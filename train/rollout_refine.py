"""
Rollout refinement objective for the polynomial Ouroboros -- a reusable training-pipeline component.

Phase-invariant objective on the soft-saturated differentiable RK4 AUTONOMOUS rollout, so a
phase-drifted-but-correct-content free run is scored well and the optimizer fixes frequency content
+ loudness instead of collapsing amplitude (the failure mode of pointwise rollout MSE on a drifting
oscillator):

    L = lam_spec * L_mrstft  +  lam_env * L_env  +  lam_tf * L_tf

  - L_mrstft : multi-resolution STFT MAGNITUDE loss (spectral convergence + log-magnitude L1).
               Magnitude discards phase -> tolerant of carrier drift; penalizes a wrong pitch/harmonic
               stack that persists across frames.
  - L_env    : Gaussian-low-passed |x| envelope L1 (per-voc normalized). The envelope IS the
               instantaneous amplitude, so REQUIRING IT TO MATCH pins the autonomous loudness/scale --
               but it must be weighted up (lam_env >> lam_spec) or the spectral term swamps it and the
               global amplitude drifts free. lam_env~10 fixes amplitude across seeds without regressing
               already-good models (validated 2026-05-28).
  - L_tf     : original one-step teacher-forced anchor (preserve the well-fit dynamics / R^2).

Backprops through the rollout with a curriculum on the horizon; because the loss is phase-invariant
the horizon can be long. See examples/finetune_rollout_spectral_poly.py (single-model entry) and
examples/train_poly_rollout.py (integrated TF + refine trainer).
"""

import glob
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

from utils import deriv_approx_dy, deriv_approx_d2y

BX, BXP = 3.0, 5.0  # soft-saturation bounds, sized so steady-state |y| ≈ 1 (env-match)
                     # sits well inside the box but transient excursions are still tamed.
                     # Old (0.5, 1.0) assumed e ≈ 1 so y was small; the env_anchor =
                     # mean((e - target_env)^2) pulls e to ≈ 0.01-0.02, so y ≈ audio/e ≈ 1.
DEFAULT_CONFIGS = ((256, 64), (512, 128), (1024, 256))


def gather_windows(dirs, n, L, start_off_ms):
    """held-out sustained vocalization windows of length L, starting start_off_ms after onset."""
    segs, sr = [], None
    for d in dirs:
        for wav in sorted(glob.glob(os.path.join(d, "*.wav"))):
            sr, af = wavfile.read(wav)
            af = af.astype(np.float64)
            on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
            s = int(on * sr) + int(start_off_ms / 1e3 * sr)
            seg = af[s:s + L]
            if len(seg) == L:
                segs.append(seg)
            if len(segs) >= n:
                return np.stack(segs)[:, :, None], sr
    return np.stack(segs)[:, :, None], sr


@lru_cache(maxsize=None)
def _hann_window(n_fft, device, dtype):
    """Cache STFT analysis windows -- they depend only on n_fft (constant across all
    steps/epochs), so the per-call torch.hann_window alloc (~6×/step in mrstft_loss) is
    pure overhead."""
    return torch.hann_window(n_fft, device=device, dtype=dtype)


def stft_mag(x, n_fft, hop):
    win = _hann_window(n_fft, x.device, x.dtype)
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
                   center=True, return_complex=True)
    return S.abs()  # (B, F, T)


_MEL_FB_CACHE = {}
def _mel_fb(n_fft, sr, n_mels, fmax, device, dtype):
    """Cached mel filterbank (n_mels, n_fft//2+1) as a torch tensor for mel-warping the STFT
    magnitude inside the loss. Uses librosa (already a codebase dep)."""
    key = (int(n_fft), int(round(sr)), int(n_mels), float(fmax), str(device), str(dtype))
    fb = _MEL_FB_CACHE.get(key)
    if fb is None:
        import librosa
        fb_np = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)  # (n_mels, F)
        fb = torch.tensor(fb_np, device=device, dtype=dtype)
        _MEL_FB_CACHE[key] = fb
    return fb


_MEL_HZ_CACHE = {}
def _mel_hz(n_mels, fmax):
    key = (int(n_mels), float(fmax))
    v = _MEL_HZ_CACHE.get(key)
    if v is None:
        import librosa
        v = librosa.mel_frequencies(n_mels=n_mels, fmax=fmax)
        _MEL_HZ_CACHE[key] = v
    return v


def mrstft_loss(xg, tgt, configs=DEFAULT_CONFIGS, eps=1e-3, sc_eps=1e-2, return_components=False,
                mel=False, sr=44100, n_mels=80, fmax=None,
                floor_fit=False, floor_pctile=25.0, floor_cutoff_hz=375.0, floor_correction=1.2):
    """multi-resolution STFT magnitude loss: spectral convergence + log-magnitude L1.

    When return_components=True, returns dict {'spec', 'sc', 'logm'} of scalars instead
    of the bare 'spec' scalar -- useful for separately logging each contributor.

    sc_eps is the denominator floor in the spectral-convergence ratio. The original
    1e-8 was unstably small for batches with silent targets (e.g. pre-onset prefix in
    cold-start windows): when ||G||_F << 1e-8 the backward grad on A blows up to ~1/eps
    per element (~1e8), overflows the chain rule through env_mamba, and crashes the
    step. 1e-2 matches the typical ||G||_F noise floor for normalized audio at our
    STFT sizes -- preserves SC behavior on voiced targets while bounding the silent-
    target gradient to ~100 per element."""
    sc_total = 0.0
    logm_total = 0.0
    for n_fft, hop in configs:
        A = stft_mag(xg, n_fft, hop)
        G = stft_mag(tgt, n_fft, hop)
        if mel:
            # Warp the linear STFT magnitude onto a mel filterbank BEFORE the SC/log-mag terms,
            # so the loss weights low/mid vocal structure the way a mel spectrogram does. fmax
            # defaults to Nyquist (nothing discarded, just mel-spaced).
            fb = _mel_fb(n_fft, sr, n_mels, fmax if fmax is not None else sr / 2.0, A.device, A.dtype)
            A = torch.einsum('mf,bft->bmt', fb, A)
            G = torch.einsum('mf,bft->bmt', fb, G)
        if floor_fit:
            # Noise-floor target for the noise-fit epoch (per sample, so each voc targets its OWN
            # floor). Pick the QUIET TIME FRAMES -- those whose BROADBAND power (>= floor_cutoff_hz,
            # the noise band, NOT total power which the LF/rumble would contaminate) falls in the
            # [0.4*floor_pctile, floor_pctile] percentile band -- and AVERAGE their spectra. That gives
            # a coherent floor SPECTRUM (a real quiet-moment spectrum), unlike a per-frequency
            # percentile which stitches a different time frame per bin and sits far below the mean.
            # floor_correction de-biases the mild downward pull of selecting low-power frames
            # (minimum-statistics correction): model the broadband power as Gamma(shape=K); the
            # unbiased factor is C(K) = (p_hi-p_lo)/[F_{K+1}(b)-F_{K+1}(a)], K = mean^2/var of the
            # broadband power. Default 1.2 ~ C(K=30). Broadband bins get this floor; LF keeps the
            # real target. Full rationale + derivation: docs/noise_floor_fit.md.
            fdim = G.shape[-2]
            if mel:
                bin_hz = torch.as_tensor(_mel_hz(fdim, fmax if fmax is not None else sr / 2.0),
                                         device=G.device, dtype=G.dtype)
            else:
                bin_hz = torch.linspace(0.0, sr / 2.0, fdim, device=G.device, dtype=G.dtype)
            bb = (bin_hz >= float(floor_cutoff_hz))                      # broadband mask (F,)
            bbp = G[:, bb, :].sum(dim=1)                                 # (B, T) broadband power/frame
            q_lo = torch.quantile(bbp, 0.4 * float(floor_pctile) / 100.0, dim=1, keepdim=True)
            q_hi = torch.quantile(bbp, float(floor_pctile) / 100.0, dim=1, keepdim=True)
            sel = ((bbp >= q_lo) & (bbp <= q_hi)).to(G.dtype)           # (B, T) quiet frames
            sel = torch.where(sel.sum(-1, keepdim=True) > 0, sel, (bbp <= q_hi).to(G.dtype))  # fallback
            w = sel.unsqueeze(1)                                         # (B, 1, T)
            N_floor = (G * w).sum(-1) / w.sum(-1).clamp_min(1.0)         # (B, F) quiet-frame mean
            N_floor = (N_floor * float(floor_correction)).unsqueeze(-1)  # (B, F, 1) bias-corrected
            G = torch.where(bb.view(1, fdim, 1), N_floor.expand_as(G), G)
        sc = torch.norm(G - A, dim=(-2, -1)) / (torch.norm(G, dim=(-2, -1)) + sc_eps)
        logm = (torch.log(G + eps) - torch.log(A + eps)).abs().mean(dim=(-2, -1))
        sc_total = sc_total + sc.mean()
        logm_total = logm_total + logm.mean()
    n = len(configs)
    sc_mean = sc_total / n
    logm_mean = logm_total / n
    spec = sc_mean + logm_mean
    if return_components:
        return {"spec": spec, "sc": sc_mean, "logm": logm_mean}
    return spec


def gaussian_envelope(x, dt, env_ms):
    """Gaussian low-pass of |x| (removes the carrier, keeps the loudness modulation). x: (B,H)."""
    sigma = (env_ms / 1e3) / dt
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    xr = F.pad(x.abs().unsqueeze(1), (radius, radius), mode="reflect")
    return F.conv1d(xr, k.view(1, 1, -1)).squeeze(1)  # (B,H)


def env_loss(xg, tgt, dt, env_ms, eps=1e-6):
    ea = gaussian_envelope(xg, dt, env_ms)
    eg = gaussian_envelope(tgt, dt, env_ms)
    return ((ea - eg).abs().mean(dim=1) / (eg.mean(dim=1) + eps)).mean()


def env_loss_log(xg, tgt, dt, env_ms, eps=1e-4):
    """Log-ratio envelope loss: |log( (env(a) + eps) / (env(g) + eps) )| averaged over time
    and batch. Symmetric in (auto, target) scale — penalizes "rollout K× too quiet" the same as
    "K× too loud", so there's no trivial-zero floor like in env_loss (where a silent rollout
    gives L=1 against a non-silent target). dB-natural: a log_e ratio is dB/8.686, so the loss
    grows linearly in dB-distance from the target envelope.

    eps acts as a soft noise floor that prevents log(0) divergence on the silence portions of
    a target. ~1e-4 matches the per-sample noise floor of the normalized blk445 audio (peak
    amplitude ~1.0 / int16 quantization 1/32768). Larger eps -> less sensitivity near silence.
    """
    ea = gaussian_envelope(xg, dt, env_ms)
    eg = gaussian_envelope(tgt, dt, env_ms)
    return (torch.log(ea + eps) - torch.log(eg + eps)).abs().mean(dim=1).mean()


def rollout_refine(model, X, dt, *, epochs=8, hmin=768, hmax=1500, batch_size=6, lr=1e-4,
                   lam_spec=1.0, lam_env=10.0, lam_tf=1.0, clip=5.0, env_ms=2.0,
                   configs=DEFAULT_CONFIGS, device="cuda", verbose=True):
    """
    In-place rollout refinement of a trained polynomial Ouroboros.

    X: numpy (N, L, 1) sustained vocalization windows (L >= hmax). Modifies `model` in place.
    Returns (history, opt): per-epoch loss history (list of dicts) and the Adam optimizer (so the
    caller can persist it via train.train.save_model).
    """
    assert hmax <= X.shape[1], "hmax must be <= window length L"
    model.train()
    D1 = deriv_approx_dy(X)
    D2 = deriv_approx_d2y(X)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Dt = torch.tensor(D1, dtype=torch.float32, device=device)
    D2t = torch.tensor(D2, dtype=torch.float32, device=device)
    var_d2 = float(D2t.var())
    P = model.kernel.poly_dim + 1
    powers = torch.arange(P, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = Xt.shape[0]
    H_sched = np.unique(np.round(np.geomspace(hmin, hmax, epochs)).astype(int))
    if verbose:
        print(f"rollout refine: {N} windows L={X.shape[1]} H {hmin}->{hmax} | "
              f"lam tf={lam_tf} spec={lam_spec} env={lam_env} env_ms={env_ms} epochs={epochs}", flush=True)

    history = []
    for epoch in range(epochs):
        H = int(H_sched[min(epoch, len(H_sched) - 1)])
        perm = torch.randperm(N)
        tot = {"spec": 0.0, "env": 0.0, "tf": 0.0}
        nb = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x, dxd, d2b = Xt[idx], Dt[idx], D2t[idx]
            z2 = (model.tau / dt) * dxd
            omega, gamma, wk, weights, _ = model.get_funcs(x, dxd.clone(), dt)  # differentiable
            tf = -(omega ** 2) * x - gamma * z2 - wk
            L_tf = ((tf - d2b) ** 2).mean() / var_d2

            om, ga, w = omega[:, :, 0], gamma[:, :, 0], weights  # w: (B,L,P,P)
            om2 = om ** 2  # square omega once over the horizon (was re-squared every substep)

            def f(xx, vv, om2k, gak, w_k):
                xpw = xx.unsqueeze(1) ** powers
                xvw = vv.unsqueeze(1) ** powers
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
                return vv, -om2k * xx - gak * vv - kern

            xc = x[:, 0, 0].detach()
            xp = z2[:, 0, 0].detach()
            xs = [xc]
            for k in range(H - 1):
                # slice step-k drives once; the 4 RK4 substeps reuse them (was 4× re-sliced)
                om2k, gak, w_k = om2[:, k], ga[:, k], w[:, k]
                k1x, k1v = f(xc, xp, om2k, gak, w_k)
                k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v, om2k, gak, w_k)
                k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v, om2k, gak, w_k)
                k4x, k4v = f(xc + k3x, xp + k3v, om2k, gak, w_k)
                xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
                xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
                xc = BX * torch.tanh(xc / BX)
                xp = BXP * torch.tanh(xp / BXP)
                xs.append(xc)
            xg = torch.stack(xs, dim=1)   # (B,H) autonomous
            tgt = x[:, :H, 0]              # (B,H) target, same time span

            L_spec = mrstft_loss(xg, tgt, configs)
            L_env = env_loss(xg, tgt, dt, env_ms)
            loss = lam_spec * L_spec + lam_env * L_env + lam_tf * L_tf
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot["spec"] += float(L_spec); tot["env"] += float(L_env); tot["tf"] += float(L_tf)
            nb += 1
        rec = {"epoch": epoch + 1, "H": H, "spec": tot["spec"] / nb,
               "env": tot["env"] / nb, "tf": tot["tf"] / nb}
        history.append(rec)
        if verbose:
            print(f"[refine ep {epoch + 1}/{epochs} H={H}] spec={rec['spec']:.4f} "
                  f"env={rec['env']:.4f} tf={rec['tf']:.4f}", flush=True)
    return history, opt
