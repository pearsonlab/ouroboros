"""Cold-start amplitude stability scan for 8 seedcull seeds.

Same diagnostic as scan_seed_amp_stability.py but using the full window
[silence_lead_in + vocalization] from a near-rest IC -- the situation finchsim's
simulate_hvc_to_audio actually faces (silence -> ignition -> sustained oscillation).
Reports per-seed CV of the rescale factor, range, median, and spec_corr.

A seed that is amplitude-stable here would let you ship one global rescale and
nothing else for the streaming synthesis.
"""
import os, sys, glob
sys.path.insert(0, "/home/pearson/code/ouroboros")

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch

from utils import deriv_approx_dy
from train.train import load_model


def _poly_rk4_step(y, v, omega, gamma, W, powers, h=1.0):
    def rhs(yy, vv):
        kern = float((yy ** powers) @ (W @ (vv ** powers)))
        return vv, -(omega * omega) * yy - gamma * vv - kern
    k1y, k1v = rhs(y, v)
    k2y, k2v = rhs(y + 0.5 * h * k1y, v + 0.5 * h * k1v)
    k3y, k3v = rhs(y + 0.5 * h * k2y, v + 0.5 * h * k2v)
    k4y, k4v = rhs(y + h * k3y, v + h * k3v)
    y = y + (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y, v


CKPT_BASE = "/home/pearson/code/ouroboros/poly_seedcull"
DATA_DIR  = "/home/pearson/code/ouroboros/data500/gabo_p9"
N_VOCS    = 10
SILENCE_PAD = 2000           # 50 ms lead-in (matches finchsim/data/paired_data.npz)
powers    = np.arange(16)


def correct(x):
    fs = len(x); sos = butter(5, 100/(0.5*fs), btype="low", output="sos")
    return x - sosfiltfilt(sos, x)


def logpsd(x, fs):
    f, P = welch(x - x.mean(), fs=fs, nperseg=min(1024, len(x)))
    m = f <= 8000
    return np.log(P[m] + 1e-20)


# Full-window cold-start segments: SILENCE_PAD lead-in + vocalization
wavs = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))[:N_VOCS]
segs = []
for w in wavs:
    sr, aud = wavfile.read(w); aud = aud.astype(np.float64)
    onoffs = np.atleast_2d(np.loadtxt(w.replace(".wav", ".txt")))
    on_i, off_i = int(round(onoffs[0][0] * sr)), int(round(onoffs[0][1] * sr))
    start = max(0, on_i - SILENCE_PAD)
    segs.append(aud[start:off_i])
dt = 1.0 / sr
L = min(len(s) for s in segs)
segs = [s[:L] for s in segs]
print(f"{len(segs)} held-out vocs, L={L} ({L*dt*1e3:.0f} ms; lead-in {SILENCE_PAD*dt*1e3:.0f} ms), "
      f"COLD START (IC=audio[0]≈silence)\n", flush=True)

print(f"{'seed':>4} {'n':>3} {'CV%':>6} {'median':>10} {'min':>10} {'max':>10} "
      f"{'max/min':>8} {'spec_corr':>10}", flush=True)

for seed in range(8):
    cd = f"{CKPT_BASE}/poly_lam1.068_seed{seed}"
    if not os.path.isdir(cd):
        print(f"{seed:>4} (no dir)"); continue
    model, _, _, _ = load_model(cd); model.eval()
    tau = float(model.tau)

    factors, corrs = [], []
    for audio in segs:
        x  = audio[None, :, None]
        dy = deriv_approx_dy(x)
        xt = torch.from_numpy(x).to(torch.float32).cuda()
        dt_t = torch.from_numpy(dy).to(torch.float32).cuda()
        with torch.no_grad():
            om, ga, _, wts, _ = model.get_funcs(xt, dt_t.clone(), dt, smoothing=False)
        om = om.cpu().numpy()[0, :, 0]
        ga = ga.cpu().numpy()[0, :, 0]
        co = wts.cpu().numpy()[0]                  # (L, 16, 16)

        # cold start: IC at window start (= silence, near 0); matches generate_autonomous
        y, v = float(audio[0]), float((tau / dt) * dy[0, 0, 0])
        raw = np.empty(L); raw[0] = y; bad = False
        for t in range(L - 1):
            W = co[t].copy(); W[1, 0] = 0; W[0, 1] = 0
            y, v = _poly_rk4_step(y, v, float(om[t]), float(ga[t]), W, powers, 1.0)
            if not (np.isfinite(y) and np.isfinite(v)):
                bad = True; break
            raw[t + 1] = y
        if bad:
            continue
        ca, cr = correct(audio), correct(raw)
        s_t, s_a = float(np.std(ca)), float(np.std(cr))
        if s_a < 1e-12:
            continue
        r = s_t / s_a
        factors.append(r)
        cr_rs = cr * r
        cc = float(np.corrcoef(logpsd(ca, 1/dt), logpsd(cr_rs, 1/dt))[0, 1])
        corrs.append(cc)
    if not factors:
        print(f"{seed:>4}    0 (no usable vocs)"); continue
    arr = np.array(factors)
    cv = arr.std() / arr.mean() * 100
    print(f"{seed:>4} {len(factors):>3d} {cv:>5.1f}% {np.median(arr):>10.3f} "
          f"{arr.min():>10.3f} {arr.max():>10.3f} {arr.max()/arr.min():>7.1f}x "
          f"{np.mean(corrs):>10.3f}", flush=True)
