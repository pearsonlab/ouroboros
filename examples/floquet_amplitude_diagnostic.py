"""
Amplitude-Floquet diagnostic: is the model's learned cycle attracting, neutral, or repelling?

For a 2-D oscillator d2x/ds2 = f(x, x'; drives), the non-trivial Floquet multiplier over the orbit is
exp(Lambda) with the area-contraction exponent

    Lambda = integral over the window of  d f / d x'      (x' = dx/ds, the model's internal velocity)

evaluated ALONG THE DATA ORBIT with the model's drives. Lambda<0 -> attracting (amplitude decays toward
the cycle), ~0 -> neutral (SHO-like, amplitude free), >0 -> repelling (amplitude grows). The predicted
log change in oscillation amplitude over the window is ~ Lambda/2 (area ~ amplitude^2).

f = -omega^2 x - gamma x' - sum_{p,k} w_{pk} x^p (x')^k, so
    d f / d x' = -gamma - sum_{p,k>=1} w_{pk} x^p k (x')^{k-1}.

Prediction to test: sign(Lambda) should match the actual rollout decay/growth (2nd-half/1st-half std),
and the spread of Lambda across seeds should track the spread of autonomous amplitude.

    python -m examples.floquet_amplitude_diagnostic --data-dir data500/gabo_p9 --n-vocs 6 \
        --runs "seed0=poly_ro_pipeline/seed0/tf" ...
"""

import argparse
import glob
import os

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import welch

from utils import deriv_approx_dy
from train.train import load_model
from train.eval import integrate_poly_autonomous, correct


def load_voc_windows(data_dir, n_vocs, off_ms, n):
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
        s = int(on * sr) + int(off_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs, sr


def peak_freq(x, sr):
    f, P = welch(x - np.mean(x), fs=sr, nperseg=min(1024, len(x)))
    P[0] = 0
    return float(f[np.argmax(P)])


def floquet_exponent(model, seg, dt):
    """Lambda = sum_s d f/d x' along the data orbit (rescaled time, ds=1 per sample)."""
    X = np.asarray(seg, dtype=np.float64)[None, :, None]
    D1 = deriv_approx_dy(X)
    Xt = torch.tensor(X, dtype=torch.float32, device="cuda")
    Dt = torch.tensor(D1, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        omega, gamma, wk, weights, _ = model.get_funcs(Xt, Dt.clone(), dt)
        z2 = (model.tau / dt) * Dt                  # internal velocity x'
        x = Xt[0, :, 0]                              # (L,)
        v = z2[0, :, 0]                              # (L,)
        ga = gamma[0, :, 0]                          # (L,)
        w = weights[0]                               # (L,P,P) indexed [l,p,k] = w_{x^p (x')^k}
        P = w.shape[-1]
        powers = torch.arange(P, device="cuda", dtype=torch.float32)
        x_pow = x[:, None] ** powers                 # (L,P) x^p
        v_pow = v[:, None] ** powers                 # (L,P) (x')^k
        dvk = torch.zeros_like(v_pow)                # d/dx' of (x')^k = k (x')^{k-1}
        dvk[:, 1:] = powers[1:][None, :] * v_pow[:, :-1]
        dkern_dv = torch.einsum("lpk,lp,lk->l", w, x_pow, dvk)
        dfdv = -ga - dkern_dv                        # (L,)
        Lam = float(dfdv.sum())                      # ds = 1 per sample
        mean_dfdv = float(dfdv.mean())
    return Lam, mean_dfdv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data500/gabo_p9")
    p.add_argument("--n-vocs", type=int, default=6)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--runs", nargs="+", required=True, help="label=dir entries")
    args = p.parse_args()

    segs, sr = load_voc_windows(args.data_dir, args.n_vocs, args.start_offset_ms, args.auto_n)
    print(f"{len(segs)} held-out vocs from {args.data_dir} (n={args.auto_n}, sr={sr})\n", flush=True)
    print(f"{'label':8s} {'Lambda':>8s} {'L/cycle':>8s} {'pred A ratio':>12s} "
          f"{'act decay':>10s} {'act std/tgt':>12s}", flush=True)
    print(f"{'':8s} {'(area)':>8s} {'':>8s} {'exp(L/2)':>12s} "
          f"{'2nd/1st':>10s} {'':>12s}", flush=True)

    Ls, decays = [], []
    for entry in args.runs:
        label, d = entry.split("=", 1)
        model, _, _, _ = load_model(d)
        model.eval()
        dt = model.tau
        lams, lpc, predA, decay, stdrat = [], [], [], [], []
        for seg in segs:
            Lam, _ = floquet_exponent(model, seg, dt)
            ncyc = peak_freq(seg, sr) * (len(seg) / sr)
            auto = integrate_poly_autonomous(model, seg, dt, noise_sd=0.0, detrend=True, verbose=False)
            tgt = correct(np.asarray(seg, dtype=np.float64))
            h1, h2 = auto[:len(auto) // 2], auto[len(auto) // 2:]
            lams.append(Lam)
            lpc.append(Lam / max(ncyc, 1e-9))
            predA.append(np.exp(np.clip(Lam / 2, -20, 20)))
            decay.append(np.nanstd(h2) / (np.nanstd(h1) + 1e-12))
            stdrat.append(np.nanstd(auto) / (np.nanstd(tgt) + 1e-12))
        L = float(np.mean(lams))
        Ls.append(L); decays.append(float(np.mean(decay)))
        print(f"{label:8s} {L:+8.2f} {np.mean(lpc):+8.4f} {np.mean(predA):12.3f} "
              f"{np.mean(decay):10.3f} {np.mean(stdrat):12.3f}", flush=True)

    Ls, decays = np.array(Ls), np.array(decays)
    if len(Ls) >= 3:
        # sign agreement: Lambda<0 should give decay<1, Lambda>0 decay>1
        agree = np.mean((np.sign(Ls) == np.sign(np.log(decays + 1e-12))))
        r = float(np.corrcoef(Ls, np.log(decays + 1e-12))[0, 1])
        print(f"\nLambda vs log(decay ratio): sign agreement {agree*100:.0f}%, corr r={r:+.2f}", flush=True)
        print(f"Lambda spread across seeds: mean {Ls.mean():+.2f} std {Ls.std():.2f} "
              f"[{Ls.min():+.2f},{Ls.max():+.2f}]", flush=True)


if __name__ == "__main__":
    main()
