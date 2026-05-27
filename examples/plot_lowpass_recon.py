"""
Teacher-forced reconstruction from raw vs. low-pass-filtered drives.

For a trained ArneodoOuroboros on one vocalization:
  1. encode the data -> drives alpha(t), beta(t), delta(t)  (teacher-forced encoder pass)
  2. optionally low-pass the drives to their slow (control-rate) component
  3. recompute the second derivative ẍ from those drives at the TRUE data states x, x'
     (teacher forcing -- the oscillation is carried by the true states, not the drives)
  4. double-integrate ẍ to reconstruct the waveform, and overlay vs the original

This tests whether the carrier-frequency content of the per-sample drives is actually needed
for reconstruction, or whether slow drives + the true states suffice.

Run from the repo root, e.g.:
    python -m examples.plot_lowpass_recon --model-dir arneodo_big/arneodo \
        --data-dir data500/gabo_p0 --cutoff-hz 50 --out-dir ./arneodo_big
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from utils import deriv_approx_dy, deriv_approx_d2y
from train.train import load_model
from train.eval import integrate_second_deriv, correct

plt.rcParams["text.usetex"] = False


def lowpass(x, cutoff_hz, fs, order=4):
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x).copy()


def r2(recon, orig):
    sse = np.sum((recon - orig) ** 2)
    sst = np.sum((orig - orig.mean()) ** 2)
    return 1 - sse / sst


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="arneodo_big/arneodo")
    p.add_argument("--data-dir", default="data500/gabo_p0")
    p.add_argument("--voc", type=int, default=0)
    p.add_argument("--vocalization", type=int, default=0)
    p.add_argument("--start-offset-ms", type=float, default=20.0,
                   help="reconstruct starting this many ms after onset (stay in sustained voc)")
    p.add_argument("--n", type=int, default=4000, help="samples to reconstruct (avoid the offset decay)")
    p.add_argument("--cutoff-hz", type=float, default=50.0)
    p.add_argument("--out-dir", default="arneodo_big")
    args = p.parse_args()

    tag = f"gabo_artificial_{args.voc}"
    model, _, _, epoch = load_model(args.model_dir)
    model.eval()
    dt = model.tau
    print(f"loaded {type(model).__name__} (epoch {epoch})")

    # one vocalization window
    sr, audio_full = wavfile.read(os.path.join(args.data_dir, f"{tag}.wav"))
    audio_full = audio_full.astype(np.float64)
    onoffs = np.atleast_2d(np.loadtxt(os.path.join(args.data_dir, f"{tag}.txt")))
    on_s, off_s = onoffs[args.vocalization]
    # sustained sub-window starting after onset (avoids the vocalization's decaying offset,
    # where open-loop double-integration destabilizes)
    start_idx = int(round(on_s * sr)) + int(round(args.start_offset_ms / 1e3 * sr))
    b = min(len(audio_full), start_idx + args.n)
    x = audio_full[start_idx:b]
    L = len(x)
    t_ms = np.arange(L) * dt * 1e3

    # encode -> drives + state z = [x, x'] (x' = dx/ds)
    xt = torch.from_numpy(x[None, :, None]).to(torch.float32).cuda()
    dy = deriv_approx_dy(x[None, :, None])
    dyt = torch.from_numpy(dy).to(torch.float32).cuda()
    with torch.no_grad():
        alpha, beta, delta, z = model._encode(xt, dyt, dt)
    g = float(model.gamma)
    al = alpha.cpu().numpy().squeeze(); be = beta.cpu().numpy().squeeze(); de = delta.cpu().numpy().squeeze()

    def xddot(a_, b_, d_):
        # ẍ (= d2x/ds2) from given drives at the TRUE states (teacher forcing)
        ad, bd, dd = (torch.from_numpy(v[None, :, None]).to(torch.float32).cuda() for v in (a_, b_, d_))
        with torch.no_grad():
            return model._rhs(ad, bd, dd, z).cpu().numpy().squeeze()

    d2_raw = xddot(al, be, de)
    al_lp, be_lp, de_lp = (lowpass(v, args.cutoff_hz, sr) for v in (al, be, de))
    d2_lp = xddot(al_lp, be_lp, de_lp)

    # drift-free metric: how well does ẍ (from raw vs low-pass drives) match the data's ẍ?
    data_d2 = deriv_approx_d2y(x[None, :, None]).squeeze()
    print(f"teacher-forced ẍ R² vs data ẍ:  raw drives={r2(d2_raw, data_d2):+.3f}   "
          f"low-pass(<{args.cutoff_hz:.0f}Hz)={r2(d2_lp, data_d2):+.3f}")

    # double-integrate ẍ (teacher-forced open loop) in rescaled time s = t/tau
    s_steps = (np.arange(L) * dt) / model.tau
    x0 = float(x[0]); xp0 = (model.tau / dt) * float(dy[0, 0, 0])
    ic = torch.tensor([x0, xp0], dtype=torch.float32, device="cuda")
    recon_raw = integrate_second_deriv(d2_raw, ic, s_steps, method="rk4", verbose=False)
    recon_lp = integrate_second_deriv(d2_lp, ic, s_steps, method="rk4", verbose=False)
    orig = correct(x)  # detrend the original the same way the reconstructions are

    n = min(len(orig), len(recon_raw), len(recon_lp))
    orig, recon_raw, recon_lp = orig[:n], recon_raw[:n], recon_lp[:n]
    r2_raw, r2_lp = r2(recon_raw, orig), r2(recon_lp, orig)
    # short-window R2 (first 10 ms) before open-loop phase drift dominates
    sw = min(n, int(round(0.01 / dt)))
    r2_raw_sw, r2_lp_sw = r2(recon_raw[:sw], orig[:sw]), r2(recon_lp[:sw], orig[:sw])
    print(f"reconstruction R2 vs original (full {n*dt*1e3:.0f}ms): raw={r2_raw:+.3f} lp={r2_lp:+.3f}")
    print(f"reconstruction R2 vs original (first 10ms): raw={r2_raw_sw:+.3f} lp={r2_lp_sw:+.3f}")

    A = 5 * float(np.std(orig))  # y-limit at the data scale (low-pass recon diverges past this)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))
    tt = t_ms[:n]
    ax1.plot(tt, orig, color="tab:orange", lw=1.0, label="original")
    ax1.plot(tt, recon_raw, color="tab:green", lw=1.0, alpha=0.7,
             label=f"recon, RAW drives (100ms R²={r2_raw:.2f})")
    ax1.plot(tt, recon_lp, color="tab:blue", lw=1.0, alpha=0.8,
             label=f"recon, LOW-PASS <{args.cutoff_hz:.0f}Hz drives (diverges)")
    ax1.set_ylim(-A, A)
    ax1.set_title("teacher-forced reconstruction (open-loop): raw drives track; low-pass drives diverge")
    ax1.set_xlabel("time (ms)"); ax1.set_ylabel("a.u."); ax1.legend(fontsize=8, loc="upper right")

    # zoom mid-window (skip the edge-IC startup transient)
    s0 = min(n - 1, int(round(0.04 / dt)))
    zsl = slice(s0, min(n, s0 + int(round(0.02 / dt))))
    ax2.plot(tt[zsl], orig[zsl], color="tab:orange", lw=1.6, label="original")
    ax2.plot(tt[zsl], recon_raw[zsl], color="tab:green", lw=1.4, alpha=0.8, label="raw-drive recon")
    ax2.plot(tt[zsl], recon_lp[zsl], color="tab:blue", lw=1.2, alpha=0.7, label="low-pass-drive recon")
    ax2.set_ylim(-A, A)
    ax2.set_title(f"20 ms zoom (mid-window): raw tracks pitch/amplitude; low-pass <{args.cutoff_hz:.0f}Hz does not")
    ax2.set_xlabel("time (ms)"); ax2.set_ylabel("a.u."); ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"teacher-forced ẍ R² vs data: raw drives {r2(d2_raw, data_d2):+.2f}  |  "
                 f"low-pass <{args.cutoff_hz:.0f}Hz {r2(d2_lp, data_d2):+.2f}", y=1.00)

    fig.tight_layout()
    out_path = os.path.join(os.path.abspath(args.out_dir), "lowpass_drive_reconstruction.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
