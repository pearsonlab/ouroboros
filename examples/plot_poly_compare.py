"""
Compare, for the polynomial Ouroboros (trained with drive low-pass), the target vocalization
against its teacher-forced and autonomous reconstructions -- waveform + spectrogram.

3x2 grid:  rows = [target, teacher-forced recon, autonomous recon];  cols = [waveform, spectrogram]

  - teacher-forced: integrate the model's predicted ẍ (computed at the TRUE states) -> integrate_model_d2
  - autonomous: integrate the ODE feeding the generated state back -> integrate_poly_autonomous

Optionally injects additive noise into the autonomous waveform to match the target's noise floor.

Run from the repo root:
    python -m examples.plot_poly_compare --model-dir poly_lp/poly --data-dir data500/gabo_p0 \
        --start-offset-ms 50 --n 4000
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, welch

from train.train import load_model
from train.eval import integrate_model_d2, integrate_poly_autonomous, correct

plt.rcParams["text.usetex"] = False


def peak_freq(x, sr):
    f, P = welch(x - np.mean(x), fs=sr, nperseg=min(1024, len(x)))
    P[0] = 0
    return float(f[np.argmax(P)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="poly_lp/poly")
    p.add_argument("--data-dir", default="data500/gabo_p0")
    p.add_argument("--voc", type=int, default=0)
    p.add_argument("--vocalization", type=int, default=0)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--noise-sd", type=float, default=0.0,
                   help="additive noise on the autonomous waveform (match target noise floor)")
    p.add_argument("--fmax", type=float, default=10000.0)
    p.add_argument("--out-dir", default="poly_lp")
    args = p.parse_args()

    tag = f"gabo_artificial_{args.voc}"
    model, _, _, epoch = load_model(args.model_dir)
    model.eval()
    dt = model.tau
    print(f"loaded {type(model).__name__} (epoch {epoch}); drive_lowpass_ms={getattr(model,'drive_lowpass_ms',0.0)}")

    sr, audio_full = wavfile.read(os.path.join(args.data_dir, f"{tag}.wav"))
    audio_full = audio_full.astype(np.float64)
    onoffs = np.atleast_2d(np.loadtxt(os.path.join(args.data_dir, f"{tag}.txt")))
    on_s = onoffs[args.vocalization][0]
    start = int(round(on_s * sr)) + int(round(args.start_offset_ms / 1e3 * sr))
    seg = audio_full[start:start + args.n]
    print(f"reconstructing {len(seg)} samples ({len(seg)*dt*1e3:.0f} ms) from +{args.start_offset_ms:.0f}ms after onset")

    print("teacher-forced reconstruction ...", flush=True)
    tf = integrate_model_d2(model, seg, dt, smoothing=True, verbose=False)
    print(f"autonomous reconstruction (process noise_sd={args.noise_sd}) ...", flush=True)
    auto = integrate_poly_autonomous(model, seg, dt, noise_sd=args.noise_sd, verbose=False)

    target = correct(seg)  # detrend the target like the reconstructions
    nmin = min(len(target), len(tf), len(auto))
    target, tf, auto = target[:nmin], tf[:nmin], auto[:nmin]
    rows = [("target", target), ("teacher-forced", tf), ("autonomous", auto)]
    for name, w in rows:
        fin = np.isfinite(w).all()
        pf = peak_freq(w, sr) if fin else float("nan")
        print(f"  {name:14s}: peak {pf:7.0f} Hz   range [{np.nanmin(w):.3f},{np.nanmax(w):.3f}]   finite={fin}")

    t_ms = np.arange(nmin) * dt * 1e3
    A = max(np.nanstd(target), np.nanstd(tf)) * 6  # shared waveform y-scale (data-ish)
    fig, axes = plt.subplots(3, 2, figsize=(15, 9))

    def spec(x):
        npg = min(256, len(x))
        f, tt, S = spectrogram(np.nan_to_num(x), fs=sr, nperseg=npg, noverlap=int(npg * 0.9))
        return f, tt * 1e3, 10 * np.log10(S + 1e-12)

    smax = max(spec(w)[2].max() for _, w in rows)
    colors = {"target": "tab:orange", "teacher-forced": "tab:green", "autonomous": "tab:blue"}
    for r, (name, w) in enumerate(rows):
        axes[r, 0].plot(t_ms, w, color=colors[name], lw=0.8)
        axes[r, 0].set_ylim(-A, A)
        axes[r, 0].set_ylabel("a.u."); axes[r, 0].set_title(f"{name} waveform  (peak {peak_freq(np.nan_to_num(w), sr):.0f} Hz)")
        f, tt, S = spec(w)
        pcm = axes[r, 1].pcolormesh(tt, f, S, shading="auto", vmin=smax - 80, vmax=smax, cmap="magma")
        axes[r, 1].set_ylim(0, args.fmax); axes[r, 1].set_ylabel("freq (Hz)"); axes[r, 1].set_title(f"{name} spectrogram")
        fig.colorbar(pcm, ax=axes[r, 1], label="dB")
    axes[2, 0].set_xlabel("time (ms)"); axes[2, 1].set_xlabel("time (ms)")
    fig.suptitle(f"poly Ouroboros (drive low-pass {getattr(model,'drive_lowpass_ms',0.0)}ms): "
                 f"target vs teacher-forced vs autonomous  |  {tag}", y=1.00)
    fig.tight_layout()
    out_path = os.path.join(os.path.abspath(args.out_dir), "poly_target_tf_autonomous.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
