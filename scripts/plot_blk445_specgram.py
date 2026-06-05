"""Spectrogram comparison for the blk445 spectral-rollout run.

Loads a checkpoint, picks a held-out cold-start val voc (silence lead-in + full
syllable), runs the closed-loop polynomial-Ouroboros integrator, and produces
a 2-row figure: target on top, autonomous reconstruction on bottom. Each row
shows waveform (left) and spectrogram (right).

The val voc selection mirrors `scripts/monitor_spectral_diagnose.py` so the
plotted voc is one autonomy_score actually evaluated.

usage:
    python scripts/plot_blk445_specgram.py [--checkpoint <path>] [--voc-idx 0]
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.train import load_model
from train.eval import integrate_poly_autonomous, generate_autonomous, correct  # noqa: E402

DATA = os.path.expanduser("~/ouroboros_data/blk445_syllC/day85")
SEED_DIR = "/home/pearson/code/ouroboros-spectral/poly_spectral_day85/seed0"


def load_cold_start_vocs(data_dir, n_max=8, silence_pad_samples=2000):
    """Match the holdout selection used in scripts/monitor_spectral_diagnose.py."""
    wavs = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    rng = np.random.default_rng(1234)
    idx = np.arange(len(wavs))
    rng.shuffle(idx)
    n_test = max(1, int(round(0.1 * len(wavs))))
    n_val = max(1, int(round(0.1 * len(wavs))))
    val_i = idx[n_test:n_test + n_val]
    val_wavs = [wavs[i] for i in val_i]

    raw, sr = [], None
    for wav in val_wavs[:n_max]:
        sr, af = wavfile.read(wav)
        if af.dtype == np.int16:
            af = af / -np.iinfo(af.dtype).min
        af = af.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        on_i = int(round(onoffs[0][0] * sr))
        off_i = int(round(onoffs[0][1] * sr))
        raw.append(af[max(0, on_i - silence_pad_samples):off_i])
    L = min(len(s) for s in raw)
    return [s[:L] for s in raw], int(sr), val_wavs[:n_max]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Specific checkpoint .tar. Default: most recent in SEED_DIR.")
    p.add_argument("--seed-dir", default=SEED_DIR)
    p.add_argument("--voc-idx", type=int, default=0,
                   help="Which held-out val voc to plot (0..7).")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Default: <seed_dir>/specgram_ckpt<N>_voc<I>.png")
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=128)
    p.add_argument("--fmax", type=float, default=8000.0)
    p.add_argument("--rescale", action="store_true",
                   help="Use generate_autonomous (rescaled to target RMS) instead of raw integrator.")
    args = p.parse_args()

    seed_dir = args.seed_dir
    if args.checkpoint is None:
        ckpts = sorted(glob.glob(os.path.join(seed_dir, "checkpoint_*.tar")),
                       key=lambda s: int(os.path.basename(s).split("_")[1].split(".")[0]))
        if not ckpts:
            raise SystemExit(f"no checkpoints under {seed_dir}")
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = args.checkpoint
    ckpt_epoch = int(os.path.basename(ckpt_path).split("_")[1].split(".")[0])

    model, _, _, _ = load_model(seed_dir if args.checkpoint is None else os.path.dirname(ckpt_path))
    # load_model picks the most recent — if we asked for a specific one, reload manually.
    if args.checkpoint is not None:
        sd = torch.load(ckpt_path, weights_only=False, map_location="cuda")
        model.load_state_dict(sd["ouroboros"])
    model.eval()

    vocs, sr, val_wavs = load_cold_start_vocs(DATA)
    if not (0 <= args.voc_idx < len(vocs)):
        raise SystemExit(f"voc-idx {args.voc_idx} out of range [0, {len(vocs)})")
    target = vocs[args.voc_idx]
    src_path = val_wavs[args.voc_idx]
    dt = 1.0 / sr

    print(f"checkpoint: {ckpt_path}", flush=True)
    print(f"voc {args.voc_idx} source: {src_path}", flush=True)
    print(f"voc length: {len(target)} samples ({len(target) * dt * 1000:.1f} ms)", flush=True)

    with torch.no_grad():
        if args.rescale:
            auto = generate_autonomous(model, target, dt, rescale=True, verbose=False)
        else:
            auto = integrate_poly_autonomous(model, target, dt, verbose=False)
    auto = np.asarray(auto, dtype=np.float64)
    auto_clean = correct(auto)
    tgt_clean = correct(np.asarray(target, dtype=np.float64))
    print(f"target std: {np.nanstd(tgt_clean):.4f}, auto std: {np.nanstd(auto_clean):.4f}", flush=True)
    print(f"auto finite frac: {np.isfinite(auto).mean():.3f}", flush=True)

    # Spectrograms
    def _spec(x):
        f, t, S = spectrogram(x, fs=sr, nperseg=args.n_fft, noverlap=args.n_fft - args.hop,
                              window="hann", detrend=False, scaling="spectrum", mode="magnitude")
        m = f <= args.fmax
        return f[m], t, np.log10(S[m] + 1e-8)

    f_t, t_t, S_t = _spec(tgt_clean)
    f_a, t_a, S_a = _spec(auto_clean)
    vmin = min(S_t.min(), S_a.min())
    vmax = max(S_t.max(), S_a.max())

    times = np.arange(len(tgt_clean)) * dt * 1000  # ms

    fig, axes = plt.subplots(2, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1, 2]})
    # Row 0: target
    axes[0, 0].plot(times, tgt_clean, lw=0.5, color="tab:orange")
    axes[0, 0].set_ylabel("target")
    axes[0, 0].set_title("waveform")
    axes[0, 0].set_xlim(times[0], times[-1])
    im0 = axes[0, 1].pcolormesh(t_t * 1000, f_t, S_t, vmin=vmin, vmax=vmax,
                                shading="auto", cmap="viridis")
    axes[0, 1].set_title("spectrogram (log10 magnitude)")
    axes[0, 1].set_ylabel("Hz")
    fig.colorbar(im0, ax=axes[0, 1], pad=0.01)

    # Row 1: autonomous. Share the waveform y-axis with the target so the
    # reconstruction is judged on the target's amplitude scale (rescaled mode);
    # in raw mode this will visually clip a saturated rollout / hide a collapsed
    # one, which is itself informative.
    color_a = "tab:blue" if not args.rescale else "tab:green"
    label_a = "autonomous (raw)" if not args.rescale else "autonomous (rescaled)"
    axes[1, 0].plot(times, auto_clean, lw=0.5, color=color_a)
    axes[1, 0].set_ylabel(label_a)
    axes[1, 0].set_xlabel("ms")
    axes[1, 0].set_xlim(times[0], times[-1])
    axes[1, 0].set_ylim(axes[0, 0].get_ylim())
    im1 = axes[1, 1].pcolormesh(t_a * 1000, f_a, S_a, vmin=vmin, vmax=vmax,
                                shading="auto", cmap="viridis")
    axes[1, 1].set_xlabel("ms")
    axes[1, 1].set_ylabel("Hz")
    fig.colorbar(im1, ax=axes[1, 1], pad=0.01)

    fig.suptitle(f"blk445 syllable_C  (checkpoint {ckpt_epoch}, val voc {args.voc_idx})",
                 fontsize=11)
    plt.tight_layout()

    if args.out is None:
        suffix = "_rescaled" if args.rescale else ""
        out = os.path.join(seed_dir, f"specgram_ckpt{ckpt_epoch}_voc{args.voc_idx}{suffix}.png")
    else:
        out = args.out
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
