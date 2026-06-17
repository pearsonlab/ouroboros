"""Cold-start recon comparison: silence_pad lead-in + full voc.

Uses load_voc_windows_coldstart so the integration IC is in silence and the model
must ignite the syllable on its own. Output: 3 rows × 4 cols (target wave, target
spec, recon wave, recon spec). The vertical dashed line marks voc onset (end of
the silence lead-in).
"""

import argparse
import glob
import os
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.signal import spectrogram

from train.train import load_model
from train.eval import generate_autonomous
from examples.run_lambda_pipeline import load_voc_windows_coldstart


def load_specific(ckpt_path):
    tmp = tempfile.mkdtemp(prefix="ckpt_plot_")
    link = os.path.join(tmp, os.path.basename(ckpt_path))
    os.symlink(os.path.abspath(ckpt_path), link)
    try:
        model, _, _, _ = load_model(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return model


def plot_wave(ax, x, sr, title, onset_ms=None):
    if x is None or not np.isfinite(x).all():
        ax.text(0.5, 0.5, "non-finite", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="red")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        return
    t = np.arange(len(x)) / sr * 1e3
    ax.plot(t, x, lw=0.6, color="steelblue")
    ax.set_xlim(t[0], t[-1])
    if onset_ms is not None:
        ax.axvline(onset_ms, ls="--", color="0.5", lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("amp")
    ax.set_title(title)


def plot_spec(ax, x, sr, title, fmax=8000, onset_ms=None):
    if x is None or not np.isfinite(x).all():
        ax.text(0.5, 0.5, "non-finite", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="red")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title)
        return
    nperseg = min(512, len(x))
    f, t, Sxx = spectrogram(x - np.mean(x), fs=sr, nperseg=nperseg,
                            noverlap=int(nperseg * 0.875), scaling="spectrum")
    m = f <= fmax
    S_db = 10 * np.log10(Sxx[m] + 1e-20)
    ax.pcolormesh(t * 1e3, f[m] / 1e3, S_db, shading="auto",
                  cmap="magma", vmin=S_db.max() - 60, vmax=S_db.max())
    if onset_ms is not None:
        ax.axvline(onset_ms, ls="--", color="white", lw=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("freq (kHz)")
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-vocs", type=int, default=3)
    p.add_argument("--silence-pad-samples", type=int, default=2205,
                   help="50 ms at 44.1 kHz")
    args = p.parse_args()

    model = load_specific(args.ckpt)
    model.eval()
    print(f"loaded {args.ckpt}")

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    test_dir = dirs[-1]
    vocs, sr = load_voc_windows_coldstart(test_dir, args.n_vocs, args.silence_pad_samples)
    dt = 1.0 / sr
    onset_ms = args.silence_pad_samples / sr * 1e3
    print(f"test_dir={test_dir} sr={sr} n_vocs={len(vocs)} L={len(vocs[0]) if vocs else 0} "
          f"onset_at={onset_ms:.1f}ms")

    n = len(vocs)
    fig, axs = plt.subplots(n, 4, figsize=(16, 3.0 * n))
    if n == 1:
        axs = axs[None, :]
    for i, voc in enumerate(vocs):
        target = np.asarray(voc, dtype=np.float64)
        # cold-start synthesis: rescale=False keeps voc-specific amplitude in the output
        recon = generate_autonomous(model, voc, dt, rescale=False, detrend=True, verbose=False)
        recon = np.asarray(recon, dtype=np.float64)
        recon_ok = np.isfinite(recon).all()
        if recon_ok:
            L = min(len(target), len(recon))
            target_p, recon_p = target[:L], recon[:L]
        else:
            target_p, recon_p = target, None
            print(f"voc {i}: recon non-finite (max |recon|={np.nanmax(np.abs(recon)):.3e})")

        plot_wave(axs[i, 0], target_p, sr, f"voc {i}  target  (wave)", onset_ms=onset_ms)
        plot_spec(axs[i, 1], target_p, sr, f"voc {i}  target  (spec)", onset_ms=onset_ms)
        plot_wave(axs[i, 2], recon_p, sr, f"voc {i}  COLD-START recon  (wave)", onset_ms=onset_ms)
        plot_spec(axs[i, 3], recon_p, sr, f"voc {i}  COLD-START recon  (spec)", onset_ms=onset_ms)

    fig.suptitle(f"COLD-START (silence lead-in + voc) — {os.path.basename(args.ckpt)} on {os.path.basename(test_dir)}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
