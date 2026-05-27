"""
Plot target vs. autonomously-generated waveforms and their spectrograms for a trained
`ArneodoOuroboros` model.

Loads the most recent checkpoint under <model-dir>, picks a held-out segment from the gabo
dataset, runs `integrate_model_autonomous`, and writes a 2x2 figure:
    row 0: waveforms      (true | autonomous)
    row 1: spectrograms   (true | autonomous)

Run from the repo root:
    python -m examples.plot_autonomous --out-dir ./arneodo_run
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.signal import spectrogram

from data.load_data import get_segmented_audio
from train.train import load_model
from train.eval import integrate_model_autonomous, correct

plt.rcParams["text.usetex"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="./arneodo_run")
    parser.add_argument("--n-integrate", type=int, default=11000, help="samples to integrate")
    parser.add_argument("--pre-onset-ms", type=float, default=10.0,
                        help="start the rollout this many ms before the vocalization onset")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fmax", type=float, default=10000.0, help="max spectrogram freq (Hz)")
    parser.add_argument("--method", default="rk4")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "gabo_data")
    model_dir = os.path.join(out_dir, "model", "arneodo")

    # load the trained model -----------------------------------------------------------
    model, _, _, epoch = load_model(model_dir)
    model.eval()
    print(f"loaded {type(model).__name__} (epoch {epoch})")

    # held-out vocalization, windowed to start `pre_onset_ms` BEFORE the first onset, so the
    # rollout is seeded from (near) rest just before vocalization and must spin the
    # oscillation up itself. analysis mode returns aud[onset - padding : offset], so passing
    # padding = pre_onset gives exactly that window (scale-consistent with training loading).
    pre_onset_s = args.pre_onset_ms / 1e3
    chunks, sr = get_segmented_audio(
        data_dir,
        data_dir,
        max_vocs=40,
        seed=args.seed + 1,
        training=False,
        padding=pre_onset_s,
        shuffle_order=True,
    )
    dt = 1 / sr
    segment = np.asarray(chunks[0]).squeeze()
    n = min(args.n_integrate, len(segment))
    segment = segment[:n]

    print(
        f"rollout starts {args.pre_onset_ms:.0f} ms before onset; "
        f"autonomously integrating {n} samples ({n * dt * 1e3:.1f} ms) ..."
    )
    x_gen = integrate_model_autonomous(
        model, segment, dt, method=args.method, detrend=True, verbose=True
    )
    print()

    # detrend the reference the same way the autonomous output is detrended
    true = correct(segment.astype(np.float64))[: len(x_gen)]
    t_ms = np.arange(len(x_gen)) * dt * 1e3

    def spec(x):
        nperseg = min(256, len(x))
        f, tt, Sxx = spectrogram(
            x, fs=sr, nperseg=nperseg, noverlap=int(nperseg * 0.9)
        )
        Sxx = 10 * np.log10(Sxx + 1e-12)
        return f, tt * 1e3, Sxx

    f_t, tt_t, S_t = spec(true)
    f_g, tt_g, S_g = spec(x_gen)
    vmin = min(S_t.max() - 80, S_g.max() - 80)  # ~80 dB dynamic range
    vmax = max(S_t.max(), S_g.max())

    # figure -----------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    # shared y-scale across the two waveform panels, so amplitudes are directly comparable
    axes[0, 0].plot(t_ms, true, color="tab:orange", lw=0.8)
    axes[0, 0].set_title("Target waveform (detrended)")
    axes[0, 1].plot(t_ms, x_gen, color="tab:blue", lw=0.8)
    axes[0, 1].set_title("Autonomous waveform")
    ymax = max(np.abs(true).max(), np.abs(x_gen).max()) * 1.05
    for ax in axes[0]:
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("a.u.")
        ax.set_ylim(-ymax, ymax)

    for ax, (f, tt, S), title in [
        (axes[1, 0], (f_t, tt_t, S_t), "Target spectrogram"),
        (axes[1, 1], (f_g, tt_g, S_g), "Autonomous spectrogram"),
    ]:
        pcm = ax.pcolormesh(tt, f, S, shading="auto", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_ylim(0, args.fmax)
        ax.set_title(title)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("frequency (Hz)")
        fig.colorbar(pcm, ax=ax, label="power (dB)")

    fig.suptitle(
        f"Arneodo autonomous integration  |  start {args.pre_onset_ms:.0f} ms pre-onset  |  "
        f"{n * dt * 1e3:.0f} ms  |  sr {sr} Hz",
        y=1.00,
    )
    fig.tight_layout()

    out_path = os.path.join(out_dir, "waveforms_and_spectrograms.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
