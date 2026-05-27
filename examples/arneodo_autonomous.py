"""
End-to-end demo for the Arneodo-2021 parameterization + autonomous integration.

Pipeline
--------
1. Generate a small synthetic dataset from the Mindlin model (`generate_vocal_dataset`).
2. Train an `ArneodoOuroboros` model (the biomechanical syrinx ODE parameterization),
   exactly as the polynomial model is trained -- teacher-forced one-step prediction of
   the second derivative -- via `train_model(..., parameterization="arneodo")`.
3. Run *fully autonomous* integration on a held-out segment: the ODE is integrated while
   feeding the generated state (x, x') back into the right-hand side, with only the
   control time series alpha(t), beta(t) (produced once by the encoder) and the initial
   condition coming from data. Plot generated vs. true waveform and report a simple
   pitch / correlation comparison.

The defaults below are a fast smoke test (a handful of vocalizations, a few epochs, a
short integration window). Scale up `n_vocs`, `n_epochs`, `batch_size`, and
`n_integrate` for a real run.

Run from the repo root:
    python -m examples.arneodo_autonomous --out-dir ./arneodo_run
(needs a working CUDA torch build -- the model is hardwired to CUDA.)
"""

import argparse
import os

# headless plotting; model_vis sets text.usetex=True at import, which breaks on
# machines without LaTeX -- force it off after matplotlib is configured.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import jax.random as jr

from data.generate_data_gabo import generate_vocal_dataset
from data.load_data import get_segmented_audio
from train.train_model import train_model
from train.eval import integrate_model_autonomous

plt.rcParams["text.usetex"] = False


def peak_freq(x: np.ndarray, dt: float) -> float:
    """dominant (non-DC) frequency of a 1-D signal, in Hz."""
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=dt)
    spec[0] = 0.0
    return float(freqs[np.argmax(spec)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="./arneodo_run", help="output directory")
    parser.add_argument("--n-vocs", type=int, default=10, help="synthetic vocalizations to generate")
    parser.add_argument("--n-epochs", type=int, default=3, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=float, default=0.25, help="chunk length (s)")
    parser.add_argument("--n-integrate", type=int, default=4000, help="samples to autonomously integrate")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--method", default="rk4", help="ODE integration method")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "gabo_data")
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. generate synthetic Mindlin data (audio .wav + onset/offset .txt) ---------------
    if not os.path.isdir(data_dir) or len(
        [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    ) < args.n_vocs:
        print(f"Generating {args.n_vocs} synthetic vocalizations into {data_dir} ...")
        generate_vocal_dataset(
            jr.PRNGKey(args.seed),
            n_vocs=args.n_vocs,
            audio_loc=data_dir,
            seg_loc=data_dir,
            func_loc=data_dir,
        )
    else:
        print(f"Reusing existing data in {data_dir}")

    # 2. train the Arneodo model --------------------------------------------------------
    print("Training ArneodoOuroboros ...")
    model = train_model(
        audio_dirs=[data_dir],
        seg_dirs=[data_dir],
        model_dir=model_dir,
        max_vocs=max(args.n_vocs * 4, 50),
        context_len=args.context_len,
        seed=args.seed,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        save_freq=max(args.n_epochs // 2, 1),
        parameterization="arneodo",
    )
    model.eval()

    # 3. autonomous integration on a held-out segment -----------------------------------
    chunks, sr = get_segmented_audio(
        data_dir,
        data_dir,
        max_vocs=args.n_vocs * 4,
        context_len=args.context_len,
        seed=args.seed + 1,  # different seed -> different shuffle than training selection
        training=True,
        extend=True,
        shuffle_order=True,
    )
    dt = 1 / sr
    segment = np.asarray(chunks[0]).squeeze()
    n = min(args.n_integrate, len(segment))
    segment = segment[:n]

    print(f"\nAutonomously integrating {n} samples ({n * dt * 1e3:.1f} ms) ...")
    x_gen = integrate_model_autonomous(
        model, segment, dt, method=args.method, detrend=True, verbose=True
    )
    print()  # newline after the progress \r

    # detrend the reference the same way for a fair visual comparison
    from train.eval import correct

    true_detrended = correct(segment.astype(np.float64))

    f_true = peak_freq(true_detrended, dt)
    f_gen = peak_freq(x_gen, dt)
    corr = float(np.corrcoef(true_detrended[: len(x_gen)], x_gen)[0, 1])
    print(f"peak frequency  true: {f_true:7.1f} Hz   generated: {f_gen:7.1f} Hz")
    print(f"waveform correlation (true vs generated): {corr:+.3f}")
    print(f"generated range: [{x_gen.min():.3g}, {x_gen.max():.3g}]")

    # 4. plot ---------------------------------------------------------------------------
    t_ms = np.arange(len(x_gen)) * dt * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    ax1.plot(t_ms, true_detrended[: len(x_gen)], label="true (detrended)", color="tab:orange")
    ax1.plot(t_ms, x_gen, label="autonomous", color="tab:blue", alpha=0.8)
    ax1.set_title(
        f"Autonomous integration  |  peak freq true {f_true:.0f} Hz / gen {f_gen:.0f} Hz  |  corr {corr:+.2f}"
    )
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("a.u.")
    ax1.legend()

    zoom = slice(0, min(len(x_gen), int(round(0.02 / dt))))  # first 20 ms
    ax2.plot(t_ms[zoom], true_detrended[: len(x_gen)][zoom], color="tab:orange", label="true")
    ax2.plot(t_ms[zoom], x_gen[zoom], color="tab:blue", alpha=0.8, label="autonomous")
    ax2.set_title("first 20 ms (zoom)")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("a.u.")
    ax2.legend()

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "autonomous_vs_true.svg")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\nSaved comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
