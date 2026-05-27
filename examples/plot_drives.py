"""
Compare a trained `ArneodoOuroboros`'s learned drives against the control inputs used to
generate the Mindlin/gabo data -- by unpacking the algebra between the two ODEs.

The two right-hand sides, grouped by monomial in (x, xdot):

    generator  ẍ = -(delta*D)        - (eps1+eps2*K) x   + (beta1+beta2*P) xdot   - C x^2 xdot
    model      ẍ =  g_phys^2 alpha   + g_phys^2 beta x    - g_phys delta_m xdot
                  + g_phys^2 x^2 - g_phys^2 x^3 - g_phys x xdot - g_phys x^2 xdot

(g_phys = gamma / tau is the physical time-scaling; gamma is the model's rescaled scalar.)

Equating coefficients of like monomials, the model drives map to the true control inputs by
INVERTING the generator's affine coefficient maps:

    D_implied = -(g_phys^2 alpha) / delta
    K_implied = (-(g_phys^2 beta) - eps1) / eps2
    P_implied = ((-g_phys delta_m) - beta1) / beta2

so a faithful model should have D_implied ~ D, K_implied ~ K, P_implied ~ P. This is NOT a
1-to-1 identity of the raw drives -- it's the affine relation between matching ODE
coefficients. Two reasons the match is still imperfect: (1) the model has x^2, x^3, x*xdot
terms (all tied to one gamma) and an x^2*xdot coeff -g_phys != -C that the generator lacks,
so it is structurally a different ODE; (2) per timestep ẍ is one equation in three unknown
drives, so the split is unidentifiable unless the drives are slow -- they are not (they
oscillate at the carrier), so we compare their low-pass (slow) component.

Run from the repo root:
    python -m examples.plot_drives --out-dir ./arneodo_run
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

from utils import deriv_approx_dy
from train.train import load_model
from data.generate_data_gabo import eps1, eps2, beta1, beta2, C, delta as delta_gen

plt.rcParams["text.usetex"] = False


def lowpass(x, cutoff_hz, fs, order=4):
    """zero-phase Butterworth low-pass; extracts the slow (control-rate) component."""
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="./arneodo_run")
    parser.add_argument("--voc", type=int, default=0, help="which gabo_artificial_<voc> file")
    parser.add_argument("--vocalization", type=int, default=0, help="which onset/offset pair")
    parser.add_argument("--pad-ms", type=float, default=20.0, help="window padding each side (ms)")
    parser.add_argument("--cutoff-hz", type=float, default=50.0,
                        help="low-pass cutoff for the learned drives' slow component")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "gabo_data")
    model_dir = os.path.join(out_dir, "model", "arneodo")
    tag = f"gabo_artificial_{args.voc}"

    model, _, _, epoch = load_model(model_dir)
    model.eval()
    g_phys = float(model.gamma) / model.tau
    print(f"loaded {type(model).__name__} (epoch {epoch}); g_phys = gamma/tau = {g_phys:.1f}")
    print(
        "model's SPURIOUS / mismatched constant coefficients (generator has 0 / -C):\n"
        f"  x^2   coeff = +g_phys^2 = {g_phys**2:+.3e}   (generator: 0)\n"
        f"  x^3   coeff = -g_phys^2 = {-g_phys**2:+.3e}   (generator: 0)\n"
        f"  x*xd  coeff = -g_phys   = {-g_phys:+.3e}       (generator: 0)\n"
        f"  x^2xd coeff = -g_phys   = {-g_phys:+.3e}       (generator -C = {-C:+.3e})"
    )

    # data: audio (x), the three control inputs, and the vocalization interval -----------
    sr, audio_full = wavfile.read(os.path.join(data_dir, f"{tag}.wav"))
    audio_full = audio_full.astype(np.float64)
    pkd = np.loadtxt(os.path.join(data_dir, f"{tag}_PKD.txt"))  # cols: P, K, D
    P_full, K_full, D_full = pkd[:, 0], pkd[:, 1], pkd[:, 2]
    onoffs = np.atleast_2d(np.loadtxt(os.path.join(data_dir, f"{tag}.txt")))
    on_s, off_s = onoffs[args.vocalization]
    dt = 1 / sr

    pad = int(round(args.pad_ms / 1e3 * sr))
    on_i, off_i = int(round(on_s * sr)), int(round(off_s * sr))
    a, b = max(0, on_i - pad), min(len(audio_full), off_i + pad)
    t_s = np.arange(a, b) * dt
    audio = audio_full[a:b]
    P, K, D = P_full[a:b], K_full[a:b], D_full[a:b]

    # learned drives over the same window ------------------------------------------------
    audio_t = torch.from_numpy(audio[None, :, None]).to(torch.float32).to("cuda")
    dy_t = torch.from_numpy(deriv_approx_dy(audio[None, :, None])).to(torch.float32).to("cuda")
    with torch.no_grad():
        alpha, beta, delta_m, _ = model.get_funcs(audio_t, dy_t, dt)
    alpha = alpha.detach().cpu().numpy().squeeze()
    beta = beta.detach().cpu().numpy().squeeze()
    delta_m = delta_m.detach().cpu().numpy().squeeze()

    # unpack the algebra: model-implied control inputs (true control-input units) --------
    g2 = g_phys**2
    D_impl = -(g2 * alpha) / delta_gen
    K_impl = (-(g2 * beta) - eps1) / eps2
    P_impl = ((-g_phys * delta_m) - beta1) / beta2

    # the implied inputs oscillate at the carrier (drives are unconstrained per-sample);
    # compare their slow component to the (slow) true control inputs
    D_impl_lp = lowpass(D_impl, args.cutoff_hz, sr)
    K_impl_lp = lowpass(K_impl, args.cutoff_hz, sr)
    P_impl_lp = lowpass(P_impl, args.cutoff_hz, sr)

    voc = slice(on_i - a, off_i - a)

    def corr(m, d):
        m, d = m[voc], d[voc]
        if m.std() < 1e-12 or d.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(m, d)[0, 1])

    panels = [
        ("D (dynamics)", D, D_impl, D_impl_lp, "constant forcing  (alpha)"),
        ("K (tension)", K, K_impl, K_impl_lp, "restoring / frequency  (beta)"),
        ("P (pressure)", P, P_impl, P_impl_lp, "linear damping  (delta)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (name, true, impl_raw, impl_lp, slot) in zip(axes, panels):
        r = corr(impl_lp, true)
        ax.plot(t_s, impl_raw, color="tab:blue", lw=0.5, alpha=0.12)
        ax.plot(t_s, true, color="tab:orange", lw=1.8, label=f"true {name}")
        ax.plot(t_s, impl_lp, color="tab:blue", lw=1.6,
                label=f"model-implied {name} (<{args.cutoff_hz:.0f} Hz)")
        ax.axvline(on_s, color="0.6", ls="--", lw=0.8)
        ax.axvline(off_s, color="0.6", ls="--", lw=0.8)
        ax.set_ylabel(name)
        ax.set_title(f"{slot}:  model-implied vs true {name}   (r over vocalization = {r:+.2f})")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")

    fig.suptitle(
        f"Model-implied vs true control inputs (affine maps inverted)  |  {tag}  |  "
        f"vocalization {args.vocalization}",
        y=1.00,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "learned_vs_true_drives.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
