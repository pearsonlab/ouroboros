"""
Quick A/B: does the constant (0,0) "alpha" forcing term help AUTONOMOUS reconstruction?

Scores the +alpha (zero-init, 1ms low-pass) poly Ouroboros against the no-alpha baseline on the
SAME held-out gabo_p9 windows, with train.eval.autonomy_score (deterministic closed-loop, log-PSD
spectral corr minus amp/pitch log-ratio penalties). Caveat: the available baseline is at 2ms low-pass,
not 1ms, so this is the no-alpha point we have, not a perfectly matched control.

    python -m examples.eval_alpha_autonomy --data-dir data500/gabo_p9 --n-vocs 6
"""

import argparse
import glob
import os

import numpy as np
from scipy.io import wavfile

from train.train import load_model
from train.eval import autonomy_score


def load_voc_windows(data_dir, n_vocs, start_offset_ms, n):
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        s = int(onoffs[0][0] * sr) + int(start_offset_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs, sr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data500/gabo_p9")
    p.add_argument("--n-vocs", type=int, default=6)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--alpha-dir", default="poly_alpha/poly")
    p.add_argument("--baseline-dir", default="poly_pipeline/poly_lam1.068_seed0")
    args = p.parse_args()

    segs, sr = load_voc_windows(args.data_dir, args.n_vocs, args.start_offset_ms, args.auto_n)
    print(f"{len(segs)} held-out vocs from {args.data_dir} (n={args.auto_n}, sr={sr})", flush=True)

    runs = [("+alpha (1ms, zero-init)", args.alpha_dir),
            ("no-alpha baseline (2ms)", args.baseline_dir)]
    for label, d in runs:
        model, _, _, _ = load_model(d)
        model.eval()
        dt = model.tau
        keep = getattr(model, "keep_const", False)
        lp = getattr(model, "drive_lowpass_ms", None)
        mean_score, per_seg, bd = autonomy_score(model, segs, dt)
        print(f"\n=== {label} ===", flush=True)
        print(f"  keep_const={keep}  drive_lowpass_ms={lp}", flush=True)
        print(f"  autonomy score   = {mean_score:+.3f}", flush=True)
        print(f"  spectral corr    = {bd['spec_corr']:.3f}", flush=True)
        print(f"  amp penalty      = {bd['amp_pen']:.3f}", flush=True)
        print(f"  pitch penalty    = {bd['pitch_pen']:.3f}", flush=True)
        print(f"  bounded fraction = {bd['bounded_frac']:.2f}", flush=True)
        print(f"  per-voc scores   = {np.round(per_seg, 3)}", flush=True)


if __name__ == "__main__":
    main()
