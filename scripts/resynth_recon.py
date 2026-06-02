"""Generate an autonomous-reconstruction WAV from a chosen checkpoint, using the
fixed (int16-normalized) val/test loader. Also writes the target voc as a WAV for
side-by-side listening.
"""

import argparse
import glob
import os
import shutil
import tempfile

import numpy as np
import torch
from scipy.io import wavfile

from train.train import load_model
from train.eval import generate_autonomous, autonomy_score
from examples.run_lambda_pipeline import load_voc_windows


def load_specific(ckpt_path):
    tmp = tempfile.mkdtemp(prefix="ckpt_synth_")
    link = os.path.join(tmp, os.path.basename(ckpt_path))
    os.symlink(os.path.abspath(ckpt_path), link)
    try:
        model, _, _, _ = load_model(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return model


def to_wav(x, sr, path):
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.abs(x).max())
    if not np.isfinite(peak) or peak < 1e-12:
        print(f"  WARNING: non-finite or empty signal at {path}")
        return
    y = (x / peak * 0.95 * 32767).astype(np.int16)
    wavfile.write(path, sr, y)
    print(f"  wrote {path}  ({len(y)} samples, sr={sr}, peak_in={peak:.4e})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-vocs", type=int, default=3)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--auto-n", type=int, default=3000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_specific(args.ckpt)
    model.eval()
    print(f"loaded {args.ckpt}")

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    test_dir = dirs[-1]
    print(f"test_dir={test_dir}")
    vocs, sr = load_voc_windows(test_dir, args.n_vocs, args.start_offset_ms, args.auto_n)
    dt = 1.0 / sr
    print(f"sr={sr} n_vocs={len(vocs)} L={len(vocs[0]) if vocs else 0}")

    score, per_seg, bd = autonomy_score(model, vocs, dt, rescale=True, cold_start=False)
    print(f"mean rescaled autonomy = {score:+.4f}  (spec={bd['spec_corr']:.2f}, "
          f"pitch_pen={bd['pitch_pen']:.2f}, bounded={bd['bounded_frac']:.2f})")
    print(f"per-voc scores: {[f'{s:+.3f}' for s in per_seg]}")

    for i, voc in enumerate(vocs):
        target = np.asarray(voc, dtype=np.float64)
        recon = generate_autonomous(model, voc, dt, rescale=True, detrend=True, verbose=False)
        to_wav(target, sr, os.path.join(args.out_dir, f"target_voc{i}.wav"))
        to_wav(recon, sr, os.path.join(args.out_dir, f"recon_voc{i}.wav"))


if __name__ == "__main__":
    main()
