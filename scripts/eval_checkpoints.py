"""Evaluate every checkpoint in a single-seed run to chart when integration stabilises.

Loads each `checkpoint_{epoch}.tar` from a poly_lam{lam}_seed{s} directory, computes:
    - train R^2 (one-step ẍ-prediction)
    - val R^2
    - mid-voc rescaled autonomy (the legacy selection metric)

Prints a per-epoch table so we can see at what R^2 the autonomous integration first
becomes finite (escapes the diverge_score=-5.0 floor) on real-finch data.

Usage:
    .venv/bin/python scripts/eval_checkpoints.py \\
        --run-dir ./finch_blk445_syllC_diag/poly_lam1.068_seed0 \\
        --data-glob '/home/pearson/ouroboros_data/blk445_syllC/day*'
"""

import argparse
import glob
import os
import re
import shutil
import tempfile

import numpy as np
import torch

from data.load_data import get_segmented_audio
from data.data_utils import get_loaders
from train.train import load_model
from train.eval import autonomy_score, eval_model_error
from examples.run_lambda_pipeline import load_voc_windows


def list_checkpoints(run_dir):
    pat = re.compile(r"checkpoint_(\d+)\.tar$")
    ckpts = []
    for f in glob.glob(os.path.join(run_dir, "checkpoint_*.tar")):
        m = pat.search(os.path.basename(f))
        if m:
            ckpts.append((int(m.group(1)), f))
    ckpts.sort()
    return ckpts


def load_specific(ckpt_path):
    """Load just this checkpoint by hiding the others in a tempdir."""
    tmp = tempfile.mkdtemp(prefix="ckpt_eval_")
    link = os.path.join(tmp, os.path.basename(ckpt_path))
    os.symlink(os.path.abspath(ckpt_path), link)
    try:
        model, _, _, epoch = load_model(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return model, epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="poly_lam{lam}_seed{s} directory containing checkpoint_*.tar")
    p.add_argument("--data-glob", required=True,
                   help="same glob the pipeline was launched with")
    p.add_argument("--max-vocs-per-shard", type=int, default=1000)
    p.add_argument("--context-len", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-val-vocs", type=int, default=3)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    # rebuild the same data state the pipeline used
    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    assert len(dirs) >= 3
    train_dirs, val_dir, _ = dirs[:-2], dirs[-2], dirs[-1]
    chunks, sr = [], None
    for d in train_dirs:
        audio, sr = get_segmented_audio(d, d, max_vocs=args.max_vocs_per_shard,
                                        context_len=args.context_len, seed=args.seed,
                                        training=True, extend=True, shuffle_order=True)
        chunks += audio
    dt = 1.0 / sr
    dls = get_loaders(np.stack(chunks, 0), num_workers=args.n_jobs,
                      batch_size=args.batch_size, train_size=0.6, cv=True,
                      seed=args.seed, dt=dt)
    val_vocs, _ = load_voc_windows(val_dir, args.n_val_vocs,
                                   args.start_offset_ms, args.auto_n)
    print(f"# train chunks={len(chunks)} sr={sr} val_vocs={len(val_vocs)} "
          f"L={len(val_vocs[0]) if val_vocs else 0}")

    ckpts = list_checkpoints(args.run_dir)
    print(f"# {len(ckpts)} checkpoint(s) in {args.run_dir}")
    print(f"{'epoch':>6}  {'train_r2':>10}  {'val_r2':>10}  {'rescaled_auto':>14}  "
          f"{'spec':>6}  {'amp_pen':>8}  {'pitch_pen':>10}  {'bounded':>8}")
    for epoch, path in ckpts:
        model, _ = load_specific(path)
        model.eval()
        with torch.no_grad():
            (_, train_r2), _, _ = eval_model_error(dls, model, dt=dt, comparison="train")
            (_, val_r2), _, _ = eval_model_error(dls, model, dt=dt, comparison="val")
        auto, _, bd = autonomy_score(model, val_vocs, dt, rescale=True, cold_start=False)
        print(f"{epoch:>6d}  {train_r2:>10.4f}  {val_r2:>10.4f}  {auto:>14.4f}  "
              f"{bd['spec_corr']:>6.2f}  {bd['amp_pen']:>8.3f}  "
              f"{bd['pitch_pen']:>10.3f}  {bd['bounded_frac']:>8.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
