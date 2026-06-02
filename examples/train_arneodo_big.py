"""
Configurable, monitored training of an ArneodoOuroboros toward a target R^2.

Gathers chunks from one or more gabo data directories, builds dataloaders, and trains in
segments -- evaluating train/test R^2 after each segment, checkpointing, and stopping early
once the target test R^2 is reached. Capacity (n_layers, d_state, d_conv, expand_factor) is
exposed so we can give the model enough capacity to fit the finite-difference target.

Run from the repo root, e.g.:
    python -m examples.train_arneodo_big --data-glob './data500/gabo_*' \
        --out-dir ./arneodo_big --d-state 16 --n-layers 4 --epochs 200 --seg 10 \
        --batch-size 32 --target-r2 0.97
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.load_data import get_segmented_audio
from data.data_utils import get_loaders
from model.model import ArneodoOuroboros
from train.train import train, save_model
from train.eval import eval_model_error
from utils import sse


def gather_loaders(data_dirs, max_vocs, context_len, batch_size, seed, n_jobs):
    chunks = []
    sr = None
    per_dir = max(1, max_vocs // len(data_dirs))
    for d in data_dirs:
        audio, sr = get_segmented_audio(
            d, d, max_vocs=per_dir, context_len=context_len, seed=seed,
            training=True, extend=True, shuffle_order=True,
        )
        chunks += audio
    print(f"gathered {len(chunks)} chunks from {len(data_dirs)} dir(s); sr={sr}")
    dt = 1 / sr
    dls = get_loaders(
        np.stack(chunks, axis=0), num_workers=n_jobs, batch_size=batch_size,
        train_size=0.6, cv=True, seed=seed, dt=dt,
    )
    return dls, dt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", required=True, help="glob matching gabo data dir(s)")
    p.add_argument("--out-dir", default="./arneodo_big")
    p.add_argument("--max-vocs", type=int, default=100000)
    p.add_argument("--context-len", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seg", type=int, default=10, help="eval/checkpoint every this many epochs")
    p.add_argument("--target-r2", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand-factor", type=int, default=10)
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0,
                   help="hard low-pass alpha/beta/delta at this Gaussian timescale (ms); "
                        "default 1 ms (slow drives, cold-start stable). 0 disables.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    run_dir = os.path.join(out_dir, "arneodo")
    os.makedirs(run_dir, exist_ok=True)

    data_dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    assert data_dirs, f"no dirs matched {args.data_glob}"

    dls, dt = gather_loaders(
        data_dirs, args.max_vocs, args.context_len, args.batch_size, args.seed, args.n_jobs
    )

    model = ArneodoOuroboros(
        d_data=1, n_layers=args.n_layers, d_state=args.d_state, d_conv=args.d_conv,
        expand_factor=args.expand_factor, tau=dt, drive_lowpass_ms=args.drive_lowpass_ms,
    )
    n_params = sum(q.numel() for q in model.parameters())
    print(f"model: n_layers={args.n_layers} d_state={args.d_state} d_conv={args.d_conv} "
          f"expand={args.expand_factor} lowpass={args.drive_lowpass_ms}ms -> {n_params} params; tau={dt:.2e}")

    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=max(args.seg, 3), min_lr=1e-10)
    model_info = {"n layers": args.n_layers, "d state": args.d_state,
                  "d conv": args.d_conv, "expand factor": args.expand_factor}

    best = -np.inf
    for start in range(0, args.epochs, args.seg):
        end = min(args.epochs, start + args.seg)
        train(
            model, opt,
            loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
            loaders=dls, scheduler=scheduler, nEpochs=end, val_freq=1, runDir=run_dir,
            dt=dt, vis_freq=0, smoothing=False, reg_weights=False, start_epoch=start,
            save_freq=max(args.seg, 1), model_info=model_info,
        )
        model.eval()
        with torch.no_grad():
            (tr, te), (trsd, tesd), _ = eval_model_error(dls, model, dt=dt, comparison="test")
        print(f"[epoch {end}] train R2={tr:.4f}+-{trsd:.4f}  test R2={te:.4f}+-{tesd:.4f}",
              flush=True)
        save_model(model, opt, os.path.join(run_dir, f"checkpoint_{end}.tar"),
                   n_layers=args.n_layers, d_state=args.d_state,
                   d_conv=args.d_conv, expand_factor=args.expand_factor)
        best = max(best, te)
        if te >= args.target_r2:
            print(f"reached target test R2 {args.target_r2} at epoch {end}", flush=True)
            break

    print(f"DONE. best test R2 = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
