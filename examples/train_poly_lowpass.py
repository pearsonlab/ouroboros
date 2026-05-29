"""
Train the ORIGINAL polynomial Ouroboros with drive low-pass "in the loop" (omega, gamma,
and the kernel weights are Gaussian low-passed at drive_lowpass_ms), monitored toward high R^2.

Same idea the other Claude applied to ArneodoOuroboros, pushed back to the original model so we
can test whether slow drives improve autonomous reconstruction for the polynomial parameterization.

Run from the repo root, e.g.:
    python -m examples.train_poly_lowpass --data-glob './data500/gabo_p*' --out-dir ./poly_lp \
        --drive-lowpass-ms 2.0 --d-state 4 --epochs 60 --seg 10 --batch-size 8
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
from model.model import Ouroboros
from model.kernels import fullPolyModule
from train.train import train, save_model
from train.eval import eval_model_error
from utils import sse


def gather_loaders(data_dirs, max_vocs, context_len, batch_size, seed, n_jobs):
    chunks, sr = [], None
    per = max(1, max_vocs // len(data_dirs))
    for d in data_dirs:
        audio, sr = get_segmented_audio(d, d, max_vocs=per, context_len=context_len, seed=seed,
                                        training=True, extend=True, shuffle_order=True)
        chunks += audio
    print(f"gathered {len(chunks)} chunks from {len(data_dirs)} dir(s); sr={sr}", flush=True)
    dt = 1 / sr
    dls = get_loaders(np.stack(chunks, 0), num_workers=n_jobs, batch_size=batch_size,
                      train_size=0.6, cv=True, seed=seed, dt=dt)
    return dls, dt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir", default="./poly_lp")
    p.add_argument("--max-vocs", type=int, default=100000)
    p.add_argument("--context-len", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seg", type=int, default=10)
    p.add_argument("--target-r2", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand-factor", type=int, default=10)
    p.add_argument("--n-kernels", type=int, default=15)
    p.add_argument("--lam", type=float, default=1.2, help="kernel-weight regularization base")
    p.add_argument("--drive-lowpass-ms", type=float, default=2.0)
    p.add_argument("--keep-const", action="store_true",
                   help="add the constant (0,0) 'alpha' forcing term (low-passed with the other drives)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(os.path.abspath(args.out_dir), "poly")
    os.makedirs(run_dir, exist_ok=True)
    data_dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    assert data_dirs, f"no dirs matched {args.data_glob}"

    dls, dt = gather_loaders(data_dirs, args.max_vocs, args.context_len, args.batch_size,
                             args.seed, args.n_jobs)

    kernel = fullPolyModule(nTerms=args.n_kernels, device="cuda", x_dim=1, z_dim=2,
                            activation=lambda x: x, lam=args.lam)
    model = Ouroboros(d_data=1, kernel=kernel, n_layers=args.n_layers, d_state=args.d_state,
                      d_conv=args.d_conv, expand_factor=args.expand_factor, tau=dt,
                      drive_lowpass_ms=args.drive_lowpass_ms, keep_const=args.keep_const)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"poly Ouroboros: n_kernels={args.n_kernels} d_state={args.d_state} expand={args.expand_factor} "
          f"lowpass={args.drive_lowpass_ms}ms lam={args.lam} -> {n_params} params; tau={dt:.2e}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=max(args.seg, 3), min_lr=1e-10)
    model_info = {"n layers": args.n_layers, "d state": args.d_state,
                  "d conv": args.d_conv, "expand factor": args.expand_factor}

    best = -np.inf
    for start in range(0, args.epochs, args.seg):
        end = min(args.epochs, start + args.seg)
        train(model, opt, loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
              loaders=dls, scheduler=scheduler, nEpochs=end, val_freq=1, runDir=run_dir,
              dt=dt, vis_freq=0, smoothing=False, reg_weights=True, start_epoch=start,
              save_freq=max(args.seg, 1), model_info=model_info)
        model.eval()
        with torch.no_grad():
            (tr, te), (trsd, tesd), _ = eval_model_error(dls, model, dt=dt, comparison="test")
        print(f"[epoch {end}] train R2={tr:.4f}  test R2={te:.4f}", flush=True)
        save_model(model, opt, os.path.join(run_dir, f"checkpoint_{end}.tar"),
                   n_layers=args.n_layers, d_state=args.d_state, d_conv=args.d_conv,
                   expand_factor=args.expand_factor)
        best = max(best, te)
        if te >= args.target_r2:
            break
    print(f"DONE. best test R2 = {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
