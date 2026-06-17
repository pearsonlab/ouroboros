"""
Integrated trainer for the (low-pass) polynomial Ouroboros: teacher-forced training THEN the
phase-invariant spectral+envelope rollout-refinement phase (train.rollout_refine), in one run.

This folds the autonomous-reconstruction objective into the training pipeline rather than running it
as a separate post-hoc fine-tune. Produces two checkpoints under <out-dir>:
  - tf/checkpoint_<tf_epochs>.tar          : teacher-forced only (reference / ablation)
  - poly/checkpoint_<tf+rollout>.tar       : TF + rollout-refined (final)

Run from the repo root:
    python -m examples.train_poly_rollout --data-glob 'data500/gabo_p[0-7]' --out-dir ./poly_ro_seed0 \
        --tf-epochs 30 --rollout-epochs 8 --lam-env 10 --drive-lowpass-ms 1.0 --d-state 4 \
        --context-len 0.1 --batch-size 8 --seed 0
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.model import Ouroboros
from model.kernels import fullPolyModule
from train.train import train, save_model
from train.eval import eval_model_error
from train.rollout_refine import gather_windows, rollout_refine
from utils import sse
from examples.train_poly_lowpass import gather_loaders


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir", required=True)
    # teacher-forced phase
    p.add_argument("--max-vocs", type=int, default=100000)
    p.add_argument("--context-len", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tf-epochs", type=int, default=30)
    p.add_argument("--seg", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand-factor", type=int, default=10)
    p.add_argument("--n-kernels", type=int, default=15)
    p.add_argument("--lam", type=float, default=1.068)
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0)
    p.add_argument("--keep-const", action="store_true")
    # rollout-refine phase
    p.add_argument("--rollout-epochs", type=int, default=8)
    p.add_argument("--rollout-hmin", type=int, default=768)
    p.add_argument("--rollout-hmax", type=int, default=1500)
    p.add_argument("--rollout-l-seg", type=int, default=1500)
    p.add_argument("--rollout-windows", type=int, default=12)
    p.add_argument("--rollout-batch", type=int, default=6)
    p.add_argument("--rollout-lr", type=float, default=1e-4)
    p.add_argument("--lam-spec", type=float, default=1.0)
    p.add_argument("--lam-env", type=float, default=10.0)
    p.add_argument("--lam-tf", type=float, default=1.0)
    p.add_argument("--env-ms", type=float, default=2.0)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    tf_dir = os.path.join(out_dir, "tf")
    poly_dir = os.path.join(out_dir, "poly")
    os.makedirs(tf_dir, exist_ok=True)
    os.makedirs(poly_dir, exist_ok=True)
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
    print(f"poly Ouroboros: lowpass={args.drive_lowpass_ms}ms lam={args.lam} keep_const={args.keep_const} "
          f"-> {n_params} params; tau={dt:.2e}", flush=True)

    # ---- teacher-forced phase ----
    opt = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, factor=0.5, patience=max(args.seg, 3), min_lr=1e-10)
    model_info = {"n layers": args.n_layers, "d state": args.d_state,
                  "d conv": args.d_conv, "expand factor": args.expand_factor}
    for start in range(0, args.tf_epochs, args.seg):
        end = min(args.tf_epochs, start + args.seg)
        train(model, opt, loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
              loaders=dls, scheduler=scheduler, nEpochs=end, val_freq=1, runDir=poly_dir,
              dt=dt, vis_freq=0, smoothing=False, reg_weights=True, start_epoch=start,
              save_freq=max(args.seg, 1), model_info=model_info)
    model.eval()
    with torch.no_grad():
        (tr, te), _, _ = eval_model_error(dls, model, dt=dt, comparison="test")
    print(f"[TF done] tf_epochs={args.tf_epochs} train R2={tr:.4f} test R2={te:.4f}", flush=True)
    save_model(model, opt, os.path.join(tf_dir, f"checkpoint_{args.tf_epochs}.tar"),
               n_layers=args.n_layers, d_state=args.d_state, d_conv=args.d_conv,
               expand_factor=args.expand_factor)

    # ---- rollout-refine phase ----
    X, _ = gather_windows(data_dirs, args.rollout_windows, args.rollout_l_seg, args.start_offset_ms)
    print(f"rollout windows: {X.shape[0]} x {X.shape[1]} samples", flush=True)
    _, ro_opt = rollout_refine(model, X, dt, epochs=args.rollout_epochs, hmin=args.rollout_hmin,
                               hmax=args.rollout_hmax, batch_size=args.rollout_batch, lr=args.rollout_lr,
                               lam_spec=args.lam_spec, lam_env=args.lam_env, lam_tf=args.lam_tf,
                               env_ms=args.env_ms)
    model.eval()
    with torch.no_grad():
        (tr2, te2), _, _ = eval_model_error(dls, model, dt=dt, comparison="test")
    print(f"[refine done] test R2 {te:.4f} -> {te2:.4f}", flush=True)
    save_model(model, ro_opt, os.path.join(poly_dir, f"checkpoint_{args.tf_epochs + args.rollout_epochs}.tar"),
               n_layers=args.n_layers, d_state=args.d_state, d_conv=args.d_conv,
               expand_factor=args.expand_factor)
    print(f"DONE seed{args.seed}: tf R2={te:.4f} final R2={te2:.4f}  ({tf_dir} | {poly_dir})", flush=True)


if __name__ == "__main__":
    main()
