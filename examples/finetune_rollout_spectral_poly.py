"""
Spectral + envelope rollout fine-tune for the polynomial Ouroboros (single-model entry point).

Thin wrapper around train.rollout_refine.rollout_refine (the reusable objective): loads a trained
poly checkpoint, gathers held-out sustained windows, and applies the phase-invariant
multi-resolution-STFT + envelope rollout objective. See train/rollout_refine.py for the loss.

Run from the repo root:
    python -m examples.finetune_rollout_spectral_poly --init poly_alpha_ms/alpha_off_seed0/poly \
        --data-glob 'data500/gabo_p[0-7]' --out-dir ./poly_ftspec_seed0 --lam-env 10
"""

import argparse
import glob
import os

from train.train import load_model, save_model
from train.rollout_refine import gather_windows, rollout_refine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True, help="poly checkpoint dir to fine-tune")
    p.add_argument("--data-glob", default="data500/gabo_p[0-7]")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-windows", type=int, default=12)
    p.add_argument("--l-seg", type=int, default=1500, help="encoder context length (>= hmax)")
    p.add_argument("--hmax", type=int, default=1500, help="max rollout horizon (samples)")
    p.add_argument("--hmin", type=int, default=768, help="initial rollout horizon (curriculum)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam-tf", type=float, default=1.0)
    p.add_argument("--lam-spec", type=float, default=1.0)
    p.add_argument("--lam-env", type=float, default=10.0)
    p.add_argument("--env-ms", type=float, default=2.0)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    args = p.parse_args()

    run_dir = os.path.join(os.path.abspath(args.out_dir), "poly")
    os.makedirs(run_dir, exist_ok=True)
    model, _, _, ep0 = load_model(args.init)
    dt = model.tau

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    X, _ = gather_windows(dirs, args.n_windows, args.l_seg, args.start_offset_ms)
    print(f"spectral FT {args.init} (epoch {ep0}); {X.shape[0]} windows L={args.l_seg}", flush=True)

    _, opt = rollout_refine(model, X, dt, epochs=args.epochs, hmin=args.hmin, hmax=args.hmax,
                            batch_size=args.batch_size, lr=args.lr, lam_spec=args.lam_spec,
                            lam_env=args.lam_env, lam_tf=args.lam_tf, env_ms=args.env_ms)

    cfg = model.omega_mamba.config
    save_model(model, opt, os.path.join(run_dir, f"checkpoint_{ep0 + args.epochs}.tar"),
               n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv,
               expand_factor=cfg.expand_factor)
    print(f"saved fine-tuned model to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
