"""
Sweep the drive low-pass timescale (ms) for ArneodoOuroboros and report the
R²-vs-smoothness tradeoff. Gathers dataloaders once, trains one model per cutoff.

Run from the repo root:
    python -m examples.sweep_lowpass --data-glob './data500/gabo_*' --out-dir ./arneodo_sweep \
        --lowpass-ms 1 2 3 4 5 --epochs 30
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import welch

from model.model import ArneodoOuroboros
from train.train import train, save_model
from train.eval import eval_model_error
from utils import sse
from examples.train_arneodo_big import gather_loaders


def beta_peak_freq(model, dls, dt):
    """dominant frequency (Hz) of the learned beta drive on a test batch."""
    x, dxdt, _ = next(iter(dls["test"]))
    x = x.cuda().float(); dxdt = dxdt.cuda().float()
    with torch.no_grad():
        _, be, _, _ = model.get_funcs(x, dxdt, dt)
    be = be[0].detach().cpu().numpy().squeeze()
    f, P = welch(be - be.mean(), fs=1 / dt, nperseg=min(1024, len(be)))
    P[0] = 0.0
    return float(f[np.argmax(P)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir", default="./arneodo_sweep")
    p.add_argument("--lowpass-ms", type=float, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-vocs", type=int, default=100000)
    p.add_argument("--context-len", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand-factor", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    data_dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    assert data_dirs, f"no dirs matched {args.data_glob}"
    dls, dt = gather_loaders(
        data_dirs, args.max_vocs, args.context_len, args.batch_size, args.seed, args.n_jobs
    )
    info = {"n layers": args.n_layers, "d state": args.d_state,
            "d conv": args.d_conv, "expand factor": args.expand_factor}

    rows = []
    for lp in args.lowpass_ms:
        run_dir = os.path.join(os.path.abspath(args.out_dir), f"lp_{lp:g}ms", "arneodo")
        os.makedirs(run_dir, exist_ok=True)
        model = ArneodoOuroboros(
            d_data=1, n_layers=args.n_layers, d_state=args.d_state, d_conv=args.d_conv,
            expand_factor=args.expand_factor, tau=dt, drive_lowpass_ms=lp,
        )
        opt = Adam(model.parameters(), lr=args.lr)
        sched = ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-10)
        train(model, opt, loss_fn=lambda y, yhat: sse(yhat, y, reduction="mean"),
              loaders=dls, scheduler=sched, nEpochs=args.epochs, val_freq=1, runDir=run_dir,
              dt=dt, vis_freq=0, smoothing=False, reg_weights=False, start_epoch=0,
              save_freq=args.epochs, model_info=info)
        model.eval()
        with torch.no_grad():
            (tr, te), _, _ = eval_model_error(dls, model, dt=dt, comparison="test")
        pf = beta_peak_freq(model, dls, dt)
        save_model(model, opt, os.path.join(run_dir, f"checkpoint_{args.epochs}.tar"),
                   n_layers=args.n_layers, d_state=args.d_state,
                   d_conv=args.d_conv, expand_factor=args.expand_factor)
        rows.append((lp, tr, te, pf))
        print(f">>> lowpass={lp:g}ms  train R2={tr:.4f}  test R2={te:.4f}  beta peak={pf:.0f}Hz", flush=True)

    print("\n==== low-pass timescale sweep (cutoff ~ 1/(2*pi*ms)) ====")
    print("  ms |  ~cutoff Hz | train R2 | test R2 | beta-drive peak Hz")
    for lp, tr, te, pf in rows:
        print(f" {lp:>3g} | {1/(2*np.pi*lp/1e3):>9.0f} | {tr:>7.3f} | {te:>6.3f} | {pf:>6.0f}")
    print("(reference: no low-pass -> test R2 ~0.987, beta peak ~2266 Hz)")


if __name__ == "__main__":
    main()
