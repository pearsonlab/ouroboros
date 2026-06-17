"""
Short rollout fine-tune for the polynomial Ouroboros (controlled experiment).

Backprops through a SHORT autonomous rollout (started from the data initial state, so phase
stays aligned and the pointwise MSE pins amplitude + frequency rather than collapsing on
long-horizon phase drift), with SOFT state saturation (x <- B*tanh(x/B), so gradients flow
even if a step would diverge -- fixing the hard-clamp bug from the earlier Arneodo attempt),
plus a teacher-forced anchor and a curriculum on the rollout horizon.

Poly RHS (rescaled time, drives low-passed by the model): d2x/ds2 = -omega^2 x - gamma x'
- sum_ij w_ij x^i x'^j, with omega(t), gamma(t), w(t) from the (differentiable) encoder.

Run from the repo root:
    python -m examples.finetune_rollout_poly --init poly_pipeline/poly_lam1.068_seed0 \
        --data-glob 'data500/gabo_p[0-7]' --out-dir ./poly_ft_seed0
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam
from scipy.io import wavfile

from utils import deriv_approx_dy, deriv_approx_d2y
from train.train import load_model, save_model

BX, BXP = 0.5, 1.0  # soft-saturation bounds (>> data/limit-cycle scale; only tame divergence)


def gather_windows(dirs, n, L, start_off_ms):
    segs, sr = [], None
    for d in dirs:
        for wav in sorted(glob.glob(os.path.join(d, "*.wav"))):
            sr, af = wavfile.read(wav)
            af = af.astype(np.float64)
            on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
            s = int(on * sr) + int(start_off_ms / 1e3 * sr)
            seg = af[s:s + L]
            if len(seg) == L:
                segs.append(seg)
            if len(segs) >= n:
                return np.stack(segs)[:, :, None], sr
    return np.stack(segs)[:, :, None], sr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True, help="poly checkpoint dir to fine-tune")
    p.add_argument("--data-glob", default="data500/gabo_p[0-7]")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-windows", type=int, default=48)
    p.add_argument("--l-seg", type=int, default=1500, help="encoder context length")
    p.add_argument("--hmax", type=int, default=400, help="max rollout horizon (samples)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam-tf", type=float, default=1.0)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    args = p.parse_args()

    run_dir = os.path.join(os.path.abspath(args.out_dir), "poly")
    os.makedirs(run_dir, exist_ok=True)
    model, _, _, ep0 = load_model(args.init)
    model.train()
    dt = model.tau

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    X, sr = gather_windows(dirs, args.n_windows, args.l_seg, args.start_offset_ms)
    D1 = deriv_approx_dy(X)
    D2 = deriv_approx_d2y(X)
    Xt = torch.tensor(X, dtype=torch.float32, device="cuda")
    Dt = torch.tensor(D1, dtype=torch.float32, device="cuda")
    D2t = torch.tensor(D2, dtype=torch.float32, device="cuda")
    var_x, var_d2 = float(Xt.var()), float(D2t.var())
    P = model.kernel.poly_dim + 1
    powers = torch.arange(P, device="cuda")
    opt = Adam(model.parameters(), lr=args.lr)
    N = Xt.shape[0]
    H_sched = np.unique(np.round(np.geomspace(80, args.hmax, args.epochs)).astype(int))
    print(f"fine-tuning {args.init} (epoch {ep0}); {N} windows L={args.l_seg} Hmax={args.hmax}", flush=True)

    for epoch in range(args.epochs):
        H = int(H_sched[min(epoch, len(H_sched) - 1)])
        perm = torch.randperm(N)
        tot_r = tot_t = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, dxd, d2b = Xt[idx], Dt[idx], D2t[idx]
            z2 = (model.tau / dt) * dxd
            omega, gamma, wk, weights, _ = model.get_funcs(x, dxd.clone(), dt)  # differentiable
            tf = -(omega ** 2) * x - gamma * z2 - wk
            L_tf = ((tf - d2b) ** 2).mean() / var_d2

            om = omega[:, :, 0]
            ga = gamma[:, :, 0]
            w = weights  # (B, L, P, P)

            def f(xx, vv, k):
                xpw = xx.unsqueeze(1) ** powers
                xvw = vv.unsqueeze(1) ** powers
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w[:, k])
                return vv, -(om[:, k] ** 2) * xx - ga[:, k] * vv - kern

            xc = x[:, 0, 0].detach()
            xp = z2[:, 0, 0].detach()
            xs = [xc]
            for k in range(H - 1):
                k1x, k1v = f(xc, xp, k)
                k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v, k)
                k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v, k)
                k4x, k4v = f(xc + k3x, xp + k3v, k)
                xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
                xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
                xc = BX * torch.tanh(xc / BX)
                xp = BXP * torch.tanh(xp / BXP)
                xs.append(xc)
            xg = torch.stack(xs, dim=1)  # (B, H)
            L_roll = ((xg - x[:, :H, 0]) ** 2).mean() / var_x

            loss = L_roll + args.lam_tf * L_tf
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_r += float(L_roll)
            tot_t += float(L_tf)
            nb += 1
        print(f"[ep {epoch + 1}/{args.epochs} H={H}] relMSE roll={tot_r / nb:.4f} tf={tot_t / nb:.4f}", flush=True)

    cfg = model.omega_mamba.config
    save_model(model, opt, os.path.join(run_dir, f"checkpoint_{ep0 + args.epochs}.tar"),
               n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv,
               expand_factor=cfg.expand_factor)
    print(f"saved fine-tuned model to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
