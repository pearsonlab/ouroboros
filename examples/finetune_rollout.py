"""
Short rollout-training fine-tune of an ArneodoOuroboros for cold-start stability.

Teacher-forced training only optimizes one-step ẍ prediction; the resulting free-running
(autonomous) dynamics can be unstable from a cold start (near-silence IC, near the model's
unstable fixed point at the origin). Here we fine-tune by backpropagating through a short
*autonomous rollout*: the model produces the control series alpha/beta/delta from the data,
we integrate the ODE from a cold-start IC feeding the generated state back in (a differentiable
RK4, state-clamped to avoid NaN), and add an MSE between the rolled-out waveform and the data.
A teacher-forced anchor (one-step ẍ MSE) keeps the model oscillating. A curriculum grows the
rollout horizon so the (initially-diverging) model can stabilize progressively.

Run from the repo root, e.g.:
    python -m examples.finetune_rollout --init arneodo_big/arneodo --data-glob './data500/gabo_p*' \
        --out-dir ./arneodo_ft --epochs 12 --batch-size 8
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam

from data.load_data import get_segmented_audio
from utils import deriv_approx_dy, deriv_approx_d2y
from train.train import load_model, save_model

BOUND_X, BOUND_XP = 3.0, 50.0  # clamp the rollout state to keep losses finite (no NaN)


def rhs(x, xp, a, b, d, g):
    g2 = g * g
    dxp = g2 * a + g2 * b * x + g2 * x * x - g2 * x * x * x - g * d * xp - g * x * xp - g * x * x * xp
    return xp, dxp


def rollout(a, b, d, g, x0, xp0, H):
    """differentiable RK4 autonomous rollout (rescaled time, step ds=1). a,b,d: (B, L)."""
    x, xp = x0, xp0
    xs = [x]
    for k in range(H - 1):
        a0, b0, d0 = a[:, k], b[:, k], d[:, k]
        a1, b1, d1 = a[:, k + 1], b[:, k + 1], d[:, k + 1]
        ah, bh, dh = 0.5 * (a0 + a1), 0.5 * (b0 + b1), 0.5 * (d0 + d1)
        k1x, k1v = rhs(x, xp, a0, b0, d0, g)
        k2x, k2v = rhs(x + 0.5 * k1x, xp + 0.5 * k1v, ah, bh, dh, g)
        k3x, k3v = rhs(x + 0.5 * k2x, xp + 0.5 * k2v, ah, bh, dh, g)
        k4x, k4v = rhs(x + k3x, xp + k3v, a1, b1, d1, g)
        x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        x = torch.clamp(x, -BOUND_X, BOUND_X)
        xp = torch.clamp(xp, -BOUND_XP, BOUND_XP)
        xs.append(x)
    return torch.stack(xs, dim=1)  # (B, H)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True, help="checkpoint dir to fine-tune from")
    p.add_argument("--data-glob", required=True)
    p.add_argument("--out-dir", default="./arneodo_ft")
    p.add_argument("--n-vocs", type=int, default=120, help="cold-start segments to fine-tune on")
    p.add_argument("--pre-onset-ms", type=float, default=20.0)
    p.add_argument("--hmax", type=int, default=2000, help="max rollout horizon (samples)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam-roll", type=float, default=1.0, help="weight on (normalized) rollout loss")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.join(os.path.abspath(args.out_dir), "arneodo"), exist_ok=True)
    run_dir = os.path.join(os.path.abspath(args.out_dir), "arneodo")

    model, _, _, ep0 = load_model(args.init)
    model.train()
    dt = model.tau
    print(f"fine-tuning from {args.init} (epoch {ep0}), gamma0={float(model.gamma):.4f}")

    # cold-start segments: analysis mode -> aud[onset - pad : offset], trimmed to hmax
    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    pad = args.pre_onset_ms / 1e3
    segs = []
    per = max(1, args.n_vocs // len(dirs))
    sr = None
    for d in dirs:
        ch, sr = get_segmented_audio(d, d, max_vocs=per, training=False, padding=pad,
                                     seed=args.seed, shuffle_order=True)
        for c in ch:
            c = np.asarray(c).squeeze()
            if len(c) >= args.hmax:
                segs.append(c[: args.hmax])
    X = np.stack(segs, axis=0)[:, :, None].astype(np.float64)  # (N, hmax, 1)
    onset_i = int(round(pad * sr))
    print(f"{X.shape[0]} cold-start segments, hmax={args.hmax}, onset at sample {onset_i}")

    dxdt = deriv_approx_dy(X)
    dx2 = deriv_approx_d2y(X)  # teacher-forced target (per-sample = d2x/ds2 since tau=dt)
    Xt = torch.tensor(X, dtype=torch.float32, device="cuda")
    Dt = torch.tensor(dxdt, dtype=torch.float32, device="cuda")
    D2 = torch.tensor(dx2, dtype=torch.float32, device="cuda")
    var_x = float(Xt.var()); var_d2 = float(D2.var())

    opt = Adam(model.parameters(), lr=args.lr)
    N = Xt.shape[0]
    # curriculum: grow horizon geometrically, starting BELOW the cold-start divergence onset
    # (~40 samples) so early epochs get finite, meaningful rollout gradients.
    H_sched = np.unique(np.round(np.geomspace(32, args.hmax, args.epochs)).astype(int))

    for epoch in range(args.epochs):
        H = int(H_sched[epoch])
        perm = torch.randperm(N)
        tot_tf = tot_roll = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            x, dxd, d2 = Xt[idx], Dt[idx], D2[idx]
            alpha, beta, delta, z = model._encode(x, dxd, dt)
            yhat = model._rhs(alpha, beta, delta, z)          # teacher-forced d2x/ds2
            L_tf = ((yhat - d2) ** 2).mean() / var_d2

            a = alpha[:, :, 0]; b = beta[:, :, 0]; d = delta[:, :, 0]; g = model.gamma
            x0 = x[:, 0, 0].detach()
            xp0 = ((model.tau / dt) * dxd[:, 0, 0]).detach()
            xg = rollout(a, b, d, g, x0, xp0, H)              # (B, H), cold start
            L_roll = ((xg - x[:, :H, 0]) ** 2).mean() / var_x

            loss = L_tf + args.lam_roll * L_roll
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_tf += float(L_tf); tot_roll += float(L_roll); nb += 1
        print(f"[epoch {epoch+1}/{args.epochs} H={H}] relMSE tf={tot_tf/nb:.4f} roll={tot_roll/nb:.4f}",
              flush=True)

    save_model(model, opt, os.path.join(run_dir, f"checkpoint_{ep0+args.epochs}.tar"),
               n_layers=model.alpha_mamba.config.n_layers, d_state=model.alpha_mamba.config.d_state,
               d_conv=model.alpha_mamba.config.d_conv, expand_factor=model.alpha_mamba.config.expand_factor)
    print(f"saved fine-tuned model; gamma={float(model.gamma):.4f}")


if __name__ == "__main__":
    main()
