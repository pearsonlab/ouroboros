"""
Make the polynomial Ouroboros's cycle ATTRACTING (fix the marginal-amplitude problem), two ways.

The autonomous amplitude is poorly constrained because on-orbit acceleration-matching leaves the
transverse Floquet exponent Lambda = integral of d f/d x' essentially free (~0, marginal). Two
fine-tunes that target the transverse/off-orbit dynamics:

  --method floq : add an amplitude-STABILITY penalty (no rollout). Evaluate the divergence field
                  d(x,x') = d f/d x' = -gamma - sum_{p,k>=1} w_pk x^p k x'^{k-1} at the data orbit
                  SCALED to (1+delta)*radius and (1-delta)*radius (same drives), and push the outer
                  one negative (contract from outside) and the inner one positive (expand from
                  inside). This makes the data radius a stable limit cycle WITHOUT imposing global
                  decay (a constant-negative Lambda would just damp to zero). Loss = TF + lam_floq*pen.

  --method noise: short-horizon NOISY teacher forcing. Perturb the start state off the orbit
                  (isotropic Gaussian, incl. transverse), roll out H << ... ~1-2 periods (short
                  enough that phase drift is negligible so pointwise MSE is valid), and require the
                  trajectory to track the TRUE orbit. The carried-forward perturbation makes the
                  model learn to contract back to the data orbit (Lambda<0 at the right radius).
                  Loss = pointwise rollout MSE (from noisy start) + lam_tf*TF.

    python -m examples.finetune_attractor_poly --init poly_ro_pipeline/seed0/tf --method floq \
        --out-dir ./poly_attr_floq_seed0
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.optim import Adam

from utils import deriv_approx_dy, deriv_approx_d2y
from train.train import load_model, save_model
from train.rollout_refine import gather_windows

BX, BXP = 0.5, 1.0


def divergence(x, v, gamma, weights, powers):
    """d f/d x' along the trajectory. x,v,gamma:(B,L); weights:(B,L,P,P); -> (B,L)."""
    xp = x.unsqueeze(-1) ** powers                  # (B,L,P) x^p
    vp = v.unsqueeze(-1) ** powers                  # (B,L,P) v^k
    dvk = torch.zeros_like(vp)
    dvk[..., 1:] = powers[1:] * vp[..., :-1]        # k v^{k-1}
    dkern = torch.einsum("blpk,blp,blk->bl", weights, xp, dvk)
    return -gamma - dkern


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", required=True)
    p.add_argument("--method", choices=["floq", "noise"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-glob", default="data500/gabo_p[0-7]")
    p.add_argument("--n-windows", type=int, default=24)
    p.add_argument("--l-seg", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam-tf", type=float, default=1.0)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    # floq
    p.add_argument("--lam-floq", type=float, default=5.0)
    p.add_argument("--delta", type=float, default=0.3, help="radius perturbation for stability penalty")
    # noise
    p.add_argument("--hmin", type=int, default=12)
    p.add_argument("--hmax", type=int, default=30)
    p.add_argument("--noise-frac", type=float, default=0.3, help="start-state noise as fraction of signal std")
    args = p.parse_args()

    run_dir = os.path.join(os.path.abspath(args.out_dir), "poly")
    os.makedirs(run_dir, exist_ok=True)
    model, _, _, ep0 = load_model(args.init)
    model.train()
    dt = model.tau

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    X, _ = gather_windows(dirs, args.n_windows, args.l_seg, args.start_offset_ms)
    D1 = deriv_approx_dy(X)
    D2 = deriv_approx_d2y(X)
    Xt = torch.tensor(X, dtype=torch.float32, device="cuda")
    Dt = torch.tensor(D1, dtype=torch.float32, device="cuda")
    D2t = torch.tensor(D2, dtype=torch.float32, device="cuda")
    var_d2 = float(D2t.var())
    P = model.kernel.poly_dim + 1
    powers = torch.arange(P, device="cuda", dtype=torch.float32)
    opt = Adam(model.parameters(), lr=args.lr)
    N = Xt.shape[0]
    H_sched = np.unique(np.round(np.geomspace(args.hmin, args.hmax, args.epochs)).astype(int))
    print(f"attractor FT [{args.method}] {args.init} (ep {ep0}); {N} windows L={args.l_seg}", flush=True)

    for epoch in range(args.epochs):
        H = int(H_sched[min(epoch, len(H_sched) - 1)])
        perm = torch.randperm(N)
        tot = {"tf": 0.0, "main": 0.0}
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, dxd, d2b = Xt[idx], Dt[idx], D2t[idx]
            z2 = (model.tau / dt) * dxd
            omega, gamma, wk, weights, _ = model.get_funcs(x, dxd.clone(), dt)
            tf = -(omega ** 2) * x - gamma * z2 - wk
            L_tf = ((tf - d2b) ** 2).mean() / var_d2

            om, ga, w = omega[:, :, 0], gamma[:, :, 0], weights
            xx, vv = x[:, :, 0], z2[:, :, 0]

            if args.method == "floq":
                # divergence at the orbit scaled to (1+/-delta)*radius (same drives)
                d_out = divergence((1 + args.delta) * xx, (1 + args.delta) * vv, ga, w, powers)
                d_in = divergence((1 - args.delta) * xx, (1 - args.delta) * vv, ga, w, powers)
                # integrate to the Floquet exponent (per-sample divergence is ~1e-3; the summed
                # Lambda is O(1) and is what actually predicts amplitude change)
                Lam_out = d_out.sum(dim=1)   # (B,) Floquet exponent at outer radius
                Lam_in = d_in.sum(dim=1)     # (B,) at inner radius
                # want Lam_out <= 0 (contract from outside), Lam_in >= 0 (expand from inside)
                L_main = (torch.relu(Lam_out) ** 2).mean() + (torch.relu(-Lam_in) ** 2).mean()
                loss = L_main * args.lam_floq + args.lam_tf * L_tf
            else:  # noise
                B = xx.shape[0]
                sx = args.noise_frac * xx.std()
                sv = args.noise_frac * vv.std()
                xc = (xx[:, 0] + sx * torch.randn(B, device="cuda")).detach()
                xp = (vv[:, 0] + sv * torch.randn(B, device="cuda")).detach()

                def f(xa, va, k):
                    xpw = xa.unsqueeze(1) ** powers
                    xvw = va.unsqueeze(1) ** powers
                    kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w[:, k])
                    return va, -(om[:, k] ** 2) * xa - ga[:, k] * va - kern

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
                xg = torch.stack(xs, dim=1)            # (B,H) noisy rollout
                tgt = xx[:, :H]                        # (B,H) true orbit
                L_main = ((xg - tgt) ** 2).mean() / (xx.var() + 1e-12)
                loss = L_main + args.lam_tf * L_tf

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot["tf"] += float(L_tf); tot["main"] += float(L_main)
            nb += 1
        print(f"[ep {epoch + 1}/{args.epochs} H={H}] {args.method}={tot['main']/nb:.4f} tf={tot['tf']/nb:.4f}", flush=True)

    cfg = model.omega_mamba.config
    save_model(model, opt, os.path.join(run_dir, f"checkpoint_{ep0 + args.epochs}.tar"),
               n_layers=cfg.n_layers, d_state=cfg.d_state, d_conv=cfg.d_conv,
               expand_factor=cfg.expand_factor)
    print(f"saved [{args.method}] model to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
