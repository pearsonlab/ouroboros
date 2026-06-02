"""Step-by-step RK4 rollout of integrate_poly_autonomous with per-step diagnostics.

Lets us see *where* and *how* the autonomous integration blows up: is it step 1
(initial-condition overflow), is it slow exponential growth, does it survive
some steps then suddenly hit nan?

Prints the state magnitudes (x, x', kernel-output, dx') at a chosen stride.
"""

import argparse
import glob
import os
import shutil
import tempfile

import numpy as np
import torch

from data.load_data import get_segmented_audio  # noqa
from train.train import load_model
from train.eval import deriv_approx_dy, correct
from examples.run_lambda_pipeline import load_voc_windows


def load_specific(ckpt_path):
    tmp = tempfile.mkdtemp(prefix="ckpt_eval_")
    link = os.path.join(tmp, os.path.basename(ckpt_path))
    os.symlink(os.path.abspath(ckpt_path), link)
    try:
        model, _, _, _ = load_model(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return model


def instrument_one(model, audio, dt, stride=50, max_steps=None):
    """Mirror integrate_poly_autonomous's noise_sd>0 path (per-sample RK4) with logging."""
    L = len(audio)
    audio_3d = audio[None, :, None]
    dy = deriv_approx_dy(audio_3d)
    audio_t = torch.from_numpy(audio_3d).to(torch.float32).to("cuda")
    dy_t = torch.from_numpy(dy).to(torch.float32).to("cuda")

    with torch.no_grad():
        omega, gamma, _, weights, _ = model.get_funcs(audio_t, dy_t, dt)
    omega = omega.detach().cpu().numpy().squeeze()
    gamma = gamma.detach().cpu().numpy().squeeze()
    weights = weights.detach().cpu().numpy()
    _, _, P, P2 = weights.shape
    ww = weights.reshape(L, 1, 1, P, P2)

    kernel = model.kernel

    x0 = float(audio[0])
    xp0 = (model.tau / dt) * float(dy[0, 0, 0])
    print(f"# IC: x={x0:.4e}  xp={xp0:.4e}  tau={model.tau:.4e}  dt={dt:.4e}  L={L}")
    print(f"# drives at t=0: omega={omega[0]:.4e}  gamma={gamma[0]:.4e}  "
          f"|w|max={np.abs(weights[0]).max():.4e}")
    print(f"{'step':>6}  {'x':>14}  {'xp':>14}  {'omega':>10}  {'gamma':>10}  "
          f"{'kern':>14}  {'dxp':>14}")
    x, xp = x0, xp0
    end = L - 1 if max_steps is None else min(max_steps, L - 1)
    for k in range(end):
        om, ga, wk = omega[k], gamma[k], ww[k]

        def f(xx, vv):
            kern = float(kernel.forward_given_weights_numpy(np.array([[[xx, vv]]]), wk).squeeze())
            return vv, -(om ** 2) * xx - ga * vv - kern, kern

        k1x, k1v, kern1 = f(x, xp)
        k2x, k2v, _ = f(x + 0.5 * k1x, xp + 0.5 * k1v)
        k3x, k3v, _ = f(x + 0.5 * k2x, xp + 0.5 * k2v)
        k4x, k4v, _ = f(x + k3x, xp + k3v)
        x_new = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        xp_new = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6

        if (k % stride == 0) or (not np.isfinite(x_new)) or (not np.isfinite(xp_new)) or k < 5:
            print(f"{k:>6d}  {x:>14.4e}  {xp:>14.4e}  {om:>10.3e}  {ga:>10.3e}  "
                  f"{kern1:>14.4e}  {k1v:>14.4e}")
        if (not np.isfinite(x_new)) or (not np.isfinite(xp_new)):
            print(f"# NON-FINITE at step {k+1}: x_new={x_new}  xp_new={xp_new}")
            return k + 1
        x, xp = x_new, xp_new
    print(f"# completed {end} steps, final x={x:.4e}, xp={xp:.4e}")
    return end


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="path to checkpoint_N.tar")
    p.add_argument("--data-glob", required=True)
    p.add_argument("--n-val-vocs", type=int, default=3)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    model = load_specific(args.ckpt)
    model.eval()
    print(f"# loaded model from {args.ckpt}")

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    val_dir = dirs[-2]
    val_vocs, sr = load_voc_windows(val_dir, args.n_val_vocs, args.start_offset_ms, args.auto_n)
    dt = 1.0 / sr
    print(f"# val_dir={val_dir} sr={sr} n_vocs={len(val_vocs)} L={len(val_vocs[0])}")

    for i, voc in enumerate(val_vocs):
        print(f"\n=== voc {i} ===")
        voc = np.asarray(voc, dtype=np.float64)
        print(f"# target std={np.std(correct(voc)):.4e}  range=[{voc.min():.3e}, {voc.max():.3e}]")
        instrument_one(model, voc, dt, stride=args.stride, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
