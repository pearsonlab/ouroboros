"""
Does EARLY-checkpoint (rescaled) autonomy predict the FINAL seed ranking? If so, the seed search can
train many seeds briefly, cull to the top, and only finish the winner.

For each seed dir poly_lam<lam>_seed<s>/ with intermediate checkpoints, scores rescaled validation
autonomy (train.eval.autonomy_score(rescale=True) -- the deployed selection metric) at each checkpoint
epoch, then reports, for each early epoch k vs the final epoch:
  - Spearman rank correlation across seeds (does early ranking match final ranking?)
  - top-1 hit (does the early-best seed == final-best seed?) and top-2 overlap.

    python -m examples.seed_cull_test --run-dir poly_seedcull --lam 1.068 --n-seeds 8 \
        --epochs 10 20 30 40 50 --val-dir data500/gabo_p8
"""

import argparse
import glob
import os
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.stats import spearmanr

from train.train import load_model
from train.eval import autonomy_score


def load_voc_windows(data_dir, n, off_ms, L):
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
        s = int(on * sr) + int(off_ms / 1e3 * sr)
        seg = af[s:s + L]
        if len(seg) == L:
            segs.append(seg)
    return segs, sr


def load_ckpt(seed_dir, epoch):
    """load a SPECIFIC checkpoint (load_model takes the highest in a dir, so isolate via a temp dir)."""
    src = os.path.abspath(os.path.join(seed_dir, f"checkpoint_{epoch}.tar"))
    td = tempfile.mkdtemp()
    os.symlink(src, os.path.join(td, f"checkpoint_{epoch}.tar"))
    model, _, _, _ = load_model(td)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="poly_seedcull")
    p.add_argument("--lam", type=float, default=1.068)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--epochs", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--val-dir", default="data500/gabo_p8")
    p.add_argument("--n-vocs", type=int, default=5)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--auto-n", type=int, default=3000)
    args = p.parse_args()

    segs, sr = load_voc_windows(args.val_dir, args.n_vocs, args.start_offset_ms, args.auto_n)
    print(f"{len(segs)} val vocs from {args.val_dir}; seeds 0..{args.n_seeds-1} at lam={args.lam:.3f}\n", flush=True)

    table = {}  # epoch -> np.array of rescaled autonomy per seed
    for ep in args.epochs:
        row = []
        for s in range(args.n_seeds):
            d = os.path.join(args.run_dir, f"poly_lam{args.lam:.3f}_seed{s}")
            ck = os.path.join(d, f"checkpoint_{ep}.tar")
            if not os.path.isfile(ck):
                row.append(np.nan); continue
            model = load_ckpt(d, ep); model.eval()
            sc, _, _ = autonomy_score(model, segs, model.tau, rescale=True)
            row.append(sc)
        table[ep] = np.array(row)
        print(f"epoch {ep:>3d}: " + "  ".join(f"s{i}={v:+.3f}" for i, v in enumerate(row)), flush=True)

    final_ep = args.epochs[-1]
    final = table[final_ep]
    order_final = np.argsort(-final)
    print(f"\nfinal (epoch {final_ep}) seed ranking (best->worst): {list(order_final)}", flush=True)
    print(f"\n{'early ep':>8s} {'Spearman rho':>13s} {'top-1 hit':>10s} {'top-2 overlap':>14s}", flush=True)
    for ep in args.epochs[:-1]:
        early = table[ep]
        ok = np.isfinite(early) & np.isfinite(final)
        rho, _ = spearmanr(early[ok], final[ok])
        order_early = np.argsort(-np.where(np.isfinite(early), early, -np.inf))
        top1 = order_early[0] == order_final[0]
        top2 = len(set(order_early[:2]) & set(order_final[:2]))
        print(f"{ep:>8d} {rho:>13.3f} {str(bool(top1)):>10s} {f'{top2}/2':>14s}", flush=True)


if __name__ == "__main__":
    main()
