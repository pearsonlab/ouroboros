"""
Multi-seed A/B: does the constant (0,0) "alpha" forcing term improve AUTONOMOUS reconstruction,
disentangled from the low-pass change and from seed noise?

Trains TWO arms over the SAME seed set, both at 1ms drive low-pass (matched control):
  - alpha_on : --keep-const  (the (0,0) y^0*ydot^0 forcing term, zero-init, low-passed with the rest)
  - alpha_off: no constant term (the standard poly model), also at 1ms

Each run is just the tested examples.train_poly_lowpass invoked as a subprocess (same code path that
produced the single-seed +0.372), so this only orchestrates + scores. Resumable: a run whose final
checkpoint already exists is skipped. After training, scores deterministic autonomy on held-out
gabo_p9 windows (train.eval.autonomy_score) and writes a paired comparison figure + CSV.

Run from the repo root:
    python -m examples.run_alpha_multiseed --train-glob 'data500/gabo_p[0-7]' --test-dir data500/gabo_p9 \
        --seeds 0 1 2 3 4 --epochs 40 --lam 1.068 --drive-lowpass-ms 1.0 --out-dir ./poly_alpha_ms
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.io import wavfile

from train.train import load_model
from train.eval import autonomy_score

plt.rcParams["text.usetex"] = False
PY = sys.executable  # the venv python running this driver


def load_voc_windows(data_dir, n_vocs, off_ms, n):
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        on = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))[0][0]
        s = int(on * sr) + int(off_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs


def train_one(arm, seed, args):
    """Train one (arm, seed) via examples.train_poly_lowpass unless its final checkpoint exists."""
    out_dir = os.path.join(os.path.abspath(args.out_dir), f"{arm}_seed{seed}")
    ckpt = os.path.join(out_dir, "poly", f"checkpoint_{args.epochs}.tar")
    if os.path.isfile(ckpt):
        print(f"[skip] {arm} seed{seed}: {ckpt} exists", flush=True)
        return out_dir
    cmd = [PY, "-m", "examples.train_poly_lowpass",
           "--data-glob", args.train_glob, "--out-dir", out_dir,
           "--lam", str(args.lam), "--drive-lowpass-ms", str(args.drive_lowpass_ms),
           "--d-state", str(args.d_state), "--context-len", str(args.context_len),
           "--epochs", str(args.epochs), "--seg", "10",
           "--batch-size", str(args.batch_size), "--seed", str(seed)]
    if arm == "alpha_on":
        cmd.append("--keep-const")
    print(f"\n[train] {arm} seed{seed}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout[-2000:])
    if res.returncode != 0:
        sys.stdout.write(res.stderr[-2000:])
        raise RuntimeError(f"training failed: {arm} seed{seed}")
    m = re.search(r"best test R2 = ([\-0-9.]+)", res.stdout)
    print(f"[done] {arm} seed{seed}: best test R2 = {m.group(1) if m else '?'}", flush=True)
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", default="data500/gabo_p[0-7]")
    p.add_argument("--test-dir", default="data500/gabo_p9")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lam", type=float, default=1.068)
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--context-len", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-test-vocs", type=int, default=6)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--out-dir", default="./poly_alpha_ms")
    args = p.parse_args()

    arms = ["alpha_on", "alpha_off"]
    # train all (arm, seed) sequentially (shared GPU -> no parallel)
    run_dirs = {}
    for seed in args.seeds:
        for arm in arms:
            run_dirs[(arm, seed)] = train_one(arm, seed, args)

    # score autonomy on the SAME held-out windows
    segs = load_voc_windows(args.test_dir, args.n_test_vocs, args.start_offset_ms, args.auto_n)
    print(f"\nscoring autonomy on {len(segs)} held-out vocs from {args.test_dir}", flush=True)
    results = {arm: [] for arm in arms}
    rows = []
    for seed in args.seeds:
        for arm in arms:
            model, _, _, _ = load_model(os.path.join(run_dirs[(arm, seed)], "poly"))
            model.eval()
            score, per, bd = autonomy_score(model, segs, model.tau)
            results[arm].append(score)
            rows.append((arm, seed, score, bd["spec_corr"], bd["amp_pen"],
                         bd["pitch_pen"], bd["bounded_frac"]))
            print(f"  {arm:9s} seed{seed}: autonomy={score:+.3f}  spec={bd['spec_corr']:.3f}  "
                  f"amp_pen={bd['amp_pen']:.3f}  pitch_pen={bd['pitch_pen']:.3f}", flush=True)

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    csv = os.path.join(out, "alpha_multiseed.csv")
    with open(csv, "w") as f:
        f.write("arm,seed,autonomy,spec_corr,amp_pen,pitch_pen,bounded_frac\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.2f}\n")

    on, off = np.array(results["alpha_on"]), np.array(results["alpha_off"])
    print(f"\n=== SUMMARY over {len(args.seeds)} seeds ===", flush=True)
    print(f"  alpha_on : mean {on.mean():+.3f}  std {on.std():.3f}  [{on.min():+.3f},{on.max():+.3f}]", flush=True)
    print(f"  alpha_off: mean {off.mean():+.3f}  std {off.std():.3f}  [{off.min():+.3f},{off.max():+.3f}]", flush=True)
    print(f"  paired diff (on-off): mean {(on-off).mean():+.3f}  per-seed {np.round(on-off,3)}", flush=True)

    # figure: paired lines + per-arm distribution
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    for i, s in enumerate(args.seeds):
        axL.plot([0, 1], [off[i], on[i]], "-o", color="0.6", zorder=1)
    axL.scatter(np.zeros_like(off), off, color="tab:red", zorder=2, label="alpha_off (1ms)")
    axL.scatter(np.ones_like(on), on, color="tab:blue", zorder=2, label="alpha_on (1ms)")
    axL.set_xticks([0, 1]); axL.set_xticklabels(["alpha_off", "alpha_on"])
    axL.set_xlim(-0.4, 1.4); axL.set_ylabel("autonomy score (held-out gabo_p9)")
    axL.set_title("paired by seed"); axL.legend(fontsize=8)
    bp = axR.boxplot([off, on], labels=["alpha_off", "alpha_on"], showmeans=True, widths=0.5)
    axR.scatter(np.full_like(off, 1), off, color="tab:red", alpha=0.7)
    axR.scatter(np.full_like(on, 2), on, color="tab:blue", alpha=0.7)
    axR.set_ylabel("autonomy score")
    axR.set_title(f"distribution ({len(args.seeds)} seeds, {args.epochs}ep, 1ms low-pass)")
    fig.suptitle("Constant (0,0) 'alpha' forcing: autonomous reconstruction, matched 1ms control")
    fig.tight_layout()
    figpath = os.path.join(out, "alpha_multiseed.png")
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved CSV  -> {csv}", flush=True)
    print(f"Saved figure -> {figpath}", flush=True)


if __name__ == "__main__":
    main()
