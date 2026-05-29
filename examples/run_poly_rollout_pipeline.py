"""
Multi-seed validation of the integrated TF + spectral/envelope rollout-refine pipeline.

For each seed: trains the integrated model (examples.train_poly_rollout -> teacher-forced THEN
rollout-refine, saving tf/ and poly/ checkpoints), then scores DETERMINISTIC autonomous
reconstruction on held-out gabo_p9 windows for BOTH the TF-only and the TF+rollout model, against
the rescale baseline. Reports the per-seed and aggregate change, and a paired figure.

Resumable: a seed whose final (poly/) checkpoint exists is not retrained.

Run from the repo root:
    python -m examples.run_poly_rollout_pipeline --train-glob 'data500/gabo_p[0-7]' \
        --test-dir data500/gabo_p9 --seeds 0 1 2 3 4 --out-dir ./poly_ro_pipeline
"""

import argparse
import glob
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.train import load_model
from examples.eval_autonomy_rescale import load_voc_windows, score_model

plt.rcParams["text.usetex"] = False
PY = sys.executable


def latest_ckpt(d):
    return sorted(glob.glob(os.path.join(d, "*.tar")),
                  key=lambda f: int(f.split("checkpoint_")[-1].split(".tar")[0]))


def train_one(seed, args):
    out_dir = os.path.join(os.path.abspath(args.out_dir), f"seed{seed}")
    if latest_ckpt(os.path.join(out_dir, "poly")):
        print(f"[skip] seed{seed}: final checkpoint exists", flush=True)
        return out_dir
    cmd = [PY, "-m", "examples.train_poly_rollout", "--data-glob", args.train_glob,
           "--out-dir", out_dir, "--tf-epochs", str(args.tf_epochs),
           "--rollout-epochs", str(args.rollout_epochs), "--lam-env", str(args.lam_env),
           "--lam-spec", str(args.lam_spec), "--lam-tf", str(args.lam_tf),
           "--drive-lowpass-ms", str(args.drive_lowpass_ms), "--lam", str(args.lam),
           "--d-state", str(args.d_state), "--context-len", str(args.context_len),
           "--batch-size", str(args.batch_size), "--rollout-windows", str(args.rollout_windows),
           "--rollout-hmax", str(args.rollout_hmax), "--rollout-l-seg", str(args.rollout_hmax),
           "--seed", str(seed)]
    print(f"\n[train] seed{seed}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout[-1500:])
    if res.returncode != 0:
        sys.stdout.write(res.stderr[-2000:])
        raise RuntimeError(f"training failed: seed{seed}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", default="data500/gabo_p[0-7]")
    p.add_argument("--test-dir", default="data500/gabo_p9")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out-dir", default="./poly_ro_pipeline")
    p.add_argument("--tf-epochs", type=int, default=30)
    p.add_argument("--rollout-epochs", type=int, default=8)
    p.add_argument("--rollout-windows", type=int, default=12)
    p.add_argument("--rollout-hmax", type=int, default=1500)
    p.add_argument("--lam", type=float, default=1.068)
    p.add_argument("--lam-spec", type=float, default=1.0)
    p.add_argument("--lam-env", type=float, default=10.0)
    p.add_argument("--lam-tf", type=float, default=1.0)
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--context-len", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-test-vocs", type=int, default=6)
    p.add_argument("--auto-n", type=int, default=3000)
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    args = p.parse_args()

    run_dirs = {s: train_one(s, args) for s in args.seeds}

    segs, sr = load_voc_windows(args.test_dir, args.n_test_vocs, args.start_offset_ms, args.auto_n)
    print(f"\nscoring autonomy on {len(segs)} held-out vocs from {args.test_dir}\n", flush=True)
    print(f"{'seed':6s} {'phase':6s} {'raw':>7s} {'rescale':>8s} {'spec':>6s} {'amp_pen':>8s} "
          f"{'envcorr':>8s} {'envL1':>7s}", flush=True)
    res = {"tf": [], "ro": []}
    rows = []
    for s in args.seeds:
        for phase, sub in [("tf", "tf"), ("ro", "poly")]:
            ck = latest_ckpt(os.path.join(run_dirs[s], sub))
            if not ck:
                print(f"  seed{s} {phase}: no checkpoint", flush=True)
                continue
            model, _, _, _ = load_model(os.path.join(run_dirs[s], sub))
            model.eval()
            r = score_model(model, segs, sr)
            res[phase].append(r["raw"])
            rows.append((s, phase, r))
            print(f"seed{s:<2d} {phase:6s} {r['raw']:+7.3f} {r['rescale']:+8.3f} {r['spec']:6.3f} "
                  f"{r['amp_pen']:8.3f} {r['envcorr']:8.3f} {r['envL1']:7.3f}", flush=True)

    out = os.path.abspath(args.out_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rollout_pipeline.csv"), "w") as f:
        f.write("seed,phase,raw,rescale,spec,amp_pen,pitch_pen,envcorr,envL1\n")
        for s, ph, r in rows:
            f.write(f"{s},{ph},{r['raw']:.4f},{r['rescale']:.4f},{r['spec']:.4f},{r['amp_pen']:.4f},"
                    f"{r['pitch_pen']:.4f},{r['envcorr']:.4f},{r['envL1']:.4f}\n")

    tf, ro = np.array(res["tf"]), np.array(res["ro"])
    if len(tf) and len(ro) and len(tf) == len(ro):
        print(f"\n=== SUMMARY over {len(tf)} seeds (autonomy raw) ===", flush=True)
        print(f"  TF-only  : mean {tf.mean():+.3f}  std {tf.std():.3f}  [{tf.min():+.3f},{tf.max():+.3f}]", flush=True)
        print(f"  TF+rollout: mean {ro.mean():+.3f}  std {ro.std():.3f}  [{ro.min():+.3f},{ro.max():+.3f}]", flush=True)
        print(f"  paired diff (ro-tf): mean {(ro-tf).mean():+.3f}  per-seed {np.round(ro-tf,3)}", flush=True)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
        for i in range(len(tf)):
            axL.plot([0, 1], [tf[i], ro[i]], "-o", color="0.6", zorder=1)
        axL.scatter(np.zeros_like(tf), tf, color="tab:red", zorder=2, label="TF-only")
        axL.scatter(np.ones_like(ro), ro, color="tab:blue", zorder=2, label="TF+rollout")
        axL.set_xticks([0, 1]); axL.set_xticklabels(["TF-only", "TF+rollout"])
        axL.set_xlim(-0.4, 1.4); axL.set_ylabel("autonomy raw (held-out gabo_p9)")
        axL.set_title("paired by seed"); axL.legend(fontsize=8)
        axR.boxplot([tf, ro], tick_labels=["TF-only", "TF+rollout"], showmeans=True, widths=0.5)
        axR.scatter(np.ones_like(tf), tf, color="tab:red", alpha=0.7)
        axR.scatter(np.full_like(ro, 2), ro, color="tab:blue", alpha=0.7)
        axR.set_ylabel("autonomy raw")
        axR.set_title(f"distribution ({len(tf)} seeds, lam_env={args.lam_env})")
        fig.suptitle("Integrated TF + spectral/envelope rollout refinement: autonomous reconstruction")
        fig.tight_layout()
        figpath = os.path.join(out, "rollout_pipeline.png")
        fig.savefig(figpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved figure -> {figpath}", flush=True)
    print(f"Saved CSV -> {os.path.join(out, 'rollout_pipeline.csv')}", flush=True)


if __name__ == "__main__":
    main()
