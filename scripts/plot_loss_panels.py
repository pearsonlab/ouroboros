"""Multi-panel loss + autonomy plot for one or two spectral-rollout runs.

Reads the TB events file under each --run-dir/seed0 and aggregates Loss/spec,
Loss/tf, Loss/env, Loss/total per epoch (mean over batches). Optionally reads
the monitor state.json next to each run for the per-checkpoint cold-start
autonomy + breakdown and overlays them.

usage:
    python scripts/plot_loss_panels.py \
        --run live:/home/pearson/code/ouroboros-spectral/poly_spectral_day85 \
              :/home/pearson/.claude/jobs/712703d8/tmp/spectral_monitor/state.json \
        --run env:/home/pearson/code/ouroboros-spectral/poly_spectral_day85_env \
              :/home/pearson/.claude/jobs/712703d8/tmp/spectral_monitor_env/state.json \
        --out /tmp/spectral_loss_panels.png

(each --run is "label:run_dir[:state_path]"; state_path is optional)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_run_arg(s):
    parts = s.split(":")
    if len(parts) == 2:
        label, run_dir = parts
        state_path = None
    elif len(parts) == 3:
        label, run_dir, state_path = parts
    else:
        raise argparse.ArgumentTypeError(
            f"--run expects 'label:run_dir' or 'label:run_dir:state_path', got {s!r}"
        )
    return {"label": label, "run_dir": run_dir, "state_path": state_path}


def read_run(run_dir):
    """Return per-batch scalars dict + an inferred batches-per-epoch."""
    seed_dir = os.path.join(run_dir, "seed0")
    evs = sorted(glob.glob(os.path.join(seed_dir, "events.out.tfevents.*")))
    if not evs:
        return None, None
    ea = EventAccumulator(evs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in ("Loss/spec", "Loss/tf", "Loss/env", "Loss/total", "Train/H",
              "Train/lam_spec_t", "Loss/nan_skip"):
        if t in tags:
            sc = ea.Scalars(t)
            out[t] = np.array([s.value for s in sc])
    # Infer batches-per-epoch from the train pipeline config: 6000 segs * 0.8 (train split) / 8 batch = 600.
    # If the run's tags reveal more we could overwrite; for now hard-code as the entry script does.
    return out, 600


def per_epoch_means(values, batches_per_epoch):
    """Group batch-indexed scalars into per-epoch means."""
    n = len(values)
    n_eps = n // batches_per_epoch
    if n_eps == 0:
        return np.array([])
    truncated = values[:n_eps * batches_per_epoch]
    return truncated.reshape(n_eps, batches_per_epoch).mean(axis=1)


def read_ckpt_autonomy(state_path):
    """Pull per-ckpt autonomy + breakdown from monitor state.json's history (if present)."""
    if not state_path or not os.path.exists(state_path):
        return {}
    with open(state_path) as f:
        state = json.load(f)
    # state["history"] is only loss snapshots, not ckpt scores. The monitor doesn't currently
    # persist per-ckpt autonomy; surface what we have via state's last_* fields if no ckpt log.
    # For now we attempt to re-derive a per-ckpt timeline by checking what the diagnose script
    # has recorded in last_val_autonomy / best_val_autonomy snapshots over time -- the file
    # is overwritten each poll so we only have the most recent. Return empty if no history.
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", type=parse_run_arg, required=True,
                   help="label:run_dir[:state_path]; repeat for multiple runs to overlay")
    p.add_argument("--out", default="/tmp/spectral_loss_panels.png")
    p.add_argument("--smooth", default="auto",
                   help="rolling-window smoothing in epochs. 'auto' (default) picks "
                        "max(1, min(5, shortest_run_epochs // 3)) so a short overlay "
                        "doesn't disappear under valid-mode convolve. Integer overrides.")
    args = p.parse_args()

    runs = []
    for r in args.run:
        scalars, bpe = read_run(r["run_dir"])
        if scalars is None:
            print(f"warn: no TB events in {r['run_dir']}/seed0", file=sys.stderr)
            continue
        per_epoch = {t: per_epoch_means(v, bpe) for t, v in scalars.items()}
        runs.append({"label": r["label"], "run_dir": r["run_dir"],
                     "scalars": scalars, "per_epoch": per_epoch})

    if not runs:
        raise SystemExit("no runs with data")

    # Auto-pick the smoothing window so the shortest run keeps multiple points.
    if isinstance(args.smooth, str) and args.smooth.lower() == "auto":
        per_run_n_eps = []
        for r in runs:
            ns = [len(v) for v in r["per_epoch"].values() if len(v) > 0]
            if ns:
                per_run_n_eps.append(min(ns))
        smooth = max(1, min(5, (min(per_run_n_eps) // 3) if per_run_n_eps else 1))
    else:
        smooth = max(1, int(args.smooth))
    print(f"smoothing window = {smooth} epoch(s)")

    panels = [
        ("Loss/spec", "spec (MRSTFT)", True),
        ("Loss/tf",   "tf anchor",     False),
        ("Loss/env",  "env L1",        False),
        ("Loss/total", "total",        True),
        ("Train/H",   "rollout H (samples)", False),
        ("Train/lam_spec_t", "lam_spec_t (warmup)", False),
    ]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.0 * n), sharex=True)

    # Stable, explicit color per known run label so the three concurrent runs are
    # distinguishable. Unknown labels fall back to matplotlib's default cycle, which
    # starts at tab:blue and would collide with `live`.
    colors = {
        "live": "tab:blue",
        "env": "tab:orange",
        "env1e4": "tab:green",
        "env1e5": "tab:red",
        "osc1e5": "tab:purple",
    }

    # Detect the spec_warmup boundary per run -- the first epoch where lam_spec_t hits
    # its max (i.e., the warmup ramp ends). Drawn as a faint vertical line per run so
    # readers know that pre-boundary total-loss values are weighted-mix artifacts of
    # the ramp, not signal.
    warmup_boundaries = []
    for r in runs:
        lam = r["per_epoch"].get("Train/lam_spec_t")
        if lam is None or len(lam) == 0:
            continue
        lam_max = float(np.max(lam))
        if lam_max <= 0:
            continue
        # epoch index where the ramp first reaches (within 1%) its max
        idx = int(np.argmax(lam >= lam_max * 0.99))
        warmup_boundaries.append((r["label"], idx))

    for ax, (tag, title, logy) in zip(axes, panels):
        for r in runs:
            v = r["per_epoch"].get(tag)
            if v is None or len(v) == 0:
                continue
            if smooth > 1 and len(v) >= smooth:
                # simple moving average
                vs = np.convolve(v, np.ones(smooth) / smooth, mode="valid")
                xs = np.arange(len(vs)) + (smooth - 1)
            else:
                vs = v
                xs = np.arange(len(vs))
            c = colors.get(r["label"], None)
            ax.plot(xs, vs, label=r["label"], color=c, lw=1.4)
        # warmup boundary marker per run
        for label, b in warmup_boundaries:
            c = colors.get(label, None)
            ax.axvline(b, color=c, alpha=0.25, lw=1.0, ls="--", zorder=0)
        ax.set_ylabel(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("epoch")

    fig.suptitle("spectral-rollout training: per-epoch losses", fontsize=12, y=1.005)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
