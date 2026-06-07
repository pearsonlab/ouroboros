"""One-shot diagnostic for the live spectral-rollout training run.

Reads the TB events file, checks for a new checkpoint, runs cold-start
autonomy_score on the latest checkpoint if it hasn't been scored yet, and emits
a single line to stdout if there is something actionable to report.

Signals (priority order):
- DEAD: training process gone.
- NAN_SPIKE: nan_skip counter jumped by > 50 since last poll.
- LOSS_SPIKE: spec_recent5 / spec_prev5 > 2.0.
- NEW_CKPT: a fresh checkpoint was found; ran autonomy_score on it.
- PLATEAU: val_autonomy hasn't improved by > 0.02 across the last 2 checkpoints.
           Falls back to "best_spec hasn't improved in 8 polls" before any
           checkpoint exists.
- HEARTBEAT: nothing actionable, but every HEARTBEAT_EVERY polls.
- UNCHANGED: not printed; monitor silent.

The autonomy check is done in a SUBPROCESS so the main monitor process never holds
a CUDA context (avoids contention with the live training run -- subprocess will
queue waiting for a GPU slot for ~30-60s during a forward of the autonomous
integrator, finish, and release).
"""

import glob
import json
import argparse
import os
import subprocess
import sys

import numpy as np


_DEFAULT_STATE = "/home/pearson/.claude/jobs/712703d8/tmp/spectral_monitor/state.json"
_DEFAULT_SEED = "/home/pearson/code/ouroboros-spectral/poly_spectral_day85/seed0"
_DEFAULT_TRAIN_PATTERN = "train_poly_spectral_blk445"

# These three module-level names are populated at __main__ time from CLI args so the
# rest of the script can keep referring to them as constants.
STATE_PATH = _DEFAULT_STATE
SEED_DIR = _DEFAULT_SEED
TRAIN_PATTERN = _DEFAULT_TRAIN_PATTERN
HEARTBEAT_EVERY = 4              # heartbeat every 4 polls = 1 hour
PLATEAU_NO_IMPROVE_POLLS = 8     # 8 polls = 2 hours with no new spec best
PLATEAU_IMPROVE_MARGIN = 0.02    # "new best" requires beating prior best by 2%
AUTONOMY_PLATEAU_CKPTS = 2       # 2 consecutive ckpts with no autonomy improvement
AUTONOMY_IMPROVE_MARGIN = 0.02   # required autonomy improvement per ckpt


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_nan_skips": 0, "poll_count": 0, "best_spec_recent5": None,
            "best_poll": 0, "last_ckpt_epoch": -1, "last_val_autonomy": None,
            "best_val_autonomy": None, "ckpts_since_best_autonomy": 0,
            "last_plateau_emit_poll": 0, "history": []}


def save_state(s):
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def process_alive():
    # The CLI lets the user disambiguate two concurrent runs by --train-pattern --
    # the live run matches on its --out-dir basename, the env run matches on a
    # different basename. Falls back to the legacy "train_poly_spectral_blk445"
    # pattern if the user didn't override (single-run case).
    out = subprocess.run(
        ["pgrep", "-f", TRAIN_PATTERN],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def latest_event_file():
    evs = sorted(glob.glob(os.path.join(SEED_DIR, "events.out.tfevents.*")))
    return evs[-1] if evs else None


def read_loss(ev_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(ev_path, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in ("Loss/spec", "Loss/tf", "Train/H"):
        if t in tags:
            sc = ea.Scalars(t)
            out[t] = np.array([s.value for s in sc])
    nan = ea.Scalars("Loss/nan_skip") if "Loss/nan_skip" in tags else []
    out["nan_skips"] = int(sum(s.value for s in nan))
    return out


def window_means(vals, w=100, n_windows=20):
    if len(vals) < w * n_windows:
        n_windows = len(vals) // w
    if n_windows == 0:
        return np.array([])
    means = []
    for i in range(n_windows):
        end = len(vals) - i * w
        start = end - w
        if start < 0:
            break
        means.append(vals[start:end].mean())
    return np.array(list(reversed(means)))


def latest_checkpoint():
    # Sort by parsed epoch number, NOT lexicographically -- checkpoint_9.tar would
    # otherwise sort after checkpoint_10.tar and silently mask all double-digit ckpts.
    ckpts = glob.glob(os.path.join(SEED_DIR, "checkpoint_*.tar"))
    if not ckpts:
        return None, -1
    def _ep(p):
        return int(os.path.basename(p).split("_")[1].split(".")[0])
    ckpts.sort(key=_ep)
    latest = ckpts[-1]
    return latest, _ep(latest)


AUTONOMY_SNIPPET = r"""
import sys, os, glob, json, warnings
import numpy as np
sys.path.insert(0, '/home/pearson/code/ouroboros-spectral')
import torch
from scipy.io import wavfile
from train.train import load_model
from train.eval import autonomy_score

DATA = os.path.expanduser('~/ouroboros_data/blk445_syllC/day85')
SR = 44100
DT = 1.0 / SR
N_VAL = 8

wavs = sorted(glob.glob(os.path.join(DATA, '*.wav')))
rng = np.random.default_rng(1234)
idx = np.arange(len(wavs))
rng.shuffle(idx)
n_test = max(1, int(round(0.1 * len(wavs))))
n_val = max(1, int(round(0.1 * len(wavs))))
val_i = idx[n_test:n_test + n_val]
val_wavs = [wavs[i] for i in val_i]

raw, sr = [], None
for wav in val_wavs[:N_VAL]:
    sr, af = wavfile.read(wav)
    if af.dtype == np.int16:
        af = af / -np.iinfo(af.dtype).min
    af = af.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        onoffs = np.atleast_2d(np.loadtxt(wav.replace('.wav', '.txt')))
    on_i = int(round(onoffs[0][0] * sr))
    off_i = int(round(onoffs[0][1] * sr))
    raw.append(af[max(0, on_i - 2000):off_i])
L = min(len(s) for s in raw)
val_vocs = [s[:L] for s in raw]

ckpt_path = sys.argv[1]
ckpt_dir = os.path.dirname(ckpt_path)
model, _, _, _ = load_model(ckpt_dir)
model.eval()
with torch.no_grad():
    score, _, bd = autonomy_score(model, val_vocs, DT, rescale=False, cold_start=True)

# Persist signed amp_pen alongside the offline cache the loss panels reads.
# autonomy_score now puts signed_amp_per_voc + signed_amp_mean in the breakdown so this
# is essentially free -- the integration already happened above.
cache_path = os.path.join(ckpt_dir, 'signed_amp_pen.json')
cache = {}
if os.path.exists(cache_path):
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except Exception:
        cache = {}
ep = int(os.path.basename(ckpt_path).split('_')[1].split('.')[0])
cache[str(ep)] = {
    'per_voc': [float(v) for v in bd.get('signed_amp_per_voc', [])],
    'mean': float(bd.get('signed_amp_mean', float('nan'))),
}
with open(cache_path, 'w') as f:
    json.dump(cache, f, indent=2)

# JSON-safe breakdown: drop the list, keep the scalar mean alongside everything else.
safe_bd = {}
for k, v in bd.items():
    if isinstance(v, bool):
        safe_bd[k] = bool(v)
    elif isinstance(v, list):
        continue  # per-voc lists go to the cache, not the stdout summary
    else:
        safe_bd[k] = float(v)
print(json.dumps({'autonomy': float(score), 'breakdown': safe_bd}))
"""


def run_autonomy_on_checkpoint(ckpt_path):
    r = subprocess.run(
        ["/home/pearson/code/ouroboros/.venv/bin/python", "-c", AUTONOMY_SNIPPET, ckpt_path],
        capture_output=True, text=True, timeout=900,
    )
    if r.returncode != 0:
        last_err = r.stderr.strip().splitlines()[-1] if r.stderr else "unknown"
        return None, {"error": last_err}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
        return out["autonomy"], out["breakdown"]
    except Exception as e:
        return None, {"error": f"parse: {e}; stdout='{r.stdout[:200]}'"}


def emit(parts):
    status = parts.pop("status")
    fields = [f"status={status}"] + [f"{k}={v}" for k, v in parts.items()]
    print(" ".join(fields), flush=True)


def main():
    state = load_state()
    state["poll_count"] = state.get("poll_count", 0) + 1

    alive = process_alive()
    ev_path = latest_event_file()
    if ev_path is None:
        emit({"status": "DEAD", "reason": "no_events_file"})
        save_state(state); return

    loss = read_loss(ev_path)
    spec = loss.get("Loss/spec", np.array([]))
    tf = loss.get("Loss/tf", np.array([]))
    H_arr = loss.get("Train/H", np.array([]))
    nan_skips = loss["nan_skips"]
    cur_H = int(H_arr[-1]) if len(H_arr) else -1
    epoch_est = int(len(spec) // 750)

    spec_w = window_means(spec, w=100, n_windows=20)
    spec_recent5 = float(spec_w[-5:].mean()) if len(spec_w) >= 5 else float("nan")
    spec_prev5 = float(spec_w[-10:-5].mean()) if len(spec_w) >= 10 else float("nan")
    spec_ratio = spec_recent5 / spec_prev5 if (spec_prev5 and not np.isnan(spec_prev5)) else float("nan")
    tf_w = window_means(tf, w=100, n_windows=5)
    tf_recent5 = float(tf_w.mean()) if len(tf_w) else float("nan")

    # Best-spec tracking with margin
    if not np.isnan(spec_recent5):
        prev_best = state.get("best_spec_recent5")
        if prev_best is None or spec_recent5 < prev_best * (1 - PLATEAU_IMPROVE_MARGIN):
            state["best_spec_recent5"] = spec_recent5
            state["best_poll"] = state["poll_count"]
        elif prev_best is None or spec_recent5 < prev_best:
            state["best_spec_recent5"] = spec_recent5

    best_spec_for_fmt = state.get("best_spec_recent5")
    if best_spec_for_fmt is None:
        best_spec_for_fmt = float("nan")
    base = {
        "poll": state["poll_count"], "epoch": epoch_est, "H": cur_H,
        "spec_recent5": f"{spec_recent5:.3g}",
        "spec_ratio": f"{spec_ratio:.3f}",
        "tf_recent5": f"{tf_recent5:.3f}",
        "nan_skips": nan_skips,
        "best_spec": f"{best_spec_for_fmt:.3g}",
        "polls_since_best": state["poll_count"] - state.get("best_poll", 0),
    }

    # 1. Process death
    if not alive:
        emit({"status": "DEAD", **base})
        save_state(state); return

    # 2. NaN spike
    nan_delta = nan_skips - state.get("last_nan_skips", 0)
    state["last_nan_skips"] = nan_skips
    if nan_delta > 50:
        emit({"status": "NAN_SPIKE", "nan_delta": nan_delta, **base})
        save_state(state); return

    # 3. Loss explosion
    if not np.isnan(spec_ratio) and spec_ratio > 2.0:
        emit({"status": "LOSS_SPIKE", **base})
        save_state(state); return

    # 4. New checkpoint -> run autonomy_score
    ckpt_path, ckpt_epoch = latest_checkpoint()
    if ckpt_path is not None and ckpt_epoch > state.get("last_ckpt_epoch", -1):
        score, bd = run_autonomy_on_checkpoint(ckpt_path)
        if score is None:
            emit({"status": "CKPT_ERR", "ckpt_epoch": ckpt_epoch,
                  "err": bd.get("error", "?"), **base})
            save_state(state); return

        prev_val = state.get("last_val_autonomy")
        prev_best_val = state.get("best_val_autonomy")
        if prev_best_val is None or score > prev_best_val + AUTONOMY_IMPROVE_MARGIN:
            state["best_val_autonomy"] = score
            state["ckpts_since_best_autonomy"] = 0
        else:
            state["ckpts_since_best_autonomy"] = state.get("ckpts_since_best_autonomy", 0) + 1
        state["last_val_autonomy"] = score
        state["last_ckpt_epoch"] = ckpt_epoch

        autonomy_line = {
            "ckpt_epoch": ckpt_epoch,
            "val_autonomy": f"{score:+.3f}",
            "val_spec_corr": f"{bd.get('spec_corr', float('nan')):+.3f}",
            "val_amp_pen": f"{bd.get('amp_pen', float('nan')):.2f}",
            "val_pitch_pen": f"{bd.get('pitch_pen', float('nan')):.2f}",
            "val_bounded": f"{bd.get('bounded_frac', float('nan')):.2f}",
            "best_val_autonomy": f"{state['best_val_autonomy']:+.3f}",
            "ckpts_since_best": state["ckpts_since_best_autonomy"],
        }
        status = "PLATEAU" if state["ckpts_since_best_autonomy"] >= AUTONOMY_PLATEAU_CKPTS else "NEW_CKPT"
        emit({"status": status, **autonomy_line, **base})
        save_state(state); return

    # 5. Loss-side plateau (before any ckpt exists, or as a fallback)
    polls_since_best = state["poll_count"] - state.get("best_poll", 0)
    if polls_since_best >= PLATEAU_NO_IMPROVE_POLLS:
        last_emit = state.get("last_plateau_emit_poll", 0)
        if state["poll_count"] - last_emit >= PLATEAU_NO_IMPROVE_POLLS:
            emit({"status": "PLATEAU_LOSS", **base})
            state["last_plateau_emit_poll"] = state["poll_count"]
            save_state(state); return

    # 6. Heartbeat
    if state["poll_count"] % HEARTBEAT_EVERY == 0:
        emit({"status": "HEARTBEAT", **base})

    state["history"].append({"poll": state["poll_count"], "epoch": epoch_est,
                             "spec_recent5": spec_recent5,
                             "tf_recent5": tf_recent5})
    if len(state["history"]) > 200:
        state["history"] = state["history"][-200:]
    save_state(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", default=_DEFAULT_SEED,
                        help="run/seed directory to watch (default: live run).")
    parser.add_argument("--state-path", default=_DEFAULT_STATE,
                        help="JSON file to persist monitor state across polls.")
    parser.add_argument("--train-pattern", default=_DEFAULT_TRAIN_PATTERN,
                        help="pgrep -f pattern that identifies the training process. "
                             "Distinguish concurrent runs by their out-dir basename.")
    args = parser.parse_args()
    SEED_DIR = args.seed_dir
    STATE_PATH = args.state_path
    TRAIN_PATTERN = args.train_pattern
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    main()
