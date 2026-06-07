"""Build a JSON cache of signed amp_pen per checkpoint for a run.

signed_amp_pen(ckpt) = mean_v log( nanstd(autonomous_rollout_v) / nanstd(target_v) )

The mean is over the same cold-start val vocs the live monitor uses (rng seed 1234,
last 10% of stems → first 8 for val). Cached at <run_dir>/seed0/signed_amp_pen.json
so we only compute new ckpts on incremental re-runs. Sign convention: positive = LOUD,
negative = QUIET; |value| matches the autonomy_score amp_pen breakdown to log-precision.

Usage:
    python scripts/build_signed_amp_cache.py <run_dir>                # current run
    python scripts/build_signed_amp_cache.py <r1> <r2> <r3>            # multiple runs
"""

import glob
import json
import os
import sys
import warnings

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.train import load_model  # noqa: E402
from train.eval import integrate_poly_autonomous, correct  # noqa: E402

DATA = os.path.expanduser("~/ouroboros_data/blk445_syllC/day85")
SR = 44100
DT = 1.0 / SR
N_VAL = 8
SILENCE_PAD = 2000


def _val_vocs():
    wavs = sorted(glob.glob(os.path.join(DATA, "*.wav")))
    rng = np.random.default_rng(1234)
    idx = np.arange(len(wavs))
    rng.shuffle(idx)
    n = max(1, int(round(0.1 * len(wavs))))
    val_wavs = [wavs[i] for i in idx[n:n + n]]
    raw_audio = []
    for wav in val_wavs[:N_VAL]:
        sr, af = wavfile.read(wav)
        if af.dtype == np.int16:
            af = af / -np.iinfo(af.dtype).min
        af = af.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        on_i = int(round(onoffs[0][0] * sr))
        off_i = int(round(onoffs[0][1] * sr))
        raw_audio.append(af[max(0, on_i - SILENCE_PAD):off_i])
    L = min(len(s) for s in raw_audio)
    return [s[:L] for s in raw_audio]


def process_run(run_dir):
    seed_dir = os.path.join(run_dir, "seed0")
    if not os.path.isdir(seed_dir):
        print(f"skip {run_dir}: no seed0", file=sys.stderr)
        return
    ckpts = glob.glob(os.path.join(seed_dir, "checkpoint_*.tar"))
    if not ckpts:
        print(f"skip {run_dir}: no ckpts", file=sys.stderr)
        return

    cache_path = os.path.join(seed_dir, "signed_amp_pen.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    def _ep(p):
        return int(os.path.basename(p).split("_")[1].split(".")[0])
    ckpts.sort(key=_ep)
    todo = [c for c in ckpts if str(_ep(c)) not in cache]
    if not todo:
        print(f"{run_dir}: all {len(ckpts)} ckpts already cached", flush=True)
        return

    val_vocs = _val_vocs()
    tgt_log = np.array([float(np.log(np.nanstd(correct(np.asarray(s, dtype=np.float64))))) for s in val_vocs])
    model, _, _, _ = load_model(seed_dir)
    print(f"{run_dir}: caching {len(todo)} new ckpts (of {len(ckpts)} total)", flush=True)

    for ckpt in todo:
        ep = _ep(ckpt)
        sd = torch.load(ckpt, weights_only=False, map_location="cuda")
        model.load_state_dict(sd["ouroboros"])
        model.eval()
        per_voc_signed = []
        for seg in val_vocs:
            seg = np.asarray(seg, dtype=np.float64)
            with torch.no_grad():
                auto = integrate_poly_autonomous(model, seg, DT, verbose=False)
            auto_rms = float(np.nanstd(np.asarray(auto, dtype=np.float64)))
            if not np.isfinite(auto_rms) or auto_rms <= 0:
                per_voc_signed.append(float("nan"))
            else:
                per_voc_signed.append(float(np.log(auto_rms)) - float(tgt_log[len(per_voc_signed)]))
        cache[str(ep)] = {
            "per_voc": per_voc_signed,
            "mean": float(np.nanmean(per_voc_signed)) if per_voc_signed else float("nan"),
        }
        # incremental save so a kill mid-loop doesn't lose progress
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"  ckpt {ep:>2}: signed mean = {cache[str(ep)]['mean']:+.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for rd in sys.argv[1:]:
        process_run(rd)
