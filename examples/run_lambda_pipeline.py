"""
Production selection pipeline for the (low-pass) polynomial Ouroboros: pick the best SEED, then
amplitude-rescale at generation.

Autonomous-reconstruction quality is dominated by the random SEED (init + batch order), not by the
kernel-weight lambda (lambda is irrelevant for autonomy; see docs/autonomous_amplitude.md). So the
recipe is: fix lambda, train several seeds, and SELECT the best seed on the VALIDATION set by
RESCALED autonomy (amplitude is a free overall-scale gauge, fixed at generation by
train.eval.generate_autonomous). The (0,0) polynomial 'alpha' term is kept on (R2-neutral here; may
help other datasets). A FILE-LEVEL holdout is used for the autonomy vocalizations:
  - training chunks from most data shards,
  - validation autonomy vocs from a held-out shard (seed selection),
  - test autonomy vocs from another held-out shard (final report + a rescaled reconstruction wav).

Pass --lam <=0 to instead sweep the standard 7-point lambda grid.

Run from the repo root:
    python -m examples.run_lambda_pipeline --data-glob 'data500/gabo_p*' --out-dir ./poly_pipeline \
        --n-epochs 50 --n-seeds 5 --lam 1.068 --drive-lowpass-ms 1.0 --d-state 4
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.io import wavfile

from data.load_data import get_segmented_audio
from data.data_utils import get_loaders
from train.model_cv import model_cv_lambdas
from train.eval import autonomy_score, generate_autonomous


def load_voc_windows(data_dir, n_vocs, start_offset_ms, n):
    """held-out sustained vocalization windows (start `start_offset_ms` after onset)."""
    segs = []
    for wav in sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n_vocs]:
        sr, af = wavfile.read(wav)
        af = af.astype(np.float64)
        onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        s = int(onoffs[0][0] * sr) + int(start_offset_ms / 1e3 * sr)
        seg = af[s:s + n]
        if len(seg) == n:
            segs.append(seg)
    return segs, sr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", default="data500/gabo_p*")
    p.add_argument("--out-dir", default="poly_pipeline")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.068,
                   help="fixed kernel-weight lambda (lambda is irrelevant for autonomy; the SEED "
                        "is what matters, so we fix lambda and select over seeds). Pass <=0 to "
                        "sweep the standard 7-point lambda grid instead.")
    p.add_argument("--keep-const", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the (0,0) polynomial 'alpha' forcing term (R2-neutral on gabo; may "
                        "help on other datasets)")
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--n-kernels", type=int, default=15)
    p.add_argument("--context-len", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-val-vocs", type=int, default=3)
    p.add_argument("--n-test-vocs", type=int, default=3)
    p.add_argument("--auto-n", type=int, default=3000, help="autonomy window length (samples)")
    p.add_argument("--start-offset-ms", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    dirs = sorted(d for d in glob.glob(args.data_glob) if os.path.isdir(d))
    assert len(dirs) >= 3, "need >=3 data shards for train/val/test holdout"
    train_dirs, val_dir, test_dir = dirs[:-2], dirs[-2], dirs[-1]
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # training chunks from the train shards
    chunks, sr = [], None
    per = 100000 // max(1, len(train_dirs))
    for d in train_dirs:
        audio, sr = get_segmented_audio(d, d, max_vocs=per, context_len=args.context_len,
                                        seed=args.seed, training=True, extend=True, shuffle_order=True)
        chunks += audio
    dt = 1 / sr
    dls = get_loaders(np.stack(chunks, 0), num_workers=args.n_jobs, batch_size=args.batch_size,
                      train_size=0.6, cv=True, seed=args.seed, dt=dt)

    # held-out autonomy vocalizations (file-level holdout)
    val_vocs, _ = load_voc_windows(val_dir, args.n_val_vocs, args.start_offset_ms, args.auto_n)
    test_vocs, _ = load_voc_windows(test_dir, args.n_test_vocs, args.start_offset_ms, args.auto_n)
    print(f"train chunks={len(chunks)} from {len(train_dirs)} shards | "
          f"val_vocs={len(val_vocs)} from {os.path.basename(val_dir)} | "
          f"test_vocs={len(test_vocs)} from {os.path.basename(test_dir)} | sr={sr}", flush=True)

    best_model = model_cv_lambdas(
        dls=dls, dt=dt, n_epochs=args.n_epochs, lr=1e-3, n_kernels=args.n_kernels,
        expand_factor=10, n_layers=3, d_state=args.d_state, d_conv=4, tau=dt,
        model_path=out_dir, save_freq=max(args.n_epochs // 5, 1),
        drive_lowpass_ms=args.drive_lowpass_ms, n_seeds=args.n_seeds,
        selection="autonomy", val_vocs=val_vocs, test_vocs=test_vocs,
        keep_const=args.keep_const, rescale_autonomy=True,
        lambdas=None if args.lam <= 0 else [args.lam],
    )

    # DEPLOYED generation: rescaled autonomous reconstruction of the held-out test vocs.
    # (Amplitude is a free gauge fixed at generation; selection above already used rescaled autonomy.)
    if test_vocs:
        score, _, bd = autonomy_score(best_model, test_vocs, dt, rescale=True)
        recon = generate_autonomous(best_model, test_vocs[0], dt, rescale=True)
        wav = (recon / (np.abs(recon).max() + 1e-12) * 0.95 * 32767).astype(np.int16)
        wavfile.write(os.path.join(out_dir, "selected_autonomous_recon.wav"), int(round(1 / dt)), wav)
        manifest = {"selected_lambda": float(args.lam), "n_seeds": int(args.n_seeds),
                    "keep_const": bool(args.keep_const), "drive_lowpass_ms": float(args.drive_lowpass_ms),
                    "rescaled_test_autonomy": score, "spec_corr": bd["spec_corr"],
                    "pitch_pen": bd["pitch_pen"], "bounded_frac": bd["bounded_frac"]}
        with open(os.path.join(out_dir, "selected_model.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"deployed (rescaled) test autonomy = {score:+.3f}  (spec={bd['spec_corr']:.2f}, "
              f"pitch_pen={bd['pitch_pen']:.2f}, bounded={bd['bounded_frac']:.2f})", flush=True)
        print(f"wrote {out_dir}/selected_autonomous_recon.wav + selected_model.json", flush=True)
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
