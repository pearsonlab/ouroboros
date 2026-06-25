"""Train the polynomial Ouroboros on a single (bird, syllable, day) with the
DDSP-style spectral-rollout loss and a cold-start initial condition.

Pipeline:
  1. File-level holdout. Glob WAV stems in --data-dir, split into train/val/test
     by stem (e.g. last 10% of stems for test, the next 10% for val, the rest
     for training).
  2. Build the edge-biased training set with `get_audio_training_edge_weighted`
     (heavily oversamples onset/offset windows, with a small mid-syllable share
     for steady-state coverage). Per-segment category labels drive the
     cold-start IC choice in the spectral training step.
  3. Load held-out cold-start val/test vocs (silence lead-in + voc).
  4. Run `model_seed_cv_spectral` (seed loop + optional culling + cold-start
     raw autonomy selection).
  5. Save the deployed (rescaled) autonomous reconstruction of the first test
     voc + a manifest with the selection breakdown.

Example:
  python -m examples.train_poly_spectral_blk445 \\
    --data-dir ~/ouroboros_data/blk445_syllC/day85 --out-dir poly_spectral_day85 \\
    --n-seeds 4 --n-epochs 50 --cull-frac 0.4 --cull-keep 2 \\
    --context-len 0.05 --silence-prefix-ms 25 --silence-suffix-ms 25 \\
    --H-min 512 --H-max 2048 --lam-spec 1.0 --lam-tf 1.0 --ic-noise-rms 1e-3
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
from scipy.io import wavfile

from data.load_data import get_audio_training_edge_weighted
from data.data_utils import get_loaders_edge
from train.model_cv import model_seed_cv_spectral
from train.eval import autonomy_score, generate_autonomous


def _stems(data_dir, audio_id="_cleaned.wav"):
    # Files in the staged dir end with `.wav` (symlinks point at *_cleaned.wav);
    # but legacy callers may have used the raw _cleaned suffix. Try both.
    wavs = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    return wavs


def _group_of(stem, stratify_sep):
    """Group key from a stem: the prefix before the FIRST occurrence of stratify_sep.
    Returns '__none__' when stratify_sep is empty or absent."""
    if not stratify_sep or stratify_sep not in stem:
        return "__none__"
    return stem.split(stratify_sep, 1)[0]


def _file_level_split(wavs, val_frac=0.1, test_frac=0.1, seed=1234, stratify_sep=None):
    """Deterministic per-stem split. If stratify_sep is provided, perform a
    per-group stratified split so val/test sets are guaranteed to cover every
    group (e.g. every (bird, syllable) prefix in a multi-syllable run)."""
    rng = np.random.default_rng(seed)
    if not stratify_sep:
        idx = np.arange(len(wavs))
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(wavs))))
        n_val = max(1, int(round(val_frac * len(wavs))))
        test_i = idx[:n_test]
        val_i = idx[n_test:n_test + n_val]
        train_i = idx[n_test + n_val:]
        return ([wavs[i] for i in train_i],
                [wavs[i] for i in val_i],
                [wavs[i] for i in test_i])
    # Stratified per-prefix
    by_group = {}
    for w in wavs:
        g = _group_of(os.path.basename(w), stratify_sep)
        by_group.setdefault(g, []).append(w)
    train, val, test = [], [], []
    for g in sorted(by_group):
        ws = by_group[g]
        idx = np.arange(len(ws))
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(ws))))
        n_val = max(1, int(round(val_frac * len(ws))))
        test.extend(ws[i] for i in idx[:n_test])
        val.extend(ws[i] for i in idx[n_test:n_test + n_val])
        train.extend(ws[i] for i in idx[n_test + n_val:])
    rng.shuffle(train)
    return train, val, test


def _coldstart_from_files(wav_files, silence_pad_samples, n_vocs, stratify_sep=None,
                          target_duration_ms=0.0):
    """Held-out cold-start vocs from an explicit list of WAV files (parallel to
    examples/_voc_windows.load_voc_windows_coldstart, but file-list-based so we can
    do file-level train/val/test holdout inside a single data directory).

    When stratify_sep is given, the selection is round-robin across groups so each
    (bird, syllable) prefix contributes ~n_vocs/N_groups vocs. n_vocs is the TOTAL
    number of returned vocs; groups with fewer than ceil(n_vocs/N) usable files
    contribute what they have.

    target_duration_ms>0 extends each voc to cover multiple consecutive
    annotations: each voc spans from (first-onset - silence_pad) to the offset
    of the LAST annotation whose end is within `target_duration_ms` of the
    first onset. Default 0 = single-syllable (legacy behaviour)."""
    if stratify_sep:
        by_group = {}
        for w in wav_files:
            g = _group_of(os.path.basename(w), stratify_sep)
            by_group.setdefault(g, []).append(w)
        groups = sorted(by_group)
        per_group = max(1, n_vocs // max(1, len(groups)))
        picked = []
        for g in groups:
            picked.extend(by_group[g][:per_group])
        wav_iter = picked[:n_vocs]
    else:
        wav_iter = wav_files[:n_vocs]
    raw = []
    sr = None
    for wav in wav_iter:
        sr, af = wavfile.read(wav)
        if af.dtype == np.int16:
            af = af / -np.iinfo(af.dtype).min
        af = af.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onoffs = np.atleast_2d(np.loadtxt(wav.replace(".wav", ".txt")))
        on_i = int(round(onoffs[0][0] * sr))
        if target_duration_ms > 0:
            # Fixed-length window so all vocs are the same length regardless
            # of source annotation density. See scripts/monitor_spectral_diagnose.py.
            target_len = silence_pad_samples + int(round(target_duration_ms / 1e3 * sr))
            start = max(0, on_i - silence_pad_samples)
            raw.append(af[start:start + target_len])
        else:
            off_i = int(round(onoffs[0][1] * sr))
            start = max(0, on_i - silence_pad_samples)
            raw.append(af[start:off_i])
    if not raw:
        return [], sr
    L = min(len(s) for s in raw)
    segs = [s[:L] for s in raw]
    return segs, sr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data-dir", required=True,
                   help="staged single-syllable directory containing {stem}.wav + {stem}.txt "
                        "pairs (run scripts/stage_finch_blk445_syllC.py first).")
    p.add_argument("--out-dir", default="poly_spectral")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--n-val-vocs", type=int, default=8)
    p.add_argument("--n-test-vocs", type=int, default=8)
    p.add_argument("--coldstart-duration-ms", type=float, default=0.0,
                   help="extend each val/test voc to cover multiple consecutive syllable "
                        "annotations spanning at least this many ms from the first onset. "
                        "0 = single-syllable (legacy). Recommended ~500 for org545.")
    p.add_argument("--stratify-sep", default=None,
                   help="If set, file-level split and cold-start voc picker stratify by the "
                        "prefix before the FIRST occurrence of this separator in the stem. "
                        "Use '__' for the multi-syllable staging (scripts/stage_finch_multi.py).")
    p.add_argument("--silence-pad-samples", type=int, default=2000,
                   help="cold-start lead-in (samples). 2000 matches the existing finchsim convention.")
    # sampler
    p.add_argument("--context-len", type=float, default=0.05,
                   help="training segment length in seconds (50 ms ~ 2205 samples @ 44.1 kHz).")
    p.add_argument("--edge-ms", type=float, default=10.0,
                   help="mid-syllable windows lie strictly inside [on+edge_ms, off-edge_ms].")
    p.add_argument("--silence-prefix-ms", type=float, default=25.0,
                   help="pre-onset audio in ONSET windows (cold-start training lead-in).")
    p.add_argument("--silence-suffix-ms", type=float, default=25.0,
                   help="post-offset audio in OFFSET windows.")
    p.add_argument("--ratio", default="0.4,0.4,0.2",
                   help="sampling ratio ONSET,OFFSET,MID (comma-separated, summed to 1).")
    p.add_argument("--max-segs", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-jobs", type=int, default=4)
    # model
    p.add_argument("--n-kernels", type=int, default=15)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--d-state", type=int, default=4)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand-factor", type=int, default=10)
    p.add_argument("--drive-lowpass-ms", type=float, default=1.0)
    p.add_argument("--keep-const", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--osc-init", action=argparse.BooleanOptionalAction, default=False,
                   help="Strategy 1: initialize each seed as a marginal van der Pol limit cycle "
                        "(small negative damping + seeded y^2*ydot re-damping + y^3 hardening) so "
                        "oscillation can ignite from silence instead of the dissipative default init.")
    p.add_argument("--checkpoint-encoder", action=argparse.BooleanOptionalAction, default=False,
                   help="gradient-checkpoint the three Mamba drive encoders. Their activations over "
                        "the doubled-length sequence dominate training memory (~6.9 GB at B=64); "
                        "recomputing them in backward trades ~one extra encoder forward per step for "
                        "the headroom to run larger batch sizes (e.g. B=128) within 11 GB.")
    p.add_argument("--use-tract", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable the LTI vocal-tract filter (FFT-domain H = OEC resonance × trachea "
                        "comb, identity-init, learnable). Adds a source/filter split: forward "
                        "deconvolves audio through H^-1 to source, polynomial generates source'', "
                        "then filters back through H to match audio''. No-op at init.")
    p.add_argument("--tract-n-sec", type=int, default=3,
                   help="Number of second-order pole/zero sections in the rational tract. Each "
                        "section adds one formant + anti-formant pair. Default 3 (~3 formants); "
                        "bird vocalizations typically show 4-6 prominent peaks so n_sec=5 or 6 "
                        "can capture finer spectral structure. Identity-init regardless of n_sec.")
    p.add_argument("--use-envelope", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable the zero-init learnable amplitude envelope e(t)=exp(lowpass(.)). "
                        "Applied as a positive scalar multiplier on the rollout source waveform "
                        "(envelope-as-output, NOT folded into the kernel). e starts at 1 so the "
                        "model at init matches legacy behavior; targets the global-amplitude "
                        "wandering directly.")
    p.add_argument("--env-lowpass-ms", type=float, default=20.0,
                   help="Time constant (ms) for the lowpass on the envelope head output before "
                        "exp. ~20 ms matches syllable amplitude-modulation timescales (10-100 Hz).")
    p.add_argument("--lam", type=float, default=1.0,
                   help="degree-grading on the polynomial kernel weights: term (i, j) is "
                        "weighted by lam**(i+j) inside the kernel-weight L2 penalty. lam<1 "
                        "softens higher-degree terms, lam>1 strengthens them, lam=1 is flat. "
                        "Penalty itself is gated by --lam-reg.")
    # loss
    p.add_argument("--lam-spec", type=float, default=1.0)
    p.add_argument("--lam-tf", type=float, default=1.0,
                   help="variance-normalized acceleration MSE anchor. Default-on so early "
                        "epochs have a smooth gradient signal before the spectral basin is informative.")
    p.add_argument("--lam-reg", type=float, default=0.0,
                   help="overall scale on the degree-graded L2 penalty `sum lam**(i+j) * weights**2` "
                        "applied to the polynomial kernel weights. 0 disables (no behavior change). "
                        "lam (above) shapes the degree grading; lam_reg scales the whole penalty.")
    p.add_argument("--lam-env-anchor", type=float, default=0.0,
                   help="weight on the mean((e - 1)^2) envelope gauge anchor. Quadratic "
                        "penalty with minimum at e=1 (its identity-init value); breaks the "
                        "spec loss's (e, x) -> (k*e, x/k) gauge invariance. The soft-tanh on "
                        "x bounds the (e -> 0, x -> infinity) direction by construction, so "
                        "only the (e -> infinity) direction needs penalizing -- which (e-1)^2 "
                        "does asymmetrically (large e penalized hard, small e mildly). "
                        "Per-sample backward grad 2*(e-1)/N is bounded and smooth -- no eps "
                        "machinery, no small-e gradient cliff. Only meaningful when "
                        "use_envelope is enabled. Default 0 disables.")
    p.add_argument("--lam-env-log", type=float, default=0.0,
                   help="weight on the LOG-RATIO envelope loss (|log((env(a)+eps)/(env(g)+eps))|). "
                        "Symmetric in (auto, target) -- penalizes shrinking past target the same as "
                        "growing past it (unlike --lam-env, which has a trivial floor of 1 at silence "
                        "and lets the model overshoot into decay). Same env_warmup ramp applies. "
                        "Default 0 disables.")
    p.add_argument("--env-log-eps", type=float, default=1e-4,
                   help="soft noise floor inside the log() of the log-ratio envelope loss; ~1e-4 "
                        "is the per-sample noise floor of normalized blk445 audio.")
    p.add_argument("--lam-env", type=float, default=0.0,
                   help="Gaussian-envelope L1 amplitude pin. 0 disables.")
    p.add_argument("--env-ms", type=float, default=2.0)
    p.add_argument("--spec-configs", default="256,64;512,128;1024,256",
                   help="MRSTFT (n_fft,hop) configs, semicolon-separated.")
    p.add_argument("--H-min", type=int, default=512)
    p.add_argument("--H-max", type=int, default=2048)
    p.add_argument("--H-schedule", choices=["geom", "linear", "const", "pow2"],
                   default="geom")
    p.add_argument("--rollout-backend",
                   choices=["eager", "graphstep", "cudagraph", "compile", "scan"],
                   default="eager",
                   help="RK4 rollout backend. 'graphstep' CUDA-graphs one RK4 step and replays it "
                        "(~3.9x vs eager at H=2048, works on Pascal, H-invariant capture) -- the "
                        "recommended fast backend here. 'cudagraph'/'compile' want --H-schedule pow2. "
                        "'scan' lowers the step once via torch.compile (sm>=70), else eager fold.")
    p.add_argument("--spec-warmup-epochs", type=int, default=5,
                   help="linearly ramp lam_spec from 0 to its target over this many epochs. "
                        "0 disables the warmup. Needed at random init because the spectral "
                        "term against quiet/onset targets is enormous and would swamp the TF anchor.")
    # Step-based curriculum overrides (None -> derive from _epochs * batches_per_epoch).
    p.add_argument("--spec-warmup-steps", type=int, default=None,
                   help="number of GLOBAL BATCHES over which to ramp lam_spec. Overrides "
                        "--spec-warmup-epochs when set; lets the curriculum stay calibrated "
                        "when the dataset size changes.")
    p.add_argument("--env-warmup-steps", type=int, default=None,
                   help="number of GLOBAL BATCHES over which to ramp lam_env / lam_env_log. "
                        "Overrides --env-warmup-epochs when set.")
    p.add_argument("--H-total-steps", type=int, default=None,
                   help="total GLOBAL BATCHES across which the H curriculum runs. Overrides "
                        "--n-epochs * batches_per_epoch when set.")
    p.add_argument("--env-warmup-epochs", type=int, default=0,
                   help="linearly ramp lam_env from 0 to its target over this many epochs. "
                        "0 disables the warmup. Set to 5+ when using a large --lam-env (e.g. 1e4+) "
                        "so the random-init env gradient doesn't blow up params before TF stabilizes.")
    p.add_argument("--ic-noise-rms", type=float, default=1e-3,
                   help="cold-start initial-condition noise RMS (training only).")
    p.add_argument("--grad-clip", type=float, default=5.0)
    # run / selection
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=None,
                   help="end LR for the linear ramp. None = constant --lr (legacy).")
    p.add_argument("--lr-ramp-epochs", type=int, default=5,
                   help="epochs over which lr ramps from --lr to --lr-end. No-op if --lr-end None.")
    p.add_argument("--save-minutes", type=float, default=0.0,
                   help="intra-epoch save cadence (minutes wall-clock). 0 disables. "
                        "Saved as inflight_latest.tar (separate from per-epoch ckpts).")
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--cull-frac", type=float, default=0.0)
    p.add_argument("--cull-keep", type=int, default=2)
    p.add_argument("--save-freq", type=int, default=5)
    p.add_argument("--max-saved", type=int, default=60,
                   help="how many per-epoch checkpoints to retain on disk before evicting the "
                        "oldest. Default 60 keeps every checkpoint of a 50-epoch run at save-freq=1.")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    wavs = _stems(data_dir)
    if not wavs:
        raise SystemExit(f"no WAV files found under {data_dir}; "
                         "run scripts/stage_finch_blk445_syllC.py first.")
    train_wavs, val_wavs, test_wavs = _file_level_split(
        wavs, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
        stratify_sep=args.stratify_sep,
    )
    train_txts = [w.replace(".wav", ".txt") for w in train_wavs]
    if args.stratify_sep:
        # Show per-group composition so accidental imbalance is obvious in the log.
        from collections import Counter
        def _group_counts(ws):
            return dict(sorted(Counter(_group_of(os.path.basename(w), args.stratify_sep)
                                       for w in ws).items()))
        print(f"file-level split: train={len(train_wavs)} val={len(val_wavs)} test={len(test_wavs)}",
              flush=True)
        print(f"  train per-group: {_group_counts(train_wavs)}", flush=True)
        print(f"  val   per-group: {_group_counts(val_wavs)}", flush=True)
        print(f"  test  per-group: {_group_counts(test_wavs)}", flush=True)
    else:
        print(f"file-level split: train={len(train_wavs)} val={len(val_wavs)} test={len(test_wavs)}",
              flush=True)

    ratio = tuple(float(r) for r in args.ratio.split(","))
    assert len(ratio) == 3, "--ratio must be ONSET,OFFSET,MID"
    spec_configs = tuple(
        tuple(int(v) for v in c.split(",")) for c in args.spec_configs.split(";")
    )

    segs, cats, sr = get_audio_training_edge_weighted(
        train_wavs, train_txts,
        context_len=args.context_len,
        edge_ms=args.edge_ms,
        silence_prefix_ms=args.silence_prefix_ms,
        silence_suffix_ms=args.silence_suffix_ms,
        ratio=ratio,
        max_segs=args.max_segs,
        seed=args.seed,
    )
    if not segs:
        raise SystemExit("edge-biased sampler returned 0 segments; check the data layout.")
    data = np.stack(segs, axis=0).astype(np.float64)
    import collections
    counts = collections.Counter(cats.tolist())
    print(f"sampler: {len(segs)} segs (ONSET={counts[0]}, OFFSET={counts[1]}, MID={counts[2]}) "
          f"L_seg={data.shape[1]} sr={sr}", flush=True)

    dt = 1.0 / sr
    dls = get_loaders_edge(
        data, cats, num_workers=args.n_jobs, batch_size=args.batch_size,
        train_size=1.0 - args.val_frac - args.test_frac,  # split inside the sampler pool
        cv=False, seed=args.seed,
    )
    # `cv=False` because we use FILE-LEVEL holdout for val/test (cleaner than splitting
    # inside the sampler pool, which would mix recordings across the split).

    val_vocs, _ = _coldstart_from_files(val_wavs, args.silence_pad_samples, args.n_val_vocs,
                                        stratify_sep=args.stratify_sep,
                                        target_duration_ms=args.coldstart_duration_ms)
    test_vocs, _ = _coldstart_from_files(test_wavs, args.silence_pad_samples, args.n_test_vocs,
                                         stratify_sep=args.stratify_sep,
                                         target_duration_ms=args.coldstart_duration_ms)
    voc_L = len(val_vocs[0]) if val_vocs else 0
    print(f"val_vocs={len(val_vocs)} test_vocs={len(test_vocs)} (cold-start L={voc_L})",
          flush=True)

    best_model = model_seed_cv_spectral(
        dls=dls, dt=dt, val_vocs=val_vocs, test_vocs=test_vocs,
        n_kernels=args.n_kernels, n_layers=args.n_layers,
        d_state=args.d_state, d_conv=args.d_conv, expand_factor=args.expand_factor,
        tau=dt, drive_lowpass_ms=args.drive_lowpass_ms, keep_const=args.keep_const,
        osc_init=args.osc_init,
        checkpoint_encoder=args.checkpoint_encoder,
        use_tract=args.use_tract, tract_n_sec=args.tract_n_sec, use_envelope=args.use_envelope,
        env_lowpass_ms=args.env_lowpass_ms,
        lam=args.lam,
        n_epochs=args.n_epochs, lr=args.lr, n_seeds=args.n_seeds,
        cull_frac=args.cull_frac, cull_keep=args.cull_keep,
        save_freq=args.save_freq, max_saved=args.max_saved, model_path=out_dir,
        H_min=args.H_min, H_max=args.H_max, H_schedule=args.H_schedule,
        lam_spec=args.lam_spec, lam_tf=args.lam_tf,
        lam_env=args.lam_env, lam_env_log=args.lam_env_log, env_log_eps=args.env_log_eps,
        env_ms=args.env_ms,
        lam_reg=args.lam_reg,
        lam_env_anchor=args.lam_env_anchor,
        spec_warmup_epochs=args.spec_warmup_epochs,
        env_warmup_epochs=args.env_warmup_epochs,
        spec_warmup_steps=args.spec_warmup_steps,
        env_warmup_steps=args.env_warmup_steps,
        H_total_steps=args.H_total_steps,
        spec_configs=spec_configs, ic_noise_rms=args.ic_noise_rms, grad_clip=args.grad_clip,
        rollout_backend=args.rollout_backend,
        lr_end=args.lr_end, lr_ramp_epochs=args.lr_ramp_epochs,
        save_minutes=args.save_minutes,
        cold_start_autonomy=True, rescale_autonomy=False,
    )

    # Deployed generation: rescaled autonomous reconstruction of the first test voc.
    manifest = {
        "data_dir": data_dir,
        "n_train_files": len(train_wavs),
        "n_val_files": len(val_wavs),
        "n_test_files": len(test_wavs),
        "selected_seed": int(getattr(best_model, "_selected_seed", -1)),
        "selected_lambda": float(getattr(best_model, "_selected_lambda", args.lam)),
        "selected_val_autonomy": float(getattr(best_model, "_selected_val_autonomy", float("nan"))),
        "selected_test_autonomy": float(getattr(best_model, "_selected_test_autonomy", float("nan"))),
        "selected_test_breakdown": dict(getattr(best_model, "_selected_test_breakdown", {})),
        "context_len_s": args.context_len,
        "silence_prefix_ms": args.silence_prefix_ms,
        "silence_suffix_ms": args.silence_suffix_ms,
        "ratio": list(ratio),
        "max_segs": args.max_segs,
        "drive_lowpass_ms": args.drive_lowpass_ms,
        "keep_const": bool(args.keep_const),
        "osc_init": bool(args.osc_init),
        "use_tract": bool(args.use_tract),
        "tract_n_sec": int(args.tract_n_sec),
        "use_envelope": bool(args.use_envelope),
        "env_lowpass_ms": args.env_lowpass_ms,
        "lam_spec": args.lam_spec,
        "lam_tf": args.lam_tf,
        "lam_env": args.lam_env,
        "lam_env_log": args.lam_env_log,
        "env_log_eps": args.env_log_eps,
        "env_ms": args.env_ms,
        "lam_reg": args.lam_reg,
        "lam_env_anchor": args.lam_env_anchor,
        "lam": args.lam,
        "spec_configs": [list(c) for c in spec_configs],
        "H_min": args.H_min,
        "H_max": args.H_max,
        "H_schedule": args.H_schedule,
        "rollout_backend": args.rollout_backend,
        "spec_warmup_epochs": args.spec_warmup_epochs,
        "env_warmup_epochs": args.env_warmup_epochs,
        "spec_warmup_steps": args.spec_warmup_steps,
        "env_warmup_steps": args.env_warmup_steps,
        "H_total_steps": args.H_total_steps,
        "ic_noise_rms": args.ic_noise_rms,
        "sr": int(sr),
    }
    if test_vocs:
        # Cold-start RAW score (the no-rescale number selection used; the metric the
        # deployment target actually has to clear) is already in selected_test_breakdown.
        # ALSO report the deployed (rescale=True) number so the manifest captures both.
        deployed_score, _, deployed_bd = autonomy_score(
            best_model, test_vocs, dt, rescale=True, cold_start=True,
        )
        manifest["rescaled_test_autonomy"] = float(deployed_score)
        def _jsonable(v):
            # breakdown values are scalars EXCEPT signed_amp_per_voc, which is a list of
            # floats -- pass lists/bools through and coerce the rest to float for JSON.
            if isinstance(v, bool):
                return bool(v)
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            return float(v)
        manifest["rescaled_test_breakdown"] = {k: _jsonable(v) for k, v in deployed_bd.items()}
        recon = generate_autonomous(best_model, test_vocs[0], dt, rescale=True)
        if np.isfinite(recon).all() and np.abs(recon).max() > 0:
            wav = (recon / (np.abs(recon).max() + 1e-12) * 0.95 * 32767).astype(np.int16)
            wavfile.write(os.path.join(out_dir, "selected_autonomous_recon.wav"),
                          int(round(1 / dt)), wav)
            print(f"wrote {out_dir}/selected_autonomous_recon.wav", flush=True)
        else:
            print("WARNING: recon non-finite or zero; skipped WAV write.", flush=True)

    with open(os.path.join(out_dir, "selected_model.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_dir}/selected_model.json", flush=True)
    print("PIPELINE DONE", flush=True)


if __name__ == "__main__":
    main()
