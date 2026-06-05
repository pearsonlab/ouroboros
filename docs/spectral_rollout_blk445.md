# Spectral-rollout polynomial Ouroboros on blk445 syllable_C

DDSP-style multi-resolution STFT magnitude loss (arxiv:1910.11480) on a
teacher-forced RK4 reconstruction of the polynomial Ouroboros, with cold-start
training on syllable onset/offset windows.

## Quick start

```bash
# 1. Stage the data (idempotent symlinks; uses ~/isilon/.../blk445).
python scripts/stage_finch_blk445_syllC.py
# day84/day85/day86 are staged under ~/ouroboros_data/blk445_syllC/.

# 2. Train (single day, single syllable).
python -m examples.train_poly_spectral_blk445 \
    --data-dir ~/ouroboros_data/blk445_syllC/day85 \
    --out-dir poly_spectral_day85 \
    --n-seeds 4 --n-epochs 50 \
    --cull-frac 0.4 --cull-keep 2 \
    --context-len 0.05 --silence-prefix-ms 25 --silence-suffix-ms 25 \
    --ratio 0.4,0.4,0.2 --max-segs 6000 \
    --H-min 512 --H-max 2000 --H-schedule geom \
    --lam-spec 1.0 --lam-tf 1.0 --ic-noise-rms 1e-3
```

Outputs land in `poly_spectral_day85/`:

- `seed{0..n}/checkpoint_*.tar` per-seed checkpoints
- `seed_cv.csv` selection table (one row per seed, val autonomy + breakdown)
- `selected_model.json` manifest with the selection breakdown, all loss hparams,
  test cold-start raw and rescaled autonomy scores
- `selected_autonomous_recon.wav` deployed (rescaled) autonomous reconstruction
  of the first test voc

## What's different from the legacy pipeline

| Knob | Legacy (run_lambda_pipeline) | This script |
|------|------------------------------|-------------|
| Loss | MSE on d²y/ds² (acceleration) | MRSTFT magnitude on TF rollout + variance-normalized d² MSE anchor |
| Training segments | uniform context_len chunks across each voc | onset-aligned + offset-aligned + mid-syllable, sampled in ratio |
| Initial condition | data IC throughout | data IC on OFFSET/MID; low-amp Gaussian noise on ONSET (cold-start training) |
| λ (kernel-weight) | swept (7-point grid) | fixed at 1.068 (λ is irrelevant for autonomy; see docs/autonomous_amplitude.md) |
| Selection | rescaled or cold-start-raw autonomy | cold-start-raw autonomy (`rescale=False`) only |
| Holdout | shard-level (multi-directory) | file-level inside a single day directory |

## Loss math (per batch)

`spectral_rollout_step(model, x, dxdt, d2x, dt, H, ...)` computes:

1. **Drives**: `omega, gamma, weighted_kernels, weights = model.get_funcs(x, dxdt.clone(), dt)`.
   The Mamba encoders see the full (B, L, 1) target; drives are low-passed
   inside the model when `drive_lowpass_ms > 0`.

2. **Teacher-forced acceleration anchor** (variance-normalized so the scale is
   stable across vocs; matches `train/rollout_refine.py:132`):

       tf_d2 = -omega² · x  -  gamma · z2  -  weighted_kernels
       L_tf = mean((tf_d2 - d2x)²) / var(d2x)

3. **Open-drive autonomous rollout** (RK4 with soft-tanh saturation at BX=0.5,
   BXP=1.0 — same kernel as `rollout_refine`). Initial condition is the data IC
   except for examples where `ic_mask == True` (ONSET category), which get
   `(x0, xp0) ~ N(0, ic_noise_rms²)`. State (x, x') updates autonomously; drives
   come from the encoded target, held open-loop for the whole H-sample horizon.

4. **Multi-resolution STFT magnitude loss** (DDSP):

       L_spec = mrstft_loss(x_rollout, x[:, :H, 0], configs)

   over `(n_fft, hop)` configs (default `(256,64);(512,128);(1024,256)`).
   Per config: spectral convergence + log-magnitude L1, summed and averaged.

5. **Optional envelope L1** (off by default; turn on with `--lam-env > 0` if the
   rollout amplitude drifts):

       L_env = env_loss(x_rollout, target, dt, env_ms)

6. **Total**: `lam_spec * L_spec + lam_tf * L_tf + lam_env * L_env`.

Curriculum: `H` ramps from `H_min` to `H_max` over training (geometric by
default). Short horizons early give a learnable signal at random init; the
soft-tanh tames blowups while the model is still figuring out the dynamics.

NaN total → skip optimizer step (counter logged to TB as `Loss/nan_skip`).
Grad-clip at `--grad-clip` (default 5.0).

## Edge-biased segment sampler

`data.load_data.get_audio_training_edge_weighted` builds three per-(wav, txt)
pools and samples to `max_segs` with the requested ratio:

- **ONSET** (cat=0): window starts at `on - silence_prefix_ms`, length `context_len`.
  The first `silence_prefix_ms` of audio is pre-onset (silence-near-silence).
- **OFFSET** (cat=1): window ends at `off + silence_suffix_ms`. Last
  `silence_suffix_ms` of audio is post-offset.
- **MID** (cat=2): non-overlapping windows lying strictly inside
  `[on + edge_ms, off - edge_ms]`.

Per-example category labels travel through the dataset to the training loop,
which uses them to set `ic_mask = (cats == ONSET)` for the cold-start IC.

## Validation / selection

`model_seed_cv_spectral` trains `n_seeds` seeds at fixed `--lam`, optionally
culls (`--cull-frac > 0`: train all to `cull_frac * n_epochs`, rank by val
cold-start autonomy, finish only top `--cull-keep`), and selects the best
seed by val `autonomy_score(..., rescale=False, cold_start=True)`.

The held-out vocs are file-level holdouts inside `--data-dir`: shuffle the
list of WAV stems with `--seed`, take the last `--test-frac` for test, the
next `--val-frac` for val, the rest for training. Each held-out voc is loaded
as `silence_pad_samples` lead-in + full vocalization (so the integration IC
at sample 0 is near-silence — the situation finchsim's real-time synthesis
target faces).

Tie-breaker on val autonomy: higher `bounded_frac` (so a
collapsed-but-lucky rollout doesn't beat a truly bounded one at the same
score).

## Smoke-test recipe (verify the pipeline runs)

```bash
python -m examples.train_poly_spectral_blk445 \
    --data-dir ~/ouroboros_data/blk445_syllC/day85 \
    --out-dir /tmp/spec_smoke \
    --val-frac 0.1 --test-frac 0.1 --n-val-vocs 2 --n-test-vocs 2 \
    --context-len 0.05 --max-segs 64 --batch-size 8 --n-jobs 0 \
    --n-kernels 3 --n-layers 2 --d-state 2 --expand-factor 2 \
    --n-epochs 2 --lr 1e-4 --n-seeds 2 \
    --cull-frac 0.5 --cull-keep 1 \
    --H-min 256 --H-max 256 --H-schedule const --save-freq 2
```

Should run in ~1 minute on a single GPU. Acceptance: `seed_cv.csv` + manifest
land in the out dir; no NaN crashes; `Loss/nan_skip` counter not heavily
incremented in TensorBoard.

For real training results (~6 h on one GPU), use the default `--n-epochs 50`,
`--n-seeds 4`, `--cull-frac 0.4 --cull-keep 2`, full model size.

## Acceptance bar (full run)

On the validation set:

- `bounded_frac >= 0.9` by epoch 10
- `spec_corr >= 0.5` by end of training
- `coldstart_raw_test_autonomy > 0` (above the divergence baseline of -5.0)

Listen to `selected_autonomous_recon.wav`: should be syllable-like with a
recognizable onset (no DC click, no immediate collapse to silence).

## Known follow-ups (out of scope for this PR)

- Multi-day train/val/test holdout (combine days 84/85/86).
- Causal / streaming Mamba in `get_funcs` (current bidirectional encoder lets
  drives "see the future" — fine for research demonstration, but a train /
  inference gap for real-time synthesis).
- Per-bird or per-syllable autosegmenter so the recipe can run on bird names /
  syllable names other than blk445 / syllable_C without code edits.

## spec_warmup_epochs

`--spec-warmup-epochs N` linearly ramps `lam_spec` from 0 → its target over the
first N epochs (default 5). Needed at random init because the rollout against a
quiet or onset target produces enormous MRSTFT values; without warmup the
spectral term swamps the TF anchor and the model fails to enter a learnable
basin. Set to 0 to disable when resuming a partly-trained checkpoint that is
already past the warmup.
