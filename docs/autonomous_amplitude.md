# Autonomous reconstruction and the free-amplitude problem

This note documents why the polynomial `Ouroboros` reconstructs vocalizations near-perfectly under
teacher forcing but produces **poorly-constrained amplitude** when run fully autonomously, what fixes
were tried, and the recipe we landed on.

## Setup

The model parameterizes the second derivative of audio as a driven nonlinear oscillator,

```
ẍ = -ω(t)² x - γ(t) ẋ - Σ_{ij} w_{ij}(t) xⁱ ẋʲ
```

with the drives ω(t), γ(t), w(t) produced by Mamba encoders and low-passed in the loop
(`drive_lowpass_ms`). Training is **teacher-forced**: one-step prediction of ẍ at the true states.
For deployment we also want **autonomous** generation — integrate the ODE feeding the *generated*
state back, with only the drives (precomputed from data) and the initial condition supplied externally
(`train.eval.integrate_poly_autonomous`).

On Mindlin/gabo synthetic data, teacher-forced fit is strong (test R² ≈ 0.73 with the 1 ms drive
low-pass; ≈ 0.99 without it), and the autonomous **pitch and spectral shape** track the target well.
The **amplitude**, however, is unreliable: the free run decays, grows, or self-sustains depending on
the seed, with the overall loudness off by factors of ~2–5.

## Diagnosis: amplitude is a marginal, untrained direction

The training data lies on a 1-D closed orbit (the limit cycle) in the 2-D (x, ẋ) phase plane.
Pointwise ẍ-matching pins the vector field's **tangential** (along-orbit: phase/frequency) component
but says almost nothing about its **transverse** (radial/amplitude) component — the data never leaves
the orbit. Amplitude stability is entirely a transverse property.

Linearizing around the periodic orbit gives two Floquet multipliers, and they are exactly our two
failure modes:

- One multiplier is structurally **= 1**, eigenvector along the flow → phase is marginal (this is the
  phase drift that breaks pointwise rollout losses).
- The other governs amplitude: `exp(Λ)` with the area-contraction exponent
  **`Λ = ∮ ∂f/∂ẋ dt`** (the net per-cycle effective damping). `Λ<0` attracting, `Λ=0` neutral
  (conservative / SHO-like, amplitude free), `Λ>0` repelling.

On-orbit fitting leaves `Λ` essentially free. Confirmed empirically with `examples/floquet_amplitude_diagnostic.py`:
across seeds, `Λ` computed along the data orbit predicts the actual rollout decay/growth at **r ≈ 1.0**,
`|Λ/cycle| ≲ 0.02` (near-neutral — *not* an attractor; a true limit cycle would be 1–2 orders of
magnitude more contracting), and `Λ` **scatters around zero with both signs across seeds**. That scatter
*is* the seed-dominated amplitude. In short: **the quantity that sets autonomous amplitude is not in the
training signal**, so it is fixed only by random initialization.

## What we tried

| Approach | What it constrains | Result |
|---|---|---|
| Constant (0,0) "alpha" forcing term | extra DC drive | R²-neutral; **no** robust autonomy gain (single-seed "win" was a low-pass confound; 5-seed median Δ≈0). Kept on anyway (may help other data). |
| Amplitude **rescale** (post-hoc) | output scale (gauge) | **+0.60**, reliable. Needs a reference loudness at inference. |
| **Envelope**-matching rollout refine | output loudness directly | helped 3/5 seeds, hurt 1/5; net slightly negative, seed-variable. |
| Spectral (multi-res STFT) rollout refine | spectral shape | improves shape but not amplitude; seed-fragile. |
| Pointwise rollout MSE (long horizon) | trajectory, pointwise | fails — phase drift makes MSE reward amplitude collapse. |
| **Λ penalty** (`floq`: make the orbit attracting) | divergence sign | converts grow→**decay**; sets stability *type*, not attractor *location*. |
| **Noise** fine-tune (denoise back to orbit) | local contraction | **collapse** (aggressive, Λ→−30) or **blow-up** (gentle, Λ→+30); BPTT through the marginal/expanding rollout is ill-conditioned. |

Two structural lessons emerged:

1. **Divergence sign ≠ amplitude.** Driving `Λ<0` on a non-exact orbit contracts toward the model's
   *own* (smaller/zero) attractor — it does not place the attractor at the data radius. Amplitude is set
   by attractor *location*, a global limit-cycle property, not by the local stability type.
2. **Gradient-based autonomous-rollout fine-tuning is ill-conditioned here.** For most seeds the rollout
   is locally expanding (Λ>0), so BPTT through it has exploding gradients; large injected noise overrides
   this into over-damping (collapse), small noise lets it blow up. No stable middle was found across
   aggressiveness × horizon.

The only interventions that reliably move autonomous amplitude **constrain the output amplitude
directly** (rescale; envelope), not the vector field.

## Recipe (what's in the pipeline)

Because the seed dominates and amplitude is a free gauge:

1. **Select over seeds, not λ.** Fix λ, train several seeds, pick the best on a held-out validation
   shard by **rescaled** autonomy (spectral shape + pitch + boundedness; amplitude gauge removed).
2. **Rescale amplitude at generation.** Run the deterministic closed-loop rollout, then match its RMS
   to a reference loudness.

Both are folded into the pipeline:

- `train.eval.autonomy_score(..., rescale=True)` — selection metric with amplitude gauge-fixed.
- `train.eval.generate_autonomous(model, audio, dt, rescale=True, ref_rms=...)` — deployed generation.
- `train.model_cv.model_cv_lambdas(..., n_seeds=N, selection="autonomy", rescale_autonomy=True,
  keep_const=True, lambdas=[λ])` — multi-seed selection by rescaled validation autonomy.
- `examples/run_lambda_pipeline.py` — end-to-end: file-level holdout, seed selection, and a rescaled
  autonomous reconstruction (`selected_autonomous_recon.wav` + `selected_model.json`).
- **Seed culling** (`model_cv_lambdas(cull_frac=…, cull_keep=…)`, exposed as `--cull-frac/--cull-keep`):
  train all seeds to `cull_frac` of the budget, finish only the top `cull_keep` by rescaled validation
  autonomy — see below.

```
python -m examples.run_lambda_pipeline --data-glob 'data500/gabo_p*' --out-dir ./poly_pipeline \
    --n-epochs 50 --n-seeds 5 --lam 1.068 --drive-lowpass-ms 1.0 --d-state 4
# with seed culling — train 8 seeds, finish the best 2 after 40% of the budget:
python -m examples.run_lambda_pipeline --n-seeds 8 --cull-frac 0.4 --cull-keep 2
```

## Validation at scale, and seed culling

**Full grid run** (`examples/run_lambda_pipeline.py`, 7 λ × {3, 8} seeds × 50 ep, the 500-voc `data500`
set, 1 ms low-pass, `keep_const`) re-confirms the diagnosis on the full dataset with the production
*rescaled* metric. Rescaled validation autonomy is **flat across λ** (per-λ mean +0.548 ± 0.0008) and
**seed-structured** (seed spread 0.51 → 0.59); the across-seed std (≈ 0.04) is ~20× the across-λ
variation (≈ 0.002), and `amp_pen = 0.00` everywhere (rescaling working). The pipeline selects the best
seed and the deployed (rescaled) reconstruction reaches **test autonomy +0.63–0.65** on the held-out
shard (best-of-3 +0.626, best-of-8 +0.653), spectral corr ≈ 0.68, pitch within ~5%, fully bounded —
i.e. more seeds → a better best, as expected when the seed is the lever.

**Seed culling — can we pick the winner early?** Since training all seeds fully is the cost, we asked
whether an *early* checkpoint's rescaled validation autonomy predicts the final seed ranking
(`examples/seed_cull_test.py`, 8 seeds at λ = 1.068). Spearman vs the final (epoch-50) ranking:

| early epoch | Spearman ρ | top-1 hit | top-2 overlap |
|---|---|---|---|
| 10 (20% budget) | 0.48 | ✗ | 1/2 |
| 20 (40% budget) | 0.86 | ✓ | 2/2 |
| 30 | 0.76 | ✓ | 2/2 |
| 40 | 0.98 | ✓ | 2/2 |

**Epoch 10 is too early** — the epoch-10 leader finished 6th of 8, and the eventual winner was only 2nd
at epoch 10. **By epoch 20 the top-1 and top-2 seeds are already correct** (the mid-pack keeps
reshuffling, so the *full* ranking only settles by ~epoch 40). Practical schedule: **train all seeds to
~40% of the budget, keep the top 1–2 by rescaled validation autonomy, and finish only those** — roughly
halving the seed-search cost. This is built into the pipeline (`--cull-frac 0.4 --cull-keep 2`, i.e.
`model_cv_lambdas(cull_frac=, cull_keep=)`). (Caveat: one dataset, one λ, 8 seeds; the 40%-budget
threshold is config-specific. Note teacher-forced R² is useless for this — it is seed-invariant; only the autonomous
rollout discriminates, and there is no cheaper on-orbit surrogate, since amplitude/shape live off-orbit.)

## Open direction

A genuine fix would place an **attracting limit cycle at the data amplitude** — i.e. make the data
orbit a near-exact periodic solution *and* contracting. That is a global property that neither a cheap
divergence penalty nor a (fragile) rollout fine-tune achieves. A phase-robust objective that constrains
the *time-varying loudness envelope* without differentiating through an unstable rollout (e.g. matching
envelopes in a way that is decoupled from the carrier phase) is the most promising untried lever; the
envelope-refine experiments are a first, seed-variable step in that direction.

## Reproduce the diagnostics

- `examples/floquet_amplitude_diagnostic.py` — Λ along the data orbit vs. actual rollout decay/growth.
- `examples/eval_autonomy_rescale.py` — raw vs. rescale-bound autonomy + envelope-match metrics.
- `examples/finetune_attractor_poly.py` — the `floq` (Λ-penalty) and `noise` fine-tunes (negative results).
- `examples/finetune_rollout_spectral_poly.py` / `train/rollout_refine.py` — spectral+envelope refine.
