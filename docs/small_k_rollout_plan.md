# Plan: small-k rollout in training + cold-start raw-autonomy seed selection

**Status:** Draft handoff. The current ouroboros pipeline trains with a single-step
ẍ-prediction loss and selects seeds by **rescaled** (amplitude-gauge-removed) autonomy.
That combination has been shown — by experiment in this branch (see
`docs/autonomous_amplitude.md`) and by direct measurement in finchsim — to leave
autonomous amplitude marginal in a way that breaks the deployed downstream synthesis.
This plan proposes the two complementary changes that line up with the unresolved
failure mode, and a concrete experimental matrix.

The complementary work in `finchsim` is described at the end (Appendix B): the synthesis
side is already implemented and validated. What is missing is a model whose drives carry
amplitude information from a cold start; this plan is how to get one.

---

## 1. Why this exists

The deployed pipeline is `finchsim`'s **brainstem → adapter → ouroboros control signals
→ synthesis ODE → audio**. The synthesis ODE is the full driven polynomial form

```
dy/ds  = v
dv/ds  = -ω(t)² y - γ(t) v - Σ_{p,k} W[p,k] y^p v^k       (W[1,0]=W[0,1]=0; W[0,0] kept)
```

stepped with fixed-step RK4 in rescaled time `s = t/τ`, **starting from rest** at
`y(0) ≈ 0`. The drives ω, γ, W come from a calibrated linear adapter mapping brainstem
LP-filtered spike rates to ouroboros control channels.

The problem (proven empirically in May 2026 across the existing 8-seed cull at λ=1.068):
**the raw autonomous amplitude of the trained model is wildly voc-dependent from a cold
start.** Concretely, with the seed currently in deployment (`poly_seedcull/poly_lam1.068_seed6`):

- Cold-start raw rescale factor `r(v) = std(correct(target)) / std(correct(raw_auto))`
  has **range 633× across 40 vocs**, median 32, CV 151%
  (`examples/scan_seed_amp_coldstart.py`).
- Other seeds in the pool either blow up cold-start (Λ>0, seeds 2/3/4) or are similarly
  wild (seeds 0/1/5/6) — except seed 7 at CV 53% / max-min 48×, still not deployable.
- With the seed's true drives fed in, the finchsim integration test produces
  `std=0.0000` (silence): the deterministic poly ODE decays from rest, and the
  raw amplitude is too voc-variable for a single shipped rescale constant to fix.

The Floquet diagnostic in `docs/autonomous_amplitude.md` already explains why: the
trained model is **near-neutral** (Λ/cycle ≈ 0), so amplitude is set by the IC + the
drive trajectory **per vocalization**, not by a contracting attractor — and the current
training objective doesn't pressure the drives to encode amplitude (one-step ẍ matching
on the limit cycle is invariant to it).

## 2. What's been tried, and what hasn't

See `docs/autonomous_amplitude.md` for the full table. Briefly, what failed and what
the failure means for the design of the next attempt:

| Approach | Horizon | Outcome | Why it failed |
|---|---|---|---|
| Pointwise rollout MSE (post-hoc FT) | 80–400 samples | Autonomy worse; amp_pen 0.47→1.59 | One carrier cycle ≈ 16 samples; phase drift past that makes pointwise MSE reward amplitude collapse. |
| Spectral + envelope rollout FT (`rollout_refine`) | 768–1500 | Seed-variable; mean down, var up | Phase-invariant but BPTT through unstable rollout, gradient pathology. |
| Λ-penalty (`floq`) | – | Decays to wrong attractor | Stability **type** ≠ attractor **location**. |
| Short noise FT | 12–24 | Λ→+30 blow-up | BPTT through expanding rollout. |
| Selection by **rescaled** autonomy | – | Picks Λ≈0 seed → cold-start unstable | Removes amplitude from the criterion; lets drives stay amp-uninformative. |

**Not yet tried and theoretically well-placed:**
1. **k-step rollout *in training* (not as FT)** with **very small k**, k ≪ one carrier
   cycle, so pointwise (y, dy) MSE is a valid signal **and** the model never sees the
   long-horizon BPTT pathology. The primitive (`utils.euler_step_k`) is already in the
   repo but **not wired into the training loss** — it's a relic of the predecessor
   `mdmarti/ouro_clean`.
2. **Selection on cold-start raw autonomy** (no rescale, IC at silence start), which
   makes the selection criterion actually test what the deployed pipeline needs:
   drives carry amplitude from t=0.

## 3. Proposal

### 3.1 Training-side change: small-k rollout consistency in the loss

Wire `utils.euler_step_k(y, dy, d2y, dt, k)` into the train loop as an additional
loss term, summed with the existing one-step ẍ-prediction MSE:

```
L = L_one_step + λ_k · L_k_step
L_k_step = MSE(y, ŷ_k) + MSE(dy, d̂y_k)        # pointwise over all (B, L−k, ·)
```

`euler_step_k` returns `((y_out, yhat_out), (dy_out, dyhat_out))`, stacking the k
intermediate predictions on the last dim, so summing over k inside the MSE is "all
intermediate steps" — the default and the desired pressure (consistency at *every*
step from 1..k, not just the kth).

**Phase drift consideration.** At sr=40 kHz and a typical syllable carrier near
~2.5 kHz, one cycle is ~16 samples. Choose k ≪ 16. The matrix below tests k ∈ {2, 4, 8}.

**Curriculum?** Probably no curriculum is needed at these horizons; the rollout is
short enough that gradients are well-conditioned from epoch 1. Start with **fixed k**;
if loss balance turns out fragile, fall back to a short curriculum (k=2 for first ~10
epochs, then k=target).

**Loss weight.** Suggest sweeping λ_k ∈ {0.1, 0.3, 1.0}. The one-step loss must remain
the dominant signal; the k-step term is a consistency *pressure*, not a replacement.

### 3.2 Selection-side change: cold-start raw autonomy

Extend `train.eval.autonomy_score` (or write a sibling) to score from **cold start**:

- IC at the start of the silence lead-in: `y(0) = audio[0]` (≈ 0), `v(0) = (τ/dt)·dxdt[0]`.
  This is the convention used by `train.eval.integrate_poly_autonomous` already.
- **Window = silence_pad + vocalization** (the `make_paired_data_v2.py` convention,
  with SILENCE_PAD=2000 for 50 ms lead-in at 40 kHz).
- **No rescaling** (`rescale=False`), so amplitude is part of the score.
- Score = `spec_corr − w_amp·|log(std_auto/std_tgt)| − w_pitch·|log(pitch_auto/pitch_tgt)|`,
  diverge → `diverge_score`.

This is exactly the form of `autonomy_score(..., rescale=False)`; the change is
**windowing** (full window incl. lead-in, not mid-voc), and using this metric in
`train.model_cv.model_cv_lambdas(selection="autonomy")` as the cull criterion.

The diagnostic at `examples/scan_seed_amp_coldstart.py` already implements the
windowing and IC; the metric just needs to be folded into the pipeline alongside it.

### 3.3 Seed-pool change

Train **≥16 fresh seeds at λ=1.068** (the existing seedcull's chosen λ; this is
known irrelevant for autonomy, so don't bother sweeping). Compute cold-start raw
autonomy at epoch 20 and at epoch 50.

Per `examples/seed_cull_test.py`, **rankings stabilize by epoch 20** for the
rescaled metric (Spearman ρ ≈ 0.86). Re-run that test on the **cold-start raw**
metric to validate the same early-cull schedule (it may or may not be as stable);
default to: train all to ep 20, cull to top-3, finish those to ep 50.

## 4. Concrete steps

### Step 1 — Wire `euler_step_k` into the train loop

**Files:**
- `utils.py:45` — `euler_step_k(y, dy, d2y, dt, k=1)`, already returns the (y, dy)
  ground truth + predictions for steps 1..k.
- `train/train.py` — current loss is the one-step ẍ MSE inside the train step. Add
  the k-step term as a sibling. The model already outputs `yhat` (= τ²·d²y/dt²), so
  compute `d2y_phys = yhat / τ²` and pass to `euler_step_k`.

Add CLI:
- `--k-rollout` (int, default 0 = off; ≥2 enables)
- `--lambda-k` (float, default 0.3)

Don't change the default behavior — keep one-step-only as the default so the existing
pipeline is undisturbed.

### Step 2 — Add cold-start raw autonomy as a selection metric

**Files:**
- `train/eval.py` — `autonomy_score` already takes `rescale: bool`. Add a
  parameter `cold_start: bool = False` that, when True, expects segments to **include
  the silence lead-in** and does NOT trim — uses the supplied window as-is. (The
  current convention is mid-voc start +50 ms.)
- `train/model_cv.py` — `model_cv_lambdas(..., rescale_autonomy=False,
  cold_start_autonomy=True, val_vocs=..., ...)`. Pass-through.

Score signature stays the same: `(score, per_seg_scores, breakdown)`.

### Step 3 — Multi-seed training run

**Files:**
- `examples/run_lambda_pipeline.py` — already supports
  `--n-seeds`, `--lam`, `--keep-const`, `--drive-lowpass-ms`. Add `--k-rollout
  --lambda-k --cold-start-selection`. With `--lam 1.068 --n-seeds 16`, this trains
  16 seeds and selects by the chosen metric.
- Train to epoch 20 first (`--n-epochs 20`); save checkpoints at 10 and 20.
- Score all 16 at epoch 20 on cold-start raw autonomy over the val shard
  (data500/gabo_p8). Cull to top-3.
- Resume the top-3 to epoch 50 (`--n-epochs 50`, the run is resumable per λ-seed
  per `train.model_cv`).

### Step 4 — Final evaluation

For each of the top-3 finished seeds:
- Held-out cold-start raw autonomy on data500/gabo_p9 (per-voc spread; want CV<30%).
- Per-voc rescale factor distribution (`examples/scan_seed_amp_coldstart.py`).
- spec_corr, pitch_pen, bounded_frac.

Pick the **winner** as the seed with the best joint criterion. Ship it.

### Step 5 — Validate in finchsim

(See Appendix B for current finchsim state.)
- Re-export paired data with the winning seed
  (`~/ouroboros_smoke/make_calib_data.py --ckpt <new seed>`).
- Re-run the adapter pipeline:
  `scripts/gen_adapter_features.py` → `scripts/fit_adapter.py` (the per-pool muscle
  filter at 3 ms/40 ms is already wired up).
- Re-run `scripts/simulate_hvc_to_audio.py`. **Success criterion: generated audio is
  not silent and has energy in the 1–8 kHz band** (see `outputs/hvc_to_audio_comparison.png`).
- If yes: ship the single shipped rescale constant (median of the winning seed's
  per-voc factors) with the model. No runtime AGC, no gamma-centering.
- If no: see §6.

## 5. Experimental matrix

Each row = one full pipeline (train 16 seeds × cull → 3 → finish → score).
Total ~16 trainings × 4 rows = 64 model fits.

| run | k_rollout | λ_k | selection | rationale |
|---|---|---|---|---|
| baseline | off | – | cold-start raw | isolate selection-only contribution |
| k2 | 2 | 0.3 | cold-start raw | minimal multi-step pressure |
| k4 | 4 | 0.3 | cold-start raw | quarter-cycle horizon |
| k8 | 8 | 0.3 | cold-start raw | half-cycle horizon; near phase-drift onset |

If `baseline` alone clears success criteria, the selection change was enough. If `k8`
clears it but smaller k don't, that's evidence the multi-step pressure was needed.
Sweep λ_k only if a k value looks promising but is dominated by the one-step loss.

**Compute estimate.** At λ=1.068, d_state=4, 50 vocs, 1 ms low-pass, 50 epochs:
~15–30 min per seed depending on GPU contention (per `poly-lowpass-autonomous`
memory). 64 seeds × 25 min = ~26 GPU-hours. With the ep20 → ep50 cull, halve to
~13 hours. Parallelize across rows if multiple GPUs are available.

## 6. Success criteria, failure plan

**Per seed (held-out, cold-start, 10+ vocs from `data500/gabo_p9` with 50 ms lead-in):**
- bounded_frac = 1.0 (no NaN/Inf during integration)
- spec_corr ≥ 0.55 (the current pool's best cold-start spec_corr is seed 0 at 0.736;
  0.55 is a modest bar)
- pitch_pen ≤ 0.10 (≈ 10% pitch error)
- **per-voc rescale-factor CV ≤ 30%** (this is the binding constraint;
  the current best — seed 7 — is at 53%)
- amp_pen ≤ 0.5 (`|log(std_auto/std_tgt)|` ≤ 0.5 ⇔ within 1.65× of correct
  amplitude with a single shipped constant)

**Deployment criterion (finchsim integration test):**
- `simulate_hvc_to_audio.py` produces generated audio with `std > 1e-4` and
  spectral energy concentrated in 1–8 kHz (per the existing comparison plot).

**Failure plan.** If no seed across all 4 experimental rows meets the per-voc CV
criterion:
1. This is empirical evidence the marginal-amplitude problem is **structural** beyond
   selection + small-k pressure. Document the result in
   `docs/autonomous_amplitude.md`.
2. Fall back to runtime AGC in `finchsim/ouroboros_ode.py` (see §A.3 in
   Appendix B), shipping seed 7 (best amp stability we have).

## 7. Pitfalls to avoid (from prior attempts)

- **Don't** use horizons that span or exceed one carrier cycle (~16 samples at this
  sr). Phase drift past that makes pointwise MSE pathological — it rewards amplitude
  collapse to minimize the phase-drifted squared error.
- **Don't** use rollout as a post-hoc FT on a converged TF model with these dynamics.
  The model is near-marginal; BPTT through an expanding rollout has exploding
  gradients, and large injected noise overrides to over-contraction. Rollout pressure
  belongs *during* TF training, with short k.
- **Don't** pursue Λ-penalty fixes — they change stability *type*, not attractor
  *location* (see "Both Λ-targeting fixes FAIL" in `docs/autonomous_amplitude.md`).
- **Don't** select by rescaled autonomy if the deployment target is real-time
  cold-start synthesis. That's the wrong metric for the wrong endpoint.

## 8. File pointers

- **Primitive (existing):** `utils.py:45` — `euler_step_k`.
- **Train loop:** `train/train.py` — where the single-step ẍ MSE lives.
- **Score:** `train/eval.py` — `autonomy_score`, `integrate_poly_autonomous`,
  `generate_autonomous`.
- **Selection pipeline:** `train/model_cv.py` — `model_cv_lambdas` (multi-seed,
  selection metric pass-through).
- **Entrypoint:** `examples/run_lambda_pipeline.py`.
- **Seed cull primitive:** `examples/seed_cull_test.py` (currently uses rescaled
  metric; reuse the early-checkpoint timing analysis with the cold-start metric).
- **Diagnostics (already in repo as of this draft):**
  - `examples/scan_seed_amp_stability.py` — per-voc rescale CV, mid-voc.
  - `examples/scan_seed_amp_coldstart.py` — per-voc rescale CV, cold start.
- **Background docs:**
  - `docs/autonomous_amplitude.md` — the full free-amplitude analysis + Floquet diagnostic.
  - `docs/ode_solver_advice.md` (in finchsim) — DC drift / streaming integrator design.
  - `docs/RA_to_ouroboros_report.md` (in finchsim) — biology + adapter design.
- **Project memory:**
  `~/.claude/projects/-home-pearson-code-ouroboros/memory/poly-lowpass-autonomous.md` —
  detailed history of the rescale + rollout fine-tune attempts in this branch.

## 9. Git hygiene

- Branch off the current ouroboros HEAD (currently `arneodo-parameterization` or
  whatever's current — check `git log -1`).
- Don't merge into main until §5 integration test passes.
- Commit messages should end with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.

---

## Appendix A: known good values (for sanity-checking new seeds)

From the existing 8-seed pool at λ=1.068 (`poly_seedcull/`):

**Mid-voc rescaled autonomy** (current selection metric; held out test on gabo_p9):
- best-of-8: seed6 = +0.65 (the shipped seed). Selection record:
  `poly_seedcull/lambda_seed_cv.csv` and `selected_model.json`.

**Mid-voc per-voc rescale factor** (10 vocs, gabo_p9, start +50 ms):
- seed 7: **CV 18%**, max/min 1.6×, spec_corr 0.636, median 1.26
- seed 5: CV 23%, max/min 1.8×, spec_corr 0.631
- seed 6: CV 24%, max/min 2.0×, spec_corr 0.660 ← current shipped seed
- seed 3: CV 26%, max/min 2.2×, spec_corr 0.583

**Cold-start per-voc rescale factor** (10 vocs, 50 ms lead-in + voc):
- seed 7: CV 53%, max/min 48×, spec_corr 0.504 ← best of the bunch, still not deployable
- seed 0: CV 80%, max/min 9×, spec_corr 0.736 ← best cold-start spec
- seed 1: CV 78%, max/min 181×, spec_corr 0.438
- seed 6: CV 144%, max/min 431×, spec_corr 0.494 ← current shipped seed (fails)
- seeds 2/3/4: blow up (non-finite rollout)

The plan succeeds when a new seed's **cold-start** numbers look like
seed 7's mid-voc numbers: CV under 30%, max/min under 3×, spec_corr above 0.55.

## Appendix B: state of the finchsim side (as of this handoff)

The synthesis-side work is **done and validated**, on branch
`finchsim/ouroboros-poly-synthesis` (off updated `main`, base ~9118523):

1. `ouroboros_ode.py` rewritten — `build_synthesis_ode(mode="poly")` (default)
   integrates the **full polynomial autonomous ODE** using ω, γ, and all 256
   `poly_coeffs` from the adapter's `u` channels. RK4, drives held per tick.
   Mask: (1,0)/(0,1) zeroed, (0,0) kept (matches `keep_const=True`). Validation
   via `scripts/validate_poly_synthesis.py`: corr **0.999** vs ouroboros
   `integrate_poly_autonomous` on all 10 v2 paired-data renditions (pitch exact,
   amplitude matched after the same RMS rescale). Brian wiring smoke test:
   corr 0.996 vs the numpy loop at the expected 1-tick latch.
2. A gentle output one-pole DC blocker is **kept available** (`dc_block=True`,
   τ=30 ms) per the user; with a stable seed it's near-no-op.
3. **Per-pool muscle filter wired up** in both `adapter.py` and
   `scripts/gen_adapter_features.py`: `xii_vs/xii_ad: 3 ms` (Adam & Elemans
   syringeal twitch), `ram/pam: 40 ms` (respiratory). Map lives in
   `adapter.MUSCLE_TAU_MS_BY_POOL`, imported by the feature script so the
   filters can never drift apart.
4. `data/paired_data.npz` in finchsim is the **40-rendition drives-only**
   export (50 ms lead-in) generated by `~/ouroboros_smoke/make_calib_data.py`
   from the current shipped seed. **Re-run that script with the new seed's
   checkpoint** to refresh `data/paired_data.npz` before re-running the
   calibration.
5. The poly synthesis is signature-compatible with the existing caller
   `scripts/simulate_hvc_to_audio.py` (mode defaults to poly; no caller changes
   needed). Re-run the integration test after the new adapter weights are fit.

**Outstanding finchsim concern:** the per-pool filter helped γ (mid-voc held-out
R² 0.484 → 0.509, clears the README ≥0.5 bar) but didn't move ω (~0.38);
the poly held-out R² is ~0.06 regardless of data/filter — the 256 poly channels are
genuinely not linearly predictable from 4 brainstem pools. That's a limitation but
**not the audibility blocker**: LASSO falls back to each channel's mean, so the
nonlinear limit-cycle terms are still present at deployment. The audibility blocker
is solely the free-amplitude problem this plan addresses.

### A.3 If the plan fails — runtime fallback

If §5 fails (no seed reaches the cold-start criterion), add an AGC to
`finchsim/ouroboros_ode.py`:

```python
# inside the network_operation, after computing the next (y, v)
ms2[0] = (1 - α) * ms2[0] + α * y * y           # EMA of y², τ_agc ≈ 30–100 ms
gain   = target_rms / max(sqrt(ms2[0]), gate_floor)
G_ode.y[0] = y * gain
```

Knobs: `target_rms` (≈ 0.01, the v2 audio scale), `τ_agc` (30–100 ms),
`gate_floor` (noise gate so silence stays silent). This is the streaming analog of
`generate_autonomous(rescale=True)`. It's seed-agnostic — works with any seed that
integrates finite, including the current seed 6.
