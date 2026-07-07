# Noise-floor fit for the harmonic-plus-noise-plus-rumble model

How the additive **noise** branch is fit to the recording noise **floor**, so the
oscillator can later be trained on a clean residual instead of collapsing.

## Why

On **raw** (un-denoised) org545 the target spectrum is dominated by a broadband
recording-noise floor plus a low-frequency lobe. If the oscillator, the additive
noise branch, and the rumble branch all train jointly under the multi-resolution
STFT-magnitude loss, the flexible noise+rumble branches soak up the floor **and** the
buried vocal energy, leaving no meaningful residual — the oscillator collapses
(γ → positive, source RMS → 0). See the run history: `..._melspec_oscwarmup5` reached
autonomy +0.94 with the oscillator contributing *nothing*.

The fix is sequential source separation:

1. **Warm-up (this doc):** freeze the oscillator, fit **noise + rumble** to the floor / LF.
2. **Residual training (later):** freeze noise + rumble, subtract them, train the
   oscillator on the residual `|STFT(raw − rumble)|² − E|STFT(noise)|²`.

For step 2 to work, the noise in step 1 must land on the **true floor** — not the
loudness-weighted average (which over-subtracts and erases the vocalization).

## The branches and the single crossover

- **rumble**: a Mamba head, **low-passed** below `--rumble-lowpass-hz` — owns the LF.
- **noise**: `sigma · noise_tract(white)`, **high-passed** above the *same* cutoff
  (`high-pass = x − lowpass(x)`), so rumble and noise are complementary and **one**
  parameter (`--rumble-lowpass-hz`) sets the whole LF/noise split. The floor-fit band
  boundary auto-derives to `1.5 ×` that cutoff (the rumble's effective upper edge).
- **sigma** (`--sigma-constant`): **one number per vocalization** (the sigma Mamba head
  mean-pools over time). It must model a *stationary floor level*, not track syllables.
  With the noise high-passed, `sigma` only ever scales the **broadband** floor, so the LF
  fit can't prop it up.

## The floor target (`--noise-floor-fit`)

For broadband bins (≥ the derived cutoff) the spectral target is replaced, **per sample**,
by an estimate of that vocalization's own noise floor. LF bins keep the real
time-resolved target so the rumble fits the LF signal.

Design choices, in order of how we arrived at them:

1. **Per sample, not batch-pooled.** Recording floors vary across vocalizations
   (~30× observed). A single batch-pooled floor is right for some vocs and
   off-by-a-constant for others — a *target* error, visible as one PSD panel dead-on and
   another shifted by a fixed dB. So each voc targets its **own** floor.

2. **Quiet-frame spectrum, not a per-frequency percentile.** A per-frequency percentile
   over time stitches a *different* time frame into each bin — incoherent, and it sits
   4–6 dB below the mean (each bin is a single exponential-ish variable, so its low
   percentile is far below its mean). Instead we pick the **quiet time frames** — those
   whose **broadband** power (not *total* power, which the LF/rumble would contaminate —
   this was the syllD 2–5 kHz contamination bug) falls in the `[0.4·p, p]` percentile band
   (`p = --floor-pctile`, default 25 → the 10–25th percentile frames) — and **average
   their spectra**. That is a real, coherent quiet-moment spectrum; the shared
   `noise_tract` learns the spectral *shape*.

3. **Bias correction (`--floor-correction`).** Selecting low-power frames biases the
   estimate downward — the classic minimum-statistics bias, exactly analogous to
   MAD → σ needing 1.4826. See the derivation below. Default 1.2.

## Deriving the correction factor (principled, not tuned)

Model the broadband power of a frame as `S ~ Gamma(shape = K, scale = θ)`. The scale θ is
the absolute noise level and **cancels** (the correction is a ratio), so only the **shape
K** matters. `K` is the effective degrees of freedom,

```
K = mean(S)² / var(S)  =  (Σ_f μ_f)² / (Σ_f μ_f²)      # inverse participation ratio
```

where `μ_f` is the noise PSD (flat noise over N bins → K = N; colored/peaked → smaller K).

We select frames with `S` in the `[p_lo, p_hi]` quantile band and average, giving
`E[S | S ∈ band]`, which underestimates `E[S]`. The correction is
`C = E[S] / E[S | S ∈ band]`. Using the identity `g · Gamma(K,1).pdf(g) = K · Gamma(K+1,1).pdf(g)`
the conditional mean collapses to a closed form:

```
C(K) = (p_hi − p_lo) / [ F_{K+1}(b) − F_{K+1}(a) ],   a = F_K⁻¹(p_lo),  b = F_K⁻¹(p_hi)
```

with `F_K` the `Gamma(K,1)` CDF (regularised lower incomplete gamma). Pure incomplete-gamma,
no simulation.

**Why total-power selection de-biases the per-bin floor:** we select on the *total*
broadband power but want the *per-bin* floor spectrum de-biased. For symmetric bins,
conditioning on low total power pulls every bin down by the same factor, so
`E[P_f | S ∈ band] / E[P_f] = E[S | S ∈ band] / E[S]`. So the total-power Gamma correction
is exactly the per-bin correction.

**Validation** (analytic `C(K_emp)` vs. simulated bias, `[10,25]` band, `K_emp = mean²/var`):

| noise | K_emp | C analytic | bias measured |
|-------|-------|-----------|---------------|
| white | 79.8  | 1.119     | 1.109         |
| pink  | 20.2  | 1.266     | 1.251         |

Matches to <1%. Our colored noise sits between, so `C ≈ 1.12–1.27`; the fixed default
**1.2 ≈ C(K≈30)**. The self-calibrating option is to compute `K_emp` from the quiet-frame
broadband power each batch and plug into `C(K)` — no magic constant. (Not yet wired in;
the fixed 1.2 only sets the absolute level by ±~0.8 dB.)

## Flags

```
--sigma-constant                 # one noise-gate number per vocalization
--use-noise-branch --use-rumble-branch
--rumble-lowpass-hz 250          # THE single LF/noise crossover (rumble LP, noise HP, floor boundary=1.5x)
--noise-floor-fit                # broadband target = per-sample quiet-frame floor
--floor-pctile 25                # quiet-frame band upper edge (frames in [0.4p, p] percentile)
--floor-correction 1.2           # minimum-statistics bias correction; C(K), see above
# oscillator frozen during the warm-up:
--lam-tf 0 --freeze-drives-epochs N --freeze-tract-epochs N --freeze-envelope-epochs N
```

Implementation: `mrstft_loss(..., floor_fit=...)` in `train/rollout_refine.py`;
noise high-pass in `filtered_noise_branch` (`train/spectral_rollout.py`); constant sigma in
`Ouroboros.get_sigma` (`model/model.py`).

## Known open issue

The constant-sigma head **mean-pools over time**, so it reads loudness-dominated features:
a voc with loud syllables but a low floor (e.g. syllD) makes the head predict a *high*
sigma even though its floor is low. Result: noise-dominated vocs land on their floor
immediately, but clean-but-loud vocs are slow to (or don't) diverge downward. Candidate
fix: pool the sigma head over a **low percentile** of its time-encoding rather than the
mean, so it estimates the floor from the quiet part of its own input.
