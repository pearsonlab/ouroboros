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
- **sigma** (`--sigma-constant`): **one scalar gain per vocalization** on the filtered
  noise. The sigma Mamba head reads the whole waveform (+ its reversal), mean-pools to a
  single number, and applies **softplus** (so it can approach 0 smoothly for clean vocs
  without a dead-relu collapse). With the noise high-passed, this gain only ever scales the
  **broadband** floor, so the LF fit can't prop it up.

## The floor target (`--noise-floor-fit`) — MASKED regression

For broadband bins (≥ the derived cutoff) the spectral loss is **masked to that
vocalization's QUIET time frames**, and there `sigma · noise_tract` is regressed onto the
real spectra of those frames (their floor). LF bins keep the loss at all frames so the
rumble fits the LF signal. The scalar gain is learned directly by this masked loss — no
scalar reduction sits in the gradient path.

Design choices, in the order we arrived at them (each fixed a failure of the last):

1. **Per sample, not batch-pooled.** Recording floors vary per vocalization; a single
   batch-pooled floor is right for some vocs and off-by-a-constant for others (visible as
   one PSD panel dead-on, another shifted by a fixed dB). Each voc targets its **own** floor.

2. **Select QUIET FRAMES by BROADBAND power, not per-frequency percentiles, not total
   power.** A per-frequency percentile stitches a different time frame into each bin
   (incoherent, and 4–6 dB below the mean). Total-power selection lets the LF/rumble pick
   the frames (the syllD 2–5 kHz contamination bug). So: select frames whose **broadband**
   power (≥ cutoff) is in the `[0.4·p, p]` percentile band and use *those* frames.

3. **MASK the loss, don't build a target spectrum, and keep g(t) out of the gradient
   reduction.** Every attempt to reduce the gate to a scalar *inside the forward pass*
   failed: mean-pool → the shared gain collapsed to 0; hard percentile → oscillated
   (109→0.95→13.8); soft low-k → still fragile — because the reduction sat in the gradient
   path and its frame-selection moved with the gate's own values. The fix is to **mask the
   broadband loss to the (target-selected, stable) quiet frames** and let the head emit one
   scalar gain trained by that masked loss. Loud frames get no broadband gradient, so
   syllables can't drag the gain up. The shared `noise_tract` learns the (universal) shape.

4. **Bias correction (`--floor-correction`).** Selecting low-power frames biases the
   estimate downward — the classic minimum-statistics bias, analogous to MAD → σ needing
   1.4826. Closed form below. The `[5,15]` band (`--floor-pctile 15`, `--floor-correction
   1.3`) tracks the true floor better than `[10,25]` (measured 1.1–1.3× vs 1.2–1.9× bias
   across voc types), at the cost of a slightly noisier per-segment estimate.

The noise-floor **shape is universal** across voc types (normalized quiet-frame PSDs of
syllA–E overlap within 1.3–2.4 dB), so one shared `noise_tract` is correct — only the
per-voc **level** (the scalar gain) needs to vary.

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

## Long-window noise-fit STAGE (`--noise-fit-only`)

The floor lives in the quiet frames between syllables. A 46 ms rollout window (the
oscillator's H) **has no quiet frames for a dense vocalizer** — syllD's 46 ms quiet-frame
floor is **67×** its true full-voc floor — so the gain fit an elevated target and plateaued
(loudness-dominated at init, e.g. syllD's gain read *highest* despite the lowest floor).

The fix exploits that this stage doesn't need the oscillator: `--noise-fit-only` **skips the
drives / TF anchor / RK4 rollout** and runs only the feedforward rumble + filtered noise.
With no O(H) rollout, the window can be long (`--context-len 0.3`, H≈13000), so segments
contain quiet gaps and the per-segment quiet-frame floor **is** the true floor. Result: the
per-voc gain specializes to each voc's true floor (syllD/E from 125×/58× to ~2×, syllA/C on
their floor) instead of a uniform loudness-driven value. Memory scales with `batch ×
window`; use `--batch-size 8` at H≈13000 on an 11 GB card.

## Rumble = time-domain MSE (`--lam-rumble-td`), noise = magnitude only

The rumble is **deterministic**, so it *can* be phase-aligned to the actual LF waveform;
the noise (stochastic) and oscillator (magnitude loss) cannot. Add a time-domain MSE of the
rumble vs the raw LF (`lowpass(raw, rumble_lowpass_hz)`, DC-subtracted):
`lam_rumble_td · mean((rumble − raw_LF)²)`. **Rumble only.** Without it the magnitude loss
leaves the rumble phase random, so `raw − rumble` **adds** LF power (measured 4–5×) instead
of cancelling. With it, time-domain `raw − rumble` removes 65–99 % of the LF. Drop the MSE
once the rumble is frozen.

## Residual subtraction (for the oscillator stage)

Because everything here is magnitude-trained *except* the (MSE'd) rumble:

- **rumble** → subtract in the **time domain**: `x = raw − rumble` (phase-aligned).
- **noise** → subtract in the **power domain**: `residual = √max(|STFT(x)|² − E|STFT(noise)|², 0)`
  (stochastic, no phase). Estimate `E|STFT(noise)|²` by a few draws.

(Subtracting the magnitude-only rumble in the time domain *without* the MSE would add LF,
not remove it — the original plan's bug.)

## Flags

```
--sigma-constant                 # one softplus scalar gain per vocalization
--use-noise-branch --use-rumble-branch
--rumble-lowpass-hz 250          # THE single LF/noise crossover (rumble LP, noise HP, floor boundary=1.5x)
--noise-floor-fit                # mask broadband loss to per-sample quiet frames
--floor-pctile 15                # quiet-frame band = [0.4p, p] pctile -> [6,15]; beats [10,25]
--floor-correction 1.3           # minimum-statistics bias correction C(K); 1.3 for [5,15], 1.2 for [10,25]
--noise-fit-only                 # skip oscillator -> long window; run only rumble + noise
--lam-rumble-td 10               # time-domain rumble MSE (phase-align); rumble-only, drop when frozen
--context-len 0.3 --H-min 13000 --H-max 13000 --batch-size 8   # long window; oscillator's H is irrelevant here
--lam-tf 0 --freeze-drives-epochs N --freeze-tract-epochs N --freeze-envelope-epochs N
```

Implementation: masked floor loss + rumble MSE in `spectral_rollout_step` /
`mrstft_loss` (`train/rollout_refine.py`, `train/spectral_rollout.py`); noise high-pass in
`filtered_noise_branch`; scalar gain in `Ouroboros.get_sigma` (`model/model.py`).

## Self-calibrating correction (optional, not wired)

`C(K)` needs only `K` = effective DOF = `mean²/var` of the **pure-noise** broadband power
(≈30 for our colored noise; ≈80 white, ≈20 pink). Estimate it from the selected quiet
frames each batch and plug into the closed form — no magic constant. NB: estimate `K` on the
*quiet frames only*; over the full segment the syllables inflate the variance and `K`
collapses (→ absurd `C`).
