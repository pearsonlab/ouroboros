"""Spectral-rollout training step for the polynomial Ouroboros.

Replaces the legacy MSE-on-d²y/ds² objective with a DDSP-style multi-resolution
STFT magnitude loss (arxiv:1910.11480) computed on a teacher-forced RK4 reconstruction
of the waveform:

  1. Encode the drives (ω(t), γ(t), w(t)) from the target audio via `model.get_funcs`
     (with optional low-pass filtering inside the model).
  2. Integrate the polynomial ODE forward in rescaled time s = t/τ with those drives
     held open-loop, but the state (x, x') autonomous. Initial condition is the data
     IC by default; for examples whose `ic_mask` is True, the IC is overridden with
     low-amplitude Gaussian noise (the cold-start training mode).
  3. Compute MRSTFT magnitude loss between the integrated waveform and the target.
     Optionally add a teacher-forced d²y/ds² MSE anchor (default on, weight 1.0) so
     early-epoch training has a smooth gradient signal before the spectral basin
     becomes informative, and an envelope L1 term (default off) for amplitude pinning.

The RK4 inner loop and soft-tanh saturation (BX=0.5, BXP=1.0) match train/rollout_refine.py,
which already validated this kernel for post-hoc fine-tuning. The novelty here is using
it as the PRIMARY objective from scratch, with per-example cold-start IC selection
driven by the edge-biased sampler's category labels.
"""

from typing import Optional

import torch

from train.rollout_refine import (
    DEFAULT_CONFIGS,
    BX,
    BXP,
    mrstft_loss,
    env_loss,
)

# ONSET category from data.load_data (kept local to avoid the import cycle the data
# module would introduce through utils).
ONSET = 0


def _filter_configs_for_horizon(configs, H):
    """Drop STFT (n_fft, hop) configs whose n_fft > H so torch.stft does not pad
    the rollout up to a longer effective window than we actually integrated."""
    keep = [(n_fft, hop) for (n_fft, hop) in configs if n_fft <= H]
    if not keep:
        # Smallest config still too big? Use a tiny one matched to H.
        n_fft = max(16, H // 2)
        keep = [(n_fft, max(1, n_fft // 4))]
    return tuple(keep)


def teacher_forced_rollout(
    model,
    x: torch.Tensor,        # (B, L, 1) target audio
    dxdt: torch.Tensor,     # (B, L, 1) numerical first derivative (per-sample)
    dt: float,
    *,
    H: int,                 # rollout horizon in samples
    ic_mask: Optional[torch.Tensor] = None,  # (B,) bool; True => silence-noise IC
    ic_noise_rms: float = 1e-3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:           # (B, H)
    """RK4 H-step rollout of the poly Ouroboros, drives encoded from target audio.

    Drives (ω, γ, kernel weights w) come from `model.get_funcs(x, dxdt.clone(), dt)`
    and are held OPEN-LOOP across the integration; the state (x, x') is updated
    autonomously. Same soft-tanh saturation as train/rollout_refine.py:136-155
    (BX=0.5 on x, BXP=1.0 on x') so a random-init model can't blow up.

    `ic_mask` is a (B,) bool tensor. Where True, the initial (x, x') is replaced with
    N(0, ic_noise_rms²) samples — this is the cold-start training mode for ONSET
    examples. Where False (or if ic_mask is None), the IC is (x[:, 0, 0], z2[:, 0, 0])
    as in rollout_refine.
    """
    B, L, _ = x.shape
    assert H <= L, f"rollout horizon H={H} must be <= segment length L={L}"

    # Encode drives in a single forward of the Mamba encoders.
    omega, gamma, _, weights, _ = model.get_funcs(x, dxdt.clone(), dt)  # ω,γ:(B,L,1); w:(B,L,P,P)
    z2 = (model.tau / dt) * dxdt  # rescaled-time velocity (B, L, 1)

    om = omega[:, :, 0]
    ga = gamma[:, :, 0]
    w = weights
    P = w.shape[-1]
    powers = torch.arange(P, device=x.device)

    # IC -- (B,) tensors of x and x' at s=0.
    xc = x[:, 0, 0].detach().clone()
    xp = z2[:, 0, 0].detach().clone()
    if ic_mask is not None and ic_mask.any():
        # NOTE: torch.randn supports a `generator=` arg even for CUDA when the
        # generator was created on the same device.
        noise_x = torch.randn(B, device=x.device, generator=rng) * ic_noise_rms
        noise_xp = torch.randn(B, device=x.device, generator=rng) * ic_noise_rms
        m = ic_mask.to(x.device).to(x.dtype)
        xc = xc * (1 - m) + noise_x * m
        xp = xp * (1 - m) + noise_xp * m

    def f(xx, vv, k):
        xpw = xx.unsqueeze(1) ** powers  # (B, P)
        xvw = vv.unsqueeze(1) ** powers  # (B, P)
        kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w[:, k])
        return vv, -(om[:, k] ** 2) * xx - ga[:, k] * vv - kern

    xs = [xc]
    for k in range(H - 1):
        k1x, k1v = f(xc, xp, k)
        k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v, k)
        k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v, k)
        k4x, k4v = f(xc + k3x, xp + k3v, k)
        xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        # soft-tanh saturation (>> data scale; only tames blow-up during training)
        xc = BX * torch.tanh(xc / BX)
        xp = BXP * torch.tanh(xp / BXP)
        xs.append(xc)
    return torch.stack(xs, dim=1)   # (B, H)


def spectral_rollout_step(
    model,
    x: torch.Tensor,        # (B, L, 1) target audio
    dxdt: torch.Tensor,     # (B, L, 1)
    d2x: torch.Tensor,      # (B, L, 1) rescaled-time target acceleration
    dt: float,
    *,
    H: int,
    configs=DEFAULT_CONFIGS,
    lam_spec: float = 1.0,
    lam_tf: float = 1.0,
    lam_env: float = 0.0,
    env_ms: float = 2.0,
    tf_var: Optional[float] = None,   # precomputed Var(d2x) over the dataset; matches rollout_refine.py:110
    ic_mask: Optional[torch.Tensor] = None,
    ic_noise_rms: float = 1e-3,
    rng: Optional[torch.Generator] = None,
) -> dict:
    """One forward + loss for the spectral-rollout objective.

    Returns a dict {'spec', 'tf', 'env', 'total'} of scalar tensors. The caller is
    responsible for backprop on 'total'. The TF anchor is the same variance-normalized
    d²y/ds² MSE used in train/rollout_refine.py:132 — keeps the model in a learnable
    basin in early epochs while the rollout horizon is short.

    `d2x` is expected to ALREADY be in rescaled-time units (i.e. multiplied by τ²/dt²
    by the caller), matching the convention used by train.train.train() for the legacy
    MSE-on-accel loss. The TF anchor compares it to the model-predicted rescaled
    acceleration tf_d2 = -ω²·x - γ·z2 - weighted_kernels.
    """
    # Encode drives once (model.get_funcs mutates dxdt in place; pass a clone).
    omega, gamma, wk, weights, _ = model.get_funcs(x, dxdt.clone(), dt)
    z2 = (model.tau / dt) * dxdt  # rescaled velocity
    tf_d2 = -(omega ** 2) * x - gamma * z2 - wk
    # Variance-normalized so the anchor scale is commensurable across vocs / runs.
    # tf_var should be the variance computed ONCE over the whole training set (see
    # rollout_refine.py:110) -- per-batch variance is unstable when the batch is mostly
    # silence (ONSET segments) and can drive the loss to explode. We clamp at a floor
    # to be extra safe.
    if tf_var is None:
        v = float(d2x.detach().var().clamp_min(1e-3).item())
    else:
        v = max(float(tf_var), 1e-6)
    L_tf = ((tf_d2 - d2x) ** 2).mean() / v

    # Rollout + spectral loss. The rollout calls model.get_funcs again with a fresh
    # clone (drives recomputed for the rollout-state forward). Could be unified with
    # the TF computation above, but keeping them separate makes the per-component
    # losses easy to reason about and matches rollout_refine's structure.
    xg = teacher_forced_rollout(
        model, x, dxdt, dt, H=H,
        ic_mask=ic_mask, ic_noise_rms=ic_noise_rms, rng=rng,
    )
    tgt = x[:, :H, 0]

    configs_H = _filter_configs_for_horizon(configs, H)
    L_spec = mrstft_loss(xg, tgt, configs_H)
    if lam_env > 0:
        L_env = env_loss(xg, tgt, dt, env_ms)
    else:
        L_env = torch.zeros((), device=x.device, dtype=x.dtype)

    total = lam_spec * L_spec + lam_tf * L_tf + lam_env * L_env
    return {"spec": L_spec, "tf": L_tf, "env": L_env, "total": total}


def horizon_for_epoch(epoch: int, n_epochs: int, H_min: int, H_max: int,
                      schedule: str = "geom") -> int:
    """Curriculum on the rollout horizon: small H early (cheap, easy gradient), full H
    by the end. 'geom' = geometric spacing; 'linear' = linear; 'const' = H_max throughout.
    """
    if n_epochs <= 1 or schedule == "const":
        return int(H_max)
    t = epoch / max(1, n_epochs - 1)
    if schedule == "linear":
        H = H_min + t * (H_max - H_min)
    elif schedule == "geom":
        import math
        H = H_min * (H_max / H_min) ** t
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return int(round(H))
