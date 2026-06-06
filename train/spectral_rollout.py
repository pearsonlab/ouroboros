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

import warnings
from typing import Optional

import torch

from train.rollout_refine import (
    DEFAULT_CONFIGS,
    BX,
    BXP,
    mrstft_loss,
    env_loss,
    env_loss_log,
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


# --- RK4 rollout backends ------------------------------------------------------------
# The autonomous RK4 integration is a sequential Python loop over the horizon H on tiny
# (B,)/(B,P) tensors, so it is launch-overhead-bound, not FLOP-bound. CUDA graphs capture
# the launch sequence once and replay it, cutting that overhead (~3x measured at H=400 on
# a GTX 1080 Ti). torch.compile(reduce-overhead) does the same plus operator fusion, but
# its Triton backend needs CUDA capability >= 7.0, so on Pascal it falls back to cudagraph.
#
# Backends ('rollout_backend'):
#   'eager'     -- plain Python loop (default; always correct, no capture).
#   'cudagraph' -- torch.cuda.make_graphed_callables (fwd+bwd graph); works on Pascal+.
#   'compile'   -- torch.compile(mode='reduce-overhead'); needs sm>=70, else -> cudagraph.
#
# Graphs require static shapes, so one graph is captured per (H, batch, dtype). Use the
# 'pow2' horizon schedule (horizon_for_epoch) to keep the number of distinct H small.
_ROLLOUT_CACHE = {}
_ROLLOUT_WARNED = set()


def _warn_once(key, msg):
    if key not in _ROLLOUT_WARNED:
        _ROLLOUT_WARNED.add(key)
        warnings.warn(msg)


def _rk4_core_factory(H, powers):
    """Build the pure RK4 integrator for a fixed horizon H, closing over H and the
    (constant) `powers` vector so the returned callable takes only the per-step tensors
    (om2, ga, w, xc, xp) -- the form torch.compile / make_graphed_callables require."""
    def core(om2, ga, w, xc, xp):
        xs = [xc]
        for k in range(H - 1):
            om2k, gak, w_k = om2[:, k], ga[:, k], w[:, k]

            def f(xx, vv):
                xpw = xx.unsqueeze(1) ** powers  # (B, P)
                xvw = vv.unsqueeze(1) ** powers  # (B, P)
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
                return vv, -om2k * xx - gak * vv - kern

            k1x, k1v = f(xc, xp)
            k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v)
            k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v)
            k4x, k4v = f(xc + k3x, xp + k3v)
            xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            # soft-tanh saturation (>> data scale; only tames blow-up during training)
            xc = BX * torch.tanh(xc / BX)
            xp = BXP * torch.tanh(xp / BXP)
            xs.append(xc)
        return torch.stack(xs, dim=1)  # (B, H)
    return core


def _rk4_step(carry, x, powers):
    """One RK4 step as a scan combine_fn: carry=(xc, xp) each (B,); x=(om2k, gak, w_k)
    are the step's drives (om2k,gak: (B,); w_k: (B,P,P)). Returns ((xc', xp'), xc').
    This is the single loop body a `torch.compile`d scan lowers once (instead of unrolling
    the whole H-step Python loop), and is identical math to f() in _rk4_core_factory."""
    xc, xp = carry
    om2k, gak, w_k = x

    def f(xx, vv):
        xpw = xx.unsqueeze(1) ** powers
        xvw = vv.unsqueeze(1) ** powers
        kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
        return vv, -om2k * xx - gak * vv - kern

    k1x, k1v = f(xc, xp)
    k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v)
    k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v)
    k4x, k4v = f(xc + k3x, xp + k3v)
    xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    xc = BX * torch.tanh(xc / BX)
    xp = BXP * torch.tanh(xp / BXP)
    return (xc, xp), xc


def _scan_rollout(om2, ga, w, xc, xp, powers, H, use_hop):
    """Express the rollout as a scan over the H-1 steps: carry=(xc, xp), per-step inputs
    are the time-leading drives. Output is (B, H) = initial xc then the H-1 step outputs.

    use_hop=True uses torch._higher_order_ops.scan (the jax.lax.scan analog), which lowers
    the step body once under torch.compile -- avoiding Dynamo's 2048x loop unroll. Requires
    being inside torch.compile (and thus sm>=70). use_hop=False runs the identical fold as
    an eager Python loop (same semantics, no compile) -- the correctness-equivalent fallback
    that also runs on hardware where torch.compile is unavailable (e.g. Pascal)."""
    n = H - 1
    # drives for steps 0..H-2, arranged time-leading (scan iterates the leading dim).
    om2_t = om2[:, :n].transpose(0, 1)         # (n, B)
    ga_t = ga[:, :n].transpose(0, 1)           # (n, B)
    w_t = w[:, :n].permute(1, 0, 2, 3)         # (n, B, P, P)

    def step(carry, x):
        return _rk4_step(carry, x, powers)

    if use_hop:
        from torch._higher_order_ops.scan import scan as _scan_hop
        _, ys = _scan_hop(step, (xc, xp), (om2_t, ga_t, w_t))  # ys: (n, B)
    else:
        carry, ys_list = (xc, xp), []
        for k in range(n):
            carry, y = step(carry, (om2_t[k], ga_t[k], w_t[k]))
            ys_list.append(y)
        ys = (torch.stack(ys_list, dim=0) if ys_list
              else om2_t.new_zeros((0, xc.shape[0])))
    return torch.cat([xc.unsqueeze(1), ys.transpose(0, 1)], dim=1)  # (B, H)


def _make_graphed(H, powers, om2, ga, w, xc, xp):
    """Capture a fwd+bwd CUDA graph of the RK4 core for these shapes, or fall back to the
    eager core if capture fails (e.g. OOM under a busy GPU). The eager fallback is cached
    so we don't re-attempt capture every step."""
    core = _rk4_core_factory(H, powers)
    try:
        sample = (
            om2.detach().clone().requires_grad_(om2.requires_grad),
            ga.detach().clone().requires_grad_(ga.requires_grad),
            w.detach().clone().requires_grad_(w.requires_grad),
            xc.detach().clone(),
            xp.detach().clone(),
        )
        return torch.cuda.make_graphed_callables(core, sample)
    except Exception as e:  # OOM during capture, unsupported op, etc.
        _warn_once(("graph_fail", H),
                   f"CUDA-graph capture failed for H={H} "
                   f"({type(e).__name__}: {e}); using eager rollout for this horizon.")
        return core


def _run_rk4(backend, om2, ga, w, xc, xp, powers, H):
    """Execute the RK4 rollout under the requested backend. om2/ga/w are (B, >=H[, P, P]);
    xc/xp are (B,). Returns (B, H)."""
    if backend == "eager" or not om2.is_cuda:
        return _rk4_core_factory(H, powers)(om2, ga, w, xc, xp)

    if backend == "compile":
        cap = torch.cuda.get_device_capability(om2.device)
        if cap[0] < 7:
            _warn_once("compile_cap",
                       f"rollout_backend='compile' needs CUDA capability >= 7.0 for the "
                       f"Triton backend; this GPU is sm_{cap[0]}{cap[1]} -> using "
                       f"'cudagraph' instead.")
            return _run_rk4("cudagraph", om2, ga, w, xc, xp, powers, H)
        key = ("compile", H, int(powers.numel()))
        fn = _ROLLOUT_CACHE.get(key)
        if fn is None:
            fn = torch.compile(_rk4_core_factory(H, powers), mode="reduce-overhead")
            _ROLLOUT_CACHE[key] = fn
        return fn(om2, ga, w, xc, xp)

    if backend == "cudagraph":
        key = ("graph", H, int(xc.shape[0]), om2.dtype, int(powers.numel()))
        fn = _ROLLOUT_CACHE.get(key)
        if fn is None:
            fn = _make_graphed(H, powers, om2, ga, w, xc, xp)
            _ROLLOUT_CACHE[key] = fn
        return fn(om2, ga, w, xc, xp)

    if backend == "scan":
        # Express the rollout as a scan. On sm>=70 compile the scan so the step body is
        # lowered once (the torch._higher_order_ops.scan HOP must run under torch.compile);
        # elsewhere run the identical eager fold (same result, no compile -- e.g. Pascal).
        cap = torch.cuda.get_device_capability(om2.device)
        if cap[0] >= 7:
            key = ("scan", H, int(powers.numel()))
            fn = _ROLLOUT_CACHE.get(key)
            if fn is None:
                fn = torch.compile(
                    lambda *a: _scan_rollout(*a, powers, H, use_hop=True),
                    mode="reduce-overhead",
                )
                _ROLLOUT_CACHE[key] = fn
            return fn(om2, ga, w, xc, xp)
        _warn_once("scan_cap",
                   f"rollout_backend='scan' lowers the scan HOP via torch.compile, which "
                   f"needs CUDA capability >= 7.0; this GPU is sm_{cap[0]}{cap[1]} -> running "
                   f"the eager fold (correct, no speedup).")
        return _scan_rollout(om2, ga, w, xc, xp, powers, H, use_hop=False)

    raise ValueError(f"unknown rollout_backend {backend!r}")


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
    drives: Optional[tuple] = None,  # precomputed (omega, gamma, weights, z2); skips get_funcs
    rollout_backend: str = "eager",  # 'eager' | 'cudagraph' | 'compile' (see _run_rk4)
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

    # Encode drives via the Mamba encoders. When the caller already ran get_funcs
    # (e.g. spectral_rollout_step computes the TF anchor from the same drives), it
    # passes them in so we don't run a redundant Mamba forward pass — the drives are
    # deterministic in (x, dxdt, dt), so reusing them is gradient-equivalent.
    if drives is None:
        omega, gamma, _, weights, _ = model.get_funcs(x, dxdt.clone(), dt)  # ω,γ:(B,L,1); w:(B,L,P,P)
        z2 = (model.tau / dt) * dxdt  # rescaled-time velocity (B, L, 1)
    else:
        omega, gamma, weights, z2 = drives

    P = weights.shape[-1]
    # reuse the kernel's cached powers vector instead of allocating a fresh arange each
    # rollout, and square omega once over the horizon (it was re-squared every RK4 substep).
    # Slice the drives to the horizon so the graphed/compiled shapes depend only on H.
    powers = model.kernel.powers[:P]
    om2 = omega[:, :H, 0] ** 2     # (B, H)
    ga_h = gamma[:, :H, 0]         # (B, H)
    w_h = weights[:, :H]           # (B, H, P, P)

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

    return _run_rk4(rollout_backend, om2, ga_h, w_h, xc, xp, powers, H)


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
    lam_env_log: float = 0.0,         # weight on the log-ratio envelope loss (env_loss_log)
    env_log_eps: float = 1e-4,        # noise floor inside the log() in env_loss_log
    env_ms: float = 2.0,
    tf_var: Optional[float] = None,   # precomputed Var(d2x) over the dataset; matches rollout_refine.py:110
    ic_mask: Optional[torch.Tensor] = None,
    ic_noise_rms: float = 1e-3,
    rng: Optional[torch.Generator] = None,
    rollout_backend: str = "eager",
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

    # Rollout + spectral loss. Reuse the drives (ω, γ, w) and rescaled velocity z2
    # already encoded above for the TF anchor instead of re-running the Mamba encoders
    # inside the rollout — they are deterministic in (x, dxdt, dt), so backprop through
    # the single shared forward sums the TF and spectral gradients exactly as before.
    xg = teacher_forced_rollout(
        model, x, dxdt, dt, H=H,
        ic_mask=ic_mask, ic_noise_rms=ic_noise_rms, rng=rng,
        drives=(omega, gamma, weights, z2),
        rollout_backend=rollout_backend,
    )
    tgt = x[:, :H, 0]

    configs_H = _filter_configs_for_horizon(configs, H)
    L_spec = mrstft_loss(xg, tgt, configs_H)
    if lam_env > 0:
        L_env = env_loss(xg, tgt, dt, env_ms)
    else:
        L_env = torch.zeros((), device=x.device, dtype=x.dtype)
    # Log-ratio envelope loss: symmetric in (auto, target) scale so the model can't satisfy it
    # by shrinking past optimum. dB-natural; see train/rollout_refine.py::env_loss_log.
    if lam_env_log > 0:
        L_env_log = env_loss_log(xg, tgt, dt, env_ms, eps=env_log_eps)
    else:
        L_env_log = torch.zeros((), device=x.device, dtype=x.dtype)

    total = lam_spec * L_spec + lam_tf * L_tf + lam_env * L_env + lam_env_log * L_env_log
    return {"spec": L_spec, "tf": L_tf, "env": L_env, "env_log": L_env_log, "total": total}


def pow2_horizon_buckets(H_min: int, H_max: int) -> list:
    """Horizon buckets in factor-of-2 jumps from H_min, with the final bucket capped at
    exactly H_max: e.g. H_min=512, H_max=2048 -> [512, 1024, 2048]. Keeping the set of
    distinct horizons small bounds the number of CUDA graphs the graphed/compiled rollout
    backends capture (one per H)."""
    hs = []
    h = int(H_min)
    while h < int(H_max):
        hs.append(h)
        h *= 2
    if not hs or hs[-1] != int(H_max):
        hs.append(int(H_max))
    return hs


def horizon_for_epoch(epoch: int, n_epochs: int, H_min: int, H_max: int,
                      schedule: str = "geom") -> int:
    """Curriculum on the rollout horizon: small H early (cheap, easy gradient), full H
    by the end. 'geom' = geometric spacing; 'linear' = linear; 'const' = H_max throughout;
    'pow2' = factor-of-2 buckets (pow2_horizon_buckets) spread evenly across the same
    n_epochs ramp -- coarse steps so the graphed backends capture only a few graphs.
    """
    if n_epochs <= 1 or schedule == "const":
        return int(H_max)
    if schedule == "pow2":
        buckets = pow2_horizon_buckets(H_min, H_max)
        # even division of the ramp across buckets; reaches the last bucket by the final epoch
        idx = min(int(epoch * len(buckets) / n_epochs), len(buckets) - 1)
        return int(buckets[idx])
    t = epoch / max(1, n_epochs - 1)
    if schedule == "linear":
        H = H_min + t * (H_max - H_min)
    elif schedule == "geom":
        H = H_min * (H_max / H_min) ** t
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return int(round(H))
