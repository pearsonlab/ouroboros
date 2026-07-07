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

The RK4 inner loop and soft-tanh saturation (BX, BXP from train/rollout_refine.py) match,
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
    gaussian_envelope,
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
#   'graphstep' -- CUDA-graph ONE RK4 step (fwd-only + recompute-fwd+bwd graphs) and replay
#                  it H-1 times through a custom autograd.Function. Correct on Pascal+ and
#                  ~3.9x faster than eager for the rollout fwd+bwd at H=2048 (measured, 1080 Ti).
#                  Capture is keyed on (B, P, dtype) -- NOT H -- so one capture serves every
#                  horizon and capture memory is O(1) (~130 MiB), unlike 'cudagraph' below.
#   'cudagraph' -- torch.cuda.make_graphed_callables of the WHOLE rollout (fwd+bwd graph).
#                  DEAD END for training: whole-loop capture does not reuse per-step scratch,
#                  so capture memory scales ~per-step (~30 GB at H=2048) and OOMs. Kept only
#                  for the forward-only/short-H cases it was originally measured on.
#   'compile'   -- torch.compile(mode='reduce-overhead'); needs sm>=70, else -> cudagraph.
#
# 'cudagraph'/'compile' capture one graph per (H, batch, dtype); use the 'pow2' horizon
# schedule to keep the number of distinct H small. 'graphstep' is H-invariant and is the
# recommended fast backend on this hardware.
_ROLLOUT_CACHE = {}
_ROLLOUT_WARNED = set()

# Clip the OU noise increment/state to +-NOISE_CLIP standard deviations. The OU has unit
# stationary variance, so +-4 truncates a ~6e-5 tail -- negligible for the noise spectrum,
# but it removes the rare Gaussian spikes that would otherwise kick the state past the
# trained region. Paired with the in-substep kernel-input clamp (see the Heun drifts), this
# keeps the high-order polynomial from overflowing fp32 -> NaN under the noise forcing.
NOISE_CLIP = 4.0


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
                # Clamp the kernel INPUT (soft-tanh) before the v^P polynomial so a runaway
                # (gamma<0) substep velocity can't overflow fp32 -> NaN. Mirrors the Heun core.
                # The LINEAR terms (-om2*x - ga*v) keep the raw state -- only the polynomial
                # needs bounding. Costs a couple (B,) tanh tensors/substep for backward (the
                # memory the old comment fretted about); worth it to keep the RK4 path from
                # diverging when the oscillator self-oscillates hard.
                xx_c = BX * torch.tanh(xx / BX)
                vv_c = BXP * torch.tanh(vv / BXP)
                xpw = xx_c.unsqueeze(1) ** powers  # (B, P)
                xvw = vv_c.unsqueeze(1) ** powers  # (B, P)
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


def _heun_core_factory(H, powers):
    """Stochastic-Heun integrator over the AUGMENTED state (x, v, eta) for the noise-forcing
    path. eta is a scalar Ornstein-Uhlenbeck process (correlation time tau_c samples, unit
    stationary variance) that is added to the momentum equation via g_eff(t)*eta, where
    g_eff = noise_gain * relu(sigma head) is the pre-scaled gate. Only the eta line carries
    the Wiener increment dW, and there with a CONSTANT coefficient sqrt(2/tau_c) -> the SDE is
    ADDITIVE, so Heun (stochastic trapezoidal) is strong order 1 with no Milstein/Ito-
    Stratonovich correction. The dW tensor is pre-sampled once per rollout (reparameterized):
    tau_c and the gate get gradients through this deterministic recurrence, dW does not.

    Unit rescaled step (h=1), matching the RK4 core. Deterministic (x, v) drift is the same
    f() as _rk4_core_factory plus the +g_eff*eta forcing; same soft-tanh state clamps."""
    def core(om2, ga, w, xc, xp, eta, g_eff, dW, tau_c):
        c1 = 1.0 - 1.0 / tau_c              # OU decay per unit step
        c2 = (2.0 / tau_c) ** 0.5           # OU diffusion coefficient (unit stationary var)
        xs = [xc]
        for k in range(H - 1):
            om2k, gak, w_k = om2[:, k], ga[:, k], w[:, k]
            g_k = g_eff[:, k]
            xi = dW[:, k].clamp(-NOISE_CLIP, NOISE_CLIP)   # bound the noise increment

            def drift(xx, vv, ee):
                # Clamp the kernel INPUT so the high-order polynomial can't overflow fp32 when a
                # noise spike kicks the (unclamped) predictor state large -- the einsum would
                # otherwise sum +inf/-inf -> NaN. Linear -om2*x -ga*v terms stay on the unclamped
                # state to keep standard integrator semantics (mirrors the eval-side clamp in
                # train.eval.integrate_poly_autonomous). Near-identity for in-range states.
                xx_c = BX * torch.tanh(xx / BX)
                vv_c = BXP * torch.tanh(vv / BXP)
                xpw = xx_c.unsqueeze(1) ** powers
                xvw = vv_c.unsqueeze(1) ** powers
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
                # dx/ds = v ; dv/ds = -om2*x - ga*v - kernel + g_eff*eta ; deta/ds = -eta/tau_c
                return vv, -om2k * xx - gak * vv - kern + g_k * ee, -ee / tau_c

            ax, av, ae = drift(xc, xp, eta)
            # predictor (Euler), with the noise increment on the eta line
            xt = xc + ax
            vt = xp + av
            et = eta + ae + c2 * xi
            axt, avt, aet = drift(xt, vt, et)
            # corrector (trapezoidal drift; same noise increment)
            xc = xc + 0.5 * (ax + axt)
            xp = xp + 0.5 * (av + avt)
            eta = (eta + 0.5 * (ae + aet) + c2 * xi).clamp(-NOISE_CLIP, NOISE_CLIP)  # bound OU state
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
        # See _rk4_core_factory.f re. why substep-clamp is not applied in training.
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


class _GraphedRK4Step:
    """One RK4 step captured as two CUDA graphs, exposed as an autograd.Function.

    g_fwd: forward-only (used in the rollout loop). g_bwd: recompute-forward + autograd.grad
    (used in backward). Saved tensors are real clones, so calling the Function H-1 times and
    backpropagating ONCE is correct -- unlike make_graphed_callables, whose shared static
    output buffers collapse a multi-call-then-single-backward into the last step's values.

    Capture is keyed on (B, P, dtype) only: the step's shapes don't depend on the horizon H,
    so one capture is replayed for every step of every horizon, with O(1) capture memory.
    `powers` is closed over at capture time (constant arange(P) for a given kernel).
    """
    def __init__(self, B, P, powers, device, dtype):
        # Own a private copy of `powers`: the captured graphs reference its data pointer for
        # their whole lifetime, but the `powers` passed in is a view into the model's kernel
        # buffer, which is freed when that model is GC'd (e.g. between seeds). Aliasing it
        # would make later replays read freed memory and emit NaNs. Holding self.powers keeps
        # the buffer alive; the closure below captures this owned tensor.
        powers = powers.detach().clone()
        self.powers = powers
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        self.s = [z(B).requires_grad_(True),        # om2k
                  z(B).requires_grad_(True),        # gak
                  z(B, P, P).requires_grad_(True),  # w_k
                  z(B).requires_grad_(True),        # xc
                  z(B).requires_grad_(True)]        # xp
        self.s_gxc, self.s_gxp = z(B), z(B)

        def step(om2k, gak, w_k, xc, xp):
            def f(xx, vv):
                # Kernel-input soft-tanh clamp (matches _rk4_core_factory.f) so graphstep RK4
                # is bit-for-bit with eager and can't overflow v^P on a gamma<0 runaway.
                xx_c = BX * torch.tanh(xx / BX)
                vv_c = BXP * torch.tanh(vv / BXP)
                xpw = xx_c.unsqueeze(1) ** powers
                xvw = vv_c.unsqueeze(1) ** powers
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
                return vv, -om2k * xx - gak * vv - kern
            k1x, k1v = f(xc, xp)
            k2x, k2v = f(xc + 0.5 * k1x, xp + 0.5 * k1v)
            k3x, k3v = f(xc + 0.5 * k2x, xp + 0.5 * k2v)
            k4x, k4v = f(xc + k3x, xp + k3v)
            xc = xc + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            return BX * torch.tanh(xc / BX), BXP * torch.tanh(xp / BXP)

        # Warm up both the grad and no-grad paths on a side stream before capture (required
        # so cuDNN/cuBLAS workspaces etc. are allocated outside the captured region).
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                oxc, oxp = step(*self.s)
                torch.autograd.grad((oxc, oxp), self.s, (self.s_gxc, self.s_gxp))
            with torch.no_grad():
                for _ in range(3):
                    step(*self.s)
        torch.cuda.current_stream().wait_stream(stream)

        self.g_fwd = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self.g_fwd):
                self.f_oxc, self.f_oxp = step(*self.s)
        self.g_bwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_bwd):
            oxc, oxp = step(*self.s)
            self.cg = torch.autograd.grad((oxc, oxp), self.s, (self.s_gxc, self.s_gxp))

        gs = self

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, om2k, gak, w_k, xc, xp):
                with torch.no_grad():
                    for buf, val in zip(gs.s, (om2k, gak, w_k, xc, xp)):
                        buf.copy_(val)
                    gs.g_fwd.replay()
                    out = (gs.f_oxc.clone(), gs.f_oxp.clone())
                ctx.save_for_backward(om2k.detach(), gak.detach(), w_k.detach(),
                                      xc.detach(), xp.detach())
                return out

            @staticmethod
            def backward(ctx, g_oxc, g_oxp):
                with torch.no_grad():
                    for buf, val in zip(gs.s, ctx.saved_tensors):
                        buf.copy_(val)
                    gs.s_gxc.copy_(g_oxc); gs.s_gxp.copy_(g_oxp)
                    gs.g_bwd.replay()
                    return tuple(c.clone() for c in gs.cg)

        self.apply = _Fn.apply


def _graphstep_rollout(om2, ga, w, xc, xp, powers, H):
    """RK4 rollout that replays a per-step CUDA graph (see _GraphedRK4Step). Same (B, H)
    output and gradients as _rk4_core_factory, ~3.9x faster at H=2048 on Pascal."""
    B, P = xc.shape[0], int(powers.numel())
    key = ("graphstep", B, P, om2.dtype)
    gstep = _ROLLOUT_CACHE.get(key)
    if gstep is None:
        gstep = _GraphedRK4Step(B, P, powers, om2.device, om2.dtype)
        _ROLLOUT_CACHE[key] = gstep
    xs = [xc]
    for k in range(H - 1):
        xc, xp = gstep.apply(om2[:, k], ga[:, k], w[:, k].contiguous(), xc, xp)
        xs.append(xc)
    return torch.stack(xs, dim=1)


def _run_rk4(backend, om2, ga, w, xc, xp, powers, H):
    """Execute the RK4 rollout under the requested backend. om2/ga/w are (B, >=H[, P, P]);
    xc/xp are (B,). Returns (B, H)."""
    if backend == "eager" or not om2.is_cuda:
        return _rk4_core_factory(H, powers)(om2, ga, w, xc, xp)

    if backend == "graphstep":
        return _graphstep_rollout(om2, ga, w, xc, xp, powers, H)

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


class _GraphedHeunStep:
    """One stochastic-Heun step over (x, v, eta) captured as two CUDA graphs, exposed as an
    autograd.Function. The noise-forcing analogue of _GraphedRK4Step: same fwd-only + recompute
    -fwd+bwd graph pair, same H-invariant capture, extended to carry the OU state `eta`, the
    pre-scaled gate `g_k` (grad-carrying -> the sigma head learns), and the pre-sampled Wiener
    increment `dW_k` (NO grad -> reparameterized external noise). tau_c is a fixed model config,
    baked into the captured kernels (and into the cache key so distinct tau_c don't collide)."""
    def __init__(self, B, P, powers, device, dtype, tau_c):
        powers = powers.detach().clone()          # own the buffer (see _GraphedRK4Step)
        self.powers = powers
        self.tau_c = float(tau_c)
        c2 = (2.0 / self.tau_c) ** 0.5
        tau_c = self.tau_c
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        # grad-carrying inputs (order fixed; backward returns grads in this order + None for dW)
        self.s = [z(B).requires_grad_(True),        # om2k
                  z(B).requires_grad_(True),        # gak
                  z(B, P, P).requires_grad_(True),  # w_k
                  z(B).requires_grad_(True),        # xc
                  z(B).requires_grad_(True),        # xp
                  z(B).requires_grad_(True),        # eta
                  z(B).requires_grad_(True)]        # g_k (= noise_gain * gate)
        self.dW = z(B)                              # external noise increment; no grad
        self.s_gxc, self.s_gxp, self.s_geta = z(B), z(B), z(B)

        def step(om2k, gak, w_k, xc, xp, eta, g_k):
            xi = self.dW.clamp(-NOISE_CLIP, NOISE_CLIP)   # bound the noise increment (matches eager)
            def drift(xx, vv, ee):
                # Clamp kernel input to prevent fp32 overflow under a noise spike; linear terms
                # stay on the unclamped state. Identical to _heun_core_factory.drift so the eager
                # and graphstep backends remain bit-for-bit equal (backend-parity test).
                xx_c = BX * torch.tanh(xx / BX)
                vv_c = BXP * torch.tanh(vv / BXP)
                xpw = xx_c.unsqueeze(1) ** powers
                xvw = vv_c.unsqueeze(1) ** powers
                kern = torch.einsum("bp,bk,bpk->b", xpw, xvw, w_k)
                return vv, -om2k * xx - gak * vv - kern + g_k * ee, -ee / tau_c
            ax, av, ae = drift(xc, xp, eta)
            xt = xc + ax; vt = xp + av; et = eta + ae + c2 * xi
            axt, avt, aet = drift(xt, vt, et)
            xo = xc + 0.5 * (ax + axt)
            vo = xp + 0.5 * (av + avt)
            eo = (eta + 0.5 * (ae + aet) + c2 * xi).clamp(-NOISE_CLIP, NOISE_CLIP)
            return BX * torch.tanh(xo / BX), BXP * torch.tanh(vo / BXP), eo

        # Warm up grad + no-grad paths on a side stream before capture (see _GraphedRK4Step).
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                oxc, oxp, oeta = step(*self.s)
                torch.autograd.grad((oxc, oxp, oeta), self.s,
                                    (self.s_gxc, self.s_gxp, self.s_geta))
            with torch.no_grad():
                for _ in range(3):
                    step(*self.s)
        torch.cuda.current_stream().wait_stream(stream)

        self.g_fwd = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self.g_fwd):
                self.f_oxc, self.f_oxp, self.f_oeta = step(*self.s)
        self.g_bwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_bwd):
            oxc, oxp, oeta = step(*self.s)
            self.cg = torch.autograd.grad((oxc, oxp, oeta), self.s,
                                          (self.s_gxc, self.s_gxp, self.s_geta))

        gs = self

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, om2k, gak, w_k, xc, xp, eta, g_k, dW_k):
                with torch.no_grad():
                    for buf, val in zip(gs.s, (om2k, gak, w_k, xc, xp, eta, g_k)):
                        buf.copy_(val)
                    gs.dW.copy_(dW_k)
                    gs.g_fwd.replay()
                    out = (gs.f_oxc.clone(), gs.f_oxp.clone(), gs.f_oeta.clone())
                ctx.save_for_backward(om2k.detach(), gak.detach(), w_k.detach(),
                                      xc.detach(), xp.detach(), eta.detach(),
                                      g_k.detach(), dW_k.detach())
                return out

            @staticmethod
            def backward(ctx, g_oxc, g_oxp, g_oeta):
                with torch.no_grad():
                    om2k, gak, w_k, xc, xp, eta, g_k, dW_k = ctx.saved_tensors
                    for buf, val in zip(gs.s, (om2k, gak, w_k, xc, xp, eta, g_k)):
                        buf.copy_(val)
                    gs.dW.copy_(dW_k)
                    gs.s_gxc.copy_(g_oxc); gs.s_gxp.copy_(g_oxp); gs.s_geta.copy_(g_oeta)
                    gs.g_bwd.replay()
                    # 7 grads for the 7 grad-carrying inputs; dW_k gets None.
                    return tuple(c.clone() for c in gs.cg) + (None,)

        self.apply = _Fn.apply


def _graphstep_heun_rollout(om2, ga, w, xc, xp, eta, g_eff, dW, powers, H, tau_c):
    """Stochastic-Heun rollout replaying a per-step CUDA graph (see _GraphedHeunStep). Same
    (B, H) output and gradients as _heun_core_factory, at graphstep speed. tau_c is part of the
    cache key so a model with a different correlation time captures its own graph."""
    B, P = xc.shape[0], int(powers.numel())
    key = ("heun", B, P, om2.dtype, float(tau_c))
    gstep = _ROLLOUT_CACHE.get(key)
    if gstep is None:
        gstep = _GraphedHeunStep(B, P, powers, om2.device, om2.dtype, tau_c)
        _ROLLOUT_CACHE[key] = gstep
    xs = [xc]
    for k in range(H - 1):
        xc, xp, eta = gstep.apply(om2[:, k], ga[:, k], w[:, k].contiguous(),
                                  xc, xp, eta, g_eff[:, k], dW[:, k])
        xs.append(xc)
    return torch.stack(xs, dim=1)


def _run_heun(backend, om2, ga, w, xc, xp, eta, g_eff, dW, powers, H, tau_c):
    """Execute the augmented-state stochastic-Heun rollout under the requested backend. Only
    'eager' and 'graphstep' are implemented for the noise path; any other backend falls back to
    eager (correct, no capture). RK4 (noise-off) is unaffected -- see _run_rk4."""
    if backend == "graphstep" and om2.is_cuda:
        return _graphstep_heun_rollout(om2, ga, w, xc, xp, eta, g_eff, dW, powers, H, tau_c)
    if backend not in ("eager", "graphstep") or not om2.is_cuda:
        _warn_once(("heun_backend", backend),
                   f"rollout_backend={backend!r} is not implemented for noise forcing; "
                   f"using the eager stochastic-Heun rollout (correct, no capture).")
    return _heun_core_factory(H, powers)(om2, ga, w, xc, xp, eta, g_eff, dW, tau_c)


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
    gate: Optional[torch.Tensor] = None,   # (B, L, 1) noise gate g(t) from model.get_sigma; None => deterministic RK4
    noise_gain: float = 1.0,               # scalar ramp/ablation multiplier on the forcing (0 => term off)
    noise_tau_samp: Optional[float] = None,  # OU correlation time in samples (required when gate is not None)
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

    if gate is None:
        return _run_rk4(rollout_backend, om2, ga_h, w_h, xc, xp, powers, H)

    # --- flow-gated colored-noise forcing (stochastic-Heun path) ---
    assert noise_tau_samp is not None, "gate given but noise_tau_samp (tau_c in samples) is None"
    # Pre-sample the per-step Wiener increments ONCE per rollout and hold them fixed through
    # the recurrence (reparameterization): xi_k ~ N(0, 1), so eta's E-M step has dt=1 in
    # rescaled-sample time. Resampled on the next call (next minibatch). Fold the ramp/ablation
    # gain into the gate so noise_gain=0 makes the forcing EXACTLY 0 (=> output independent of
    # dW) and grad still reaches the sigma head through g_eff.
    dW = torch.randn(B, H, device=x.device, generator=rng)
    eta0 = torch.zeros(B, device=x.device)          # OU state; relaxes to stationary in ~tau_c
    g_eff = noise_gain * gate[:, :H, 0]              # (B, H)
    return _run_heun(rollout_backend, om2, ga_h, w_h, xc, xp, eta0, g_eff, dW,
                     powers, H, float(noise_tau_samp))


def filtered_noise_branch(model, x, dxdt, dt, H, rng=None):
    """Additive noise source (harmonic-plus-noise mode).

    white noise (reparameterized: sampled once, held fixed) -> model.noise_tract (a LOW-ORDER
    rational pole/zero filter, same parameterization as the vocal tract, too coarse to synthesize
    sharp harmonic peaks) -> amplitude-modulate by the sigma gate g(t). Returns (B, H) to be ADDED
    to the tract output. Differentiable in the sigma head + noise_tract; the low filter order is
    what forces the oscillator (not the noise) to carry the tonal/harmonic structure.
    """
    B = x.shape[0]
    dev = x.device
    g = model.get_sigma(x, dxdt.clone(), dt)[:, :H, 0]                 # (B, H) AM gate >= 0
    w = torch.randn(B, H, device=dev, dtype=x.dtype, generator=rng)    # white noise
    colored = model.noise_tract.apply(w[..., None])[..., 0]           # (B, H) low-order rational
    return g * colored                                                # sigma AM gate


def rumble_branch(model, x, dxdt, dt, H):
    """Deterministic low-frequency 'rumble' source (harmonic-plus-noise-plus-rumble mode).

    model.get_rumble runs a parallel Mamba head and band-limits its output to < rumble_lowpass_hz.
    Returns (B, H) to be ADDED to the tract output OUTSIDE the tract, alongside the noise branch.
    A dedicated cheap channel for the sub-cutoff recording floor so the oscillator isn't forced to
    spend capacity on it -- the oscillator is left FULL-RANGE (not high-passed), so it can still
    reach below the cutoff when a vocalization has genuine LF content. Differentiable in the head.
    """
    r = model.get_rumble(x, dxdt.clone(), dt)      # (B, L, 1) band-limited LF source
    return r[:, :H, 0]                             # (B, H)


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
    lam_reg: float = 0.0,             # scale on the degree-graded L2 penalty on kernel weights
    lam_env_anchor: float = 0.0,      # scale on mean((e - 1)^2) envelope gauge anchor; pulls e toward identity, per-sample backward grad bounded by 2|e-1|/N
    lam_tract_k_anchor: float = 0.0,  # scale on (K/K0 - 1)^2 tract-gain gauge anchor; pins K=softplus(K_raw) near its data-init K0, closing the (K, source) gauge the env anchor leaves open
    lam_env_max_anchor: float = 0.0,  # scale on mean((max_t e - 1)^2); pins the envelope PEAK to 1 (absolute scale) without penalizing its time-variance (shape)
    tf_var: Optional[float] = None,   # precomputed Var(d2x) over the dataset; matches rollout_refine.py:110
    ic_mask: Optional[torch.Tensor] = None,
    ic_noise_rms: float = 1e-3,
    rng: Optional[torch.Generator] = None,
    rollout_backend: str = "eager",
    noise_gain: float = 0.0,          # ramp/ablation multiplier on the OU forcing (0 => off)
    osc_gain: float = 1.0,            # gain on the deterministic (tract) output; warmup ramp gates it in
    mel_spec: bool = False,           # compute the MRSTFT magnitude loss on mel-warped spectra
    mel_n_mels: int = 80,
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

    # Flow-gated colored-noise forcing (opt-in). The noise realization can only be supervised
    # in distribution by the phase-discarding MRSTFT magnitude loss below -- never pathwise --
    # so this path is only reachable with loss_mode='spectral_rollout' (guarded in train.train).
    # The gate g(t) is a learned Mamba head; noise_tau_samp is the fixed OU correlation time.
    # Skip the sigma encoder entirely while noise_gain==0 (warmup): the rollout then uses the
    # exact deterministic RK4 path (no wasted encoder forward, no Heun), and the stochastic-Heun
    # path engages only once the gain ramp lifts off at noise_start_step.
    noise_on = getattr(model, "enable_noise_forcing", False) and noise_gain > 0
    gate = model.get_sigma(x, dxdt.clone(), dt) if noise_on else None
    noise_tau_samp = ((model.noise_tau_ms / 1e3) / dt) if noise_on else None

    # Learnable amplitude envelope e(t): computed once here and reused by both the TF anchor
    # (immediately below) and the rollout (further down). None when the model has no envelope.
    e = model.get_envelope(x, dxdt.clone(), dt) if getattr(model, "use_envelope", False) else None

    # TF anchor (variance-normalized 1-step acceleration MSE; a basin-keeper for early epochs).
    # When the model carries the envelope, the ODE describes a UNIT-amplitude source -- e carries
    # amplitude in the rollout -- so the anchor must be computed on the envelope-NORMALIZED DATA
    # s = x/e (target d2x/e), NOT on the raw audio. We rescale the DATA, not the prediction, so
    # the kernel is evaluated at unit amplitude (plain poly; no reciprocal-e gains). Computing it
    # on the raw audio instead would fit the ODE's limit cycle to the full audio amplitude and
    # fight e. This anchor is OFF by default (lam_tf may be 0); this just makes its form correct
    # if it is enabled. e is floored to avoid 0/0 in deep silence. The tract is intentionally
    # left out of the anchor (it is identity at init and reshapes spectrum, not amplitude).
    # Short-circuit when lam_tf<=0: the envelope-normalized form scales as ~1/e and would
    # log enormous (then potentially inf) values at deep-silence ONSET segments. With lam_tf=0
    # the value isn't even backproppable, so skipping it saves a kernel forward and keeps the
    # TB curves readable. L_tf is returned as a zero tensor for downstream stacking.
    if lam_tf <= 0:
        L_tf = torch.zeros((), device=x.device, dtype=x.dtype)
    else:
        if e is not None:
            ef = e.clamp_min(1e-6)
            s, s2 = x / ef, z2 / ef
            wk_s = model.kernel.forward_given_weights(torch.cat([s, s2], dim=-1), weights.clone())
            tf_d2 = -(omega ** 2) * s - gamma * s2 - wk_s
            tf_target = d2x / ef
        else:
            tf_d2 = -(omega ** 2) * x - gamma * z2 - wk
            tf_target = d2x
        # Variance-normalized so the anchor scale is commensurable across vocs / runs.
        # tf_var should be the variance computed ONCE over the whole training set (see
        # rollout_refine.py:110) -- per-batch variance is unstable when the batch is mostly
        # silence (ONSET segments) and can drive the loss to explode. We clamp at a floor
        # to be extra safe.
        if tf_var is None:
            v = float(tf_target.detach().var().clamp_min(1e-3).item())
        else:
            v = max(float(tf_var), 1e-6)
        L_tf = ((tf_d2 - tf_target) ** 2).mean() / v

    # Rollout + spectral loss. Reuse the drives (ω, γ, w) and rescaled velocity z2
    # already encoded above for the TF anchor instead of re-running the Mamba encoders
    # inside the rollout — they are deterministic in (x, dxdt, dt), so backprop through
    # the single shared forward sums the TF and spectral gradients exactly as before.
    xg = teacher_forced_rollout(
        model, x, dxdt, dt, H=H,
        ic_mask=ic_mask, ic_noise_rms=ic_noise_rms, rng=rng,
        drives=(omega, gamma, weights, z2),
        rollout_backend=rollout_backend,
        gate=gate, noise_gain=noise_gain, noise_tau_samp=noise_tau_samp,
    )
    # NON-TRAINABLE STABILIZER on the raw RK4 source: subtract the per-segment mean
    # so any DC drift the integrator accumulated is gone BEFORE env(t) multiplies it.
    # Without this, env*DC becomes an additive DC modulated by env(t), which gets
    # fed into the (trainable) tract's frequency response at f=0 and can interact
    # badly with the formant filter. Differentiable, identity-on-AC. Mirrors the
    # eval-side detrend placement in train.eval.integrate_poly_autonomous._finish
    # (which uses the scipy butter HPF for stronger low-frequency rolloff; here
    # DC subtraction is enough because the STFT loss ignores the 0-frequency bin).
    xg = xg - xg.mean(dim=1, keepdim=True)

    # Apply the learnable amplitude envelope e(t) and the linear vocal tract H to the rolled-
    # out source BEFORE comparing to audio. e(t) multiplies the waveform per-timestep -- a
    # plain positive gain (no kernel reciprocal-e terms; a small e just makes the output
    # quiet), low-passed to ~20 ms. H is the forward LTI tract (source -> radiated audio).
    # Both are identity at init (e == 1, H == 1), so this reduces to the bare rollout, and
    # both are learned only through this MRSTFT comparison (the env penalty stays off). e is
    # the SAME tensor used by the TF anchor above (one env encode per step).
    if e is not None:
        xg = e[:, :H, 0] * xg  # (B, H)
    if getattr(model, "use_tract", False):
        xg = model.tract.apply(xg[..., None])[..., 0]  # (B, H)

    # Oscillator warmup gate: multiply the deterministic (oscillator -> env -> tract) output by
    # osc_gain. Held at 0 during warmup (train.train schedule) so the rumble+noise branches fit
    # the SPECTRAL loss FIRST, then ramped to 1 to bring the oscillator in for the harmonics. At
    # osc_gain=0 the tract output is 0, so the spectral loss sends NO gradient to the oscillator /
    # tract / envelope -- they are not recruited to the spectrum. (The TF anchor still uses the
    # drives directly, weight lam_tf, so the oscillator idles in a learnable basin meanwhile.)
    if osc_gain != 1.0:
        xg = osc_gain * xg

    # Harmonic-plus-noise: add the parallel filtered-noise branch OUTSIDE the tract. The
    # oscillator rollout above stayed fully deterministic (gate=None => RK4), so this is a
    # clean additive source -- no ODE coupling, no collapse. noise_gain ramps/gates it in.
    if getattr(model, "use_noise_branch", False) and noise_gain > 0:
        xg = xg + noise_gain * filtered_noise_branch(model, x, dxdt, dt, H, rng=rng)

    # Rumble branch: deterministic band-limited (< rumble_lowpass_hz) LF source added OUTSIDE the
    # tract, alongside the noise. Takes the sub-cutoff recording floor off the oscillator's plate.
    if getattr(model, "use_rumble_branch", False):
        xg = xg + rumble_branch(model, x, dxdt, dt, H)

    # Backward gradient clip at the rolled-out audio: caps the spec loss's backward
    # contribution norm to SPEC_GRAD_MAX_NORM before it flows back into env_mamba's
    # pscan and the polynomial dynamics. The 1/(A_i + eps) term in mrstft_loss can
    # produce per-bin grads ~1e5; after iSTFT-equivalent and pscan amplification these
    # easily overflow fp32 and freeze the model via the post-backward NaN check. Norm
    # clip preserves direction. Non-finite values are zeroed so a single bad bin
    # doesn't poison the whole chain. Identity in forward; only affects backward.
    if torch.is_grad_enabled() and xg.requires_grad:
        SPEC_GRAD_MAX_NORM = 10.0
        def _clip(g, max_norm=SPEC_GRAD_MAX_NORM):
            # Async clip: scale by min(1, max_norm/(n + eps)) instead of `if n > max_norm`.
            # The branch form calls `.__bool__()` on a 0-D CUDA tensor, forcing a host
            # sync per backward (~24% per-epoch overhead measured on 1080 Ti). The
            # clamp(max=1) keeps direction identical and avoids the sync.
            g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
            n = g.norm()
            scale = (max_norm / (n + 1e-12)).clamp(max=1.0)
            return g * scale
        xg.register_hook(_clip)

    tgt = x[:, :H, 0]

    configs_H = _filter_configs_for_horizon(configs, H)
    spec_parts = mrstft_loss(xg, tgt, configs_H, return_components=True,
                             mel=mel_spec, sr=1.0 / dt, n_mels=mel_n_mels)
    L_spec = spec_parts["spec"]
    L_sc = spec_parts["sc"]
    L_logm = spec_parts["logm"]
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

    # Degree-graded L2 penalty on the polynomial kernel weights. The weighting
    # `model.kernel.lam ** (i + j)` shapes how each polynomial term is penalised
    # by its total degree; `lam_reg` is the overall scale and defaults to 0 (off).
    # Operates on the same `weights` tensor returned by get_funcs above -- no extra
    # forward pass. Mirrors the legacy MSE-accel reg_weights block in train.train,
    # which never ran in spectral mode prior to this.
    if lam_reg > 0:
        P = weights.shape[-1]
        deg = torch.arange(P, dtype=weights.dtype, device=weights.device)
        lam_grid = float(model.kernel.lam) ** (deg.view(P, 1) + deg.view(1, P))  # (P, P)
        L_reg = (lam_grid * weights ** 2).sum(dim=(-1, -2, -3)).mean()
    else:
        L_reg = torch.zeros((), device=x.device, dtype=x.dtype)

    # Gauge-fixing anchor on the envelope: quadratic penalty mean((e - 1)^2) pulls e toward
    # the identity-init value e=1. Without this, the spec loss is gauge-invariant under
    # (e, x) -> (k*e, x/k) for any k > 0 -- which the kernel-weight L2 reg then exploits
    # by sending k -> infinity to shrink the polynomial weights to zero. The soft-tanh on
    # x (|x| <= BX) makes the (e -> 0, x -> infinity) direction self-bounded; only
    # (e -> infinity) is unbounded, and the quadratic asymmetrically penalizes e>1 more
    # than e<1. Per-sample backward grad is 2*(e_i - 1)/N -- bounded and smooth, no eps
    # machinery and no small-e gradient cliff.
    # Pointwise envelope supervision: pull the model's envelope head e(t) toward
    # the target audio's Gaussian-low-passed |x| amplitude envelope at every sample,
    # rather than the gauge-only (e - 1)^2 pull toward identity. Both share the same
    # role (fix the (e, x) -> (k*e, x/k) gauge), but this version gives the env_mamba
    # an explicit per-sample target instead of a constant. Mean square error over
    # the horizon, matching the (B, H) shapes from gaussian_envelope.
    if lam_env_anchor > 0 and e is not None:
        with torch.no_grad():
            tgt_env = gaussian_envelope(tgt, dt, env_ms)  # (B, H), no grad through target
        L_env_anchor = (e[:, :H, 0] - tgt_env).pow(2).mean()
    else:
        L_env_anchor = torch.zeros((), device=x.device, dtype=x.dtype)

    # Gauge-fixing anchor on the tract GAIN K = softplus(K_raw): quadratic pull toward its
    # data-init value K0 (model.tract.K_anchor_target). Closes the (K, source) -> (c*K, source/c)
    # gauge the envelope anchor leaves open -- pins the absolute output gain WITHOUT flattening
    # the envelope. Relative (K/K0 - 1)^2 so the weight is scale-free; K is a scalar so this is a
    # tiny, bounded-gradient term (dL/dK_raw = 2(K/K0 - 1)/K0 * sigmoid(K_raw)).
    tract = getattr(model, "tract", None)
    if lam_tract_k_anchor > 0 and tract is not None and getattr(tract, "K_anchor_target", None) is not None:
        K = torch.nn.functional.softplus(tract.K_raw)
        K0 = tract.K_anchor_target.clamp_min(1e-8)
        L_k_anchor = (K / K0 - 1.0).pow(2)
    else:
        L_k_anchor = torch.zeros((), device=x.device, dtype=x.dtype)

    # Scale-only anchor on the envelope: quadratic penalty on the departure of the PEAK of
    # e(t) from 1, mean over batch: mean((max_t e - 1)^2). Unlike lam_env_anchor (which pins
    # e(t) to a per-sample target and thus penalizes the whole shape), this fixes only the
    # absolute SCALE of the envelope -- its time-variance (syllable shaping) is unpenalized,
    # so e is free to vary but can't roam its overall level up (the runaway seen with only a
    # K anchor). Grad flows only through each sample's argmax timestep (max-pool subgradient).
    if lam_env_max_anchor > 0 and e is not None:
        e_max = e[:, :H, 0].amax(dim=1)          # (B,) per-sample envelope peak
        L_env_max = (e_max - 1.0).pow(2).mean()
    else:
        L_env_max = torch.zeros((), device=x.device, dtype=x.dtype)

    total = (
        lam_spec * L_spec
        + lam_tf * L_tf
        + lam_env * L_env
        + lam_env_log * L_env_log
        + lam_reg * L_reg
        + lam_env_anchor * L_env_anchor
        + lam_tract_k_anchor * L_k_anchor
        + lam_env_max_anchor * L_env_max
    )
    return {"spec": L_spec, "sc": L_sc, "logm": L_logm,
            "tf": L_tf, "env": L_env, "env_log": L_env_log,
            "reg": L_reg, "env_anchor": L_env_anchor,
            "k_anchor": L_k_anchor, "env_max": L_env_max, "total": total}


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


def horizon_for_step(step: int, total_steps: int, H_min: int, H_max: int,
                     schedule: str = "geom") -> int:
    """Curriculum on the rollout horizon, indexed by global TRAINING STEP (batch index)
    rather than epoch. Same formulas as horizon_for_epoch -- units differ. Lets the
    schedule stay calibrated when the dataset size changes (longer epochs no longer
    stretch the H ramp). 'geom' = geometric; 'linear' = linear; 'const' = H_max
    throughout; 'pow2' = factor-of-2 buckets spread evenly across total_steps.
    """
    if total_steps <= 1 or schedule == "const":
        return int(H_max)
    if schedule == "pow2":
        buckets = pow2_horizon_buckets(H_min, H_max)
        idx = min(int(step * len(buckets) / total_steps), len(buckets) - 1)
        return int(buckets[idx])
    t = step / max(1, total_steps - 1)
    if schedule == "linear":
        H = H_min + t * (H_max - H_min)
    elif schedule == "geom":
        H = H_min * (H_max / H_min) ** t
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return int(round(H))


def horizon_for_epoch(epoch: int, n_epochs: int, H_min: int, H_max: int,
                      schedule: str = "geom") -> int:
    """Epoch-indexed thin wrapper around horizon_for_step. Kept for backward
    compatibility with callers that don't track global step count."""
    return horizon_for_step(epoch, n_epochs, H_min, H_max, schedule)
