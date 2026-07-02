import torch
import numpy as np
from utils import (
    sst,
    sse,
    deriv_approx_d2y,
    deriv_approx_dy,
    butter_filter,
)
from torchdiffeq import odeint_adjoint

from scipy.interpolate import make_interp_spline
from scipy.signal import welch

"""
tools for evaluating model performance. covers both regular evaluation and model integration

"""


def correct(data: np.ndarray) -> np.ndarray:
    """
    corrects model integration using a low-pass filter. low-pass filters the data (assuming that
    in the integration window, your integrated signal shouldn't see any oscillations that are too slow
    (/100 oscillations in the integration window), then subtracts out to remove trends

    inputs
    -----
        - data: integrated second derivative
    returns
    -----
        - corrected integration
    """
    corrected = data.copy()
    low_pass = butter_filter(data, cutoff=100, fs=len(data), btype="low")
    corrected = data - low_pass

    return corrected


def integrate_model_d2(
    model: torch.nn.Module,
    audio: torch.FloatTensor,
    dt: float,
    method: str = "rk4",
    use_omega: bool = True,
    use_gamma: bool = True,
    use_nonlinearity: bool = True,
    null_comparison: bool = False,
    smoothing: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    integrate second derivative prediction from an ouroboros model for a segment of audio

    inputs
    -----
        - model: a trained ouroboros
        - audio: audio used as input
        - dt:audio sampling spacing
        - method: integration method
        - use_omega: whether to use omega model output
        - use_gamma: whether to use gamma model output
        - use_nonlinearity: whether to use model nonlinearity
        - null_comparison: whether we should just integrate audio (e.g., omega=gamma = 1)
        - smoothing: whether we should smooth model functions before integration
        - verbose: whether to be verbose or not

    returns
    -----
        - integrated (and corrected) second derivative
    """

    L = len(audio)

    t_steps = np.arange(0, L * dt + dt / 2, dt)[:L]

    audio = audio[None, :, None]
    dy = deriv_approx_dy(audio)

    audio = torch.from_numpy(audio).to(torch.float32).to("cuda")
    dy = torch.from_numpy(dy).to(torch.float32).to("cuda")
    ic = torch.hstack([audio[0, 0, 0], dy[0, 0, 0] / dt])

    with torch.no_grad():
        omega, gamma, _, weights, _ = model.get_funcs(
            audio, dy, dt, smoothing=smoothing
        )

    audio = audio.detach().cpu().numpy().squeeze()
    dy = dy.detach().cpu().numpy().squeeze() / dt * model.tau
    omega, gamma = (
        omega.detach().cpu().numpy().squeeze(),
        gamma.detach().cpu().numpy().squeeze(),
    )
    weights = weights.detach().cpu().numpy().squeeze()
    z = np.stack([audio, dy], axis=-1)

    if null_comparison:
        if use_omega:
            yhat = -audio
        else:
            yhat = 0
        if use_gamma:
            yhat += -dy

        if use_nonlinearity:
            kernel = model.kernel.forward_given_weights_numpy(
                z, weights[None, :, :, :]
            ).squeeze()

            yhat += -kernel
    else:
        if use_omega:
            yhat = -(omega**2) * audio
        else:
            yhat = 0

        if use_gamma:
            yhat += -gamma * dy

        if use_nonlinearity:
            kernel = model.kernel.forward_given_weights_numpy(
                z, weights[None, :, :, :]
            ).squeeze()

            yhat += -kernel

    yhat = yhat / model.tau**2

    integrated = integrate_second_deriv(
        yhat, ic=ic, eval_times=t_steps, method=method, verbose=verbose
    )

    return integrated


def integrate_estimated_d2(
    audio: torch.FloatTensor, dt: float, method: str = "rk4", verbose: bool = True
) -> np.ndarray:
    """
    comparison integration: estimate second derivative from audio, then integrate

    inputs
    -----
        - audio: audio to estimate
        - dt: audio sampling timestep
        - method: integration method
        - verbose: whether to print things

    returns
    -----
        - integrated and corrected second derivative
    """

    L = len(audio)

    t_steps = np.arange(0, L * dt + dt / 2, dt)[:L]

    yhat = deriv_approx_d2y(audio[None, :, None]).squeeze() / dt**2
    dy = deriv_approx_dy(audio[None, :, None]).squeeze()

    ic = (
        torch.from_numpy(np.hstack([audio[0], dy[0] / dt])).to(torch.float32).to("cuda")
    )
    integrated = integrate_second_deriv(
        yhat, ic, t_steps, method=method, verbose=verbose
    )

    return integrated


def integrate_second_deriv(
    deriv_approx: np.ndarray,
    ic: torch.FloatTensor,
    eval_times: np.ndarray,
    method: str = "rk4",
    verbose: bool = True,
) -> np.ndarray:
    """
    integrates an approximate second derivative

    inputs
    -----
        - deriv_approx: second deriv approximation
        - ic: initial conditions
        - eval_times: integration evaluation times. these will be kept, regardless of integration method
        - method: integration method
        - verbose: print progress or not

    returns
    -----
        - integrated, corrected second derivative
    """

    deriv_interp = make_interp_spline(eval_times, deriv_approx)

    # print(ic.shape)
    def dz_hat(t, z):

        if verbose:
            print(
                f"{(t - eval_times[0]) / (eval_times[-1] - eval_times[0]) * 100:0.3f}%,",
                end="\r",
            )

        t = t.detach().cpu().numpy()
        dz2 = torch.from_numpy(np.array([deriv_interp(t)])).to("cuda").to(torch.float32)

        dz1 = z[1]

        return torch.hstack([dz1, dz2])

    eval_times = torch.from_numpy(eval_times)

    with torch.no_grad():
        yhat = odeint_adjoint(
            dz_hat, ic, eval_times, adjoint_params=(), method=method, options=dict()
        ).transpose(0, 1)

    yhat = yhat[0].detach().cpu().numpy().squeeze()
    yhat = correct(yhat)
    return yhat


def eval_model_error(
    dls: dict, model: torch.nn.Module, dt: float, comparison: str = "val"
) -> tuple[tuple[float, float], tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """
    assess second derivative prediction error on the whole dataset. returns
    results in terms of r^2

    inputs
    -----
        - dls: dictionary of torch dataloaders, containing one for train and one for test
        - model: a trained ouroboros
        - dt: audio sampling timestep
        - comparison: val or test. datasegment to evaluate on

    returns
    -----
        (train,test) mean r2
        (train, test) sd r2
        (train, test) all r2
    """

    # tf = model.trend_filtering

    model.eval()

    train_errors = []
    test_errors = []
    preds, reals = [], []
    train_r2 = []
    test_r2 = []
    # model.trend_filtering=False

    for idx, batch in enumerate(dls["train"]):
        with torch.no_grad():
            x, dxdt, dx2dt2 = batch  # each is bsz x seq len x n neurons + 1

            x = x.to("cuda").to(torch.float32)
            dxdt = dxdt.to("cuda").to(torch.float32)
            dx2 = dx2dt2.to("cuda").to(torch.float32) / (dt**2)
            dx2hat, state_pred = model(x, dxdt, dt)  # state: B x L x SD

            # change: scaling to "true" d2y
            dx2hat = (
                dx2hat / model.tau**2
            ) 

            yhat = dx2hat
            y = dx2

            err = sse(yhat, y, reduction="none")  
            err = err.detach().cpu().numpy().squeeze()
            
            tot = sst(y, reduction="none")
            tot = tot.detach().cpu().numpy().squeeze()
            
            train_r2.append(1 - err / tot)

            train_errors.append(err)

    for idx, batch in enumerate(dls[comparison]):
        with torch.no_grad():
            x, dxdt, dx2dt2 = batch  # each is bsz x seq len x n neurons + 1
            

            x = x.to("cuda").to(torch.float32)
            dxdt = dxdt.to("cuda").to(torch.float32)
            dx2 = dx2dt2.to("cuda").to(torch.float32) / (dt**2)
            dx2hat, state_pred = model(x, dxdt, dt)  # state: B x L x SD

            # change: scaling to "true" d2y
            dx2hat = (
                dx2hat / model.tau**2
            )  # * (model.tau*dt)**2 #update to match new tau scaling

            yhat = dx2hat
            # y starts as x[1:0]
            y = dx2

            err = sse(yhat, y, reduction="none")  
            err = err.detach().cpu().numpy().squeeze()
            tot = sst(
                y, reduction="none"
            )  
            tot = tot.detach().cpu().numpy().squeeze()
            assert tot.shape == err.shape
            test_r2.append(1 - err / tot)
            reals.append(y.detach().cpu().numpy().squeeze())
            preds.append(dx2hat.detach().cpu().numpy().squeeze())

            test_errors.append(err)

    mean_r2_train = np.nanmean(np.hstack(train_r2))
    mean_r2_test = np.nanmean(np.hstack(test_r2))
    sd_r2_train = np.nanstd(np.hstack(train_r2))
    sd_r2_test = np.nanstd(np.hstack(test_r2))

    print(f"Train r2: {mean_r2_train} +- {sd_r2_train}")
    print(f"{comparison} r2: {mean_r2_test} +- {sd_r2_test}")
    # model.trend_filtering=tf

    return (
        (mean_r2_train, mean_r2_test),
        (sd_r2_train, sd_r2_test),
        (np.hstack(train_r2), np.hstack(test_r2)),
    )


def pad_with_nan(array: np.ndarray, target_len: int) -> np.ndarray:
    """
    pads an array to a certain length with nans

    inputs
    -----
        - l: array to pad
        - target_len: desired length

    returns
    -----
        - padded array
    """

    l1 = len(array)

    diff = target_len - l1
    if diff > 0:
        return np.hstack(
            [
                array,
                np.nan
                * np.ones(
                    diff,
                ),
            ]
        )
    else:
        return array


def integrate_poly_autonomous(
    model: torch.nn.Module,
    audio: np.ndarray,
    dt: float,
    method: str = "rk4",
    detrend: bool = True,
    noise_sd: float = 0.0,
    noise_gain: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
    return_envelope: bool = False,
    return_source: bool = False,
    return_drives: bool = False,
) -> np.ndarray:
    """
    Fully autonomous (closed-loop) integration of a polynomial `Ouroboros`.

    Precomputes the drives omega(t), gamma(t) and the kernel weights w(t) from `audio` (these are
    low-passed inside `get_funcs` if the model was trained with `drive_lowpass_ms`), then integrates

        dx/ds  = x'
        dx'/ds = -omega(s)^2 x - gamma(s) x' - kernel(x, x'; w(s))

    in the model's rescaled time s = t/tau, feeding the generated state (x, x') back into the
    nonlinearity each step. Only the drives (from data) and the initial condition come from outside.
    """

    L = len(audio)
    t_steps = np.arange(0, L * dt + dt / 2, dt)[:L]
    s_steps = t_steps / model.tau

    # Use the model's device so this works on either CPU or CUDA. Lets the monitor
    # evaluate ckpts on CPU when GPU is contention-saturated by ongoing training.
    dev = next(model.parameters()).device
    audio_3d = audio[None, :, None]
    dy = deriv_approx_dy(audio_3d)
    audio_t = torch.from_numpy(audio_3d).to(torch.float32).to(dev)
    dy_t = torch.from_numpy(dy).to(torch.float32).to(dev)

    with torch.no_grad():
        omega, gamma, _, weights, _ = model.get_funcs(audio_t, dy_t, dt)
    omega = omega.detach().cpu().numpy().squeeze()
    gamma = gamma.detach().cpu().numpy().squeeze()
    weights = weights.detach().cpu().numpy()  # (1, L, P, P)
    _, _, P, P2 = weights.shape
    w_flat = weights.reshape(L, P * P2)

    omega_interp = make_interp_spline(s_steps, omega)
    gamma_interp = make_interp_spline(s_steps, gamma)
    w_interp = make_interp_spline(s_steps, w_flat)  # vector-valued spline over time

    x0 = float(audio[0])
    xp0 = (model.tau / dt) * float(dy[0, 0, 0])
    ic = torch.tensor([x0, xp0], dtype=torch.float32, device=dev)
    kernel = model.kernel

    # learnable amplitude envelope e(s) (all-ones when the model has no envelope) and the
    # forward vocal tract H. The source ODE is rolled out with the PLAIN kernel (no e), then
    # the generated source is multiplied by e(s) and filtered by H -- matching the training
    # path in train.spectral_rollout. Both are identity at init.
    use_tract = getattr(model, "use_tract", False)
    use_env = getattr(model, "use_envelope", False)
    with torch.no_grad():
        e_seq = (
            model.get_envelope(audio_t, dy_t.clone(), dt).detach().cpu().numpy().squeeze()
            if use_env
            else np.ones(L)
        )

    # Learned flow-gated colored-noise forcing (matches training's stochastic-Heun path).
    # g_gate = the ReLU gate g(t) from the sigma head. Computed whenever the model carries the
    # noise head (independent of noise_gain) so it can be returned as a drive and plotted like
    # omega/gamma even in deterministic eval. The OU forcing is only APPLIED when noise_gain > 0
    # (use_learned_noise); noise_tau_c is the OU correlation time in samples.
    noise_on = getattr(model, "enable_noise_forcing", False)
    g_gate = None
    with torch.no_grad():
        _g = model.get_sigma(audio_t, dy_t.clone(), dt)   # non-None iff the sigma head exists
    if _g is not None:                                    # OU forcing OR additive noise branch
        g_gate = np.atleast_1d(_g.detach().cpu().numpy().squeeze()).astype(np.float64)
    use_learned_noise = noise_on and noise_gain > 0
    if use_learned_noise:
        g_seq = g_gate
        noise_tau_c = (model.noise_tau_ms / 1e3) / dt

    def _finish(x_src: np.ndarray) -> np.ndarray:
        """detrend the raw RK4 source FIRST (non-trainable HPF stabilizes the
        integrator output before any downstream scaling), then envelope-scale,
        then filter through the trainable tract. Order matters: env(t) only
        modulates the AC content of the source, and the tract sees a stable
        signal -- no DC ride-through into the formant filter."""
        x_src = np.asarray(x_src, dtype=np.float64)
        if detrend:
            x_src = correct(x_src)
        x_src = x_src * e_seq[: len(x_src)]
        if use_tract:
            xt = torch.from_numpy(x_src[None, :, None]).to(torch.float32).to(dev)
            with torch.no_grad():
                x_src = model.tract.apply(xt).detach().cpu().numpy().squeeze()
        # Harmonic-plus-noise: add the additive filtered-noise branch OUTSIDE the tract, so the
        # autonomous reconstruction matches training (harmonic + noise floor). Gated by noise_gain.
        if getattr(model, "use_noise_branch", False) and noise_gain > 0:
            from train.spectral_rollout import filtered_noise_branch
            Hn = len(x_src)
            gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
            with torch.no_grad():
                nb = filtered_noise_branch(model, audio_t, dy_t.clone(), dt, Hn,
                                           rng=gen).detach().cpu().numpy().squeeze()
            x_src = x_src + noise_gain * np.asarray(nb, dtype=np.float64)[:len(x_src)]
        return x_src

    if use_learned_noise:
        # Learned flow-gated OU forcing, stochastic Heun on (x, v, eta) -- the numpy mirror of
        # train.spectral_rollout._heun_core_factory, with the same soft-tanh clamps. eta is an
        # OU process (correlation time noise_tau_c samples), gated by g_eff = noise_gain*g(t)
        # and added to the momentum equation only. Additive noise => Heun is strong order 1.
        from train.rollout_refine import BX, BXP
        rng = np.random.default_rng(seed)
        tau_c = float(noise_tau_c)
        c2 = (2.0 / tau_c) ** 0.5
        x, xp, eta = x0, xp0, 0.0
        xs = [x]
        ww = weights.reshape(L, 1, 1, P, P2)
        for k in range(L - 1):
            om, ga, wk = omega[k], gamma[k], ww[k]
            g_eff = noise_gain * float(g_seq[k])
            xi = rng.standard_normal()

            def drift(xx, vv, ee):
                xx_c = BX * np.tanh(xx / BX)
                vv_c = BXP * np.tanh(vv / BXP)
                kern = float(
                    kernel.forward_given_weights_numpy(np.array([[[xx_c, vv_c]]]), wk).squeeze()
                )
                return vv, -(om ** 2) * xx - ga * vv - kern + g_eff * ee, -ee / tau_c

            ax, av, ae = drift(x, xp, eta)
            xt, vt, et = x + ax, xp + av, eta + ae + c2 * xi
            axt, avt, aet = drift(xt, vt, et)
            x = x + 0.5 * (ax + axt)
            xp = xp + 0.5 * (av + avt)
            eta = eta + 0.5 * (ae + aet) + c2 * xi
            x = BX * np.tanh(x / BX)
            xp = BXP * np.tanh(xp / BXP)
            xs.append(x)
        src_pre = np.array(xs)
        out = _finish(src_pre)
        rv = (out,)
        if return_envelope:
            rv = rv + (e_seq[: len(out)].copy(),)
        if return_source:
            rv = rv + (src_pre[: len(out)].copy(),)
        if return_drives:
            alpha = weights[0, :, 0, 0].astype(np.float64).copy()
            rv = rv + ({"omega": omega[: len(out)].astype(np.float64).copy(),
                        "gamma": gamma[: len(out)].astype(np.float64).copy(),
                        "alpha": alpha[: len(out)],
                        "sigma": (g_gate[: len(out)].copy() if g_gate is not None else None)},)
        return rv if len(rv) > 1 else rv[0]

    if noise_sd > 0:
        # stochastic forcing (Euler-Maruyama on the velocity) to sustain a noise-driven
        # oscillation at the data amplitude. RK4 drift per sample (drives held constant over
        # the 1-sample step, fine since they are low-passed), plus additive noise on x'.
        rng = np.random.default_rng(seed)
        x, xp = x0, xp0
        xs = [x]
        ww = weights.reshape(L, 1, 1, P, P2)
        for k in range(L - 1):
            om, ga, wk = omega[k], gamma[k], ww[k]

            def f(xx, vv):
                kern = float(kernel.forward_given_weights_numpy(np.array([[[xx, vv]]]), wk).squeeze())
                return vv, -(om**2) * xx - ga * vv - kern

            k1x, k1v = f(x, xp)
            k2x, k2v = f(x + 0.5 * k1x, xp + 0.5 * k1v)
            k3x, k3v = f(x + 0.5 * k2x, xp + 0.5 * k2v)
            k4x, k4v = f(x + k3x, xp + k3v)
            x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6 + noise_sd * rng.standard_normal()
            xs.append(x)
        src_pre = np.array(xs)
        out = _finish(src_pre)
        rv = (out,)
        if return_envelope:
            rv = rv + (e_seq[: len(out)].copy(),)
        if return_source:
            rv = rv + (src_pre[: len(out)].copy(),)
        if return_drives:
            # omega/gamma already extracted at top of fn; alpha is the (0,0)
            # polynomial weight that becomes a constant forcing per timestep.
            alpha = weights[0, :, 0, 0].astype(np.float64).copy()
            rv = rv + ({"omega": omega[: len(out)].astype(np.float64).copy(),
                        "gamma": gamma[: len(out)].astype(np.float64).copy(),
                        "alpha": alpha[: len(out)],
                        "sigma": (g_gate[: len(out)].copy() if g_gate is not None else None)},)
        return rv if len(rv) > 1 else rv[0]

    # Manual RK4 with the same soft-tanh state saturation as train/spectral_rollout.py
    # (BX, BXP). The previous odeint_adjoint path ran the bare polynomial ODE with no
    # state bound, so any transient that pushed |y| past ~1 (where the 15th-order kernel
    # terms dominate) blew up. The training rollout already clamps both states after
    # each substep; doing the same here closes the train/val gap and stops the
    # autonomous integrator from diverging when the trajectory grazes the edge of the
    # trained region.
    from train.rollout_refine import BX, BXP
    x, xp = x0, xp0
    xs = [x]
    ww = weights.reshape(L, 1, 1, P, P2)
    for k in range(L - 1):
        om, ga, wk = omega[k], gamma[k], ww[k]

        def f(xx, vv):
            # Clamp BEFORE the kernel eval so the 16th-order polynomial is never
            # evaluated outside the trained region, even within an RK4 substep.
            # Without this, the inter-substep extrapolation (xc + 0.5*k1x) can push
            # the polynomial input arbitrarily far past BX/BXP, overflow numpy
            # float64, and produce NaN that propagates past the end-of-step tanh
            # clamp (since tanh(NaN)=NaN). Linear -om^2*xx and -ga*vv terms stay
            # on the unclamped state to preserve standard RK4 semantics.
            xx_c = BX * np.tanh(xx / BX)
            vv_c = BXP * np.tanh(vv / BXP)
            kern = float(
                kernel.forward_given_weights_numpy(np.array([[[xx_c, vv_c]]]), wk).squeeze()
            )
            return vv, -(om ** 2) * xx - ga * vv - kern

        k1x, k1v = f(x, xp)
        k2x, k2v = f(x + 0.5 * k1x, xp + 0.5 * k1v)
        k3x, k3v = f(x + 0.5 * k2x, xp + 0.5 * k2v)
        k4x, k4v = f(x + k3x, xp + k3v)
        x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        xp = xp + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        x = BX * np.tanh(x / BX)
        xp = BXP * np.tanh(xp / BXP)
        xs.append(x)
    src_pre = np.array(xs)
    out = _finish(src_pre)
    rv = (out,)
    if return_envelope:
        rv = rv + (e_seq[: len(out)].copy(),)
    if return_source:
        rv = rv + (src_pre[: len(out)].copy(),)
    if return_drives:
        alpha = weights[0, :, 0, 0].astype(np.float64).copy()
        rv = rv + ({"omega": omega[: len(out)].astype(np.float64).copy(),
                    "gamma": gamma[: len(out)].astype(np.float64).copy(),
                    "alpha": alpha[: len(out)],
                    "sigma": (g_gate[: len(out)].copy() if g_gate is not None else None)},)
    return rv if len(rv) > 1 else rv[0]


def autonomy_score(
    model: torch.nn.Module,
    segments: list,
    dt: float,
    fmax: float = 8000.0,
    w_amp: float = 1.0,
    w_pitch: float = 1.0,
    diverge_score: float = -5.0,
    method: str = "rk4",
    rescale: bool = False,
    cold_start: bool = False,
    return_trajectories: bool = False,
    noise_gain: float = 0.0,
) -> tuple:
    """
    Validation metric for AUTONOMOUS reconstruction quality (model-selection criterion).

    For each pre-windowed (sustained) vocalization segment, run the model's autonomous
    integration and score it against the target by phase-robust spectral (log-PSD)
    correlation, minus log-ratio penalties on amplitude and pitch:

    `noise_gain` (default 0.0 = deterministic) is forwarded to integrate_poly_autonomous: when
    >0 and the model has the flow-gated OU forcing (enable_noise_forcing), the reconstruction is
    the NOISE-ON output (the model's actual generative signal), not the deterministic backbone.
    Ignored for models without the noise head. Both the returned trajectories (for spectrograms)
    and the score then reflect the noise-driven output.

        score = spectral_corr(autonomous, target)
                - w_amp   * |log(std_auto   / std_target)|
                - w_pitch * |log(pitch_auto / pitch_target)|

    A divergent / collapsed rollout (non-finite or ~zero) gets `diverge_score`. Poly-only.

    If `rescale=True`, the autonomous output's RMS is matched to the target's before scoring (the
    deployed recipe -- amplitude is an arbitrary overall-scale gauge fixed at generation by
    `generate_autonomous`). This zeroes `amp_pen` for bounded rollouts, so selection then turns on
    the genuinely-constrained quantities (spectral shape + pitch + boundedness). A divergent rollout
    is still detected on the RAW output and gets `diverge_score` (collapse is not rescaled away).

    If `cold_start=True`, the caller is asserting that each `segments[i]` already includes a silence
    lead-in. The integration IC is the first sample of the segment, which then equals near-silence
    and exercises the ignition path -- the situation our real-time synthesis target actually faces.
    Mechanically the metric is unchanged; this flag exists so callers can request the cold-start
    scoring contract and so the chosen mode is recorded in the breakdown. Typically paired with
    `rescale=False` (amplitude must contribute to the score, because a single shipped rescale
    constant cannot fix voc-variable cold-start amplitude).

    returns
    -----
        - mean score over segments
        - per-segment scores (list)
        - breakdown dict (mean spectral corr, amp penalty, pitch penalty, bounded fraction,
          plus the `cold_start` and `rescale` flags for downstream logging)
    """
    fs = 1.0 / dt

    def _logpsd(x):
        f, P = welch(x - np.mean(x), fs=fs, nperseg=min(1024, len(x)))
        m = f <= fmax
        return np.log(P[m] + 1e-20)

    def _peak(x):
        f, P = welch(x - np.mean(x), fs=fs, nperseg=min(1024, len(x)))
        P[0] = 0
        return float(f[np.argmax(P)])

    scores, specs, amps, pits, bounded = [], [], [], [], []
    # Signed amp = log(auto_rms / target_rms). Positive = loud, negative = quiet.
    # amp_pen is |signed_amp|, so this gives direction at no extra integration cost.
    # The "raw" version uses the unrescaled auto_n, so the sign reflects what the
    # autonomous integrator actually produced (not what the deployed rescale recipe
    # would emit).
    signed_amps_raw = []
    # If `return_trajectories`, capture (tgt_n, auto_n_pre_rescale) per segment so the
    # caller can render audio / spectrograms without a separate integration pass.
    trajectories = [] if return_trajectories else None
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        tgt = correct(seg)
        if return_trajectories:
            auto, env, src, drives = integrate_poly_autonomous(
                model, seg, dt, method=method, noise_sd=0.0, noise_gain=noise_gain,
                detrend=True, verbose=False, return_envelope=True,
                return_source=True, return_drives=True)
        else:
            auto = integrate_poly_autonomous(model, seg, dt, method=method, noise_sd=0.0,
                                             noise_gain=noise_gain,
                                             detrend=True, verbose=False)
            env = src = drives = None
        n = min(len(tgt), len(auto))
        tgt_n, auto_n = tgt[:n], auto[:n]
        if return_trajectories:
            env_n = env[:n] if env is not None else np.ones(n)
            src_n = src[:n] if src is not None else auto_n.copy()
            drives_n = ({k: (v[:n].copy() if v is not None else None) for k, v in drives.items()}
                        if drives is not None else None)
            trajectories.append((tgt_n.copy(), auto_n.copy(), env_n.copy(),
                                 src_n.copy(), drives_n))
        if (not np.isfinite(auto_n).all()) or np.nanstd(auto_n) < 1e-9:
            scores.append(diverge_score)
            bounded.append(0.0)
            specs.append(np.nan); amps.append(np.nan); pits.append(np.nan)
            signed_amps_raw.append(float("nan"))
            continue
        bounded.append(1.0)
        # Capture the SIGNED amp BEFORE optional rescaling so we know which direction
        # the raw autonomous rollout drifts.
        signed_amp_raw = float(np.log((np.nanstd(auto_n) + 1e-12) / (np.nanstd(tgt_n) + 1e-12)))
        signed_amps_raw.append(signed_amp_raw)
        if rescale:  # gauge-fix amplitude to the target RMS (the deployed recipe)
            auto_n = auto_n * (np.nanstd(tgt_n) / (np.nanstd(auto_n) + 1e-12))
        sc = float(np.corrcoef(_logpsd(tgt_n), _logpsd(auto_n))[0, 1])
        amp = abs(np.log((np.nanstd(auto_n) + 1e-12) / (np.nanstd(tgt_n) + 1e-12)))
        pit = abs(np.log((_peak(auto_n) + 1e-9) / (_peak(tgt_n) + 1e-9)))
        scores.append(sc - w_amp * amp - w_pitch * pit)
        specs.append(sc); amps.append(amp); pits.append(pit)

    breakdown = {
        "spec_corr": float(np.nanmean(specs)) if len(specs) else float("nan"),
        "amp_pen": float(np.nanmean(amps)) if len(amps) else float("nan"),
        "pitch_pen": float(np.nanmean(pits)) if len(pits) else float("nan"),
        "bounded_frac": float(np.mean(bounded)) if len(bounded) else 0.0,
        "cold_start": bool(cold_start),
        "rescale": bool(rescale),
        # Signed amp_pen (loud/quiet direction) -- mean over vocs and per-voc list.
        "signed_amp_mean": (float(np.nanmean(signed_amps_raw))
                            if any(np.isfinite(signed_amps_raw)) else float("nan")),
        "signed_amp_per_voc": [float(v) for v in signed_amps_raw],
    }
    if return_trajectories:
        return float(np.mean(scores)), scores, breakdown, trajectories
    return float(np.mean(scores)), scores, breakdown


def generate_autonomous(
    model: torch.nn.Module,
    audio: np.ndarray,
    dt: float,
    rescale: bool = True,
    ref_rms: float = None,
    method: str = "rk4",
    detrend: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    DEPLOYED autonomous generation: closed-loop integration + amplitude rescaling.

    Autonomous amplitude is a poorly-constrained, marginal direction (the transverse Floquet
    exponent is left free by on-orbit teacher-forced fitting), so the raw free-running amplitude
    decays/grows/varies by seed. The overall scale is an arbitrary audio-unit gauge, so we fix it
    at generation: run the deterministic closed-loop rollout, then (if `rescale`) match its RMS to
    a reference -- `ref_rms` if given, else the input window's own (detrended) RMS for reconstruction.

    Poly-only. Returns the (rescaled) generated waveform; a divergent/collapsed rollout is returned
    unscaled.
    """
    auto = integrate_poly_autonomous(model, audio, dt, method=method, detrend=detrend,
                                     noise_sd=0.0, verbose=verbose)
    if rescale and np.isfinite(auto).all() and np.nanstd(auto) > 1e-9:
        target = ref_rms if ref_rms is not None else float(np.nanstd(correct(np.asarray(audio, dtype=np.float64))))
        auto = auto * (target / (np.nanstd(auto) + 1e-12))
    return auto
