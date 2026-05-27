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


def integrate_model_autonomous(
    model: torch.nn.Module,
    audio: np.ndarray,
    dt: float,
    method: str = "rk4",
    detrend: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    fully autonomous (closed-loop) integration of an `ArneodoOuroboros` model.

    Unlike `integrate_model_d2`, which replays the model's predicted second derivative
    evaluated at the *data* points, this integrates the biomechanical syrinx ODE while
    feeding the last generated state (x, x') back into the right-hand side. The state is
    therefore generated entirely by the integrator -- only the control time series
    alpha(t), beta(t) (produced once by the encoder from `audio`, as in the paper's
    neurally-driven synthesis), the scalar gamma, and the initial condition come from
    outside the integrator.

    We integrate in the model's rescaled time s = t / tau, where the state is z = [x, x']
    with x' = dx/ds, gamma is the learned (rescaled) scalar `model.gamma`, and

        dx/ds  = x'
        dx'/ds = g^2 a + g^2 b x + g^2 x^2 - g^2 x^3 - g (d + x + x^2) x'

    with g = gamma, a = alpha(s), b = beta(s), d = delta(s) (the linear-damping series).

    inputs
    -----
        - model: a trained ArneodoOuroboros
        - audio: 1-D audio segment used to produce alpha(t), beta(t) and the IC
        - dt: audio sampling spacing (seconds)
        - method: integration method (passed to torchdiffeq)
        - detrend: whether to low-pass detrend the generated waveform (as in `correct`)
        - verbose: print integration progress

    returns
    -----
        - the autonomously generated waveform x, sampled at the same points as `audio`
    """

    L = len(audio)
    t_steps = np.arange(0, L * dt + dt / 2, dt)[:L]
    s_steps = t_steps / model.tau  # rescaled time s = t / tau

    audio_3d = audio[None, :, None]
    dy = deriv_approx_dy(audio_3d)  # per-sample first derivative dx/dn

    audio_t = torch.from_numpy(audio_3d).to(torch.float32).to("cuda")
    dy_t = torch.from_numpy(dy).to(torch.float32).to("cuda")

    with torch.no_grad():
        alpha, beta, delta, gamma = model.get_funcs(audio_t, dy_t, dt)

    alpha = alpha.detach().cpu().numpy().squeeze()
    beta = beta.detach().cpu().numpy().squeeze()
    delta = delta.detach().cpu().numpy().squeeze()
    gamma = float(gamma.detach().cpu().numpy())
    g2 = gamma**2

    # control parameters as smooth functions of rescaled time
    alpha_interp = make_interp_spline(s_steps, alpha)
    beta_interp = make_interp_spline(s_steps, beta)
    delta_interp = make_interp_spline(s_steps, delta)

    # initial condition in rescaled time: x(0) and x'(0) = dx/ds = (tau/dt) * dx/dn
    x0 = float(audio[0])
    xp0 = (model.tau / dt) * float(dy[0, 0, 0])
    ic = torch.tensor([x0, xp0], dtype=torch.float32, device="cuda")

    def dz_hat(s, z):
        if verbose:
            print(
                f"{(s - s_steps[0]) / (s_steps[-1] - s_steps[0]) * 100:0.3f}%,",
                end="\r",
            )

        s_np = s.detach().cpu().numpy()
        a = float(alpha_interp(s_np))
        b = float(beta_interp(s_np))
        d = float(delta_interp(s_np))

        x = z[0]
        xp = z[1]

        dx = xp
        dxp = (
            g2 * a
            + g2 * b * x
            + g2 * x**2
            - g2 * x**3
            - gamma * d * xp
            - gamma * x * xp
            - gamma * x**2 * xp
        )

        return torch.hstack([dx.reshape(1), dxp.reshape(1)])

    eval_times = torch.from_numpy(s_steps).to(ic.device)

    with torch.no_grad():
        sol = odeint_adjoint(
            dz_hat, ic, eval_times, adjoint_params=(), method=method, options=dict()
        ).transpose(0, 1)

    x_gen = sol[0].detach().cpu().numpy().squeeze()
    if detrend:
        x_gen = correct(x_gen)
    return x_gen


def integrate_poly_autonomous(
    model: torch.nn.Module,
    audio: np.ndarray,
    dt: float,
    method: str = "rk4",
    detrend: bool = True,
    noise_sd: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    fully autonomous (closed-loop) integration of a polynomial `Ouroboros`.

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

    audio_3d = audio[None, :, None]
    dy = deriv_approx_dy(audio_3d)
    audio_t = torch.from_numpy(audio_3d).to(torch.float32).to("cuda")
    dy_t = torch.from_numpy(dy).to(torch.float32).to("cuda")

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
    ic = torch.tensor([x0, xp0], dtype=torch.float32, device="cuda")
    kernel = model.kernel

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
        x_gen = np.array(xs)
        return correct(x_gen) if detrend else x_gen

    def dz_hat(s, z):
        if verbose:
            print(
                f"{(s - s_steps[0]) / (s_steps[-1] - s_steps[0]) * 100:0.3f}%,",
                end="\r",
            )
        s_np = s.detach().cpu().numpy()
        om = float(omega_interp(s_np))
        ga = float(gamma_interp(s_np))
        w = w_interp(s_np).reshape(1, 1, P, P2)
        x = float(z[0])
        xp = float(z[1])
        kern = float(
            kernel.forward_given_weights_numpy(np.array([[[x, xp]]]), w).squeeze()
        )
        dxp = -(om**2) * x - ga * xp - kern
        return torch.tensor([xp, dxp], dtype=torch.float32, device=z.device)

    eval_times = torch.from_numpy(s_steps).to(ic.device)
    with torch.no_grad():
        sol = odeint_adjoint(
            dz_hat, ic, eval_times, adjoint_params=(), method=method, options=dict()
        ).transpose(0, 1)

    x_gen = sol[0].detach().cpu().numpy().squeeze()
    if detrend:
        x_gen = correct(x_gen)
    return x_gen


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


