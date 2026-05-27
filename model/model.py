import math

import torch
from torch import nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig
from model.model_utils import smooth
from typing import Tuple, Union
# from model.kernels import *

import numpy as np
from tqdm import tqdm

# plt.rcParams['text.usetex'] = True
import gc

"""
these classes are the body of the Ouroboros method. An Ouroboros consists of three parallel Mamba encoders:
an Omega, a Gamma, and a kernel encoder. The output of each encoder is linearly mapped to our latent features.
These features are then used to reconstruct the estimated second derivative of the audio. See figure 1 of our 
paper for a visual illustration of the procedure.

Models are implemented using the mamba.py package (https://github.com/alxndrTL/mamba.py/),
with elements used included in `third_party` for convenience.

"""


class Ouroboros(nn.Module):
    def __init__(
        self,
        d_data: int,
        kernel: nn.Module,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 1,
        device: str = "cuda",
        tau: float = 1 / 10000,
        smooth_len: float = 0.001,
        drive_lowpass_ms: float = 0.0,
    ):

        super().__init__()

        self.device = device
        ## if stacking on data dimension, d should be 4* d_data (y,dy,rev y, rev dy)
        ## if stacking on time dimension, d should be 2*d_data (y,dy). We are stacking on the time dimension.
        omegaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        gammaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.omega_mamba = Mamba(omegaConfig).to(device)
        self.gamma_mamba = Mamba(gammaConfig).to(device)

        self.omega_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained
        self.gamma_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained

        kernelConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.kernel_mamba = Mamba(kernelConfig).to(device)

        self.tau = tau
        self.smooth_len = smooth_len
        # if > 0, low-pass the drives omega(t), gamma(t), and the kernel weights w(t) with a
        # zero-phase Gaussian (sigma = drive_lowpass_ms) -- "low-pass in the loop". Matches the
        # mechanism added to ArneodoOuroboros so the drives stay slow / control-rate.
        self.drive_lowpass_ms = drive_lowpass_ms
        self.kernel = kernel
        self.kernel.tau = self.tau
        self.names = [r"$\omega$", r"$\gamma$", "weighted kernels", "states"]

    def _lowpass(self, x: torch.FloatTensor, dt: float) -> torch.FloatTensor:
        """centered zero-phase Gaussian low-pass along time of a (B, L, C) control series."""
        sigma = (self.drive_lowpass_ms / 1e3) / dt  # samples
        if sigma <= 0:
            return x
        radius = max(1, int(round(3 * sigma)))
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kern = torch.exp(-0.5 * (t / sigma) ** 2)
        kern = kern / kern.sum()
        C = x.shape[-1]
        k = kern.view(1, 1, -1).expand(C, 1, -1)
        xc = F.pad(x.transpose(1, 2), (radius, radius), mode="reflect")
        return F.conv1d(xc, k, groups=C).transpose(1, 2)

    def _lowpass_weights(self, weights: torch.FloatTensor, dt: float) -> torch.FloatTensor:
        """low-pass the polynomial kernel weights (B, L, P, P) along time."""
        B, L, P, P2 = weights.shape
        w = self._lowpass(weights.reshape(B, L, P * P2), dt)
        return w.reshape(B, L, P, P2)

    def forward(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        predicts second derivative at time t.
        all other predictions should be done in the train loop (train_utils.py)

        inputs
        ------
            - x: cleaned audio segment
            - dxdt: first derivative estimate, scaled by sample interval dt
            - dt: sample interval
            - smoothing: whether we smooth functions during training. We do not, but you can

        outputs
        ------
            - yhat: model predicted second derivative, scaled by model time constant tau^2
            - weights: kernel weights over whole segment. these are regularized towards simplicity during training
        """

        dxdt *= self.tau  # this is now \tau dxdt

        dxdt /= dt  # this is now \tau dx

        B, L, D = x.shape

        # x: x_0, x_dt, x_2dt,...
        smooth_len = int(round(self.smooth_len / dt))

        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]

        # we feed the audio and its first derivative, along with a time-reversed version,
        # to each encoder
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)  # stack on time dimension

        omegaControl = self.omega_mamba(x_in)[:, L:, :]
        gammaControl = self.gamma_mamba(x_in)[:, L:, :]
        kernelControl = self.kernel_mamba(x_in)[:, L:, :]

        omega = self.omega_net(
            omegaControl
        ).abs()  # Since we take omega^2 anyway, we take the absolute value to prevent things from switching around too much
        gamma = self.gamma_net(gammaControl)
        weighted_kernels, weights = self.kernel(z, kernelControl)
        if self.drive_lowpass_ms > 0:
            # low-pass the drives in the loop, then recompute the nonlinearity from the
            # low-passed weights so yhat is consistent with the (slow) drives
            omega = self._lowpass(omega, dt)
            gamma = self._lowpass(gamma, dt)
            weights = self._lowpass_weights(weights, dt)
            weighted_kernels = self.kernel.forward_given_weights(z, weights)
        elif smoothing:
            # smooth our model functions, if we choose to do so. I do not.
            omega = smooth(omega, smooth_len)
            gamma = smooth(gamma, smooth_len)

        z1 = z[:, :, :1]
        z2 = z[:, :, 1:]

        yhat = -(omega**2) * z1 - gamma * z2 - weighted_kernels

        return yhat, weights

    def get_funcs(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
        max_len_t: int = 4,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        Union[list, torch.FloatTensor],
    ]:
        """
        given data, returns learned model functions --- latent features

        inputs
        ------
            - x: audio segment
            - dxdt: first derivative estimate, scaled by timestep dt
            - dt: sampling timestep
            - smoothing: whether to smooth model functions. Here, we typically smooth omega & gamma
            - max_len_t: maximum audio length that we will process simultaneously. triggers sequential processing, if x is too long

        returns
        ------
            - omega: instantaneous frequency term
            - gamma: instantaneous damping termp
            - weighted_kernels: nonlinearity term
            - weights: weights on kernels in nonlinearity
            - states: control signals for omega, gamma, kernels. I typically do not use these.
        """

        B, L, D = x.shape

        ## as in forward
        dxdt *= self.tau
        dxdt /= dt

        max_len_s = int(round(max_len_t / dt))

        smooth_len = int(
            round(self.smooth_len / dt)
        )  # convert smooth len from seconds to samples

        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]
        if L > max_len_s:
            # if a function is too long, use sequential processing to retrieve latent features
            print("using step by step functions")
            yhat, omega, gamma, weighted_kernels, weights = self.funcs_by_step(
                z, dt, smoothing=smoothing, step_size=max_len_s
            )  # update for chunked steps
            return omega, gamma, weighted_kernels, weights, []

        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)  ## stack on time dimension

        omegaControl = self.omega_mamba(x_in)[:, L:, :]
        gammaControl = self.gamma_mamba(x_in)[:, L:, :]
        kernelControl = self.kernel_mamba(x_in)[:, L:, :]

        omega = self.omega_net(omegaControl).abs()
        gamma = self.gamma_net(gammaControl)
        weighted_kernels, weights = self.kernel(z, kernelControl)

        if self.drive_lowpass_ms > 0:
            omega = self._lowpass(omega, dt)
            gamma = self._lowpass(gamma, dt)
            weights = self._lowpass_weights(weights, dt)
            weighted_kernels = self.kernel.forward_given_weights(z, weights)
        elif smoothing:
            omega = smooth(omega.abs(), smooth_len)
            gamma = smooth(gamma, smooth_len)

        return (
            omega,
            gamma,
            weighted_kernels,
            weights,
            torch.cat([omegaControl, gammaControl, kernelControl], dim=-1),
        )

    def integrate(
        self,
        x,
        dxdt,
        dx2,
        dt,
        method="RK45",
        st=0.05,
        scaled=True,
        with_residual=False,
        smoothing=True,
        strategy="interp",
        oversample_prop=1,
    ):
        """
        don't use this.
        """
        print(
            "Don't use this to integrate. Instead, use the integration methods in train/eval.py"
        )
        return

    def funcs_by_step(
        self,
        z: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
        step_size: int = 10000,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """
        if your audio segment is too long, we'll just get these functions step by step
        rather than all at once. This function assumes you do NOT need to backprop at all. That would probably take forever.
        So I would recommend keeping that as is.

        inputs
        ------
            - z: stacked x, \tau dx -- input to mamba models
            - dt: sampling timestep
            - smoothing: whether we smooth the latent features
            - step_size: maximum number of samples to run through the model at once

        returns
        ------
            - yhat: model estimated second derivative
            - omegas: instantaneous frequency
            - gammas: instantaneous damping
            - kernel: nonlinearity
            - weights: kernel weights for nonlinearity
        """

        smooth_len = int(round(self.smooth_len / dt))

        omega_cache = [
            (
                None,
                torch.zeros(
                    (
                        1,
                        self.omega_mamba.config.d_model
                        * self.omega_mamba.config.expand_factor,
                        self.omega_mamba.config.d_conv,
                    ),
                    device=self.device,
                ),
            )
            for _ in self.omega_mamba.layers
        ]
        gamma_cache = [
            (
                None,
                torch.zeros(
                    (
                        1,
                        self.gamma_mamba.config.d_model
                        * self.gamma_mamba.config.expand_factor,
                        self.gamma_mamba.config.d_conv,
                    ),
                    device=self.device,
                ),
            )
            for _ in self.gamma_mamba.layers
        ]
        weights_cache = [
            (
                None,
                torch.zeros(
                    (
                        1,
                        self.kernel_mamba.config.d_model
                        * self.kernel_mamba.config.expand_factor,
                        self.kernel_mamba.config.d_conv,
                    ),
                    device=self.device,
                ),
            )
            for _ in self.kernel_mamba.layers
        ]

        B, L, D = z.shape
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)
        # B,L,D = x_in.shape
        L_new = x_in.shape[1]

        omegas, gammas, weights, kernel = [], [], [], []
        with torch.no_grad():
            for ii in tqdm(
                range(0, L_new, step_size),
                total=L_new // step_size + 1,
                desc=f"iterating through segment of length {L}",
            ):
                if np.mod(ii, 10000) == 0:
                    ## clean up stuff every so often
                    gc.collect()
                end_ind = min(L_new, ii + step_size)
                s = x_in[:, ii:end_ind, :]

                omega, omega_cache = self.omega_mamba.step(s, omega_cache)
                gamma, gamma_cache = self.omega_mamba.step(s, gamma_cache)
                w, weights_cache = self.omega_mamba.step(s, weights_cache)

                start_ind = max(0, ii - L)
                s = z[:, start_ind : end_ind - L, :]
                if end_ind - L > 0:
                    start_ind = max(0, L - ii)
                    omega = self.omega_net(omega[:, start_ind:]).abs()
                    gamma = self.gamma_net(gamma[:, start_ind:])
                    weights.append(w[:, start_ind:].detach().cpu().numpy().squeeze())
                    omegas.append(omega.detach().cpu().numpy().squeeze())
                    gammas.append(gamma.detach().cpu().numpy().squeeze())
                    weighted_kernels, _ = self.kernel(s, w[:, start_ind:], smooth_len)
                    kernel.append(weighted_kernels.detach().cpu().numpy().squeeze())

        z[:, :, 1] /= dt
        omegas = np.concatenate(omegas)
        gammas = np.concatenate(gammas)
        weights = np.concatenate(weights)
        kernel = np.concatenate(kernel)

        z1 = z[:, :, :1].detach().cpu().numpy().squeeze() / dt
        z2 = z[:, :, 1:].detach().cpu().numpy().squeeze()

        yhat = -(omegas**2) * z1 - gammas * z2 - kernel

        if smoothing:
            omegas = smooth(omegas[None, :, None], smooth_len).squeeze()
            gammas = smooth(gammas[None, :, None], smooth_len).squeeze()

        return yhat, omegas, gammas, kernel, weights


class ArneodoOuroboros(nn.Module):
    """
    Variant of `Ouroboros` that parameterizes the second derivative using the
    biomechanical syrinx ODE of Arneodo et al. 2021 (Current Biology) instead of a
    full polynomial kernel. In physical time the (paper) model reads:

        dx/dt = y
        dy/dt = gamma^2 alpha + gamma^2 beta x + gamma^2 x^2
                - gamma^2 x^3 - gamma x y - gamma x^2 y

    where x is the labial displacement, gamma is a (constant) time-scaling factor,
    and alpha, beta are the two bird-controlled parameters (sub-syringeal pressure
    and syringeal-muscle tension).

    This implementation additionally includes a *linear* damping term -gamma * delta * y
    (delta(t) a third control series), which the strict paper polynomial lacks. It lets the
    model represent sources with a linear damping/anti-damping term -- e.g. the Mindlin/gabo
    data generator's `B * xdot` (van-der-Pol-style) term. delta can be either sign; delta < 0
    is anti-damping (pumps energy, sustaining oscillation). Three parallel Mamba encoders
    output alpha(t), beta(t), delta(t); gamma is a single learned positive scalar
    (`self.gamma`), as in the paper where gamma is constant. delta's head is zero-initialized,
    so the model *starts* at delta == 0 (the strict paper form) and only learns linear
    damping if it reduces the loss -- the extended form is a strict superset of the paper one.

    As in `Ouroboros`, the model works in rescaled time s = t/tau: `forward` scales the
    input first derivative to x' = dx/ds = (tau/dt) * dxdt, and the training target is
    d2x/ds2 = tau^2 * d2x/dt2. Substituting s = t/tau into the ODE above (the linear term
    -gamma*delta*y rescales to -g*delta*x', matching the other damping terms) leaves the
    form identical with a rescaled scalar g = tau * gamma, so this is a drop-in replacement
    for the polynomial RHS and the existing training target scaling is unchanged:

        d2x/ds2 = g^2 alpha + g^2 beta x + g^2 x^2 - g^2 x^3 - g (delta + x + x^2) x'

    `self.gamma` is exactly this rescaled g. (With tau = dt, g = dt * gamma_phys, an O(1)
    learnable scalar.)
    """

    def __init__(
        self,
        d_data: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 1,
        device: str = "cuda",
        tau: float = 1 / 10000,
        smooth_len: float = 0.001,
        gamma_init: float = 1.0,
        drive_lowpass_ms: float = 0.0,
    ):

        super().__init__()

        self.device = device

        # we stack x and (scaled) dx on the time dimension, so d_model = 2 * d_data
        alphaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        betaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )
        deltaConfig = MambaConfig(
            d_model=2 * d_data,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        self.alpha_mamba = Mamba(alphaConfig).to(device)
        self.beta_mamba = Mamba(betaConfig).to(device)
        self.delta_mamba = Mamba(deltaConfig).to(device)

        self.alpha_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained: alpha can be either sign
        self.beta_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained: beta can be either sign
        self.delta_net = nn.Linear(
            in_features=2 * d_data, out_features=d_data, device=device
        )  # output unconstrained: delta can be either sign (negative => anti-damping)

        # Zero-initialize the control heads so the model starts at alpha = beta = delta = 0,
        # i.e. yhat = g^2 (x^2 - x^3) - g (x + x^2) x' (bounded, structural terms only --
        # exactly the strict-paper-form dynamics). Without this the heads emit O(1) forcing
        # that is ~10-100x the target second-derivative scale, which makes the loss explode
        # and conditioning poor at the start of training. Because delta is also zero-init,
        # the extended (linear-damping) form is a strict superset of the paper form: the
        # model starts strict and learns linear damping only if it reduces the loss.
        for net in (self.alpha_net, self.beta_net, self.delta_net):
            nn.init.zeros_(net.weight)
            nn.init.zeros_(net.bias)

        # gamma (rescaled time-scaling factor) is a single positive scalar, learned as
        # softplus(log_gamma). Initialize log_gamma so that softplus(log_gamma) ~ gamma_init.
        inv_softplus = math.log(math.expm1(gamma_init))  # inverse of softplus
        self.log_gamma = nn.Parameter(torch.tensor(float(inv_softplus), device=device))

        self.tau = tau
        self.smooth_len = smooth_len
        # if >0, hard-constrain the control series alpha/beta/delta to vary no faster than
        # this timescale (ms) by low-pass filtering the Mamba heads' outputs (always, train
        # + eval). Encodes a known physiological control-rate prior and prevents the drives
        # from carrying carrier-frequency content. Gaussian sigma = drive_lowpass_ms.
        self.drive_lowpass_ms = drive_lowpass_ms
        self.names = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$"]

    @property
    def gamma(self) -> torch.FloatTensor:
        """rescaled, strictly-positive time-scaling scalar g = softplus(log_gamma)."""
        return F.softplus(self.log_gamma)

    def _lowpass(self, x: torch.FloatTensor, dt: float) -> torch.FloatTensor:
        """
        centered, zero-phase Gaussian low-pass along time (sigma = self.drive_lowpass_ms),
        applied to a control series x of shape (B, L, 1). Differentiable (fixed kernel);
        reflection-padded to avoid edge artifacts / phase delay.
        """
        sigma = (self.drive_lowpass_ms / 1e3) / dt  # samples
        if sigma <= 0:
            return x
        radius = max(1, int(round(3 * sigma)))
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kern = torch.exp(-0.5 * (t / sigma) ** 2)
        kern = (kern / kern.sum()).view(1, 1, -1)
        xc = F.pad(x.transpose(1, 2), (radius, radius), mode="reflect")
        return F.conv1d(xc, kern).transpose(1, 2)

    def _encode(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        shared input prep + encoders. Returns (alpha, beta, delta, z) where z = [x, x'] is
        the state in rescaled time (x' = dx/ds = (tau/dt) * dxdt).
        """

        # scale first derivative to rescaled time: x' = dx/ds = (tau/dt) * dxdt.
        # use out-of-place ops so we never mutate the caller's tensor.
        dxdt = dxdt * (self.tau / dt)

        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]

        # feed the state and its time-reversed copy to each encoder, then keep the
        # forward-time half (matches Ouroboros.forward)
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)

        alphaControl = self.alpha_mamba(x_in)[:, L:, :]
        betaControl = self.beta_mamba(x_in)[:, L:, :]
        deltaControl = self.delta_mamba(x_in)[:, L:, :]

        alpha = self.alpha_net(alphaControl)
        beta = self.beta_net(betaControl)
        delta = self.delta_net(deltaControl)

        if self.drive_lowpass_ms > 0:
            # hard timescale constraint: low-pass the drives (always, train + eval)
            alpha = self._lowpass(alpha, dt)
            beta = self._lowpass(beta, dt)
            delta = self._lowpass(delta, dt)
        elif smoothing:
            smooth_len = int(round(self.smooth_len / dt))
            alpha = smooth(alpha, smooth_len)
            beta = smooth(beta, smooth_len)
            delta = smooth(delta, smooth_len)

        return alpha, beta, delta, z

    def _rhs(
        self,
        alpha: torch.FloatTensor,
        beta: torch.FloatTensor,
        delta: torch.FloatTensor,
        z: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        the Arneodo RHS in rescaled time, given alpha, beta, delta and state z = [x, x'].
        returns the predicted second derivative d2x/ds2. The damping is
        -g (delta + x + x^2) x': the (delta) term is the linear-damping extension, the
        (x + x^2) terms are the strict paper nonlinear damping.
        """
        g = self.gamma
        x = z[:, :, :1]
        xp = z[:, :, 1:]
        g2 = g * g
        return (
            g2 * alpha
            + g2 * beta * x
            + g2 * x**2
            - g2 * x**3
            - g * delta * xp
            - g * x * xp
            - g * x**2 * xp
        )

    def forward(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        predicts the second derivative at time t via the Arneodo ODE.

        inputs
        ------
            - x: cleaned audio segment (B, L, d_data)
            - dxdt: first derivative estimate (unscaled; scaled internally by tau/dt)
            - dt: sample interval
            - smoothing: whether to smooth alpha, beta. We do not, but you can

        outputs
        ------
            - yhat: predicted second derivative, scaled by tau^2 (i.e. d2x/ds2)
            - alpha: the alpha(t) time series. Returned in the second slot (where the
              polynomial model returns kernel weights) so the train loop signature
              matches; it is NOT regularized (train with reg_weights=False).
        """

        alpha, beta, delta, z = self._encode(x, dxdt, dt, smoothing=smoothing)
        yhat = self._rhs(alpha, beta, delta, z)
        return yhat, alpha

    def get_funcs(
        self,
        x: torch.FloatTensor,
        dxdt: torch.FloatTensor,
        dt: float,
        smoothing: bool = False,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """
        given data, returns the learned model functions for the Arneodo parameterization.

        inputs
        ------
            - x: audio segment (B, L, d_data)
            - dxdt: first derivative estimate (unscaled)
            - dt: sampling timestep
            - smoothing: whether to smooth alpha, beta, delta

        returns
        ------
            - alpha: alpha(t) control time series (B, L, d_data)
            - beta: beta(t) control time series (B, L, d_data)
            - delta: delta(t) linear-damping control time series (B, L, d_data)
            - gamma: the learned scalar g (rescaled time-scaling factor)
        """

        alpha, beta, delta, _ = self._encode(x, dxdt, dt, smoothing=smoothing)
        return alpha, beta, delta, self.gamma
