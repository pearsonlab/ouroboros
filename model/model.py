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
        keep_const: bool = False,
        osc_init: bool = False,
        gamma_init: float = -0.01,
        vdp_init: float = 0.02,
        cubic_init: float = 0.01,
        const_init: float = 1e-3,
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
        # zero-phase Gaussian (sigma = drive_lowpass_ms) -- "low-pass in the loop". Keeps the
        # drives from re-encoding the audio carrier and improves autonomous behavior.
        self.drive_lowpass_ms = drive_lowpass_ms
        self.kernel = kernel
        self.kernel.tau = self.tau
        # if True, ADD the constant (0,0) "alpha"-like forcing term (y^0*ydot^0) to the kernel
        # instead of zeroing it; it is low-passed at drive_lowpass_ms like the other drives.
        self.keep_const = keep_const
        if keep_const:
            self.kernel.keep_const = True
            # zero-init the (0,0) output of the kernel weight head so the alpha forcing starts at
            # 0 and is learned gently (otherwise the random constant disrupts early training).
            with torch.no_grad():
                self.kernel.weights.weight[0].zero_()
                self.kernel.weights.bias[0].zero_()
        # Strategy 1 "oscillator init": start every seed as a marginal van der Pol limit cycle
        # instead of the default coin-flip-sign, ~40x-too-stiff linear damping. At the default
        # init |gamma| ~ 0.4 (envelope tau ~ 0.1 ms) with random per-seed sign, so ~half of seeds
        # are purely dissipative (oscillation dies instantly -> the model can only AM-modulate
        # input/IC noise). osc_init makes the damping a small NEGATIVE constant (slow energy
        # injection, envelope tau ~ ms) and seeds an amplitude-dependent re-damping (+y^2*ydot)
        # plus a hardening cubic (+y^3) so the growth saturates into a bounded cycle. The
        # corresponding control-head WEIGHTS are zeroed so these start as clean constants the
        # encoders can later modulate; only the structural prior is injected.
        #
        # Net damping in ydot-dot is -(gamma_init + vdp_init*y^2)*ydot, zero at amplitude
        # A* = sqrt(-gamma_init/vdp_init) (~0.7 with the defaults), i.e. a limit cycle near |y|~0.7.
        if osc_init:
            with torch.no_grad():
                # gamma: pure small negative constant (anti-damping) -> energy injection
                self.gamma_net.weight.zero_()
                self.gamma_net.bias.fill_(float(gamma_init))
                # kernel polynomial weights: index (i,j) -> row i*W + j in the flattened
                # (poly_dim+1)x(poly_dim+1) output of kernel.weights. yhat subtracts
                # weighted_kernels, so a +w[i,j] adds -w[i,j]*y^i*ydot^j to ydot-dot.
                W = self.kernel.poly_dim + 1
                def _seed(i, j, val):
                    idx = i * W + j
                    self.kernel.weights.weight[idx].zero_()
                    self.kernel.weights.bias[idx].fill_(float(val))
                _seed(2, 1, vdp_init)    # +y^2*ydot -> amplitude-dependent re-damping (saturation)
                _seed(3, 0, cubic_init)  # +y^3      -> hardening spring (bounds amplitude)
                if keep_const:
                    # tiny constant drive to break the y=0 fixed-point symmetry so the cycle ignites
                    _seed(0, 0, const_init)
        self.names = [r"$\omega$", r"$\gamma$", "weighted kernels", "states"]

    def _lowpass(self, x: torch.FloatTensor, dt: float, lp_ms: float = None) -> torch.FloatTensor:
        """centered zero-phase Gaussian low-pass along time of a (B, L, C) control series.
        lp_ms overrides the timescale (defaults to self.drive_lowpass_ms). The kernel radius is
        capped at L-1 so very slow (large-sigma) low-passes work on short segments (reduce to
        ~a segment-wide average)."""
        ms = self.drive_lowpass_ms if lp_ms is None else lp_ms
        sigma = (ms / 1e3) / dt  # samples
        if sigma <= 0:
            return x
        L = x.shape[1]
        radius = max(1, min(int(round(3 * sigma)), L - 1))
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
