import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mambapy.mamba import Mamba, MambaConfig
from model.model_utils import smooth
from typing import Optional, Tuple, Union
# from model.kernels import *
import math

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


class Tract(nn.Module):
    r"""
    Linear (LTI) vocal-tract filter as a learnable RATIONAL transfer function in the
    Laplace domain (a polynomial filter, like the Arneodo/Perl OEC + trachea equivalent
    circuit), applied in the rFFT domain.

        H(jw) = K * prod_k [ (jw)^2 + 2 zeta_z,k w_z,k (jw) + w_z,k^2 ]
                          / [ (jw)^2 + 2 zeta_p,k w_p,k (jw) + w_p,k^2 ]  *  (1 - r e^{-jw tau})
                \________________ rational pole/zero cascade _______________/  \__ trachea comb __/

    Each second-order section is a complex-conjugate POLE pair (a formant/resonance) and a
    ZERO pair (an anti-resonance); n_sec sections give a rational filter with n_sec
    resonances, and the paper's 3-state OEC circuit is one particular special case. Unlike a
    zero-phase magnitude bump this carries the proper resonant PHASE (a pole pair's phase
    swings ~+/-pi across its resonance). With real coefficients H(jw) is Hermitian, so the
    rFFT -> *H -> irFFT in `apply` returns a real, correctly-phased signal. Applied FORWARD
    only (source -> audio), so no invertibility / min-phase constraint is needed.

    Stability: dampings zeta_p > 0 (softplus) keep the poles in the left half-plane, so
    |den| > 0 and the filter is bounded. Identity at init: each section's zeros are
    initialized EQUAL to its poles (numerator == denominator => section == 1), K = 1, r = 0,
    so H == 1 -- a fresh model leaves the rollout untouched and resonances emerge only as
    training separates the zeros from the poles.
    """

    def __init__(
        self,
        device: str = "cuda",
        n_sec: int = 3,
        f0_inits=(0.04, 0.10, 0.20),  # pole center freqs (cycles/sample); ~1.6/4/8 kHz @ 40 kHz
        zeta_init: float = 0.1,        # pole/zero damping at init (Q ~ 5)
        tau_init: float = 10.0,
        pad: int = 128,
        r_max: float = 0.99,
        use_comb: bool = True,
    ):
        super().__init__()
        self.device = device
        self.pad = pad
        self.r_max = r_max
        self.n_sec = n_sec

        f0s = list(f0_inits)[:n_sec]
        if len(f0s) < n_sec:  # spread any extra sections across (0, 0.5)
            f0s = [0.5 * (i + 1) / (n_sec + 1) for i in range(n_sec)]

        def _logit(p):  # inverse of 0.5*sigmoid, p in (0, 0.5)
            return math.log(p / (0.5 - p))

        zr = math.log(math.expm1(zeta_init))  # inverse softplus
        # pole parameters (one per section)
        self.f0_raw = nn.Parameter(torch.tensor([_logit(f) for f in f0s], device=device))
        self.zeta_p_raw = nn.Parameter(torch.full((n_sec,), float(zr), device=device))
        # zero parameters -- initialized EQUAL to the poles so each section == 1 (identity).
        self.fz_raw = nn.Parameter(self.f0_raw.detach().clone())
        self.zeta_z_raw = nn.Parameter(self.zeta_p_raw.detach().clone())
        # Global tract gain K = softplus(K_raw). Softplus has bounded gradient
        # (= sigmoid(K_raw) in (0, 1)), so once training pushes K_raw moderately
        # positive the rate of K change saturates at 1 — unlike exp() where K
        # grew exponentially in K_raw and amplitude could runaway. In the small-K
        # regime (K << 1) softplus(K_raw) ≈ exp(K_raw), so init in this corner
        # behaves like the old log_K parameterization. Init K_raw=0 → K=log(2)≈0.69
        # (vs the old K=1 at log_K=0); the entry script's data-driven init
        # picks K_raw = softplus_inv(target_K) so the absolute gain is correct.
        self.K_raw = nn.Parameter(torch.zeros((), device=device))
        # Target gain for the optional K-gauge anchor (see train.spectral_rollout). Persistent
        # buffer so it rides save/load + device moves; holds the INITIAL K = softplus(K_raw).
        # model_cv's data-driven K_raw init updates it so the anchor pins K near its data-matched
        # starting gain, not the constructor default. Closes the (K, source) -> (c*K, source/c)
        # gauge the envelope anchor leaves open. (load_model tolerates its absence in old ckpts.)
        self.register_buffer("K_anchor_target", F.softplus(self.K_raw.detach().clone()))
        # trachea comb: reflection r = r_max*tanh(r_raw) (init 0 -> no comb); delay tau samples.
        # Disabled (use_comb=False) for the noise branch: a comb imposes periodic spectral teeth,
        # exactly the harmonic-like structure the low-order noise filter must NOT be able to make
        # -- without it the noise filter is a pure broadband pole/zero envelope.
        self.use_comb = use_comb
        if use_comb:
            self.r_raw = nn.Parameter(torch.zeros((), device=device))
            self.tau_raw = nn.Parameter(
                torch.tensor(math.log(math.expm1(tau_init)), device=device)
            )

    def _transfer(self, n: int, device, dtype) -> torch.Tensor:
        """complex rational transfer function H(f) on the rFFT grid of a length-n signal."""
        kf = torch.arange(n // 2 + 1, device=device, dtype=dtype)
        w = 2 * math.pi * (kf / n)                    # rad/sample, >= 0
        jw = torch.complex(torch.zeros_like(w), w)    # j*w
        jw2 = jw * jw                                 # = -w^2

        wp = 2 * math.pi * (0.5 * torch.sigmoid(self.f0_raw))   # (n_sec,) pole freqs
        zp = F.softplus(self.zeta_p_raw)                        # (n_sec,) pole dampings > 0
        wz = 2 * math.pi * (0.5 * torch.sigmoid(self.fz_raw))   # (n_sec,) zero freqs
        zz = F.softplus(self.zeta_z_raw)                        # (n_sec,) zero dampings > 0

        H = torch.ones_like(jw) * F.softplus(self.K_raw)
        for k in range(self.n_sec):
            num = jw2 + (2 * zz[k] * wz[k]) * jw + wz[k] ** 2
            den = jw2 + (2 * zp[k] * wp[k]) * jw + wp[k] ** 2
            H = H * (num / den)

        if not self.use_comb:
            return H
        r = self.r_max * torch.tanh(self.r_raw)
        tau = F.softplus(self.tau_raw)
        ang = -2 * math.pi * (kf / n) * tau
        comb = 1.0 - r * torch.complex(torch.cos(ang), torch.sin(ang))
        return H * comb

    def apply(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        filter a (B, L, 1) real waveform along time by H (forward). Reflection-padded to
        limit circular-convolution edge ringing.
        """
        B, L, C = x.shape
        pad = min(self.pad, max(0, L - 1))
        xc = x.transpose(1, 2)
        if pad > 0:
            xc = F.pad(xc, (pad, pad), mode="reflect")
        n = xc.shape[-1]
        X = torch.fft.rfft(xc, dim=-1)
        H = self._transfer(n, x.device, x.dtype)
        y = torch.fft.irfft(X * H[None, None, :], n=n, dim=-1)
        if pad > 0:
            y = y[..., pad : pad + L]
        return y.transpose(1, 2)


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
        alpha_lowpass_ms: float = 0.0,
        keep_const: bool = False,
        osc_init: bool = False,
        checkpoint_encoder: bool = False,
        gamma_init: float = -0.01,
        vdp_init: float = 0.02,
        cubic_init: float = 0.01,
        const_init: float = 1e-3,
        use_tract: bool = False,
        tract_n_sec: int = 3,
        use_envelope: bool = False,
        env_lowpass_ms: float = 20.0,
        enable_noise_forcing: bool = False,
        noise_tau_ms: float = 5.0,
        noise_init_bias: float = 0.1,
        use_noise_branch: bool = False,
        noise_tract_n_sec: int = 3,
        sigma_lowpass_ms: float = 0.0,
        sigma_constant: bool = False,
        use_rumble_branch: bool = False,
        rumble_lowpass_hz: float = 250.0,
        noise_highpass_hz: float = 0.0,   # noise high-pass cutoff; 0 -> use rumble_lowpass_hz (complementary)
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
        # Optional EXTRA low-pass on alpha (the constant kernel term [0,0] = pressure-analog DC
        # drive). 0 = off (alpha smoothed at drive_lowpass_ms like the rest). >0 smooths alpha at
        # this longer timescale so the driving pressure varies slowly (syllable-scale), matching the
        # physical syringeal pressure wave.
        self.alpha_lowpass_ms = alpha_lowpass_ms
        # if True, gradient-checkpoint the three Mamba drive encoders in get_funcs: their
        # activations over the doubled-length sequence (x_in is 2L) dominate training memory
        # (~6.9 GB at B=64, vs ~60 MiB for the RK4 rollout), so recomputing them in backward
        # is what unlocks larger batch sizes. ~1 extra encoder forward per step in exchange.
        self.checkpoint_encoder = checkpoint_encoder
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

        # ---- learnable amplitude envelope + vocal-tract filter (opt-in) ----
        # Both default off, so an Ouroboros built/loaded without the flags is the legacy
        # poly model. They act ONLY on the rolled-out waveform (see get_envelope / Tract and
        # train.spectral_rollout): the synthesized source x(t) is multiplied by e(t) and
        # filtered by H, then compared to the audio. e is NOT in the kernel / 2nd-derivative,
        # so there are no reciprocal-e gains and a small e simply makes the output quiet.
        self.use_tract = use_tract
        self.use_envelope = use_envelope
        self.env_lowpass_ms = env_lowpass_ms

        if use_envelope:
            # a fourth parallel Mamba encoder emits the (log) amplitude envelope e(t).
            envConfig = MambaConfig(
                d_model=2 * d_data,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            self.env_mamba = Mamba(envConfig).to(device)
            # head outputs log e(t); e = exp(.). Zero-init => e starts at 1 (no scaling),
            # so the model begins exactly at the un-enveloped rollout and learns amplitude
            # only if it lowers the loss.
            self.env_net = nn.Linear(
                in_features=2 * d_data, out_features=d_data, device=device
            )
            nn.init.zeros_(self.env_net.weight)
            nn.init.zeros_(self.env_net.bias)
            self.names = self.names + [r"$e$"]

        if use_tract:
            self.tract = Tract(device=device, n_sec=tract_n_sec)
            self.tract_n_sec = tract_n_sec

        # ---- flow-gated colored-noise forcing (opt-in) ----
        # Adds an Ornstein-Uhlenbeck colored-noise term g(t)*eta to the momentum equation
        # ONLY (see the stochastic-Heun path in train.spectral_rollout). eta is one scalar
        # OU state per oscillator with fixed correlation time noise_tau_ms; g(t) = ReLU of a
        # new Mamba head parallel to omega/gamma -- a learned, nonnegative, time-varying gate
        # that sets where/how much noise is injected (it absorbs the intensity, so there is no
        # separate sigma0). Off by default: a model built without the flag is bit-identical to
        # the deterministic poly model and adds no parameters.
        self.enable_noise_forcing = enable_noise_forcing
        self.noise_tau_ms = noise_tau_ms
        # Optional low-pass on the sigma gate g(t) (0 = off, the default -- g stays sharp). When
        # >0, get_sigma smooths g at this timescale (same zero-phase Gaussian as the drives), so
        # the noise amplitude envelope can't snap abruptly.
        self.sigma_lowpass_ms = sigma_lowpass_ms
        self.sigma_constant = sigma_constant
        # ---- harmonic-plus-noise: additive filtered-noise branch (opt-in) ----
        # An alternative to the in-ODE OU forcing: keep the oscillator PURELY deterministic
        # (RK4) and add, OUTSIDE the tract, a parallel noise source -- white noise, amplitude-
        # modulated by the sigma gate g(t), passed through a learned time-varying filter (a
        # per-frame magnitude response from a new Mamba head), summed with the tract output.
        # This is the DDSP harmonic-plus-filtered-noise decomposition: the oscillator carries
        # the tonal/harmonic content, the noise branch carries the broadband/aperiodic floor,
        # and the two can't fight (the noise never touches the ODE, so no collapse, no Heun).
        self.use_noise_branch = use_noise_branch
        # The sigma gate is the noise AM for the additive branch too, so build it for EITHER
        # mode. get_sigma() keys off the head existing, not off enable_noise_forcing.
        if enable_noise_forcing or use_noise_branch:
            sigmaConfig = MambaConfig(
                d_model=2 * d_data,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            self.sigma_mamba = Mamba(sigmaConfig).to(device)
            # Gate head g(t) = relu(sigma_net(.)). The bias is init to a small POSITIVE constant
            # (noise_init_bias, default 0.1) so the pre-activation is positive-biased and the
            # ReLU starts LIVE: with the default (random, zero-ish-bias) Linear init ~1/3 of
            # seeds had an all-negative pre-activation -> gate identically 0 -> relu'=0 -> the
            # head received ZERO gradient and could never learn (measured across seeds). The
            # gate being "on" at init is harmless because the actual forcing = noise_gain*g*eta
            # and the noise_gain ramp (train.train) holds noise_gain at 0 until noise_start_step.
            # noise_init_bias<=0 restores the dead-ReLU risk and is only for deliberate ablation.
            self.sigma_net = nn.Linear(
                in_features=2 * d_data, out_features=d_data, device=device
            )
            nn.init.constant_(self.sigma_net.bias, float(noise_init_bias))
            self.names = self.names + [r"$\sigma$"]

        if use_noise_branch:
            self.noise_tract_n_sec = noise_tract_n_sec
            # Low-order rational (pole/zero) filter for the noise branch -- SAME parameterization
            # as the vocal Tract (n_sec second-order pole/zero sections, identity at init). Kept
            # deliberately LOW order so it can shape a broadband spectral envelope but CANNOT
            # synthesize sharp harmonic peaks -- that forces the oscillator to carry the tonal /
            # harmonic structure instead of the noise modelling everything. Applied to white noise;
            # the sigma gate g(t) provides the time-varying amplitude (see filtered_noise_branch).
            self.noise_tract = Tract(device=device, n_sec=noise_tract_n_sec, use_comb=False)
            self.names = self.names + [r"$H_{noise}$"]

        # Deterministic low-frequency "rumble" source, band-limited to < rumble_lowpass_hz, added
        # alongside tract + noise (OUTSIDE the tract). A dedicated cheap channel for the sub-cutoff
        # recording floor so the oscillator isn't forced to spend capacity matching it -- but the
        # oscillator is left FULL-RANGE (not high-passed), so it can still reach below the cutoff
        # when a vocalization has genuine LF content. Zero-init head -> starts silent, learned gently.
        self.use_rumble_branch = use_rumble_branch
        # Noise high-pass cutoff. 0 -> use rumble_lowpass_hz (complementary crossover). Set > 0 to
        # DECOUPLE: e.g. rumble low-passes at 300 (covers the LF signal a bit higher) while the noise
        # high-passes at 250, giving a 250-300 overlap where the rumble carries the deterministic LF
        # and the noise carries the floor -- fills the crossover residual bump.
        self.noise_highpass_hz = float(noise_highpass_hz)
        if use_rumble_branch:
            self.rumble_lowpass_hz = float(rumble_lowpass_hz)
            rumbleConfig = MambaConfig(
                d_model=2 * d_data,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            self.rumble_mamba = Mamba(rumbleConfig).to(device)
            self.rumble_net = nn.Linear(2 * d_data, d_data, device=device)
            nn.init.zeros_(self.rumble_net.weight)
            nn.init.zeros_(self.rumble_net.bias)
            self.names = self.names + [r"rumble"]

    def _lowpass_fft(self, x: torch.FloatTensor, cutoff_hz: float, dt: float) -> torch.FloatTensor:
        """Zero-phase soft-mask rFFT low-pass along time of a (B, L, C) series: unity below
        cutoff_hz, raised-cosine rolloff to 0 by 1.5*cutoff_hz. Fixed (non-learnable) given the
        cutoff; differentiable through rfft/irfft. Used to band-limit the rumble source."""
        B, L, C = x.shape
        sig = x.transpose(1, 2)                                   # (B, C, L)
        Xf = torch.fft.rfft(sig, dim=-1)
        freqs = torch.fft.rfftfreq(L, d=dt, device=x.device)     # (L//2+1,) Hz
        hi = 1.5 * cutoff_hz
        t = ((hi - freqs) / (hi - cutoff_hz)).clamp(0.0, 1.0)     # 1 below cutoff, 0 above hi
        mask = 0.5 - 0.5 * torch.cos(math.pi * t)                # raised-cosine transition band
        out = torch.fft.irfft(Xf * mask, n=L, dim=-1)            # (B, C, L)
        return out.transpose(1, 2)                                # (B, L, C)

    def get_rumble(
        self, x: torch.FloatTensor, dxdt: torch.FloatTensor, dt: float
    ) -> Optional[torch.FloatTensor]:
        """Deterministic low-frequency source r(t), band-limited to < rumble_lowpass_hz, shape
        (B, L, 1), or None when use_rumble_branch is False. Parallel Mamba head over the same
        [x, x'] state as the drives; zero-init so it starts silent. Added to the tract+noise
        output OUTSIDE the tract (see train.spectral_rollout / train.eval)."""
        if not getattr(self, "use_rumble_branch", False):
            return None
        dxdt = dxdt * (self.tau / dt)  # rescaled velocity; out-of-place (no caller mutation)
        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)
        if self.checkpoint_encoder and torch.is_grad_enabled():
            r_out = checkpoint(self.rumble_mamba, x_in, use_reentrant=False)[:, L:, :]
        else:
            r_out = self.rumble_mamba(x_in)[:, L:, :]
        r = self.rumble_net(r_out)                          # (B, L, 1) raw LF source
        return self._lowpass_fft(r, self.rumble_lowpass_hz, dt)

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
        """low-pass the polynomial kernel weights (B, L, P, P) along time. The constant term
        (alpha = [0,0], the pressure-analog DC drive) optionally gets an EXTRA, longer low-pass
        (alpha_lowpass_ms) so it varies on a slow syllable-scale timescale."""
        B, L, P, P2 = weights.shape
        w = self._lowpass(weights.reshape(B, L, P * P2), dt).reshape(B, L, P, P2)
        if getattr(self, "alpha_lowpass_ms", 0.0) > 0:
            a = self._lowpass(w[:, :, 0, 0:1], dt, lp_ms=self.alpha_lowpass_ms)  # (B,L,1)
            w = w.clone()
            w[:, :, 0, 0] = a[:, :, 0]
        return w

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

        # The three Mamba encoders over x_in (length 2L) are the training-memory bottleneck.
        # When checkpoint_encoder is on (and we're building a graph), recompute them in
        # backward instead of storing their activations -- trades ~1 extra encoder forward
        # for the headroom to push batch size past what fits otherwise.
        if self.checkpoint_encoder and torch.is_grad_enabled():
            omegaControl = checkpoint(self.omega_mamba, x_in, use_reentrant=False)[:, L:, :]
            gammaControl = checkpoint(self.gamma_mamba, x_in, use_reentrant=False)[:, L:, :]
            kernelControl = checkpoint(self.kernel_mamba, x_in, use_reentrant=False)[:, L:, :]
        else:
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

    def get_envelope(
        self, x: torch.FloatTensor, dxdt: torch.FloatTensor, dt: float
    ) -> Optional[torch.FloatTensor]:
        """
        learnable amplitude envelope e(t) = softplus(lowpass(env_head)) / log(2),
        shape (B, L, 1), or None when use_envelope is False. A fourth Mamba encoder
        reads the same [x, x'] state as the drives; the head is zero-init so the
        pre-activation is 0, giving softplus(0)/log(2) = 1 -- the identity gauge
        starting point. Always low-passed at env_lowpass_ms (default 20 ms) so e is
        the slow amplitude gauge. Applied as a plain multiplicative factor on the
        rolled-out waveform (e * x), NOT in the kernel.

        Why softplus(...)/log(2) instead of exp(...):
          - softplus(x) is BOUNDED in growth: softplus(x) ~= x for large positive x,
            so even softplus(1000) = 1000 (no fp32 inf). exp(88) = inf in fp32, and
            the env_anchor + (e-1)^2 gradients combined with env_mamba drift could
            push the pre-exp tensor over that threshold on rare batches, producing
            inf * 0 = NaN in the rolled-out spec loss and triggering the cascade we
            saw on 2026-06-18 (env_lowpass_ms=2 run on blk445).
          - softplus' derivative is sigmoid(x), bounded in [0, 1]. exp's derivative
            is exp(x) itself -- unboundedly amplifying. The new envelope therefore
            cannot blow up backward gradients on extreme batches.
          - Identity-init preserved: softplus(0)/log(2) = log(2)/log(2) = 1, so
            existing (e-1)^2 anchor and all the resume/load paths are unchanged.
          - Small-e behaviour also smoother: silence regions have softplus(very
            negative) -> 0 smoothly, vs exp which has the same limit but with
            unbounded gradient on the way down.
        """
        if not self.use_envelope:
            return None
        dxdt = dxdt * (self.tau / dt)  # rescaled velocity; out-of-place (no caller mutation)
        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)
        # Match the three drive encoders' checkpointing policy (see get_funcs): when
        # checkpoint_encoder is on, recompute env_mamba in backward instead of storing its
        # length-2L activations. Without this, env_mamba alone keeps ~600 MiB at B=64.
        if self.checkpoint_encoder and torch.is_grad_enabled():
            env_out = checkpoint(self.env_mamba, x_in, use_reentrant=False)[:, L:, :]
        else:
            env_out = self.env_mamba(x_in)[:, L:, :]
        e_pre = self.env_net(env_out)
        e_pre = self._lowpass(e_pre, dt, lp_ms=self.env_lowpass_ms)
        return F.softplus(e_pre) / math.log(2)

    def get_sigma(
        self, x: torch.FloatTensor, dxdt: torch.FloatTensor, dt: float
    ) -> Optional[torch.FloatTensor]:
        """flow-gated noise gate g(t) = relu(sigma_net(sigma_mamba(x_in))), shape (B, L, 1),
        or None when enable_noise_forcing is False. A parallel Mamba head reads the same
        [x, x'] state as the drives; ReLU keeps the gate nonnegative. Unlike the envelope it
        is deliberately NOT low-passed -- the gate is allowed to be sharp so it can track fast
        onset/offset structure. Multiplies the OU noise eta in the momentum equation (see the
        stochastic-Heun path in train.spectral_rollout); at init the noise_gain ramp holds the
        term off regardless of g. Also serves as the amplitude modulation for the additive
        filtered-noise branch (harmonic-plus-noise mode)."""
        if not hasattr(self, "sigma_net"):   # built for enable_noise_forcing OR use_noise_branch
            return None
        dxdt = dxdt * (self.tau / dt)  # rescaled velocity; out-of-place (no caller mutation)
        z = torch.cat([x, dxdt], dim=-1)
        L = z.shape[1]
        x_in = torch.cat([torch.flip(z, [1]), z], dim=1)
        # Match the drive/envelope encoders' checkpointing policy (see get_funcs).
        if self.checkpoint_encoder and torch.is_grad_enabled():
            g_out = checkpoint(self.sigma_mamba, x_in, use_reentrant=False)[:, L:, :]
        else:
            g_out = self.sigma_mamba(x_in)[:, L:, :]
        if getattr(self, "sigma_constant", False):
            # ONE scalar gain per vocalization on the filtered noise. The Mamba sees the whole
            # waveform (+ its reversal) and mean-pools to a single number; the spectral loss is MASKED
            # to the quiet frames (see mrstft_loss), so this gain is trained to match the noise floor
            # -- "see the whole voc, output a gain for the quiet parts". Softplus (not relu) so it can
            # approach 0 smoothly for clean vocs instead of dead-relu collapsing. See
            # docs/noise_floor_fit.md.
            g_out = g_out.mean(dim=1, keepdim=True)                     # (B, 1, F) full-voc summary
            return F.softplus(self.sigma_net(g_out)).expand(-1, L, -1)  # (B, L, 1) constant gain
        g = F.relu(self.sigma_net(g_out))
        # Optional smoothing (opt-in via sigma_lowpass_ms). The Gaussian kernel has all-positive,
        # unit-sum weights, so low-passing a nonnegative gate keeps it nonnegative.
        if getattr(self, "sigma_lowpass_ms", 0.0) > 0:
            g = self._lowpass(g, dt, lp_ms=self.sigma_lowpass_ms)
        return g

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
