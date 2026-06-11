import math
import numpy as np
from abc import abstractmethod
import torch
from torch import nn
from model.model_utils import smooth
from typing import Optional, Tuple

"""
These kernel functions define the nonlinearity of our model. They are separate to make analysis
a little easier and code a little cleaner.
In general, these require a number of terms, an x dim, a z dim, and an activation function.
The activation function we typically use is the identity, but one could further constrain 
the structure of the nonlinearity by changing the activation

For all experiments, we use the FullPolyModule.
This can be extended to other kernels, but if you wanto to use them you can implement them yourselves
"""


class kernelModule(nn.Module):
    def __init__(self, nTerms, device, x_dim, z_dim, activation):

        super().__init__()

        self.nTerms = nTerms
        self.device = device
        self.d = x_dim
        self.n = z_dim
        self.activation = activation
        self.tau = 1
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def get_weights(self, z, smooth_len):

        return smooth(self.activation(self.weights(z)), smooth_len)


class fullPolyModule(kernelModule):
    """
    uses products of polynomials to estimate functions. Takes control
    functions from mamba encoders and projects to weights, then calculates
    kernel based on those weights.
    """

    nTerms: int  # number of polynomial terms included
    device: str  # cuda or cpu
    x_dim: int  # data dimension
    z_dim: int  # number of inputs
    lam: float  # base regularization weight on polynomial weights
    activation: callable  # activation function for weights

    def __init__(self, nTerms, device, x_dim, z_dim, lam=0.01, activation=lambda x: x):

        super().__init__(nTerms, device, x_dim, z_dim, activation)

        self.poly_dim = nTerms

        self.weights = nn.Linear(self.n, (self.poly_dim + 1) ** 2).to(self.device)
        self.powers = torch.arange(0, self.poly_dim + 1, device=self.device)
        self.lam = lam
        # if True, keep the constant (0,0) term (the "alpha"-like forcing y^0*ydot^0).
        # The linear (1,0)/(0,1) terms are always zeroed (omega/gamma handle those).
        self.keep_const = False
        # per-term total-degree exponent (1 - (p + q)) used for the optional envelope
        # gain. A term w_{pq} * y^p * ydot^q receives a multiplicative gain
        # e^{1-(p+q)} when an envelope e(t) is supplied. This is the division-free,
        # clamp-able form of "normalize the source by e, evaluate the RHS, rescale by
        # e": distributing the leading e through e * w_{pq} (y/e)^p (ydot/e)^q gives
        # w_{pq} y^p ydot^q e^{1-(p+q)}. Linear terms (p+q=1) get exponent 0 (e cancels,
        # so omega/gamma stay scale-free); only the >=2-degree nonlinearity is reweighted.
        # deg_exp[p, q] = 1 - (p + q), shape (poly_dim+1, poly_dim+1).
        deg = self.powers[:, None] + self.powers[None, :]
        self.register_buffer("deg_exp", (1 - deg).to(torch.float32))

    def _env_gains(
        self, env: torch.FloatTensor, g_max: float
    ) -> torch.FloatTensor:
        """
        per-term envelope gains G_{pq}(t) = clamp(e(t)^{1-(p+q)}, max=g_max).

        Computed in log space (no division, so e -> 0 at onset can never blow up):
        log G_{pq} = (1-(p+q)) * log e, upper-clamped at log(g_max), then exponentiated.
        The clamp caps the gain of the high-degree terms in the near-silence region
        (e below ~g_max^{-1/(deg-1)}); there the signal y^p ydot^q -> 0 dominates, so each
        term vanishes smoothly instead of diverging.

        inputs
        ------
            - env: envelope e(t), shape (B, L, 1), strictly positive
            - g_max: upper clamp on the per-term gain
        returns
        ------
            - gains: (B, L, poly_dim+1, poly_dim+1)
        """
        log_e = torch.log(env.clamp_min(1e-12))  # (B, L, 1)
        log_g = self.deg_exp[None, None] * log_e[..., 0][..., None, None]
        log_g = torch.clamp(log_g, max=math.log(g_max))
        return torch.exp(log_g)

    def forward(
        self,
        x: torch.FloatTensor,
        z: torch.FloatTensor,
        env: Optional[torch.FloatTensor] = None,
        g_max: float = 100.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        computes weights and polynomial kernel

        inputs:
        -----
            - x: input data (audio and first derivative)
            - z: input control weights (from mamba encoder)
            - env: optional envelope e(t), shape (B, L, 1). When given, each term is
                reweighted by clamp(e^{1-(p+q)}, max=g_max) (see `_env_gains`); when
                None the kernel is exactly the legacy (un-enveloped) polynomial.
            - g_max: upper clamp on the per-term envelope gain
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
            - weights: learned weights on polynomial terms (RAW, without env gains, so
                regularization and low-pass recompute see the learned function itself)
        """

        B, L, d = x.shape
        _, _, n = z.shape
        weights = self.activation(self.weights(z))

        weights = weights.view(B, L, self.poly_dim + 1, self.poly_dim + 1)
        ### constant term
        if not self.keep_const:
            weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0
        # env gains are applied to a *copy* used for evaluation only; the returned
        # `weights` stay raw so regularization/low-pass operate on the learned function.
        w_eff = weights if env is None else weights * self._env_gains(env, g_max)
        power_mat = (x[:, :, :, None].expand(-1, -1, -1, self.poly_dim + 1)).pow(
            self.powers
        )
        ## power mat is now: B x L x d x p
        ## we want to turn it into a B x L x 1 x p x p matrix

        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = torch.einsum("bldp,bldk -> blpk", z1, z2)
        ###
        ## batch x time x 1 (y [z1] or dy [z2]) x n poly
        ## -> batch x time x y degree x dy degree

        x = torch.einsum("blpd,blpd -> bl", w_eff, power_mat)

        return x[:, :, None], weights

    def forward_given_weights(
        self,
        x: torch.FloatTensor,
        weights: torch.FloatTensor,
        env: Optional[torch.FloatTensor] = None,
        g_max: float = 100.0,
    ) -> torch.FloatTensor:
        """
        computes polynomial kernel given weights on polynomial terms

        inputs:
        -----
            - x: input data (audio and first derivative)
            - weights: weights on polynomial terms
            - env: optional envelope e(t), shape (B, L, 1); see `forward`
            - g_max: upper clamp on the per-term envelope gain
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
        """

        B, L, d = x.shape
        weights = weights.view(B, L, self.poly_dim + 1, self.poly_dim + 1)
        ### constant term
        if not self.keep_const:
            weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0
        w_eff = weights if env is None else weights * self._env_gains(env, g_max)
        power_mat = (x[:, :, :, None].expand(-1, -1, -1, self.poly_dim + 1)).pow(
            self.powers
        )

        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = torch.einsum("bldp,bldk -> blpk", z1, z2)
        x = torch.einsum("blpd,blpd -> bl", w_eff, power_mat)

        return x[:, :, None]

    def forward_given_weights_numpy(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        env: Optional[float] = None,
        g_max: float = 100.0,
    ) -> np.ndarray:
        """
        computes polynomial kernel given weights on polynomial terms.
        same as the above method, but using numpy instead of torch

        inputs:
        -----
            - x: input data (audio and first derivative)
            - weights: weights on polynomial terms
            - env: optional envelope e (scalar or array broadcastable to (B, L)); when
                given each term is reweighted by clamp(e^{1-(p+q)}, max=g_max), matching
                the torch path used in training
            - g_max: upper clamp on the per-term envelope gain
        returns:
        -----
            - computed kernel: weighted sum of product of polynomials
        """

        powers = self.powers.detach().cpu().numpy()
        if len(x.shape) != 3:
            x = np.reshape(x, (x.shape[0], -1, 2 * self.d))
        B, L, d = x.shape
        # if weights.shape != (B,L,self.d,self.poly_dim-1):
        weights = np.reshape(weights, (B, L, self.poly_dim + 1, self.poly_dim + 1))

        # constant term
        if not self.keep_const:
            weights[:, :, 0, 0] = weights[:, :, 0, 0] * 0
        ### y, ydot terms
        weights[:, :, 1, 0] = weights[:, :, 1, 0] * 0
        weights[:, :, 0, 1] = weights[:, :, 0, 1] * 0

        if env is not None:
            deg_exp = self.deg_exp.detach().cpu().numpy()  # (P+1, P+1)
            log_e = np.log(np.maximum(np.asarray(env, dtype=weights.dtype), 1e-12))
            log_g = deg_exp[None, None] * log_e.reshape(B, L, 1, 1)
            log_g = np.minimum(log_g, np.log(g_max))
            weights = weights * np.exp(log_g)

        power_mat = np.power(
            np.tile(x[:, :, :, None], (1, 1, 1, self.poly_dim + 1)), powers
        )
        z1 = power_mat[:, :, :1, :]
        z2 = power_mat[:, :, 1:, :]
        power_mat = np.einsum("bldp,bldk->blpk", z1, z2)
        x = np.einsum("blpd,blpd->bl", weights, power_mat)

        return x[:, :, None]
