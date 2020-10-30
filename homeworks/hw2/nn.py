"""Layers and models defined for hw2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw1.utils import rescale, descale
import torch.distributions as D
import math
from homeworks.hw1.nn import MaskedLinear, MaskedLinearOutput, MaskedLinearInOut


class MaskedLinearInOut(nn.Linear):
    """Masked direct input-output linear layer for MADE."""

    def __init__(self, n_dims, n_components, bias=True):
        super().__init__(n_dims, n_dims * n_components, bias)
        # make mask
        mk = (torch.arange(n_dims) + 1).repeat_interleave(n_components)[:, None]
        mask = (mk > mk.T).squeeze()
        # make mask parameter so can be moved to correct device by model.to(device)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class Made(nn.Module):
    """Made conditioner for flow model."""

    def __init__(self, n_dims, hidden, n_components):
        super().__init__()
        self.n_dims = n_dims
        prev_mk = (torch.arange(n_dims) + 1)[:, None]
        layers = [MaskedLinear(n_dims, hidden[0], n_dims, prev_mk=prev_mk)]
        mk = layers[-1].mk
        for lidx, out_features in enumerate(hidden[1:]):
            layers.append(nn.ReLU())
            layers.append(MaskedLinear(hidden[lidx], out_features, n_dims, prev_mk=mk))
            mk = layers[-1].mk
        layers.append(nn.ReLU())
        self.sequential = nn.Sequential(*layers)
        self.pis = layers.append(MaskedLinearOutput(hidden[-1], n_dims * n_components, n_dims, prev_mk=mk))
        self.locs = layers.append(MaskedLinearOutput(hidden[-1], n_dims * n_components, n_dims, prev_mk=mk))
        self.logscales = layers.append(MaskedLinearOutput(hidden[-1], n_dims * n_components, n_dims, prev_mk=mk))
        self.inout_pis = MaskedLinearInOut(n_dims, n_components)
        self.inout_locs = MaskedLinearInOut(n_dims, n_components)
        self.inout_logscales = MaskedLinearInOut(n_dims, n_components)

    def forward(self, x):
        return self.sequential(x) + self.inout(x)


class GaussianMixCDF(nn.Module):
    """Logistic mixture CDF for elementwise flow model."""

    def __init__(self, pis_nonscaled, los, logscale):
        super().__init__()
        self.pis_nonscaled = pis_nonscaled
        self.loc = scale
        self.logscale = logscale

    def forward(self, x):
        pis = F.softmax(self.pis_nonscaled, dim=0)[None, :]
        scale = self.logscale.exp()[None, :]
        loc = self.loc[None, :]
        cdfs = 0.5 * (1 + torch.erf((x - loc) / (scale * 2**0.5)))
        logpdfs = -self.logscale[None, :] - torch.tensor((math.pi*2)**0.5).log() - (x - loc)**2 / (2 * scale**2)
        logjacob = torch.logsumexp(logpdfs + pis.log(), dim=1)
        z = (pis * cdfs).sum(dim=1)
        return z, logjacob


class AutoregressiveFlow(nn.Module):
    """Autoregressie flow."""

    def __init__(self, n_dims, n_components):
        super().__init__()
        self.n_dims = n_dims
        self.n_components = n_components
        self.conditioner = Made(n_dims, [10, 10, 10], n_components)
        distribs = 

    def forward(self, x):
        if x.shape[-1] != self.n_dims:
            raise Exception(f"Data dimension {x.shape[-1]} does not correspond to model n_dim {self.n_dims}.")
        params = 
        z = torch.zeros_like(x)
        logjacobs = torch.zeros_like(x)
        for dim in range(self.n_dims):
            z[:, dim], logjacobs[:, dim] = self.distribs[dim](x[:, dim][:, None])
        return z, logjacobs


        distribs = []
        for dim in range(n_dims):
            distribs.append(GaussianMixCDF())
        self.distribs = nn.ModuleList(distribs)


