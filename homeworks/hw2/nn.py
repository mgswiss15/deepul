"""Layers and models defined for hw2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw1.utils import rescale, descale
import torch.distributions as D
import math


class GaussianMixCDF(nn.Module):
    """Logistic mixture CDF for elementwise flow model."""

    def __init__(self, n_components=5, min=-5., max=5.):
        super().__init__()
        self.pis_nonscaled = nn.Parameter(torch.ones(n_components) / n_components)
        self.loc = nn.Parameter(torch.linspace(min, max, n_components))
        self.logscale = nn.Parameter(torch.zeros(n_components))

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

    def __init__(self, n_dims):
        super().__init__()
        self.n_dims = n_dims
        distribs = []
        for dim in range(n_dims):
            distribs.append(GaussianMixCDF())
        self.distribs = nn.ModuleList(distribs)

    def forward(self, x):
        if x.shape[-1] != self.n_dims:
            raise Exception(f"Data dimension {x.shape[-1]} does not correspond to model n_dim {self.n_dims}.")
        z = torch.zeros_like(x)
        logjacobs = torch.zeros_like(x)
        for dim in range(self.n_dims):
            z[:, dim], logjacobs[:, dim] = self.distribs[dim](x[:, dim][:, None])
        return z, logjacobs

