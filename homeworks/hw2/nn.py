"""Layers and models defined for hw2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw1.utils import rescale, descale
import math
from homeworks.hw1.nn import MaskedLinear, MaskedLinearOutput


class MaskedLinearInOut(nn.Linear):
    """Masked direct input-output linear layer for MADE."""

    def __init__(self, n_dims, out_features, bias=True):
        super().__init__(n_dims, out_features, bias)
        # make mask
        prev_mk = (torch.arange(n_dims) + 1)[:, None]
        mk = (torch.arange(n_dims) + 1).repeat_interleave(int(out_features / n_dims))[:, None]
        mask = (mk > prev_mk.T).squeeze()
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
        self.out = MaskedLinearOutput(hidden[-1], n_dims * n_components * 3, n_dims, prev_mk=mk)
        self.inout = MaskedLinearInOut(n_dims, n_dims * n_components * 3)

    def forward(self, x):
        seq = self.sequential(x)
        params = self.out(seq) + self.inout(x)
        return params


class AutoregressiveFlow(nn.Module):
    """Autoregressie flow."""

    def __init__(self, n_dims, n_components, madelayers, transformertype):
        super().__init__()
        self.n_dims = n_dims
        self.n_components = n_components
        self.conditioner = Made(n_dims, madelayers, n_components)
        # self.transformer = self.transformergauss if transformertype=='gauss' else self.transformerlogistic
        self.transformer = self.transformergauss

    def forward(self, x):
        if x.shape[-1] != self.n_dims:
            raise Exception(f"Data dimension {x.shape[-1]} does not correspond to model n_dim {self.n_dims}.")
        params = self.conditioner(x)
        return self.transformer(x, params)

    def transformergauss(self, x, params):
        params = params.view(-1, self.n_dims, 3, self.n_components)
        pis = F.softmax(params[:, :, 0, :], dim=-1)
        locs = params[:, :, 1, :]
        logscales = params[:, :, 2, :]
        scales = logscales.exp()
        cdfs = pis * (0.5 * (1 + torch.erf((x[..., None] - locs) / (scales * 2**0.5))))
        z = cdfs.sum(dim=2)
        logpdfs = -logscales - torch.tensor((math.pi*2)**0.5).log() - (x[..., None] - locs)**2 / (2 * scales**2)
        logjacobs = torch.logsumexp(logpdfs + pis.log(), dim=2)
        return z, logjacobs

    def transformerlogistic(self, x, params):
        """This does not work, loss eventually explodes (too small) and parameters fall to nans"""
        params = params.view(-1, self.n_dims, 3, self.n_components)
        pis = F.softmax(params[:, :, 0, :], dim=-1)
        locs = params[:, :, 1, :]
        logscales = params[:, :, 2, :]
        scales = logscales.exp()
        xnorm = (x[..., None] - locs) / scales
        cdfs = pis * torch.sigmoid(xnorm)
        z = cdfs.sum(dim=2)
        logpdfs = -xnorm - logscales + 2 * xnorm
        logjacobs = torch.logsumexp(logpdfs + pis.log(), dim=2)
        return z, logjacobs
