"""Layers and models defined for hw3."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """VAE simple mlp encoder."""

    def __init__(self, x_dim, z_dim, hidden):
        super().__init__()
        hidden.append(z_dim * 2)
        layers = [nn.Linear(x_dim, hidden[0])]
        for hidx, h_dim in enumerate(hidden[:-1]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(h_dim, hidden[hidx+1]))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        params = self.sequential(x)
        location, logscale = torch.chunk(params, 2, dim=1)
        return location, logscale


class Decoder(nn.Module):
    """VAE simple mlp encoder."""

    def __init__(self, x_dim, z_dim, hidden):
        super().__init__()
        hidden.append(x_dim * 2)
        layers = [nn.Linear(z_dim, hidden[0])]
        for hidx, h_dim in enumerate(hidden[:-1]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(h_dim, hidden[hidx+1]))
        self.sequential = nn.Sequential(*layers)

    def forward(self, z):
        params = self.sequential(z)
        location, logscale = torch.chunk(params, 2, dim=1)
        return location, logscale


class Reparametrization(nn.Module):
    """VAE Gaussian reparametrization."""

    def __init__(self):
        super().__init__()

    def forward(self, location, scale):
        epsilon = torch.randn_like(location)
        z = epsilon * scale + location
        return z


class VAE(nn.Module):
    """Basic VAE model."""

    def __init__(self, x_dim, z_dim, hiddenE, hiddenD):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, z_dim, hiddenE)
        self.reparam = Reparametrization()
        self.decoder = Decoder(x_dim, z_dim, hiddenE)

    def forward(self, x):
        zlocation, zlogscale = self.encoder(x)
        zsample = self.reparam(zlocation, zlogscale.exp())
        xlocation, xlogscale = self.decoder(zsample)
        return xlocation, xlogscale, zlocation, zlogscale

    def sample(self, n_samples, device):
        zsample = torch.randn((n_samples, self.z_dim), device=device)
        xlocation, xlogscale = self.decoder(zsample)
        return xlocation, xlogscale

