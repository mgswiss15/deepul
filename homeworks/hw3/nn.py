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
        mu, logstd = torch.chunk(params, 2, dim=1)
        return mu, logstd


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
        mu, logstd = torch.chunk(params, 2, dim=1)
        return mu, logstd


class Reparametrization(nn.Module):
    """VAE Gaussian reparametrization."""

    def __init__(self):
        super().__init__()

    def forward(self, mu, scale):
        epsilon = torch.randn_like(mu)
        z = epsilon * scale + mu
        return z


class VAE(nn.Module):
    """Basic VAE model."""

    def __init__(self, x_dim, z_dim, hiddenE, hiddenD):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, z_dim, hiddenE)
        self.reparam = Reparametrization()
        self.decoder = Decoder(x_dim, z_dim, hiddenD)

    def forward(self, x):
        mu_z, logstd_z = self.encoder(x)
        zsample = torch.randn_like(mu_z) * logstd_z.exp() + mu_z
        # zsample = self.reparam(mu_z, logstd_z.exp())
        mu_x, logstd_x = self.decoder(zsample)
        return mu_x, logstd_x, mu_z, logstd_z

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            zsample = torch.randn((n_samples, self.z_dim), device=device)
            mu_x, logstd_x = self.decoder(zsample)
            return mu_x, logstd_x

