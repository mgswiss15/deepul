"""Layers and models defined for hw3."""

import torch
import torch.nn as nn
from homeworks.hw2.nn import Made
import torch.distributions as D


class EncoderMLP(nn.Module):
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


class DecoderMLP(nn.Module):
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


class EncoderConv(nn.Module):
    """VAE convolutional encoder."""

    def __init__(self, x_dim, z_dim, hidden):
        super().__init__()
        in_dim = x_dim
        layers = [nn.Conv2d(in_dim, hidden[0], 3, 1, 1)]
        for hidx, h_dim in enumerate(hidden[:-1]):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(h_dim, hidden[hidx+1], 3, 2, 1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden[-1]*16, 2*z_dim))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        params = self.sequential(x)
        mu, logstd = torch.chunk(params, 2, dim=1)
        return mu, logstd


class DecoderConv(nn.Module):
    """VAE convolutional encoder."""

    def __init__(self, x_dim, z_dim, hidden):
        super().__init__()
        self.hidden = hidden
        self.l0 = nn.Linear(z_dim, hidden[0] * 16)
        layers = []
        for hidx, h_dim in enumerate(hidden[:-1]):
            layers.append(nn.ReLU())
            layers.append(nn.ConvTranspose2d(h_dim, hidden[hidx+1], 4, 2, 1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden[-1], 3, 3, 1, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        x = self.l0(x)
        x = x.reshape(-1, self.hidden[0], 4, 4)
        mu = self.sequential(x)
        return mu, torch.zeros_like(mu)


class VAE(nn.Module):
    """Basic VAE model."""

    def __init__(self, x_dim, z_dim, encoder, decoder, hiddenE=[32, 64, 128, 256], hiddenD=[128, 128, 64, 32]):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = encoder(x_dim, z_dim, hiddenE)
        self.reparam = Reparametrization()
        self.decoder = decoder(x_dim, z_dim, hiddenD)

    def forward(self, x):
        mu_z, logstd_z = self.encoder(x)
        zsample = self.reparam(mu_z, logstd_z.exp())
        mu_x, logstd_x = self.decoder(zsample)
        return mu_x, logstd_x, mu_z, logstd_z

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            zsample = torch.randn((n_samples, self.z_dim), device=device)
            mu_x, logstd_x = self.decoder(zsample)
            return mu_x, logstd_x


class AFPrior(nn.Module):
    """AF prior for VAE."""

    def __init__(self, n_dims, madelayers):
        super().__init__()
        self.n_dims = n_dims
        self.conditioner = Made(n_dims, madelayers, 2)

    def forward(self, z):
        params = self.conditioner(z)
        loc, logscale = torch.chunk(params.view(z.shape[0], -1, 2), 2, dim=-1)
        loc, logscale = loc.squeeze(dim=-1), logscale.squeeze(dim=-1)
        scale = logscale.exp()
        eps = z * scale + loc
        logjacobs = logscale.sum(dim=1)
        return eps, logjacobs

    def log_prob(self, z):
        eps, logjacobs = self(z)
        dist_eps = D.Normal(torch.zeros_like(eps), torch.ones_like(eps))
        logpdf_eps = dist_eps.log_prob(eps).sum(dim=1)
        logpdf_z = logpdf_eps + logjacobs
        return logpdf_z

    def sample(self, n_samples, device):
        eps = torch.randn(n_samples, self.n_dims).to(device)
        z = torch.zeros_like(eps)
        for d in range(self.n_dims):
            params = self.conditioner(z)
            loc, logscale = torch.chunk(params.view(n_samples, -1, 2), 2, dim=-1)
            loc, logscale = loc.squeeze(dim=-1), logscale.squeeze(dim=-1)
            z[:, d] = (eps[:, d] - loc[:, d]) / (logscale[:, d].exp())
        return z


class VaeAf(nn.Module):
    """Basic VAE model."""

    def __init__(self, x_dim, z_dim, madelayers, encoder, decoder, hiddenE=[32, 64, 128, 256], hiddenD=[128, 128, 64, 32]):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = encoder(x_dim, z_dim, hiddenE)
        self.reparam = Reparametrization()
        self.decoder = decoder(x_dim, z_dim, hiddenD)
        self.prior = AFPrior(z_dim, madelayers)

    def log_posterior(self, z, loc, scale):
        dist = D.Normal(loc, scale)
        return dist.log_prob(z).sum(dim=1)

    def forward(self, x):
        mu_z, logstd_z = self.encoder(x)
        zsample = self.reparam(mu_z, logstd_z.exp())
        mu_x, _ = self.decoder(zsample)
        logprior = self.prior.log_prob(zsample)
        logposterior = self.log_posterior(zsample, mu_z, logstd_z.exp())
        return mu_x, logprior, logposterior

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            zsample = self.prior.sample(n_samples, device)
            mu_x, _ = self.decoder(zsample)
            return mu_x

    def loss_func(self, x, mu_x, logprior, logposterior):
        """ELBO loss."""
        rec = 0.5*((mu_x - x)**2).sum(dim=(1, 2, 3)).mean()
        kl = (logposterior - logprior).mean()
        nelbo = rec + kl
        # print(f"loss func {nelbo} {rec} {kl}")
        return {'nelbo':nelbo, 'rec':rec, 'kl':kl}
