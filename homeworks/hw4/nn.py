"""Layers and modules for hw4."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorMlp(nn.Module):
    """MLP generator for GAN."""

    def __init__(self, zdim, hidden, xdim):
        super().__init__()
        self.zdim = zdim
        hidden.append(xdim)
        layers = [nn.Linear(zdim, hidden[0])]
        indim = hidden[0]
        for outdim in hidden[1:]:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(indim, outdim))
            indim = outdim
        self.sequential = nn.Sequential(*layers)

    def forward(self, z):
        x = self.sequential(z)
        return x

    def sample(self, nsamples):
        z = torch.randn(nsamples, self.zdim)
        x = self(z)
        return x

    def loss_func(self, target, logit):
        loss = -F.binary_cross_entropy_with_logits(logit, target)
        return loss


class GeneratorMlp2(GeneratorMlp):
    """MLP generator for GAN."""

    def loss_func(self, target, logit):
        target = 1. - target
        loss = F.binary_cross_entropy_with_logits(logit, target)
        return loss


class DiscriminatorMlp(nn.Module):
    """MLP discriminator for GAN."""

    def __init__(self, xdim, hidden):
        super().__init__()
        hidden.append(xdim)
        layers = [nn.Linear(xdim, hidden[0])]
        indim = hidden[0]
        for outdim in hidden[1:]:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(indim, outdim))
            indim = outdim
        layers.append(nn.ReLU())
        layers.append(nn.Linear(indim, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        logit = self.sequential(x)
        return logit

    def loss_func(self, target, logit):
        loss = F.binary_cross_entropy_with_logits(logit, target)
        return loss


class Gan(nn.Module):
    """Base GAN class."""

    def __init__(self, xdim, zdim, generator, generatorhidden, discriminator, discriminatorhidden):
        super().__init__()
        self.generator = generator(zdim, generatorhidden, xdim)
        self.discriminator = discriminator(xdim, discriminatorhidden)

    def sample(self, nsamples):
        self.eval()
        with torch.no_grad():
            sample = self.generator.sample(nsamples)
        return sample

    def discriminate(self, samples):
        self.eval()
        with torch.no_grad():
            logits = self.discriminator(samples)
            outputs = torch.sigmoid(logits)
        return outputs



