"""Layers and modules for bigan."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Generator(nn.Module):
    """MLP generator for BiGAN."""

    def __init__(self, xdim, zdim, hidden):
        super().__init__()
        self.zdim = zdim
        self.xdim = xdim
        layers = [nn.Linear(zdim, hidden)]
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.BatchNorm1d(hidden, affine=False))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, xdim[0]*xdim[1]))
        layers.append(nn.Tanh())
        self.sequential = nn.Sequential(*layers)

    def forward(self, z):
        x = self.sequential(z)
        x = x.view(-1, self.xdim[0], self.xdim[1])
        return x

    def sample(self, nsamples):
        device = next(self.parameters()).device
        z = torch.randn(nsamples, self.zdim).to(device)
        x = self(z)
        return x, z


class Encoder(nn.Module):
    """MLP encoder for BiGAN."""

    def __init__(self, xdim, zdim, hidden):
        super().__init__()
        layers = [nn.Linear(xdim[0]*xdim[1], hidden)]
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.BatchNorm1d(hidden, affine=False))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden, zdim))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, -1)
        z = self.sequential(x)
        return z


class Discriminator(nn.Module):
    """MLP discriminator for BiGAN."""

    def __init__(self, xdim, zdim, hidden):
        super().__init__()
        layers = [nn.Linear(xdim[0]*xdim[1]+zdim, hidden)]
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.BatchNorm1d(hidden, affine=False))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x, z):
        n = x.shape[0]
        x = x.view(n, -1)
        x = torch.cat((x, z), dim=1)
        logit = self.sequential(x)
        return logit


class BiGan(nn.Module):
    """BiGan as in Donague et al 2017: Adversarial feature learning."""

    def __init__(self, xdim, zdim, hidden):
        super().__init__()
        self.generator = Generator(xdim, zdim, hidden)
        self.encoder = Encoder(xdim, zdim, hidden)
        self.discriminator = Discriminator(xdim, zdim, hidden)

    def forward(self, data):
        n = data.shape[0]
        zhat = self.encoder(data)
        datahat, z = self.generator.sample(n)
        zhatc = zhat.clone().detach()
        datahatc = datahat.clone().detach()
        logitsencoder = self.discriminator(data, zhatc)
        logitsgenerator = self.discriminator(datahatc, z)
        logits = torch.cat((logitsencoder, logitsgenerator), dim=0)
        targets = torch.cat((logitsencoder.new_ones(logitsencoder.shape),
                             logitsgenerator.new_zeros(logitsgenerator.shape)), dim=0)
        # lossD = (-logitsencoder.log() - (1 - logitsgenerator).log()).mean()
        lossD = F.binary_cross_entropy_with_logits(logits, targets)
        logitsencoder = self.discriminator(data, zhat)
        logitsgenerator = self.discriminator(datahat, z)
        logits = torch.cat((logitsencoder, logitsgenerator), dim=0)
        targets = torch.cat((logitsencoder.new_zeros(logitsencoder.shape),
                             logitsgenerator.new_ones(logitsgenerator.shape)), dim=0)
        lossGE = F.binary_cross_entropy_with_logits(logits, targets)
        # lossGE = (-(1 - logitsencoder).log() - logitsgenerator.log()).mean()
        return lossD, lossGE

    def sample(self, nsamples):
        self.eval()
        with torch.no_grad():
            samples, _ = self.generator.sample(nsamples)
        return samples[:, None, :, :]

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            z = self.encoder(data)
            x = self.generator(z)
        return x[:, None, :, :]


class BiGanLearner():
    """Class for training BiGAN."""

    def __init__(self, model, optimizer, trainloader, device, callback_list=[]):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.device = device
        self.callback_list = callback_list
        self.batchsize = self.trainloader.batch_size
        self.epoch = 0
        self.losses = {'D': [], 'GE': []}
        for cb in self.callback_list:
            cb.init_learner(self)

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            for batch in self.trainloader:
                self.updates(batch)
            self.callback('epoch_end')
        self.callback('fit_end')
        return self.losses['D']

    def updates(self, batch):
        """Careful here about the order of backward and optimizer updates."""
        self.model.train()
        self.callback('batch_begin')
        batch = batch[0].to(self.device)
        lossD, lossGE = self.model(batch)       
        self.losses['D'].append(lossD.item())
        self.losses['GE'].append(lossGE.item())
        self.optimizer['generator'].zero_grad()
        self.optimizer['encoder'].zero_grad()
        lossGE.backward()
        self.optimizer['generator'].step()
        self.optimizer['encoder'].step()
        self.optimizer['discriminator'].zero_grad()
        lossD.backward()
        self.optimizer['discriminator'].step()
        self.callback('batch_end')

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)
