"""Layers and modules for hw4."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class GeneratorMlp(nn.Module):
    """MLP generator for GAN."""

    def __init__(self, zdim, hidden, xdim):
        super().__init__()
        self.zdim = zdim
        hidden.append(xdim)
        layers = [nn.Linear(zdim, hidden[0])]
        indim = hidden[0]
        for outdim in hidden[1:]:
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Linear(indim, outdim))
            indim = outdim
        # layers.append(nn.Linear(indim, xdim))
        self.sequential = nn.Sequential(*layers)

    def forward(self, z):
        x = self.sequential(z)
        return x

    def sample(self, nsamples):
        device = next(self.parameters()).device
        z = torch.randn(nsamples, self.zdim).to(device)
        x = self(z)
        return x

    @staticmethod
    def loss_func(target, logit, *args):
        loss = -F.binary_cross_entropy_with_logits(logit, target)
        return loss


class GeneratorMlp2(GeneratorMlp):
    """MLP generator for GAN."""

    @staticmethod
    def loss_func(target, logit, *args):
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
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Linear(indim, outdim))
            indim = outdim
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Linear(indim, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        logit = self.sequential(x)
        return logit

    @staticmethod
    def loss_func(target, logit, *args):
        loss = F.binary_cross_entropy_with_logits(logit, target)
        return loss


class Gan(nn.Module):
    """Base GAN class."""

    def __init__(self, xdim, zdim, generator, generatorhidden, discriminator, discriminatorhidden):
        super().__init__()
        self.generator = generator(zdim, generatorhidden, xdim)
        self.discriminator = discriminator(xdim, discriminatorhidden)

    def sample(self, nsamples):
        sample = self.generator.sample(nsamples)
        return sample

    def discriminate(self, samples):
        logits = self.discriminator(samples)
        outputs = torch.sigmoid(logits)
        return outputs


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


class Upsample_Conv2d(nn.Module):
    """Spatial Upsampling with Nearest Neighbors."""

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depthtospace = DepthToSpace(block_size=2)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depthtospace(x)
        x = self.conv2d(x)
        return x


class Downsample_Conv2d(nn.Module):
    """Spatial Downsampling with Spatial Mean Pooling."""

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.spacetodepth = SpaceToDepth(block_size=2)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.spacetodepth(x)
        x = sum(torch.chunk(x, 4, dim=1)) / 4.0
        x = self.conv2d(x)
        return x


class ResnetBlockUp(nn.Module):
    """Resblock used in generator."""

    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.sequenatial = nn.Sequential(nn.BatchNorm2d(in_dim),
                                         nn.ReLU(),
                                         nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                                         nn.BatchNorm2d(n_filters),
                                         nn.ReLU(),
                                         Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
                                        )
        self.shortcut = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        y = self.sequenatial(x)
        x = self.shortcut(x)
        return y + x


class ResnetBlockDown(nn.Module):
    """Resblock used in discriminator."""

    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.sequenatial = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                                         nn.ReLU(),
                                         Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
                                        )
        self.shortcut = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        y = self.sequenatial(x)
        x = self.shortcut(x)
        return y + x


class GeneratorCifar(nn.Module):
    """Generator for cifar."""

    def __init__(self, xdim, zdim, n_filters=128):
        super().__init__()
        self.n_filters = n_filters
        self.zdim = zdim
        self.linear0 = nn.Linear(zdim, 4*4*256)
        self.sequential = nn.Sequential(ResnetBlockUp(in_dim=256, n_filters=n_filters),
                                        ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
                                        ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
                                        nn.BatchNorm2d(n_filters),
                                        nn.ReLU(),
                                        nn.Conv2d(n_filters, xdim, kernel_size=(3, 3), padding=1),
                                        nn.Tanh()
                                       )

    def forward(self, z):
        x = self.linear0(z)
        x = x.view(-1, 256, 4, 4)
        x = self.sequential(x)
        return x

    def sample(self, nsamples):
        device = next(self.parameters()).device
        z = torch.randn(nsamples, self.zdim).to(device)
        x = self(z)
        return x

    @staticmethod
    def loss_func(target, criticout, *args):
        loss = - (1 - target) * criticout
        return loss.mean()


class CriticCifar(nn.Module):
    """Critic for cifar."""

    def __init__(self, xdim, n_filters=128, lbd=10):
        super().__init__()
        self.n_filters = n_filters
        self.lbd = lbd
        self.sequential = nn.Sequential(ResnetBlockDown(in_dim=xdim, n_filters=n_filters),
                                        ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
                                        ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
                                        nn.ReLU(),
                                        nn.AvgPool2d(kernel_size=4)
                                        )
        self.linear = nn.Linear(n_filters, 1)

    def forward(self, x):
        criticout = self.sequential(x)
        criticout = self.linear(criticout.squeeze())
        return criticout, x

    def loss_func(self, target, criticout, x, *args):
        real, fake = torch.chunk(x, 2, dim=0)
        eps = torch.rand((real.shape[0], 1, 1, 1)).to(x.device)
        xinterpolate = eps*real + (1-eps)*fake.detach()
        criticinterpolate, _ = self(xinterpolate)
        grads = torch.autograd.grad(criticinterpolate.mean(), xinterpolate, create_graph=True)
        gploss = (torch.sum(grads[0]**2, dim=(1, 2, 3))**0.5 - 1)**2
        criticreal, criticfake = torch.chunk(criticout, 2, dim=0)
        criticloss = - criticreal + criticfake
        return criticloss.mean() + self.lbd * gploss.mean()


class GanCifar(nn.Module):
    """GAN for Cifar."""

    def __init__(self, xdim, zdim, n_filters, lbd):
        super().__init__()
        self.generator = GeneratorCifar(xdim, zdim, n_filters)
        self.discriminator = CriticCifar(xdim, n_filters, lbd)

    def sample(self, nsamples):
        sample = self.generator.sample(nsamples)
        return sample

    def discriminate(self, samples):
        self.eval()
        with torch.no_grad():
            criticout = self.discriminator(samples)
            criticout = torch.split(criticout, 2 * criticout.shape[0] // 3, dim=0)
        self.train()
        return criticout

