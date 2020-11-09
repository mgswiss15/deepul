"""Layers and models Real NVP part of hw2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw2.utils import rescale, descale, jitter


def dequantize(x, colcats, alpha=0.05, forward=True):
    """Logit gequantization from 4.1 of RealNVP."""

    n, h, w, c = x.shape
    n_dims = h*w*c

    def logit(z):
        n = z.shape[0]
        out = z.log() - (1 - z).log()
        logjacobs = (1/z + 1/(1-z)).view(n, -1).sum(dim=1)
        return out, logjacobs

    minx, _ = logit(torch.tensor([alpha, ]))
    maxx, _ = logit(torch.tensor([1 - alpha, ]))
    if forward:
        x = x.float()
        x = jitter(x, colcats)
        x = alpha + (1. - 2*alpha) * x
        logjacobs = torch.tensor([- 2*alpha*n_dims, ])
        x, ljd = logit(x)
        logjacobs += ljd.mean()
        x, ljd = rescale(x, minx, maxx)
        logjacobs += ljd
        return x.permute(0, 3, 1, 2), logjacobs
    else:
        x = descale(x, minx, maxx)
        x = torch.sigmoid(x)
        x = (x-alpha)/(1-alpha)
        return x.permute(0, 2, 3, 1)


class ResBlock(nn.Module):
    """Convolutional residual block as in PixelCNN."""

    def __init__(self, n_filters, k_size):
        super().__init__()
        if (n_filters % 2) != 0:
            raise Exception(f"n_filters must be even, not {n_filters}.")
        half_filters = n_filters // 2
        self.conv0 = nn.Conv2d(n_filters, half_filters, 1)
        self.conv1 = nn.Conv2d(half_filters, half_filters, k_size, padding=(k_size//2))
        self.conv2 = nn.Conv2d(half_filters, n_filters, 1)

    def forward(self, x):
        h = F.relu(self.conv0(x))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        return h + x


class AffineCoupling(nn.Module):
    """Affine coupling for Real NVP."""

    def __init__(self, mask, conditioner, img_shape):
        super().__init__()
        self.register_buffer('mask', mask)
        self.conditioner = conditioner
        self.scale = nn.Parameter(torch.ones(img_shape))
        self.shift = nn.Parameter(torch.ones(img_shape))

    def forward(self, x):
        logscale, t = self.get_params(x)
        z = x * self.mask + ~self.mask * ((x * logscale.exp()) + t)
        logjacobs = (~self.mask * logscale).sum(dim=(1, 2, 3))
        return z, logjacobs

    def reverse(self, z):
        logscale, t = self.get_params(z)
        x = z * self.mask + ~self.mask * (z - t) * (-logscale).exp()
        return x

    def get_params(self, data):
        params = self.conditioner(data * self.mask)
        t, s = torch.chunk(params, 2, dim=1)
        logscale = (self.scale * torch.tanh(s) + self.shift)
        return logscale, t


class SimpleResnet(nn.Module):
    """Simple resnet for RealNVP."""

    def __init__(self, in_channels, n_filters, k_size, n_blocks, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, n_filters, k_size, padding=(k_size//2))
        resblocks = [nn.ReLU()]
        for _ in range(n_blocks):
            resblocks.append(ResBlock(n_filters, k_size))
        self.resblocks = nn.Sequential(*resblocks)
        self.convL = nn.Conv2d(n_filters, out_channels, k_size, padding=(k_size//2))

    def forward(self, x):
        x = self.conv0(x)
        x = self.resblocks(x)
        return self.convL(x)


class Squeeze(nn.Module):
    """Squeeze layer from Real NVP."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        n, c, h, w = x.shape
        if (h % 2) != 0 or (w % 2) !=0:
            raise Exception(f"Height and width have to be even, not {h} and {w}.")
        x = x.view(n, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(n, c * 4, h // 2, w // 2)
        return x

    @staticmethod
    def reverse(z):
        z = Unsqueeze().forward(z)
        return z


class Unsqueeze(nn.Module):
    """Unsqueeze layer from Real NVP."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        n, c, h, w = x.shape
        if (c % 4) != 0:
            raise Exception(f"Channels have to be divisible by 4 not {c}.")
        x = x.view(n, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(n, c // 4, h * 2, w * 2)
        return x

    @staticmethod
    def reverse(z):
        z = Squeeze().forward(z)
        return z


class ActNorm(nn.Module):
    """ActNorm as in Glow paper."""

    def __init__(self, n_filters, img_shape):
        super().__init__()
        self.n_filters = n_filters
        self.img_shape = img_shape
        self.logscale = nn.Parameter(torch.zeros(n_filters))
        self.shift = nn.Parameter(torch.zeros(n_filters))

    def forward(self, x):
        x = self.logscale[None, :, None, None].exp() * x + self.shift[None, :, None, None]
        logjacobs = self.img_shape[0]*self.img_shape[1]*self.logscale.sum(dim=0, keepdim=True)
        return x, logjacobs[None, :]

    def reverse(self, z):
        z = (z - self.shift[None, :, None, None]) * (-self.logscale).exp()[None, :, None, None]
        return z


class RealNVP(nn.Module):
    """Real NVP model."""

    def __init__(self, in_channels, n_filters, k_size, n_blocks, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.checkermask = self.get_checkermask(img_shape)
        self.channelmask = self.get_channelmask(in_channels * 4)
        layers = []
        mask = self.checkermask[None, None, :, :]
        for _ in range(4):
            conditioner = SimpleResnet(in_channels, n_filters, k_size, n_blocks, in_channels * 2)
            layers.append(AffineCoupling(mask, conditioner, img_shape))
            mask = ~mask
            layers.append(ActNorm(in_channels, img_shape))
        layers.append(Squeeze())
        mask = self.channelmask[None, :, None, None]
        for _ in range(3):
            conditioner = SimpleResnet(in_channels * 4, n_filters, k_size, n_blocks, in_channels * 4 * 2)
            layers.append(AffineCoupling(mask, conditioner, (img_shape[0] // 2, img_shape[1] // 2)))
            mask = ~mask
            layers.append(ActNorm(in_channels * 4, img_shape))
        layers.append(Unsqueeze())
        mask = self.checkermask[None, None, :, :]
        for _ in range(3):
            conditioner = SimpleResnet(in_channels, n_filters, k_size, n_blocks, in_channels * 2)
            layers.append(AffineCoupling(mask, conditioner, img_shape))
            mask = ~mask
            layers.append(ActNorm(in_channels, img_shape))
        self.transforms = nn.ModuleList(layers)

    def forward(self, data):
        logjacobs = 0
        for transform in self.transforms:
            if isinstance(transform, Squeeze) or isinstance(transform, Unsqueeze):
                data = transform(data)
            else:
                data, lj = transform(data)
                logjacobs = logjacobs + lj
        return data, logjacobs

    def reverse(self, data):
        for transform in reversed(self.transforms):
            data = transform.reverse(data)
        return data

    def get_checkermask(self, img_shape):
        h, w = img_shape
        mask = torch.tensor([[True, False], [False, True]])
        mask = mask.repeat(h // 2, w//2)
        return mask

    def get_channelmask(self, n_channels):
        mask = torch.tensor([True, True, False, False])
        mask = mask.repeat(n_channels // 4)
        return mask

    def sample_data(self, n_samples, device):
        print(f"Sampling ...")
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, 3, *self.img_shape).to(device)
            samples = self.reverse(z)
        return samples
