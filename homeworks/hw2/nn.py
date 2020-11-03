"""Layers and models defined for hw2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw2.utils import jitter, bisection
import math
from homeworks.hw1.nn import MaskedLinear, MaskedLinearOutput, PixelCNN, MaskedConv2dSingle


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
        self.out = MaskedLinearOutput(hidden[-1], n_dims * n_components, n_dims, prev_mk=mk)
        self.inout = MaskedLinearInOut(n_dims, n_dims * n_components)

    def forward(self, x):
        seq = self.sequential(x)
        params = self.out(seq) + self.inout(x)
        return params


class AutoregressiveFlow(nn.Module):
    """Autoregressie flow."""

    def __init__(self, n_dims, n_components, madelayers, transformertype):
        if (n_components % 3) != 0:
            raise Exception(f"Invalid n_components {n_components}, has to be divisible by 3.")
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
        params = params.view(-1, self.n_dims, 3, self.n_components // 3)
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
        params = params.view(-1, self.n_dims, 3, self.n_components // 3)
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


class Mlp(nn.Module):
    """Simple MLP as a conditioner for coupling flows."""

    def __init__(self, in_features, out_features, hidden):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        layers = []
        for out_features in hidden:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, self.out_features))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class AffineTransform(nn.Module):
    """Affine transform for flow model used in Real NVP."""

    def __init__(self, n_dims, hidden, odd=True):
        super().__init__()
        self.n_dims = n_dims
        self.odd = odd
        self.conditioner = Mlp(n_dims, n_dims * 2, hidden)
        self.register_buffer('maskodd', torch.tensor([0, 1]).repeat_interleave(n_dims // 2))
        self.register_buffer('maskeven', torch.tensor([1, 0]).repeat_interleave(n_dims // 2))
        self.scale = nn.Parameter(torch.ones(n_dims))
        self.shift = nn.Parameter(torch.zeros(n_dims))

    def forward(self, x):
        if x.shape[-1] != self.n_dims:
            raise Exception(f"Data dimension {x.shape[-1]} does not correspond to model n_dim {self.n_dims}.")
        mask = self.maskodd if self.odd else self.maskeven
        params = self.conditioner(x * mask)
        params = params.view(-1, self.n_dims, 2)
        logscale = self.scale * torch.tanh(params[..., 0]) + self.shift
        z = x * mask + (1 - mask) * (x * logscale.exp() + params[..., 1])
        return z, (1 - mask) * logscale


class RealNVP(nn.Module):
    """Simple Real NVP model."""

    def __init__(self, n_dims, hidden, n_transforms):
        super().__init__()
        transforms = []
        odd = True
        for _ in range(n_transforms):
            transforms.append(AffineTransform(n_dims, hidden, odd))
            odd = not odd
        self.transforms = nn.ModuleList(transforms)

    def forward(self, data):
        logjacobs = 0
        for transform in self.transforms:
            data, lj = transform(data)
            logjacobs = logjacobs + lj
        return data, logjacobs


class PixelCNNConditioner(PixelCNN):
    """PixelCNN conditioner for autoregressive flow with CDF transforms."""

    def __init__(self, in_channels, n_filters, kernel_size, n_layers, n_components):
        super().__init__(in_channels, n_filters, kernel_size, n_layers, colcats=2, n_classes=0)
        self.layers = self.layers[:-1]
        self.layers.append(MaskedConv2dSingle('B', n_filters, n_components, 1, self.n_classes))


class PixelCNNFlow(nn.Module):
    """Autoregressie flow."""

    def __init__(self, in_channels, n_components, n_filters, kernel_size, n_layers, transformertype):
        if (n_components % 3) != 0:
            raise Exception(f"Invalid n_components {n_components}, has to be divisible by 3.")
        super().__init__()
        self.n_components = n_components
        self.conditioner = PixelCNNConditioner(in_channels, n_filters, kernel_size, n_layers, n_components)
        self.transformer = self.transformergauss

    def forward(self, x):
        params = self.conditioner(x)
        return self.transformer(x, params)

    def transformergauss(self, x, params):
        h, w = x.shape[-2:]
        params = params.view(-1, self.n_components // 3, 3, h, w)
        pis = F.softmax(params[:, :, 0, :, :], dim=1)
        locs = params[:, :, 1, :, :]
        logscales = params[:, :, 2, :, :]
        scales = logscales.exp()
        cdfs = pis * (0.5 * (1 + torch.erf((x - locs) / (scales * 2**0.5))))
        z = cdfs.sum(dim=1, keepdim=True)
        logpdfs = -logscales - torch.tensor((math.pi*2)**0.5).log() - (x - locs)**2 / (2 * scales**2)
        logjacobs = torch.logsumexp(logpdfs + pis.log(), dim=1, keepdim=True)
        return z, logjacobs

    def sample_data(self, n_samples, image_shape, device):
        print('Sampling ...')
        self.eval()
        with torch.no_grad():
            h, w = image_shape
            samples = torch.bernoulli(torch.ones(n_samples, 1, h, w))
            samples = jitter(samples, 2.).to(device)
            for hi in range(h):
                print(f'Row {hi} ...', flush=True)
                for wi in range(w):
                    params = self.conditioner(samples)[:, :, hi, wi]
                    samples[:, 0, hi, wi] = self.invcdf(params)
        return samples.permute(0, 2, 3, 1)

    def invcdf(self, params):
        n = params.shape[0]
        params = params.view(-1, self.n_components // 3, 3)
        pis = F.softmax(params[:, :, 0], dim=1)
        locs = params[:, :, 1]
        logscales = params[:, :, 2]
        scales = logscales.exp()
        z = torch.rand((n, 1)).to("cuda")
        def cdfz(x):
            cdfs = pis * (0.5 * (1 + torch.erf((x - locs) / (scales * 2**0.5))))
            return cdfs.sum(dim=1, keepdim=True) - z
        x = bisection(cdfz, n)
        return x.squeeze()



