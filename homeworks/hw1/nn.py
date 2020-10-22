"""Layers and models defined for hw1."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw1.utils import rescale, descale


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE."""

    def __init__(self, in_features, out_features, n_dims, bias=True, prev_mk=None):
        super().__init__(in_features, out_features, bias)
        self.mk = torch.randint(prev_mk.min(), n_dims, (out_features, 1))
        mask = (self.mk >= prev_mk.T).squeeze()
        # make mask parameter so can be moved to correct device by model.to(device)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class MaskedLinearOutput(nn.Linear):
    """Masked output linear layer for MADE."""

    def __init__(self, in_features, out_features, n_dims, bias=True, prev_mk=None):
        super().__init__(in_features, out_features, bias)
        # make mask
        mk = (torch.arange(n_dims) + 1).repeat_interleave(int(out_features / n_dims))[:, None]
        mask = (mk > prev_mk.T).squeeze()
        # make mask parameter so can be moved to correct device by model.to(device)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class MaskedLinearInOut(nn.Linear):
    """Masked directn input-output linear layer for MADE."""

    def __init__(self, in_features, n_dims, bias=True):
        super().__init__(in_features, in_features, bias)
        # make mask
        mk = (torch.arange(n_dims) + 1).repeat_interleave(int(in_features / n_dims))[:, None]
        mask = (mk > mk.T).squeeze()
        # make mask parameter so can be moved to correct device by model.to(device)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)


class Made(nn.Module):
    """Made model."""

    def __init__(self, in_features, hidden, n_dims):
        super().__init__()
        self.in_features = in_features
        prev_mk = (torch.arange(n_dims) + 1).repeat_interleave(int(in_features / n_dims))[:, None]
        layers = [MaskedLinear(in_features, hidden[0], n_dims, prev_mk=prev_mk)]
        mk = layers[-1].mk
        for lidx, out_features in enumerate(hidden[1:]):
            layers.append(nn.ReLU())
            layers.append(MaskedLinear(hidden[lidx], out_features, n_dims, prev_mk=mk))
            mk = layers[-1].mk
        layers.append(nn.ReLU())
        layers.append(MaskedLinearOutput(hidden[-1], in_features, n_dims, prev_mk=mk))
        self.sequential = nn.Sequential(*layers)
        self.inout = MaskedLinearInOut(in_features, n_dims)

    def forward(self, x):
        return self.sequential(x) + self.inout(x)


class MaskedConv2dBSingle(nn.Conv2d):
    """Masked 2D convolution type B for single color channel as defined in PixelCNN paper."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid+1] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)


class MaskedConv2dASingle(nn.Conv2d):
    """Masked 2D convolution type A for single colour channel as defined in PixelCNN paper."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)


class PixelCNN(nn.Module):
    """PixelCNN model for single color masked convolutions."""

    def __init__(self, in_channels, n_filters, kernel_size, n_layers):
        super().__init__()
        layers = [MaskedConv2dASingle(in_channels, n_filters, kernel_size)]
        for _ in range(n_layers):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2dBSingle(n_filters, n_filters, kernel_size))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dBSingle(n_filters, n_filters, 1))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dBSingle(n_filters, n_filters, 1))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dBSingle(n_filters, in_channels, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class MaskedConv2dB(nn.Conv2d):
    """Masked 2D convolution type B as defined in PixelCNN paper."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, indepchannels=False):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        if (self.kernel_size[0] % 2) == 0:
            raise Exception(f"Invalid kernel_size {kernel_size}, has to be odd.")
        else:
            mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid] = 1.
        redin = in_channels // 3
        redout = out_channels // 3
        if (redin != in_channels / 3):
            raise Exception(f"Invalid in_channels {in_channels}, has to be divisible by 3.")
        if (redout != out_channels / 3):
            raise Exception(f"Invalid out_channels {out_channels}, has to be divisible by 3.")
        greenin = 2 * redin
        greenout = 2 * redout
        if indepchannels:
            mask[:redin, :redout, mid, mid] = 1.
            mask[redin:greenin, redout:greenout, mid, mid] = 1.
            mask[greenin:, greenout:, mid, mid] = 1.
        else:
            mask[:redin, :, mid, mid] = 1.
            mask[redin:greenin, redout:, mid, mid] = 1.
            mask[greenin:, greenout:, mid, mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)


class MaskedConv2dA(nn.Conv2d):
    """Masked 2D convolution type A as defined in PixelCNN paper."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, indepchannels=False):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        if (self.kernel_size[0] % 2) == 0:
            raise Exception(f"Invalid kernel_size {kernel_size}, has to be odd.")
        else:
            mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid] = 1.
        if not indepchannels:
            redin = in_channels // 3
            redout = out_channels // 3
            if (redin != in_channels / 3):
                raise Exception(f"Invalid in_channels {in_channels}, has to be divisible by 3.")
            if (redout != out_channels / 3):
                raise Exception(f"Invalid out_channels {out_channels}, has to be divisible by 3.")
            greenin = 2 * redin
            greenout = 2 * redout
            mask[:redin, redout:, mid, mid] = 1.
            mask[redin:greenin, greenout:, mid, mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)


class ResBlock(nn.Module):
    """Residual block for PixelCNN."""

    def __init__(self, in_channels, kernel_size, indepchannels=False):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = MaskedConv2dB(in_channels, mid_channels, 1, indepchannels=indepchannels)
        self.conv2 = MaskedConv2dB(mid_channels, mid_channels, kernel_size, indepchannels=indepchannels)
        self.conv3 = MaskedConv2dB(mid_channels, in_channels, 1, indepchannels=indepchannels)

    def forward(self, x):
        y = self.conv1(F.relu(x))
        y = self.conv2(F.relu(y))
        y = self.conv3(F.relu(y))
        return x + y


class LayerNorm(nn.LayerNorm):
    """Layer norm for masked conv2d in PixelCNN."""

    def __init__(self, normalized_shape):
        super().__init__(normalized_shape, elementwise_affine=False)

    def forward(self, x):
        n, c, h, w = x.shape
        # careful here about splitting colour channels as in loss_func
        x = x.view(n, c // 3, 3, h, w).permute(0, 2, 3, 4, 1)
        x = super().forward(x)
        return x.permute(0, 4, 1, 2, 3).reshape(n, c, h, w)


class PixelCNNResidual(nn.Module):
    """PixelCNN model with residual blocks."""

    def __init__(self, in_channels, n_filters, kernel_size, n_resblocks, colcats, indepchannels=False):
        super().__init__()
        self.colcats = colcats
        self.indepchannels = indepchannels
        layers = []
        layers = [MaskedConv2dA(in_channels, n_filters, kernel_size, indepchannels=indepchannels)]
        for _ in range(n_resblocks):
            layers.append(LayerNorm(n_filters // 3))
            layers.append(ResBlock(n_filters, kernel_size, indepchannels=indepchannels))
        for _ in range(2):
            layers.append(LayerNorm(n_filters // 3))
            layers.append(nn.ReLU())
            layers.append(MaskedConv2dB(n_filters, n_filters, 1, indepchannels=indepchannels))
        layers.append(LayerNorm(n_filters // 3))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dB(n_filters, in_channels*colcats, 1, indepchannels=indepchannels))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w, c = image_shape
            samples = torch.multinomial(torch.ones(self.colcats)/self.colcats,
                                        n_samples*c*h*w, replacement=True)
            samples = samples.reshape(n_samples, c, h, w).to(device, dtype=torch.float)
            samples = rescale(samples, 0., self.colcats - 1.)
            if self.indepchannels:
                print(f"Sampling new examples with independent channels ...", flush=True)
                print(f"Rows: ", end="", flush=True)
                for hi in range(h):
                    print(f"{hi}", end=" ", flush=True)
                    for wi in range(w):
                        logits = self(samples)[:, :, hi, wi].squeeze()
                        logits = logits.view(n_samples, self.colcats, c).permute(0, 2, 1)
                        logits = logits.reshape(n_samples * c, self.colcats)
                        probs = logits.softmax(dim=1)
                        samples_flat = torch.multinomial(probs, 1).squeeze()
                        samples[:, :, hi, wi] = rescale(samples_flat.view(n_samples, c), 0., self.colcats - 1.)
                print(f"", flush=True)  # print newline symbol after all rows
            else:
                print(f"Sampling new examples with dependent channels ...", flush=True)
                for ci in range(c):
                    print(f"Channel {ci}", flush=True)
                    print(f"Rows: ", end="", flush=True)
                    for hi in range(h):
                        print(f"{hi}", end=" ", flush=True)
                        for wi in range(w):
                            logits = self(samples)[:, :, hi, wi].squeeze()
                            logits = logits.view(n_samples, self.colcats, c)[:, :, ci].squeeze()
                            probs = logits.softmax(dim=1)
                            samples_flat = torch.multinomial(probs, 1).squeeze()
                            samples[:, ci, hi, wi] = rescale(samples_flat, 0., self.colcats - 1.)
                    print(f"", flush=True)  # print newline symbol after all rows
        return descale(samples.permute(0, 2, 3, 1), 0., self.colcats - 1.)
