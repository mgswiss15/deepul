import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """2D conv layer with masking from PixelCNN (van den Oord, 2016).

    Autoregressive through masking previous pixels in conv operations.
    This is single channel only so does not do RGB autoregressive structure.

    Args:
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution
    kernel_size (int or tuple): Size of the convolving kernel

    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)
        # masking matrix
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        cp = kernel_size // 2
        mask[:, :, cp+1:,:] = 0. # rows below
        mask[:, :, cp, cp:] = 0. # itself and everything to the right
        self.register_buffer('mask', mask)
        self.padding = 'same'

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.mask, self.bias)

class PixelCnn(nn.Module):
    """PixelCNN reimplemented from van den Oord (2016).

    Ensures autoregressive masking through maksing kernels.
    This is single channel only so does not do RGB autoregressive structure.

    Args:
        channels (list): output channels in conv2d
        kernels (list): kernel sizes in conv2d
    """
    def __init__(self, channels, kernels):
        super().__init__()
        assert len(channels) == len(kernels), f'Lists channels and kernels are unequel length'
        self.depth = len(channels)

        # conv layers with relus
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        in_channels = 1
        for i in range(self.depth):
            conv2d = MaskedConv2d(in_channels, channels[i], kernel_size=kernels[i])
            conv2d_relu = nn.Sequential(conv2d, nn.ReLU())
            self.layers.append(conv2d_relu)
            skip = nn.Sequential(MaskedConv2d(in_channels, channels[-1] // 2, kernel_size=kernels[i]), nn.ReLU())
            self.skips.append(skip)
            in_channels = channels[i]
        self.pre_final = nn.Sequential(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1), nn.ReLU())
        self.final = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        # recode to [-1, 1]
        x = torch.where(x == 0., -1., 1.)
        skips = []
        for i in range(self.depth):
            # skips.append(self.skips[i](x))
            x = self.layers[i](x)
            x = self.pre_final(x)
        # skip connections
        # x = x + torch.sum(torch.stack(skips), dim=0)
        x = self.final(x)
        return x


class ResBlock(nn.Module):
    """ResBlock as in PixelCNN (van den Oord, 2016))"""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            MaskedConv2d(channels // 2, channels // 2, 3),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class PixelCnn_Residual(nn.Module):
    """PixelCNN with residual blocks reimplemented from van den Oord (2016).

    Ensures autoregressive masking through maksing kernels.
    This is single channel only so does not do RGB autoregressive structure.

    Args:
        channels (int): output channels in conv2d
        kernel (int): kernel sizes in conv2d
        resblocks (int): number of resblocks
    """
    def __init__(self, channels, kernel, resblocks):
        super().__init__()

        # conv layers with relus
        self.first = MaskedConv2d(1, channels, kernel_size=kernel) 
        self.layers = nn.ModuleList()
        self.resblocks = resblocks
        for i in range(resblocks):
            self.layers.append(ResBlock(channels))
        self.pre_final = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels // 2, kernel_size=1))
        self.final = nn.Sequential(nn.ReLU(), nn.Conv2d(channels // 2, 1, kernel_size=1))

    def forward(self, x):
        # recode to [-1, 1]
        x = torch.where(x == 0., -1., 1.)
        x = self.first(x)
        for i in range(self.resblocks):
            x = self.layers[i](x)
        x = self.pre_final(x)
        x = self.final(x)
        return x