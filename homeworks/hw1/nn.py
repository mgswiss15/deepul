"""Layers and models defined for hw1."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw1.utils import rescale, descale
from torch.nn.modules.utils import _pair


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE."""

    def __init__(self, in_features, out_features, n_dims, bias=True, prev_mk=None):
        super().__init__(in_features, out_features, bias=False)
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
    """Masked direct input-output linear layer for MADE."""

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


class MaskedConv2dSingle(nn.Conv2d):
    """Masked 2D convolution type B for single color channel as defined in PixelCNN paper."""

    def __init__(self, masktype, in_channels, out_channels, kernel_size, n_classes=0):
        if masktype not in ['A', 'B']:
            raise Exception(f"Mask type has to be A or B, not {masktype}.")
        self.n_classes = n_classes
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid] = 1.
        if masktype == 'B':
            mask[:, :, mid, mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)
        if n_classes > 0:
            self.condbias = nn.Parameter(torch.Tensor(out_channels, n_classes))
            nn.init.kaiming_uniform_(self.condbias)

    def forward(self, input):
        data = input[0] if isinstance(input, list) else input
        conv = self._conv_forward(data, self.weight * self.mask)
        if self.n_classes > 0:
            cond = F.linear(input[1], self.condbias, bias=None)
            return conv + cond[:, :, None, None]
        return conv


class PixelCNN(nn.Module):
    """PixelCNN model for single color masked convolutions."""

    def __init__(self, in_channels, n_filters, kernel_size, n_layers, colcats=2, n_classes=0):
        super().__init__()
        self.colcats = colcats
        self.n_classes = n_classes
        layers = [MaskedConv2dSingle('A', in_channels, n_filters, kernel_size, n_classes)]
        for _ in range(n_layers):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2dSingle('B', n_filters, n_filters, kernel_size, n_classes))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dSingle('B', n_filters, n_filters, 1, n_classes))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dSingle('B', n_filters, n_filters, 1, n_classes))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2dSingle('B', n_filters, in_channels, 1, n_classes))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # batches are lists of data, labels in conditional models so take just data
        logits = x[0] if isinstance(x, list) else x
        for layer in self.layers:
            if isinstance(layer, MaskedConv2dSingle) and self.n_classes > 0:
                logits = layer([logits, x[1]])
            else:
                logits = layer(logits)
        return logits

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w = image_shape
            samples = torch.bernoulli(torch.ones(n_samples, 1, h, w))
            samples = rescale(samples, 0., self.colcats - 1.).to(device)
            if self.n_classes > 0:
                labels = torch.arange(self.n_classes).repeat_interleave(n_samples // self.n_classes)
                labels = F.one_hot(labels, self.n_classes).to(device, dtype=torch.float)
            for hi in range(h):
                for wi in range(w):
                    data = [samples, labels] if self.n_classes > 0 else samples
                    logits = self(data)
                    samples[:, 0, hi, wi] = torch.bernoulli(torch.sigmoid(logits))[:, 0, hi, wi]
                    samples[:, 0, hi, wi] = rescale(samples[:, 0, hi, wi], 0., self.colcats - 1.)
        return descale(samples.permute(0, 2, 3, 1), 0., self.colcats - 1.)


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution as defined in PixelCNN paper."""

    def __init__(self, masktype, in_channels, out_channels, kernel_size, indepchannels=False):
        if masktype not in ['A', 'B']:
            raise Exception(f"Mask type has to be A or B, not {masktype}.")
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=True)
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
        if masktype == 'B' and indepchannels:
            mask[:, :, mid, mid] = 1.
            # mask[redout:greenout, redin:greenin, mid, mid] = 1.
            # mask[greenout:, greenin:, mid, mid] = 1.
        elif masktype == 'B':
            mask[:, :redin, mid, mid] = 1.
            mask[redout:, redin:greenin, mid, mid] = 1.
            mask[greenout:, greenin:, mid, mid] = 1.
        elif masktype == 'A' and not indepchannels:
            mask[redout:, :redin, mid, mid] = 1.
            mask[greenout:, redin:greenin, mid, mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)


class ResBlock(nn.Module):
    """Residual block for PixelCNN.

    NOTE: The forward step here is strange.
    Should not it first do the conv and then the relu?
    I guess depends on what comes in.
    """

    def __init__(self, in_channels, kernel_size, indepchannels=False):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = MaskedConv2d('B', in_channels, mid_channels, 1, indepchannels=indepchannels)
        self.conv2 = MaskedConv2d('B', mid_channels, mid_channels, kernel_size, indepchannels=indepchannels)
        self.conv3 = MaskedConv2d('B', mid_channels, in_channels, 1, indepchannels=indepchannels)

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
        return x
        n, c, h, w = x.shape
        # careful here about splitting colour channels as in loss_func
        x = x.view(n, 3, c // 3, h, w).permute(0, 1, 3, 4, 2)
        x = super().forward(x)
        return x.permute(0, 1, 4, 2, 3).reshape(n, c, h, w)


class PixelCNNResidual(nn.Module):
    """PixelCNN model with residual blocks."""

    def __init__(self, in_channels, n_filters, kernel_size, n_resblocks, colcats, indepchannels=False):
        super().__init__()
        self.colcats = colcats
        self.indepchannels = indepchannels
        layers = []
        layers = [MaskedConv2d('A', in_channels, n_filters, kernel_size, indepchannels=indepchannels)]
        for _ in range(n_resblocks):
            layers.append(LayerNorm(n_filters // 3))
            layers.append(ResBlock(n_filters, kernel_size, indepchannels=indepchannels))
        for _ in range(2):
            layers.append(LayerNorm(n_filters // 3))
            layers.append(nn.ReLU())
            layers.append(MaskedConv2d('B', n_filters, n_filters, 1, indepchannels=indepchannels))
        layers.append(LayerNorm(n_filters // 3))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2d('B', n_filters, in_channels*colcats, 1, indepchannels=indepchannels))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # batches are lists of data, labels in conditional models so take just data
        x = x[0] if isinstance(x, list) else x
        return self.sequential(x)

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w, c = image_shape
            samples = torch.multinomial(torch.ones(self.colcats)/self.colcats,
                                        n_samples*c*h*w, replacement=True)
            samples = samples.view(n_samples, c, h, w)
            samples = rescale(samples, 0., self.colcats - 1.).to(device)
            if self.indepchannels:
                print(f"Sampling new examples with independent channels ...", flush=True)
                print(f"Rows: ", end="", flush=True)
                for hi in range(h):
                    print(f"{hi}", end=" ", flush=True)
                    for wi in range(w):
                        logits = self(samples)[:, :, hi, wi].squeeze()
                        logits = logits.view(n_samples, c, self.colcats)
                        probs = logits.softmax(dim=2)
                        for ci in range(c):
                            samples[:, ci, hi, wi] = torch.multinomial(probs[:, ci, :], 1).squeeze()
                        samples[:, :, hi, wi] = rescale(samples[:, :, hi, wi], 0., self.colcats - 1.)
                print(f"", flush=True)  # print newline symbol after all rows
            else:
                print(f"Sampling new examples with dependent channels ...", flush=True)
                print(f"Rows: ", end="", flush=True)
                for hi in range(h):
                    print(f"{hi}", end=" ", flush=True)
                    for wi in range(w):
                        for ci in range(c):
                            logits = self(samples)[:, :, hi, wi].squeeze()
                            logits = logits.view(n_samples, c, self.colcats)[:, ci, :].squeeze()
                            probs = logits.softmax(dim=1)
                            samples[:, ci, hi, wi] = torch.multinomial(probs, 1).squeeze()
                            samples[:, ci, hi, wi] = rescale(samples[:, ci, hi, wi], 0., self.colcats - 1.)
                print(f"", flush=True)  # print newline symbol after all rows
        return descale(samples.permute(0, 2, 3, 1), 0., self.colcats - 1.)


class Conv2dVertical(nn.Conv2d):
    """Masked 2D convolution for vertical stack as defined in Conditional
    Image Generation with PixelCNN Decoders."""

    def __init__(self, in_channels, out_channels, kernel_size, mask):
        padding = kernel_size // 2
        self.mask = mask
        super().__init__(in_channels, out_channels, (padding, kernel_size), padding=padding, bias=False)
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight2)

    def forward(self, input):
        a1 = F.conv2d(input, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        a2 = F.conv2d(input, self.weight2, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        h = input.shape[2]
        return a1[:, :, :h, :], a2[:, :, :h, :]


class Conv2dHorizontal(nn.Conv2d):
    """Masked 2D convolution for horizontal stack as defined in Conditional
    Image Generation with PixelCNN Decoders."""

    def __init__(self, masktype, in_channels, out_channels, kernel_size):
        self.masktype = masktype
        if masktype not in ['A', 'B']:
            raise Exception(f"Mask type has to be A or B, not {masktype}.")
        padding = kernel_size // 2
        ksize = kernel_size // 2
        ksize += 1 if masktype == 'B' else 0
        super().__init__(in_channels, out_channels, (1, ksize), padding=(0, padding), bias=False)
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight2)

    def forward(self, input):
        a1 = F.conv2d(input, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        a2 = F.conv2d(input, self.weight2, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        w = input.shape[3]
        return a1[:, :, :, :w], a2[:, :, :, :w]

class MaskedConv2dVertical(nn.Conv2d):
    """Masked 2D convolution for vertical stack as defined in Conditional
    Image Generation with PixelCNN Decoders."""

    def __init__(self, in_channels, out_channels, kernel_size, mask):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.mask = mask
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight2)

    def forward(self, input):
        a1 = F.conv2d(input, self.weight*self.mask, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        a2 = F.conv2d(input, self.weight2*self.mask, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        return a1, a2


class MaskedConv2dHorizontal(nn.Conv2d):
    """Masked 2D convolution for horizontal stack as defined in Conditional
    Image Generation with PixelCNN Decoders."""

    def __init__(self, in_channels, out_channels, kernel_size, mask):
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.mask = mask
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight2)

    def forward(self, input):
        a1 = F.conv2d(input, self.weight*self.mask, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        a2 = F.conv2d(input, self.weight2*self.mask, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
        return a1, a2


class GatedLayer(nn.Module):
    """Gated convolution layer as defined in Conditional Image Generation with PixelCNN Decoders """

    def __init__(self, masktype, n_filters, kernel_size, masks):
        super().__init__()
        maskV, maskHA, maskHB = masks
        if masktype not in ['A', 'B']:
            raise Exception(f"Mask type has to be A or B, not {masktype}.")
        maskH = maskHA if masktype == 'A' else maskHB
        self.vertical = MaskedConv2dVertical(n_filters, n_filters, kernel_size, maskV)
        self.horizontal = MaskedConv2dHorizontal(n_filters, n_filters, kernel_size, maskH)
        self.oneconv1 = nn.Conv2d(2*n_filters, 2*n_filters, 1, bias=False)
        self.oneconv2 = nn.Conv2d(n_filters, n_filters, 1, bias=False)

    def forward(self, vert, horiz):
        va1, va2 = self.vertical(vert)
        connect = self.oneconv1(torch.cat((va1, va2), dim=1))
        c1, c2 = torch.chunk(connect, 2, dim=1)
        ha1, ha2 = self.horizontal(horiz)
        ha1, ha2 = ha1+c1, ha2+c2
        vertout = torch.tanh(va1) * torch.sigmoid(va2)
        horizout = torch.tanh(ha1) * torch.sigmoid(ha2)
        horizout = self.oneconv2(horizout) + horiz
        return vertout, horizout


class PixelCNNGated(nn.Module):
    """PixelCNN model with residual blocks."""

    def __init__(self, in_channels, n_filters, kernel_size, n_layers, colcats):
        super().__init__()
        self.colcats = colcats
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        masks = self.makemasks()
        self.start = nn.Conv2d(in_channels, n_filters, 1)
        layers = [GatedLayer('A', n_filters, kernel_size, masks)]
        for _ in range(n_layers):
            layers.append(GatedLayer('B', n_filters, kernel_size, masks))
        self.gated = nn.ModuleList(layers)
        self.prefinal = nn.Conv2d(n_filters, n_filters, 1)
        self.final = nn.Conv2d(n_filters, in_channels*colcats, 1)

    def makemasks(self):
        kernel_size = nn.modules.utils._pair(self.kernel_size)
        print('-------------', kernel_size)
        mask = torch.zeros(self.n_filters, self.n_filters, *kernel_size)
        if (kernel_size[0] % 2) == 0:
            raise Exception(f"Invalid kernel_size {kernel_size}, has to be odd.")
        else:
            mid = kernel_size[0] // 2
        mask[:, :, :mid, :] = 1.
        maskV = nn.Parameter(mask, requires_grad=False)
        mask = torch.zeros_like(mask)
        mask[:, :, mid, :mid] = 1
        maskHA = nn.Parameter(mask, requires_grad=False)
        mask = torch.zeros_like(mask)
        mask[:, :, mid, :mid+1] = 1
        maskHB = nn.Parameter(mask, requires_grad=False)
        return maskV, maskHA, maskHB

    def forward(self, x):
        # batches are lists of data, labels in conditional models so take just data
        x = x[0] if isinstance(x, list) else x
        x = self.start(x)
        vert, horiz = x, x
        for layer in self.gated:
            vert, horiz = layer(vert, horiz)
        x = F.relu(self.prefinal(horiz))
        return self.final(x)

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w, c = image_shape
            samples = torch.multinomial(torch.ones(self.colcats)/self.colcats,
                                        n_samples*c*h*w, replacement=True)
            samples = samples.reshape(n_samples, c, h, w)
            samples = rescale(samples, 0., self.colcats - 1.).to(device)
            print(f"Sampling new examples with dependent channels ...", flush=True)
            print(f"Rows: ", end="", flush=True)
            for hi in range(h):
                print(f"{hi}", end=" ", flush=True)
                for wi in range(w):
                    logits = self(samples)[:, :, hi, wi].squeeze()
                    logits = logits.view(n_samples, c, self.colcats)
                    probs = logits.reshape(n_samples*c, self.colcats).softmax(dim=1)
                    samples_flat = torch.multinomial(probs, 1).squeeze()
                    samples_flat = samples_flat.view(n_samples, c)
                    samples[:, :, hi, wi] = rescale(samples_flat, 0., self.colcats - 1.)
            print(f"", flush=True)  # print newline symbol after all rows
        return descale(samples.permute(0, 2, 3, 1), 0., self.colcats - 1.)
