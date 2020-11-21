"""VQVAE modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from homeworks.hw3.utils import rescale, descale

class EncoderVqVae(nn.Module):
    """Encoder for VQVAE"""

    def __init__(self, codedim, xchannels=3):
        super().__init__()
        layers = [nn.Conv2d(xchannels, codedim, 4, 2, 1)]
        layers.append(nn.Conv2d(codedim, codedim, 4, 2, 1))
        layers.append(ResBlock(codedim))
        layers.append(ResBlock(codedim))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        out = self.sequential(x)
        return out


class DecoderVqVae(nn.Module):
    """Decoder for VQVAE"""

    def __init__(self, codedim, xchannels=3):
        super().__init__()
        layers = [ResBlock(codedim)]
        layers.append(ResBlock(codedim))
        layers.append(nn.ConvTranspose2d(codedim, codedim, 4, 2, 1))
        layers.append(nn.ConvTranspose2d(codedim, xchannels, 4, 2, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, z):
        out = self.sequential(z)
        return out


class ResBlock(nn.Module):
    """Resblock for VQVAE."""

    def __init__(self, codedim):
        super().__init__()
        layers = [nn.ReLU()]
        layers.append(nn.Conv2d(codedim, codedim, 3, 1, 1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(codedim, codedim, 1, 1, 0))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        out = self.sequential(x)
        return x + out


class VqVae(nn.Module):
    """VQVAE model."""

    def __init__(self, xchannels, codedim, ncodes):
        super().__init__()
        self.codedim = codedim
        self.ncodes = ncodes
        self.codes = nn.Parameter(torch.rand(ncodes, codedim)*(2/ncodes)-(1/ncodes))
        self.encoder = EncoderVqVae(codedim, xchannels)
        self.decoder = DecoderVqVae(codedim, xchannels)

    def forward(self, x, learnprior=False):
        zenc = self.encoder(x)
        n, c, h, w = zenc.shape
        idx = self.get_codes(zenc.detach())
        zdec = torch.index_select(self.codes, dim=0, index=idx)
        zdec = zdec.view(n, h, w, c).permute(0, 3, 1, 2)
        xhat = self.decoder((zdec - zenc).detach() + zenc)
        if learnprior:
            z = idx.view(n, 1, h, w)
            return z
        else:
            return xhat, zenc, zdec

    def get_codes(self, ze):
        n, c, h, w = ze.shape
        ze = ze.permute(0, 2, 3, 1).reshape(-1, self.codedim).expand(self.ncodes, -1, -1)
        codes = self.codes[:, None, :].expand_as(ze)
        distmat = torch.norm(ze-codes, 2, dim=2)
        minvals, minidx = distmat.min(dim=0)
        # z = minidx.view(n, 1, h, w)
        return minidx.long()

    def loss_func(self, x, xhat, ze, zd, z, beta=0.25):
        reconstruct = 0.5*((x - xhat)**2).sum(dim=(1, 2, 3)).mean()
        vq = ((ze.detach() - zd)**2).sum(dim=(1, 2, 3)).mean()
        commit = ((ze - zd.detach())**2).sum(dim=(1, 2, 3)).mean()
        return reconstruct + vq + beta*commit

    def sample(self, zsample):
        self.eval()
        with torch.no_grad():
            zdec = torch.index_select(self.codes, dim=0, index=zsample.view(-1).long())
            zdec = zdec.view(zsample.shape[0], 8, 8, self.codedim).permute(0, 3, 1, 2)
            xhat = self.decoder(zdec)
            return xhat


class MaskedConv2d(nn.Conv2d):
    """Masked 2D convolution for single color channel as defined in PixelCNN paper."""

    def __init__(self, masktype, in_channels, out_channels, kernel_size):
        if masktype not in ['A', 'B']:
            raise Exception(f"Mask type has to be A or B, not {masktype}.")
        padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        mid = self.kernel_size[0] // 2
        mask = torch.zeros_like(self.weight)
        mask[:, :, :mid, :] = 1.
        mask[:, :, mid, :mid] = 1.
        if masktype == 'B':
            mask[:, :, mid, mid] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, input):
        data = input[0] if isinstance(input, list) else input
        conv = self._conv_forward(data, self.weight * self.mask)
        return conv


class ResBlockPixelCNN(nn.Module):
    """Residual block for single channel PixelCNN."""

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = MaskedConv2d('B', in_channels, mid_channels, 1)
        self.conv2 = MaskedConv2d('B', mid_channels, mid_channels, kernel_size)
        self.conv3 = MaskedConv2d('B', mid_channels, in_channels, 1)

    def forward(self, x):
        y = self.conv1(F.relu(x))
        y = self.conv2(F.relu(y))
        y = self.conv3(F.relu(y))
        return x + y


class PixelCNN(nn.Module):
    """PixelCNN model with residual blocks."""

    def __init__(self, in_channels, n_filters, kernel_size, n_resblocks, n_cats):
        super().__init__()
        self.n_cats = n_cats
        layers = []
        layers = [MaskedConv2d('A', in_channels, n_filters, kernel_size)]
        for _ in range(n_resblocks):
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(ResBlockPixelCNN(n_filters, kernel_size))
        for _ in range(2):
            layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU())
            layers.append(MaskedConv2d('B', n_filters, n_filters, 1))
        layers.append(nn.ReLU())
        layers.append(MaskedConv2d('B', n_filters, n_cats, 1))
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        out = self.sequential(x)
        return out

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w = image_shape
            samples = torch.multinomial(torch.ones(self.n_cats)/self.n_cats,
                                        n_samples*h*w, replacement=True)
            samples = samples.view(n_samples, 1, h, w)
            samples = rescale(samples, 0., self.n_cats - 1.).to(device)
            print(f"Sampling new examples ...", flush=True)
            for hi in range(h):
                for wi in range(w):
                    logits = self(samples)[:, :, hi, wi].squeeze()
                    probs = logits.softmax(dim=1)
                    samples[:, 0, hi, wi] = torch.multinomial(probs, 1).squeeze()
                    samples[:, 0, hi, wi] = rescale(samples[:, 0, hi, wi], 0., self.n_cats - 1.)
        return samples


class GatedLayer(nn.Module):
    """Gated condovlution layer as in Oords Conditional PixelCNN 2016 paper."""

    def __init__(self, n_features, ksize):
        super().__init__()
        self.vertical1 = VerticalStack(n_features, n_features*2, ksize)
        self.verticala = VerticalStack(n_features, n_features, ksize, bias=False)
        self.verticalb = VerticalStack(n_features, n_features, ksize, bias=False)
        self.horizontal1 = HorizontalStack(n_features, n_features*2, ksize)
        self.horizontala = HorizontalStack(n_features, n_features, ksize, bias=False)
        self.horizontalb = HorizontalStack(n_features, n_features, ksize, bias=False)
        self.conv1 = nn.Conv2d(n_features*2, n_features*2, 1)
        self.conv2 = nn.Conv2d(n_features, n_features, 1)

    def forward(self, vert, horiz):
        # vertical
        vert = self.vertical1(vert)
        horiz = self.horizontal1(horiz)
        horiz = horiz + self.conv1(vert)
        v1, v2 = vert.chunk(2, dim=1)
        h1, h2 = horiz.chunk(2, dim=1)
        vert = self.verticala(v1) * self.verticalb(v2)
        horiz = self.horizontala(h1) * self.horizontalb(h2)
        horiz = self.conv2(horiz) + horiz
        return vert, horiz


class VerticalStack(nn.Conv2d):
    """Vertical stack for PixelCNN."""

    def __init__(self, in_features, out_features, ksize, bias=True):
        padding = ksize // 2
        super().__init__(in_features, out_features, ksize, 1, padding, bias=bias)
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.mask[:, :, :padding, :] = 1.

    def forward(self, x):
        out = self._conv_forward(x, self.weight * self.mask)
        return out


class HorizontalStack(nn.Conv2d):
    """Horizontal stack for PixelCNN."""

    def __init__(self, in_features, out_features, ksize, bias=True):
        super().__init__(in_features, out_features, (1, ksize), 1, (0, ksize // 2), bias=bias)
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.mask[:, :, :, :(ksize // 2)] = 1.

    def forward(self, x):
        out = self._conv_forward(x, self.weight * self.mask)
        return out


class PixelCNNGated(nn.Module):
    """Gated PixelCNN as in van Oords papers."""

    def __init__(self, in_channels, n_filters, kernel_size, n_blocks, n_cats):
        super().__init__()
        self.n_cats = n_cats
        self.vertical1 = VerticalStack(in_channels, n_filters, kernel_size)
        self.horizontal1 = HorizontalStack(in_channels, n_filters, kernel_size)
        self.gated = nn.ModuleList()
        for _ in range(n_blocks):
            self.gated.append(GatedLayer(n_filters, kernel_size))
        self.vertical2 = VerticalStack(n_filters, n_filters, kernel_size)
        self.horizontal2 = HorizontalStack(n_filters, n_filters, kernel_size)
        self.vertical3 = VerticalStack(n_filters, n_cats, kernel_size)
        self.horizontal3 = HorizontalStack(n_filters, n_cats, kernel_size)

    def forward(self, x):
        vert = self.vertical1(x)
        horiz = self.horizontal1(x)
        for layer in self.gated:
            vert, horiz = layer(vert, horiz)
        out = self.vertical2(vert) + self.horizontal2(horiz)
        out = F.relu(out)
        out = self.vertical3(out) + self.horizontal3(out)
        return out

    def sample_data(self, n_samples, image_shape, device):
        self.eval()
        with torch.no_grad():
            h, w = image_shape
            samples = torch.multinomial(torch.ones(self.n_cats)/self.n_cats,
                                        n_samples*h*w, replacement=True)
            samples = samples.view(n_samples, 1, h, w)
            samples = rescale(samples, 0., self.n_cats - 1.).to(device)
            print(f"Sampling new examples ...", flush=True)
            for hi in range(h):
                for wi in range(w):
                    logits = self(samples)[:, :, hi, wi].squeeze()
                    probs = logits.softmax(dim=1)
                    samples[:, 0, hi, wi] = torch.multinomial(probs, 1).squeeze()
                    samples[:, 0, hi, wi] = rescale(samples[:, 0, hi, wi], 0., self.n_cats - 1.)
        return samples


class VqLearner():
    """Class for model training."""

    def __init__(self, model, optimizer, trainloader, testloader, loss_func, device, callback_list=[], clip_grads=False):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.loss_func = loss_func
        self.device = device
        self.clip_grads = clip_grads
        self.callback_list = callback_list
        for cb in self.callback_list:
            cb.init_learner(self)

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        losses_train = []
        losses_test = self.eval_epoch()
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            print(f"Training epoch {self.epoch} ...", flush=True)
            losses = self.train_epoch()
            losses_train.extend(losses)
            losses = self.eval_epoch()
            losses_test.extend(losses)
            print(f"Losses: train = {losses_train[-1]}, test = {losses_test[-1]}.", flush=True)
        self.callback('fit_end')
        return losses_train, losses_test

    def train_epoch(self):
        self.callback('train_epoch_begin')
        losses = []
        self.model.train()
        for batch in self.trainloader:
            self.callback('train_batch_begin')
            batch = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            out = self.model(batch[0])
            loss = self.loss_func(batch[0], *out)
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def eval_epoch(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            n_samples = 0.
            for batch in self.testloader:
                batch = [b.to(self.device) for b in batch]
                out = self.model(batch[0])
                batch_size = batch[0].shape[0]
                loss += self.loss_func(batch[0], *out).item() * batch_size
                n_samples += batch_size
            losses.append(loss / n_samples)
        return losses

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)


class LearnerPixelCNN():
    """Class for model training."""

    def __init__(self, model, optimizer, trainloader, testloader, loss_func, device, clip_grads=False):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.loss_func = loss_func
        self.device = device
        self.clip_grads = clip_grads

    def fit(self, epochs):
        losses_train = []
        losses_test = self.eval_epoch()
        for epoch in range(epochs):
            print(f"Training epoch {epoch} ...", flush=True)
            losses = self.train_epoch()
            losses_train.extend(losses)
            losses = self.eval_epoch()
            losses_test.extend(losses)
            print(f"Losses: train = {losses_train[-1]}, test = {losses_test[-1]}.", flush=True)
        return losses_train, losses_test

    def train_epoch(self):
        losses = []
        self.model.train()
        for batch in self.trainloader:
            self.optimizer.zero_grad()
            batch = [bpart.to(self.device) for bpart in batch]
            out = self.model(batch[0])
            loss = self.loss_func(out, batch[-1])
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            losses.append(loss.item())
        return losses

    def eval_epoch(self):
        losses = []
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            n_samples = 0.
            for batch in self.testloader:
                batch = [bpart.to(self.device) for bpart in batch]
                out = self.model(batch[0])
                batch_size = batch[0].shape[0]
                loss += self.loss_func(out, batch[-1]).item() * batch_size
                n_samples += batch_size
            losses.append(loss / n_samples)
        return losses
