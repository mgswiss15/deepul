"""VQVAE modules."""

import torch
import torch.nn as nn


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


# class Vq(torch.autograd.Function):
#     """Vector quantization."""

#     @staticmethod
#     def forward(ctx, x, codes):
#         n, c, h, w = x.shape
#         ncodes, codedim = codes.shape
#         x = x.permute(0, 2, 3, 1).reshape(-1, codedim).expand(ncodes, -1, -1)
#         codes = codes[:, None, :].expand_as(x)
#         distmat = torch.norm(x-codes, 2, dim=2)
#         minvals, minidx = distmat.min(dim=0)
#         # z = minidx.view(n, 1, h, w)
#         return minidx

#     @staticmethod
#     def backward(ctx, out_grad):
#         return out_grad, out_grad


class VqVae(nn.Module):
    """VQVAE model."""

    def __init__(self, xchannels, codedim, ncodes):
        super().__init__()
        self.codedim = codedim
        self.ncodes = ncodes
        self.codes = nn.Parameter(torch.rand(ncodes, codedim)*(2/ncodes)-(1/ncodes))
        self.encoder = EncoderVqVae(codedim, xchannels)
        self.decoder = DecoderVqVae(codedim, xchannels)

    def forward(self, x):
        zenc = self.encoder(x)
        n, c, h, w = zenc.shape
        idx = self.get_codes(zenc.detach())
        zdec = torch.index_select(self.codes, dim=0, index=idx)
        zdec = zdec.view(n, h, w, c).permute(0, 3, 1, 2)
        xhat = self.decoder((zdec - zenc).detach() + zenc)
        # print(f"zenc {zenc.requires_grad}")
        # print(f"idx {idx.requires_grad}")
        # print(f"zdec {zdec.requires_grad}")
        # print(f"xhat {xhat.requires_grad}")
        # zenc.register_hook(lambda grad: print(f"zenc grad {grad.shape} {grad[0, 0, ...]}."))
        # zdec.register_hook(lambda grad: print(f"zdec grad {grad.shape} {grad[0, 0, ...]}."))
        # self.codes.register_hook(lambda grad: print(f"codes grad {grad.shape} {grad[:10, :10]}."))
        return xhat, zenc, zdec

    def get_codes(self, ze):
        n, c, h, w = ze.shape
        ze = ze.permute(0, 2, 3, 1).reshape(-1, self.codedim).expand(self.ncodes, -1, -1)
        codes = self.codes[:, None, :].expand_as(ze)
        distmat = torch.norm(ze-codes, 2, dim=2)
        minvals, minidx = distmat.min(dim=0)
        # z = minidx.view(n, 1, h, w)
        return minidx.long()

    def loss_func(self, x, xhat, ze, zd, beta=0.25):
        reconstruct = 0.5*((x - xhat)**2).sum(dim=(1, 2, 3)).mean()
        vq = ((ze.detach() - zd)**2).sum(dim=(1, 2, 3)).mean()
        commit = ((ze - zd.detach())**2).sum(dim=(1, 2, 3)).mean()
        return reconstruct + vq + beta*commit

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            zsample = torch.randn((n_samples, self.codedim, 32, 32), device=device)
            mu_x = self.decoder(zsample)
            return mu_x


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
