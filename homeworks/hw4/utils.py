"""Utility functions for hw4."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from deepul.utils import savefig
from collections import Counter, defaultdict


class Learner():
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
        self.epoch = 0
        for cb in self.callback_list:
            cb.init_learner(self)

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        losses_train = defaultdict(list)
        losses_test = self.eval_epoch()
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            print(f"Training epoch {self.epoch} ...", flush=True)
            losses = self.train_epoch()
            for key, value in losses.items():
                losses_train[key].extend(value)
            # losses_train.extend(losses)
            losses = self.eval_epoch()
            for key, value in losses.items():
                losses_test[key].extend(value)
            # losses_test.extend(losses)
            print(f"Losses: train = {losses_train['nelbo'][-1]}, test = {losses_test['nelbo'][-1]}.", flush=True)
        self.callback('fit_end')
        return losses_train, losses_test

    def train_epoch(self):
        self.callback('train_epoch_begin')
        losses = defaultdict(list)
        self.model.train()
        for batch in self.trainloader:
            self.callback('train_batch_begin')
            batch = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            out = self.model(batch[0])
            loss = self.loss_func(batch[0], *out)
            loss['nelbo'].backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            for key, value in loss.items():
                losses[key].append(value.item())
        return losses

    def eval_epoch(self):
        losses = defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            loss = {'nelbo':0., 'rec':0., 'kl':0.}
            n_samples = 0.
            for batch in self.testloader:
                batch = [b.to(self.device) for b in batch]
                out = self.model(batch[0])
                batch_size = batch[0].shape[0]
                ll = self.loss_func(batch[0], *out)
                loss = {key: loss.get(key) + ll.get(key).item()*batch_size for key in ll.keys()}
                n_samples += batch_size
            for key, value in loss.items():
                losses[key].append(value / n_samples)
        return losses

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)


def rescale(x, min, max):
    """Rescale x to [-1, 1]."""

    return 2. * (x - min) / (max - min) - 1.
    # return x + 0.


def descale(x, min, max):
    """Descale x from [-1, 1] to [min, max]."""

    return (x + 1.) * (max - min) / 2. + min
    # return x + 0.


def jitter(x, colcats):
    """Jitter by uniform noise and rescale to 0, 1."""

    return (x + torch.rand_like(x)) / colcats


def reload_modelstate(model, optimizer, modelpath):
    """Reload mode state from checkpoint saved in modelpath."""

    checkpoint = torch.load(modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses_train, losses_test = checkpoint['losses_train'], checkpoint['losses_test']
    print(f"Loded model from {modelpath}.")
    return model, optimizer, losses_train, losses_test


def save_lr_plot(lr, epochs, fname):
    plt.figure()
    x = np.linspace(0, epochs, len(lr))

    plt.plot(x, lr, label='learning rate')
    plt.legend()
    plt.title("Learning rate schedule")
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    savefig(fname)


def dequantize(x, colcats, alpha=0.05, forward=True):
    """Logit gequantization from 4.1 of RealNVP."""

    def logit(p):
        out = p.log() - (1 - p).log()
        return out

    minx = logit(torch.tensor([alpha, ]))
    maxx = logit(torch.tensor([1 - alpha, ]))
    if forward:
        x = x.float()
        x = jitter(x, colcats)
        x = alpha + (1. - 2*alpha) * x
        x = logit(x)
        x = rescale(x, minx, maxx)
        return x.permute(0, 3, 1, 2).contiguous()
    else:
        x = descale(x, minx, maxx)
        x = torch.sigmoid(x)
        x = (x-alpha)/(1-2*alpha) * colcats
        return x.permute(0, 2, 3, 1)
