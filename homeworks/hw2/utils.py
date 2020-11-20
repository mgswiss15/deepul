"""Utility functions for homeworks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from deepul.utils import savefig


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
            loss = self.loss_func(*out)
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
                loss += self.loss_func(*out).item() * batch_size
                n_samples += batch_size
            losses.append(loss / n_samples)
        return losses

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)


def rescale(x, min, max):
    """Rescale x to [-1, 1]."""

    n, h, w, c = x.shape
    n_dims = h*w*c

    out = 2. * (x - min) / (max - min) - 1.
    logjabobs = (2. / (max - min) * n_dims).log()
    return out, logjabobs


def descale(x, min, max):
    """Descale x from [-1, 1] to [min, max]."""

    return (x + 1.) * (max - min) / 2. + min


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


def prep_data(data, colcats, dtype=torch.float):
    targets = torch.from_numpy(data).to(dtype)
    targets = targets.permute(0, 3, 1, 2)
    data = jitter(targets, colcats)
    return targets, data


def bisection(func, n):
    a = (torch.ones((n, 1))*(-10.)).to("cuda")
    b = -a
    while True:
        m = (a + b) / 2.
        mask = (func(m) * func(a)) < 0
        b = mask * m + ~mask * b
        a = ~mask * m + mask * a
        if ((b-a) < 1e-5).all():
            m = (a + b) / 2.
            break
    return m


def save_lr_plot(lr, epochs, fname):
    plt.figure()
    x = np.linspace(0, epochs, len(lr))

    plt.plot(x, lr, label='learning rate')
    plt.legend()
    plt.title("Learning rate schedule")
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    savefig(fname)
