"""Utility functions for hw2."""

import torch


class Learner():
    """Class for model training."""

    def __init__(self, model, optimizer, scheduler, trainloader, testloader, loss_func, device, clip_grads=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
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
            # if not isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            #     self.scheduler.step()
        return losses_train, losses_test

    def train_epoch(self):
        losses = []
        self.model.train()
        for batch in self.trainloader:
            batch = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            out = self.model(batch[0])
            loss = self.loss_func(*out)
            loss.backward()
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            # if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            #     self.scheduler.step()
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


def rescale(x, min, max):
    """Rescale x to [-1, 1]."""

    return 2. * (x - min) / (max - min) - 1.


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
