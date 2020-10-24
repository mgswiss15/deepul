"""Utility functions for hw1."""

import torch


class Learner():
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
            out = self.model(batch[:-1])
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
                out = self.model(batch[:-1])
                batch_size = batch[0].shape[0]
                loss += self.loss_func(out, batch[-1]).item() * batch_size
                n_samples += batch_size
            losses.append(loss / n_samples)
        return losses


def rescale(x, min, max):
    """Rescale x to [-1, 1]."""

    return 2. * (x - min) / (max - min) - 1.


def descale(x, min, max):
    """Descale x from [-1, 1] to [min, max]."""

    return (x + 1.) * (max - min) / 2. + min


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
    data = rescale(targets, 0., colcats - 1.)
    return targets, data
