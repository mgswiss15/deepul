"""Learner for GANs in hw4."""

import torch


class Learner():
    """Class for model training."""

    def __init__(self, model, optimizer, trainloader, discriminator_steps, device, callback_list=[]):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.discriminator_steps = discriminator_steps
        self.device = device
        self.callback_list = callback_list
        self.batchsize = self.trainloader.batch_size
        self.epoch = 0
        for cb in self.callback_list:
            cb.init_learner(self)
            print(cb)

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        losses_train = []
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            print(f"Training epoch {self.epoch} ...", flush=True)
            losses = self.train_epoch()
            losses_train.extend(losses)
            print(f"Losses: train = {losses_train[-1]}.", flush=True)
            self.callback('epoch_end')
        self.callback('fit_end')
        return losses_train

    def train_epoch(self):
        self.callback('train_epoch_begin')
        losses = []
        self.model.train()
        losses.extend(self.discriminator_update(self.discriminator_steps))
        self.generator_update()
        return losses

    def discriminator_update(self, steps):
        losses = []
        for bidx, batch in enumerate(self.trainloader):
            self.callback('discriminator_update_begin')
            self.optimizer['discriminator'].zero_grad()
            batch = batch[0].to(self.device)
            batchhat = self.model.generator.sample(self.batchsize).to(self.device)
            batch = torch.cat((batch, batchhat), dim=0)
            targets = torch.zeros_like(batch, device=self.device)
            targets[:self.batchsize, :] = 1.
            logits = self.model.discriminator(batch)
            loss = self.model.discriminator.loss_func(targets, logits)
            loss.backward()
            losses.append(loss.item())
            self.optimizer['discriminator'].step()
            if bidx == steps:
                break
        return losses

    def generator_update(self):
        self.callback('generator_update_begin')
        self.optimizer['generator'].zero_grad()
        batch = self.model.generator.sample(self.batchsize).to(self.device)
        targets = torch.zeros_like(batch, device=self.device)
        logits = self.model.discriminator(batch)
        loss = self.model.generator.loss_func(targets, logits)
        loss.backward()
        self.optimizer['generator'].step()

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)
