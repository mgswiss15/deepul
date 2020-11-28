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
        self.discriminator_lossfunc = self.model.discriminator.loss_func
        self.generator_lossfunc = self.model.generator.loss_func
        self.losses_discriminator = []
        self.losses_generator = []

    def fit(self, epochs):
        self.epochs = epochs
        self.callback('fit_begin')
        self.model.train()
        for self.epoch in range(epochs):
            self.callback('epoch_begin')
            self.losses_discriminator.extend(self.discriminator_update(self.discriminator_steps))
            self.losses_generator.append(self.generator_update())
            self.callback('epoch_end')
        self.callback('fit_end')
        return self.losses_discriminator

    def discriminator_update(self, steps):
        losses = []
        self.callback('discriminator_update_begin')
        for bidx, batch in enumerate(self.trainloader):
            self.optimizer['discriminator'].zero_grad()
            batch = batch[0].to(self.device)
            batchhat = self.model.generator.sample(self.batchsize)
            batch = torch.cat((batch, batchhat), dim=0)
            targets = torch.zeros(batch.shape[0], 1).to(self.device)
            targets[:self.batchsize, :] = 1.
            out = self.model.discriminator(batch)
            loss = self.discriminator_lossfunc(targets, *out)
            losses.append(loss.item())
            loss.backward()
            self.optimizer['discriminator'].step()
            self.callback('discriminator_step_end')
            if bidx == (steps - 1):
                break
        return losses

    def generator_update(self):
        self.callback('generator_update_begin')
        self.optimizer['generator'].zero_grad()
        batch = self.model.generator.sample(self.batchsize)
        targets = torch.zeros(batch.shape[0], 1, device=self.device)
        out = self.model.discriminator(batch)
        loss = self.generator_lossfunc(targets, *out)
        loss.backward()
        self.optimizer['generator'].step()
        self.callback('generator_step_end')
        return loss.item()

    def callback(self, cb_name, *args, **kwargs):
        for cb in self.callback_list:
            cb_method = getattr(cb, cb_name, None)
            if cb_method:
                cb_method(*args, **kwargs)
