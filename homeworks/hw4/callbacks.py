"""Callbacks inspired by fastai."""
import torch
import wandb
from homeworks.hw4.utils import q1_gan_plot, show_samples
import matplotlib.pyplot as plt

class Callback():
    """Base class for callbacks."""

    def init_learner(self, learner):
        self.learner = learner

    @property
    def class_name(self):
        return self.__class__.__name__.lower()


class SampleData(Callback):
    """Callback for sampling data."""

    def __init__(self, nsamples, nsamplesfinal=None, samplingepoch=None, sampling_freq=None):
        self.samplingepoch = samplingepoch
        self.nsamples = nsamples
        if nsamplesfinal is None:
            self.nsamplesfinal = nsamples
        else:
            self.nsamplesfinal = nsamplesfinal
        self.samples = {}
        self.sampling_freq = sampling_freq

    def epoch_end(self):
        if (self.learner.epoch == self.samplingepoch) or (self.learner.epoch % self.sampling_freq == 0):
            self.samples = self.generator_sample(self.nsamples)
            self.learner.samples = self.samples.to("cpu")
            if self.learner.epoch == self.samplingepoch:
                self.learner.samples_epoch = self.learner.samples

    def fit_end(self):
        self.samples = self.generator_sample(self.nsamplesfinal)
        self.learner.samples = self.samples.to("cpu")

    def generator_sample(self, nsamples):
        self.learner.model.eval()
        with torch.no_grad():
            steps = nsamples // 100
            smpl = []
            for i in range(steps):
                smpl.append(self.learner.model.sample(100))
            samples = torch.cat(smpl, dim=0)
        self.learner.model.train()
        return samples


class Discriminate(Callback):
    """Callback for discriminating fakes."""

    def __init__(self, fakes, samplingepoch=None, sampling_freq=None):
        self.fakes = fakes[:, None]
        self.samplingepoch = samplingepoch
        self.classes = {}
        self.sampling_freq = sampling_freq

    def epoch_end(self):
        if (self.learner.epoch == self.samplingepoch) or (self.learner.epoch % self.sampling_freq == 0):
            self.classes = self.discriminate()
            self.learner.classes = self.classes.to("cpu")
            if self.learner.epoch == self.samplingepoch:
                self.learner.classes_epoch = self.learner.classes

    def fit_end(self):
        self.classes = self.discriminate()
        self.learner.classes = self.classes.to("cpu")

    def discriminate(self):
        self.learner.model.eval()
        with torch.no_grad():
            classes = self.learner.model.discriminate(self.fakes)
        self.learner.model.train()
        return classes.squeeze()


class Wandb(Callback):
    """Callback for wandb."""

    def __init__(self, data, log_freq, fakes, plot_freq, title):
        self.data = data
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.fakes = fakes
        self.title = title

    def fit_begin(self):
        wandb.watch(self.learner.model, log="all", log_freq=self.log_freq)

    def epoch_end(self):
        wandb.log({"discriminator loss": self.learner.losses_discriminator[-1],
                   "generator loss": self.learner.losses_generator[-1]})
        if self.learner.epoch % self.plot_freq == 0:
            if self.title == "q1_b":
                fig = q1_gan_plot(self.data,
                                  self.learner.samples.numpy(),
                                  self.fakes,
                                  self.learner.classes.numpy(),
                                  f"{self.title} {self.learner.epoch}")
            elif self.title == "q2":
                fig = show_samples(self.learner.samples,
                                   title=f"{self.title} {self.learner.epoch}")
            wandb.log({f"generations": wandb.Image(fig)})
            plt.close(fig)

    def fit_end(self):
        if self.title == "q1_b":
            fig = q1_gan_plot(self.data,
                              self.learner.samples.numpy(),
                              self.fakes,
                              self.learner.classes.numpy(),
                              f"{self.title} end")
        elif self.title == "q2":
            fig = show_samples(self.learner.samples[:100, ...],
                               title=f"{self.title} end")
        wandb.log({f"generations": wandb.Image(fig)})
        plt.close(fig)


class Scheduler(Callback):
    """Setup and update scheduler."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def discriminator_step_end(self):
        wandb.log({"critic lr": self.scheduler['discriminator'].get_last_lr()[0]})
        self.scheduler['discriminator'].step()

    def generator_step_end(self):
        wandb.log({"generator lr": self.scheduler['generator'].get_last_lr()[0]})
        self.scheduler['generator'].step()
