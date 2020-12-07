"""Callbacks inspired by fastai."""
import torch
import wandb
from homeworks.hw4.utils import q1_gan_plot, show_samples
import matplotlib.pyplot as plt
import torch.optim as optim
import homeworks.hw4.bigan as bigan
import numpy as np


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
            if self.title in {"q1_a", "q1_b"}:
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
        if self.title in {"q1_a", "q1_b"}:
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


class Printing(Callback):
    """Monitoring printouts."""

    def epoch_begin(self):
        print(f"Epoch {self.learner.epoch} begin. ", flush=True)

    def epoch_end(self):
        print(f"Epoch {self.learner.epoch} end. ", flush=True)


class BiGan(Callback):
    """Callback for wandb for BiGan."""

    def __init__(self, scheduler, log_freq, nsamples, sampling_freq, recdata, zdim, trainloader, testloader, classifierepochs=15):
        self.scheduler = scheduler
        self.log_freq = log_freq
        self.nsamples = nsamples
        self.sampling_freq = sampling_freq
        self.recdata = recdata
        self.zdim = zdim
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifierepochs = classifierepochs

    def fit_begin(self):
        wandb.watch(self.learner.model, log="all", log_freq=self.log_freq)
        self.learner.random_losses = self.train_classifier()

    def batch_begin(self):
        wandb.log({"learning rate": self.scheduler['discriminator'].get_last_lr()[0]})

    def batch_end(self):
        self.scheduler['discriminator'].step()
        self.scheduler['generator'].step()
        self.scheduler['encoder'].step()
        wandb.log({"D loss": self.learner.losses['D'][-1],
                   "GE loss": self.learner.losses['GE'][-1]})

    def epoch_end(self):
        if (self.learner.epoch % self.sampling_freq == 0):
            # samples
            samples = self.learner.model.sample(self.nsamples).to("cpu")
            fig = show_samples(samples, title=f"MNIST {self.learner.epoch}")
            wandb.log({f"generations": wandb.Image(fig)})
            plt.close(fig)
            # reconstructions
            reconstructions = self.learner.model.reconstruct(self.recdata)
            print(f"shapes {self.recdata.shape} {reconstructions.shape}")
            reconstructions = torch.cat((self.recdata, reconstructions), dim=0).to("cpu")
            fig = show_samples(reconstructions, nrow=20, title=f"MNIST {self.learner.epoch}")
            wandb.log({f"reconstructions": wandb.Image(fig)})
            plt.close(fig)

    def fit_end(self):
        # samples
        samples = self.learner.model.sample(self.nsamples).to("cpu")
        fig = show_samples(samples, title=f"MNIST final")
        wandb.log({f"generations": wandb.Image(fig)})
        plt.close(fig)
        self.learner.final_samples = samples
        # reconstructions
        reconstructions = self.learner.model.reconstruct(self.recdata)
        reconstructions = torch.cat((self.recdata, reconstructions), dim=0).to("cpu")
        fig = show_samples(reconstructions, nrow=20, title=f"MNIST {self.learner.epoch}")
        wandb.log({f"reconstructions": wandb.Image(fig)})
        plt.close(fig)
        self.learner.final_reconstructions = reconstructions
        # classification
        self.learner.pretrained_losses = self.train_classifier()
        self.plot_class_loss(self.learner.pretrained_losses, self.learner.random_losses)

    def train_classifier(self):
        classifier = bigan.LinearClassifier(self.zdim).to(self.learner.device)
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        learner = bigan.LearnerClassifier(self.learner.model.encoder, classifier, optimizer,
                                          self.trainloader, self.testloader,
                                          classifier.loss_func, self.learner.device)
        losses_test = np.array(learner.fit(self.classifierepochs)).T
        return losses_test

    def plot_class_loss(self, pretrained_losses, random_losses, title="Classification losses"):
        fig = plt.figure()
        xs = np.arange(len(pretrained_losses))
        plt.plot(xs, pretrained_losses, label='bigan')
        xs = np.arange(len(random_losses))
        plt.plot(xs, random_losses, label='random init')
        plt.legend()
        plt.title(title)
        wandb.log({f"Classification losses": wandb.Image(fig)})
        plt.close(fig)
