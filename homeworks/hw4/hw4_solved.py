""" Solutions for hw4."""

import homeworks.hw4.nn as nn_
import torch.optim as optim
from homeworks.hw4.learner import Learner
from torch.utils.data import DataLoader, TensorDataset
from deepul.hw4_helper import *
import homeworks.hw4.callbacks as cb
import homeworks.hw4.bigan as bigan

# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
LEARN_RATE = 0.001
MAX_EPOCHS = 10
BATCH_SIZE = 128
GENFREQ = 50
DSTEPS = 1
DHIDDEN = [10]
GHIDDEN = [10]
ZDIM = 1


def q1_a(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at the end of training
    """

    traindata = torch.from_numpy(train_data).float()
    trainloader = DataLoader(TensorDataset(traindata), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    xdim = traindata.shape[1]

    print(f"Training q1_a on {DEVICE}.")

    model = nn_.Gan(xdim, ZDIM, nn_.GeneratorMlp, GHIDDEN, nn_.DiscriminatorMlp, DHIDDEN).to(DEVICE)

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE)

    callbacks = [cb.SampleData(5000, 5000, 0, GENFREQ)]
    fakes = torch.linspace(-1., 1., 1000)
    callbacks.append(cb.Discriminate(fakes.to(DEVICE), 0, GENFREQ))
    callbacks.append(cb.Wandb(train_data, 1, fakes.numpy(), GENFREQ, "q1_a"))
    learner = Learner(model, optimizer, trainloader, DSTEPS, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(learner.samples_epoch.to("cpu").numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes_epoch.to("cpu").numpy())
    outputs.append(learner.samples.to("cpu").numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes.to("cpu").numpy())

    return outputs


def q1_b(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (100,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (100,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at the end of training
    """

    traindata = torch.from_numpy(train_data).float()
    trainloader = DataLoader(TensorDataset(traindata), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    xdim = traindata.shape[1]

    print(f"Training q1_b on {DEVICE}.")

    model = nn_.Gan(xdim, ZDIM, nn_.GeneratorMlp2, GHIDDEN, nn_.DiscriminatorMlp, DHIDDEN).to(DEVICE)

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE)

    callbacks = [cb.SampleData(5000, 5000, 0, GENFREQ)]
    fakes = torch.linspace(-1., 1., 1000)
    callbacks.append(cb.Discriminate(fakes.to(DEVICE), 0, GENFREQ))
    callbacks.append(cb.Wandb(train_data, 1, fakes.numpy(), GENFREQ, "q1_b"))
    learner = Learner(model, optimizer, trainloader, DSTEPS, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(learner.samples_epoch.to("cpu").numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes_epoch.to("cpu").numpy())
    outputs.append(learner.samples.to("cpu").numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes.to("cpu").numpy())

    return outputs


def q2(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
    """

    traindata = torch.from_numpy(train_data).float()
    # traindata = traindata[:200, ...]
    trainloader = DataLoader(TensorDataset(traindata), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    xdim = traindata.shape[1]
    NFILTERS = 128
    NSAMPLES = 1000
    LBD = 10

    print(f"Training q2 on {DEVICE}.")

    model = nn_.GanCifar(xdim, ZDIM, NFILTERS, LBD).to(DEVICE)

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE, betas=[0, 0.9], eps=2e-4)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE, betas=[0, 0.9], eps=2e-4)

    scheduler = {}
    total_steps = DSTEPS * MAX_EPOCHS
    scheduler['discriminator'] = optim.lr_scheduler.OneCycleLR(optimizer['discriminator'],
                                                               LEARN_RATE, total_steps,
                                                               pct_start=0., anneal_strategy='linear')
    total_steps = MAX_EPOCHS
    scheduler['generator'] = optim.lr_scheduler.OneCycleLR(optimizer['generator'],
                                                           LEARN_RATE, total_steps,
                                                           pct_start=0., anneal_strategy='linear')

    callbacks = [cb.SampleData(100, NSAMPLES, 0, GENFREQ)]
    callbacks.append(cb.Wandb(None, 10, None, GENFREQ, "q2"))
    callbacks.append(cb.Scheduler(scheduler))
    callbacks.append(cb.Printing())
    learner = Learner(model, optimizer, trainloader, DSTEPS, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(learner.samples.permute(0, 2, 3, 1).to("cpu").numpy())

    return outputs


def q3(train_data, test_data):
    """
    train_data: A PyTorch dataset that contains (n_train, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST
    test_data: A PyTorch dataset that contains (n_test, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST

    Returns
    - a (# of training iterations,) numpy array of BiGAN minimax losses evaluated every minibatch
    - a (100, 28, 28, 1) numpy array of BiGAN samples that lie in [0, 1]
    - a (40, 28, 28, 1) numpy array of 20 real image / reconstruction pairs
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on the BiGAN encoder evaluated every epoch 
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on a random encoder evaluated every epoch 
    """

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)

    xdim = (28, 28)

    print(f"Training q3 on {DEVICE}.")

    model = bigan.BiGan(xdim, ZDIM, DHIDDEN[0]).to(DEVICE)

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE, betas=[0.5, 0.9], eps=2e-4)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE, betas=[0.5, 0.9], eps=2e-4)
    optimizer['encoder'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE, betas=[0.5, 0.9], eps=2e-4)

    scheduler = {}
    total_steps = MAX_EPOCHS * len(trainloader)
    scheduler['discriminator'] = optim.lr_scheduler.OneCycleLR(optimizer['discriminator'],
                                                               LEARN_RATE, total_steps,
                                                               pct_start=0., anneal_strategy='linear')
    scheduler['generator'] = optim.lr_scheduler.OneCycleLR(optimizer['generator'],
                                                           LEARN_RATE, total_steps,
                                                           pct_start=0., anneal_strategy='linear')
    scheduler['encoder'] = optim.lr_scheduler.OneCycleLR(optimizer['discriminator'],
                                                               LEARN_RATE, total_steps,
                                                               pct_start=0., anneal_strategy='linear')

    # reconstructions
    batch = next(iter(testloader))
    recdata = batch[0][:20, ...].to(DEVICE)
    callbacks = [cb.BiGan(scheduler, 10, 100, 10, recdata)]
    learner = bigan.BiGanLearner(model, optimizer, trainloader, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(self.learner.final_samples.permute(0, 3, 1, 2).numpy())
    outputs.append(self.learner.final_reconstructions.permute(0, 3, 1, 2).numpy())

    outputs.append(torch.rand(40, 28, 28, 1).numpy())
    outputs.append(torch.randn(100, 1).numpy())
    outputs.append(torch.randn(100, 1).numpy())

    return outputs


