""" Solutions for hw4."""

import homeworks.hw4.nn as nn_
import torch.optim as optim
from homeworks.hw4.learner import Learner
from torch.utils.data import DataLoader, TensorDataset
from deepul.hw4_helper import *
import homeworks.hw4.callbacks as cb

# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
RELOAD = False
TRAIN = True
LEARN_RATE = 0.001
MAXLEARN_RATE = LEARN_RATE*100
MAX_EPOCHS = 10
BATCH_SIZE = 128


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

    train_data = torch.from_numpy(train_data).float()
    trainloader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    XDIM = 1
    ZDIM = 1
    DISCRIMINATOR_STEPS = 1

    model = nn_.Gan(XDIM, ZDIM, nn_.GeneratorMlp, [10, 10, 10], nn_.DiscriminatorMlp, [10, 10, 10])

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE)

    callbacks = [cb.SampleData(5000, 0)]
    fakes = torch.linspace(-1., 1., 1000)
    callbacks.append(cb.Discriminate(fakes.to(DEVICE), 0))
    learner = Learner(model, optimizer, trainloader, DISCRIMINATOR_STEPS, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(learner.samples[0].numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes[0].numpy())
    outputs.append(learner.samples['end'].numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes['end'].numpy())

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

    train_data = torch.from_numpy(train_data).float()
    trainloader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    XDIM = 1
    ZDIM = 1
    DISCRIMINATOR_STEPS = 10

    model = nn_.Gan(XDIM, ZDIM, nn_.GeneratorMlp2, [10, 10, 10], nn_.DiscriminatorMlp, [10, 10, 10])

    optimizer = {}
    optimizer['discriminator'] = optim.Adam(model.discriminator.parameters(), lr=LEARN_RATE)
    optimizer['generator'] = optim.Adam(model.generator.parameters(), lr=LEARN_RATE)

    callbacks = [cb.SampleData(5000, 0)]
    fakes = torch.linspace(-1., 1., 1000)
    callbacks.append(cb.Discriminate(fakes.to(DEVICE), 0))
    learner = Learner(model, optimizer, trainloader, DISCRIMINATOR_STEPS, DEVICE, callbacks)
    losses_train = learner.fit(MAX_EPOCHS)

    outputs = [np.array(losses_train).T]
    outputs.append(learner.samples[0].numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes[0].numpy())
    outputs.append(learner.samples['end'].numpy())
    outputs.append(fakes.numpy())
    outputs.append(learner.classes['end'].numpy())

    return outputs
