""" Solutions for hw2."""

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from deepul.hw2_helper import resultsdir
import homeworks.hw2.nn as nn_
from homeworks.hw2.utils import Learner, rescale, reload_modelstate, prep_data
from pathlib import Path
from deepul.utils import show_samples
import torch.distributions as D


# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
RELOAD = False
TRAIN = True
LEARN_RATE = 0.001
MAX_EPOCHS = 10
BATCH_SIZE = 128


def q1_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets, or
             for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity). 
      Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents 
      mapping the train set data points through our flow to the latent space. 
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q1_a {dset_id} on {DEVICE}.")

    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()

    n_dims = train_data.shape[-1]

    N_COMPONENTS = 10

    def loss_func(z, logjacobs, aggregate=True):
        """Flow loss func: NLL for uniform z."""
        logpdf = logjacobs.sum(dim=1)
        logpdf = logpdf.mean() if aggregate else logpdf
        return -logpdf

    model = nn_.AutoregressiveFlow(n_dims, N_COMPONENTS * 3, [30, 30, 30, 30, 30], 'gauss')
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data),
                            batch_size=BATCH_SIZE)
    learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
    losses_train, losses_test = learner.fit(MAX_EPOCHS)

    with torch.no_grad():
        # heatmap
        dx, dy = 0.025, 0.025
        if dset_id == 1:  # face
            x_lim = (-4, 4)
            y_lim = (-4, 4)
        elif dset_id == 2:  # two moons
            x_lim = (-1.5, 2.5)
            y_lim = (-1, 1.5)
        y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                        slice(x_lim[0], x_lim[1] + dx, dx)]
        mesh_xs = torch.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
        z, logjacobs = model(mesh_xs)
        probs = (-loss_func(z, logjacobs, aggregate=False)).exp()

        # latents
        z, _ = model(train_data)

    return np.array(losses_train), np.array(losses_test), probs.numpy(), z.numpy()


def q1_b(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets, or
             for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity). 
      Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in R^2. This represents 
      mapping the train set data points through our flow to the latent space. 
    """
    
    # DEVICE is defined and assigned to this module in main
    print(f"Training q1_b {dset_id} on {DEVICE}.")

    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()

    n_dims = train_data.shape[-1]

    # z distribution
    MN = D.MultivariateNormal(torch.zeros(n_dims), torch.eye(n_dims))

    def loss_func(z, logjacobs, aggregate=True):
        """Flow loss func: NLL for standard normal z."""
        logpdf = MN.log_prob(z) + logjacobs.sum(dim=1)
        logpdf = logpdf.mean() if aggregate else logpdf
        return -logpdf

    model = nn_.RealNVP(n_dims, [30, 30, 30], 5)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data),
                            batch_size=BATCH_SIZE)
    learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
    losses_train, losses_test = learner.fit(MAX_EPOCHS)

    with torch.no_grad():
        # heatmap
        dx, dy = 0.025, 0.025
        if dset_id == 1:  # face
            x_lim = (-4, 4)
            y_lim = (-4, 4)
        elif dset_id == 2:  # two moons
            x_lim = (-1.5, 2.5)
            y_lim = (-1, 1.5)
        y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                        slice(x_lim[0], x_lim[1] + dx, dx)]
        mesh_xs = torch.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
        z, logjacobs = model(mesh_xs)
        probs = (-loss_func(z, logjacobs, aggregate=False)).exp()

        # latents
        z, _ = model(train_data)

    return np.array(losses_train), np.array(losses_test), probs.numpy(), z.numpy()
