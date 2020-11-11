""" Solutions for hw3."""

import torch
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader
from deepul.hw3_helper import resultsdir
from homeworks.hw3.utils import Learner, reload_modelstate, save_lr_plot, rescale, descale
from pathlib import Path
import torch.distributions as D
import homeworks.hw3.callbacks as cb
from homeworks.hw3.nn import VAE

# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
RELOAD = False
TRAIN = True
LEARN_RATE = 0.001
MAXLEARN_RATE = LEARN_RATE*100
MAX_EPOCHS = 10
BATCH_SIZE = 128


def q1(train_data, test_data, part, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q1_a on {DEVICE}.")

    modelpath = f'{resultsdir}/q1_a_model.pickle'

    train_data = torch.from_numpy(train_data)
    xmin, _ = train_data.min(dim=0)
    xmax, _ = train_data.max(dim=0)
    train_data = rescale(train_data, xmin, xmax)
    test_data = rescale(torch.from_numpy(test_data), xmin, xmax)

    x_dim = train_data.shape[1]
    Z_DIM = 2

    def loss_func(data, xlocation, xlogscale, zlocation, zlogscale):
        """ELBO loss."""
        xdist = D.Normal(xlocation.reshape(-1), xlogscale.reshape(-1).exp())
        n = data.shape[0]
        # xcov = (2*xlogscale).exp()
        # rec_loss = 0.5*x_dim*math.log(2*math.pi)
        # rec_loss += xlogscale.sum(dim=1)
        # rec_loss += 0.5 * ((data-xlocation)**2 * xcov).sum(dim=1)
        # rec_loss = rec_loss.mean()
        rec = xdist.log_prob(data.view(-1)).sum() / n
        kl = 0.5 * (-2*zlogscale - 1 + (2*zlogscale).exp() + zlocation**2).sum(dim=1).mean()
        nelbo = rec + kl
        return {'nelbo':nelbo, 'rec':rec, 'kl':kl}

    model = VAE(x_dim, Z_DIM, [32, 32], [32, 32])
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = [], []

    if TRAIN:
        trainloader = DataLoader(TensorDataset(train_data, torch.zeros(train_data.shape[0])),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data, torch.zeros(test_data.shape[0])),
                                batch_size=BATCH_SIZE)

        callback_list = []

        learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE, callback_list)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        losses_train.extend(l_train)
        losses_test.extend(l_test)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
    else:
        print(f"No training, only generating data from last available model.")

    xlocation, xlogscale = model.sample(100, DEVICE)
    samples = xlocation.to("cpu")
    samples_noisy = torch.randn_like(samples) * xlogscale.exp() + xlocation

    train_elbo = [x[0] for x in losses_train]
    train_rec = [x[1] for x in losses_train]
    train_kl = [x[2] for x in losses_train]
    train_losses = np.array([[train_elbo], [train_rec], [train_kl]])

    test_elbo = [x[0] for x in losses_test]
    test_rec = [x[1] for x in losses_test]
    test_kl = [x[2] for x in losses_test]
    test_losses = np.array([[test_elbo], [test_rec], [test_kl]])

    return train_losses, test_losses, samples.numpy(), samples_noisy.numpy()

