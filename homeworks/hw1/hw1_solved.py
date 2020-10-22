""" Solutions for hw1."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from deepul.hw1_helper import resultsdir
import homeworks.hw1.nn as nn_
from homeworks.hw1.utils import Learner, rescale, reload_modelstate
from pathlib import Path


# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
RELOAD = False
TRAIN = True
SHORTTRAINING = False


def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q1_a {dset_id} on {DEVICE}.")

    MAX_EPOCHS = 100
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 10

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    def loss_alex(data, params):
        def logsumexp(x):
            maxx = x.max()
            return maxx + (x-maxx).exp().sum().log()
        logp = thetas - logsumexp(thetas)
        logp_data = logp[data]
        return -logp_data.mean()

    def loss_likelihood(data, params):
        """Negative log likelihood defined by hand."""
        logp = thetas - torch.logsumexp(thetas, dim=0)
        logp_data = logp[data]
        return -logp_data.mean()

    def loss_crossentropy(data, params):
        """Negative log likelihood for softmax defined categorical distribution."""
        params = params.expand(data.shape[0], -1)
        loss = F.cross_entropy(params, data)
        return loss

#     loss_func = loss_alex
#     loss_func = loss_likelihood
    loss_func = loss_crossentropy

    thetas = nn.Parameter(torch.zeros(d))

    losses_train = []
    losses_test = [loss_func(test_data, thetas.detach()).item()]

    optimizer = optim.SGD([thetas], lr=LEARN_RATE)
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = loss_func(train_data, thetas)
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())
        loss = loss_func(test_data, thetas.detach()).item()
        losses_test.append(loss)

    probs = F.softmax(thetas.detach(), dim=0)

    return np.array(losses_train), np.array(losses_test), probs.numpy()


def q1_b(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q1_b {dset_id} on {DEVICE}.")

    MAX_EPOCHS = 1000
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 1
    MIX_COMPONENTS = 4
    
    pis_nonscaled = nn.Parameter(torch.ones(MIX_COMPONENTS)/MIX_COMPONENTS)
    mus = nn.Parameter(torch.linspace(0.+0.5, d-1.5, MIX_COMPONENTS))
    log_scales = nn.Parameter(torch.zeros(MIX_COMPONENTS))

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)

    def loss_func(data, pis_nonscaled, mus, log_scales, aggregate=True):
        """Negative log likelihood for discretized mixture of logistics."""
        pis = F.softmax(pis_nonscaled, dim=0)
        scales = log_scales.exp()
        data = data[:, None]
        pis, mus, scales = pis[None, :], mus[None, :], scales[None, :]
        cdf_plus = torch.sigmoid((data + 0.5 - mus)/scales)
        cdf_minus = torch.sigmoid((data - 0.5 - mus)/scales)
        mask_plus = (data < (d-1))
        cdf_plus = (cdf_plus * mask_plus + ~mask_plus)
        mask_minus = (data > 0)
        cdf_minus = (cdf_minus * mask_minus)
        comp_prob = (cdf_plus - cdf_minus).clamp(1e-7)  # clamping for numerical stability (log afterwards)
        component = pis.log() + comp_prob.log()
        logpdf = torch.logsumexp(component, dim=1)
        if aggregate:
            logpdf = logpdf.mean()
        return -logpdf

    losses_train = []
    losses_test = [loss_func(test_data, pis_nonscaled.detach(),
                             mus.detach(), log_scales.detach()).item()]

    optimizer = optim.SGD([pis_nonscaled, mus, log_scales], lr=LEARN_RATE)
    for epoch in range(MAX_EPOCHS):
        optimizer.zero_grad()
        loss = loss_func(train_data, pis_nonscaled, mus, log_scales)
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())
        loss = loss_func(test_data, pis_nonscaled.detach(),
                         mus.detach(), log_scales.detach()).item()
        losses_test.append(loss)

    data_fake = torch.linspace(0, d-1, d)
    neglogprobs = loss_func(data_fake, pis_nonscaled.detach(),
                            mus.detach(), log_scales.detach(), aggregate=False)
    probs = (-neglogprobs).exp()

    return np.array(losses_train), np.array(losses_test), probs.numpy()


def q2_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """

    """ YOUR CODE HERE """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q2_a {dset_id} on {DEVICE}.")

    MAX_EPOCHS = 20
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 0.001
    BATCH_SIZE = 128
    N_DIMS = 2

    train_targets = torch.from_numpy(train_data)
    test_targets = torch.from_numpy(test_data)
    train_size, test_size = train_targets.shape[0], test_targets.shape[0]
    train_data = F.one_hot(train_targets, d).float().view(train_size, -1)
    test_data = F.one_hot(test_targets, d).float().view(test_size, -1)

    def loss_func(logits, targets):
        """Negative log likelihood for AR model."""
        logits = logits.view(logits.shape[0], N_DIMS, d).permute(0, 2, 1)
        return F.cross_entropy(logits, targets)

    model = nn_.Made(N_DIMS*d, [64, 64], N_DIMS).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data, train_targets),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data, test_targets),
                            batch_size=test_data.shape[0])
    learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
    losses_train, losses_test = learner.fit(MAX_EPOCHS)

    model.eval()
    with torch.no_grad():
        ints = torch.arange(d)
        data_fake = torch.cartesian_prod(ints, ints)
        data_in = F.one_hot(data_fake, d).float().view(d**N_DIMS, -1).to(DEVICE)
        logits = model(data_in)
        logits = logits.view(logits.shape[0], N_DIMS, d)
        probs = F.softmax(logits, dim=-1)
        probs_data = torch.masked_select(probs.to(torch.device("cpu")), F.one_hot(data_fake, d).bool())
        probs_data = probs_data.view(d**N_DIMS, N_DIMS)
        probs_joint = torch.prod(probs_data, dim=1)
        probs = probs_joint.view(d, d)

    return np.array(losses_train), np.array(losses_test), probs.numpy()


def q2_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q2_b {dset_id} on {DEVICE}.")

    MAX_EPOCHS = 50
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 0.001
    BATCH_SIZE = 128

    h, w = image_shape
    n_dims = h * w

    train_data = torch.from_numpy(train_data).float()
    test_data = torch.from_numpy(test_data).float()
    train_size, test_size = train_data.shape[0], test_data.shape[0]
    train_data = train_data.view(train_size, -1)
    test_data = test_data.view(test_size, -1)

    def loss_func(logits, targets):
        """Binary cross etnropy."""
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return loss.sum(dim=1).mean(dim=0)

    model = nn_.Made(n_dims, [1024, 1024, 1024], n_dims).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data, train_data),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data, test_data),
                            batch_size=test_data.shape[0])
    learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
    losses_train, losses_test = learner.fit(MAX_EPOCHS)
    losses_train = [x / n_dims for x in losses_train]
    losses_test = [x / n_dims for x in losses_test]

    def sample_data(n_samples):
        freqs = (train_data == 1.).float().sum(dim=0) / train_size
        samples = torch.bernoulli(freqs.expand(n_samples, -1)).to(DEVICE)
        for i in range(n_dims):
            logits = model(samples)
            samples[:, i] = torch.bernoulli(torch.sigmoid(logits))[:, i]
        return samples.view(n_samples, h, w, 1)

    model.eval()
    with torch.no_grad():
        samples = sample_data(100).to(torch.device("cpu"))

    return np.array(losses_train), np.array(losses_test), samples.numpy()


def q3_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q3_a {dset_id} on {DEVICE}.")

    MAX_EPOCHS = 10
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 1e-3
    BATCH_SIZE = 128

    train_targets = torch.from_numpy(train_data).float()
    test_targets = torch.from_numpy(test_data).float()
    train_targets = train_targets.permute(0, 3, 1, 2)
    test_targets = test_targets.permute(0, 3, 1, 2)
#     train_min = torch.min(train_targets)
#     train_max = torch.max(train_targets)
#     train_data = 2 * (train_targets - train_min) / (train_max - train_min) - 1
#     test_data = 2 * (test_targets - train_min) / (train_max - train_min) - 1
    train_data = train_targets
    test_data = test_targets

    c, h, w = train_data.shape[1:]
    n_dims = c * h * w

    def loss_func(logits, targets):
        """Binary cross etnropy."""
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return loss.sum(dim=(1, 2, 3)).mean(dim=0)

    model = nn_.PixelCNN(in_channels=1, n_filters=64, kernel_size=7, n_layers=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data, train_targets),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data, test_targets),
                            batch_size=BATCH_SIZE)
    learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
    losses_train, losses_test = learner.fit(MAX_EPOCHS)
    losses_train = [x / n_dims for x in losses_train]
    losses_test = [x / n_dims for x in losses_test]

    def sample_data(n_samples):
        train_size = train_targets.shape[0]
        freqs = (train_targets.sum(dim=0) / train_size).expand(n_samples, -1, -1, -1)
        samples = torch.bernoulli(freqs)
        samples = samples.to(DEVICE)
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    logits = model(samples)
                    samples[:, ci, hi, wi] = torch.bernoulli(torch.sigmoid(logits))[:, ci, hi, wi]
        return samples.permute(0, 2, 3, 1)

    model.eval()
    with torch.no_grad():
        samples = sample_data(100).to("cpu")

    return np.array(losses_train), np.array(losses_test), samples.numpy()


def q3_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q3_b {dset_id} on {DEVICE}.")

    modelpath = f'{resultsdir}/q3_b_dset{dset_id}_model.pickle'

    MAX_EPOCHS = 20
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 1e-3
    BATCH_SIZE = 128
    COLCATS = 4

    train_targets = torch.from_numpy(train_data).long()
    test_targets = torch.from_numpy(test_data).long()
    train_targets = train_targets.permute(0, 3, 1, 2)
    test_targets = test_targets.permute(0, 3, 1, 2)
    train_data = rescale(train_targets, 0., COLCATS - 1.)
    test_data = rescale(test_targets, 0., COLCATS - 1.)
    # train_data = train_targets.float()
    # test_data = test_targets.float()

    c, h, w = train_data.shape[1:]
    n_dims = c * h * w

    def loss_func(logits, targets):
        """Cross etnropy."""
        logits = logits.view(-1, COLCATS, c, h, w)
        loss = F.cross_entropy(logits, targets, reduction='none')
        return loss.sum(dim=(1, 2, 3)).mean(dim=0)

    model = nn_.PixelCNNResidual(in_channels=c, n_filters=120, kernel_size=7, n_resblocks=8,
                                 colcats=COLCATS, indepchannels=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = [], []

    if TRAIN:
        trainloader = DataLoader(TensorDataset(train_data, train_targets),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data, test_targets),
                                batch_size=BATCH_SIZE)
        learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
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

    losses_train = [x / n_dims for x in losses_train]
    losses_test = [x / n_dims for x in losses_test]

    samples = model.sample_data(100, image_shape, DEVICE).to("cpu")

    return np.array(losses_train), np.array(losses_test), samples.numpy()


def q3_c(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of samples with values in {0, 1, 2, 3}
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q3_c {dset_id} on {DEVICE}.")

    modelpath = f'{resultsdir}/q3_c_dset{dset_id}_model.pickle'

    MAX_EPOCHS = 20
    if SHORTTRAINING:
        MAX_EPOCHS = 1
    LEARN_RATE = 1e-3
    BATCH_SIZE = 128
    COLCATS = 4

    train_targets = torch.from_numpy(train_data).long()
    test_targets = torch.from_numpy(test_data).long()
    train_targets = train_targets.permute(0, 3, 1, 2)
    test_targets = test_targets.permute(0, 3, 1, 2)
    # train_data = train_targets.float()
    # test_data = test_targets.float()
    train_data = rescale(train_targets, 0., COLCATS - 1.)
    test_data = rescale(test_targets, 0., COLCATS - 1.)

    c, h, w = train_data.shape[1:]
    n_dims = c * h * w

    def loss_func(logits, targets):
        """Cross etnropy."""
        logits = logits.view(-1, COLCATS, c, h, w)
        loss = F.cross_entropy(logits, targets, reduction='none')
        return loss.sum(dim=(1, 2, 3)).mean(dim=0)

    model = nn_.PixelCNNResidual(in_channels=c, n_filters=120, kernel_size=7, n_resblocks=8,
                                 colcats=COLCATS, indepchannels=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = [], []

    if TRAIN:
        trainloader = DataLoader(TensorDataset(train_data, train_targets),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data, test_targets),
                                batch_size=BATCH_SIZE)
        learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        losses_train.extend(l_train)
        losses_test.extend(l_test)

        Path(modelpath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
    else:
        print(f"No training, only generating data from last available model.")

    losses_train = [x / n_dims for x in losses_train]
    losses_test = [x / n_dims for x in losses_test]

    samples = model.sample_data(100, image_shape, DEVICE).to("cpu")

    return np.array(losses_train), np.array(losses_test), samples.numpy()
