""" Solutions for hw2."""

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from deepul.hw2_helper import resultsdir
import homeworks.hw2.nn as nn_
from homeworks.hw2.utils import Learner, reload_modelstate, prep_data, save_lr_plot
from pathlib import Path
import torch.distributions as D
from homeworks.hw2.realnvp import dequantize, RealNVP
import homeworks.hw2.callbacks as cb
from deepul.utils import show_samples

# init DEVICE, RELOAD and TRAIN will be redefined in main from argparse
DEVICE = torch.device("cpu")
RELOAD = False
TRAIN = True
LEARN_RATE = 0.001
MAXLEARN_RATE = LEARN_RATE*100
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


def q2(train_data, test_data):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    H = W = 20
    Note that you should dequantize your train and test data, your dequantized pixels should all lie in [0,1]

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in [0, 1], where [0,0.5] represents a black pixel
      and [0.5,1] represents a white pixel. We will show your samples with and without noise. 
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q2 on {DEVICE}.")

    modelpath = f'{resultsdir}/q2_model.pickle'

    COLCATS = 2

    train_targets, train_data = prep_data(train_data, COLCATS)
    test_targets, test_data = prep_data(test_data, COLCATS)

    n_dims = 20*20*3

    N_COMPONENTS = 10

    def loss_func(z, logjacobs, aggregate=True):
        """Flow loss func: NLL for uniform z."""
        logpdf = logjacobs.sum(dim=(1, 2, 3))
        logpdf = logpdf.mean() if aggregate else logpdf
        return -logpdf

    model = nn_.PixelCNNFlow(1, N_COMPONENTS * 3, 64, 7, 5, 'gauss').to(DEVICE)
    # model = nn_.PixelCNNFlow(1, N_COMPONENTS * 3, 32, 7, 2, 'gauss')
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

    samples = model.sample_data(100, (20, 20), DEVICE).to("cpu")

    return np.array(losses_train), np.array(losses_test), samples.numpy()


def q3_a(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    # DEVICE is defined and assigned to this module in main
    print(f"Training q3_a on {DEVICE}.")

    modelpath = f'{resultsdir}/q3_a_model.pickle'

    COLCATS = 4

    train_data, ljd_train = dequantize(torch.from_numpy(train_data), COLCATS, forward=True)
    test_data, ljd_test = dequantize(torch.from_numpy(test_data), COLCATS, forward=True)
    # train_data = train_data[:100, ...]
    # test_data = test_data[:50, ...]

    img_shape = train_data.shape[-2:]
    n_dims = img_shape[0] * img_shape[1] * 3

    # z distribution
    MN = D.MultivariateNormal(torch.zeros(n_dims, device=DEVICE), torch.eye(n_dims, device=DEVICE))

    def loss_func(z, logjacobs, aggregate=True):
        """Flow loss func: NLL for standard normal z."""
        logpdf = MN.log_prob(z.view(-1, n_dims)) + logjacobs
        logpdf = logpdf.mean() if aggregate else logpdf
        return -logpdf

    # model = RealNVP(3, 8, 3, 2, img_shape).to(DEVICE)
    model = RealNVP(3, 128, 3, 8, img_shape).to(DEVICE)
    # model = RealNVP(3, 64, 3, 4, img_shape).to(DEVICE)
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

        callback_list = [cb.CombinedScheduler('lr', ['cosine_sched', 'cosine_sched'],
                                              0.2, LEARN_RATE, MAXLEARN_RATE, LEARN_RATE),
                         cb.InitActNorm(BATCH_SIZE, DEVICE)]

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
        save_lr_plot(learner.schedule['lr'], MAX_EPOCHS, f'{resultsdir}/q3_a_lrschedule.png')
    else:
        print(f"No training, only generating data from last available model.")

    losses_train = [(x - ljd_train) / n_dims for x in losses_train]
    losses_test = [(x - ljd_test) / n_dims for x in losses_test]

    samples = model.sample_data(100, DEVICE).to("cpu")
    samples = dequantize(samples, COLCATS, forward=False)

    def interpolations(n_rows):
        print("Interpolations ...")
        testidx = torch.randint(0, test_data.shape[0], (n_rows*2,))
        imgs = torch.index_select(test_data, 0, testidx).to(DEVICE)
        imgs_plot = dequantize(imgs.to("cpu"), COLCATS, forward=False).numpy()
        show_samples(imgs_plot * 255.0, f'{resultsdir}/q3_a_data.png', nrow=6, title='Data')
        model.eval()
        with torch.no_grad():
            zs, _ = model(imgs)
            z1 = torch.repeat_interleave(zs[:n_rows, ...], 6, dim=0)
            z2 = torch.repeat_interleave(zs[n_rows:, ...], 6, dim=0)
            weights = torch.linspace(0., 1., 6, device=DEVICE).repeat(5)[:, None, None, None]
            zs = weights * z1 + (1-weights) * z2
            inter = model.reverse(zs)
            print('Inter stats', inter.max(), inter.min())
        return dequantize(inter.to("cpu"), COLCATS, forward=False)

    inter = interpolations(5)

    return np.array(losses_train), np.array(losses_test), samples.numpy(), inter.numpy()


def q3_b(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

     # DEVICE is defined and assigned to this module in main
    print(f"Training q3_b on {DEVICE}.")

    modelpath = f'{resultsdir}/q3_b_model.pickle'

    COLCATS = 4

    train_data = dequantize(torch.from_numpy(train_data), COLCATS, forward=True)
    test_data = dequantize(torch.from_numpy(test_data), COLCATS, forward=True)
    train_data = train_data[:100, ...]
    test_data = test_data[:50, ...]

    img_shape = train_data.shape[-2:]
    n_dims = img_shape[0] * img_shape[1] * 3

    # z distribution
    MN = D.MultivariateNormal(torch.zeros(n_dims, device=DEVICE), torch.eye(n_dims, device=DEVICE))

    def loss_func(z, logjacobs, aggregate=True):
        """Flow loss func: NLL for standard normal z."""
        logpdf = MN.log_prob(z.view(-1, n_dims)) + logjacobs
        logpdf = logpdf.mean() if aggregate else logpdf
        return -logpdf

    model = RealNVP(3, 8, 3, 2, img_shape, badmasks=True).to(DEVICE)
    # model = RealNVP(3, 128, 3, 8, img_shape, badmasks=True).to(DEVICE)
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

        callback_list = [cb.CombinedScheduler('lr', ['cosine_sched', 'cosine_sched'],
                                              0.2, LEARN_RATE, LEARN_RATE*100, LEARN_RATE),
                         cb.InitActNorm(BATCH_SIZE, DEVICE)]

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
        save_lr_plot(learner.schedule['lr'], MAX_EPOCHS, f'{resultsdir}/q3_b_lrschedule.png')
    else:
        print(f"No training, only generating data from last available model.")

    losses_train = [x / n_dims for x in losses_train]
    losses_test = [x / n_dims for x in losses_test]

    samples = model.sample_data(100, DEVICE)
    samples = dequantize(samples, COLCATS, forward=False).to("cpu")

    def interpolations(n_rows):
        print("Interpolations ...")
        testidx = torch.randint(0, test_data.shape[0], (n_rows*2,))
        imgs = torch.index_select(test_data, 0, testidx).to(DEVICE)
        imgs_plot = dequantize(imgs.to("cpu"), COLCATS, forward=False).numpy()
        show_samples(imgs_plot * 255.0, f'{resultsdir}/q3_a_data.png', nrow=6, title='Interpolations')
        model.eval()
        with torch.no_grad():
            zs, _ = model(imgs)
            z1 = torch.repeat_interleave(zs[:n_rows, ...], 6, dim=0)
            z2 = torch.repeat_interleave(zs[n_rows:, ...], 6, dim=0)
            weights = torch.linspace(0., 1., 6, device=DEVICE).repeat(5)[:, None, None, None]
            zs = weights * z1 + (1-weights) * z2
            inter = model.reverse(zs)
        return dequantize(inter.to("cpu"), COLCATS, forward=False)

    inter = interpolations(5)
    print('Iter stats', iter.max(), iter.min())

    return np.array(losses_train), np.array(losses_test), samples.numpy(), inter.numpy()
