""" Solutions for hw3."""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from deepul.hw3_helper import resultsdir
from deepul.utils import save_scatter_2d, show_samples, savefig
from homeworks.hw3.utils import Learner, reload_modelstate, save_lr_plot, rescale, descale, dequantize
from pathlib import Path
import torch.distributions as D
import homeworks.hw3.callbacks as cb
import homeworks.hw3.nn as nn_
from collections import defaultdict
import matplotlib.pyplot as plt
from homeworks.hw3.vqvae import VqVae, VqLearner, PixelCNN
from homeworks.hw1.utils import Learner


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

    modelpath = f'{resultsdir}/q1_{part}_dset{dset_id}_model.pickle'

    train_data = torch.from_numpy(train_data)
    xmin, _ = train_data.min(dim=0)
    xmax, _ = train_data.max(dim=0)
    train_data = rescale(train_data, xmin, xmax)
    test_data = rescale(torch.from_numpy(test_data), xmin, xmax)
    save_scatter_2d(train_data, title='Train data',
                    fname=f'{resultsdir}/q1_{part}_dset{dset_id}_traindata.png')

    x_dim = train_data.shape[1]
    Z_DIM = 2

    def loss_func(data, mu_x, logstd_x, mu_z, logstd_z):
        """ELBO loss."""
        dist_x = D.Normal(mu_x.reshape(-1), logstd_x.reshape(-1).exp())
        rec = -(dist_x.log_prob(data.view(-1))).sum() / data.shape[0]
        kl = (-logstd_z - 0.5 + 0.5*((2*logstd_z).exp() + mu_z**2)).sum(dim=1).mean()
        nelbo = rec + kl
        return {'nelbo':nelbo, 'rec':rec, 'kl':kl}

    model = nn_.VAE(x_dim, Z_DIM, nn_.EncoderMLP, nn_.DecoderMLP, [16, 16], [16, 16]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = defaultdict(list), defaultdict(list)

    if TRAIN:
        print(f"Training q1_a on {DEVICE}.")
        trainloader = DataLoader(TensorDataset(train_data, torch.zeros(train_data.shape[0])),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data, torch.zeros(test_data.shape[0])),
                                batch_size=BATCH_SIZE)

        callback_list = []

        learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE, callback_list)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        for key, value in l_train.items():
            losses_train[key].extend(value)
        for key, value in l_test.items():
            losses_test[key].extend(value)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
    else:
        print(f"No training, only generating data from last available model.")

    mu_x, logstd_x = model.sample(1000, DEVICE)
    samples = descale(mu_x.to("cpu"), xmin, xmax)
    samples_noisy = descale(torch.randn_like(samples) * logstd_x.exp().to("cpu") + mu_x.to("cpu"), xmin, xmax)

    model.eval()
    with torch.no_grad():
        recdata = test_data[:10, ...]
        reconstruct, _, _, _ = model(recdata.to(DEVICE))
        reconstruct = reconstruct.to("cpu")
        plt.figure()
        plt.title('Reconstructions')
        c = list(range(10))
        plt.scatter(recdata[:, 0], recdata[:, 1], c=c, marker='o', cmap=plt.cm.Set1)
        plt.scatter(reconstruct[:, 0], reconstruct[:, 1], c=c, marker='^', cmap=plt.cm.Set1)
        savefig(f'{resultsdir}/q1_{part}_dset{dset_id}_reconstruct.png')

    train_losses = np.array([losses_train['nelbo'], losses_train['rec'], losses_train['kl']]).T
    test_losses = np.array([losses_test['nelbo'], losses_test['rec'], losses_test['kl']]).T

    return train_losses, test_losses, samples_noisy.numpy(), samples.numpy()


def q2_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    modelpath = f'{resultsdir}/q2_a_dset{dset_id}_model.pickle'

    Z_DIM = 16
    COLCATS = 256

    train_data = dequantize(torch.from_numpy(train_data), COLCATS)
    test_data = dequantize(torch.from_numpy(test_data), COLCATS)
    # train_data = train_data[:200, ...]
    # test_data = test_data[:200, ...]

    x_dim = train_data.shape[1]

    def loss_func(data, mu_x, logstd_x, mu_z, logstd_z):
        """ELBO loss."""
        rec = 0.5*((mu_x - data)**2).sum(dim=(1, 2, 3)).mean()
        kl = (-logstd_z - 0.5 + 0.5*((2*logstd_z).exp() + mu_z**2)).sum(dim=1).mean()
        nelbo = rec + kl
        return {'nelbo':nelbo, 'rec':rec, 'kl':kl}

    model = nn_.VAE(x_dim, Z_DIM, nn_.EncoderConv, nn_.DecoderConv).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = defaultdict(list), defaultdict(list)

    if TRAIN:
        print(f"Training q1_a on {DEVICE}.")
        trainloader = DataLoader(TensorDataset(train_data),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data),
                                batch_size=BATCH_SIZE)

        callback_list = [cb.CombinedScheduler('lr', ['cosine_sched', 'cosine_sched'],
                                              0.2, LEARN_RATE, MAXLEARN_RATE, LEARN_RATE)]

        learner = Learner(model, optimizer, trainloader, testloader, loss_func, DEVICE, callback_list)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        for key, value in l_train.items():
            losses_train[key].extend(value)
        for key, value in l_test.items():
            losses_test[key].extend(value)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
        save_lr_plot(learner.schedule['lr'], MAX_EPOCHS, f'{resultsdir}/q2_a_dset{dset_id}_lrschedule.png')
    else:
        print(f"No training, only generating data from last available model.")

    mu_x, logstd_x = model.sample(100, DEVICE)
    samples = dequantize(mu_x.to("cpu"), COLCATS, forward=False)

    model.eval()
    with torch.no_grad():
        recdata = test_data[:50, ...]
        reconstruct, _, _, _ = model(recdata.to(DEVICE))
        reconstruct = torch.cat((recdata, reconstruct.to("cpu")), dim=1).view(100, 3, 32, 32)
        reconstruct = dequantize(reconstruct, COLCATS, forward=False)

    def interpolations():
        print("Interpolations ...")
        testidx = torch.randint(0, test_data.shape[0], (20,))
        imgs = torch.index_select(test_data, 0, testidx).to(DEVICE)
        imgs_plot = dequantize(imgs.to("cpu"), COLCATS, forward=False).numpy()
        show_samples(imgs_plot, fname=f'{resultsdir}/q2_a_dset{dset_id}_data.png',title='Data')
        model.eval()
        with torch.no_grad():
            zs, _ = model.encoder(imgs)
            z1 = torch.repeat_interleave(zs[:10, ...], 10, dim=0)
            z2 = torch.repeat_interleave(zs[10:, ...], 10, dim=0)
            weights = torch.linspace(0., 1., 10, device=DEVICE).repeat(10)[:, None]
            zs = weights * z1 + (1-weights) * z2
            inter, _ = model.decoder(zs)
        return dequantize(inter.to("cpu"), COLCATS, forward=False)

    inter = interpolations()

    train_losses = np.array([losses_train['nelbo'], losses_train['rec'], losses_train['kl']]).T
    test_losses = np.array([losses_test['nelbo'], losses_test['rec'], losses_test['kl']]).T

    return train_losses, test_losses, samples.numpy(), reconstruct.numpy(), inter.numpy()


def q2_b(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    modelpath = f'{resultsdir}/q2_b_dset{dset_id}_model.pickle'

    Z_DIM = 16
    COLCATS = 256

    train_data = dequantize(torch.from_numpy(train_data), COLCATS)
    test_data = dequantize(torch.from_numpy(test_data), COLCATS)
    # train_data = train_data[:200, ...]
    # test_data = test_data[:200, ...]

    x_dim = train_data.shape[1]

    model = nn_.VaeAf(x_dim, Z_DIM, [512, 512], nn_.EncoderConv, nn_.DecoderConv).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = defaultdict(list), defaultdict(list)

    if TRAIN:
        print(f"Training q1_a on {DEVICE}.")
        trainloader = DataLoader(TensorDataset(train_data),
                                 batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(TensorDataset(test_data),
                                batch_size=BATCH_SIZE)

        callback_list = [cb.CombinedScheduler('lr', ['cosine_sched', 'cosine_sched'],
                                              0.2, LEARN_RATE, MAXLEARN_RATE, LEARN_RATE)]

        learner = Learner(model, optimizer, trainloader, testloader, model.loss_func, DEVICE, callback_list)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        for key, value in l_train.items():
            losses_train[key].extend(value)
        for key, value in l_test.items():
            losses_test[key].extend(value)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
        save_lr_plot(learner.schedule['lr'], MAX_EPOCHS, f'{resultsdir}/q2_b_dset{dset_id}_lrschedule.png')
    else:
        print(f"No training, only generating data from last available model.")

    mu_x = model.sample(100, DEVICE)
    samples = dequantize(mu_x.to("cpu"), COLCATS, forward=False)

    model.eval()
    with torch.no_grad():
        recdata = test_data[:50, ...]
        reconstruct, _, _ = model(recdata.to(DEVICE))
        reconstruct = torch.cat((recdata, reconstruct.to("cpu")), dim=1).view(100, 3, 32, 32)
        reconstruct = dequantize(reconstruct, COLCATS, forward=False)

    def interpolations():
        print("Interpolations ...")
        testidx = torch.randint(0, test_data.shape[0], (20,))
        imgs = torch.index_select(test_data, 0, testidx).to(DEVICE)
        imgs_plot = dequantize(imgs.to("cpu"), COLCATS, forward=False).numpy()
        show_samples(imgs_plot, fname=f'{resultsdir}/q2_b_dset{dset_id}_data.png',title='Data')
        model.eval()
        with torch.no_grad():
            zs, _ = model.encoder(imgs)
            z1 = torch.repeat_interleave(zs[:10, ...], 10, dim=0)
            z2 = torch.repeat_interleave(zs[10:, ...], 10, dim=0)
            weights = torch.linspace(0., 1., 10, device=DEVICE).repeat(10)[:, None]
            zs = weights * z1 + (1-weights) * z2
            inter, _ = model.decoder(zs)
        return dequantize(inter.to("cpu"), COLCATS, forward=False)

    inter = interpolations()

    train_losses = np.array([losses_train['nelbo'], losses_train['rec'], losses_train['kl']]).T
    test_losses = np.array([losses_test['nelbo'], losses_test['rec'], losses_test['kl']]).T

    return train_losses, test_losses, samples.numpy(), reconstruct.numpy(), inter.numpy()


def q3(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """

    modelpath = f'{resultsdir}/q3_model.pickle'

    COLCATS = 256
    CODEDIM = 256
    NCODES = 128

    train_data = dequantize(torch.from_numpy(train_data), COLCATS)
    test_data = dequantize(torch.from_numpy(test_data), COLCATS)
    # train_data = train_data[:200, ...]
    # test_data = test_data[:200, ...]

    n, c, h, w = train_data.shape
    n_dims = c*h*w

    model = VqVae(c, CODEDIM, NCODES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    trainloader = DataLoader(TensorDataset(train_data),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data),
                            batch_size=BATCH_SIZE)

    if RELOAD and Path(modelpath).exists():
        model, optimizer, losses_train, losses_test = reload_modelstate(model, optimizer, modelpath)
    else:
        losses_train, losses_test = [], []

    if TRAIN:
        print(f"Training q1_a on {DEVICE}.")

        callback_list = []

        learner = VqLearner(model, optimizer, trainloader, testloader, model.loss_func, DEVICE, callback_list)
        l_train, l_test = learner.fit(MAX_EPOCHS)
        losses_train.extend(l_train)
        losses_test.extend(l_test)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses_train': losses_train,
                    'losses_test': losses_test},
                   modelpath)
        print(f"Saved model to {modelpath}.")
        # save_lr_plot(learner.schedule['lr'], MAX_EPOCHS, f'{resultsdir}/q3_dset{dset_id}_lrschedule.png')
    else:
        print(f"No training, only generating data from last available model.")

    # reconstructions
    model.eval()
    with torch.no_grad():
        recdata = test_data[:50, ...]
        reconstruct, _, _ = model(recdata.to(DEVICE))
        reconstruct = torch.cat((recdata, reconstruct.to("cpu")), dim=1).view(100, 3, 32, 32)
        reconstruct = dequantize(reconstruct, COLCATS, forward=False)

    losses_train_vqvae = np.array([x / n_dims for x in losses_train])
    losses_test_vqvae = np.array([x / n_dims for x in losses_test])


    # train prior for generations
    z_train = torch.empty(n, 1, 8, 8)
    z_test = torch.empty(test_data.shape[0], 1, 8, 8)

    model.eval()
    with torch.no_grad():
        i = 0
        for batch in trainloader:
            ztemp = model(batch[0].to(DEVICE), learnprior=True)
            z_train[i:i+BATCH_SIZE, ...] = ztemp.to("cpu")
            i += BATCH_SIZE
        i = 0
        for batch in testloader:
            ztemp = model(batch[0].to(DEVICE), learnprior=True)
            z_test[i:i+BATCH_SIZE, ...] = ztemp.to("cpu")
            i += BATCH_SIZE
        print(f"Generated z datasets for training prior.")

    train_targets, train_data = z_train.long(), rescale(z_train, 0., NCODES - 1.)
    test_targets, test_data = z_test.long(), rescale(z_test, 0., NCODES - 1.)

    trainloader = DataLoader(TensorDataset(train_data, train_targets),
                             batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(TensorDataset(test_data, test_targets),
                            batch_size=BATCH_SIZE)

    modelprior = PixelCNN(1, 512, 3, 10, NCODES).to(DEVICE)
    optimizerprior = optim.Adam(modelprior.parameters(), lr=LEARN_RATE)

    def loss_funcprior(logits, targets):
        """Cross entropy."""
        logits = logits.view(-1, 1, NCODES, 8, 8).permute(0, 2, 1, 3, 4)
        loss = F.cross_entropy(logits, targets, reduction='none')
        return loss.sum(dim=(1, 2, 3)).mean(dim=0)

    losses_train, losses_test = [], []
    learnerprior = Learner(modelprior, optimizerprior, trainloader, testloader, loss_funcprior, DEVICE)
    l_train, l_test = learnerprior.fit(MAX_EPOCHS)
    losses_train.extend(l_train)
    losses_test.extend(l_test)

    losses_train_pixelcnn = np.array([x / 64 for x in losses_train])
    losses_test_pixelcnn = np.array([x / 64 for x in losses_test])

    model.eval()
    modelprior.eval()
    with torch.no_grad():
        samplesz = modelprior.sample_data(100, (8, 8), DEVICE)
        samplesz = descale(samplesz, 0., NCODES - 1.)
        samples = model.sample(samplesz).to("cpu")
        samples = dequantize(samples, COLCATS, forward=False)

    return losses_train_vqvae, losses_test_vqvae, losses_train_pixelcnn, losses_test_pixelcnn, samples.numpy(), reconstruct.numpy()
