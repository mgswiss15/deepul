"""Utility functions for hw4."""

import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def q1_gan_plot(data, samples, xs, ys, title):
    fig = plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    return fig


def show_samples(samples, nrow=10, title='Samples'):
    grid_img = make_grid(samples, nrow=nrow)
    fig = plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    return fig
