from .utils import *

resultsdir = '../hw2/results'


def make_moons(n_samples=100, *, shuffle=True, noise=None, random_state=None):
    """Stolen from sklearn Make two interleaving half circles
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int or two-element tuple, optional (default=100)
        If int, the total number of points generated.
        If two-element tuple, number of points in each of two moons.
    shuffle : bool, optional (default=True)
        Whether to shuffle the samples.
    noise : double or None (default=None)
        Standard deviation of Gaussian noise added to the data.
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : array of shape [n_samples, 2]
        The generated samples.
    y : array of shape [n_samples]
        The integer labels (0 or 1) for class membership of each sample.
    """

    if isinstance(n_samples, int):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError:
            raise ValueError('`n_samples` can be either an int or '
                             'a two-element tuple.')

    generator = np.random.RandomState(random_state)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if shuffle:
        idx = generator.permutation(X.shape[0])
        X = X[idx, :]
        y = y[idx]
        # X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def make_scatterplot(points, title=None, filename=None):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    if title is not None:
        plt.title(title)
    # if filename is not None:
    #     plt.savefig("q1_{}.png".format(filename))

######################
##### Question 1 #####
######################

def load_smiley_face(n):
    count = n
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]

def load_half_moons(n):
    return make_moons(n_samples=n, noise=0.1)

def q1_sample_data_1():
    train_data, train_labels = load_smiley_face(2000)
    test_data, test_labels = load_smiley_face(1000)
    return train_data, train_labels, test_data, test_labels

def q1_sample_data_2():
    train_data, train_labels = load_half_moons(2000)
    test_data, test_labels = load_half_moons(1000)
    return train_data, train_labels, test_data, test_labels

def visualize_q1_data(dset_type):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    else:
        raise Exception('Invalid dset_type:', dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
    ax1.set_title('Train Data')
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=1, c=train_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    ax2.set_title('Test Data')
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=1, c=test_labels)
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    print(f'Dataset {dset_type}')
    plt.show()

def show_2d_samples(samples, fname=None, title='Samples'):
    plt.figure()
    plt.title(title)
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlabel('x1')
    plt.ylabel('x2')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def show_2d_latents(latents, labels, fname=None, title='Latent Space'):
    plt.figure()
    plt.title(title)
    plt.scatter(latents[:, 0], latents[:, 1], s=1, c=labels)
    plt.xlabel('z1')
    plt.ylabel('z2')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def show_2d_densities(densities, dset_type, fname=None, title='Densities'):
    plt.figure()
    plt.title(title)
    dx, dy = 0.025, 0.025
    if dset_type == 1: # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_type == 2: # moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    else:
        raise Exception('Invalid dset_type:', dset_type)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    # mesh_xs = ptu.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2))
    # densities = np.exp(ptu.get_numpy(self.log_prob(mesh_xs)))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.pcolor(x, y, densities.reshape([y.shape[0], y.shape[1]]))
    plt.xlabel('z1')
    plt.ylabel('z2')
    if fname is not None:
        savefig(fname)
    else:
        plt.show()

def q1_save_results(dset_type, part, fn):
    if dset_type == 1:
        train_data, train_labels, test_data, test_labels = q1_sample_data_1()
    elif dset_type == 2:
        train_data, train_labels, test_data, test_labels = q1_sample_data_2()
    else:
        raise Exception('Invalid dset_type:', dset_type)

    train_losses, test_losses, densities, latents = fn(train_data, test_data, dset_type)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    save_training_plot(train_losses, test_losses, f'Q1({part}) Dataset {dset_type} Train Plot',
                       f'{resultsdir}/q1_{part}_dset{dset_type}_train_plot.png')
    show_2d_densities(densities, dset_type, fname=f'{resultsdir}/q1_{part}_dset{dset_type}_densities.png')
    show_2d_latents(latents, train_labels, f'{resultsdir}/q1_{part}_dset{dset_type}_latents.png')


######################
##### Question 2 #####
######################

def visualize_q2_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
    name = 'Shape'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs] * 255
    show_samples(images, title=f'{name} Samples')

def q2_save_results(fn):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))

    train_losses, test_losses, samples = fn(train_data, test_data)
    samples = np.clip(samples.astype('float') * 2.0, 0, 1.9999)
    floored_samples = np.floor(samples)

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q2 Dataset Train Plot',
                       f'{resultsdir}/q2_train_plot.png')
    show_samples(samples * 255.0 / 2.0, f'{resultsdir}/q2_samples.png')
    show_samples(floored_samples * 255.0, f'{resultsdir}/q2_flooredsamples.png', title='Samples with Flooring')

######################
##### Question 3 #####
######################

def visualize_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    name = 'CelebA'

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype(np.float32) / 3.0 * 255.0
    show_samples(images, title=f'{name} Samples')

def get_q3_data():
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))
    return train_data, test_data


def q3_save_results(fn, part):
    data_dir = get_data_dir(2)
    train_data, test_data = load_pickled_data(join(data_dir, 'celeb.pkl'))

    train_losses, test_losses, samples, interpolations = fn(train_data, test_data)
    samples = samples.astype('float')
    interpolations = interpolations.astype('float')

    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    save_training_plot(train_losses, test_losses, f'Q3 Dataset Train Plot',
                       f'{resultsdir}/q3_{part}_train_plot.png')
    show_samples(samples * 255.0, f'{resultsdir}/q3_{part}_samples.png')
    show_samples(interpolations * 255.0, f'{resultsdir}/q3_{part}_interpolations.png', nrow=6, title='Interpolations')
