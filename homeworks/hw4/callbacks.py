"""Callbacks inspired by fastai."""


class Callback():
    """Base class for callbacks."""

    def init_learner(self, learner):
        self.learner = learner

    @property
    def class_name(self):
        return self.__class__.__name__.lower()


class SampleData(Callback):
    """Callback for sampling data."""

    def __init__(self, nsamples, samplingepoch=None):
        self.samplingepoch = samplingepoch
        self.nsamples = nsamples
        self.samples = {}

    def epoch_end(self):
        if self.learner.epoch == self.samplingepoch:
            self.samples[self.samplingepoch] = self.generator_sample()

    def fit_end(self):
        self.samples['end'] = self.generator_sample()
        self.learner.samples = self.samples

    def generator_sample(self):
        samples = self.learner.model.sample(self.nsamples)
        return samples


class Discriminate(Callback):
    """Callback for discriminating fakes."""

    def __init__(self, fakes, samplingepoch=None):
        self.fakes = fakes[:, None]
        self.samplingepoch = samplingepoch
        self.classes = {}

    def epoch_end(self):
        if self.learner.epoch == self.samplingepoch:
            self.classes[self.samplingepoch] = self.discriminate()

    def fit_end(self):
        self.classes['end'] = self.discriminate()
        self.learner.classes = self.classes

    def discriminate(self):
        classes = self.learner.model.discriminate(self.fakes)
        return classes.squeeze()
