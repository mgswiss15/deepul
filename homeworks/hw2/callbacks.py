"""Callbacks inspired by fastai."""

import math


class Callback():
    """Base class for callbacks."""

    def init_learner(self, learner):
        self.learner = learner

    @property
    def class_name(self):
        return self.__class__.__name__.lower()


class ParamScheduler(Callback):
    """Callback for parameter scheduling."""

    def __init__(self, param_name, sched_func, *args):
        self.param_name = param_name
        self.sched_func = getattr(self, sched_func, None)
        self.args = args

    def fit_begin(self, *args, **kwargs):
        self.learner.schedule = getattr(self.learner, 'schedule', {})
        self.learner.schedule[self.param_name] = []
        self.iters = len(self.learner.trainloader)
        for pg in self.learner.optimizer.param_groups:
            pg[self.param_name] = self.args[0]

    def train_epoch_begin(self, *args, **kwargs):
        self.iter = 0.

    def train_batch_begin(self, *args, **kwargs):
        epoch_iter = self.learner.epoch + (self.iter / self.iters)
        for pg in self.learner.optimizer.param_groups:
            position = epoch_iter / self.learner.epochs
            pg[self.param_name] = self.sched_func(position, *self.args)
        self.learner.schedule[self.param_name].append(pg[self.param_name])
        self.iter += 1.

    def linear_sched(self, position, start, end, *args, **kwargs):
        return start + position * (end-start)

    def cosine_sched(self, position, start, end, *args, **kwargs):
        return start + (1. + math.cos(math.pi * (1. - position))) * (end-start)/2.

    def fixed_sched(self, position, start, *args, **kwargs):
        return start


class CombinedScheduler(Callback):
    """"Callback for combined parameter schedulling."""

    def __init__(self, param_name, sched_list, pct_switch, start, mid, end):
        self.param_name = param_name
        if len(sched_list) != 2:
            raise Exception(f"Number of schedules {len(sched_list)} must be equal 2.")
        if pct_switch > 1. or pct_switch < 0.:
            raise Exception(f"Switch percent has to be between 0 and 1, not {pct_switch}.")
        self.pct_switch = pct_switch
        self.sched_list = sched_list
        self.start, self.mid, self.end = start, mid, end

    def fit_begin(self, *args, **kwargs):
        self.learner.schedule = getattr(self.learner, 'schedule', {})
        self.learner.schedule[self.param_name] = []
        self.iters = len(self.learner.trainloader)
        self.split = self.pct_switch * self.learner.epochs
        for pg in self.learner.optimizer.param_groups:
            pg[self.param_name] = self.start
        self.p1 = ParamScheduler(self.param_name, self.sched_list[0], self.start, self.mid)
        self.p2 = ParamScheduler(self.param_name, self.sched_list[1], self.mid, self.end)
        self.p1.init_learner(self.learner)
        self.p2.init_learner(self.learner)

    def train_epoch_begin(self, *args, **kwargs):
        self.iter = 0.

    def train_batch_begin(self, *args, **kwargs):
        epoch_iter = self.learner.epoch + (self.iter / self.iters)
        for pg in self.learner.optimizer.param_groups:
            if epoch_iter < self.split:
                position = epoch_iter / self.split
                pg[self.param_name] = self.p1.sched_func(position, self.start, self.mid)
            else:
                position = (epoch_iter-self.split) / (self.learner.epochs-self.split)
                pg[self.param_name] = self.p2.sched_func(position, self.mid, self.end)
        self.learner.schedule[self.param_name].append(pg[self.param_name])
        self.iter += 1.
