"""Hooks inspired by fastai."""


class Hook():
    """Base class for hooks."""

    def __init__(self, layer, forward=False, backward=False):
        self.forward = forward
        self.backward = backward
        self.hook_forward = layer.register_forward_hook(self.forward_hook) if forward else None
        self.hook_backward = layer.register_forward_hook(self.backward_hook) if backward else None

    def __del__(self):
        self.remove()

    def remove(self):
        if self.forward:
            self.hook_forward.remove()
        if self.backward:
            self.hook_backward.remove()

    @staticmethod
    def forward_hook(self, *args):
        pass

    @staticmethod
    def backward_hook(self, *args):
        pass


class ChannelOutputsHook(Hook):
    """Hook for monotoring layer outputs for actnorm initialization."""

    def __init__(self, layer, forward=True, backward=False):
        super().__init__(layer, forward, backward)

    @staticmethod
    def forward_hook(self, input, output):
        if not hasattr(self, 'output_stats'):
            self.output_stats = {}
        if isinstance(output, tuple):
            output = output[0]
        self.output_stats['means'] = output.mean(dim=(0, 2, 3)).detach()
        self.output_stats['stds'] = output.std(dim=(0, 2, 3)).detach()


class OutputStatsHook(Hook):
    """Hook for monotoring layer outputs."""

    def __init__(self, layer, forward=True, backward=False):
        print(f'Initialising hook for {layer}.')
        super().__init__(layer, forward, backward)

    @staticmethod
    def forward_hook(self, input, output):
        if not hasattr(self, 'output_stats'):
            self.output_stats = {'means': [], 'stds': []}
        if isinstance(output, tuple):
            output = output[0]
        self.output_stats['means'].append(output.mean().item())
        self.output_stats['stds'].append(output.std().item())
        print(f"In hook: {output.mean()}")
