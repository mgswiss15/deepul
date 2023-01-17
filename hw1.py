import torch
import torch.nn as nn
import torch.nn.functional as F


# quick an dirty implementation for 2D inputs
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.mask = torch.ones(out_features, in_features)
        self.mask[:, in_features//2:] = 0.
        # print(f'Mask {self.mask}')

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedOut(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.mask = torch.ones(out_features, in_features)
        self.mask[:out_features//2, :] = 0.
        # print(f'Mask {self.mask}')

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    def __init__(self, n_hidden, d):
        super().__init__()
        self.l1 = MaskedLinear(2, n_hidden)
        self.l11 = nn.Linear(n_hidden, n_hidden*2)
        self.l12 = nn.Linear(n_hidden*2, n_hidden*2)
        self.l2 = MaskedOut(n_hidden*2, 2*d)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l11(x))
        x = F.relu(self.l12(x))
        return self.l2(x)


# quick an dirty implementation for onehot encoding
class MADE_oh(nn.Module):
    def __init__(self, n_hidden, d):
        super().__init__()
        self.l1 = MaskedLinear(2*d, n_hidden)
        self.l11 = nn.Linear(n_hidden, n_hidden*2)
        self.l12 = nn.Linear(n_hidden*2, n_hidden*2)
        self.l2 = MaskedOut(n_hidden*2, 2*d)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l11(x))
        x = F.relu(self.l12(x))
        return self.l2(x)
