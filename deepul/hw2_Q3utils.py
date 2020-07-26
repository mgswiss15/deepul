''' Generated from Q3_utils.ipynb. Do not edit .py file.'''
from deepul.exp_utils2 import *
from torch.distributions import Normal

class AffineCoupling(nn.Module):
    def __init__(self, channels, res_filters, res_blocks):
        super(AffineCoupling, self).__init__()
        self.resnet = SimpleResnet(channels, channels * 2, res_filters, res_blocks)
        self.scale_layer = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0)
        )
       
    def forward(self, x):
        mask = self.make_mask(x)
        z = x * mask
        log_scale, shift = self.resnet(z).chunk(2, dim=1)
        log_scale = self.scale_layer(log_scale)
        z2 = (1 - mask) * (x * log_scale.exp() + shift)
        z = z + z2
        log_det_jacobian = ((1 - mask) * log_scale).sum(dim=[1, 2, 3])
        xprime = self.reverse(z.detach())
        return z, log_det_jacobian
    
    def reverse(self, z):
        mask = self.make_mask(z)
        x = z * mask
        log_scale, shift = self.resnet(x).chunk(2, dim=1)
        log_scale = self.scale_layer(log_scale)
        x2 = (1 - mask) * (z - shift) / log_scale.exp()
        x = x + x2
        return x

    def make_mask(self, x):
        raise NotImplementedError('Should be implemented by subclasses')
    

class AffineCouplingChecker(AffineCoupling):
    '''Checkerboard masking for coupling layher as in RealNVP'''
    def __init__(self, start_from, channels, res_filters, res_blocks):
        super(AffineCouplingChecker, self).__init__(channels, res_filters, res_blocks)
        self.start_from = start_from
    
    def make_mask(self, x):
        mask = torch.zeros_like(x)
        if self.start_from==1:
            mask[:, :, ::2, ::2] = 1
            mask[:, :, 1::2, 1::2] = 1
        elif self.start_from==0:
            mask[:, :, 1::2, ::2] = 1
            mask[:, :, ::2, 1::2] = 1
        else:
            raise Exception(f'start_from should be 0 or 1, not {start_from}')
        return mask

            
class AffineCouplingChannel(AffineCoupling):
    '''Channel-wise masking as in RealNVP'''
    def __init__(self, start_from, channels, res_filters, res_blocks):
        super(AffineCouplingChannel, self).__init__(channels, res_filters, res_blocks)
        self.start_from = start_from
    
    def make_mask(self, x):
        mask = torch.zeros_like(x)
        if self.start_from==1:
            mask[:, ::2, :, :] = 1
        elif self.start_from==0:
            mask[:, 1::2, :, :] = 1
        else:
            raise Exception(f'start_from should be 0 or 1, not {start_from}')
        return mask
            

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
        
    @staticmethod
    def forward(x):
        b, c, h, w = x.shape
        return x.view(b, c*4, h//2, w//2)

    
class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()
        
    @staticmethod
    def forward(x):
        b, c, h, w = x.shape
        return x.view(b, c//4, h*2, w*2)

def dequantize(data, alpha=0.05):
    '''Dequantize with logits as in RealNVP'''
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data, dtype=torch.float)
    z = alpha + (1 - alpha) * (data / 4)
    logit = z.log() - (1-z).log()  # logits
    out = (logit - logit.min())/(logit.max()-logit.min())
    return out
    


class ResBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, 1, stride=1, padding=0),
        )
        
    def forward(self, x):
        out = self.sequential(x)
        return x + out
    

class SimpleResnet(nn.Module):
    def __init__(self, channels, channels_out=2, res_filters=128, res_blocks=8):
        super(SimpleResnet, self).__init__()
        reslayers = []
        for _ in range(res_blocks):
            reslayers.append(ResBlock(res_filters))
        resblock = nn.Sequential(*reslayers)
        self.sequential = nn.Sequential(
            nn.Conv2d(channels, res_filters, 3, stride=1, padding=1),
            resblock,
            nn.ReLU(),
            nn.Conv2d(res_filters, channels_out, 3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return self.sequential(x)

    

class SigmoidLayer(nn.Module):
    def __init__(self):
        super(SigmoidLayer, self).__init__()
    
    def forward(self, x):
        z = torch.sigmoid(x)
        det = z * (1 - z)
        return z, det
    
class RealNVP(nn.Module):
    def __init__(self, image_shape, res_filters=128, res_blocks=8, device='cuda', base_distrib='Normal'):
        super(RealNVP, self).__init__()
        self.image_shape = image_shape
        self.res_filters = res_filters
        self.res_blocks = res_blocks
        self.device = device
        self.channels = image_shape[0]
        self.base_distrib = base_distrib
        if self.base_distrib=='Normal':
            self.base = Normal(torch.zeros(image_shape, device=device), torch.ones(image_shape, device=device))
        elif self.base_distrib=='Uniform':
            self.base = Uniform(torch.zeros(image_shape, device=device), torch.ones(image_shape, device=device))
            self.sigmoid = SigmoidLayer()
        else:
            raise NotImplementedError('Sorry, base distribution can only be Uniform or Normal')
        start_from = torch.tensor([1])
        layers = []
        for _ in range(4):
            layers.extend([
                AffineCouplingChecker(start_from, self.channels, self.res_filters, self.res_blocks),
                ActNorm(self.channels)
            ])
            start_from = torch.tensor([1, 0])[start_from]
        self.coupling1 = nn.ModuleList(layers)
        self.squeeze = Squeeze()
        layers = []
        for _ in range(3):
            layers.extend([
                AffineCouplingChannel(start_from, self.channels*4, self.res_filters, self.res_blocks),
                ActNorm(self.channels*4)
            ])
            start_from = torch.tensor([1, 0])[start_from]
        self.coupling2 = nn.ModuleList(layers)
        self.unsqueeze = Unsqueeze()
        layers = []
        for _ in range(3):
            layers.extend([
                AffineCouplingChecker(start_from, self.channels, self.res_filters, self.res_blocks),
                ActNorm(self.channels)
            ])
            start_from = torch.tensor([1, 0])[start_from]
        self.coupling3 = nn.ModuleList(layers)
        
    def forward(self, x):
        log_det_jacobian = torch.tensor([0.], device=self.device)
        for layer in self.coupling1:
            x, ldt = layer(x)
            log_det_jacobian = log_det_jacobian + ldt
        x = self.squeeze(x)
        for layer in self.coupling2:
            x, ldt = layer(x)
            log_det_jacobian = log_det_jacobian + ldt
        x = self.unsqueeze(x)
        for layer in self.coupling3:
            x, ldt = layer(x)
            log_det_jacobian = log_det_jacobian + ldt
        return x, log_det_jacobian
    
    def reverse(self, z):
        for layer in reversed(self.coupling3):
            z = layer.reverse(z)
        z = self.squeeze(z)
        for layer in reversed(self.coupling2):
            z = layer.reverse(z)
        z = self.unsqueeze(z)
        for layer in reversed(self.coupling1):
            z = layer.reverse(z)
        return z
            
    def log_prob_x_from_z(self, z, log_det_jacobian):
        if self.base_distrib == 'Uniform':
            log_prob = log_det_jacobian.abs()  # use independent uniform - this breaks
        else:
            log_prob_z = self.base.log_prob(z)
            log_prob = self.base.log_prob(z) + log_det_jacobian[:, None, None, None]  # can sum log_probs cause independent
        return log_prob
    
    def loss_function(self, z, log_det_jacobian):
        loss = -self.log_prob_x_from_z(z, log_det_jacobian).sum((1, 2, 3)).mean()
        return loss
    
    def eval_log_prob(self, x):
        self.eval()
        with torch.no_grad():
            z, log_det_jacobian = self(x)
            log_prob = self.log_prob_x_from_z(z, log_det_jacobian)
        return log_prob
    
    def sampling(self, size):
        print(f'Begin sampling')
        self.eval()
        with torch.no_grad():
            c, h, w = self.image_shape
            z = torch.randn((size, c, h, w), device=self.device)
            x = self.reverse(z)
            images = x.permute(0, 2, 3, 1)
        print(f'Done sampling')
        return images
    
    def interpolate(self, x):
        print(f'Begin interpolate')
        self.eval()
        with torch.no_grad():
            print('original x', x.max(), x.min())
            z, _ = self(x[:30, ...])
            zb = z[:4, ...].clone()
            for i, phi1 in enumerate(torch.linspace(0, np.pi/2, 5)):
                for j, phi2 in enumerate(torch.linspace(0, np.pi/2, 6)):
                    z[i*6 + j, ...] = torch.cos(phi1) * (
                    torch.cos(phi2)*zb[0, ...] + torch.sin(phi2)*zb[1, ...]) + torch.sin(phi1) * (
                    torch.cos(phi2)*zb[2, ...] + torch.sin(phi2)*zb[3, ...])
            xprime = self.reverse(z)
            print('Reconstructed x', xprime.max(), xprime.min())
            images = xprime.permute(0, 2, 3, 1)
            return images

class ActNorm(nn.Module):
    '''ActNorm as in Glow paper'''
    def __init__(self, channels, needs_init=True):
        super(ActNorm, self).__init__()
        self.needs_init = needs_init
        self.scale = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
       
    def forward(self, x):
        h, w = x.shape[2:]
        z = x * self.scale[None, :, None, None]
        z = z + self.bias[None, :, None, None]
        log_det_jacobian = self.scale.abs().log().sum() * h * w
        # init params from first forward pass
        if self.needs_init:
            with torch.no_grad():
                means = z.mean(dim=[0, 2, 3])
                stds = z.std(dim=[0, 2, 3])
                self.bias.data = - means / stds
                self.scale.data = 1 / stds
            self.needs_init = False
        return z, log_det_jacobian
    
    def reverse(self, z):
        x = z - self.bias[None, :, None, None]
        x = x / self.scale[None, :, None, None]
        return x
   
def init_actnorm(model, data_loader, device):
    model.eval()
    with torch.no_grad():   
        batch = next(iter(data_loader)).to(device)
        model(batch)
