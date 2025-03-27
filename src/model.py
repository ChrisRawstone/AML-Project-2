# src/model.py
# do not remove this comment or the comment above


import torch
import torch.nn as nn
import torch.distributions as td

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), reinterpreted_batch_ndims=3)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        logp_x_given_z = self.decoder(z).log_prob(x)
        kl_qp = q.log_prob(z) - self.prior().log_prob(z)
        return torch.mean(logp_x_given_z - kl_qp)

    def sample(self, n_samples=1):
        z = self.prior().sample((n_samples,))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)

import torch
import torch.nn as nn

def new_encoder(M):
    """
    Example encoder network for a VAE with no saturating activations like softmax.
    We use ReLU + BatchNorm in intermediate layers, and produce 2*M outputs (mean and log-std).
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),   # 16 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(16),

        nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 x 7 x 7
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32 x 4 x 4
        nn.ReLU(),

        nn.Flatten(),             # Flatten to shape (batch_size, 32*4*4 = 512)
        nn.Linear(512, 2 * M),    # Outputs (mean, log_std) of dim M each
    )

def new_decoder(M):
    """
    Example decoder network for a VAE with no softmax in intermediate layers.
    Typically, we apply a final Sigmoid if input data are in [0,1].
    For natural images in [0,1], the final Sigmoid is common.
    For e.g. [-1,1], use Tanh. 
    """
    return nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),

        nn.Unflatten(-1, (32, 4, 4)),  # shape = (batch_size, 32, 4, 4)
        nn.BatchNorm2d(32),

        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),  # 32 x 7 x 7
        nn.ReLU(),
        nn.BatchNorm2d(32),

        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 16 x 14 x 14
        nn.ReLU(),
        nn.BatchNorm2d(16),

        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # 1 x 28 x 28

        # If data is in [0,1], it's common to apply Sigmoid here:
        nn.Sigmoid(),
    )