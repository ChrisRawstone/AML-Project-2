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

def new_encoder(M):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16x14x14
        nn.Softmax(dim=1),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32x7x7
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32x4x4
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

def new_decoder(M):
    return nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )