# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PiecewiseLinearCurve(torch.nn.Module):
    def __init__(self, c_0, c_1,num_decoders,f=None, G=None, n_intervals=10,device='cpu',):
        super(PiecewiseLinearCurve, self).__init__()
        if G is None and f is None:
            raise ValueError("Either G or vae must be provided.")
        self.num_decoders = num_decoders
        c_0 = c_0.detach().clone().requires_grad_(False).to(device)
        c_1 = c_1.detach().clone().requires_grad_(False).to(device)
        self.n_intervals = n_intervals
        self.t_space = torch.linspace(0, 1, n_intervals + 1,device=device)[None, :]
        pertubation = torch.randn(c_0.shape[0], n_intervals - 1,device=device) * 0.1 * torch.norm(c_1 - c_0, 2)
        self.c_free = nn.Parameter((c_1 - c_0) * self.t_space[:, 1:-1] + c_0 + pertubation)  # initial value of c as a linear interpolation between c_0 and c_1
        self.c_0 = c_0
        self.c_1 = c_1
        self.G = G
        self.f = f

        self.initial_c = self.c().detach().clone()

    def c(self):
        return torch.cat([self.c_0, self.c_free, self.c_1], dim=1)

    def curve_grad(self):
        c = self.c()
        diffs = c[:, 1:] - c[:, :-1]
        dt = 1 / self.n_intervals
        return diffs / dt

    def norm(self,x,y):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        expected_norm = 0
        for idx in range(self.num_decoders):
            G = self.G(idx)(y) 
            expected_norm = x.T @ G @ x
        return expected_norm / self.num_decoders
    
    def speed(self):
        return np.array([self.norm(curve_velocity, curve_position).item() for curve_velocity, curve_position in zip(self.curve_grad().T, self.c().T)])

    def length(self):
        # uses finite differences instead of G
        c = self.c()
        norm2 = 0
        dt = 1 / self.n_intervals
        indices = [(i, j) for i in range(self.num_decoders) for j in range(i, self.num_decoders)]
        for i in range(1, self.n_intervals + 1):
            expected_value = 0
            for idx in indices:
                f_c_s = self.f(c[:, i].unsqueeze(0),idx[0])
                f_c_s_minus_1 = self.f(c[:, i - 1].unsqueeze(0),idx[1])
                expected_value += torch.sum((f_c_s - f_c_s_minus_1) ** 2).sqrt()
            norm2 += expected_value / len(indices)
        return norm2
    
    def riemann_sum(self):
        # uses G directly
        # TODO
        c = self.c()
        c_dot = self.curve_grad()
        norm2 = 0
        for i in range(self.n_intervals):
            norm2 += self.norm(c_dot[:,i].unsqueeze(1),c[:,i])
        return norm2 / self.n_intervals

    def finite_diff(self):
        # uses finite differences instead of G
        c = self.c()
        norm2 = 0
        dt = 1 / self.n_intervals
        indices = [(i, j) for i in range(self.num_decoders) for j in range(i, self.num_decoders)]
        for i in range(1, self.n_intervals + 1):
            expected_value = 0
            for idx in indices:
                f_c_s = self.f(c[:, i].unsqueeze(0),idx[0])
                f_c_s_minus_1 = self.f(c[:, i - 1].unsqueeze(0),idx[1])
                expected_value += torch.sum((f_c_s - f_c_s_minus_1) ** 2)/dt
            norm2 += expected_value / len(indices)
        return norm2
    
    def forward(self):
        # calculate the energy of the curve
        if self.f is not None:
            return self.finite_diff()
        else:
            return self.riemann_sum()
    

class PolynomialCurve(torch.nn.Module):
    def __init__(self,c_0,c_1,f=None, G=None, degree=2,n_intervals=10,device='cpu'):
        super(PolynomialCurve, self).__init__()
        if G is None and f is None:
            raise ValueError("Either G or vae must be provided.")
        c_0 = c_0.detach().clone().requires_grad_(False).to(device)
        c_1 = c_1.detach().clone().requires_grad_(False).to(device)
        self.device = device
        self.degree = degree
        self.t_points = n_intervals+1
        self.t_space = torch.linspace(0,1,self.t_points,device=device)[None,:]
        self.t_tensor = torch.vstack([self.t_space[0]**i for i in range(degree+1)])
        pertubation = torch.randn(c_0.shape[0],degree-1,device=device)*3*torch.norm(c_1-c_0,2)
        self.w = nn.Parameter(torch.zeros((c_0.shape[0],degree-1),device=device)+pertubation) # initial value of c as a linear interpolation between c_0 and c_1
        
        self.c_0 = c_0
        self.c_1 = c_1
        self.G = G
        self.f = f

        self.initial_c = self.c().detach().clone()

    
    def c(self):
        w_0 = torch.zeros(self.w.shape[0],1,device=self.device)
        w_K = -torch.sum(self.w,dim=1)[:,None]
        w = torch.cat([w_0,self.w,w_K],dim=1)
        return w @ self.t_tensor+(1-self.t_space)*self.c_0+self.t_space*self.c_1

    def curve_grad(self):
        t_tensor_grad = self.t_tensor[:-1,:]
        w_K = -torch.sum(self.w,dim=1)[:,None]
        degrees = torch.arange(1,self.degree,device=device)
        w = torch.cat([degrees*self.w,self.degree*w_K],dim=1)
        return  w @ t_tensor_grad+self.c_1-self.c_0
    
    def norm(self,x,y):
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return x.T @ self.G(y) @ x
    
    def speed(self):
        return np.array([self.norm(curve_velocity, curve_position).item() for curve_velocity, curve_position in zip(self.curve_grad().T, self.c().T)])


    def riemann_sum(self):
        c = self.c()
        c_dot = self.curve_grad()
        norm2 = 0
        for i in range(c_dot.shape[1]):
            norm2 += c_dot[:,i] @ self.G(c[:,i]) @ c_dot[:,i]
        return norm2 / c_dot.shape[1]
    
    def finite_diff(self):
        c = self.c()
        norm2 = 0
        for i in range(1, self.t_points):
            f_c_s = self.f(c[:, i].unsqueeze(0))
            f_c_s_minus_1 = self.f(c[:, i - 1].unsqueeze(0))
            norm2 += torch.sum((f_c_s - f_c_s_minus_1) ** 2 )
        return norm2
    
    def forward(self):
        # calculate the energy of the curve
        if self.f is not None:
            return self.finite_diff()
        else:
            return self.riemann_sum()
    
    
def optimize(curve, n_steps=10):
    # L-BFGS

    optimizer = torch.optim.LBFGS(curve.parameters(),
                                    history_size=10,
                                    max_iter=4,
                                    line_search_fn="strong_wolfe")
    #optimizer = torch.optim.Adam(curve.parameters(), lr=1e-2)
    def closure():
        optimizer.zero_grad()
        objective = curve()
        objective.backward()
        return objective
    for i in range(n_steps):
        if i % (n_steps // min(n_steps,10)) == 0:
            print(f"Step {i}, Loss: {curve().item()}")
        optimizer.step(closure)

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)
    



class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

    def jacobian(self, z):
        """
        Calculate the Jacobian of the mean with respect to the input z.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        
        Returns:
        jacobian: [torch.Tensor]
                  A tensor of dimension `(batch_size, feature_dim1 * feature_dim2, M)`
        """
        z = z.requires_grad_(True)
        if len(z.shape) == 1:  
            z = z.unsqueeze(0)
        means = self.decoder_net(z)
        batch_size = means.shape[0]
        jacobian = torch.autograd.functional.jacobian(lambda z: self.decoder_net(z).view(batch_size, -1), z)
        jacobian = jacobian.view(batch_size, -1, *z.shape[1:]).detach().clone().requires_grad_(False)
        return jacobian
    
    def pull_back_metric(self, z):
        """
        Compute the pull-back metric of the decoder network at the given latent variable z.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.

        Returns:
        metric: [torch.Tensor]
                A tensor of dimension `(batch_size, M, M)`
        """
        jacobian = self.jacobian(z)
        metric = torch.matmul(jacobian.transpose(1, 2), jacobian)
        return metric


class EnsembleVAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoders, encoder,M=3):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoders: list [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(EnsembleVAE, self).__init__()
        self.M = M
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)
        self.encoder = encoder

    def elbo(self, x,i):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoders[i](z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x,i):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x,i)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs*model.M
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                decoder_idx = random.randint(0, model.M - 1)
                loss = model(x,decoder_idx)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break
def plot_posterior_samples(model, data_loader, device, posterior_plot_name, n_samples=20, grid_size=100, curves=None):
    n_samples = min(len(data_loader.dataset), n_samples)
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q_x = model.encoder(x)
            z_x = q_x.mean
            latents.append(z_x.cpu().numpy())
            labels.append(y)
            #if len(latents) >= n_samples:
            #    break
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Create a grid for contour plotting
    x_min, x_max = latents[:, 0].min() - 1, latents[:, 0].max() + 1
    y_min, y_max = latents[:, 1].min() - 1, latents[:, 1].max() + 1
    x_vals, y_vals = np.meshgrid(np.linspace(x_min, x_max, grid_size), 
                                    np.linspace(y_min, y_max, grid_size))
    
    grid_points = np.stack([x_vals.ravel(), y_vals.ravel()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        prior_density = model.prior().log_prob(grid_tensor).exp().cpu().numpy()  # Compute prior probability density
        prior_density = prior_density.reshape(grid_size, grid_size)

    # Plot the latent space
    plt.figure(figsize=(8, 6))
    plt.contourf(x_vals, y_vals, prior_density, levels=7, cmap="viridis") 
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.8, marker='.')
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if isinstance(model.prior, GaussianPrior):
        model_prior_name = "Gaussian Prior"
    plt.title(f"Latent Space with Aggregate Posterior and {model_prior_name}")

    # Plot curves if provided
    if curves is not None:
        for curve in curves:
            plt.plot(curve[0], curve[1], linewidth=2, label='Geodesic Curve',color='orange')

    # Legend
    scatter_legend = plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='black', markersize=10, label="Aggregate Posterior Samples")
    prior_patch = mpatches.Patch(color="yellow", label="Prior Density")    
    plt.legend(handles=[scatter_legend, prior_patch], loc="upper right")
    plt.savefig(f"plots/{posterior_plot_name}", dpi=300, bbox_inches="tight")
    plt.show()

def plot_speeds(speeds, output_file="speeds_plot.png"):
    """
    Plot speeds along segments for multiple curves.

    Parameters:
    speeds: [np.ndarray]
        Array of shape (n_curves, n_segments) containing speeds along n_curves.
    output_file: [str]
        File name to save the plot (default: "speeds_plot.png").
    """
    n_curves, n_segments = speeds.shape
    segments = np.arange(n_segments)

    plt.figure(figsize=(10, 6))
    for i in range(n_curves):
        plt.plot(segments, speeds[i])

    plt.xlabel("Segment")
    plt.ylabel("Speed")
    plt.title("Speed Along Segments for Multiple Curves")
    plt.grid(True)
    plt.savefig("plots/" + output_file, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    import random
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
            args.num_decoders
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
            args.num_decoders
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)],
            GaussianEncoder(new_encoder()),
            args.num_decoders
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        n_plot_samples = args.num_curves

        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        test_data_iter = iter(test_data_loader)
       
        x, _ = next(test_data_iter)
        x = x.to(device)
        z = model.encoder(x).mean

        # Choose args.num_curves different pairs at random
        torch.manual_seed(0)
        np.random.seed(0)
        pairs = []
        while len(pairs) < args.num_curves:
            idx = np.random.choice(len(z), 2, replace=False)
            if idx.tolist() not in pairs:
                pairs.append(idx.tolist())

        curves = []
        speeds = []
        for i in range(args.num_curves):
            print(f"Optimizing curve {i}")
            idx = pairs[i]
            z0, z1 = z[idx[0]].unsqueeze(0), z[idx[1]].unsqueeze(0)
            f = lambda x,i: model.decoders[i](x).mean
            J = lambda i: model.decoders[i].pull_back_metric
            curve = PiecewiseLinearCurve(z0.T, z1.T,args.num_decoders,f=f, G=J, n_intervals=args.num_t, device=device).to(device)
            #curve = PolynomialCurve(z0.T, z1.T, f=f, G=model.decoder.pull_back_metric, degree=3,n_intervals=args.num_t, device=device).to(device)
            with torch.no_grad(): speed = curve.speed()
            with torch.no_grad(): length = curve.length()
            print(f"Speeds before optimization: mean={np.mean(speed):.4f} ± std={np.std(speed):.4f}")
            print(f"length before optimization: {length}")
            optimize(curve, n_steps=5)
            c = curve.c().detach().cpu().numpy()
            with torch.no_grad(): speed = curve.speed()
            with torch.no_grad(): length = curve.length()
            
            print(f"Speeds after optimization: mean={np.mean(speed):.4f} ± std={np.std(speed):.4f}")
            print(f"length after optimization: {length}")
            speeds.append(speed)
            
            curves.append(c)
            print(f"Curve {i} optimized with energy {curve()}")
            print("-"*80)
        curves = np.array(curves)
        speeds = np.array(speeds)
        # Plot the result in the latent space
        with torch.no_grad():
            base_name = f"posterior_{curve.__class__.__name__}_{len(curves)}"
            existing_files = glob.glob(f"plots/{base_name}_*.pdf")
            file_count = len(existing_files) + 1
            plot_posterior_samples(model, mnist_test_loader, device, f"{base_name}_{file_count}.pdf", n_samples=n_plot_samples, grid_size=100, curves=curves)
            base_name = f"speed_{curve.__class__.__name__}_{len(curves)}"
            existing_files = glob.glob(f"plots/{base_name}_*.pdf")
            file_count = len(existing_files) + 1
            plot_speeds(speeds, output_file=f"{base_name}_{file_count}.pdf")
