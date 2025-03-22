import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import os
import math
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torchvision.utils import save_image

############################################################
# Model Components (same as before)
############################################################

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
        elbo = torch.mean(logp_x_given_z - kl_qp)
        return elbo

    def sample(self, n_samples=1):
        z = self.prior().sample((n_samples,))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)

############################################################
# Geodesic Computation with History Tracking
############################################################

def compute_geodesic(model, z_start, z_end, num_segments=10, lr=1e-2, steps=2000):
    """
    Compute the geodesic connecting z_start to z_end in latent space
    by minimizing the discrete energy:
         E = sum_{s=1}^{S} || f(z_s) - f(z_{s-1}) ||^2.
         
    This function also records the energy at each optimization step and
    returns the initial (linear) path.
    
    Returns:
       z_opt      : optimized geodesic path (tensor of shape (S+1, latent_dim))
       energy_hist: list of energy values at each optimization step
       z_initial  : initial (linear interpolation) geodesic path.
    """
    device = z_start.device
    # Create initial path by linear interpolation.
    tgrid = torch.linspace(0, 1, num_segments+1, device=device).unsqueeze(-1)
    z_init = z_start + tgrid * (z_end - z_start)
    z_initial = z_init.detach().clone()  # store the initial unoptimized path
    z_vars = nn.Parameter(z_init)

    optimizer = torch.optim.Adam([z_vars], lr=lr)
    energy_history = []

    for step in range(steps):
        # Ensure endpoints are fixed.
        with torch.no_grad():
            z_vars[0] = z_start
            z_vars[-1] = z_end

        # Decode the latent points (use decoder mean).
        decoded = model.decoder(z_vars).mean  # e.g., shape (S+1, 1, 28, 28)
        # Compute discrete energy between consecutive decoded points.
        diffs = decoded[1:] - decoded[:-1]
        energy = (diffs**2).sum()
        energy_history.append(energy.item())

        optimizer.zero_grad()
        energy.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:4d}, energy={energy.item():.4f}")

    return z_vars.detach(), energy_history, z_initial

############################################################
# Plotting Functions
############################################################

def plot_energy_curve(energy_history, save_path=None):
    """
    Plot the energy (loss) over the optimization steps.
    """
    plt.figure(figsize=(8,6))
    plt.plot(energy_history, linewidth=2)
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy')
    plt.title('Energy Curve During Geodesic Optimization')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_geodesic_comparison(latents, labels, z_initial, z_optimized, z_start, z_end, save_path=None):
    """
    Plot all latent points (colored by label) and overlay both the initial
    (dashed) and optimized (solid) geodesic curves between the selected endpoints.
    """
    # Convert tensors to numpy arrays.
    if hasattr(latents, 'detach'):
        latents_np = latents.detach().cpu().numpy()
    else:
        latents_np = latents
    if hasattr(labels, 'detach'):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    cmap = plt.get_cmap('viridis', len(np.unique(labels_np)))
    
    plt.figure(figsize=(8,6))
    plt.scatter(latents_np[:,0], latents_np[:,1], c=labels_np, cmap=cmap, 
                edgecolor='k', s=40, alpha=0.7)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Comparison: Initial vs Optimized Geodesic")
    plt.grid(True)
    
    # Plot initial geodesic (dashed magenta line).
    z_initial_np = z_initial.detach().cpu().numpy() if hasattr(z_initial, 'detach') else z_initial
    plt.plot(z_initial_np[:,0], z_initial_np[:,1], 'o--', color='magenta', linewidth=2, markersize=8,
             label="Initial Geodesic")
    
    # Plot optimized geodesic (solid blue line).
    z_opt_np = z_optimized.detach().cpu().numpy() if hasattr(z_optimized, 'detach') else z_optimized
    plt.plot(z_opt_np[:,0], z_opt_np[:,1], 'o-', color='blue', linewidth=2, markersize=8,
             label="Optimized Geodesic")
    
    # Mark endpoints.
    z_start_np = z_start.detach().cpu().numpy() if hasattr(z_start, 'detach') else z_start
    z_end_np   = z_end.detach().cpu().numpy() if hasattr(z_end, 'detach') else z_end
    plt.scatter(z_start_np[0], z_start_np[1], color='green', s=120, marker='s', label='Start')
    plt.scatter(z_end_np[0], z_end_np[1], color='red', s=120, marker='s', label='End')
    
    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

############################################################
# Main Script (Geodesics mode)
############################################################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="geodesics",
                        choices=["train", "sample", "eval", "geodesics"],
                        help="What action to perform?")
    parser.add_argument("--experiment-folder",
                        type=str,
                        default="experiment",
                        help="Folder to save/load checkpoints.")
    parser.add_argument("--samples",
                        type=str,
                        default="samples.png",
                        help="File to save samples.")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Torch device.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32)
    parser.add_argument("--epochs-per-decoder",
                        type=int,
                        default=50)
    parser.add_argument("--latent-dim",
                        type=int,
                        default=2)
    parser.add_argument("--num-segments",
                        type=int,
                        default=10,
                        help="Number of segments for geodesic.")
    parser.add_argument("--steps",
                        type=int,
                        default=1000,
                        help="Optimization steps for geodesic.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Data preparation (subsample MNIST for 3 classes)
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).float()/255.
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_raw = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    test_raw  = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())

    train_data = subsample(train_raw.data, train_raw.targets, num_train_data, num_classes)
    test_data  = subsample(test_raw.data, test_raw.targets, num_train_data, num_classes)

    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Build model networks.
    M = args.latent_dim

    def new_encoder():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16x14x14
            nn.Softmax(dim=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x7x7
            nn.Softmax(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # 32x4x4
            nn.Flatten(),
            nn.Linear(512, 2*M),
        )

    def new_decoder():
        return nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32,4,4)),
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

    prior   = GaussianPrior(M)
    decoder = GaussianDecoder(new_decoder())
    encoder = GaussianEncoder(new_encoder())
    model   = VAE(prior, decoder, encoder).to(device)

    # Modes: train, sample, eval, geodesics.
    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        def noise(x, std=0.05):
            eps = std * torch.randn_like(x)
            return torch.clamp(x + eps, min=0.0, max=1.0)

        for epoch in range(args.epochs_per_decoder):
            running_loss = 0.0
            with tqdm(mnist_train_loader, desc=f"Epoch {epoch+1}/{args.epochs_per_decoder}") as pbar:
                for i, (x, _) in enumerate(pbar):
                    x = x.to(device)
                    x = noise(x)
                    optimizer.zero_grad()
                    loss = model(x)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.set_postfix({"loss": f"{running_loss/(i+1):.4f}"})
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")
    
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64,1,28,28), args.samples)
            data, _ = next(iter(mnist_test_loader))
            data = data.to(device)
            recon_mean = model.decoder(model.encoder(data).mean).mean
            out = torch.cat([data.cpu(), recon_mean.cpu()], dim=0)
            save_image(out, "recon_means.png", nrow=data.size(0))
            print("Saved samples and reconstructions.")
    
    elif args.mode == "eval":
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                elbos.append(model.elbo(x))
        mean_elbo = torch.stack(elbos).mean()
        print("Mean test ELBO =", mean_elbo.item())
    
    elif args.mode == "geodesics":
        # Load the trained model.
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        
        # Get a batch from the test set.
        data_batch, labels_batch = next(iter(mnist_test_loader))
        data_batch = data_batch.to(device)
        z_enc = model.encoder(data_batch).mean  # shape (batch_size, M)
        
        # Define the target points in latent space
        target_start = torch.tensor([0.0, 0.0], device=device)
        target_end = torch.tensor([-2.0, 6], device=device)

        # Compute Euclidean distances for each encoded point
        distances_start = torch.norm(z_enc - target_start, dim=1)
        distances_end = torch.norm(z_enc - target_end, dim=1)

        # Choose the points that minimize the distances
        z_start = z_enc[distances_start.argmin()]
        z_end = z_enc[distances_end.argmin()]

        print("Closest point to (0,0):", z_start)
        print("Closest point to (-1,5.5):", z_end)
        # Compute the optimized geodesic, energy history, and store the initial path.
        z_opt, energy_hist, z_initial = compute_geodesic(
            model,
            z_start,
            z_end,
            num_segments=args.num_segments,
            lr=1e-2,
            steps=args.steps
        )
        print("Optimized geodesic obtained.")
        
        # Decode the optimized geodesic to visualize interpolation.
        with torch.no_grad():
            imgs_curve = model.decoder(z_opt).mean  # shape (S+1, 1, 28, 28)
        save_image(imgs_curve, "geodesic_path.png", nrow=imgs_curve.size(0))
        print("Saved geodesic images as 'geodesic_path.png'")
        
        # Compute latent representations for all test data.
        all_latents = []
        all_labels = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                all_latents.append(model.encoder(x).mean)
                all_labels.append(y)
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Plot a comparison: initial (linear) vs. optimized geodesic.
        plot_geodesic_comparison(all_latents, all_labels, z_initial, z_opt, z_start, z_end,
                                 save_path="latent_space_geodesic_comparison.png")
        print("Saved latent space comparison plot as 'latent_space_geodesic_comparison.png'")
        
        # Plot the energy curve during optimization.
        plot_energy_curve(energy_hist, save_path="energy_curve.png")
        print("Saved energy curve plot as 'energy_curve.png'")