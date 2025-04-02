# emsemble_vae.py
# do not remove this comment or the comment above

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import pdb
import seaborn as sns
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

def plot_curve_speed(model, z_curve, ensemble=False, save_path=None):
    """
    Computes and plots the speed (expected Euclidean distance between consecutive decoded outputs)
    along a latent curve.
    
    For ensemble mode:
      - For each latent point, decodes using all decoders and flattens the outputs.
      - For each segment (between consecutive latent points), computes the difference between every
        pair of decoder outputs (one from each latent point), then averages their L2 norms.
      
    For single-decoder mode, it computes speeds as before.
    
    Parameters:
        model: VAE model or EnsembleVAE.
               For ensemble, model.decoders is a list of decoders.
        z_curve (Tensor): Tensor of shape (S+1, latent_dim) representing the latent curve.
        ensemble (bool): If True, computes expected speeds over all decoder pairs.
        save_path (str, optional): If provided, saves the plot to this path.
        
    Returns:
        speeds (Tensor): A tensor of shape (S,) with the computed speed for each segment.
    """
    # Ensure z_curve is a torch.Tensor.
    if not isinstance(z_curve, torch.Tensor):
        device = next(model.parameters()).device
        z_curve = torch.from_numpy(z_curve).to(device)
    
    with torch.no_grad():
        if ensemble:
            # Precompute decoded outputs for each latent point.
            # For each z, compute outputs from all decoders; shape: (M, D)
            all_outputs = []
            for z in z_curve:
                outputs = [decoder(z.unsqueeze(0)).mean for decoder in model.decoders]
                outputs = torch.stack(outputs, dim=0).view(len(model.decoders), -1)
                all_outputs.append(outputs)
            # Stack outputs: shape (S+1, M, D)
            all_outputs = torch.stack(all_outputs, dim=0)
            
            speeds = []
            S = z_curve.shape[0] - 1  # number of segments
            for i in range(S):
                outputs_i = all_outputs[i]     # shape (M, D)
                outputs_i1 = all_outputs[i+1]    # shape (M, D)
                # Compute differences for every decoder pair: shape (M, M, D)
                diffs = outputs_i.unsqueeze(1) - outputs_i1.unsqueeze(0)
                # Compute L2 norm for each pair: shape (M, M)
                norms = torch.norm(diffs, p=2, dim=2)
                # Average over all M^2 pairs gives the expected speed for the segment.
                segment_speed = norms.mean()
                speeds.append(segment_speed)
            speeds = torch.stack(speeds)  # shape (S,)
        else:
            # Single decoder: decode and compute speeds as before.
            decoded = model.decoder(z_curve).mean
            decoded_flat = decoded.view(decoded.size(0), -1)
            diffs = decoded_flat[1:] - decoded_flat[:-1]
            speeds = torch.norm(diffs, p=2, dim=1)
    
    # Plot speeds along the curve.
    plt.figure(figsize=(8, 4))
    plt.plot(speeds.cpu().numpy(), marker='o', linestyle='-', color='blue')
    plt.xlabel('Segment Index')
    plt.ylabel('Speed (Euclidean distance)')
    plt.title('Speed Along the Geodesic Curve')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return speeds

def plot_latent_geodesics(all_latents, all_labels, geodesics, 
                          title="Latent Variables and Geodesics", 
                          save_path="latent_geodesics.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({'font.size': 14})
    
    plt.figure(figsize=(8, 6))
    
    # Plot latent codes
    scatter = plt.scatter(all_latents[:, 0], all_latents[:, 1],
                          c=all_labels, cmap='tab10', alpha=0.8, s=40)
    
    # Plot only the final optimized geodesic curves.
    for i, (initial, final) in enumerate(geodesics):
        final = final.cpu() if final.is_cuda else final
        initial = initial.cpu() if initial.is_cuda else initial
        if i == 0:
            plt.plot(final[:, 0], final[:, 1], color='red', linestyle='-', lw=2, label="Optimized Curve")
            plt.plot(initial[:, 0], initial[:, 1], color='blue', linestyle='--', lw=2, label="Initial Curve", alpha=0.5)
        else:
            plt.plot(final[:, 0], final[:, 1], color='red', linestyle='-', lw=2)
            plt.plot(initial[:, 0], initial[:, 1], color='blue', linestyle='--', lw=2, alpha=0.5)
    
    # Create legend patches for classes.
    unique_labels = sorted(torch.unique(all_labels).tolist())
    class_handles = []
    for label in unique_labels:
        color = scatter.cmap(scatter.norm(label))
        patch = mpatches.Patch(color=color, label=f"Class {label}")
        class_handles.append(patch)
    
    # Merge geodesic and class legend entries.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(class_handles)
    labels.extend([f"Class {label}" for label in unique_labels])
    
    plt.legend(handles=handles, labels=labels, frameon=True, fancybox=True, shadow=True, framealpha=0.7)
    plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_curve_reconstructions(model, z_curve, ensemble, title="Reconstruction", save_path=None):
    """
    Decode a series of latent codes along a curve and plot the reconstructed images in a row.
    
    For ensemble mode, each latent point is decoded using all decoders and the outputs
    are averaged to yield the model-average reconstruction.
    
    Parameters:
        model: VAE model (or EnsembleVAE) where:
               - For single decoder, model.decoder returns a distribution with a 'mean'.
               - For ensemble, model.decoders is a list of decoders.
        z_curve (Tensor): Latent curve of shape (S+1, latent_dim).
        ensemble (bool): If True, use ensemble averaging over decoders.
        title (str): Title of the plot.
        save_path (str, optional): If provided, save the plot to this path.
    """
    model.eval()
    with torch.no_grad():
        if ensemble:
            decoded_points = []
            # For each latent point, decode using all decoders and average the outputs.
            for z in z_curve:
                outputs = [decoder(z.unsqueeze(0)).mean for decoder in model.decoders]
                avg_output = torch.stack(outputs, dim=0).mean(dim=0)  # shape: (1, channels, height, width)
                decoded_points.append(avg_output.squeeze(0))
            decoded = torch.stack(decoded_points)  # shape: (S+1, channels, height, width)
        else:
            decoded = model.decoder(z_curve).mean  # shape: (S+1, channels, height, width)
    
    decoded = decoded.cpu()
    num_points = decoded.shape[0]
    fig, axes = plt.subplots(1, num_points, figsize=(num_points * 2, 2))
    if num_points == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        img = decoded[i].squeeze()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{i}")
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def compute_curve_length_mc(model, z_curve, n_samples=10, ensemble=False):
    """
    Compute the expected curve length via Monte Carlo sampling.
    
    When ensemble is True, for each segment of the latent curve, n_samples random decoder pairs are sampled.
    For each sample, the endpoints are decoded (using rsample for stochastic decoders), the L2 distance is computed,
    and the distances are averaged over samples.
    
    When ensemble is False, the decoding is performed using a single decoder (i.e. model.decoder).
    
    Parameters:
        model: VAE or EnsembleVAE where:
               - For ensemble: model.decoders is a list of decoders.
               - For single decoder: model.decoder is used.
        z_curve (Tensor): Latent curve of shape (S+1, latent_dim).
        n_samples (int): Number of Monte Carlo samples per segment.
        ensemble (bool): Whether to use an ensemble MC sampling method.
        
    Returns:
        total_length (float): The expected sum of Euclidean distances between consecutive decoded outputs.
    """
    S = z_curve.shape[0] - 1  # number of segments
    
    if ensemble:
        M = len(model.decoders)    # number of decoders in the ensemble
        total_length = 0.0

        for i in range(S):
            samples = []
            for _ in range(n_samples):
                # For each sample, randomly choose a pair of decoders.
                l_idx = torch.randint(0, M, (1,), device=z_curve.device).item()
                k_idx = torch.randint(0, M, (1,), device=z_curve.device).item()
                # Decode the endpoints of the segment using the chosen decoders.
                f_l = model.decoders[l_idx](z_curve[i].unsqueeze(0)).rsample().view(-1)
                f_k = model.decoders[k_idx](z_curve[i+1].unsqueeze(0)).rsample().view(-1)
                length = torch.norm(f_l - f_k, p=2)
                samples.append(length)
            # Average over the MC samples for this segment.
            segment_length = torch.stack(samples).mean()
            total_length += segment_length
        return total_length.item()
    
    else:
        # Single-decoder case.
        decoded = model.decoder(z_curve).mean  # shape: (S+1, channels, height, width)
        decoded_flat = decoded.view(decoded.size(0), -1)
        diffs = decoded_flat[1:] - decoded_flat[:-1]
        segment_lengths = torch.norm(diffs, p=2, dim=1)
        total_length = segment_lengths.sum().item()
        return total_length

def compute_curve_length(model, z_curve, ensemble=False):
    """
    Computes the approximate length of a curve in observation space.
    
    Parameters:
        model: VAE model (or EnsembleVAE) where:
               - For a single decoder, model.decoder returns a distribution with a 'mean'.
               - For ensembles, model.decoders is a list of decoders.
        z_curve (Tensor): A tensor of shape (S+1, latent_dim) representing the curve in latent space.
        ensemble (bool): If True, compute the expected curve length over all decoder pairs.
        
    Returns:
        total_length (float): The expected sum of Euclidean distances between consecutive decoded outputs.
    """
    import torch

    with torch.no_grad():
        if ensemble:
            # Precompute decoded outputs for each latent point.
            # For each latent point, obtain outputs from each decoder, and flatten them.
            all_outputs = []
            for z in z_curve:
                outputs = [decoder(z.unsqueeze(0)).mean for decoder in model.decoders]
                # Each output: shape (1, channels, height, width); flatten to (1, D)
                outputs = torch.stack(outputs, dim=0).view(len(model.decoders), -1)  # shape: (M, D)
                all_outputs.append(outputs)
            # Stack all decoded outputs: shape (S+1, M, D)
            all_outputs = torch.stack(all_outputs, dim=0)
            
            total_length = 0.0
            S = z_curve.shape[0] - 1  # number of segments
            # For each segment, compute the L2 distance for every decoder pair and average.
            for i in range(S):
                outputs_i = all_outputs[i]   # shape: (M, D)
                outputs_i1 = all_outputs[i+1]  # shape: (M, D)
                # Expand dims to broadcast: shape becomes (M, M, D)
                diffs = outputs_i.unsqueeze(1) - outputs_i1.unsqueeze(0)
                # Compute L2 norm for each pair: shape (M, M)
                norms = torch.norm(diffs, p=2, dim=2)
                # Average over all decoder pairs for this segment.
                segment_length = norms.mean()
                total_length += segment_length
        else:
            # Single-decoder case: decode using model.decoder
            decoded = model.decoder(z_curve).mean  # shape: (S+1, channels, height, width)
            decoded_flat = decoded.view(decoded.size(0), -1)
            diffs = decoded_flat[1:] - decoded_flat[:-1]
            segment_lengths = torch.norm(diffs, p=2, dim=1)
            total_length = segment_lengths.sum().item()
    
    return total_length

def compute_energy(model, z_curve):
    """
    Compute the energy of a curve in latent space using equation 8.7 in the DGGM book.    
    Parameters:
        model: VAE model 
        z_curve (Tensor): A tensor of shape (S+1, 2) representing the curve in latent space.
        
    Returns:
        energy (Tensor): The computed energy (scalar) that supports gradients.
    """
    dt = 1.0 / (z_curve.shape[0] - 1)
    decoded = model.decoder(z_curve).mean  
    diff = decoded[1:] - decoded[:-1]
    energy = (diff ** 2).view(diff.size(0), -1).sum() / dt
    return energy

def compute_model_average_energy(model, z_curve):

    """
    Compute the model-average curve energy by enumerating *all* decoder pairs,
    rather than sampling. This is feasible for a small number of decoders (e.g., M <= 3).
    
    Parameters:
        model: EnsembleVAE with an ensemble of decoders.
        z_curve (Tensor): Latent curve of shape (S+1, latent_dim).
        
    Returns:
        total_energy (Tensor): Scalar energy of the curve.
    """
    S = z_curve.shape[0] - 1  # Number of segments
    M = len(model.decoders)   # Number of decoders
    
    # Precompute decoded outputs for all latent points:
    # all_outputs[i] will be shape (M, D), 
    # with D = flattened dimensionality of decoder outputs.
    all_outputs = []
    for z in z_curve:
        outputs = [decoder(z.unsqueeze(0)).mean for decoder in model.decoders]
        outputs = torch.stack(outputs, dim=0).view(M, -1)  # shape: (M, D)
        all_outputs.append(outputs)
    all_outputs = torch.stack(all_outputs, dim=0)  # shape: (S+1, M, D)
    #pdb.set_trace()
    
    total_energy = 0.0
    
    for i in range(S):
        # outputs_i: shape (M, D)
        # outputs_i1: shape (M, D)
        outputs_i = all_outputs[i]
        outputs_i1 = all_outputs[i + 1]
        #pdb.set_trace()
        
        # Expand dimensions to broadcast over all pairs (l, k).
        # diffs will have shape (M, M, D).
        diffs = outputs_i.unsqueeze(1) - outputs_i1.unsqueeze(0)
        #pdb.set_trace()
        
        # Sum of squared differences along the D dimension => shape (M, M).
        squared_diffs = diffs.pow(2).sum(dim=2)
        #pdb.set_trace()
        
        # Average over all M^2 pairs => scalar.
        segment_energy = squared_diffs.mean() 
        #pdb.set_trace()

        total_energy += segment_energy

    
    dt = 1.0 / (S)   
    total_energy /= dt 
    return total_energy

def compute_model_average_energy_dec_mc(model, z_curve, n_samples=10):
    """
    Compute the model-average curve energy via Monte Carlo sampling.
    For each curve segment, sample n_samples random decoder pairs.
    For each MC sample, a pair of decoders is randomly chosen (independently)
    and used to decode the two endpoint latent variables for the segment.
    The squared norm of the difference is computed for each sample and then averaged.

    Parameters:
        model: EnsembleVAE with an ensemble of decoders.
        z_curve (Tensor): Latent curve of shape (S+1, latent_dim).
        n_samples (int): Number of MC samples (random decoder pair choices) per segment.
        
    Returns:
        total_energy (Tensor): Scalar energy of the curve.
    """
    S = z_curve.shape[0] - 1  # number of segments
    M = len(model.decoders)   # number of decoders

    total_energy = 0.0

    for i in range(S):
        samples_diff = []
        for _ in range(n_samples):
            # For each sample, randomly choose a pair of decoders independently.
            l_idx = torch.randint(0, M, (1,), device=z_curve.device).item()
            k_idx = torch.randint(0, M, (1,), device=z_curve.device).item()
            # Decode the endpoints of the segment using the chosen decoders.
            f_l = model.decoders[l_idx](z_curve[i].unsqueeze(0)).rsample().view(-1)
            f_k = model.decoders[k_idx](z_curve[i+1].unsqueeze(0)).rsample().view(-1)
            diff = f_l - f_k
            samples_diff.append(diff.pow(2).sum())
        
        # Average the squared differences for this segment.
        segment_energy = torch.stack(samples_diff).mean()
        total_energy += segment_energy

    dt = 1.0 / S
    total_energy /= dt
    return total_energy

def compute_model_average_energy_mc(model, z_curve, n_samples=10):
    """
    Compute the model-average curve energy via Monte Carlo sampling.
    For each curve segment, a random pair of decoders is chosen.
    For each latent point of the segment, n_samples are drawn (via rsample)
    from the corresponding decoder, and the mean squared norm of the differences
    is computed.

    Parameters:
        model: EnsembleVAE with an ensemble of decoders.
        z_curve (Tensor): Latent curve of shape (S+1, latent_dim).
        n_samples (int): Number of MC samples per segment.
        
    Returns:
        total_energy (Tensor): Scalar energy of the curve.
    """
    S = z_curve.shape[0] - 1      # number of segments
    M = len(model.decoders)       # number of decoders

    total_energy = 0.0

    for i in range(S):
        # Randomly select one decoder for the starting point and one for the ending point.
        l_idx = torch.randint(0, M, (1,), device=z_curve.device).item()
        k_idx = torch.randint(0, M, (1,), device=z_curve.device).item()

        samples_diff = []
        for _ in range(n_samples):
            # Sample from the chosen decoders, each sample is a new Monte Carlo draw.
            f_l = model.decoders[l_idx](z_curve[i].unsqueeze(0)).rsample().view(-1)
            f_k = model.decoders[k_idx](z_curve[i+1].unsqueeze(0)).rsample().view(-1)
            diff = f_l - f_k
            samples_diff.append(diff.pow(2).sum())
        
        # Average the squared differences over the MC samples.
        segment_energy = torch.stack(samples_diff).mean()
        total_energy += segment_energy

    dt = 1.0 / S
    total_energy /= dt
    return total_energy

def compute_geodesic_adam(
    model,           # VAE model with .decoder(...) -> distribution
    z_start,         # Tensor of shape (latent_dim,)  -- endpoint A
    z_end,           # Tensor of shape (latent_dim,)  -- endpoint B
    num_segments=20, # S: total segments so there are S+1 points
    lr=0.5,
    max_iter=200,   # number of Adam iterations
    ensemble=False   # Use ensemble energy
):
    """
    Optimize the geodesic connecting z_start to z_end in the data space of the decoder
    using the Adam optimizer with a learning rate decay schedule. Only the interior latent points are optimized.

    Returns:
        (initial_curve, final_curve): a tuple each of shape (S+1, latent_dim)
    """
    # Compute initial full curve via linear interpolation.
    t_full = torch.linspace(0, 1, num_segments+1, device=z_start.device)
    initial_curve = z_start.unsqueeze(0) * (1 - t_full).unsqueeze(1) + z_end.unsqueeze(0) * t_full.unsqueeze(1)
    
    # If no interior points exist, return the endpoints.
    if num_segments < 1:
        return initial_curve, initial_curve

    # Only the interior points (exclude endpoints) will be optimized.
    t_interior = t_full[1:-1]
    z_interior = z_start.unsqueeze(0) * (1 - t_interior).unsqueeze(1) + z_end.unsqueeze(0) * t_interior.unsqueeze(1)
    z_interior = torch.nn.Parameter(z_interior)

    optimizer = torch.optim.Adam([z_interior], lr=lr)
    # Example: Exponential decay scheduler: every iteration, lr = lr * gamma.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    for itr in range(1, max_iter+1):
        optimizer.zero_grad()
        # Reconstruct full curve with fixed endpoints.
        z_vars = torch.cat([z_start.unsqueeze(0), z_interior, z_end.unsqueeze(0)], dim=0)
        if ensemble:
            energy = compute_model_average_energy_dec_mc(model, z_vars)
        else:
            energy = compute_energy(model, z_vars)
        energy.backward()
        optimizer.step()
        scheduler.step(energy.item())  # Update the learning rate
        
        if itr % 5 == 0:
            grad_norm = z_interior.grad.norm().item() if z_interior.grad is not None else 0.0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iteration {itr}: energy = {energy.item():.4f}, gradient norm = {grad_norm:.6f}, lr = {current_lr:.6f}")
            
    final_curve = torch.cat([z_start.unsqueeze(0), z_interior.detach(), z_end.unsqueeze(0)], dim=0)
    return initial_curve, final_curve

def compute_geodesic(
    model,           # VAE model with .decoder(...) -> distribution
    z_start,         # Tensor of shape (latent_dim,)  -- endpoint A
    z_end,           # Tensor of shape (latent_dim,)  -- endpoint B
    num_segments=20, # S: total segments so there are S+1 points
    lr=0.001,
    max_iter=1000,   # total LBFGS iterations
    ensemble=False   # Use ensemble energy
):
    """
    Optimize the geodesic connecting z_start to z_end in the data space of the decoder.
    This version optimizes only the interior points.
    
    Returns a tuple (initial_curve, final_curve) each of shape (S+1, latent_dim).
    """
    # Compute initial full curve via linear interpolation.
    t_full = torch.linspace(0, 1, num_segments+1).to(z_start.device)
    initial_curve = z_start.unsqueeze(0) * (1 - t_full).unsqueeze(1) + z_end.unsqueeze(0) * t_full.unsqueeze(1)
    
    # If no interior points exist, return the endpoints.
    if num_segments < 1:
        return initial_curve, initial_curve

    # Only the interior points (exclude endpoints) will be optimized.
    t_interior = t_full[1:-1]
    z_interior = (z_start.unsqueeze(0) * (1 - t_interior).unsqueeze(1) +
                  z_end.unsqueeze(0) * t_interior.unsqueeze(1))
    z_interior = torch.nn.Parameter(z_interior)

    optimizer = torch.optim.LBFGS(
        [z_interior],
        lr=lr,
        max_iter=max_iter,
        history_size=20,
        line_search_fn="strong_wolfe"
    )
    counter = [0]

    # Use a dictionary to store the decoder_choices persistently across closure calls.
    cache = {"decoder_choices": None}

    def closure():
        optimizer.zero_grad()
        counter[0] += 1
        # Reconstruct full curve with fixed endpoints.
        z_vars = torch.cat([z_start.unsqueeze(0), z_interior, z_end.unsqueeze(0)], dim=0)
        
        if ensemble:
            energy = compute_model_average_energy_mc(model, z_vars)
        else:
            energy = compute_energy(model, z_vars)

        #if counter[0] % 10 == 0:
        print(f"Iteration {counter[0]}: energy = {energy.item():.4f}")
        energy.backward()

       # Print gradient norm for z_interior.
        if z_interior.grad is not None:
            print(f"Gradient norm: {z_interior.grad.norm().item():.6f}")
        
        
        return energy

    # for i in range(10):
    #     print(f"Outer iteration {i+1}")
    optimizer.step(closure)

    # Reconstruct the final curve.
    final_curve = torch.cat([z_start.unsqueeze(0), z_interior.detach(), z_end.unsqueeze(0)], dim=0)
    return initial_curve, final_curve

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

class EnsembleVAE(nn.Module):
    """
    Define a VAE with an ensemble of decoders.
    
    """
    def __init__(self, prior, decoders, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoders: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
         
        """
        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)
        self.encoder = encoder
    
    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data by randomly selecting one decoder.

        """

        q = self.encoder(x)
        z = q.rsample()

        # Randomly select one decoder for this batch
        decoder = random.choice(self.decoders)
        log_prob = decoder(z).log_prob(x)

        elbo = torch.mean(log_prob - q.log_prob(z) + self.prior().log_prob(z))
        return elbo

    def sample(self, n_samples=1, average=True):
        """
        Sample from the model.
        
        Parameters:
            n_samples (int): Number of samples to generate.
            average (bool): If True, returns the ensemble average sample;
                            otherwise returns samples from each decoder.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        samples = [decoder(z).sample() for decoder in self.decoders]
        samples = torch.stack(samples)  # Shape: (num_decoders, n_samples, ...)
        if average:
            return torch.mean(samples, dim=0)  # Ensemble average
        return samples

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.
        """
        return -self.elbo(x)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
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
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
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

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

def plot_training_loss(loss_history, title="Training Loss", xlabel="Iteration", ylabel="Loss", save_path=None):
    """
    Plot the training loss.

    Parameters:
        loss_history (list): List of loss values recorded during training.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str): If provided, saves the plot to this path.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

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
    # Decoders are sampled randomly during training
    # so we need to account for the number of decoders in the total number of steps.
    num_decoders = len(model.decoders) if hasattr(model, "decoders") else 1
    num_steps = len(data_loader) * epochs * num_decoders
    epoch = 0
    loss_history = []

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
                loss = model(x)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                loss_history.append(loss_val)

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
    return loss_history


def new_encoder(M):
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

def new_decoder(M):
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

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
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
        default=200,
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
        default=1,
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
        default=25,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: random)",
    )
    parser.add_argument(
        "--num_train_data",
        type=int,
        default=2048,
        help="random seed (default: random)",
    )



    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)
    
    # After parsing
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = args.num_train_data
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



    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        # Construct ensemble of decoders
        if args.num_decoders > 1:
            decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
            model = EnsembleVAE(
                GaussianPrior(M),
                decoders,
                GaussianEncoder(new_encoder()),
            ).to(device)
        # Single decoder
        else:
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_history = train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        plot_training_loss(loss_history, save_path=experiments_folder+"/training_loss.png")
        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
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
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
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
        ensemble = False
        if args.num_decoders > 1:
            ensemble = True
            decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
            model = EnsembleVAE(
                GaussianPrior(M),
                decoders,
                GaussianEncoder(new_encoder()),
            ).to(device)
        else:
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # Encode test data to get latent representations
        all_latents = []
        all_labels = []
        for x, y in mnist_test_loader:
            x = x.to(device)
            with torch.no_grad():
                q = model.encoder(x)
                # Use mean of q as latent representation
                latent = q.mean
            all_latents.append(latent)
            all_labels.append(y)
        all_labels = torch.cat(all_labels, dim=0).cpu()
        all_latents = torch.cat(all_latents, dim=0).cpu()

        # Choose random pairs from encoded latent codes
        num_pairs = args.num_curves
        # Ensure reproducibility
        indices = torch.randperm(all_latents.shape[0])[:2*num_pairs].reshape(num_pairs, 2)
        geodesics = []
        latent_pairs = []

        
        # For each chosen latent pair:
        for pair in tqdm(indices):
            z0, z1 = all_latents[pair[0]].to(device), all_latents[pair[1]].to(device)
            latent_pairs.append((z0.cpu().numpy(), z1.cpu().numpy()))
            if ensemble:
                initial_curve, final_curve = compute_geodesic_adam(
                    model, z0, z1, num_segments=args.num_t, ensemble=True
                )
            else:
                initial_curve, final_curve = compute_geodesic(model, z0, z1, num_segments=args.num_t)
            # Stack both curves into a tuple for later plotting.
            geodesics.append((initial_curve, final_curve))

        initial_lengths = [
            compute_curve_length_mc(model, curve.to(device), ensemble=ensemble) for curve, _ in geodesics
        ]
        final_lengths = [
            compute_curve_length_mc(model, curve.to(device), ensemble=ensemble) for _, curve in geodesics
        ]

        print("Initial geodesic lengths:", initial_lengths)
        print("Optimized geodesic lengths:", final_lengths)

        # Plot curve speeds for final curves
        for i, (initial_curve, final_curve) in enumerate(geodesics):
            plot_curve_speed(model, initial_curve, ensemble=ensemble, save_path=args.experiment_folder + f"/initial_curve_{i}_speeds.png")
            plot_curve_speed(model, final_curve, ensemble=ensemble, save_path=args.experiment_folder + f"/final_curve_{i}_speeds.png")

        # Plot the latent variables and the geodesics.
        plot_latent_geodesics(all_latents, all_labels, geodesics, save_path=args.experiment_folder + "/latent_geodesics.png")

        # Plot reconstructions from the linear (initial) interpolation.
        plot_curve_reconstructions(model, initial_curve, ensemble=ensemble, title="Linear Interpolation Reconstructions", save_path=args.experiment_folder + "/linear_interpolation_reconstructions.png")

        # Plot reconstructions from the optimized geodesic.
        plot_curve_reconstructions(model, final_curve, ensemble=ensemble, title="Optimized Geodesic Reconstructions", save_path=args.experiment_folder + "/optimized_geodesic_reconstructions.png")


