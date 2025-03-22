import torch
import torch.nn as nn
import torch.autograd.functional as Fauto

##############################################################################
# Pull-back metric geodesic using LBFGS (optimizing only interior points)
##############################################################################

def decoder_mean_jacobian(model, z):
    """
    Compute the Jacobian of the decoder's mean output with respect to z.
    z: torch.Tensor of shape (M,) representing a single latent point.
    Returns:
      J: torch.Tensor of shape (D, M), where D is the flattened dimension of f(z).
    """
    def forward_fn(z_in):
        # z_in: shape (M,). Reshape to (1, M), decode, then flatten.
        z_in_2d = z_in.unsqueeze(0)
        out = model.decoder(z_in_2d).mean  # shape: (1, C, H, W)
        return out.view(-1)                # shape: (D,)
    
    # Build a fresh graph for the Jacobian.
    z_for_jac = z.detach().clone().requires_grad_(True)
    J = Fauto.jacobian(forward_fn, z_for_jac, create_graph=True)
    return J

def compute_geodesic_pullback_lbfgs(
    model,
    z_start,
    z_end,
    num_segments=10,
    lr=1e-2,
    outer_steps=100
):
    """
    Compute the geodesic between z_start and z_end under the pull-back metric,
    using LBFGS to optimize only the interior points (z_1,...,z_{S-1}).
    
    We discretize the full path as (z_0, z_1, ..., z_S) with z_0=z_start and z_S=z_end fixed.
    The discrete energy is defined as:
      E = sum_{s=1}^{S} || J_f( (z_s+z_{s-1})/2 ) * (z_s - z_{s-1}) ||^2.
    
    Returns:
      z_opt       : Tensor of shape (S+1, M) representing the optimized full geodesic.
      energy_hist : List of energy (loss) values at each LBFGS outer iteration.
      z_initial   : The initial full path (linear interpolation) for comparison.
    """
    device = z_start.device
    latent_dim = z_start.shape[0]

    # Create full initial path by linear interpolation.
    tgrid = torch.linspace(0, 1, num_segments+1, device=device).unsqueeze(-1)  # shape (S+1, 1)
    z_init_full = z_start + tgrid * (z_end - z_start)  # shape (S+1, M)
    z_initial = z_init_full.detach().clone()

    # Only parameterize the interior points (z_1,...,z_{S-1}).
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2 for interior optimization.")
    z_interior_init = z_init_full[1:-1].detach().clone()  # shape (S-1, M)
    z_interior = nn.Parameter(z_interior_init)

    optimizer = torch.optim.LBFGS([z_interior], lr=lr, max_iter=20, history_size=10)
    energy_history = []

    def closure():
        optimizer.zero_grad()
        # Reconstruct the full path: fixed endpoints and interior points.
        z_full = torch.cat([
            z_start.unsqueeze(0),
            z_interior,
            z_end.unsqueeze(0)
        ], dim=0)  # shape (S+1, M)

        cost = 0.0
        # Compute the discrete pull-back energy along the path.
        for s in range(1, num_segments+1):
            z_prev = z_full[s-1]
            z_curr = z_full[s]
            d_z = z_curr - z_prev
            z_mid = 0.5 * (z_curr + z_prev)
            J = decoder_mean_jacobian(model, z_mid)  # shape (D, M)
            cost_segment = (J @ d_z).pow(2).sum()
            cost = cost + cost_segment

        # Instead of cost.backward(), compute gradients manually.
        grad_val = torch.autograd.grad(cost, z_interior, retain_graph=False, allow_unused=True)[0]
        if grad_val is None:
            grad_val = torch.zeros_like(z_interior)
        z_interior.grad = grad_val
        return cost

    for i in range(outer_steps):
        loss_val = optimizer.step(closure)  # LBFGS will call closure multiple times internally.
        energy_history.append(loss_val.item())
        if i % 1 == 0:
            print(f"[LBFGS] Outer step {i:3d}, energy = {loss_val.item():.4f}")

    # Reconstruct the final full path.
    with torch.no_grad():
        z_full_opt = torch.cat([
            z_start.unsqueeze(0),
            z_interior,
            z_end.unsqueeze(0)
        ], dim=0)
    return z_full_opt.detach(), energy_history, z_initial