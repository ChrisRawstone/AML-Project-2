import torch
import torch.nn as nn
import torch.autograd.functional as Fauto
from tqdm import tqdm
from time import perf_counter

##############################################################################
# Pull-back metric geodesic using LBFGS (optimizing only interior points)
##############################################################################

def decoder_jvp(model, z, v):
    """
    Compute the Jacobian-vector product J_f(z) @ v without computing full Jacobian.
    z: torch.Tensor of shape (M,)
    v: torch.Tensor of shape (M,)
    Returns:
      jvp_result: torch.Tensor of shape (D,), the directional derivative.
    """
    z = z.detach().clone().requires_grad_(True)
    return Fauto.jvp(lambda z_: model.decoder(z_.unsqueeze(0)).mean.view(-1), (z,), (v,), create_graph=True)[1]

def compute_geodesic_pullback_lbfgs(
    model,
    z_start,
    z_end,
    num_segments,
    lr,
    outer_steps,
    optimizer_type="adam"  # Added parameter for optimizer type
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
    # TODO: implement third order parametrization.

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS([z_interior], lr=lr, max_iter=20, history_size=20)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam([z_interior], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
    
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
            jvp_result = decoder_jvp(model, z_mid, d_z)
            cost_segment = jvp_result.pow(2).sum()
            cost = cost + cost_segment
            # TODO: can add compute of velocity at the same time here.

        # Instead of cost.backward(), compute gradients manually.
        grad_val = torch.autograd.grad(cost, z_interior, retain_graph=False, allow_unused=True)[0]
        if grad_val is None:
            grad_val = torch.zeros_like(z_interior)
        z_interior.grad = grad_val
        return cost

    for i in tqdm(range(outer_steps)):
        if optimizer_type == "lbfgs":
            loss_val = optimizer.step(closure)
        else:
            loss_val = closure()
            optimizer.step()
        energy_history.append(loss_val.item())
        print(f"[{optimizer_type.upper()}] Outer step {i:3d}, energy = {loss_val.item():.4f}")

    # Reconstruct the final full path.
    with torch.no_grad():
        z_full_opt = torch.cat([
            z_start.unsqueeze(0),
            z_interior,
            z_end.unsqueeze(0)
        ], dim=0)
    return z_full_opt.detach(), energy_history, z_initial

def compute_segment_speeds(model, z_path):
    """
    Given the optimized geodesic path z_path (of shape (S+1, M)),
    compute the speed for each segment.
    
    Returns:
      speeds: list of speeds for each segment.
    """
    num_segments = z_path.shape[0] - 1
    speeds = []
    for s in range(1, num_segments+1):
        z_prev = z_path[s-1]
        z_curr = z_path[s]
        d_z = z_curr - z_prev
        z_mid = 0.5 * (z_curr + z_prev)
        # Compute the speed as the norm of the directional derivative
        jvp_result = decoder_jvp(model, z_mid, d_z)
        speed = jvp_result.norm().item()
        speeds.append(speed)
    return speeds