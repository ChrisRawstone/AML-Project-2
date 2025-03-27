# src/geodesics.py
# ----------------------------------------------------------------
# Pull-back geodesic computation with debug statements and gradient fixes.
# The continuous energy:
#    E[c] = ∫₀¹ ‖J_f(c(t)) c'(t)‖² dt
# is discretized with a midpoint rule:
#    E ≈ ∑(s=1 to S) ‖J_f((z_s + z_(s-1))/2) · (z_s - z_(s-1))‖² * (1/S)
# ----------------------------------------------------------------

import torch
import torch.nn as nn
from tqdm import tqdm

def decoder_jvp(model, z, v, debug=False):
    """
    Compute the Jacobian-vector product J_f(z) @ v without forming the full Jacobian.
    
    Args:
      model: A model with a 'decoder' that outputs a distribution.
      z (torch.Tensor): Latent point at which to evaluate the Jacobian (shape: latent_dim,).
      v (torch.Tensor): The vector to multiply (shape: latent_dim,).
      debug (bool): If True, print debug information.
    
    Returns:
      jvp_result (torch.Tensor): The directional derivative, i.e. J_f(z) @ v.
    """
    if debug:
        print("[DEBUG] Entering decoder_jvp")
        print("[DEBUG]  Initial z (requires_grad=%s):" % z.requires_grad, z)
        print("[DEBUG]  v:", v)
    # Only detach if z is not already part of the computational graph.
    if not z.requires_grad:
        z = z.clone().detach().requires_grad_(True)
    def decode_fn(z_):
        out = model.decoder(z_.unsqueeze(0)).mean.view(-1)
        if debug:
            print("[DEBUG] decode_fn => out shape:", out.shape)
        return out
    # Set create_graph=True so the derivative remains connected for gradient computation.
    jvp_result = torch.autograd.functional.jvp(decode_fn, (z,), (v,), create_graph=True)[1]
    if debug:
        print("[DEBUG] jvp_result norm:", jvp_result.norm().item())
    return jvp_result

def compute_geodesic_pullback_lbfgs(
    model,
    z_start,
    z_end,
    num_segments=10,
    lr=1e-3,
    outer_steps=5,
    optimizer_type="lbfgs",
    line_search_fn="strong_wolfe",
    device="cuda",
    debug=False
):
    """
    Compute the geodesic between z_start and z_end under the pull-back metric,
    using LBFGS (or Adam) on the interior points.
    
    The energy is approximated by:
      E ≈ ∑ₛ ‖J_f((zₛ+zₛ₋₁)/2)(zₛ - zₛ₋₁)‖² * (1/num_segments)
    
    Args:
      model: A VAE-like model with a 'decoder' method.
      z_start (torch.Tensor): Starting latent point (shape: latent_dim,).
      z_end (torch.Tensor): Ending latent point (shape: latent_dim,).
      num_segments (int): Number of segments S.
      lr (float): Learning rate.
      outer_steps (int): Number of outer optimization steps.
      optimizer_type (str): "lbfgs" or "adam".
      line_search_fn (str): For LBFGS (e.g. "strong_wolfe").
      device (str): "cuda" or "cpu".
      debug (bool): If True, print detailed debug statements.
    
    Returns:
      z_full_opt (torch.Tensor): Optimized full path (shape: (S+1, latent_dim)).
      energy_history (list): Energy values from each outer iteration.
      z_initial (torch.Tensor): The initial linear path.
      final_energy (float): The final energy computed on the optimized path.
    """
    if debug:
        print("[DEBUG] compute_geodesic_pullback_lbfgs => Starting")
        print("[DEBUG]  z_start:", z_start)
        print("[DEBUG]  z_end:", z_end)
        print("[DEBUG]  lr:", lr)
        print("[DEBUG]  line_search_fn:", line_search_fn)
    
    z_start = z_start.to(device)
    z_end   = z_end.to(device)

    # Create a linear initial path: shape (num_segments+1, latent_dim)
    tgrid = torch.linspace(0, 1, num_segments + 1, device=device).unsqueeze(-1)
    z_init_full = z_start + tgrid * (z_end - z_start)
    z_initial = z_init_full.detach().clone()

    if num_segments < 2:
        raise ValueError("num_segments must be >= 2 for interior optimization.")

    # Parameterize only the interior points (they already require grad).
    z_interior = nn.Parameter(z_init_full[1:-1].clone().detach())

    # Create optimizer.
    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [z_interior],
            lr=lr,
            max_iter=200,
            history_size=20,
            line_search_fn=line_search_fn if line_search_fn != "none" else None
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam([z_interior], lr=lr)
    else:
        raise ValueError("[DEBUG] Unsupported optimizer_type: " + optimizer_type)

    energy_history = []
    delta_t = 1.0 / num_segments  # Δt for the discrete integration.

    def closure():
        optimizer.zero_grad()
        # Reconstruct the full path (concatenate fixed endpoints and interior points).
        z_full = torch.cat([z_start.unsqueeze(0), z_interior, z_end.unsqueeze(0)], dim=0)
        cost = 0.0
        for s in range(1, num_segments + 1):
            z_prev = z_full[s - 1]
            z_curr = z_full[s]
            d_z = z_curr - z_prev
            z_mid = 0.5 * (z_prev + z_curr)
            if debug:
                print(f"[DEBUG] Segment {s}:")
                print("   z_prev:", z_prev)
                print("   z_curr:", z_curr)
                print("   z_mid:", z_mid)
                print("   d_z:", d_z)
            jvp = decoder_jvp(model, z_mid, d_z, debug=debug)
            cost_segment = jvp.pow(2).sum() * delta_t
            if debug:
                print(f"[DEBUG]  Segment {s} cost: {cost_segment.item():.6f}")
            cost += cost_segment
        if debug:
            print("[DEBUG] closure => Total cost:", cost.item())
        cost.backward()
        return cost

    for i in range(outer_steps):
        if optimizer_type == "lbfgs":
            loss_val = optimizer.step(closure)
        else:
            loss_val = closure()
            optimizer.step()
        energy_val = loss_val.item()
        energy_history.append(energy_val)
        print(f"[DEBUG] Outer iteration {i+1}/{outer_steps}, Energy = {energy_val:.6f}")

    with torch.no_grad():
        z_full_opt = torch.cat([z_start.unsqueeze(0), z_interior, z_end.unsqueeze(0)], dim=0)
        final_cost = 0.0
        for s in range(1, num_segments + 1):
            z_prev = z_full_opt[s - 1]
            z_curr = z_full_opt[s]
            d_z = z_curr - z_prev
            z_mid = 0.5 * (z_prev + z_curr)
            jvp = decoder_jvp(model, z_mid, d_z, debug=False)
            final_cost += jvp.pow(2).sum().item() * delta_t
    print(f"[DEBUG] Final cost after optimization = {final_cost:.6f}")

    return z_full_opt.detach(), energy_history, z_initial, final_cost