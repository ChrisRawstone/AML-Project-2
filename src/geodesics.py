import torch
import torch.nn as nn
import torch.autograd.functional as Fauto
from tqdm import tqdm

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
    optimizer_type="lbfgs"  # Added parameter for optimizer type
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
            # Δt = 1/num_segments; multiplying by num_segments scales the cost properly.
            cost_segment = jvp_result.pow(2).sum() * num_segments
            cost = cost + cost_segment

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

# New helper function to compute pullback metric using full Jacobian
def decoder_pullback_metric(model, z):
    """
    Compute the pullback metric J^T J for a single latent z of shape (M,).
    Returns a (M, M) matrix.
    """
    z_ = z.detach().clone().requires_grad_(True)
    # We'll flatten the decoded output
    decoded = model.decoder(z_.unsqueeze(0)).mean.view(-1)
    # Compute full Jacobian w.r.t. z
    J = torch.autograd.functional.jacobian(lambda z_in: model.decoder(z_in.unsqueeze(0)).mean.view(-1), z_)
    # shape of J is (D, M); we want (M, M) by J.T @ J
    G = J.t() @ J
    return G.detach()

def metric_norm(G, x):
    """
    Compute x^T G x for a vector x in R^M.
    """
    return (x.unsqueeze(0) @ G @ x.unsqueeze(-1)).squeeze()

# New class PiecewiseLinearCurve
class PiecewiseLinearCurve(nn.Module):
    def __init__(self, z_start, z_end, n_intervals=10, device='cpu'):
        super().__init__()
        # We'll store the endpoints as fixed buffers
        self.register_buffer('z0', z_start.detach().clone())
        self.register_buffer('z1', z_end.detach().clone())
        self.n_intervals = n_intervals

        # Create interior points
        tgrid = torch.linspace(0, 1, n_intervals+1, device=device)
        # initialize as a small random perturbation around linear interp
        self.c_free = nn.Parameter(
            self.z0 + (self.z1 - self.z0) * tgrid[1:-1].unsqueeze(-1) +
            0.1 * torch.randn_like(self.z0).unsqueeze(0) * torch.norm(self.z1 - self.z0)
        )

    def c(self):
        # Return the full sequence of points [z0, c_free..., z1], shape: (n_intervals+1, M)
        return torch.cat([
            self.z0.unsqueeze(0),
            self.c_free,
            self.z1.unsqueeze(0)
        ], dim=0)

    def c_dot(self):
        # discrete difference
        c_full = self.c()
        diffs = c_full[1:] - c_full[:-1]
        return diffs  # shape: (n_intervals, M)

    def forward(self, model):
        """
        Compute the discrete energy based on equation 8.7:
        E = sum_{i=1}^{n_intervals} || f(c_i) - f(c_{i-1}) ||^2,
        where f is the decoder of the VAE.
        """
        c_full = self.c()  # shape: (n_intervals+1, latent_dim)
        norm2 = 0.0
        for i in range(1, self.n_intervals + 1):
            # Decode consecutive points along the curve
            f_c = model.decoder(c_full[i].unsqueeze(0)).mean
            f_prev = model.decoder(c_full[i-1].unsqueeze(0)).mean
            norm2 += torch.sum((f_c - f_prev) ** 2)
        return norm2

# New function compute_geodesic_piecewise
def compute_geodesic_piecewise(
    model,
    z_start,
    z_end,
    num_segments=10,
    lr=1e-3,
    outer_steps=50,
    optimizer_type="lbfgs",
    device="cpu"
):
    """
    Similar to compute_geodesic_pullback_lbfgs, but uses a piecewise linear approach
    with a full Jacobian for the pull-back metric. Returns the path plus energy history.
    """
    # Build curve module
    curve = PiecewiseLinearCurve(z_start, z_end, n_intervals=num_segments, device=device).to(device)

    # Choose optimizer
    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(curve.parameters(), lr=lr, max_iter=20, history_size=20)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(curve.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    energy_history = []

    def closure():
        optimizer.zero_grad()
        cost = curve(model)
        cost.backward()
        return cost

    for i in range(outer_steps):
        if optimizer_type == "lbfgs":
            loss_val = optimizer.step(closure)
        else:
            loss_val = closure()
            optimizer.step()
        energy_history.append(loss_val.item())
        print(f"[{optimizer_type.upper()}] Step {i}, energy = {loss_val.item():.4f}")

    # Return the optimized path
    with torch.no_grad():
        final_curve = curve.c().detach()
    return final_curve, energy_history