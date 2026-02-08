import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
import random
import numpy as np
from torch.autograd import grad
from PINN.networks import Net
import sys
import torch.nn.functional as F

mu = 4 * np.pi * 1e-7 # vacuum permeability
epsilon = 8.854e-12 # vacuum permittivity
I_0 = 1.0 # current amplitude
r_0 = 0.05 # solenoid radius
c2 = 1.0 / (mu * epsilon)  # speed of light squared (vacuum)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def _smooth_step(x: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Smooth approximation of Heaviside(x)."""
    return torch.sigmoid(sharpness * x)

def J_theta(r: torch.Tensor, z: torch.Tensor, t: torch.Tensor, args) -> torch.Tensor:
    """Approximate azimuthal current density for two finite solenoids.

    Modeled as a smooth current sheet around r=args.r_coil, active over two z-intervals:
    [args.z1_left, args.z1_right] and [args.z2_left, args.z2_right].
    """
    # radial "sheet" around the coil radius
    Jr = torch.exp(-((r - args.r_coil) / args.sigma_r) ** 2)

    # smooth boxcar in z for each solenoid segment
    H = _smooth_step
    z1 = H(z - args.z1_left, args.z_sharpness) * H(args.z1_right - z, args.z_sharpness)
    z2 = H(z - args.z2_left, args.z_sharpness) * H(args.z2_right - z, args.z_sharpness)
    Jz = z1 + z2

    # Time-harmonic drive: I(t) = I0 * sin(2π f t), with f=50Hz by default.
    time_factor = torch.sin(2.0 * torch.pi * args.frequency_hz * t)

    return args.current * time_factor * Jr * Jz

def PDE(u: torch.Tensor, v: torch.Tensor, r: torch.Tensor, z: torch.Tensor, t: torch.Tensor, args):
    """Coupled PDE residuals using axis-regular parameterization.

    Re-parameterize to remove 1/r and 1/r^2 singularities at the axis:
      A_theta(r,z,t) = r * u(r,z,t)
      A_r(r,z,t)     = r * v(r,z,t)

    Derived residuals (from your provided component equations):

      res_r =
        (3 v_r + r (v_rr + v_zz + (1/c^2) v_tt))
        - 4 u_z - 2 r u_rz

      res_theta =
        (3 u_r + r (u_rr + u_zz + (1/c^2) u_tt + 2 v_zz))
        - μ0 J_theta(r,z,t)

    Returns:
      (res_r, res_theta) tensors shaped like u/v.
    """
    # derivatives of u
    u_r = d(u, r)
    u_rr = d(u_r, r)
    u_z = d(u, z)
    u_zz = d(d(u, z), z)
    u_tt = d(d(u, t), t)
    u_rz = d(d(u, r), z)

    # derivatives of v
    v_r = d(v, r)
    v_rr = d(v_r, r)
    v_zz = d(d(v, z), z)
    v_tt = d(d(v, t), t)

    res_r = (3.0 * v_r + r * (v_rr + v_zz + (1.0 / c2) * v_tt)) - 4.0 * u_z - 2.0 * r * u_rz
    res_theta = (3.0 * u_r + r * (u_rr + u_zz + (1.0 / c2) * u_tt + 2.0 * v_zz)) - mu * J_theta(
        r, z, t, args
    )

    return res_r, res_theta

def boundary_derivative(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper for ∂A/∂x."""
    return d(A, x)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        r_max = args.r_max
        # r is sampled in [0, r_max], with extra density near r=0 (axis)
        r = (r_max * torch.rand((args.n_f, 1), dtype=torch.float) ** 2).requires_grad_(True)
        # z is sampled from -z_max to z_max
        z = (args.z_max * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        # t is sampled over one drive period (seconds)
        t = (args.t_max * torch.rand((args.n_f, 1), dtype=torch.float)).requires_grad_(True)

        # Scale network output so that A_theta is in a realistic magnitude range.
        # In air, ν=1/μ is huge, so the physical solution typically has very small A.
        out = PINN(torch.cat([r, z, t], dim=1))
        # PINN outputs u,v; physical potentials are A_theta=r*u and A_r=r*v
        u = args.u_scale * out[:, 0:1]
        v = args.v_scale * out[:, 1:2]
        A_r = r * v
        A_theta = r * u
        res_r, res_theta = PDE(u, v, r, z, t, args)
        mse_PDE = args.criterion(res_r, torch.zeros_like(res_r)) + args.criterion(res_theta, torch.zeros_like(res_theta))

        # boundary
        # r is outer boundary
        r_bc = (args.r_max * torch.ones((args.n_b_l, 1), dtype=torch.float)).requires_grad_(True)
        # z is sampled away from the center gap region (avoid over-constraining near z≈0)
        n_half = args.n_b_l // 2
        z_exclude = args.z_gap_exclude
        z_bc_low = -args.z_max + ((args.z_max - z_exclude) * torch.rand((n_half, 1), dtype=torch.float))
        z_bc_high = z_exclude + ((args.z_max - z_exclude) * torch.rand((n_half, 1), dtype=torch.float))
        z_bc = torch.cat([z_bc_low, z_bc_high], dim=0).requires_grad_(True)
        t_bc = (args.t_max * torch.rand((args.n_b_l, 1), dtype=torch.float)).requires_grad_(True)
        out_bc = PINN(torch.cat([r_bc, z_bc, t_bc], dim=1))
        u_bc = args.u_scale * out_bc[:, 0:1]
        v_bc = args.v_scale * out_bc[:, 1:2]
        A_r_bc = r_bc * v_bc
        A_theta_bc = r_bc * u_bc
        dA_r_dr_bc = boundary_derivative(A_r_bc, r_bc)
        dA_theta_dr_bc = boundary_derivative(A_theta_bc, r_bc)
        mse_BC = args.criterion(dA_r_dr_bc, torch.zeros_like(dA_r_dr_bc)) + args.criterion(dA_theta_dr_bc, torch.zeros_like(dA_theta_dr_bc))

        # z-boundary (Neumann): ∂/∂z(...) = 0 at z = ±z_max
        n_half_z = args.n_b_z // 2
        r_zbc = (r_max * torch.rand((args.n_b_z, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_zbc = torch.cat(
            [
                (-args.z_max * torch.ones((n_half_z, 1), dtype=torch.float)),
                (args.z_max * torch.ones((args.n_b_z - n_half_z, 1), dtype=torch.float)),
            ],
            dim=0,
        ).requires_grad_(True)
        t_zbc = (args.t_max * torch.rand((args.n_b_z, 1), dtype=torch.float)).requires_grad_(True)
        out_zbc = PINN(torch.cat([r_zbc, z_zbc, t_zbc], dim=1))
        u_zbc = args.u_scale * out_zbc[:, 0:1]
        v_zbc = args.v_scale * out_zbc[:, 1:2]
        A_r_zbc = r_zbc * v_zbc
        A_theta_zbc = r_zbc * u_zbc
        dA_r_dz_zbc = boundary_derivative(A_r_zbc, z_zbc)
        dA_theta_dz_zbc = boundary_derivative(A_theta_zbc, z_zbc)
        mse_ZBC = args.criterion(dA_r_dz_zbc, torch.zeros_like(dA_r_dz_zbc)) + args.criterion(dA_theta_dz_zbc, torch.zeros_like(dA_theta_dz_zbc))

        # symmetry about z=0 (even): A_theta(r,z,t) == A_theta(r,-z,t)
        r_sym = r_max * torch.rand((args.n_sym, 1), dtype=torch.float) ** 2
        z_sym = args.z_max * torch.rand((args.n_sym, 1), dtype=torch.float)
        t_sym = args.t_max * torch.rand((args.n_sym, 1), dtype=torch.float)
        out_sym = PINN(torch.cat([r_sym, z_sym, t_sym], dim=1))
        u_sym = args.u_scale * out_sym[:, 0:1]
        v_sym = args.v_scale * out_sym[:, 1:2]
        A_r_sym = r_sym * v_sym
        A_theta_sym = r_sym * u_sym
        mse_SYM = args.criterion(A_r_sym, torch.zeros_like(A_r_sym)) + args.criterion(A_theta_sym, torch.zeros_like(A_theta_sym))

        # symmetry on-axis (targets Bz(0,z)=2A_theta(0,z)): A_theta(0,z,t) == A_theta(0,-z,t)
        r_sym0 = torch.zeros((args.n_sym_axis, 1), dtype=torch.float)
        z_sym0 = args.z_max * torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        t_sym0 = args.t_max * torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        out_sym0 = PINN(torch.cat([r_sym0, z_sym0, t_sym0], dim=1))
        u_sym0 = args.u_scale * out_sym0[:, 0:1]
        v_sym0 = args.v_scale * out_sym0[:, 1:2]
        A_r_sym0 = r_sym0 * v_sym0
        A_theta_sym0 = r_sym0 * u_sym0
        mse_SYM0 = args.criterion(A_r_sym0, torch.zeros_like(A_r_sym0)) + args.criterion(A_theta_sym0, torch.zeros_like(A_theta_sym0))

        # mid-plane Neumann symmetry: ∂A_theta/∂z = 0 at z=0
        r_mid = (r_max * torch.rand((args.n_mid, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_mid = torch.zeros((args.n_mid, 1), dtype=torch.float).requires_grad_(True)
        t_mid = (args.t_max * torch.rand((args.n_mid, 1), dtype=torch.float)).requires_grad_(True)
        out_mid = PINN(torch.cat([r_mid, z_mid, t_mid], dim=1))
        u_mid = args.u_scale * out_mid[:, 0:1]
        v_mid = args.v_scale * out_mid[:, 1:2]
        A_r_mid = r_mid * v_mid
        A_theta_mid = r_mid * u_mid
        dA_theta_dz_mid = boundary_derivative(A_theta_mid, z_mid)
        mse_MID = args.criterion(dA_theta_dz_mid, torch.zeros_like(dA_theta_dz_mid)) + args.criterion(A_r_mid, torch.zeros_like(A_r_mid))

        # initial condition: A_theta(r,z,t=0) = 0
        r_ic = (r_max * torch.rand((args.n_f, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_ic = (args.z_max * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        t_ic = torch.zeros((args.n_f, 1), dtype=torch.float).requires_grad_(True)
        out_ic = PINN(torch.cat([r_ic, z_ic, t_ic], dim=1))
        u_ic = args.u_scale * out_ic[:, 0:1]
        v_ic = args.v_scale * out_ic[:, 1:2]
        A_r_ic = r_ic * v_ic
        A_theta_ic = r_ic * u_ic
        mse_IC = args.criterion(A_theta_ic, torch.zeros_like(A_theta_ic)) + args.criterion(A_r_ic, torch.zeros_like(A_r_ic))

        # loss
        loss = args.PDE_panelty * mse_PDE + args.BC_panelty * mse_BC + args.ZBC_panelty * mse_ZBC + args.SYM_panelty * mse_SYM + args.SYM0_panelty * mse_SYM0 + args.MID_panelty * mse_MID + args.IC_panelty * mse_IC
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_ZBC.item(), mse_SYM.item(), mse_SYM0.item(), mse_MID.item(), mse_IC.item(), loss.item()])
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, PDE: {mse_PDE.item()}, BC: {mse_BC.item()}, ZBC: {mse_ZBC.item()}, SYM: {mse_SYM.item()}, SYM0: {mse_SYM0.item()}, MID: {mse_MID.item()}, IC: {mse_IC.item()}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    
    # plot_A_3d(PINN, args=args, scale=args.A_scale, z_min=-0.03, z_max=0.03)
    # plot_A_2d(PINN, args=args, scale=args.A_scale, z0=0.0, t=0.5 * args.t_max)
    plot_A_components_2d(PINN, args=args, z0=0.0, t=0.5 * args.t_max)
    return loss_history

def plot_A_3d(
    PINN,
    args=None,
    *,
    x_max: float | None = None,
    z_min: float = -0.05,
    z_max: float = 0.05,
    n_x: int = 11,
    n_y: int = 11,
    n_z: int = 5,
    t: float = 0.0,
    scale: float = 1.0,
    auto_scale: bool = True,
    target_arrow: float | None = None,
    stride: int = 1,
):
    """3D quiver plot of A(x,y,z) for z in [z_min, z_max].

    The PINN is axisymmetric and outputs (u, v) where A_theta=r*u and A_r=r*v (A_z=0).
    We evaluate on a 3D Cartesian grid, convert to (A_x, A_y, A_z), and plot vectors.
    """
    device = next(PINN.parameters()).device

    if args is not None and x_max is None:
        x_max = float(args.r_max)
    if x_max is None:
        x_max = 0.1

    x = torch.linspace(-x_max, x_max, n_x, device=device)
    y = torch.linspace(-x_max, x_max, n_y, device=device)
    z = torch.linspace(z_min, z_max, n_z, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    R = torch.sqrt(X**2 + Y**2)
    inside = R <= x_max
    R_eval = torch.clamp(R, min=1e-6)

    # Evaluate network on (r,z,t) for each grid point
    T = torch.full_like(R_eval, float(t))
    inp = torch.cat([R_eval.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        out = PINN(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        if args is not None:
            u = args.u_scale * u
            v = args.v_scale * v
        A_theta = (R_eval.reshape(-1, 1) * u)
        A_r = (R_eval.reshape(-1, 1) * v)

    # Convert cylindrical -> Cartesian at each point
    inv_r = 1.0 / R_eval.reshape(-1, 1)
    cos_t = X.reshape(-1, 1) * inv_r
    sin_t = Y.reshape(-1, 1) * inv_r
    A_x = A_r * cos_t - A_theta * sin_t
    A_y = A_r * sin_t + A_theta * cos_t
    A_z = torch.zeros_like(A_x)

    # Mask outside cylinder and reshape to grid
    mask = inside.reshape(-1, 1)
    A_x = A_x.masked_fill(~mask, 0.0)
    A_y = A_y.masked_fill(~mask, 0.0)
    A_z = A_z.masked_fill(~mask, 0.0)

    # Auto-scale so arrows are visible in plot units (meters)
    if auto_scale:
        if target_arrow is None:
            # roughly 15% of the plotted radius
            target_arrow = 0.15 * float(x_max)
        mag = torch.sqrt(A_x**2 + A_y**2 + A_z**2).reshape(-1)
        mag_inside = mag[inside.reshape(-1)]
        mag_max = float(mag_inside.max().detach().cpu().item()) if mag_inside.numel() else 0.0
        if mag_max > 0.0:
            scale = float(target_arrow) / mag_max

    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()
    Zn = Z.detach().cpu().numpy()
    U = (A_x.reshape(n_x, n_y, n_z) * scale).detach().cpu().numpy()
    V = (A_y.reshape(n_x, n_y, n_z) * scale).detach().cpu().numpy()
    W = (A_z.reshape(n_x, n_y, n_z) * scale).detach().cpu().numpy()

    # Optional thinning
    xs = slice(None, None, stride)
    ys = slice(None, None, stride)
    zs = slice(None, None, stride)

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    # Plot with y as the vertical axis by swapping plotted coordinates:
    # - matplotlib's vertical axis is the 3rd coordinate ("z" in plotting coords)
    # - we map physical (x, y, z) -> plotted (x, z, y)
    ax.quiver(
        Xn[xs, ys, zs],  # x
        Zn[xs, ys, zs],  # plotted y-axis (depth) shows physical z
        Yn[xs, ys, zs],  # plotted z-axis (vertical) shows physical y
        U[xs, ys, zs],
        W[xs, ys, zs] * 0.0,  # A_z is 0, keep depth component zero
        V[xs, ys, zs],  # vertical component uses physical A_y
        length=1.0,
        normalize=False,
        linewidth=0.6,
        arrow_length_ratio=0.25,
    )
    ax.set_title(f"3D vector potential A (y vertical) (z ∈ [{z_min}, {z_max}], t={t})")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(-x_max, x_max)
    plt.show()

def plot_A_2d(
    PINN,
    args=None,
    *,
    z0: float = 0.0,
    t: float = 0.0,
    x_max: float | None = None,
    grid_n: int = 101,
    r_min: float = 1e-6,
    stride: int = 5,
    scale: float = 1.0,
    auto_scale: bool = True,
    target_arrow: float | None = None,
    show_magnitude: bool = True,
):
    """2D quiver plot of A(x,y,z0) on the x–y plane at fixed z0 and time t.

    The PINN is axisymmetric and outputs (u, v) where A_theta=r*u and A_r=r*v (A_z=0).
    We evaluate on a Cartesian x–y grid, convert to (A_x, A_y), and plot vectors.
    """
    device = next(PINN.parameters()).device

    if args is not None and x_max is None:
        x_max = float(args.r_max)
    if x_max is None:
        x_max = 0.1

    x = torch.linspace(-x_max, x_max, grid_n, device=device)
    y = torch.linspace(-x_max, x_max, grid_n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    inside = R <= x_max
    R_eval = torch.clamp(R, min=float(r_min))

    Z = torch.full_like(R_eval, float(z0))
    T = torch.full_like(R_eval, float(t))
    inp = torch.cat([R_eval.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        out = PINN(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        if args is not None:
            u = args.u_scale * u
            v = args.v_scale * v
        A_theta = (R_eval.reshape(-1, 1) * u)
        A_r = (R_eval.reshape(-1, 1) * v)

    inv_r = 1.0 / R_eval.reshape(-1, 1)
    cos_t = X.reshape(-1, 1) * inv_r
    sin_t = Y.reshape(-1, 1) * inv_r
    A_x = A_r * cos_t - A_theta * sin_t
    A_y = A_r * sin_t + A_theta * cos_t

    mask = inside.reshape(-1, 1)
    A_x = A_x.masked_fill(~mask, 0.0)
    A_y = A_y.masked_fill(~mask, 0.0)

    if auto_scale:
        if target_arrow is None:
            target_arrow = 0.12 * float(x_max)
        mag = torch.sqrt(A_x**2 + A_y**2).reshape(-1)
        mag_inside = mag[inside.reshape(-1)]
        mag_max = float(mag_inside.max().detach().cpu().item()) if mag_inside.numel() else 0.0
        if mag_max > 0.0:
            scale = float(target_arrow) / mag_max

    Axg = (A_x.reshape(grid_n, grid_n) * scale).detach().cpu().numpy()
    Ayg = (A_y.reshape(grid_n, grid_n) * scale).detach().cpu().numpy()
    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    if show_magnitude:
        C = np.hypot(Axg, Ayg)
        Q = ax.quiver(
            Xn[::stride, ::stride],
            Yn[::stride, ::stride],
            Axg[::stride, ::stride],
            Ayg[::stride, ::stride],
            C[::stride, ::stride],
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            pivot="mid",
        )
        fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04, label="|A| (scaled)")
    else:
        ax.quiver(
            Xn[::stride, ::stride],
            Yn[::stride, ::stride],
            Axg[::stride, ::stride],
            Ayg[::stride, ::stride],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            pivot="mid",
        )

    ax.set_title(f"A on x–y plane (z={z0}, t={t})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-x_max, x_max)
    ax.grid(True, alpha=0.25)
    plt.show()


def plot_A_components_2d(
    PINN,
    args=None,
    *,
    z0: float = 0.0,
    t: float = 0.0,
    x_max: float | None = None,
    grid_n: int = 201,
    r_min: float = 1e-6,
):
    """Diagnostic plots of cylindrical components A_r and A_theta on the x–y plane.

    Evaluates the axisymmetric PINN at r=sqrt(x^2+y^2) and plots scalar fields:
    - A_r(x,y,z0,t)
    - A_theta(x,y,z0,t)
    """
    device = next(PINN.parameters()).device

    if args is not None and x_max is None:
        x_max = float(args.r_max)
    if x_max is None:
        x_max = 0.1

    x = torch.linspace(-x_max, x_max, grid_n, device=device)
    y = torch.linspace(-x_max, x_max, grid_n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    inside = R <= x_max
    R_eval = torch.clamp(R, min=float(r_min))

    Z = torch.full_like(R_eval, float(z0))
    T = torch.full_like(R_eval, float(t))
    inp = torch.cat([R_eval.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        out = PINN(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        if args is not None:
            u = args.u_scale * u
            v = args.v_scale * v
        A_theta = (R_eval.reshape(-1, 1) * u).reshape(grid_n, grid_n)
        A_r = (R_eval.reshape(-1, 1) * v).reshape(grid_n, grid_n)

    # mask outside cylinder for clean visualization (don't force values to 0,
    # because 0 can map to a saturated color depending on vmin/vmax)
    A_r_inside = A_r[inside]
    A_theta_inside = A_theta[inside]

    # Print summary stats to confirm whether A_theta is truly negligible
    eps = 1e-30
    max_ar = float(A_r_inside.abs().max().detach().cpu().item()) if A_r_inside.numel() else 0.0
    max_at = float(A_theta_inside.abs().max().detach().cpu().item()) if A_theta_inside.numel() else 0.0
    print(
        f"[A components @ z={z0}, t={t}] max|A_r|={max_ar:.3e}, max|A_theta|={max_at:.3e}, "
        f"max|A_theta|/max|A_r|={max_at/(max_ar+eps):.3e}"
    )

    Ar_np = A_r.detach().cpu().numpy()
    Ath_np = A_theta.detach().cpu().numpy()
    inside_np = inside.detach().cpu().numpy()
    extent = (-x_max, x_max, -x_max, x_max)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    cmap = plt.cm.coolwarm.copy()
    # Render masked/outside region as transparent (shows white background)
    cmap.set_bad(color="white", alpha=0.0)
    Ar_ma = np.ma.array(Ar_np.T, mask=~inside_np.T)
    Ath_ma = np.ma.array(Ath_np.T, mask=~inside_np.T)

    axes[0].set_facecolor("white")
    axes[1].set_facecolor("white")

    im0 = axes[0].imshow(Ar_ma, origin="lower", extent=extent, cmap=cmap)
    axes[0].set_title(f"$A_r$ on x–y (z={z0}, t={t})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(Ath_ma, origin="lower", extent=extent, cmap=cmap)
    axes[1].set_title(f"$A_\\theta$ on x–y (z={z0}, t={t})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.show()

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            # outputs: [u, v] where A_theta=r*u and A_r=r*v
            self.seq_net = [3, 100, 100, 100, 100, 100, 100, 2]
            self.epochs = 300
            self.n_f = 10000
            self.n_b_l = 5000
            self.n_b_z = 5000
            self.n_sym = 5000
            self.n_sym_axis = 10000
            self.n_mid = 5000
            self.PDE_panelty = 1.0
            self.BC_panelty = 1.0
            self.ZBC_panelty = 1.0
            self.SYM_panelty = 1.0
            self.SYM0_panelty = 1.0
            self.MID_panelty = 1.0
            self.IC_panelty = 1.0
            self.optimizer = torch.optim.Adam
            self.lr = 0.001
            self.criterion = torch.nn.MSELoss()
            self.r_max = 0.1
            self.z_max = 0.1
            self.frequency_hz = 1.0e4
            self.t_max = 1.0 / float(self.frequency_hz)
            self.r_coil = 0.085 / 2.0 # 0.0425 m
            self.sigma_r = 0.01
            self.z_sharpness = 10.0
            self.gap = 0.06
            self.coil_length = 0.18
            self.z1_left = -(self.gap / 2.0 + self.coil_length)
            self.z1_right = -(self.gap / 2.0)
            self.z2_left = (self.gap / 2.0)
            self.z2_right = (self.gap / 2.0 + self.coil_length)
            self.current = 100.0
            # output scaling for u,v so A=r*(u or v) has reasonable magnitude
            self.u_scale = 1e-2
            self.v_scale = 1e-2
            self.activation = torch.tanh
            self.z_gap_exclude = 0.05
            self.A_scale = 1000.0
            

    args = ARGS()
    history = train(args)