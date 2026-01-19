import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import grad
from PINN.networks import Net
import sys

mu = 4 * np.pi * 1e-7 # vacuum permeability
epsilon = 8.854e-12 # vacuum permittivity
I_0 = 1.0 # current amplitude
r_0 = 0.05 # solenoid radius

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

# use cylindrical coordinates (r, theta, z)
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


def mu_r_field(r: torch.Tensor, z: torch.Tensor, args) -> torch.Tensor:
    """Relative permeability μ_r(r,z) for a ferrite ring inside each coil.

    Ring geometry (front view): core_r_inner <= r <= core_r_outer
    Axial extent (side view): matches coil 1 and coil 2 z-intervals.

    Uses smooth (sigmoid) transitions for differentiability.
    """
    H = _smooth_step
    # radial shell
    r_in = H(r - args.core_r_inner, args.core_sharpness)
    r_out = H(args.core_r_outer - r, args.core_sharpness)
    radial_shell = r_in * r_out

    # axial extent: same as coil segments
    z1 = H(z - args.z1_left, args.z_sharpness) * H(args.z1_right - z, args.z_sharpness)
    z2 = H(z - args.z2_left, args.z_sharpness) * H(args.z2_right - z, args.z_sharpness)
    axial = z1 + z2

    core_mask = radial_shell * axial
    return 1.0 + (args.mu_r_core - 1.0) * core_mask


def PDE(u, r, z, t, args):
    """Axisymmetric PDE in cylindrical coordinates, stabilized at r≈0.

    The naive cylindrical form for A_theta contains 1/r and 1/r^2 terms, which are
    singular on the axis. Enforce regularity by parameterizing:

      A_theta(r,z,t) = r * u(r,z,t)

    With this substitution, the operator simplifies to:

      (∇² - 1/r²)A_theta = r*(u_rr + u_zz) + 3*u_r

    so no divisions by r are required and r=0 is well-defined.
    """
    # Use variable-permeability form:
    #   -∂z(ν ∂Aθ/∂z) - ∂r(ν (1/r) ∂(rAθ)/∂r) = Jθ
    # where ν = 1/μ.
    #
    # With Aθ = r*u:
    #   ∂Aθ/∂z = r*u_z
    #   (1/r) ∂(rAθ)/∂r = (1/r) ∂(r^2 u)/∂r = 2u + r*u_r
    mu_local = mu * mu_r_field(r, z, args)
    nu = 1.0 / mu_local

    A_z = d(r * u, z)
    u_r = d(u, r)
    flux_r = nu * (2.0 * u + r * u_r)

    lhs = -d(nu * A_z, z) - d(flux_r, r)
    return lhs - J_theta(r, z, t, args)

def A_theta_boundary(u, r):
    # d/dr (A_theta) where A_theta = r*u  ->  dA/dr = u + r*u_r
    return u + r * d(u, r)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    # calculate J_theta for r=0.05 and z from -0.1 to 0.1 and t from 0 to 1
    # z = torch.linspace(-0.1, 0.1, 500, dtype=torch.float).reshape(-1, 1)
    # r = 0.05 * torch.ones((z.shape[0], 1), dtype=torch.float)
    

    # test = J_theta(r, z, args)
    # plot test
    # plt.plot(z.detach().cpu().numpy().squeeze(), test.detach().cpu().numpy().squeeze())
    # plt.show()
    # print(test.shape)
    
    # sys.exit()

    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        r_max = 0.1
        # r is sampled in [0, r_max], with extra density near r=0 (axis)
        r = (r_max * torch.rand((args.n_f, 1), dtype=torch.float) ** 2).requires_grad_(True)
        # z is sampled from -0.1 to 0.1
        z = (0.1 * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        # t is randomly sampled between 0 and 1 (seconds)
        t = torch.rand((args.n_f, 1), dtype=torch.float).requires_grad_(True)

        # Scale network output so that A_theta is in a realistic magnitude range.
        # In air, ν=1/μ is huge, so the physical solution typically has very small A.
        u = args.u_scale * PINN(torch.cat([r, z, t], dim=1))
        PDE_ = PDE(u, r, z, t, args)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # boundary
        # r is always 0.1
        r_bc = (0.1 * torch.ones((args.n_b_l, 1), dtype=torch.float)).requires_grad_(True)
        # z is sampled from -0.1 to -0.05, and then sampled from 0.05 to 0.1
        n_half = args.n_b_l // 2
        z_bc_low = -0.1 + (0.05 * torch.rand((n_half, 1), dtype=torch.float))  # [-0.1, -0.05]
        z_bc_high = 0.05 + (0.05 * torch.rand((n_half, 1), dtype=torch.float))  # [0.05, 0.1]
        z_bc = torch.cat([z_bc_low, z_bc_high], dim=0).requires_grad_(True)
        t_bc = torch.rand((args.n_b_l, 1), dtype=torch.float).requires_grad_(True)
        u_bc = args.u_scale * PINN(torch.cat([r_bc, z_bc, t_bc], dim=1))
        dA_dr_bc = A_theta_boundary(u_bc, r_bc)
        mse_BC = args.criterion(dA_dr_bc, torch.zeros_like(dA_dr_bc))

        # z-boundary (Neumann): dA_theta/dz = 0 at z = ±0.1
        n_half_z = args.n_b_z // 2
        r_zbc = (r_max * torch.rand((args.n_b_z, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_zbc = torch.cat(
            [
                (-0.1 * torch.ones((n_half_z, 1), dtype=torch.float)),
                (0.1 * torch.ones((args.n_b_z - n_half_z, 1), dtype=torch.float)),
            ],
            dim=0,
        ).requires_grad_(True)
        t_zbc = torch.rand((args.n_b_z, 1), dtype=torch.float).requires_grad_(True)
        u_zbc = args.u_scale * PINN(torch.cat([r_zbc, z_zbc, t_zbc], dim=1))
        A_zbc = r_zbc * u_zbc
        dA_dz_bc = d(A_zbc, z_zbc)
        mse_ZBC = args.criterion(dA_dz_bc, torch.zeros_like(dA_dz_bc))

        # symmetry about z=0 (even): u(r,z,t) == u(r,-z,t)
        r_sym = r_max * torch.rand((args.n_sym, 1), dtype=torch.float) ** 2
        z_sym = 0.1 * torch.rand((args.n_sym, 1), dtype=torch.float)  # [0, 0.1]
        t_sym = torch.rand((args.n_sym, 1), dtype=torch.float)
        u_pos = args.u_scale * PINN(torch.cat([r_sym, z_sym, t_sym], dim=1))
        u_neg = args.u_scale * PINN(torch.cat([r_sym, -z_sym, t_sym], dim=1))
        mse_SYM = args.criterion(u_pos - u_neg, torch.zeros_like(u_pos))

        # symmetry on-axis (targets Bz(0,z)=2u(0,z)): u(0,z,t) == u(0,-z,t)
        r_sym0 = torch.zeros((args.n_sym_axis, 1), dtype=torch.float)
        z_sym0 = 0.1 * torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        t_sym0 = torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        u_pos0 = args.u_scale * PINN(torch.cat([r_sym0, z_sym0, t_sym0], dim=1))
        u_neg0 = args.u_scale * PINN(torch.cat([r_sym0, -z_sym0, t_sym0], dim=1))
        mse_SYM0 = args.criterion(u_pos0 - u_neg0, torch.zeros_like(u_pos0))

        # mid-plane Neumann symmetry: ∂A_theta/∂z = 0 at z=0
        r_mid = (r_max * torch.rand((args.n_mid, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_mid = torch.zeros((args.n_mid, 1), dtype=torch.float).requires_grad_(True)
        t_mid = torch.rand((args.n_mid, 1), dtype=torch.float).requires_grad_(True)
        u_mid = args.u_scale * PINN(torch.cat([r_mid, z_mid, t_mid], dim=1))
        A_mid = r_mid * u_mid
        dA_dz_mid = d(A_mid, z_mid)
        mse_MID = args.criterion(dA_dz_mid, torch.zeros_like(dA_dz_mid))

        # Initial condition: with a sinusoidal drive, current is 0 at t=0, so A_theta should be ~0.
        # Enforce A_theta(r,z,t=0)=0 over the full domain (this is non-trivial, unlike enforcing at r=0).
        r_ic = (r_max * torch.rand((args.n_f, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_ic = (0.1 * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        t_ic = torch.zeros((args.n_f, 1), dtype=torch.float).requires_grad_(True)
        u_ic = args.u_scale * PINN(torch.cat([r_ic, z_ic, t_ic], dim=1))
        A_ic = r_ic * u_ic
        mse_IC = args.criterion(A_ic, torch.zeros_like(A_ic))

        # loss
        loss = (
            args.PDE_panelty * mse_PDE
            + args.BC_panelty * mse_BC
            + args.ZBC_panelty * mse_ZBC
            + args.SYM_panelty * mse_SYM
            + args.SYM0_panelty * mse_SYM0
            + args.MID_panelty * mse_MID
            + args.IC_panelty * mse_IC
        )
        loss_history.append(
            [
                mse_PDE.item(),
                mse_BC.item(),
                mse_ZBC.item(),
                mse_SYM.item(),
                mse_SYM0.item(),
                mse_MID.item(),
                mse_IC.item(),
                loss.item(),
            ]
        )
        if epoch % 10 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC_r: {:.08e}, BC_z: {:.08e}, SYM: {:.08e}, SYM0: {:.08e}, MID: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    epoch,
                    mse_PDE.item(),
                    mse_BC.item(),
                    mse_ZBC.item(),
                    mse_SYM.item(),
                    mse_SYM0.item(),
                    mse_MID.item(),
                    mse_IC.item(),
                    loss.item(),
                )
            )
        loss.backward()
        optimizer.step()

    # plot_result(PINN, scale=1000.0)
    plot_B_yz(PINN, scale_factor=10.0, t=0.495)

def plot_B_yz(
    PINN,
    z_min: float = -0.1,
    z_max: float = 0.1,
    n_z: int = 200,
    scale_factor: float = 100.0,
    t: float = 0.5,
):
    """Plot scalar Bz on the z-axis (r=0).

    With A_theta = r*u:
      B_z(0,z) = 2u(0,z)
    """
    device = next(PINN.parameters()).device

    z = torch.linspace(z_min, z_max, n_z, device=device).reshape(-1, 1)
    r = torch.zeros_like(z)
    t_f = torch.full_like(z, float(t))

    inp = torch.cat([r, z, t_f], dim=1)
    with torch.no_grad():
        u = PINN(inp)
        Bz = scale_factor * (2.0 * u)

    z_np = z.squeeze(1).detach().cpu().numpy()
    Bz_np = Bz.squeeze(1).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.plot(z_np, Bz_np, linewidth=2)
    ax.set_title(f"On-axis Bz at r=0 (t={t})")
    ax.set_xlabel("z")
    ax.set_ylabel("Bz (scaled)")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_result(
    PINN,
    z_slices=(-0.05, 0.0, 0.05),
    grid_n: int = 200,
    r_min: float = 0.01,
    r_max: float = 0.1,
    t: float = 0.5,
    step: int = 8,
    scale: float = 1.0,
):
    """Plot vector potential A (azimuthal) on the x-y plane for fixed z at time t.

    A has only a theta component: A = A_theta(r,z,t) * e_theta.
    In Cartesian coordinates (x,y):
      e_theta = (-y/r, x/r)
      A_x = A_theta * (-y/r)
      A_y = A_theta * ( x/r)
    """
    device = next(PINN.parameters()).device

    x = torch.linspace(-r_max, r_max, grid_n, device=device)
    y = torch.linspace(-r_max, r_max, grid_n, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)
    inside = R <= r_max
    R_eval = torch.clamp(R, min=r_min)

    fig, axes = plt.subplots(1, len(z_slices), figsize=(5 * len(z_slices), 4), constrained_layout=True)
    if len(z_slices) == 1:
        axes = [axes]

    for ax, z0 in zip(axes, z_slices):
        Z = torch.full_like(R_eval, float(z0), device=device)
        T = torch.full_like(R_eval, float(t), device=device)
        inp = torch.cat([R_eval.reshape(-1, 1), Z.reshape(-1, 1), T.reshape(-1, 1)], dim=1)
        u = PINN(inp).reshape(grid_n, grid_n)
        A_theta = R_eval * u
        A_theta = A_theta.masked_fill(~inside, 0.0)

        # Convert azimuthal component to Cartesian vector field on x-y plane
        eps = 1e-12
        inv_r = 1.0 / torch.clamp(R, min=eps)
        Ax = A_theta * (-Y * inv_r)
        Ay = A_theta * (X * inv_r)

        # Downsample for quiver
        Xn = X.detach().cpu().numpy()
        Yn = Y.detach().cpu().numpy()
        U = (Ax * scale).detach().cpu().numpy()
        V = (Ay * scale).detach().cpu().numpy()
        C = np.hypot(U, V)

        Q = ax.quiver(
            Xn[::step, ::step],
            Yn[::step, ::step],
            U[::step, ::step],
            V[::step, ::step],
            C[::step, ::step],
            cmap="jet",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            pivot="mid",
        )
        fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04, label="|A| (arb.)")
        ax.set_title(f"A (tangential), z={z0}, t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [3, 100, 100, 100, 100, 100, 100, 1]
            self.epochs = 500
            self.n_f = 10000
            # self.n_f_1 = 10000
            # self.n_f_2 = 10000
            self.n_b_l = 5000
            self.n_b_z = 5000
            self.n_sym = 5000
            self.n_sym_axis = 10000
            self.n_mid = 5000
            # Loss weights: raw mse_PDE is ~1e-3..1e-2 while mse_BC is ~1e-22,
            # so BC needs a very large weight to matter in the total loss.
            self.PDE_panelty = 1.0
            self.BC_panelty = 1e19
            self.ZBC_panelty = 1e19
            self.SYM_panelty = 1e18
            self.SYM0_panelty = 1e19
            self.MID_panelty = 1e18
            self.BC_inner_panelty = 1.0
            self.IC_panelty = 1e19
            self.lr = 0.001
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam
            self.activation = torch.tanh
            self.activ_name = 'tanh'
            self.x_left = -0.1
            self.x_right = 0.1
            self.y_left = -0.1
            self.y_right = 0.1
            # solenoid current model (two coils separated by 0.1 gap)
            self.current = 1.0
            self.frequency_hz = 50.0
            self.r_coil = 0.04
            self.z1_left = -0.1
            self.z1_right = -0.05
            self.z2_left = 0.05
            self.z2_right = 0.1
            # smoothing controls for trainable "sheet" current
            self.sigma_r = 0.002
            self.z_sharpness = 200.0
            # ferrite ring (MnZn core) inside each coil
            self.core_r_inner = 0.02
            self.core_r_outer = 0.04
            self.core_sharpness = 400.0
            # MnZn ferrite can have very high μr; start moderate for training stability
            self.mu_r_core = 200.0
            # output scaling for u so A_theta=r*u has realistic magnitude
            self.u_scale = 1e-6


    args = ARGS()
    train(args)