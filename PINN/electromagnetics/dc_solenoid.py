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


def J_theta(r: torch.Tensor, z: torch.Tensor, args) -> torch.Tensor:
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

    return args.current * Jr * Jz


def PDE(A_theta, r, z, args):
    # Magnetostatics: (∇² - 1/r²)Aθ = -μ Jθ  ->  residual = LHS + μ Jθ
    lhs = d(d(A_theta, r), r) + 1 / r * d(A_theta, r) + d(d(A_theta, z), z) - 1 / r**2 * A_theta
    return lhs + mu * J_theta(r, z, args)

def A_theta_boundary(A_theta, r):
    return d(A_theta, r)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    # calculate J_theta for r=0.05 and z from -0.1 to 0.1
    z = torch.linspace(-0.1, 0.1, 500, dtype=torch.float).reshape(-1, 1)
    r = 0.05 * torch.ones((z.shape[0], 1), dtype=torch.float)
    test = J_theta(r, z, args)
    # plot test
    plt.plot(z.detach().cpu().numpy().squeeze(), test.detach().cpu().numpy().squeeze())
    plt.show()
    print(test.shape)
    
    sys.exit()

    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        r_max = 0.1
        # r is sampled from 0.01 to 0.1
        r = (0.01 + (r_max - 0.01) * torch.rand((args.n_f, 1), dtype=torch.float)).requires_grad_(True)
        # z is sampled from -0.1 to 0.1
        z = (0.1 * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)


        A_theta = PINN(torch.cat([r, z], dim=1))
        PDE_ = PDE(A_theta, r, z, args)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # boundary
        # r is always 0.1
        r_bc = (0.1 * torch.ones((args.n_b_l, 1), dtype=torch.float)).requires_grad_(True)
        # z is sampled from -0.1 to -0.05, and then sampled from 0.05 to 0.1
        n_half = args.n_b_l // 2
        z_bc_low = -0.1 + (0.05 * torch.rand((n_half, 1), dtype=torch.float))  # [-0.1, -0.05]
        z_bc_high = 0.05 + (0.05 * torch.rand((n_half, 1), dtype=torch.float))  # [0.05, 0.1]
        z_bc = torch.cat([z_bc_low, z_bc_high], dim=0).requires_grad_(True)
        A_theta_bc = PINN(torch.cat([r_bc, z_bc], dim=1))
        dA_dr_bc = A_theta_boundary(A_theta_bc, r_bc)
        mse_BC = args.criterion(dA_dr_bc, torch.zeros_like(dA_dr_bc))

        # initial condition
        # r is always 0.01
        r_0 = (0.01 * torch.ones((args.n_f, 1), dtype=torch.float)).requires_grad_(True)
        # z is sampled from -0.1 to 0.1
        z_0 = (0.1 * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        A_theta_0 = PINN(torch.cat([r_0, z_0], dim=1))
        mse_IC = args.criterion(A_theta_0, torch.zeros_like(A_theta_0))

        # loss
        loss = args.PDE_panelty * mse_PDE + args.BC_panelty * mse_BC + args.IC_panelty * mse_IC
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_IC.item(), loss.item()])
        if epoch % 10 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), mse_IC.item(), loss.item()
                )
            )
        loss.backward()
        optimizer.step()

    plot_result(PINN)

def plot_B_yz(PINN, z_min: float = -0.1, z_max: float = 0.1, n_r: int = 120, n_z: int = 200, r_min: float = 0.01, r_max: float = 0.1, step: int = 6, scale_factor: float = 100.0):
    """Plot magnetic field B on the y-z plane (x=0) using axisymmetry.

    Uses:
      B_r = -∂Aθ/∂z
      B_z = (1/r) ∂(r Aθ)/∂r
      B_theta = 0

    On the y-z plane, y=±r and the plotted components are:
      B_y(y,z) = sign(y) * B_r(r,z)
      B_z(y,z) = B_z(r,z)
    """
    device = next(PINN.parameters()).device

    r = torch.linspace(r_min, r_max, n_r, device=device).reshape(-1, 1)
    z = torch.linspace(z_min, z_max, n_z, device=device).reshape(1, -1)
    r_m, z_m = torch.meshgrid(r.squeeze(1), z.squeeze(0), indexing="ij")

    r_f = r_m.reshape(-1, 1).clone().detach().requires_grad_(True)
    z_f = z_m.reshape(-1, 1).clone().detach().requires_grad_(True)
    inp = torch.cat([r_f, z_f], dim=1)
    A = PINN(inp)

    B_r = (-d(A, z_f)).reshape(n_r, n_z)
    B_z = ((1.0 / r_f) * d(r_f * A, r_f)).reshape(n_r, n_z)

    # mirror to y in [-r_max, r_max]
    y_m_pos = r_m
    y_m_neg = -r_m
    z_m_full = torch.cat([z_m, z_m], dim=0)
    y_m_full = torch.cat([y_m_neg, y_m_pos], dim=0)

    B_y_full = torch.cat([-B_r, B_r], dim=0)
    B_z_full = torch.cat([B_z, B_z], dim=0)

    Y = y_m_full.detach().cpu().numpy()
    Z = z_m_full.detach().cpu().numpy()
    By = (scale_factor * B_y_full).detach().cpu().numpy()
    Bz = (scale_factor * B_z_full).detach().cpu().numpy()
    Bmag = np.hypot(By, Bz)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    # Quiver plot with arrow color = |B| (similar to typical EM field plots)
    # swap axes: horizontal=z, vertical=y
    Q = ax.quiver(
        Z[::step, ::step],   # x-axis: z
        Y[::step, ::step],   # y-axis: y
        Bz[::step, ::step],  # x-component: Bz
        By[::step, ::step],  # y-component: By
        Bmag[::step, ::step],
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        pivot="mid",
    )
    fig.colorbar(Q, ax=ax, fraction=0.046, pad=0.04, label="|B| (arb.)")
    ax.set_title("Magnetic field B (z-y plane, x=0)")
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def plot_result(PINN, z_slices=(-0.05, 0.0, 0.05), grid_n: int = 200, r_min: float = 0.01, r_max: float = 0.1):
    """Plot A_theta on the x-y plane for several fixed z values."""
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
        inp = torch.cat([R_eval.reshape(-1, 1), Z.reshape(-1, 1)], dim=1)
        A_theta = PINN(inp).reshape(grid_n, grid_n)
        A_plot = A_theta.masked_fill(~inside, float("nan"))

        img = ax.imshow(
            A_plot.detach().cpu().numpy().T,
            origin="lower",
            extent=(-r_max, r_max, -r_max, r_max),
            cmap="jet",
        )
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"A_theta (z={z0})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # plt.show()

    # Field plot on the y-z plane
    plot_B_yz(PINN, z_min=-0.1, z_max=0.1, n_r=120, n_z=200, r_min=r_min, r_max=r_max, step=6, scale_factor=13.0)

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [2, 100, 100, 100, 100, 100, 100, 1]
            self.epochs = 300
            self.n_f = 10000
            # self.n_f_1 = 10000
            # self.n_f_2 = 10000
            self.n_b_l = 5000
            self.PDE_panelty = 1.0
            self.BC_panelty = 1.0
            self.BC_inner_panelty = 1.0
            self.IC_panelty = 1.0
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
            self.r_coil = 0.05
            self.z1_left = -0.1
            self.z1_right = -0.05
            self.z2_left = 0.05
            self.z2_right = 0.1
            # smoothing controls for trainable "sheet" current
            self.sigma_r = 0.002
            self.z_sharpness = 200.0


    args = ARGS()
    train(args)