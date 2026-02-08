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

# use cylindrical coordinates (r, theta, z)
def _smooth_step(x: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Smooth approximation of Heaviside(x)."""
    return torch.sigmoid(sharpness * x)


def _smoothstep01(u: torch.Tensor) -> torch.Tensor:
    """C1 smoothstep on [0,1]: 3u^2 - 2u^3, with clamping for stability."""
    u = torch.clamp(u, 0.0, 1.0)
    return 3.0 * u * u - 2.0 * u * u * u


def _softplus_beta(x: torch.Tensor, beta: float) -> torch.Tensor:
    """Softplus with explicit beta (everywhere differentiable)."""
    return F.softplus(x * beta) / beta


def _pulse_current(t: torch.Tensor, args) -> torch.Tensor:
    """Everywhere-differentiable pulsed current waveform (peak I_pk).

    Implements the form shown in the attached notes:
      θ(t) = 2π f t
      I(t) = I_pk * [(1-w(θ)) R(θ) + w(θ) * D(θ)/D0]
    """
    two_pi = 2.0 * torch.pi
    theta = two_pi * float(args.frequency_hz) * t
    # Make θ_r a tensor (same device/dtype as t) for torch.sin compatibility
    theta_r = t.new_tensor(two_pi * float(args.rise_fraction))  # rise ends at θ_r
    T = 1.0 / float(args.frequency_hz)
    t_d = (1.0 - float(args.rise_fraction)) * T

    # smooth periodic gate
    w = torch.sigmoid(torch.sin(theta - theta_r) / float(args.gate_eps))

    # rise progress: u_r(θ) = sin(θ/2) / sin(θ_r/2)
    denom_r = torch.sin(theta_r / 2.0)
    u_r = torch.sin(theta / 2.0) / (denom_r + 1e-12)
    R = _smoothstep01(u_r)

    # decay progress: u_d(θ) = sin((θ-θ_r)/2) / sin((2π-θ_r)/2)
    denom_d = torch.sin((two_pi - theta_r) / 2.0)
    u_d = torch.sin((theta - theta_r) / 2.0) / (denom_d + 1e-12)
    Delta = torch.clamp(u_d, 0.0, 1.0) * t_d

    # nonlinear core used inside softplus
    q = torch.exp(0.13 * Delta) * torch.sin(321.0 * Delta) / 0.74
    D = _softplus_beta(q, float(args.softplus_beta))
    D0 = _softplus_beta(torch.zeros_like(D), float(args.softplus_beta))

    return float(args.I_pk) * ((1.0 - w) * R + w * (D / (D0 + 1e-12)))


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

    # Pulsed drive (everywhere differentiable)
    I_t = _pulse_current(t, args)
    J_mag = I_t / (torch.pi * float(args.wire_radius) ** 2)  # uniform over πa^2
    return J_mag * Jr * Jz


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


def PDE(
    phi: torch.Tensor,
    u_theta: torch.Tensor,
    v_r: torch.Tensor,
    r: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    args,
):
    """Coupled (phi, A_r, A_theta) system in axisymmetric cylindrical coordinates.

    Potentials:
      B = ∇×A
      E = -∇φ - ∂A/∂t

    Gauge (note sign):
      ∇·A = (1/c^2) ∂φ/∂t

    Axisymmetry: ∂/∂θ = 0, and we assume A_z = 0 (per your instruction).

    To avoid 1/r singularities on the axis, we parameterize:
      A_theta(r,z,t) = r * u_theta(r,z,t)
      A_r(r,z,t)     = r * v_r(r,z,t)

    PDEs (from the attached notes, specialized to axisymmetry and A_z=0):
      ∇²φ + (1/c^2) φ_tt = 0                              (ρ ≈ 0)
      ΔA_r - A_r/r^2 - (1/c^2) A_r,tt = (2/c^2) ∂r(φ_t)   (J_r = 0)
      ΔA_θ - A_θ/r^2 - (1/c^2) A_θ,tt = -μ0 J_θ

    Gauge residual:
      (1/r) ∂(r A_r)/∂r - (1/c^2) φ_t = 0
    """
    r_safe = torch.clamp(r, min=1e-8)

    # Construct vector potential components
    A_theta = r * u_theta
    A_r = r * v_r

    # Scalar potential wave equation (ρ≈0)
    phi_r = d(phi, r)
    phi_rr = d(phi_r, r)
    phi_zz = d(d(phi, z), z)
    phi_tt = d(d(phi, t), t)
    lap_phi = phi_rr + (1.0 / r_safe) * phi_r + phi_zz
    res_phi = lap_phi + (1.0 / c2) * phi_tt

    # Vector potential equations (component-wise)
    # Use scalar Laplacian Δf = f_rr + (1/r) f_r + f_zz (axisymmetry)
    def lap_scalar(f: torch.Tensor) -> torch.Tensor:
        f_r = d(f, r)
        f_rr = d(f_r, r)
        f_zz = d(d(f, z), z)
        return f_rr + (1.0 / r_safe) * f_r + f_zz

    Atheta_tt = d(d(A_theta, t), t)
    Ar_tt = d(d(A_r, t), t)

    lap_Atheta = lap_scalar(A_theta)
    lap_Ar = lap_scalar(A_r)

    # φ coupling appears only in A_r for axisymmetry (θ-gradient term is zero)
    phi_t = d(phi, t)
    dphi_t_dr = d(phi_t, r)

    res_A_theta = lap_Atheta - (A_theta / (r_safe**2)) - (1.0 / c2) * Atheta_tt + mu * J_theta(r, z, t, args)
    res_A_r = lap_Ar - (A_r / (r_safe**2)) - (1.0 / c2) * Ar_tt - (2.0 / c2) * dphi_t_dr

    # Gauge residual: (1/r) ∂(rA_r)/∂r - (1/c^2) φ_t
    # with A_r = r v_r, so (1/r)∂(rA_r)/∂r = 2 v_r + r v_r,r (no singularity)
    v_rr = d(v_r, r)
    div_A = 2.0 * v_r + r * v_rr
    res_gauge = div_A - (1.0 / c2) * phi_t

    return res_phi, res_A_theta, res_A_r, res_gauge

def A_theta_boundary(u, r):
    # d/dr (A_theta) where A_theta = r*u  ->  dA/dr = u + r*u_r
    return u + r * d(u, r)

def A_r_boundary(v, r):
    # d/dr (A_r) where A_r = r*v  ->  dA/dr = v + r*v_r
    return v + r * d(v, r)

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
        phi = out[:, 0:1]
        u_theta = args.u_scale * out[:, 1:2]
        v_r = args.v_scale * out[:, 2:3]
        res_phi, res_A_theta, res_A_r, res_gauge = PDE(phi, u_theta, v_r, r, z, t, args)
        mse_PDE_phi = args.criterion(res_phi, torch.zeros_like(res_phi))
        mse_PDE_A = args.criterion(res_A_theta, torch.zeros_like(res_A_theta)) + args.criterion(
            res_A_r, torch.zeros_like(res_A_r)
        )
        mse_gauge = args.criterion(res_gauge, torch.zeros_like(res_gauge))
        mse_PDE = mse_PDE_phi + mse_PDE_A + args.gauge_penalty * mse_gauge

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
        u_bc = args.u_scale * out_bc[:, 1:2]
        v_bc = args.v_scale * out_bc[:, 2:3]
        dAθ_dr_bc = A_theta_boundary(u_bc, r_bc)
        dAr_dr_bc = A_r_boundary(v_bc, r_bc)
        mse_BC = args.criterion(dAθ_dr_bc, torch.zeros_like(dAθ_dr_bc)) + args.criterion(
            dAr_dr_bc, torch.zeros_like(dAr_dr_bc)
        )

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
        u_zbc = args.u_scale * out_zbc[:, 1:2]
        v_zbc = args.v_scale * out_zbc[:, 2:3]
        Aθ_zbc = r_zbc * u_zbc
        Ar_zbc = r_zbc * v_zbc
        dAθ_dz_bc = d(Aθ_zbc, z_zbc)
        dAr_dz_bc = d(Ar_zbc, z_zbc)
        mse_ZBC = args.criterion(dAθ_dz_bc, torch.zeros_like(dAθ_dz_bc)) + args.criterion(
            dAr_dz_bc, torch.zeros_like(dAr_dz_bc)
        )

        # symmetry about z=0 (even): u(r,z,t) == u(r,-z,t)
        r_sym = r_max * torch.rand((args.n_sym, 1), dtype=torch.float) ** 2
        z_sym = args.z_max * torch.rand((args.n_sym, 1), dtype=torch.float)  # [0, z_max]
        t_sym = args.t_max * torch.rand((args.n_sym, 1), dtype=torch.float)
        out_pos = PINN(torch.cat([r_sym, z_sym, t_sym], dim=1))
        out_neg = PINN(torch.cat([r_sym, -z_sym, t_sym], dim=1))
        phi_pos = out_pos[:, 0:1]
        phi_neg = out_neg[:, 0:1]
        u_pos = args.u_scale * out_pos[:, 1:2]
        u_neg = args.u_scale * out_neg[:, 1:2]
        v_pos = args.v_scale * out_pos[:, 2:3]
        v_neg = args.v_scale * out_neg[:, 2:3]
        # symmetry about z=0: phi, A_theta even; A_r odd
        mse_SYM = (
            args.criterion(phi_pos - phi_neg, torch.zeros_like(phi_pos))
            + args.criterion(u_pos - u_neg, torch.zeros_like(u_pos))
            + args.criterion(v_pos + v_neg, torch.zeros_like(v_pos))
        )

        # symmetry on-axis (targets Bz(0,z)=2u(0,z)): u(0,z,t) == u(0,-z,t)
        r_sym0 = torch.zeros((args.n_sym_axis, 1), dtype=torch.float)
        z_sym0 = args.z_max * torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        t_sym0 = args.t_max * torch.rand((args.n_sym_axis, 1), dtype=torch.float)
        out_pos0 = PINN(torch.cat([r_sym0, z_sym0, t_sym0], dim=1))
        out_neg0 = PINN(torch.cat([r_sym0, -z_sym0, t_sym0], dim=1))
        u_pos0 = args.u_scale * out_pos0[:, 1:2]
        u_neg0 = args.u_scale * out_neg0[:, 1:2]
        v_pos0 = args.v_scale * out_pos0[:, 2:3]
        v_neg0 = args.v_scale * out_neg0[:, 2:3]
        mse_SYM0 = args.criterion(u_pos0 - u_neg0, torch.zeros_like(u_pos0)) + args.criterion(
            v_pos0 + v_neg0, torch.zeros_like(v_pos0)
        )

        # mid-plane Neumann symmetry: ∂A_theta/∂z = 0 at z=0
        r_mid = (r_max * torch.rand((args.n_mid, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_mid = torch.zeros((args.n_mid, 1), dtype=torch.float).requires_grad_(True)
        t_mid = (args.t_max * torch.rand((args.n_mid, 1), dtype=torch.float)).requires_grad_(True)
        out_mid = PINN(torch.cat([r_mid, z_mid, t_mid], dim=1))
        u_mid = args.u_scale * out_mid[:, 1:2]
        v_mid = args.v_scale * out_mid[:, 2:3]
        Aθ_mid = r_mid * u_mid
        Ar_mid = r_mid * v_mid
        dAθ_dz_mid = d(Aθ_mid, z_mid)
        # mid-plane: ∂A_theta/∂z = 0 and A_r = 0 (odd symmetry)
        mse_MID = args.criterion(dAθ_dz_mid, torch.zeros_like(dAθ_dz_mid)) + args.criterion(
            Ar_mid, torch.zeros_like(Ar_mid)
        )

        # Initial condition: with a sinusoidal drive, current is 0 at t=0, so A_theta should be ~0.
        # Enforce A_theta(r,z,t=0)=0 over the full domain (this is non-trivial, unlike enforcing at r=0).
        r_ic = (r_max * torch.rand((args.n_f, 1), dtype=torch.float) ** 2).requires_grad_(True)
        z_ic = (args.z_max * (2 * torch.rand((args.n_f, 1), dtype=torch.float) - 1)).requires_grad_(True)
        t_ic = torch.zeros((args.n_f, 1), dtype=torch.float).requires_grad_(True)
        out_ic = PINN(torch.cat([r_ic, z_ic, t_ic], dim=1))
        phi_ic = out_ic[:, 0:1]
        u_ic = args.u_scale * out_ic[:, 1:2]
        v_ic = args.v_scale * out_ic[:, 2:3]
        Aθ_ic = r_ic * u_ic
        Ar_ic = r_ic * v_ic
        mse_IC = (
            args.criterion(Aθ_ic, torch.zeros_like(Aθ_ic))
            + args.criterion(Ar_ic, torch.zeros_like(Ar_ic))
            + args.phi_ic_penalty * args.criterion(phi_ic, torch.zeros_like(phi_ic))
        )

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
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_ZBC.item(), mse_IC.item(), loss.item()])
        if epoch % 10 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC_r: {:.08e}, BC_z: {:.08e}, IC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), mse_ZBC.item(), mse_IC.item(), loss.item()
                )
            )
        loss.backward()
        optimizer.step()

    plot_A_2d(PINN, args=args, scale=1000.0, z0 = 0.0, t = 0.5 * args.t_max)
    # plot_A_3d(PINN, args=args, scale=1000.0, z_min=-0.05, z_max=0.05)
    # plot_B_yz(PINN, args=args, scale_factor=10.0, t=0.5 * args.t_max)

def plot_B_yz(
    PINN,
    z_min: float = -0.25,
    z_max: float = 0.25,
    n_z: int = 200,
    scale_factor: float = 100.0,
    t: float = 0.0,
    args=None,
):
    """Plot scalar Bz on the z-axis (r=0).

    With A_theta = r*u:
      B_z(0,z) = 2u(0,z)
    """
    device = next(PINN.parameters()).device
    if args is not None:
        z_min = float(-args.z_max)
        z_max = float(args.z_max)

    z = torch.linspace(z_min, z_max, n_z, device=device).reshape(-1, 1)
    r = torch.zeros_like(z)
    t_f = torch.full_like(z, float(t))

    inp = torch.cat([r, z, t_f], dim=1)
    with torch.no_grad():
        out = PINN(inp)
        u = out[:, 1:2]  # u_theta
        if args is not None:
            u = args.u_scale * u
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
    t: float = 0.0,
    step: int = 8,
    scale: float = 1.0,
    args=None,
):
    """Plot vector potential A on the x-y plane for fixed z at time t.

    Here A has two components: A = A_r e_r + A_theta e_theta, with A_z=0.
    In Cartesian coordinates (x,y):
      e_r     = (x/r, y/r)
      e_theta = (-y/r, x/r)
      A_x = A_r * (x/r) + A_theta * (-y/r)
      A_y = A_r * (y/r) + A_theta * ( x/r)
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
        out = PINN(inp)
        u = out[:, 1:2]  # u_theta
        v = out[:, 2:3]  # v_r
        if args is not None:
            u = args.u_scale * u
            v = args.v_scale * v
        u = u.reshape(grid_n, grid_n)
        v = v.reshape(grid_n, grid_n)
        A_theta = R_eval * u
        A_r = R_eval * v
        A_theta = A_theta.masked_fill(~inside, 0.0)
        A_r = A_r.masked_fill(~inside, 0.0)

        # Convert azimuthal component to Cartesian vector field on x-y plane
        eps = 1e-12
        inv_r = 1.0 / torch.clamp(R, min=eps)
        Ax = A_r * (X * inv_r) + A_theta * (-Y * inv_r)
        Ay = A_r * (Y * inv_r) + A_theta * (X * inv_r)

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

    The PINN is axisymmetric and outputs cylindrical components (A_r, A_theta), with A_z=0.
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
        u = out[:, 1:2]  # u_theta
        v = out[:, 2:3]  # v_r
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

    The PINN is axisymmetric and outputs cylindrical components (A_r, A_theta), with A_z=0.
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
        u = out[:, 1:2]  # u_theta
        v = out[:, 2:3]  # v_r
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
    ax.quiver(
        Xn[xs, ys, zs],
        Yn[xs, ys, zs],
        Zn[xs, ys, zs],
        U[xs, ys, zs],
        V[xs, ys, zs],
        W[xs, ys, zs],
        length=1.0,
        normalize=False,
        linewidth=0.6,
        arrow_length_ratio=0.25,
    )
    ax.set_title(f"3D vector potential A (z ∈ [{z_min}, {z_max}], t={t})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-x_max, x_max)
    ax.set_zlim(z_min, z_max)
    plt.show()


if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            # outputs: [phi, u_theta, v_r] where A_theta=r*u_theta and A_r=r*v_r
            self.seq_net = [3, 100, 100, 100, 100, 100, 100, 3]
            self.epochs = 200
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
            # training/plot domain (meters)
            self.r_max = 0.10
            self.z_max = 0.25
            # exclude a small region around z=0 when sampling r=r_max boundary points
            self.z_gap_exclude = 0.05
            # solenoid current model (two coils separated by 0.1 gap)
            # pulsed drive parameters
            self.I_pk = 100.0
            self.frequency_hz = 1.0e4
            self.rise_fraction = 0.2
            self.gate_eps = 0.05
            self.softplus_beta = 50.0
            self.wire_radius = 1.0e-3
            # ---- Geometry (meters) from experiment description ----
            # Wire is wound into circular loops of diameter 8.5 cm
            self.r_coil = 0.085 / 2.0  # 0.0425 m
            # Two solenoids separated by a 6 cm axial gap (between facing ends)
            self.gap = 0.06
            # Solenoid axial length (meters)
            self.coil_length = 0.18
            self.z1_left = -(self.gap / 2.0 + self.coil_length)
            self.z1_right = -(self.gap / 2.0)
            self.z2_left = (self.gap / 2.0)
            self.z2_right = (self.gap / 2.0 + self.coil_length)
            # smoothing controls for trainable "sheet" current
            self.sigma_r = 0.002
            self.z_sharpness = 200.0
            # MnZn ferrite toroidal cores (treated as homogeneous bodies)
            # outer diameter 8 cm -> outer radius 0.04 m
            # inner diameter 1 cm -> inner radius 0.005 m (hollow region inside)
            self.core_r_inner = 0.01 / 2.0  # 0.005 m
            self.core_r_outer = 0.08 / 2.0  # 0.04 m
            self.core_sharpness = 400.0
            # MnZn ferrite can have very high μr; start moderate for training stability
            self.mu_r_core = 200.0
            # output scaling so A components are in realistic magnitude range
            self.u_scale = 1e-6
            self.v_scale = 1e-6
            # coupling / gauge penalties
            self.gauge_penalty = 1.0
            self.phi_ic_penalty = 1.0
            self.t_max = 1.0 / float(self.frequency_hz)


    args = ARGS()
    train(args)