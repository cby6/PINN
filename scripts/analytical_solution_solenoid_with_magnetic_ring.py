import numpy as np
import matplotlib.pyplot as plt

# NOTE:
# This is a semi-analytical baseline for the updated PINN problem with a ferrite ring.
# A fully analytical solution for variable μ(r,z) is non-trivial; here we:
# 1) Approximate the coils as many circular current loops (Biot–Savart, numeric integration).
# 2) Optionally apply a simple local scaling inside the ferrite region: B ≈ μ_r(r,z) * B_free,
#    A ≈ μ_r(r,z) * A_free. This is only a heuristic baseline.

mu0 = 4 * np.pi * 1e-7  # vacuum permeability
epsilon = 8.854e-12  # vacuum permittivity
I_0 = 1.0  # current amplitude
f_0 = 50.0  # frequency (Hz)
c = 1.0 / np.sqrt(mu0 * epsilon)  # speed of light


def I(t: float) -> float:
    return float(I_0 * np.sin(2.0 * np.pi * f_0 * t))


def _sigmoid(x: np.ndarray | float, sharpness: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-sharpness * x))


def mu_r_field(r: np.ndarray | float, z: np.ndarray | float, *, sharpness: float = 400.0) -> np.ndarray:
    """Relative permeability μ_r(r,z) for the MnZn ferrite ring.

    Matches the updated PINN geometry:
      - Ring: 0.02 <= r <= 0.04
      - Axial extent: same as coils (two segments)
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    core_r_inner = 0.02
    core_r_outer = 0.04
    mu_r_core = 200.0

    z1_left, z1_right = -0.1, -0.05
    z2_left, z2_right = 0.05, 0.1

    r_in = _sigmoid(r - core_r_inner, sharpness)
    r_out = _sigmoid(core_r_outer - r, sharpness)
    radial = r_in * r_out

    z1 = _sigmoid(z - z1_left, sharpness) * _sigmoid(z1_right - z, sharpness)
    z2 = _sigmoid(z - z2_left, sharpness) * _sigmoid(z2_right - z, sharpness)
    axial = z1 + z2

    mask = radial * axial
    return 1.0 + (mu_r_core - 1.0) * mask


def loop_Aphi_Br_Bz(
    r: float,
    z: float,
    a: float,
    z0: float,
    current: float,
    *,
    n_phi: int = 720,
) -> tuple[float, float, float]:
    """Compute (A_phi, B_r, B_z) from a single circular loop by numerical Biot–Savart.

    Assumes observation point lies on the x-axis (y=0) in cylindrical coordinates (r,z).
    By axisymmetry, this defines the field in the full (r,z) plane.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi

    # loop position (a cosφ, a sinφ, z0)
    cx = a * np.cos(phi)
    cy = a * np.sin(phi)

    # dl = (-a sinφ, a cosφ, 0) dφ
    dlx = -a * np.sin(phi) * dphi
    dly = a * np.cos(phi) * dphi

    # observation point at (r, 0, z)
    Rx = r - cx
    Ry = -cy
    Rz = z - z0
    R2 = Rx * Rx + Ry * Ry + Rz * Rz
    R = np.sqrt(R2)
    R3 = R2 * R + 1e-30

    # On the x-axis, e_phi points +y, so A_phi == A_y
    Ay = (mu0 * current / (4.0 * np.pi)) * np.sum(dly / (R + 1e-30))

    # dl × R
    cxr_x = dly * Rz
    cxr_z = dlx * Ry - dly * Rx

    pref = mu0 * current / (4.0 * np.pi)
    Bx = pref * np.sum(cxr_x / R3)
    Bz = pref * np.sum(cxr_z / R3)

    # On x-axis, B_r = Bx
    return float(Ay), float(Bx), float(Bz)


def solenoid_Aphi_Br_Bz(
    r: float,
    z: float,
    t: float,
    *,
    r_coil: float = 0.04,
    z1_left: float = -0.1,
    z1_right: float = -0.05,
    z2_left: float = 0.05,
    z2_right: float = 0.1,
    loops_per_segment: int = 120,
    turns_per_segment: float = 50.0,
    include_core_scaling: bool = True,
) -> tuple[float, float, float]:
    """Semi-analytical solenoid approximation using many current loops.

    Each coil segment is approximated as `loops_per_segment` discrete loops carrying
    current I(t) * (turns_per_segment / loops_per_segment).

    If `include_core_scaling=True`, apply local scaling:
      B ≈ μ_r(r,z) * B_free, A ≈ μ_r(r,z) * A_free
    """
    current = I(float(t))
    z1 = np.linspace(z1_left, z1_right, loops_per_segment)
    z2 = np.linspace(z2_left, z2_right, loops_per_segment)
    z_all = np.concatenate([z1, z2])
    w = float(turns_per_segment / loops_per_segment)

    Aphi = 0.0
    Br = 0.0
    Bz = 0.0
    for z0 in z_all:
        a, br, bz = loop_Aphi_Br_Bz(r, z, r_coil, float(z0), current * w)
        Aphi += a
        Br += br
        Bz += bz

    if include_core_scaling:
        mur = float(mu_r_field(r, z))
        Aphi *= mur
        Br *= mur
        Bz *= mur

    return Aphi, Br, Bz


if __name__ == "__main__":
    # Example point inside ferrite ring and within coil 2 z-range
    r_pt = 0.03
    z_pt = 0.075

    t_values = np.linspace(0.0, 0.1, 200)
    A_values = []
    Bmag_values = []
    for tt in t_values:
        Aphi, Br, Bz = solenoid_Aphi_Br_Bz(r_pt, z_pt, float(tt))
        A_values.append(Aphi)
        Bmag_values.append(np.hypot(Br, Bz))
    # print deciles of A_values and Bmag_values
    print(np.percentile(A_values, [0, 25, 50, 75, 100]))
    print(np.percentile(Bmag_values, [0, 25, 50, 75, 100]))
    exit()

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    ax.plot(t_values, A_values, label=r"$A_\theta(r,z,t)$ (approx)")
    ax.plot(t_values, Bmag_values, label=r"$|B(r,z,t)|$ (approx)")
    ax.set_xlabel("t (s)")
    ax.set_title(
        f"Approx solenoid+ring at r={r_pt:.3f}, z={z_pt:.3f} (μ_r≈{mu_r_field(r_pt,z_pt):.1f})"
    )
    ax.legend()
    plt.show()
