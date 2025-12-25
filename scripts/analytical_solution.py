import numpy as np
import matplotlib.pyplot as plt

mu = 4 * np.pi * 1e-7 # vacuum permeability
epsilon = 8.854e-12 # vacuum permittivity
I_0 = 1 # current amplitude
tau = 0.1e-9 # pulse width
f_0 = 50 # frequency
c = 1.0 / np.sqrt(mu * epsilon) # speed of light
r_0 = 0.01 # reference radius


def I(t):
    return I_0 * np.sin(2 * np.pi * f_0 * t)

def Az(x, y, t):
    r = np.sqrt(x**2 + y**2)
    return mu * I(t) / (2 * np.pi) * np.log(r / r_0)

def B(x, y, t):
    r = np.sqrt(x**2 + y**2)
    return mu * I(t) / (2 * np.pi * r)

if __name__ == "__main__":
    x = 0.02
    y = 0.02
    t = 0.05
    # fix x and y, plot Az vs t
    t_values = np.linspace(0, 0.1, 100)
    Az_values = [Az(x, y, t) for t in t_values]
    B_values = [B(x, y, t) for t in t_values]
    # plot Az and B vs t
    plt.plot(t_values, Az_values, label='Az')
    plt.plot(t_values, B_values, label='B')
    plt.legend()
    plt.show()
