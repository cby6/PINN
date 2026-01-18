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
I_0 = 1 # current amplitude
# tau = 0.1e-9 # pulse width
f_0 = 50 # frequency
c = 1.0 / np.sqrt(mu * epsilon) # speed of light
r_0 = 0.01 # wire radius

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


# choose a small spatial width for localization
sigma_r = 0.1 # spatial width
# def S(x: torch.Tensor, y: torch.Tensor, sigma: float = sigma_r) -> torch.Tensor:
#     """Spatial Gaussian envelope centered at (0, 0)."""
#     r2 = x**2 + y**2
#     return torch.exp(-r2 / (2.0 * sigma**2))

def I(t):
    return I_0 * torch.sin(2 * torch.pi * f_0 * t)

def PDE(Az, x_f, y_f, t_f):
    # return d(d(Az, x_f), x_f) + d(d(Az, y_f), y_f) - mu * epsilon * d(d(Az, t_f), t_f) - mu * I(t_f) * S(x_f, y_f)
    return d(d(Az, x_f), x_f) + d(d(Az, y_f), y_f) - mu * epsilon * d(d(Az, t_f), t_f) - mu * I(t_f)

def Az_boundary(x, y, t):
    r = torch.sqrt(x**2 + y**2)
    return -(mu * I(t) / (2 * np.pi)) * torch.log(r / r_0) # * S(x, y)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # inside
        x_Az = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        y_Az = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        t_Az = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        Az = PINN(torch.cat([x_Az, y_Az, t_Az], dim=1))
        PDE_ = PDE(Az, x_Az, y_Az, t_Az)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # boundary

        # x_rand_1 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)
        # x_rand_2 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)

        # y_rand_1 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)
        # y_rand_2 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)

        # t_rand_1 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)
        # t_rand_2 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)
        # t_rand_3 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)
        # t_rand_4 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
        #             (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
        #             ).requires_grad_(True)

        # xbc_l = (args.x_left * torch.ones_like(x_rand_1)).requires_grad_(True)
        # xbc_r = (args.x_right * torch.ones_like(x_rand_1)).requires_grad_(True)
        # ybc_l = (args.y_left * torch.ones_like(y_rand_1)).requires_grad_(True)
        # ybc_r = (args.y_right * torch.ones_like(y_rand_1)).requires_grad_(True)

        # # boundary_condition left
        # BC_left = PINN(torch.cat([xbc_l, y_rand_1, t_rand_1], dim=1)) - Az_boundary(xbc_l, y_rand_1, t_rand_1)
        # mse_BC_left = args.criterion(BC_left, torch.zeros_like(BC_left))

        # # boundary_condition right
        # BC_right = PINN(torch.cat([xbc_r, y_rand_2, t_rand_2], dim=1)) - Az_boundary(xbc_r, y_rand_2, t_rand_2)
        # mse_BC_right = args.criterion(BC_right, torch.zeros_like(BC_right))

        # # boundary_condition down
        # BC_down = PINN(torch.cat([x_rand_1, ybc_l, t_rand_3], dim=1)) - Az_boundary(x_rand_1, ybc_l, t_rand_3)
        # mse_BC_down = args.criterion(BC_down, torch.zeros_like(BC_down))

        # # boundary_condition up
        # BC_up = PINN(torch.cat([x_rand_2, ybc_r, t_rand_4], dim=1)) - Az_boundary(x_rand_2, ybc_r, t_rand_4)
        # mse_BC_up = args.criterion(BC_up, torch.zeros_like(BC_up))

        # mse_BC = mse_BC_left + mse_BC_right + mse_BC_down + mse_BC_up

        # Sample angles uniformly on [0, 2π)
        theta = 2.0 * torch.pi * torch.rand(size=(args.n_b_l, 1), dtype=torch.float)

        # Circle of radius 0.01
        x_bc = 0.01 * torch.cos(theta).requires_grad_(True)
        y_bc = 0.01 * torch.sin(theta).requires_grad_(True)

        # Random times along the boundary
        t_bc = ((args.t_left + args.t_right) / 2
                + (args.t_right - args.t_left) * (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
            ).requires_grad_(True)

        # On r = 0.01 with r0 = 0.01, analytic Az is 0 → homogeneous Dirichlet BC
        Az_bc_pred = PINN(torch.cat([x_bc, y_bc, t_bc], dim=1))
        mse_BC = args.criterion(Az_bc_pred, torch.zeros_like(Az_bc_pred))

        # initial conditions: Az(x, y, 0) = 0 and ∂Az/∂t(x, y, 0) = 0
        t_0 = (args.t_left * torch.ones_like(x_Az)).requires_grad_(True)
        Az_0 = PINN(torch.cat([x_Az, y_Az, t_0], dim=1))
        # displacement IC
        mse_IC_disp = args.criterion(Az_0, torch.zeros_like(Az_0))
        # velocity IC: time derivative at t = t_left
        dAz_dt_0 = d(Az_0, t_0)
        mse_IC_vel = args.criterion(dAz_dt_0, torch.zeros_like(dAz_dt_0))
        mse_IC = mse_IC_disp + mse_IC_vel

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

    time = 0.05
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    xx = torch.linspace(args.x_left, args.x_right, 125).cpu()
    yy = torch.linspace(args.y_left, args.y_right, 125).cpu()
    x1, y1 = torch.meshgrid([xx, yy], indexing="ij")
    s1 = x1.shape
    x1 = x1.reshape((-1, 1)).requires_grad_(True)
    y1 = y1.reshape((-1, 1)).requires_grad_(True)
    t1 = (time * torch.ones_like(x1)).requires_grad_(True)
    x = torch.cat([x1, y1, t1], dim=1)
    Az = PINN(x)
    print(Az.shape)

    Az_out = Az.reshape(s1)
    out = Az_out.cpu().T.detach().numpy()[::-1, :]
    im1 = ax[0].imshow(out, cmap='jet')
    plt.colorbar(im1, ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title(f'Az (t={time})')

    amplifying_factor = 8000
    dAz_dy = d(Az, y1) * amplifying_factor
    dAz_dx = d(Az, x1) * amplifying_factor
    Bx = dAz_dy.reshape(s1)
    By = (-dAz_dx).reshape(s1)
    Xg = x1.reshape(s1).detach().cpu().numpy()
    Yg = y1.reshape(s1).detach().cpu().numpy()
    U = Bx.detach().cpu().numpy()
    V = By.detach().cpu().numpy()
    step = 5
    C = np.hypot(U[::step, ::step], V[::step, ::step])
    Q = ax[1].quiver(
        Xg[::step, ::step], Yg[::step, ::step],
        U[::step, ::step], V[::step, ::step],
        C, cmap='jet', angles='xy', scale_units='xy', scale=1.0, width=0.002
    )
    plt.colorbar(Q, ax=ax[1])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title(f'Magnetic field B (t={time})')


    ax[2].plot(loss_history)
    ax[2].set_yscale('log')
    ax[2].legend(('PDE loss', 'BC loss', 'Total loss'))

    # plt.savefig('./result/loss.png')
    plt.show()


if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [3, 50, 50, 50, 50, 50, 50, 1]
            self.epochs = 2000
            self.n_f = 10000
            # self.n_f_1 = 10000
            # self.n_f_2 = 10000
            self.n_b_l = 5000
            self.PDE_panelty = 1.0
            self.BC_panelty = 1.0
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
            self.t_left = 0
            self.t_right = 0.1

    args = ARGS()
    train(args)