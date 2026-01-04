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
r_0 = 0.01 # wire radius

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def PDE(Az, x_f, y_f):
    return d(d(Az, x_f), x_f) + d(d(Az, y_f), y_f) + mu * I_0

def Az_boundary(x, y):
        r = torch.sqrt(x**2 + y**2)
        return -(mu * I_0 / (2 * np.pi)) * torch.log(r / r_0)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        r_max = min(args.x_right, -args.x_left, args.y_right, -args.y_left)
        theta = 2.0 * torch.pi * torch.rand(size=(args.n_f, 1), dtype=torch.float)
        # sqrt for uniform sampling over disk area
        r = r_max * torch.sqrt(torch.rand(size=(args.n_f, 1), dtype=torch.float))
        x_Az = (r * torch.cos(theta)).requires_grad_(True)
        y_Az = (r * torch.sin(theta)).requires_grad_(True)

        Az = PINN(torch.cat([x_Az, y_Az], dim=1))
        PDE_ = PDE(Az, x_Az, y_Az)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # outer boundary
        theta = 2.0 * torch.pi * torch.rand(size=(args.n_b_l, 1), dtype=torch.float)
        x_bc = 0.1 * torch.cos(theta).requires_grad_(True)
        y_bc = 0.1 * torch.sin(theta).requires_grad_(True)
        Az_bc_pred = PINN(torch.cat([x_bc, y_bc], dim=1))
        # mse_BC_inner = args.criterion(Az_bc_pred, torch.zeros_like(Az_bc_pred))
        mse_BC = args.criterion(Az_bc_pred, Az_boundary(x_bc, y_bc))

        # initial condition
        Az_0 = PINN(torch.cat([x_Az, y_Az], dim=1))
        mse_IC = args.criterion(Az_0, torch.zeros_like(Az_0))

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
        # print weights of the network
        # print(PINN.features.keys())
        # for key in PINN.features.keys():
        #     print(key)
        #     print(min(PINN.features[key].weight[0]), max(PINN.features[key].weight[0]))
        #     print(min(PINN.features[key].bias), max(PINN.features[key].bias))

    # plot Az in 2D space (DC solution, no time dependence)
    plot_Az(PINN, args, loss_history)

    return loss_history

def plot_Az(PINN, args, loss_history, grid_n: int = 125, step: int = 5, amplifying_factor: float = 7000.0):
    """Plot Az + B field (DC solution, no time dependence)."""
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 1.2])

    xx = torch.linspace(args.x_left, args.x_right, grid_n)
    yy = torch.linspace(args.y_left, args.y_right, grid_n)
    xg, yg = torch.meshgrid([xx, yy], indexing="ij")
    s1 = xg.shape

    ax_az = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    x1 = xg.reshape((-1, 1)).clone().detach().requires_grad_(True)
    y1 = yg.reshape((-1, 1)).clone().detach().requires_grad_(True)

    x = torch.cat([x1, y1], dim=1)
    Az = PINN(x)

    Az_out = Az.reshape(s1)
    out = Az_out.detach().cpu().T.numpy()[::-1, :]
    im = ax_az.imshow(out, cmap="jet")
    fig.colorbar(im, ax=ax_az, fraction=0.046, pad=0.04)
    ax_az.set_xticks([])
    ax_az.set_yticks([])
    ax_az.set_xlabel("x")
    ax_az.set_ylabel("y")
    ax_az.set_title("Az")

    dAz_dy = d(Az, y1) * amplifying_factor
    dAz_dx = d(Az, x1) * amplifying_factor
    Bx = dAz_dy.reshape(s1)
    By = (-dAz_dx).reshape(s1)
    Xg = x1.reshape(s1).detach().cpu().numpy()
    Yg = y1.reshape(s1).detach().cpu().numpy()
    U = Bx.detach().cpu().numpy()
    V = By.detach().cpu().numpy()

    C = np.hypot(U[::step, ::step], V[::step, ::step])
    Q = ax_b.quiver(
        Xg[::step, ::step],
        Yg[::step, ::step],
        U[::step, ::step],
        V[::step, ::step],
        C,
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(Q, ax=ax_b, fraction=0.046, pad=0.04)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("y")
    ax_b.set_title("Magnetic field B")

    ax_loss = fig.add_subplot(gs[1, :])
    loss_arr = np.asarray(loss_history)
    ax_loss.plot(loss_arr[:, 0], label="PDE loss")
    ax_loss.plot(loss_arr[:, 1], label="BC loss")
    ax_loss.plot(loss_arr[:, 2], label="IC loss")
    ax_loss.plot(loss_arr[:, 3], label="Total loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_title("Loss history")
    ax_loss.legend()

    fig.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [2, 100, 100, 100, 100, 100, 100, 1]
            self.epochs = 400
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


    args = ARGS()
    train(args)
    