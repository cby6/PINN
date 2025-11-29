import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import grad
from PINN.networks import Net

mu = 4 * np.pi * 1e-7 # vacuum permeability
epsilon = 8.854e-12 # vacuum permittivity
I_0 = 1.0 # current amplitude
tau = 0.1e-9 # pulse width
c = 1.0 / np.sqrt(mu * epsilon) # speed of light
beta = 50
r_0 = 1.0 # reference radius

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def I(t):
    return 2 * I_0 / np.pi * torch.atan(beta * torch.tan(np.pi * t / tau))

def PDE(Az, x_f, y_f, t_f):
    return d(d(Az, x_f), x_f) + d(d(Az, y_f), y_f) - mu * epsilon * d(d(Az, t_f), t_f) + mu * I(t_f)

def Az_boundary(x, y, t):
        r = torch.sqrt(x**2 + y**2)
        return -(mu * I(t) / (2 * np.pi)) * torch.log(r / r_0)

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
        x_rand = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)).requires_grad_(True)

        y_rand = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
                (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)).requires_grad_(True)

        t_rand = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)).requires_grad_(True)

        xbc_l = (args.x_left * torch.ones_like(x_rand)).requires_grad_(True)
        xbc_r = (args.x_right * torch.ones_like(x_rand)).requires_grad_(True)
        ybc_l = (args.y_left * torch.ones_like(y_rand)).requires_grad_(True)
        ybc_r = (args.y_right * torch.ones_like(y_rand)).requires_grad_(True)

        # boundary_condition left
        BC_left = PINN(torch.cat([xbc_l, y_rand, t_rand], dim=1)) - Az_boundary(xbc_l, y_rand, t_rand)
        mse_BC_left = args.criterion(BC_left, torch.zeros_like(BC_left))
        
        # boundary_condition right
        BC_right = PINN(torch.cat([xbc_r, y_rand, t_rand], dim=1)) - Az_boundary(xbc_r, y_rand, t_rand)
        mse_BC_right = args.criterion(BC_right, torch.zeros_like(BC_right))

        # boundary_condition bottom
        BC_bottom = PINN(torch.cat([x_rand, ybc_l, t_rand], dim=1)) - Az_boundary(x_rand, ybc_l, t_rand)
        mse_BC_bottom = args.criterion(BC_bottom, torch.zeros_like(BC_bottom))

        # boundary_condition top
        BC_top = PINN(torch.cat([x_rand, ybc_r, t_rand], dim=1)) - Az_boundary(x_rand, ybc_r, t_rand)
        mse_BC_top = args.criterion(BC_top, torch.zeros_like(BC_top))

        mse_BC = mse_BC_left + mse_BC_right + mse_BC_bottom + mse_BC_top

        # initial condition
        t_0 = args.t_left * torch.ones_like(x_Az)
        Az_0 = PINN(torch.cat([x_Az, y_Az, t_0], dim=1)) - Az_boundary(x_Az, y_Az, t_0)
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

    # plot Az in 2D space at t = 0.5
    time = 0.3
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    xx = torch.linspace(-1, 1, 400).cpu()
    yy = torch.linspace(-1, 1, 400).cpu()
    x1, y1 = torch.meshgrid([xx, yy], indexing="ij")
    s1 = x1.shape
    x1 = x1.reshape((-1, 1)).requires_grad_(True)
    y1 = y1.reshape((-1, 1)).requires_grad_(True)
    t1 = (time * torch.ones_like(x1)).requires_grad_(True)
    x = torch.cat([x1, y1, t1], dim=1)
    Az = PINN(x)
    Az_out = Az.reshape(s1)
    out = Az_out.cpu().T.detach().numpy()[::-1, :]
    im1 = ax[0].imshow(out, cmap='jet')
    plt.colorbar(im1, ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Az')

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
    ax[1].set_title('Magnetic field B (t=0.5)')


    ax[2].plot(loss_history)
    ax[2].set_yscale('log')
    ax[2].legend(('PDE loss', 'BC loss', 'Total loss'))

    # plt.savefig('./result/loss.png')
    plt.show()

    return loss_history

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [3, 50, 50, 50, 50, 50, 50, 1]
            self.epochs = 900
            self.n_f = 20000
            self.n_f_1 = 10000
            self.n_f_2 = 10000
            self.n_b_l = 5000
            self.PDE_panelty = 1.0
            self.BC_panelty = 1.0
            self.IC_panelty = 1.0
            self.lr = 0.001
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam
            self.activation = torch.tanh
            self.activ_name = 'tanh'
            self.x_left = -1
            self.x_right = 1
            self.y_left = -1
            self.y_right = 1
            self.t_left = 0
            self.t_right = 1

    args = ARGS()
    train(args)
    