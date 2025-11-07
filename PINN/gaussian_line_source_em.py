import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import grad
from networks import Net


mu = 4 * np.pi * 1e-7 # vacuum permeability
epsilon = 8.854e-12 # vacuum permittivity
I_0 = 1.0 # current amplitude
tau = 0.1e-9 # pulse width
c = 1.0 / np.sqrt(mu * epsilon) # speed of light

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def PDE(Ez, x_f, y_f, t_f):
    return d(d(Ez, x_f), x_f) + d(d(Ez, y_f), y_f) - mu * epsilon * d(d(Ez, t_f), t_f) + mu * I_0 * torch.exp(-t_f**2 / tau**2)

def train(args):
    setup_seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    PINN = Net(seq_net=args.seq_net, activation=args.activation)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(100):
        # for epoch in range(args.epochs):
        optimizer.zero_grad()
        # inside
        x_Ez = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        y_Ez = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        t_Ez = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        Ez = PINN(torch.cat([x_Ez, y_Ez, t_Ez], dim=1))
        PDE_ = PDE(Ez, x_Ez, y_Ez, t_Ez)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))

        # boundary

        x_rand_1 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        x_rand_2 = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)

        y_rand_1 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        y_rand_2 = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)

        t_rand_1 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)
        t_rand_2 = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                    (torch.rand(size=(args.n_b_l, 1), dtype=torch.float) - 0.5)
                    ).requires_grad_(True)

        xbc_l = (args.x_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        xbc_r = (args.x_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        ybc_l = (args.y_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)
        ybc_r = (args.y_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        # boundary_condition down
        Ez_b_down = PINN(torch.cat([x_rand_1, ybc_l, t_rand_1], dim=1))
        BC_down = d(Ez_b_down, y_rand_1) + 1 / c * d(Ez_b_down, t_rand_1)
        mse_BC_down = args.criterion(BC_down, torch.zeros_like(BC_down))

        # boundary_condition up
        Ez_b_up = PINN(torch.cat([x_rand_2, ybc_r, t_rand_2], dim=1))
        BC_up = d(Ez_b_up, y_rand_2) - 1 / c * d(Ez_b_up, t_rand_2)
        mse_BC_up = args.criterion(BC_up, torch.zeros_like(BC_up))

        # boundary_condition left
        Ez_b_left = PINN(torch.cat([xbc_l, y_rand_1, t_rand_1], dim=1))
        BC_left = d(Ez_b_left, x_rand_1) + 1 / c * d(Ez_b_left, t_rand_1)
        mse_BC_left = args.criterion(BC_left, torch.zeros_like(BC_left))

        # boundary_condition right
        Ez_b_right = PINN(torch.cat([xbc_r, y_rand_2, t_rand_2], dim=1))
        BC_right = d(Ez_b_right, x_rand_2) - 1 / c * d(Ez_b_right, t_rand_2)
        mse_BC_right = args.criterion(BC_right, torch.zeros_like(BC_right))

        mse_BC = mse_BC_down + mse_BC_up + mse_BC_left + mse_BC_right

        # initial condition
        Ez_0 = PINN(torch.cat([[0], [0], [0]], dim=1))
        mse_IC = args.criterion(Ez_0, torch.zeros_like(Ez_0))

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

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [3, 50, 50, 50, 50, 50, 50, 1]
            self.epochs = 60000
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