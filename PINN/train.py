import torch
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import grad
from networks import Net

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def PDE(u, x_f, y_f):
    return d(d(u, x_f), x_f) + d(d(u, y_f), y_f) + 1


def is_neumann_boundary_x(u, x, y):
    return d(u, x)

def is_neumann_boundary_y(u, x, y):
    return d(u, y)

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
        x_f = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        y_f = ((args.y_left + args.y_right) / 2 + (args.y_right - args.y_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float) - 0.5)
               ).requires_grad_(True)

        u_f = PINN(torch.cat([x_f, y_f], dim=1))
        PDE_ = PDE(u_f, x_f, y_f)
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
        # 局部采点
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
        xbc_l = (args.x_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        xbc_r = (args.x_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        ybc_l = (args.y_left * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)
        ybc_r = (args.y_right * torch.ones_like(x_rand_1)
                 ).requires_grad_(True)

        # is_neumann_boundary  下
        u_b_1 = (PINN(torch.cat([x_rand_1, ybc_l], dim=1)))
        BC_1 = is_neumann_boundary_y(u_b_1, x_rand_1, ybc_l)
        mse_BC_1 = args.criterion(BC_1, torch.zeros_like(BC_1))

        # is_neumann_boundary上
        u_b_2 = PINN(torch.cat([x_rand_2, ybc_r], dim=1))
        BC_2 = is_neumann_boundary_y(u_b_2, x_rand_2, ybc_r)
        mse_BC_2 = args.criterion(BC_2, torch.zeros_like(BC_2))

        # is_dirichlet_boundary左
        u_b_3 = PINN(torch.cat([xbc_l, y_rand_1], dim=1))
        mse_BC_3 = args.criterion(u_b_3, torch.zeros_like(u_b_3))

        # is_dirichlet_boundary右
        u_b_4 = PINN(torch.cat([xbc_r, y_rand_2], dim=1))
        mse_BC_4 = args.criterion(u_b_4, torch.zeros_like(u_b_4))

        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3 + mse_BC_4
        # loss
        loss = args.PDE_panelty * mse_PDE + args.BC_panelty * mse_BC
        loss_history.append([mse_PDE.item(), mse_BC.item(), loss.item()])
        if epoch % 10 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    class ARGS():
        def __init__(self):
            self.seq_net = [2, 50, 50, 50, 50, 50, 50, 1]
            self.epochs = 60000
            self.n_f = 20000
            self.n_f_1 = 10000
            self.n_f_2 = 10000
            self.n_b_l = 5000
            self.PDE_panelty = 1.0
            self.BC_panelty = 1.0
            self.lr = 0.001
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam
            self.activation = torch.tanh
            self.activ_name = 'tanh'
            self.x_left = 1
            self.x_right = 1
            self.y_left = -1
            self.y_right = 1

    args = ARGS()
    train(args)