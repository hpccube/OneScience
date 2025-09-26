import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import *

from onescience.models.mlp import FullyConnected


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train High Dimension Poisson Equation using WAN",
        fromfile_prefix_chars="+",
    )
    parser.add_argument(
        "--s", type=int, default=12160, help="The random seed")
    parser.add_argument(
        "--d", type=int, default=1, help="The dimension of space")
    parser.add_argument(
        "--co",
        type=str,
        default="0",
        help="The number of gpu used",
    )
    parser.add_argument(
        "--i", type=int, default=2000, help="The iteration number")
    parser.add_argument(
        "--b", type=int, default=1000, help="The beta number")
    return parser.parse_args(arg_list)


def sampleCubeMC(dim, l_bounds, u_bounds, N=100):
    """Monte Carlo Sampling

    Get the sampling points by Monte Carlo Method.

    Args:
        dim:      The dimension of space
        l_bounds: The lower boundary
        u_bounds: The upper boundary
        N:        The number of sample points

    Returns:
        numpy.array: An array of sample points
    """
    sample = []
    for i in range(dim):
        sample.append(np.random.uniform(
            l_bounds[i], u_bounds[i], [N, 1]))
    data = np.concatenate(sample, axis=1)
    return data


args = get_arguments()
torch.manual_seed(args.s)
np.random.seed(args.s)
DIMENSION = args.d
a = [-1, 0]
b = [1, 1]
# Netword
DIM_INPUT = DIMENSION
NUM_UNIT = 30
DIM_OUTPUT = 1
NUM_BLOCKS = 4
# Optimizer
IS_DECAY = 0
LEARN_RATE = 1e-3
LEARN_FREQUENCY = 50
LEARN_LOWWER_BOUND = 1e-5
LEARN_DECAY_RATE = 0.99
LOSS_FN = nn.MSELoss()
# Training
CUDA_ORDER = args.co
NUM_INT_SAMPLE = 2000
NUM_BD_SAMPLE = 50
NUM_TRAIN_TIMES = 1
NUM_ITERATION = args.i

# Testing
NUM_TEST_SAMPLE = 10000
TEST_FREQUENCY = 100
# Loss weight
BETA = args.b
# Save model
IS_SAVE_MODEL = 1


class AllenCahn1DEquation(object):
    def __init__(self, dimension, device):
        self.D = dimension
        self.device = device

    def f(self, X):
        f = torch.zeros([X.shape[0], 1])
        return f.reshape(-1, 1).detach().to(self.device)

    def interior(self, N=100):
        # a = [-1,-1]
        # b = [1,1]
        eps = np.spacing(1)
        l_bounds = [a[0] + eps, a[1] + eps]
        u_bounds = [b[0] - eps, b[1]]
        X = torch.FloatTensor(sampleCubeMC(
            self.D + 1, l_bounds, u_bounds, N))
        return X.requires_grad_(True).to(self.device)

    def pe_boundary(self, x_l, x_r, n=100):
        t_points = torch.rand([n, 1]) * (b[1] - a[1]) + a[1]
        xl_points = torch.ones_like(t_points) * x_l
        xr_points = torch.ones_like(t_points) * x_r
        xt_bd_l, xt_bd_r = torch.cat((xl_points, t_points), dim=1), torch.cat(
            (xr_points, t_points), dim=1
        )
        return xt_bd_l.requires_grad_(True).to(self.device), xt_bd_r.requires_grad_(
            True
        ).to(self.device)

    def init_boundary(self, dataset, n=100):
        with h5py.File(dataset, "r") as f:
            x = f["x-coordinate"]
            t = f["t-coordinate"][0]
            u = f["tensor"][0, 0, :]
            idx = np.random.choice(
                x.shape[0] - 1, n, replace=False)
            idx = np.sort(idx)
            x_init, t_init, ue_init = (
                x[idx].reshape(-1, 1),
                np.ones([n, 1]) * t,
                u[idx].reshape(-1, 1),
            )
            xt_init = torch.Tensor(
                np.concatenate([x_init, t_init], axis=1))
            ue_init = torch.Tensor(ue_init)
        return xt_init.requires_grad_(True).to(self.device), ue_init.to(self.device)

    def end_boundary(self, n=100):
        eps = np.spacing(1)
        l_bound = [a[0] + eps]
        b_bound = [b[0] - eps]
        X = torch.FloatTensor(sampleCubeMC(
            self.D, l_bound, b_bound, n))
        t_points = torch.ones([n, 1]) * b[1]
        xt_end = torch.cat([X, t_points], dim=1)
        return xt_end.requires_grad_(True).to(self.device)

    def generate_test(self, dataset, N=10000):
        with h5py.File(dataset, "r") as f:
            x = f["x-coordinate"][:].reshape(-1, 1)
            # take watch of the last timing
            t = f["t-coordinate"][:].reshape(-1, 1)
            x_mesh, t_mesh = np.meshgrid(x, t)
            xt = np.concatenate(
                [x_mesh.flatten()[:, None], t_mesh.flatten()[
                    :, None]], axis=1
            )
            idx = np.random.choice(
                xt.shape[0] - 1, N, replace=False)
            idx = np.sort(idx)
            xt = torch.Tensor(xt[idx, :]).to(self.device)
            u = f["tensor"][0, :, :].reshape(-1, 1)
            u = torch.Tensor(u[idx, :]).to(self.device)
        return xt, u


def fun_w(x, eq):
    I1 = 0.210987
    x_list = torch.split(x, 1, 1)
    up = 1.0
    low = -1.0
    # **************************************************
    x_scale_list = []

    h_len = (up - low) / 2.0
    for i in range(eq.D):
        x_scale = (x_list[i] - low - h_len) / h_len
        x_scale_list.append(x_scale)
    # ************************************************
    z_x_list = []
    for i in range(eq.D):
        supp_x = torch.greater(
            1 - torch.abs(x_scale_list[i]), 0)
        z_x = torch.where(
            supp_x,
            torch.exp(
                1 / (torch.pow(x_scale_list[i], 2) - 1)) / I1,
            torch.zeros_like(x_scale_list[i]),
        )
        z_x_list.append(z_x)
    # ***************************************************
    w_val = 1
    for i in range(eq.D):
        w_val = w_val * z_x_list[i]
    dw = torch.autograd.grad(
        w_val,
        x,
        grad_outputs=torch.ones_like(w_val),
        create_graph=True,
        retain_graph=True,
    )[0]
    dw = torch.where(torch.isnan(
        dw), torch.zeros_like(dw), dw)
    return (w_val, dw)


def grad_u(x, model):
    fun_u = model(x)
    grad_u = torch.autograd.grad(
        fun_u,
        x,
        grad_outputs=torch.ones_like(fun_u),
        create_graph=True,
        retain_graph=True,
    )[0]
    du_x, du_t = grad_u[:, :1], grad_u[:, -1:]
    return fun_u, du_x, du_t


def loss(
    eq, model_u, model_v, xt_int, xt_bd_l, xt_bd_r, xt_init, xt_end, ue_init, beta
):
    # domain squares size
    sp_s = 2  # spatial size
    te_s = 1  # temporal size
    st_s = sp_s * te_s  # time-space size
    # compute some basic gradient values
    u, dux, dut = grad_u(xt_int, model_u)
    v, dvx, dvt = grad_u(xt_int, model_v)
    w, wx = fun_w(xt_int[:, :-1], eq)
    # compute time terms
    # u(x,T) * \phi(x,T)
    w_end, _ = fun_w(xt_end[:, :-1], eq)
    u_end, v_end = model_u(xt_end), model_v(xt_end)
    int_l1 = sp_s * torch.mean(u_end * (w_end * v_end))
    # u(x,0) * \phi(x,0)
    w_init, _ = fun_w(xt_init[:, :-1], eq)
    v_init = model_v(xt_init)
    int_r1 = sp_s * torch.mean(ue_init * (w_init * v_init))
    # u(x,t) * \phi(x,t)_t = u(x,t) * w(x) * v(x,t)_t
    int_r2 = st_s * torch.mean(u * (w * dvt))
    # compute space terms
    int_l2 = 5 * st_s * torch.mean(u**3 * (w * v))
    int_r3 = 5 * st_s * torch.mean(u * (w * v))
    int_l3 = 0.0001 * st_s * \
        torch.mean(dux * (wx * v + w * dvx))
    # compute the L2 norm of the test function
    norm = st_s * torch.mean((w * v) ** 2)
    # loss interior
    loss_int = torch.pow(
        int_l1 + int_l2 + int_l3 - int_r1 - int_r2 - int_r3, 2) / norm

    # loss_bd, loss_init
    # periodic bd
    # u_bd_l, u_bd_r = model_u(xt_bd_l), model_u(xt_bd_r)
    u_bd_l, dux_l, _ = grad_u(xt_bd_l, model_u)
    u_bd_r, dux_r, _ = grad_u(xt_bd_r, model_u)
    loss_bd = LOSS_FN(u_bd_l, u_bd_r) + \
        LOSS_FN(dux_l, dux_r)

    # initial bd
    u_init = model_u(xt_init)
    loss_init = LOSS_FN(u_init, ue_init)

    # loss_u , loss_v
    loss_u = loss_int + beta * loss_bd + 100 * beta * loss_init
    loss_v = -torch.log(loss_int)
    return loss_u, loss_v


# Test function for Resulting model
def TEST(model, x_test, u_real):
    with torch.no_grad():
        u_pred = model(x_test)
        Error = u_real - u_pred
        L2error = torch.sqrt(
            torch.sum(Error * Error) / torch.sum(u_real * u_real))
        Maxerror = torch.max(torch.abs(Error))
    return L2error.cpu().detach().numpy(), Maxerror.cpu().detach().numpy()


def train_pipeline():
    # define device
    DEVICE = torch.device(
        f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    equation = "AC1"
    dataset = "../data/1D_Allen-Cahn_0.0001_5.hdf5"
    # define equation
    Eq = AllenCahn1DEquation(DIMENSION, DEVICE)
    # define model
    model_u = FullyConnected(
        in_features=args.d + 1,
        layer_size=64,
        out_features=5,
        num_layers=6,
        activation_fn="tanh",
    ).to(DEVICE)
    model_v = FullyConnected(
        in_features=args.d + 1,
        layer_size=64,
        out_features=5,
        num_layers=6,
        activation_fn="tanh",
    ).to(DEVICE)
    optu = torch.optim.Adam(model_u.parameters(), lr=0.0001)
    optv = torch.optim.Adam(model_v.parameters(), lr=0.001)

    # training data
    xt_int = Eq.interior(NUM_INT_SAMPLE)
    xt_bd_l, xt_bd_r = Eq.pe_boundary(a[0], b[0], n=500)
    xt_init, ue_init = Eq.init_boundary(dataset, n=500)
    xt_end = Eq.end_boundary(n=100)
    # testing data
    x_test, u_real = Eq.generate_test(dataset)

    elapsed_time = 0
    training_history = []
    epoch_list = []
    min_l2 = 100
    global loss_u, loss_v
    dir_path = os.getcwd() + f"/{equation}/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for step in tqdm(range(NUM_ITERATION + 1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optu.param_groups:
                if p["lr"] > LEARN_LOWWER_BOUND:
                    p["lr"] = p["lr"] * LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")
            for p in optv.param_groups:
                if p["lr"] > LEARN_LOWWER_BOUND:
                    p["lr"] = p["lr"] * LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")

        start_time = time.time()

        start_t = time.time()
        for _ in range(2):
            loss_u, loss_v = loss(
                Eq,
                model_u,
                model_v,
                xt_int,
                xt_bd_l,
                xt_bd_r,
                xt_init,
                xt_end,
                ue_init,
                BETA,
            )
            optv.zero_grad()
            loss_v.backward()
            optv.step()
        for _ in range(1):
            loss_u, loss_v = loss(
                Eq,
                model_u,
                model_v,
                xt_int,
                xt_bd_l,
                xt_bd_r,
                xt_init,
                xt_end,
                ue_init,
                BETA,
            )
            optu.zero_grad()
            loss_u.backward()
            optu.step()
        end_t = time.time() - start_t
        epoch_list.append(end_t)
        elapsed_time = elapsed_time + time.time() - start_time
        if step % TEST_FREQUENCY == 0:
            loss_u = loss_u.cpu().detach().numpy()
            loss_v = loss_v.cpu().detach().numpy()
            L2error, Maxerror = TEST(
                model_u, x_test, u_real)

            tqdm.write(
                f"Step: {step:>5} | "
                f"Loss_u: {loss_u:>12.5f} | "
                f"Loss_v: {loss_v:>10.5f} | "
                f"L2 error: {L2error:>7.5e} | "
                f"Max error: {Maxerror:>7.5e} |"
                f"Time: {elapsed_time:>7.2f} |"
            )
            training_history.append(
                [step, loss_u, loss_v, L2error,
                    Maxerror, elapsed_time]
            )
            if L2error < min_l2:
                min_l2 = L2error
                torch.save(
                    {
                        "model_u_state_dict": model_u.state_dict(),
                        "model_v_state_dict": model_v.state_dict(),
                    },
                    os.path.join(
                        dir_path, equation + "_check_point_me.pt"),
                )
                print("Save model min L2error!")
    training_history = np.array(training_history)
    print("l2r_min:", np.min(training_history[:, 3]))
    history = pd.DataFrame(training_history[:, 3])
    history.to_csv(os.path.join(
        dir_path, f"WAN-{equation}.csv"))
    loss_history = pd.DataFrame(training_history[:, 1])
    loss_history.to_csv(os.path.join(
        dir_path, f"WAN-{equation}-loss_history.csv"))

    epoch_list = np.array(epoch_list)

    np.savetxt(
        os.path.join(
            dir_path, f"epoch_time-{equation}.csv"),
        epoch_list,
        delimiter=",",
        header="epoch_time",
        comments="",
    )

    if IS_SAVE_MODEL:
        torch.save(model_u.state_dict(),
                   os.path.join(dir_path, "WAN-U_net"))
        print("Weak Adversarial Network Saved!")

    metric = []
    with torch.no_grad():
        model_me = torch.load(
            os.path.join(dir_path, equation + "_check_point_me.pt"), weights_only=True
        )
        model_u.load_state_dict(
            model_me["model_u_state_dict"])
        x_infer = Eq.interior(N=10000)
        start_time = time.time()
        _ = model_u(x_infer)
        end_time = time.time() - start_time
        metric.append(end_time)
        print(f"infer time: {end_time}")
        L2error, Maxerror = TEST(model_u, x_test, u_real)
        metric.append(L2error)
        metric.append(Maxerror)
        metric_list = np.array(metric)

        np.savetxt(
            os.path.join(
                dir_path, f"metric-{equation}.csv"),
            metric_list,
            delimiter=",",
            header="infer_time, L2error, Maxerror",
            comments="",
        )
        print(
            f"L2 error: {L2error}" f"Max error: {Maxerror}")
    return model_u


if __name__ == "__main__":
    model = train_pipeline()
    print("done")
