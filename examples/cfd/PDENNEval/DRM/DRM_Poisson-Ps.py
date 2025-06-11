import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import *
import os
import argparse
from onescience.utils.pdenneval.dfvm_GenerateData import *


# Parser
parser = argparse.ArgumentParser(description='DFVM')
parser.add_argument('--dimension', type=int, default=100, metavar='N',
                    help='dimension of the problem (default: 100)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--beta', type=int, default=1000, metavar='N',
                    help='weight of boundary loss (default: 1000)')
seed = parser.parse_args().seed
# Omega space domain
DIMENSION = parser.parse_args().dimension     # Dimension
a = [ 0 for _ in range(DIMENSION)]
b = [ 1 for _ in range(DIMENSION)]
# Network
DIM_INPUT  = DIMENSION   # Input dimension
NUM_UNIT   = 40          # Number of neurons per layer
DIM_OUTPUT = 1           # Output dimension
NUM_LAYERS = 6           # Number of layers in the model
# Optimizer
IS_DECAY   = 0
LEARN_RATE         = 1e-3    # Learning rate
LEARN_FREQUENCY    = 50     # Learning rate change interval
LEARN_LOWWER_BOUND = 1e-5
LEARN_DECAY_RATE   = 0.99
LOSS_FN            = nn.MSELoss()
# Training
CUDA_ORDER = "0"
NUM_TRAIN_SMAPLE   = 10000    # Training set size
NUM_TRAIN_TIMES    = 1       # Number of training samples
NUM_ITERATION      = 100000  # Number of iterations per sample
# Re-sampling
IS_RESAMPLE = 0
SAMPLE_FREQUENCY   = 2000     # Resampling interval
# Testing
NUM_TEST_SAMPLE    = 10000
TEST_FREQUENCY     = 1     # Output interval
# Loss weight
BETA = 1000                  # Weight of boundary loss function
# Save model
IS_SAVE_MODEL = 1



class PossionQuation(object):
    def __init__(self, dimension, device):
        self.D      = dimension
        self.device = device

    def f(self, X):
        f = -2 * torch.ones(len(X), 1).to(self.device)
        return f.detach()

    def g(self, X):
        x = X[:,0]
        u = torch.where(x<0.5, x.pow(2), (x-1).pow(2))
        return u.reshape(-1,1).detach()

    def u_exact(self, X):
        x = X[:,0]
        u = torch.where(x<0.5, x.pow(2), (x-1).pow(2))
        return u.reshape(-1,1).detach()

    # Sampling inside the region
    def interior(self, N=100):
        eps = np.spacing(1)
        l_bounds = [l+eps for l in a]
        u_bounds = [u-eps for u in b]
        X = torch.FloatTensor( sampleCubeMC(self.D, l_bounds, u_bounds, N) )
        return X.requires_grad_(True).to(self.device)

    # Boundary sampling
    def boundary(self, n=100):
        x_boundary = []
        for i in range( self.D ):
            x = np.random.uniform(a[i], b[i], [2*n, self.D]) 
            x[:n,i] = b[i]
            x[n:,i] = a[i]
            x_boundary.append(x)
        x_boundary = np.concatenate(x_boundary, axis=0)
        x_boundary = torch.FloatTensor(x_boundary).requires_grad_(True).to(self.device)
        return x_boundary


# Boundary loss function
def loss_boundary(Eq, model, x_boundary):
    # x_boundary = Eq.boundary(100)
    u_theta    = model(x_boundary).reshape(-1,1)
    u_bd       = Eq.g(x_boundary).reshape(-1,1)
    loss_bd    = LOSS_FN(u_theta, u_bd) 
    return loss_bd

# Test function
def TEST(Eq, model, NUM_TESTING):
    with torch.no_grad():
        x_test = torch.Tensor(NUM_TESTING, Eq.D).uniform_(a[0], b[0]).requires_grad_(True).to(Eq.device)
        begin  = time.time()
        u_real = Eq.u_exact(x_test).reshape(1,-1)
        end    = time.time()
        u_pred = model(x_test).reshape(1,-1)
        Error  =  u_real - u_pred
        L2error  = torch.sqrt( torch.mean(Error*Error) )/ torch.sqrt( torch.mean(u_real*u_real) )
        MaxError = torch.max(torch.abs(Error))
    return L2error.cpu().detach().numpy(), MaxError.cpu().detach().numpy(), end-begin


class MLP(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_width=40):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x.to(torch.float32))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_pipeline():
    # define device
    DEVICE = torch.device(f"cuda:{CUDA_ORDER}" if torch.cuda.is_available() else "cpu")
    print(f"Currently using {DEVICE}")
    # define equation
    Eq = PossionQuation(DIMENSION, DEVICE)
    # define model
    # torch.set_default_dtype(torch.float64)
    model = MLP(DIM_INPUT, DIM_OUTPUT, NUM_UNIT).to(DEVICE) # .double()
    optA   = torch.optim.Adam(model.parameters(), lr=LEARN_RATE) 

    x = Eq.interior(NUM_TRAIN_SMAPLE)
    fTerm    = Eq.f(x).detach().reshape(-1,1)
    x_boundary = Eq.boundary(100)

    # Network iteration
    elapsed_time     = 0    # Timing
    training_history = []  

    for step in tqdm(range(NUM_ITERATION+1)):
        if IS_DECAY and step and step % LEARN_FREQUENCY == 0:
            for p in optA.param_groups:
                if p['lr'] > LEARN_LOWWER_BOUND:
                    p['lr'] = p['lr']*LEARN_DECAY_RATE
                    print(f"Learning Rate: {p['lr']}")
        # if IS_RESAMPLE and step and step % SAMPLE_FREQUENCY == 0:
        #     x = Eq.interior(NUM_TRAIN_SMAPLE)
        #     fTerm    = Eq.f(x).detach().reshape(-1,1)
        
        start_time = time.time()
        # Forward pass: compute predicted y by passing x to the model.
        u_pred   = model(x)
        du       = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
        loss_int = torch.mean( 0.5*torch.sum(du*du,1).unsqueeze(1)-fTerm*u_pred )
        loss_bd  = loss_boundary(Eq, model, x_boundary)
        loss     = loss_int + BETA*loss_bd
        optA.zero_grad()
        loss.backward()
        optA.step()

        epoch_time = time.time() - start_time
        elapsed_time = elapsed_time + epoch_time
        if step % TEST_FREQUENCY == 0:
                loss_int     = loss_int.cpu().detach().numpy()
                loss_bd      = loss_bd.cpu().detach().numpy()
                loss         = loss.cpu().detach().numpy()
                L2error,ME,T = TEST(Eq, model, NUM_TEST_SAMPLE)
                if step and step % 1000 == 0:
                    tqdm.write( f'\nStep: {step:>5}, '
                                f'Loss_r: {loss_int:>10.5f}, '
                                f'Loss_b: {loss_bd:>10.5f}, '
                                f'Loss: {loss:>10.5f}, '                                     
                                f'L2 error: {L2error:.5f}, '                                     
                                f'Time: {elapsed_time:.2f}')
                training_history.append([step, L2error, ME, loss, elapsed_time, epoch_time, T])

    training_history = np.array(training_history)
    print(np.min(training_history[:,1]))
    print(np.min(training_history[:,2]))

    save_time = time.localtime()
    save_time = f'[{save_time.tm_mday:0>2d}{save_time.tm_hour:0>2d}{save_time.tm_min:0>2d}]'
    dir_path  = os.getcwd() + f'/PossionEQ_seed{seed}/'
    file_name = f'{DIMENSION}DIM-DRM-{NUM_ITERATION}itr-{SAMPLE_FREQUENCY}N-.csv'
    file_path = dir_path + file_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savetxt(file_path, training_history,
                delimiter =",",
                header    ="step, L2error, MaxError, loss, elapsed_time, epoch_time, inference_time",
                comments  ='')
    print('Training History Saved!')

    if IS_SAVE_MODEL:
        torch.save(model.state_dict(), dir_path + f'{DIMENSION}DIM-DRM_net')
        print('DRM Network Saved!')


if __name__ == "__main__":
    setup_seed(seed)
    train_pipeline()
