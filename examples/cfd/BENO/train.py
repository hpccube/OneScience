import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pprint as pp
from timeit import default_timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import scatter
from torchvision.transforms import GaussianBlur
import sys, os
from onescience.utils.beno.utilities import *
from onescience.utils.beno.util import record_data, to_cpu, to_np_array, make_dir
from onescience.models.beno.BE_MPNN import HeteroGNS
import random
from loguru import logger
import matplotlib.tri as tri
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings('ignore')
fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--dataset_type', default="32x32", type=str,
                     help='dataset type default="32x32"')
parser.add_argument('--epochs', default=1000, type=int,
                    help='Epochs default=1000')
parser.add_argument('--lr', default=0.00001, type=float,
                    help='learning rate default=0.00001')
parser.add_argument('--inspect_interval', default=100, type=int,
                    help='inspect interval default=100')
parser.add_argument('--init_boudary_loc', default="regular", type=str,
                    help='choose from "random" or "regular" ')
parser.add_argument('--trans_layer', default=3, type=int,
                    help='Layer of Transformer')
parser.add_argument('--boundary_dim', default=128, type=int,
                    help='Layer of Transformer')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size default=1')
parser.add_argument('--act', default="relu", type=str,
                    help='activation choose from "relu","elu","leakyrelu","silu')
parser.add_argument('--nmlp_layers',default=2,type=int,
                    help='number of layers of GNS')
parser.add_argument('--ns',default=10,type=int,
                    help='number of the number of neighbor nodes')


if 'ipykernel' in sys.modules:
    is_jupyter = True
    try:
        # 在 Notebook 环境中，使用默认值或手动覆盖参数
        args = parser.parse_args([])
        # 手动覆盖参数
        args.batch_size = 1 #原始数据集中边界形状各不相同，内部点数量也各不相同，若batchsize不设为1，会导致大小不一致
        args.lr = 0.00005
        args.boundary_dim = 128
        args.act = 'silu'
        args.nmlp_layers = 3
        args.trans_layer = 3
        args.ns = 10
    except:
        pass 
else:
    is_jupyter = False
    args = parser.parse_args()
pp.pprint(args.__dict__)



# ## ===============================================================

# DATA_PATH = f"./data/Dirichlet/"
# f_all = np.load(DATA_PATH + "RHS_N32_mix_all.npy")
# sol_all = np.load(DATA_PATH + "SOL_N32_mix_all.npy")
# bc_all=np.load(DATA_PATH + "BC_N32_mix_all.npy")
# ntrain = 900
# ntest =100

## ===============================================================

# ## ===============================================================

# DATA_PATH = f"./data/Neumann/"
# f_all = np.load(DATA_PATH + "RHS_N32_mix_all.npy")
# sol_all = np.load(DATA_PATH + "SOL_N32_mix_all.npy")
# bc_all=np.load(DATA_PATH + "BC_N32_mix_all.npy")
# ntrain = 900
# ntest =100

## ===============================================================

DATA_PATH = f"./data/"
f_all = np.load(DATA_PATH + "RHS_N32_10.npy")
sol_all = np.load(DATA_PATH + "SOL_N32_10.npy")
bc_all=np.load(DATA_PATH + "BC_N32_10.npy")
ntrain = 7
ntest =3

# ===============================================================
gblur = GaussianBlur(kernel_size=5, sigma=5) 
batch_size = args.batch_size
batch_size2 = args.batch_size
width = 64
ker_width = 256
depth = 4
edge_features = 7
node_features = 10
ns=args.ns
epochs = args.epochs

learning_rate = args.lr
inspect_interval = args.inspect_interval

runtime = np.zeros(2, )
t1 = default_timer()

resolution = 32
s = resolution
n=s**2


trans_layer = args.trans_layer

path = 'Resolution_' + str(s) + '_poisson' + \
    '_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + '_Transformer_layer' + str(args.trans_layer) +\
    '_Rolling' + args.init_boudary_loc+'_ns'+str(args.ns)+\
    '_nheads2'+'_bddim'+str(args.boundary_dim)+"_act"+args.act+'lr'+str(args.lr)+'_nmlp_layers'+str(args.nmlp_layers)
path_model = './model/' + path
make_dir(path_model)


logger.add(os.path.join('log', '{}.log'.format(
            path)), rotation="500 MB", level="INFO")
logger.info(path)
   

coord_all=f_all[:,:,0:2]
bc_euco=bc_all[:,:,0:2]
bc_value=bc_all[:,:,2].reshape(-1,128,1)
bc_value=torch.tensor(bc_value)
bc_value_1=bc_value[0:ntrain,:,:]
bc_euco=torch.tensor(bc_euco)
bcv_normalizer = GaussianNormalizer(bc_value_1) #边界部分标准化
bc_value = bcv_normalizer.encode(bc_value)
bc_euco= to_np_array(torch.cat([bc_euco,bc_value],dim=-1))

all_a = f_all[:,:,2]
cells_state=f_all[:,:,3] # node type \in {0,1,2,3}

all_a_smooth = to_np_array(gblur(torch.tensor(all_a.reshape(all_a.shape[0], resolution, resolution))).flatten(start_dim=1))   
all_a_reshape = all_a_smooth.reshape(-1, resolution, resolution)
all_a_gradx = np.concatenate([
    all_a_reshape[:,1:2] - all_a_reshape[:,0:1],    #上边界梯度
    (all_a_reshape[:,2:] - all_a_reshape[:,:-2]) / 2, #中间部分梯度
    all_a_reshape[:,-1:] - all_a_reshape[:,-2:-1], #下边界梯度
], 1)
all_a_gradx = all_a_gradx.reshape(-1, n)
all_a_grady = np.concatenate([
    all_a_reshape[:,:,1:2] - all_a_reshape[:,:,0:1], #左边界梯度
    (all_a_reshape[:,:,2:] - all_a_reshape[:,:,:-2]) / 2,#中间部分梯度
    all_a_reshape[:,:,-1:] - all_a_reshape[:,:,-2:-1], #右边界梯度
], 2)
all_a_grady = all_a_grady.reshape(-1, n)
all_u = sol_all[:,:,0]

train_a = torch.FloatTensor(all_a[:ntrain]) 
train_a_smooth = torch.FloatTensor(all_a_smooth[:ntrain])
train_a_gradx = torch.FloatTensor(all_a_gradx[:ntrain])
train_a_grady = torch.FloatTensor(all_a_grady[:ntrain])
train_u = torch.FloatTensor(all_u[:ntrain])

test_a = torch.FloatTensor(all_a[ntrain:])
test_a_smooth = torch.FloatTensor(all_a_smooth[ntrain:])
test_a_gradx = torch.FloatTensor(all_a_gradx[ntrain:])
test_a_grady = torch.FloatTensor(all_a_grady[ntrain:])
test_u = torch.FloatTensor(all_u[ntrain:])

bc_euco_train=bc_euco[:ntrain,:,:]
bc_euco_test=bc_euco[ntrain:,:,:]

cells_state_train=cells_state[:ntrain,:]
cells_state_test=cells_state[ntrain:,:]

#* normalization
indomain_a = np.array([])
indomain_u = np.array([])
for j in range(ntrain):
    outdomain_idx=np.array([],dtype=int)
    indomain_idx=np.array([],dtype=int)
    for p in range(f_all.shape[1]): 
        if (cells_state[j][p]!=0):  # 判断节点是否为域外节点 0为域内节点
            outdomain_idx=np.append(outdomain_idx,int(p))
    indomain_idx = list(set([i for i in range(resolution*resolution)]) - set(list(outdomain_idx))) #得到域内节点
    indomain_u = np.append(indomain_u,sol_all[j][indomain_idx])  # 提取域内目标值 u
    indomain_a = np.append(indomain_a,f_all[j][indomain_idx][:,2])  # 提取域内特征 a

indomain_u=torch.tensor(indomain_u)                 
indomain_a=torch.tensor(indomain_a)


a_normalizer = GaussianNormalizer(indomain_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = GaussianNormalizer(x=indomain_u) 
train_u = u_normalizer.encode(train_u)


grid_input=f_all[-1,:,0:2] 
meshgenerator = MeshGenerator([[0,1],[0,1]],[s,s], grid_input = grid_input)  #根据输入的网格坐标 
data_train = []
for j in range(ntrain):
    #提取网格内与边界的节点索引
    mesh_idx_temp=[p for p in range(resolution**2)]  # 初始化所有节点编号和参数生成用于图神经网络训练的数据 
    outdomain_idx=np.array([])  
    for p in range(f_all.shape[1]): 
        if (cells_state[j][p]!=0):   # 非内域点（如边界点）
            outdomain_idx=np.append(outdomain_idx,p)
    for p in range(len(outdomain_idx)): # 保留内域点
            mesh_idx_temp.remove(outdomain_idx[p])

    dist2bd_x=np.array([0,0])[np.newaxis,:]
    dist2bd_y=np.array([0,0])[np.newaxis,:]
    for p in range(len(mesh_idx_temp)):
        indomain_x = coord_all[j][mesh_idx_temp[p]][0] #获取域内节点的x坐标
        indomain_y = coord_all[j][mesh_idx_temp[p]][1] #获取域内节点的y坐标
        
        horizon_bd_y = np.where(bc_euco_train[j,:,0].round(4) == indomain_x.round(4))[0]
        dist2bd_y_temp = np.array(
            [np.abs(bc_euco_train[j,horizon_bd_y[0],1] - indomain_y),
             np.abs(bc_euco_train[j,horizon_bd_y[1],1] - indomain_y)
            ]
        )
        dist2bd_y = np.vstack([dist2bd_y,dist2bd_y_temp[np.newaxis,:]])
        horizon_bd_x = np.where(bc_euco_train[j,:,1].round(4) == indomain_y.round(4))[0]
        dist2bd_x_temp = np.array(
            [np.abs(bc_euco_train[j,horizon_bd_x[0],0] - indomain_x),
             np.abs(bc_euco_train[j,horizon_bd_x[1],0] - indomain_x)
            ]
        )
        dist2bd_x = np.vstack([dist2bd_x,dist2bd_x_temp[np.newaxis,:]])
    dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
    dist2bd_x = torch.tensor(dist2bd_x[1:]).float() # [num, 2]

    
    idx = meshgenerator.sample(mesh_idx_temp)#构造图结构的数据
    grid = meshgenerator.get_grid()
    xx=to_np_array(grid[:,0])
    yy=to_np_array(grid[:,1])
    triang = tri.Triangulation(xx, yy)
    tri_edge = triang.edges    
    
    edge_index = meshgenerator.ball_connectivity(ns=10,tri_edge=tri_edge)
    edge_attr = meshgenerator.attributes(theta=train_a[j,:])
    cell_state_train_current = torch.FloatTensor(cells_state_train[j])
    
    train_x = torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                             train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                             train_a_grady[j, idx].reshape(-1, 1), dist2bd_x,dist2bd_y,
                            ], dim=1)
    train_x_2 = torch.cat([grid, torch.zeros([grid.shape[0],4]), dist2bd_x,dist2bd_y,
                            ], dim=1)
    bd_coord_input = torch.tensor(bc_euco_train[j])  
        
    bd_coord_input_1=bd_coord_input.clone()
    bd_coord_input_1[:,2]=0

    #将数据存储到 HeteroData 中
    data=HeteroData() 
    data['G1'].x=train_x #node features ▲u=f 
    data['G1'].boundary=bd_coord_input_1 #boundary value=0 
    data['G1'].edge_features=edge_attr
    data['G1'].sample_idx=idx
    data['G1'].edge_index=edge_index
    data['G1'].cell_state = cell_state_train_current
    
    data['G2'].x=train_x_2  ##node features ▲u=0
    data['G2'].boundary=bd_coord_input #boundary value=g(x)
    data['G2'].edge_features=edge_attr
    data['G2'].sample_idx=idx
    data['G2'].edge_index=edge_index
    
    #结合G1和G2
    data['G1+2'].y=train_u[j, idx]
    data_train.append(data)

data_test = []
for j in range(ntest):
    mesh_idx_temp=[p for p in range(resolution**2)]
    outdomain_idx=np.array([])
    for p in range(f_all.shape[1]): 
        if (cells_state[j+ntrain][p]!=0):  
            outdomain_idx=np.append(outdomain_idx,p)
        
    for p in range(len(outdomain_idx)):
            mesh_idx_temp.remove(outdomain_idx[p])     
    
    dist2bd_x=np.array([0,0])[np.newaxis,:]
    dist2bd_y=np.array([0,0])[np.newaxis,:]
    for p in range(len(mesh_idx_temp)):
        indomain_x = coord_all[j+ntrain][mesh_idx_temp[p]][0]
        indomain_y = coord_all[j+ntrain][mesh_idx_temp[p]][1]
        
        horizon_bd_y = np.where(bc_euco_test[j,:,0].round(4) == indomain_x.round(4))[0]
        
        dist2bd_y_temp = np.array(
            [np.abs(bc_euco_test[j,horizon_bd_y[0],1] - indomain_y),
             np.abs(bc_euco_test[j,horizon_bd_y[1],1] - indomain_y)
            ]
        )
        dist2bd_y = np.vstack([dist2bd_y,dist2bd_y_temp[np.newaxis,:]])
        horizon_bd_x = np.where(bc_euco_test[j,:,1].round(4) == indomain_y.round(4))[0]
       
        dist2bd_x_temp = np.array(
            [np.abs(bc_euco_test[j,horizon_bd_x[0],0] - indomain_x),
             np.abs(bc_euco_test[j,horizon_bd_x[1],0] - indomain_x)
            ]
        )
        dist2bd_x = np.vstack([dist2bd_x,dist2bd_x_temp[np.newaxis,:]])
    dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
    dist2bd_x = torch.tensor(dist2bd_x[1:]).float() # [num, 2]
    
    
    idx = meshgenerator.sample(mesh_idx_temp)
    grid = meshgenerator.get_grid()
    
    xx=to_np_array(grid[:,0])   
    yy=to_np_array(grid[:,1])
    triang = tri.Triangulation(xx, yy)
    tri_edge = triang.edges    

    edge_index = meshgenerator.ball_connectivity(ns=10,tri_edge=tri_edge)
    edge_attr = meshgenerator.attributes(theta=test_a[j,:])
    cell_state_test_current = torch.FloatTensor(cells_state_test[j])
    test_x = torch.cat([grid, test_a[j, idx].reshape(-1, 1),
                        test_a_smooth[j, idx].reshape(-1, 1), test_a_gradx[j, idx].reshape(-1, 1),
                        test_a_grady[j, idx].reshape(-1, 1),dist2bd_x,dist2bd_y
                       ], dim=1)
    test_x_2 = torch.cat([grid, torch.zeros([grid.shape[0],4]), dist2bd_x,dist2bd_y
                            ], dim=1)
    bd_coord_input = torch.tensor(bc_euco_test[j])  
    bd_coord_input_1=bd_coord_input.clone()
    bd_coord_input_1[:,2]=0

    data=HeteroData() 
    data['G1'].x=test_x #node features ▲u=f
    data['G1'].boundary=bd_coord_input_1 #boundary value=0
    data['G1'].edge_features=edge_attr
    data['G1'].sample_idx=idx
    data['G1'].edge_index=edge_index
    data['G1'].cell_state = cell_state_test_current

    data['G2'].x=test_x_2  ##node features ▲u=0
    data['G2'].boundary=bd_coord_input #boundary value=g(x)
    data['G2'].edge_features=edge_attr
    data['G2'].sample_idx=idx
    data['G2'].edge_index=edge_index
    
    data['G1+2'].y=test_u[j, idx]
     
    data_test.append(data)


train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)
t2 = default_timer()

logger.info('preprocessing finished, time used:{}', t2-t1)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if args.act == 'leakyrelu':
    activation = nn.LeakyReLU 
elif args.act == 'elu':
    activation = nn.ELU
elif args.act == 'relu':
    activation = nn.ReLU
else:
    activation = nn.SiLU
    
model = HeteroGNS(nnode_in_features = node_features, nnode_out_features = 1, nedge_in_features = edge_features, nmlp_layers=args.nmlp_layers,
             activation = activation,boundary_dim = args.boundary_dim,trans_layer = trans_layer).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2)
myloss = LpLoss(size_average=False)
u_normalizer.cuda(device)
ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))
model.train()

data_record = {}
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out.view(-1, 1), batch['G1+2'].y.view(-1,1))

        loss.backward()
        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch['G1'].sample_idx.view(batch_size, -1)),
            u_normalizer.decode(batch['G1+2'].y.view(batch_size, -1), sample_idx=batch['G1'].sample_idx.view(batch_size, -1))) #G1和G2的sanmple_idx是一样的
        
        optimizer.step()
        train_mse += loss.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch['G1'].sample_idx.view(batch_size2,-1))
            test_l2 += myloss(out, batch['G1+2'].y.view(batch_size2, -1)).item()

    t3 = default_timer()
    ttrain[ep] = train_l2/(ntrain)
    ttest[ep] = test_l2/ntest
    logger.info(f"Epoch {ep:03d}     train_Loss: {train_mse/len(train_loader):.6f}  \t train_L2: {train_l2/(ntrain):.6f}\t test_L2: {test_l2/ntest:.6f}")
    record_data(data_record, [ep, train_mse/len(train_loader), train_l2/(ntrain), test_l2/ntest], ["epoch", "train_MSE", "train_L2", "test_L2"])
    if ep % inspect_interval == 0 or ep == epochs - 1:
        record_data(data_record, [ep, to_cpu(model.state_dict())], ["save_epoch", "state_dict"])
        pickle.dump(data_record, open(path_model, "wb"))