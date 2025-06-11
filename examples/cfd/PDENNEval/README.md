# PDE求解模型集：PDENNEval

这是一个求解PDE的模型集，包括12个微分方程的神经网络模型。

## 介绍

PDENNEval 对 12 种用于偏微分方程（PDE）的神经网络（NN）方法进行了全面和系统的评估，其中包括 6 种基于函数学习的神经网络方法：[DRM](https://arxiv.org/abs/1710.00211), [PINN](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125), [WAN](https://arxiv.org/abs/1907.08272), [DFLM](https://arxiv.org/abs/2001.06145), [RFM](https://arxiv.org/abs/2207.13380), [DFVM](https://arxiv.org/abs/2305.06863v2), 以及 6 种基于算子学习的神经网络方法： [U-Net](https://arxiv.org/abs/1505.04597), [MPNN](https://arxiv.org/abs/2202.03376), [FNO](https://arxiv.org/abs/2010.08895), [DeepONet](https://arxiv.org/abs/1910.03193), [PINO](https://arxiv.org/abs/2111.03794), [U-NO](https://arxiv.org/abs/2204.11127). 在这个代码库中，我们提供了所有评估方法的代码参考。

## 软件库依赖

本项目依赖一些常用的Python库，包括：
pytorch
onescience
deepxde
torch_geometric
scikit-learn
h5py
tensorboard matplotlib tqdm等等
我们提供了模型对应的镜像，可以直接运行项目代码。

## 数据集

我们评估中使用的数据来自两个来源： [PDEBench](https://arxiv.org/abs/2210.07182) 和自生成的数据。

PDEBench 提供了涵盖广泛偏微分方程（PDE）的大规模数据集。

我们目前准备了3种数据集，包括一维扩散吸收问题数据，二维达西流问题数据集和二维潜水流数据集，并上传到了超算互联网平台上，可以下载使用。

| PDE | File Name | File Size | 
| :--- | :---: | :---: |
| 1D Diffusion-Sorption | 1D_diff-sorp_NA_NA.h5 | 4.0G |
| 2D Darcy Flow | 2D_DarcyFlow_beta1.0_Train.hdf5 | 1.3G |
| 2D Shallow Water | 2D_rdb_NA_NA.h5 | 6.2G |



## DCU上运行方法
deeponet


python train.py ./configs/config_2D_Darcy_Flow.yaml

python train.py ./configs/config_1D_Diffusion-Sorption.yaml

python train.py ./configs/config_2D_SWE.yaml

dflm

python DFLM_Poisson-Ph.py --dimension 3 --seed 0

python DFLM_Poisson-Ps.py --dimension 2 --seed 0

DFM,DRM and DFLM are similarly

FNO

python train.py ./config/config_2D_Darcy_Flow.yaml

python train.py ./config/config_1D_Diffusion-Sorption.yaml

python train.py ./config/config_2D_SWE.yaml

MPNN

cannot run torch_cluster

PINN

python PINN_Poisson-Ph.py --dimension 3 --seed 0

python PINN_Poisson-Ps.py --dimension 2 --seed 0

python run.py config_pinn_darcy.yaml

python run.py config_pinn_diff-sorp.yaml

python run.py config_pinn_swe2d.yaml

PINO

python train.py ./configs/4x/config_2D_Darcy_Flow.yaml

python train.py ./configs/4x/config_2D_SWE.yaml

python train.py ./configs/4x/config_1D_Diffusion-Sorption.yaml

RFM

python RFM_Poisson-PH.py --dimension 3

python RFM_Poisson-PH.py --dimension 20

python RFM_Poisson-PH_improved.py --dimension 3

python RFM_Poisson-PL.py

Unet

python train.py ./configs/config_2D_Darcy_Flow.yaml

python train.py ./configs/config_1D_Diffusion-Sorption.yaml

python train.py ./configs/config_2D_SWE.yaml

UNO

python train.py ./config/config_2D_Darcy_Flow.yaml

python train.py ./config/config_2D_SWE.yaml

WAN

python WAN_Poisson-PH.py --d 2 --i 15000 --s 0 --b 1000

python WAN_Poisson-PS.py --s 0
