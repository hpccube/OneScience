# WAN

不同的脚本对应不同的PDE方程

>```python
># the solved problems of different files
>WAN_Ad1.py  # 1D Advection equation
>WAN_DS1.py  # 1D Diffusion-Sorption equation
>WAN_Bu1.py  # 1D Burgers equation 
>WAN_AC1.py  # 1D Allen-Cahn equation
>WAN_DR1.py  # 1D Diffusion-Reaction equation
>WAN_AC2.py  # 2D Allen-Cahn equation
>WAN_DF2.py  # 2D Darcy-Flow equation
>WAN_BS2.py  # 2D Black-Scholes equation
>```

## 训练
```python
 # training codes
 python WAN_Ad1.py --s 0 --i 15000 --b 1000
 python WAN_DS1.py --s 0 --i 15000 --b 1000
 python WAN_Bu1.py --s 0 --i 15000 --b 1000
 python WAN_AC1.py --s 0 --i 15000 --b 1000 
 python WAN_DR1.py --s 0 --i 15000 --b 1000
 python WAN_AC2.py --s 0 --i 15000 --b 1000
 python WAN_DF2.py --s 0 --i 15000 --b 1000
 python WAN_BS2.py --s 0 --i 15000 --b 1000
```
### 命令行参数说明

- `--s`  ：随机种子，用于控制实验的随机性，保证结果可复现。

- `--d`  ：空间维度，指定所求解的泊松方程的维度。

- `--co` ：使用的 GPU 编号，支持多 GPU 情况下指定具体设备。

- `--i`  ：训练迭代次数，控制训练过程中梯度更新的总步数。

- `--b`  ：beta 参数值，控制的是边界条件损失（loss_bd）和初始条件损失（loss_init）在总损失中的权重。

