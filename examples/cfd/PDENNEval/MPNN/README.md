# MPNN

## 配置文件

我们在一维和二维时变问题上评估 MPNN。配置文件保存在 `config` 目录中。以下是一些 MPNN 特定参数的说明：

* PDE 信息参数:
    * `temporal_domain`：元组，时间域的取值范围。
    * `resolution_t`：整数，数据的原始时间分辨率。
    * `spatial_domain`：字符串，可用 `eval` 函数解析为元组列表，空间域的取值范围。
    * `resolution`：整数，数据的原始空间分辨率。
    * `variables`：字典，包含 PDE 参数。
    * `num_outputs`：整数，待求解变量的数量。

* 训练参数:
    * `neighbors`：整数，消息传递时使用的邻居数量。默认一维问题设为3，二维问题设为1。
    * `time_window`：整数，输入/输出时间步的数量。我们设为10，等于其他自回归方法中 `initial_step` 参数值，以保证公平比较。
    * `unrolling`：整数，展开前向步数的数量。（默认1）
    * `unroll_step`：整数，推前训练时反向传播的时间步数量。由于 MPNN 与其他自回归方法在训练策略上存在细微差别（详见[这里](https://github.com/zhouzy36/PDENNEval/tree/main/src/MPNN#training-strategy)），此参数借用自其他自回归方法，保证单个 MPNN 训练 epoch 的迭代次数几乎相等。（默认1）

* 模型参数:
    * `hidden_features`：整数，节点特征维度数量。
    * `hidden_layer`：整数，GNN 层数。

为保证可重复性，以下表格给出了我们求解不同 PDE 所用的训练超参数。

| PDE 名称                   | 空间分辨率 / 下采样率         | 时间分辨率 / 下采样率        | 学习率 | 批大小 | 权重衰减 | 邻居数量 | 训练轮数 | 学习率调度           |
| :------------------------- | :--------------------------- | :--------------------------- | :----- | :----- | :------- | :------- | :------- | :------------------- |
| 1D Advection               | 1024/4                      | 201/5                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Diffusion-Reaction      | 1024/4                      | 101/1                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Burgers                 | 1024/4                      | 201/5                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Diffusion-Sorption      | 1024/4                      | 101/1                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Compressible NS         | 1024/4                      | 101/1                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Allen Cahn              | 1024/4                      | 101/1                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 1D Cahn Hilliard           | 1024/4                      | 101/1                       | 1.e-4  | 64     | 1.e-8    | 3        | 500      | "StepLR", 100, 0.5    |
| 2D Shallow Water           | 128/2                       | 101/1                       | 1.e-4  | 8      | 1.e-8    | 1        | 500      | "StepLR", 100, 0.5    |
| 2D Compressible NS         | 128/2                       | 21/1                        | 1.e-4  | 16     | 1.e-8    | 1        | 500      | "StepLR", 100, 0.5    |
| 2D Burgers                 | 128/2                       | 101/1                       | 1.e-4  | 8      | 1.e-8    | 1        | 500      | "StepLR", 100, 0.5    |
| 2D Allen-Cahn              | 128/2                       | 101/1                       | 1.e-4  | 8      | 1.e-8    | 1        | 500      | "StepLR", 100, 0.5    |
| 2D Black-Scholes-Barenblatt| 128/2                       | 101/1                       | 1.e-4  | 8      | 1.e-8    | 1        | 500      | "StepLR", 100, 0.5    |

## 损失函数

MPNN 采用带有**时间捆绑技巧**的自回归方法求解 PDE。与其他自回归方法一次前向计算预测下一时间步解不同，MPNN 会一次预测接下来 $l$（即 `time_window`）个时间步的解，形式化表达为：

$$
(\hat{u}^{k+1}, ..., \hat{u}^{k+l}) = f_{\theta}(\hat{u}^{k-l+1}, ..., \hat{u}^k).
$$

这将求解器调用次数减少了 $l$ 倍，从而减少了 rollout 时间。损失函数为标准形式：

$$
\mathcal{L} = \frac{1}{N} \sum_{k=0}^{N-1} l(\hat{u}^{k+1}, u^{k+1}),
$$

其中 $u^{k+1}$ 是由高阶数值方法生成的真实数据，$l$ 是我们实现中的均方误差（MSE）。

## 训练策略

我们使用**推前技巧（pushforward trick）**训练 MPNN。对于一个包含 $L$ 个时间步解的样本，随机抽取时间点 $t$，执行 (`unrolling` + 1) 次前向计算，但仅对最后一步传播损失。这与其他自回归方法不同，后者总是从固定时间点（与 `init_step` 参数相关）开始自回归循环，并在训练期间在剩余时间点累积损失。虽然我们用推前技巧训练 U-Net，但 U-Net 的自回归循环对所有样本均从固定的 `init_step` 时间点开始。更多训练细节请参见其[论文](https://arxiv.org/abs/2202.03376)的附录F。

## 训练

1. 检查配置文件中的以下参数：
    1. `file_name` 和 `saved_folder` 路径是否正确；
    2. `if_training` 是否为 `True`；
2. 设置训练超参数，如学习率、批大小等。可以使用我们提供的默认值；
3. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${配置文件名}
# 示例：CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_1D_Advection.yaml
```

## 继续训练

1. 修改配置文件：
    1. 确保 `if_training` 为 `True`；
    2. 设置 `continue_training` 为 `True`；
    3. 将 `model_path` 设置为重新训练所用的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${配置文件名}
```

## 测试

1. 修改配置文件：
    1. 将 `if_training` 设置为 `False`；
    2. 将 `model_path` 设置为待评估模型的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${配置文件名}
```