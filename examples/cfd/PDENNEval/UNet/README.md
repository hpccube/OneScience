# U-Net

## 配置文件

`config` 目录包含许多命名格式为 `config_{1/2/3}D_{PDE name}.yaml` 的 `yaml` 配置文件，所有训练和测试的参数均保存在其中。以下是一些 U-Net 特定参数的说明：

* 训练参数:
    * `training_type`：字符串，设置为 `autoregressive` 表示使用自回归损失的自回归训练，设置为 `single` 表示使用单步损失的单步训练。
    * `pushforward`：布尔值，设置为 `True` 表示使用推前训练，同时 `training_type` 也必须设置为 `True`。
    * `initial_step`：整数，输入时间步的数量。（默认：10）
    * `unroll_step`：整数，推前训练时反向传播的时间步数量。（默认：20）

* 模型参数:
    * `in_channels`：整数，输入通道数，等于待求解变量的数量。例如，1D 可压缩 NS 方程有3个变量需要求解：密度、压力和速度。
    * `out_channels`：整数，输出通道数，等于 `in_channels`。
    * `init_features`：整数，U-Net 第一层上采样块的通道数。

其他参数含义较为明确。为保证可重复性，以下表格给出了我们求解不同 PDE 所用的训练超参数。

| PDE 名称                    | 空间分辨率 / 下采样率            | 时间分辨率 / 下采样率           | 学习率 | 训练轮数 | 批大小 | 权重衰减       | 初始步数 | 展开步数    |
| :-------------------------- | :----------------------------- | :----------------------------- | :----- | :------- | :----- | :------------- | :------- | :--------- |
| 1D Advection                | 1024/4                        | 201/5                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Diffusion-Reaction       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Burgers                  | 1024/4                        | 201/5                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Diffusion-Sorption       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Allen Cahn               | 1024/4                        | 101/1                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Cahn Hilliard            | 1024/4                        | 101/1                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 1D Compressible NS          | 1024/4                        | 101/1                          | 1.e-3  | 500      | 64     | 1.e-4          | 10       | 20         |
| 2D Burgers                  | 128/1                         | 101/1                          | 1.e-3  | 500      | 8      | 1.e-4          | 10       | 20         |
| 2D Compressible NS          | 128/2                         | 21/1                           | 1.e-3  | 500      | 32     | 1.e-4          | 10       | 20         |
| 2D DarcyFlow                | 128/1                         | -                             | 1.e-3  | 500      | 64     | 1.e-4          | 1        | 1          |
| 2D Shallow Water            | 128/1                         | 101/1                          | 1.e-3  | 500      | 8      | 1.e-4          | 10       | 20         |
| 2D Allen Cahn               | 128/1                         | 101/1                          | 1.e-3  | 500      | 8      | 1.e-4          | 10       | 20         |
| 2D Black-Scholes-Barenblatt | 128/1                         | 101/1                          | 1.e-3  | 500      | 8      | 1.e-4          | 10       | 20         |
| 3D Compressible NS          | 128/2                         | 21/1                           | 1.e-3  | 500      | 2      | 1.e-4          | 10       | 20         |
| 3D Eular                    | 128/2                         | 21/1                           | 1.e-3  | 500      | 2      | 1.e-4          | 10       | 20         |
| 3D Maxwell                  | 32                            | 8                             | 1.e-3  | 500      | 2      | 1.e-4          | 2        | -          |

## 损失函数

U-Net 以自回归方式求解 PDE，模型 $f_{\theta}$ 基于前 $l$（即 `initial_step`）时间步的解 $\{\hat{u}^{k-l+1}, ..., \hat{u}^k\}$ 预测下一时间步解 $\hat{u}^{k+1}$。过程形式化表示为：

$$
\hat{u}^{k+1} = f_{\theta}(\hat{u}^{k-l+1}, ..., \hat{u}^k).
$$

损失函数形式为：

$$
\mathcal{L} = \frac{1}{N} \sum_{k=0}^{N-1} l(f_{\theta}(\hat{u}^{k-l+1}, ..., \hat{u}^k), u^{k+1})
$$

其中 $u^{k+1}$ 是由高阶数值方法生成的真实数据，$l$ 是我们实现中的均方误差（MSE）。

实际上，前 $l$ 个时间步的解 $\{u^0, ..., u^{l-1}\}$ 可能由其他高阶方法生成，作为模型的初始输入。剩余时间步的解由训练好的模型自回归生成，后续输入也包含模型的预测结果。

## 训练策略

**单步训练**：模型输入始终来自真实数据。损失函数形式为：

$$
\mathcal{L} = \frac{1}{N} \sum_{k=0}^{N-1} l(f_{\theta}(u^{k-l+1}, ..., u^k), u^{k+1}).
$$

**自回归训练**：标准训练策略。初始输入来自真实数据，中间输入来自模型预测。损失函数形式同上。

**推前技巧**：该技巧由 [MPNN](https://arxiv.org/abs/2202.03376) 提出，用于促进更稳定且更快速的自回归训练。使用推前技巧时，我们仅对最后 $M$（即 `unroll_step`）个时间步的梯度进行反向传播。该实现采用自 [PDEBench](https://arxiv.org/abs/2210.07182)。损失函数形式为：

$$
\mathcal{L} = \frac{1}{M} \sum_{k=N-M}^{N-1} l(f_{\theta}(u^{k-l+1}, ..., u^k), u^{k+1}).
$$

## 训练

1. 检查配置文件中的以下参数：
    1. `file_name` 和 `saved_folder` 路径是否正确；
    2. `if_training` 是否为 `True`；
2. 设置训练超参数，如学习率、批大小等。可以使用我们提供的默认值；
3. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config_filename}
# 示例：CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Darcy_Flow.yaml

## 继续训练

1. 修改配置文件：
    1. 确保 `if_training` 为 `True`；
    2. 设置 `continue_training` 为 `True`；
    3. 将 `model_path` 设置为重新训练所用的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config_filename}
```

## 测试

1. 修改配置文件：
    1. 将 `if_training` 设置为 `False`；
    2. 将 `model_path` 设置为待评估模型的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/${config_filename}
```