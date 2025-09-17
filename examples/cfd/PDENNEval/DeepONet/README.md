# DeepOnet

## 配置文件

`config` 目录包含许多命名格式为 `config_{1/2/3}D_{PDE name}.yaml` 的 `yaml` 配置文件。以下是一些 DeepOnet 特定参数的说明：

* 训练参数:
    * `training_type`：字符串，`single` 指定 DeepOnet 的非自回归处理。
    * `initial_step`：整数，输入时间步的数量。（默认：1）

* 模型参数:
    * `input_size`：整数，指定每个维度空间网格的长度，用于计算对应于 branch 网络的 MLP 的最终输入形状。例如二维网格 256*256 应设置 `input_size`=256。
    * `in_channels`：整数，输入通道数，等于待求解变量的数量。例如，1D 可压缩 NS 方程有3个变量需要求解：密度、压力和速度。
    * `out_channels`：整数，输出通道数。
    * `query_dim`：整数，MLP 对应 trunk 网络输入的查询位置长度。

    以 1D 示例为例，`input_size`=256，`in_channels`=1，branch 网络的 MLP 应输入形状为 $(N_b,256)$ 的张量，其中 $N_b$ 是批大小，256 是 `input_size^{spatial_dimension} \times in_channels \times initial_step`。

训练时使用的超参数如下：

| PDE 名称                    | 空间分辨率 / 下采样率            | 时间分辨率 / 下采样率           | 学习率 | 训练轮数 | 批大小 | 权重衰减       |
| :-------------------------- | :----------------------------- | :----------------------------- | :----- | :------- | :----- | :------------- |
| 1D Advection                | 1024/4                        | 201/5                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Diffusion-Reaction       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Burgers                  | 1024/4                        | 201/5                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Diffusion-Sorption       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Allen Cahn               | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Cahn Hilliard            | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 1D Compressible NS          | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4          |
| 2D Burgers                  | 128/1                         | 101/1                          | 1.e-3  | 200      | 50     | 1.e-4          |
| 2D Compressible NS          | 128/1                         | 21/1                           | 1.e-3  | 200      | 50     | 1.e-4          |
| 2D DarcyFlow                | 128/1                         | -                             | 1.e-3  | 500      | 50     | 1.e-4          |
| 2D Shallow Water            | 128/1                         | 101/1                          | 1.e-3  | 200      | 50     | 1.e-4          |
| 2D Allen Cahn               | 128/1                         | 101/1                          | 1.e-3  | 200      | 50     | 1.e-4          |
| 2D Black-Scholes-Barenblatt | 128/1                         | 101/1                          | 1.e-3  | 200      | 50     | 1.e-4          |

## 损失函数

DeepOnet 以非自回归方式求解 PDE，模型 $f_{\theta}$ 基于初始时间步的解（`initial_step`=1）$\hat{u}^0$，预测所有时间步的解集合 $\{\hat{u}^0 \dots \hat{u}^T\}$。

损失函数形式为：

$$
\mathcal{L} = \frac{1}{N_b} \sum_{i=1}^{N_b} l(u_{\text{pred}}, u)
$$

其中 \(N_b\) 是批大小，\(u\) 和 \(u_{\text{pred}}\) 分别是真实解和预测解，\(l\) 是我们实现中的均方误差（MSE）。

## 训练

1. 检查配置文件中的以下参数：
    1. `file_name` 和 `saved_folder` 路径是否正确；
    2. `if_training` 是否为 `True`；
2. 设置训练超参数，如学习率、批大小等。可以使用我们提供的默认值；
3. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/${config_filename}
# 示例：CUDA_VISIBLE_DEVICES=0 python train.py ./config/config_2D_Darcy_Flow.yaml
```

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