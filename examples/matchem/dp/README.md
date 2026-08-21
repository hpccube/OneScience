# DeepMD（DP）训练示例

本目录包含基于 DeepMD-kit PyTorch 后端的分子动力学势函数训练、微调和推理示例。传统 DeepMD、DPA3 和 DPA4 共用同一个 Conda 环境和同一套 `dp` 命令。

## 前置条件

1. 已完成 `onescience[matchem]` 基础环境安装。
2. 使用前设置自己的 Conda 环境名并加载统一环境。默认环境名为
   `onescience311`；如果实际环境使用其他名称，请先覆盖
   `MATCHEM_CONDA_NAME`：

   ```bash
   # 默认环境
   export MATCHEM_CONDA_NAME=onescience311
   source ../matchem_env.sh

   # 例如使用自定义环境时
   export MATCHEM_CONDA_NAME=my_onescience_env
   source ../matchem_env.sh
   ```

   `matchem_env.sh` 是环境名默认值的唯一设置位置，各算例的
   `submit*.sh` 不会再次覆盖该变量。通过 `sbatch --export=ALL`
   提交时，当前设置会传入计算节点。

3. 首次使用时一键安装统一 DeepMD-kit DCU 版：

   ```bash
   bash dp_install.sh
   ```

安装脚本会自动完成以下工作：

- 使用现有环境中的 DTK PyTorch 2.5.1、TensorFlow 2.18 和 Triton 3.1；
- 下载 vesin 0.6.1，以及针对 DTK PyTorch 2.5.1 编译并校验过的 vesin-torch 0.6.1 wheel；
- 自动补齐固定版本的 DeepMD Python 运行依赖，不重新安装 Torch、TensorFlow 或 Triton；
- 拉取 Gitee `dpa4-torch251` 分支并固定到提交 `40a7d99fa46c8ff1e75b5be9d64540d95dbac184`；
- 编译安装 PyTorch + TensorFlow 双后端 DeepMD-kit；
- 隔离账号环境中的外部 Intel MKL，并审计 PyTorch 自定义算子的动态库依赖；
- 保留并安装原有的预编译 C++/LAMMPS 接口包。

源码、下载文件和临时构建结果默认保存在 `${HOME}/.cache/onescience/deepmd-unified`，不会在本目录中生成 wheel 或源码文件夹。
在交互式终端运行时，脚本会询问 DeepMD-kit 源码目录和 C++ 接口目录；直接
回车使用上述用户缓存目录。DeepMD Python 包始终安装到当前 Conda 环境的
`site-packages`，安装器不允许把源码或 C++ 接口放入项目的 `dp/` 目录。

也可以在执行前通过绝对路径指定：

```bash
DEEPMD_SRC_DIR=/your/source/path \
DEEPMD_CPP_DIR=/your/cpp/path \
bash dp_install.sh
```

## 目录说明

```text
dp/
├── demo/                              # DeepMD 算例
├── README.md                          # 本说明文档
└── dp_install.sh                      # DeepMD-kit DCU 一键安装脚本
```

`demo/` 中包括：

- `water_se_e2_a_pt/`：传统 `se_e2_a` 模型（PyTorch）；
- `water_se_atten_pt/`：attention/DPA1 模型（PyTorch）；
- `water_se_e2_a_tf/`：TensorFlow 示例（本轮未验收）；
- `dpa3_finetune_CH/`：DPA3 CH 微调；
- `dpa4_train_water/`：DPA4 water 从头训练；
- `dpa4_finetune_CH/`：DPA4-beta CH 微调；
- `dpa4_test_CH/`：DPA4-beta `dp test` 直接推理。

## 快速开始

### 单卡训练

```bash
cd demo/water_se_e2_a_pt
sbatch submit_1card.sh
```

### 多卡训练

```bash
cd demo/water_se_e2_a_pt
sbatch submit_4card.sh   # 四卡
sbatch submit_8card.sh   # 八卡
```

### DPA3 微调

```bash
cd demo/dpa3_finetune_CH
sbatch submit.sh
```

DPA3 默认读取 `${ONESCIENCE_MODELS_DIR}/deepmd/DPA-3.1-3M.pt`，也可以显式指定：

```bash
DPA3_BASE_MODEL=/path/to/DPA3.pt sbatch --export=ALL submit.sh
```

### DPA4 训练、微调和推理

```bash
# 从头训练
cd demo/dpa4_train_water
sbatch submit_1card.sh

# 基于 DPA4-beta 的 OMat24 分支微调
cd ../dpa4_finetune_CH
sbatch submit.sh

# dp test 直接推理
cd ../dpa4_test_CH
sbatch submit.sh
```

DPA4 基座模型按以下顺序解析：`DPA4_BASE_MODEL` 用户覆盖、`${ONESCIENCE_MODELS_DIR}/deepmd/DPA4-beta.pt`、算例目录内的 `DPA4-beta.pt`。

### 短流程测试

复制一份输入文件并修改其中的 `numb_steps`、`disp_freq` 和
`save_freq`，再通过 `INPUT_JSON` 指定测试配置。例如：

```bash
cp input_torch.json input_test.json
# 将 input_test.json 中的训练步数改为所需值
INPUT_JSON="$PWD/input_test.json" sbatch --export=ALL submit_1card.sh
```

每个作业的 checkpoint 和统计文件写入算例下独立的 `run_<jobid>/`，避免并发作业
争用同一个 HDF5 统计文件；Slurm 日志写入当前算例目录。DPA4 算例会在系统临时
目录中生成展开数据路径后的输入文件，并在作业退出时自动删除。

## 注意事项

- 当前统一范围是传统 PyTorch、DPA3、DPA4 eager 和普通 DDP；DPA4 输入配置保持 `use_compile=false`。
- TensorFlow 后端参与 DeepMD 编译并保留 CLI，但本轮没有验收 TensorFlow DCU 训练。
- 原有 `dp_cpp_dcu.tar.gz` 继续服务旧 C++/LAMMPS 推理；Python 环境统一不代表新版 C++ 接口已经验收。
- `matchem_env.sh` 默认使用 `onescience311`；其他环境应在安装或提交作业前通过 `MATCHEM_CONDA_NAME` 指定。
- 队列名以脚本中的 `#SBATCH --partition` 为默认值，提交时可用 `sbatch -p <partition>` 覆盖。
