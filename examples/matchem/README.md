# OneScience MatChem 领域使用指南

> 本目录假设已完成 `onescience[matchem]` 的基础安装（参见仓库外层安装文档）。
> 这里仅介绍 MatChem 领域特有组件、环境入口与使用方式。

---

## 0. 前置条件

- 已按仓库外层安装文档创建 **Python 3.11** 的 conda 环境并完成 `bash install.sh matchem`。
  > **注意**：LAMMPS / DeepMD C++ 预编译组件均基于 Python 3.11（cp311）构建，其他 Python 版本的环境无法使用。
- 集群需可加载 `sghpcdas/25.6`、`sghpc-mpi-gcc/26.3` 模块（`matchem_env.sh` 会自动加载）。

---

## 1. 环境架构概览

| 能力 | 所属目录 | 说明 |
|------|---------|------|
| MACE 训练 | `mace/` | 已包含在 `onescience[matchem]` 基础环境中 |
| UMA 训练 | `uma/` | 已包含在 `onescience[matchem]` 基础环境中 |
| MatRIS 训练 | `matris/` | 已包含在 `onescience[matchem]` 基础环境中 |
| MatterGen 训练/微调/生成 | `mattergen/` | 源码内嵌在 `src/onescience/`，随 `onescience[matchem]` 一起安装 |
| MatterSim 推理/微调 | `mattersim/` | 源码内嵌在 `src/onescience/`，随 `onescience[matchem]` 一起安装 |
| DP 训练 | `dp/` | 需额外编译安装 deepmd-kit（PyTorch 后端） |
| NEP 训练 | `nep/` | 需额外编译安装 MatPL（DCU 原生算子） |
| LAMMPS 推理 | `tools/lmp/` | 需自行编译/解压 LAMMPS with HIP，支持 DP/NEP/MACE 后端 |

**核心原则**：所有环境变量与模块加载统一走 **`matchem_env.sh`**，子目录下的 `*_env.sh` 逐步收敛到该文件。

---

## 2. 环境入口

每次使用本目录功能前，请先加载统一环境：

```bash
source matchem_env.sh
```

> 默认激活的 conda 环境名为 `test_pip`。如果你使用其他环境名，请覆盖该变量：
> ```bash
> MATCHEM_CONDA_NAME=your_env source matchem_env.sh
> ```
> 所有 `submit.sh` 中已内置 `export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"`，提交作业时会自动沿用当前设置；如需切换环境，在提交前重新设置 `MATCHEM_CONDA_NAME` 即可。

`matchem_env.sh` 会完成：
- 加载 `sghpcdas/25.6`、`sghpc-mpi-gcc/26.3` 模块
- 激活 conda 环境
- 设置 `ONESCIENCE_DATASETS_DIR`、`ONESCIENCE_MODELS_DIR`、`device`、`LD_LIBRARY_PATH` 等运行时变量
- 定义 `LAMMPS_INSTALL_DIR`、`DEEPMD_SRC_DIR`、`DP_CPP_DIR`、`MATPL_SRC_DIR` 等关键路径变量

> 如果你需要使用 **DP 训练**、**NEP 训练** 或 **LAMMPS 推理**，对应的安装脚本（`dp_install.sh`、`matpl_install.sh`、`lmp_install.sh`）会自动将实际安装路径写回 `matchem_env.sh`，无需手动编辑。
>
> 如需自定义路径，可在运行安装脚本前通过环境变量指定：
> ```bash
> export DEEPMD_SRC_DIR=/your/deepmd-kit_dcu
> export MATPL_SRC_DIR=/your/matpl_dcu
> export LAMMPS_INSTALL_DIR=/your/lammps_dcu
> export DP_CPP_DIR=/your/dp_cpp_dcu
> ```

---

## 3. 领域特有组件安装（按需）

### Step 1: DP 训练环境（可选）

```bash
cd dp
bash dp_install.sh
```

**说明**：
- 脚本默认从 gitee 自动拉取 DCU 版源码（`deepmd-kit_dcu`）到 `dp/deepmd-kit`；生产环境建议提前上传源码，并通过 `DEEPMD_SRC_DIR` 指定路径
- **请勿**自行 `git clone` 上游 `deepmodeling/deepmd-kit` 到该目录——上游源码不含 DCU 支持，且脚本检测到目录已存在会跳过拉取，导致编译出错误版本
- 自动检测 PyTorch 路径并启用 ROCm 后端编译，默认安装 PyTorch + TensorFlow 双后端
- C++ 接口默认下载预编译包；如需源码编译，设置 `COMPILE_DP_CPP=1`

---

### Step 2: NEP 训练环境（可选）

```bash
cd nep
# 可选：通过环境变量指定 MatPL 源码路径，默认拉取到 nep/matpl_dcu
# export MATPL_SRC_DIR=/your/matpl_dcu
bash matpl_install.sh
```

**说明**：
- 脚本默认从 gitee 自动拉取 MatPL DCU 源码（分支 `nep-dcu/2026.3`），自动 patch CMakeLists 并完成 C++ 算子编译
- 各 NEP 算例的 `submit.sh` 中已 inline 维护 MatPL 运行时环境

---

### Step 3: LAMMPS 安装（可选，若需分子动力学推理）

```bash
cd tools/lmp
# 可选：通过环境变量指定安装路径，默认使用 $(pwd)/lammps_dcu 和 $(pwd)/dp_cpp_dcu
# export LAMMPS_INSTALL_DIR=/your/lammps_dcu
# export DP_CPP_DIR=/your/dp_cpp_dcu
bash lmp_install.sh
```

**说明**：
- 运行时不需要暴露 LAMMPS 源码路径
- `tools/lmp/lmp_script/` 下的脚本仅用于**编译时**环境配置
- DeepMD C++ 接口（`dpplugin.so`）通过 `DP_CPP_DIR` 管理，运行时需设置 `LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib`

---

## 4. 模型训练入口

### MACE 训练

```bash
cd mace/demo
bash run.sh --config configs/DMC.yaml
```

### UMA 训练

```bash
cd uma/demo
bash run.sh --config configs/oc20_ef_4dcu.yaml
```

### MatRIS 训练

```bash
cd matris/demo
# 根据实际算例执行
```

### MatterSim 推理与微调

```bash
cd mattersim
python single_point.py
```

MatterSim 的结构弛豫、MD 和 OneScience `torchrun` 微调命令见
`mattersim/README.md`。

### MatterGen 训练

```bash
cd mattergen/demo
bash run.sh --config configs/train_8dcu.yaml --submit
```

MatterGen 的晶体生成、属性微调、数据预处理和训练参数说明见
`mattergen/README.md`。

### DP 训练

```bash
cd dp/demo/water_se_e2_a_pt
# 单卡
dp --pt train input_torch.json

# 多卡
torchrun --nproc_per_node=4 -m deepmd --pt train input_torch.json
```

### NEP 训练

```bash
cd nep/demo/nep_Cu
sbatch submit.sh
```

> **切换 conda 环境**：如果当前环境名不是 `test_pip`，提交前请先设置环境变量：
> ```bash
> MATCHEM_CONDA_NAME=your_env sbatch --export=ALL submit.sh
> ```

---

## 5. LAMMPS 推理示例

所有推理算例统一在 `tools/lmp/` 下，`submit.sh` 均通过 `source matchem_env.sh` 加载环境。

```bash
# DP 推理
cd tools/lmp/deepmd/dp_alloy_npt
sbatch submit.sh

# NEP 推理
cd tools/lmp/nep/Cu
sbatch submit.sh

# MACE 推理
cd tools/lmp/mace/LiGaClF
sbatch submit.sh
```

> **MACE 模型文件**：`submit.sh` 会优先从 `${ONESCIENCE_MODELS_DIR}/mace/mace-mpa-0-medium.model-lammps.pt` 链接模型；如果集群模型库中没有，请从 [MACE 官方 releases](https://github.com/ACEsuit/mace-mp/releases) 下载后放入算例目录。

> **切换 conda 环境**：如果当前环境名不是 `test_pip`，提交前请先设置环境变量：
> ```bash
> MATCHEM_CONDA_NAME=your_env sbatch --export=ALL submit.sh
> ```

---

## 6. 目录速查

```
examples/matchem/
├── matchem_env.sh          # 统一环境入口
├── README.md               # 本文件：基于 PyPI 安装后的使用指南
├── dp/                     # DeepMD-kit 训练
├── mace/                   # MACE 训练
├── matris/                 # MatRIS 训练
├── mattergen/              # MatterGen 训练、微调与晶体生成
├── mattersim/              # MatterSim 推理与微调
├── nep/                    # NEP 训练
├── uma/                    # UMA 训练
└── tools/lmp/              # LAMMPS 推理
```

---

## 7. 已知问题与注意事项

1. **DeepMD TensorFlow 后端**：DCU 平台 TensorFlow 存在 MLIR kernel_gen JIT 编译缺陷，PyTorch 后端训练为推荐路径。
2. **LAMMPS 安装路径**：在 `matchem_env.sh` 中定义 `LAMMPS_INSTALL_DIR`，运行脚本通过该变量获取路径。
3. **DeepMD C++ 接口**：通过 `DP_CPP_DIR` 管理，LAMMPS+DP 推理时需设置 `LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib`。
4. **FastEq 源码依赖**、**MatPL 源码依赖**：均为外部私有/合作源码，需单独获取。
5. **队列名**：各 `submit.sh` 中 `#SBATCH --partition=hx1hdexclu12` 为本集群队列配置，迁移到其他集群时需自行修改。

---

## 8. 扩展功能预览（FastEq-hip）

| 场景 | 支持状态 |
|------|---------|
| MACE 在 ASE 中的推理加速 | 已支持 |
| MACE 训练加速 | 暂不支持 |
| MACE 在 LAMMPS 中的推理加速 | 暂不支持 |
