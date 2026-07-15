# OneScience MatChem 领域使用指南

> 本目录假设已完成 `onescience[matchem]` 的基础安装（参见仓库外层安装文档）。
> 这里仅介绍 MatChem 领域特有组件、环境入口与使用方式。

---

## 1. 环境架构概览

| 能力 | 所属目录 | 说明 |
|------|---------|------|
| MACE 训练 | `mace/` | 已包含在 `onescience[matchem]` 基础环境中 |
| UMA 训练 | `uma/` | 已包含在 `onescience[matchem]` 基础环境中 |
| MatRIS 训练 | `matris/` | 已包含在 `onescience[matchem]` 基础环境中 |
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

> 默认激活的 conda 环境名为 `matchem_pip`。如果你安装时使用了其他环境名（例如 `test_pip`），请覆盖该变量：
> ```bash
> MATCHEM_CONDA_NAME=test_pip source matchem_env.sh
> ```

`matchem_env.sh` 会完成：
- 加载 `sghpcdas/25.6`、`sghpc-mpi-gcc/26.3` 模块
- 激活 conda 环境
- 设置 `ONESCIENCE_DATASETS_DIR`、`ONESCIENCE_MODELS_DIR`、`device`、`LD_LIBRARY_PATH` 等运行时变量
- 定义 `LAMMPS_INSTALL_DIR`、`DEEPMD_SRC_DIR`、`DP_CPP_DIR`、`MATPL_SRC_DIR` 等关键路径变量

> 如果你需要使用 **DP 训练**、**NEP 训练** 或 **LAMMPS 推理**，请先编辑 `matchem_env.sh`，将以下变量修改为你的实际路径，然后重新 `source matchem_env.sh`：
> ```bash
> export DEEPMD_SRC_DIR=/path/to/deepmd-kit_dcu
> export MATPL_SRC_DIR=/path/to/matpl_dcu
> export LAMMPS_INSTALL_DIR=/path/to/lammps_dcu
> export DP_CPP_DIR=/path/to/dp_cpp_dcu
> ```
>
> 或者运行对应的 `dp_install.sh`、`matpl_install.sh`、`lmp_install.sh`，它们会自动将这些路径写回 `matchem_env.sh`。

---

## 3. 领域特有组件安装（按需）

### Step 1: DP 训练环境（可选）

```bash
cd dp
# 获取 deepmd-kit 源码（生产环境建议提前上传，通过 DEEPMD_SRC_DIR 指定）
git clone https://github.com/deepmodeling/deepmd-kit.git
bash dp_install.sh
```

**说明**：
- `dp_install.sh` 会自动检测 PyTorch 路径并启用 ROCm 后端编译
- 默认安装 PyTorch + TensorFlow 双后端
- C++ 接口默认跳过编译；如需自行编译，设置 `COMPILE_DP_CPP=1`

---

### Step 2: NEP 训练环境（可选）

```bash
cd nep
export MATPL_SRC_DIR=/path/to/matpl_dcu  # 默认使用 $(pwd)/matpl_dcu
bash matpl_install.sh
```

**说明**：
- `matpl_install.sh` 会自动生成 `dcu_install.sh` 并完成 C++ 扩展编译
- 各 NEP 算例的 `submit.sh` 中已 inline 维护 MatPL 运行时环境

---

### Step 3: LAMMPS 安装（可选，若需分子动力学推理）

将 DCU 适配优化后的 LAMMPS 安装包解压到目标目录，并在 `matchem_env.sh` 中定义：

```bash
export LAMMPS_INSTALL_DIR="/path/to/your/lammps_dcu"
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

---

## 6. 目录速查

```
examples/matchem/
├── matchem_env.sh          # 统一环境入口
├── README.md               # 本文件：基于 PyPI 安装后的使用指南
├── dp/                     # DeepMD-kit 训练
├── mace/                   # MACE 训练
├── matris/                 # MatRIS 训练
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

---

## 8. 扩展功能预览（FastEq-hip）

| 场景 | 支持状态 |
|------|---------|
| MACE 在 ASE 中的推理加速 | 已支持 |
| MACE 训练加速 | 暂不支持 |
| MACE 在 LAMMPS 中的推理加速 | 暂不支持 |
