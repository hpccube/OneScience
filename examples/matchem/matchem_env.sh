#!/bin/bash
# ==========================================
# MatChem 统一环境配置脚本
# 用途：加载模块、激活 conda、导出各组件路径
# 用法：source matchem_env.sh
# 注意：默认激活 conda 环境 test_pip；若使用其他环境名，请执行：
#       MATCHEM_CONDA_NAME=your_env source matchem_env.sh
# ==========================================

# ---------- 1. 基础环境配置 ----------
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"

# ---------- 2. OneScience 运行时环境变量 ----------
export ONESCIENCE_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets"
export ONESCIENCE_MODELS_DIR="/public/share/sugonhpcapp01/onestore/onemodels"
export device="gpu"  # 根据实际平台改为 gpu 或 dcu
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# ---------- 3. 外部软件路径（TODO: 修改为你自己的实际路径，默认当前目录） ----------
# DeepMD-kit 源码目录
export DEEPMD_SRC_DIR="${DEEPMD_SRC_DIR:-./deepmd-kit}"

# MatPL 源码目录
export MATPL_SRC_DIR="${MATPL_SRC_DIR:-./MatPL}"

# LAMMPS 安装目录
export LAMMPS_INSTALL_DIR="${LAMMPS_INSTALL_DIR:-./lammps_dcu}"

# DeepMD C++ 接口目录
export DP_CPP_DIR="${DP_CPP_DIR:-./dp_cpp_dcu}"

# ---------- 4. 加载集群模块与 conda ----------
set +u
source ~/.bashrc
set -u
module load sghpcdas/25.6        # DTK / PyTorch 等 SDK
module load sghpc-mpi-gcc/26.3   # MPI 与 GCC 编译器

conda activate "$MATCHEM_CONDA_NAME"

# ---------- 6. LAMMPS 运行时环境 ----------
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib

echo "✅ MatChem 环境已激活: ${MATCHEM_CONDA_NAME}"
