#!/bin/bash
# ==========================================
# MatChem 统一环境配置脚本
# 用途：加载模块、激活 conda、导出各组件路径
# 用法：source matchem_env.sh
# ==========================================

# ---------- 1. 基础环境配置 ----------
export MATCHEM_CONDA_NAME=matchem_test
export ONESCIENCE_MAIN_DIR=/public/home/easyscience2024/wangrui/onescience

# ---------- 2. 训练软件源码/安装路径 ----------
export DEEPMD_SRC_DIR=/public/home/easyscience2024/wangrui/software/deepmd-kit_dcu
export MATPL_SRC_DIR=/public/home/easyscience2024/wangrui/software/matpl_dcu

# ---------- 3. LAMMPS 与 C++ 接口路径 ----------
export LAMMPS_INSTALL_DIR="/public/home/easyscience2024/wangrui/software/lammps_dcu"
export DP_CPP_DIR="/public/home/easyscience2024/wangrui/software/dp_cpp_dcu"

# ---------- 4. 加载集群模块与 conda ----------
source ~/.bashrc
module load sghpcdas/25.6        # DTK / PyTorch 等 SDK
module load sghpc-mpi-gcc/26.3   # MPI 与 GCC 编译器

conda activate $MATCHEM_CONDA_NAME

# ---------- 5. 加载 OneScience 环境变量 ----------
source $ONESCIENCE_MAIN_DIR/env.sh

# ---------- 6. LAMMPS 运行时环境 ----------
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib
