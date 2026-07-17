#!/bin/bash
#SBATCH --job-name=LiCaClF_MACE
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"
source "$SCRIPT_DIR/../../../../matchem_env.sh"

# LAMMPS 运行时库路径
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/dcc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:$LD_LIBRARY_PATH
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH=${SITE_PACKAGES}/torch/lib:$LD_LIBRARY_PATH

# 优先使用系统 hwloc，避免 conda hwloc 与集群 PMIx 在 MPI_Finalize 时冲突
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/hwloc/lib:$LD_LIBRARY_PATH

export PATH=${LAMMPS_INSTALL_DIR}/bin:$PATH

cd "$SCRIPT_DIR"

# MACE 模型文件：优先从 ONESCIENCE_MODELS_DIR 链接，若不存在则提示用户自行下载
MACE_MODEL_NAME="mace-mpa-0-medium.model-lammps.pt"
MACE_MODEL_SRC="${ONESCIENCE_MODELS_DIR:-}/mace/${MACE_MODEL_NAME}"
if [ -f "${MACE_MODEL_SRC}" ] && [ ! -f "${MACE_MODEL_NAME}" ]; then
    echo ">>> 链接 MACE 模型文件: ${MACE_MODEL_SRC}"
    ln -s "${MACE_MODEL_SRC}" ./
elif [ ! -f "${MACE_MODEL_NAME}" ]; then
    echo "[警告] 未找到 MACE 模型文件 ${MACE_MODEL_NAME}"
    echo "       请从 https://github.com/ACEsuit/mace-mp/releases 下载后放入当前目录"
fi

# ML-MACE only supports single-card simulations, deprecated by LAMMPS officials
mpirun -np 1 lmp_mpi -k on g 1 -sf kk -pk kokkos gpu/aware off newton on neigh half -in in.lmp

