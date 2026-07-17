#!/bin/bash
#SBATCH --job-name=dp_pfd
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=dcu:8
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

# DeepMD C++ 接口依赖 TensorFlow C++ 运行时
export LD_LIBRARY_PATH=${SITE_PACKAGES}/tensorflow:$LD_LIBRARY_PATH

# 优先使用系统 hwloc，避免 conda hwloc 与集群 PMIx 在 MPI_Finalize 时冲突
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/hwloc/lib:$LD_LIBRARY_PATH

# DeepMD LAMMPS 插件路径
export LAMMPS_PLUGIN_PATH=${DP_CPP_DIR}/lib

export PATH=${LAMMPS_INSTALL_DIR}/bin:$PATH

cd "$SCRIPT_DIR"
chmod +x mpi_bind.sh

# 8卡 GPU 加速（通过 mpi_bind.sh 绑定 NUMA 与 DCU）
mpirun -np 8 ./mpi_bind.sh in.lmp lmp_8gpu.log
