#!/bin/bash
#SBATCH --job-name=Cu_NEP
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
export LD_LIBRARY_PATH=${LAMMPS_INSTALL_DIR}/lib_override:$LD_LIBRARY_PATH
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH=${SITE_PACKAGES}/torch/lib:$LD_LIBRARY_PATH

cd "$SCRIPT_DIR"
chmod +x mpi_bind.sh
mpirun -np 8 ./mpi_bind.sh in.lmp

