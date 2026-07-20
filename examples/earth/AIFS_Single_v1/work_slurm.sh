#!/bin/bash
#SBATCH -p hx1hdexclu12
#SBATCH -N 1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J AIFS
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

# --- 日志目录 ---
mkdir -p logs

source ../../../env.sh

export OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname)

echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_NTASKS=$SLURM_NTASKS

srun -u --mpi=pmix\
    bash -c "
    source ../earth_env.sh
    source ../export_DDP_vars.sh
    python train.py
    "
