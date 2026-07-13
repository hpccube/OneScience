#!/bin/bash
#SBATCH --job-name=dpa3_finetune
#SBATCH --partition=hpctest02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

SCRIPT_DIR="$SLURM_SUBMIT_DIR"

source /public/software/sghpc_sdk/Linux_x86_64/25.6/das/conda/etc/profile.d/conda.sh
source "$SCRIPT_DIR/../../../matchem_env.sh"

# 限制并行度，规避 ROCm kernel launch 问题
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export OMP_NUM_THREADS=1

cd "$SCRIPT_DIR"
dp --pt train input_finetune.json --finetune ./DPA-3.1-3M.pt
