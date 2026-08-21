#!/usr/bin/env bash
#SBATCH --job-name=dp_pt_se_e2_a_8card
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -Eeuo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../../../matchem_env.sh"

# DeepMD PyTorch 自定义算子需要能够找到当前环境的 Torch 动态库。
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-1}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/run_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
torchrun --standalone --nproc_per_node=8 -m deepmd --pt train "${INPUT_JSON:-${SCRIPT_DIR}/input_torch.json}"
