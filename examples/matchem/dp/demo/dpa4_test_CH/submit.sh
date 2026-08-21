#!/usr/bin/env bash
#SBATCH --job-name=dpa4_test_CH
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -Eeuo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../../../matchem_env.sh"

# DeepMD PyTorch 自定义算子需要能够找到当前环境的 Torch 动态库。
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

SHARED_MODEL="${ONESCIENCE_MODELS_DIR}/deepmd/DPA4-beta.pt"
LOCAL_MODEL="${SCRIPT_DIR}/../dpa4_finetune_CH/DPA4-beta.pt"
if [[ -n "${DPA4_BASE_MODEL:-}" ]]; then
    MODEL="${DPA4_BASE_MODEL}"
elif [[ -f "${SHARED_MODEL}" ]]; then
    MODEL="${SHARED_MODEL}"
else
    MODEL="${LOCAL_MODEL}"
fi
if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: DPA4 model not found: ${MODEL}" >&2
    exit 1
fi

SYSTEM="${SYSTEM:-${ONESCIENCE_DATASETS_DIR}/matchem/dp/dpa4_finetune/val_CH/sys_100}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/run_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
dp --pt test -m "${MODEL}" -s "${SYSTEM}" -n "${NUMB_TEST:-1}" \
    -d "${DETAIL_PREFIX:-detail}" --model-branch "${MODEL_BRANCH:-OMat24}"
