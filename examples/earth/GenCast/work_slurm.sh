#!/bin/bash
#SBATCH -p normal
#SBATCH -N 2
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH -J GenCast
#SBATCH --time=72:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --exclusive

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "${PROJECT_DIR}" ]]; then
  PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fi
PROJECT_DIR=$(cd -- "${PROJECT_DIR}" && pwd)
WORKSPACE_DIR=$(cd -- "${PROJECT_DIR}/../../.." && pwd)
ONESCIENCE_SRC=${ONESCIENCE_SRC:-${WORKSPACE_DIR}/onescience/src}
NUM_DEVICES=${NUM_DEVICES:-8}
NUM_PROCESSES=${NUM_PROCESSES:-${SLURM_NTASKS:-2}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_DEVICES * NUM_PROCESSES))}
MAX_STEPS=${MAX_STEPS:-10}
CONFIG=${CONFIG:-conf/config.yaml}

cd "${PROJECT_DIR}"

if [[ ! -f "${ONESCIENCE_SRC}/onescience/models/gencast/__init__.py" ]]; then
  echo "GenCast OneScience package not found under ${ONESCIENCE_SRC}" >&2
  exit 1
fi

echo "START TIME: $(date)"
echo "HOST: $(hostname)"
echo "NUM_DEVICES: ${NUM_DEVICES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}"

module purge
module load sghpcdas/25.6
module load sghpc-mpi-gcc/26.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gencast_develop

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONPATH="${ONESCIENCE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY all_proxy
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eno1}
export JAX_NUM_PROCESSES=${NUM_PROCESSES}
export JAX_COORDINATOR_ADDRESS=${JAX_COORDINATOR_ADDRESS:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}
export JAX_COORDINATOR_PORT=${JAX_COORDINATOR_PORT:-12355}

srun --ntasks="${NUM_PROCESSES}" --ntasks-per-node=1 -u python train.py \
  --config "${CONFIG}" \
  --parallel-mode pmap \
  --num-devices "${NUM_DEVICES}" \
  --num-processes "${NUM_PROCESSES}" \
  --coordinator-address "${JAX_COORDINATOR_ADDRESS}" \
  --coordinator-port "${JAX_COORDINATOR_PORT}" \
  --max-steps "${MAX_STEPS}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}"

echo "END TIME: $(date)"
