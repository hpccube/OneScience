#!/bin/bash
# set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/model_garden/google--medgemma-27b-text-it/snapshots/master}
PARQUET_DIR=${PARQUET_DIR:-${ONESCIENCE_DATASETS_DIR}/medgemma/medqa}
RL_OUTPUT_DIR=${RL_OUTPUT_DIR:-./outputs/reinforcement_learning_with_hugging_face}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-256}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-64}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
USE_LORA=${USE_LORA:-0}

export USE_TF=0
export TRANSFORMERS_NO_TF=1
export WANDB_DISABLED=true
export MEDQA_PARQUET_DIR="${PARQUET_DIR}"
export BITSANDBYTES_NOWELCOME=1

mkdir -p "$(dirname "${RL_OUTPUT_DIR}")"
OUTPUT_DIR_ABS="$(cd "$(dirname "${RL_OUTPUT_DIR}")" && pwd)/$(basename "${RL_OUTPUT_DIR}")"

echo "Using local model: ${MODEL_PATH}"
echo "Using local MedQA parquet: ${PARQUET_DIR}"
echo "Output dir: ${OUTPUT_DIR_ABS}"
echo "Use LoRA: ${USE_LORA}"

EXTRA_ARGS=()
if [[ "${USE_LORA}" == "1" ]]; then
    EXTRA_ARGS+=(--use_lora)
fi

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
${LAUNCHER:-python} ./notebook_conver/reinforcement_learning_with_hugging_face.py \
    --model_path "${MODEL_PATH}" \
    --parquet_dir "${PARQUET_DIR}" \
    --output_dir "${RL_OUTPUT_DIR}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
