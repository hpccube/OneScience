#!/usr/bin/env bash

set -euo pipefail

: "${ROCM_PATH:?Set ROCM_PATH before running scGPT scripts}"
: "${CONDA_PREFIX:?Activate the target conda environment before running scGPT scripts}"

source "${ROCM_PATH}/cuda/env.sh"

SCGPT_PYTHON="${SCGPT_PYTHON:-python}"
SCGPT_SITE_PACKAGES="$("${SCGPT_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SCGPT_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"

SCGPT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCGPT_EXAMPLE_DIR="$(cd "${SCGPT_SCRIPT_DIR}/.." && pwd)"
SCGPT_REPO_ROOT="$(cd "${SCGPT_EXAMPLE_DIR}/../../.." && pwd)"

source "${SCGPT_REPO_ROOT}/env.sh"

: "${ONESCIENCE_MODELS_DIR:?Set ONESCIENCE_MODELS_DIR before running scGPT scripts}"
: "${ONESCIENCE_DATASETS_DIR:?Set ONESCIENCE_DATASETS_DIR before running scGPT scripts}"

SCGPT_MODEL_ROOT="${SCGPT_MODEL_ROOT:-${ONESCIENCE_MODELS_DIR}/scGPT}"
SCGPT_DATASET_ROOT="${SCGPT_DATASET_ROOT:-${ONESCIENCE_DATASETS_DIR}/scGPT}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-${SCGPT_MODEL_ROOT}/scGPT_human}"
SCGPT_PANCREAS_DIR="${SCGPT_PANCREAS_DIR:-${SCGPT_DATASET_ROOT}/annotation_pancreas}"
SCGPT_INFERENCE_DATA="${SCGPT_INFERENCE_DATA:-${SCGPT_PANCREAS_DIR}/demo_test.h5ad}"
SCGPT_FINETUNE_DATA="${SCGPT_FINETUNE_DATA:-${SCGPT_PANCREAS_DIR}/demo_train.h5ad}"
SCGPT_OUTPUT_ROOT="${SCGPT_OUTPUT_ROOT:-${SCGPT_EXAMPLE_DIR}/outputs}"
SCGPT_DEVICE="${SCGPT_DEVICE:-cuda}"
SCGPT_TORCHRUN="${SCGPT_TORCHRUN:-torchrun}"

export SCGPT_MODEL_ROOT SCGPT_DATASET_ROOT
export PYTHONPATH="${SCGPT_REPO_ROOT}/src:${SCGPT_SITE_PACKAGES}:${PYTHONPATH:-}"

scgpt_require_file() {
    local path="$1"
    if [[ ! -r "${path}" ]]; then
        echo "Required file is not readable: ${path}" >&2
        exit 2
    fi
}

scgpt_require_model() {
    scgpt_require_file "${SCGPT_MODEL_DIR}/args.json"
    scgpt_require_file "${SCGPT_MODEL_DIR}/best_model.pt"
    scgpt_require_file "${SCGPT_MODEL_DIR}/vocab.json"
}

scgpt_launch() {
    local program="$1"
    local num_devices
    shift
    num_devices="$("${SCGPT_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
    if [[ "${SCGPT_DEVICE}" == cuda* && "${num_devices}" -lt 1 ]]; then
        echo "No CUDA/DTK device is visible to PyTorch" >&2
        exit 2
    fi
    if [[ "${SCGPT_DEVICE}" == cuda* && "${num_devices}" -gt 1 ]]; then
        echo "Detected ${num_devices} visible devices; launching distributed scGPT"
        exec "${SCGPT_TORCHRUN}" \
            --standalone \
            --nproc-per-node "${num_devices}" \
            "${program}" "$@"
    fi
    exec "${SCGPT_PYTHON}" "${program}" "$@"
}

mkdir -p "${SCGPT_OUTPUT_ROOT}"
cd "${SCGPT_EXAMPLE_DIR}"
