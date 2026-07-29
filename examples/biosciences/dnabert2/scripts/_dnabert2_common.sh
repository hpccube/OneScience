#!/usr/bin/env bash
set -euo pipefail

: "${ROCM_PATH:?Set ROCM_PATH before running DNABERT-2 scripts}"
: "${CONDA_PREFIX:?Activate the target conda environment before running DNABERT-2 scripts}"

source "${ROCM_PATH}/cuda/env.sh"

DNABERT2_PYTHON="${DNABERT2_PYTHON:-python}"
DNABERT2_SITE_PACKAGES="$("${DNABERT2_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${DNABERT2_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"

DNABERT2_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DNABERT2_EXAMPLE_DIR="$(cd "${DNABERT2_SCRIPT_DIR}/.." && pwd)"
DNABERT2_REPO_ROOT="$(cd "${DNABERT2_EXAMPLE_DIR}/../../.." && pwd)"

source "${DNABERT2_REPO_ROOT}/env.sh"

: "${ONESCIENCE_MODELS_DIR:?Set ONESCIENCE_MODELS_DIR before running DNABERT-2 scripts}"
: "${ONESCIENCE_DATASETS_DIR:?Set ONESCIENCE_DATASETS_DIR before running DNABERT-2 scripts}"

DNABERT2_MODEL_ROOT="${ONESCIENCE_MODELS_DIR}/DNABERT-2"
DNABERT2_DATASET_ROOT="${ONESCIENCE_DATASETS_DIR}/DNABERT-2_dataset"
DNABERT2_SAMPLE_DATA="${DNABERT2_DATASET_ROOT}/sample_data"
DNABERT2_GUE_ROOT="${DNABERT2_DATASET_ROOT}/GUE"
DNABERT2_OUTPUT_ROOT="${DNABERT2_OUTPUT_DIR:-${DNABERT2_EXAMPLE_DIR}/outputs}"

export DNABERT2_MODEL_ROOT DNABERT2_DATASET_ROOT
export PYTHONPATH="${DNABERT2_SITE_PACKAGES}:${PYTHONPATH:-}"
cd "${DNABERT2_EXAMPLE_DIR}"

dnabert2_require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 2
    fi
}

dnabert2_require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 2
    fi
}

dnabert2_require_model() {
    dnabert2_require_dir "${DNABERT2_MODEL_ROOT}"
    dnabert2_require_file "${DNABERT2_MODEL_ROOT}/config.json"
    if ! compgen -G "${DNABERT2_MODEL_ROOT}/*.safetensors" >/dev/null \
        && ! compgen -G "${DNABERT2_MODEL_ROOT}/pytorch_model*.bin" >/dev/null; then
        echo "No model weights found in ${DNABERT2_MODEL_ROOT}" >&2
        exit 2
    fi
}
