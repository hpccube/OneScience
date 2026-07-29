#!/usr/bin/env bash
set -euo pipefail

: "${ROCM_PATH:?Set ROCM_PATH before running Boltz scripts}"
: "${CONDA_PREFIX:?Activate the target conda environment before running Boltz scripts}"

source "${ROCM_PATH}/cuda/env.sh"

BOLTZ_PYTHON="${BOLTZ_PYTHON:-python}"
BOLTZ_SITE_PACKAGES="$("${BOLTZ_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${BOLTZ_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"

BOLTZ_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLTZ_EXAMPLE_DIR="$(cd "${BOLTZ_SCRIPT_DIR}/.." && pwd)"
BOLTZ_REPO_ROOT="$(cd "${BOLTZ_EXAMPLE_DIR}/../../.." && pwd)"

source "${BOLTZ_REPO_ROOT}/env.sh"

: "${ONESCIENCE_MODELS_DIR:?Set ONESCIENCE_MODELS_DIR before running Boltz scripts}"
: "${ONESCIENCE_DATASETS_DIR:?Set ONESCIENCE_DATASETS_DIR before running Boltz scripts}"

BOLTZ_MODEL_ROOT="${ONESCIENCE_MODELS_DIR}/Boltz"
BOLTZ_DATASET_ROOT="${ONESCIENCE_DATASETS_DIR}/Boltz_dataset"
BOLTZ_BOLTZ1_CHECKPOINT="${BOLTZ_MODEL_ROOT}/boltz1_conf.ckpt"
BOLTZ_BOLTZ2_CHECKPOINT="${BOLTZ_MODEL_ROOT}/boltz2_conf.ckpt"
BOLTZ_AFFINITY_CHECKPOINT="${BOLTZ_MODEL_ROOT}/boltz2_aff.ckpt"
BOLTZ_CCD_FILE="${BOLTZ_DATASET_ROOT}/ccd.pkl"
BOLTZ_MOLS_DIR="${BOLTZ_DATASET_ROOT}/mols"
BOLTZ_TRAIN_DATASET="${BOLTZ_TRAIN_DATASET:-rcsb}"
BOLTZ_TRAINING_ROOT="${BOLTZ_DATASET_ROOT}/training/${BOLTZ_TRAIN_DATASET}"
BOLTZ_DEFAULT_TARGET_DIR="${BOLTZ_TRAINING_ROOT}/targets"
BOLTZ_DEFAULT_MSA_DIR="${BOLTZ_TRAINING_ROOT}/msa"
BOLTZ_DEFAULT_SYMMETRIES="${BOLTZ_DATASET_ROOT}/training/symmetry.pkl"

# The legacy layout placed every benchmark input and output below one root.
# Keep that behavior as the default while allowing local Boltz predictions and
# evaluations to be combined with shared references and third-party results.
BOLTZ_SHARED_RESULTS_ROOT="${BOLTZ_SHARED_RESULTS_ROOT:-${BOLTZ_RESULTS_ROOT:-${BOLTZ_DATASET_ROOT}/boltz_results_final}}"
BOLTZ_LOCAL_RESULTS_ROOT="${BOLTZ_LOCAL_RESULTS_ROOT:-${BOLTZ_RESULTS_ROOT:-${BOLTZ_SHARED_RESULTS_ROOT}}}"
BOLTZ_LOCAL_EVAL_ROOT="${BOLTZ_LOCAL_EVAL_ROOT:-${BOLTZ_LOCAL_RESULTS_ROOT}/evals_local}"
BOLTZ_REPORT_ROOT="${BOLTZ_REPORT_ROOT:-${BOLTZ_LOCAL_RESULTS_ROOT}/aggregate_local}"

export BOLTZ_MODEL_ROOT BOLTZ_DATASET_ROOT
export BOLTZ_SHARED_RESULTS_ROOT BOLTZ_LOCAL_RESULTS_ROOT
export BOLTZ_LOCAL_EVAL_ROOT BOLTZ_REPORT_ROOT

export PYTHONPATH="${BOLTZ_SITE_PACKAGES}:${PYTHONPATH:-}"
cd "${BOLTZ_EXAMPLE_DIR}"

boltz_require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 2
    fi
}

boltz_require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 2
    fi
}
