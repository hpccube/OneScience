#!/usr/bin/env bash

set -euo pipefail

: "${ROCM_PATH:?Load sghpc-mpi-gcc/26.3 before running Geneformer scripts}"
: "${CONDA_PREFIX:?Activate the bio_test conda environment before running Geneformer scripts}"

source "${ROCM_PATH}/cuda/env.sh"

GENEFORMER_PYTHON="${GENEFORMER_PYTHON:-python}"
GENEFORMER_SITE_PACKAGES="$(
    "${GENEFORMER_PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${GENEFORMER_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"

GENEFORMER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENEFORMER_EXAMPLE_DIR="$(cd "${GENEFORMER_SCRIPT_DIR}/.." && pwd)"
GENEFORMER_REPO_ROOT="$(cd "${GENEFORMER_EXAMPLE_DIR}/../../.." && pwd)"

source "${GENEFORMER_REPO_ROOT}/env.sh"

GENEFORMER_MODEL_ROOT="${GENEFORMER_MODEL_ROOT:-${ONESCIENCE_MODELS_DIR}/Geneformer}"
GENEFORMER_DATASET_ROOT="${GENEFORMER_DATASET_ROOT:-${ONESCIENCE_DATASETS_DIR}/Geneformer}"
GENEFORMER_V1_MODEL="${GENEFORMER_V1_MODEL:-${GENEFORMER_MODEL_ROOT}/Geneformer-V1-10M}"
GENEFORMER_V1_CELL_MODEL="${GENEFORMER_V1_CELL_MODEL:-${GENEFORMER_MODEL_ROOT}/fine_tuned_models/Geneformer-V1-10M_CellClassifier_cardiomyopathies_220224}"
GENEFORMER_V1_CORPUS="${GENEFORMER_V1_CORPUS:-${GENEFORMER_DATASET_ROOT}/Genecorpus-30M/genecorpus_30M_2048.dataset}"
GENEFORMER_V1_LENGTHS="${GENEFORMER_V1_LENGTHS:-${GENEFORMER_DATASET_ROOT}/Genecorpus-30M/genecorpus_30M_2048_lengths.pkl}"
GENEFORMER_CELL_DATA="${GENEFORMER_CELL_DATA:-${GENEFORMER_DATASET_ROOT}/Genecorpus-30M/example_input_files/cell_classification/disease_classification/human_dcm_hcm_nf.dataset}"
GENEFORMER_GENE_EXAMPLE_ROOT="${GENEFORMER_DATASET_ROOT}/Genecorpus-30M/example_input_files/gene_classification/dosage_sensitive_tfs"
GENEFORMER_GENE_DATA="${GENEFORMER_GENE_DATA:-${GENEFORMER_GENE_EXAMPLE_ROOT}/gc-30M_sample50k.dataset}"
GENEFORMER_GENE_CLASSES="${GENEFORMER_GENE_CLASSES:-${GENEFORMER_GENE_EXAMPLE_ROOT}/dosage_sensitivity_TFs.pickle}"
GENEFORMER_OUTPUT_ROOT="${GENEFORMER_OUTPUT_ROOT:-${GENEFORMER_EXAMPLE_DIR}/outputs}"

export PYTHONPATH="${GENEFORMER_REPO_ROOT}/src:${PYTHONPATH:-}"

geneformer_require_file() {
    local path="$1"
    if [[ ! -r "${path}" ]]; then
        echo "Required file is not readable: ${path}" >&2
        exit 2
    fi
}

geneformer_require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "Required directory does not exist: ${path}" >&2
        exit 2
    fi
}

geneformer_require_model() {
    local path="$1"
    geneformer_require_file "${path}/config.json"
    if [[ ! -r "${path}/model.safetensors" && ! -r "${path}/pytorch_model.bin" ]]; then
        echo "No model.safetensors or pytorch_model.bin found in ${path}" >&2
        exit 2
    fi
}

mkdir -p "${GENEFORMER_OUTPUT_ROOT}"
cd "${GENEFORMER_EXAMPLE_DIR}"
