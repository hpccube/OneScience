#!/usr/bin/env bash
set -euo pipefail

: "${ROCM_PATH:?Set ROCM_PATH before running State scripts}"
: "${CONDA_PREFIX:?Activate the target conda environment before running State scripts}"

source "${ROCM_PATH}/cuda/env.sh"

STATE_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="${STATE_SITE_PACKAGES}/fastpt/torch/lib:${LD_LIBRARY_PATH:-}"

STATE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_EXAMPLE_DIR="$(cd "${STATE_SCRIPT_DIR}/.." && pwd)"
STATE_REPO_ROOT="$(cd "${STATE_EXAMPLE_DIR}/../../.." && pwd)"
STATE_RUNNER_DIR="${STATE_SCRIPT_DIR}/runner"

source "${STATE_REPO_ROOT}/env.sh"

echo "ONESCIENCE_PATH:" "${STATE_REPO_ROOT}"
echo "${ONESCIENCE_DATASETS_DIR}"
echo "${ONESCIENCE_MODELS_DIR}"

cd "${STATE_EXAMPLE_DIR}"

export PYTHONPATH="${STATE_SITE_PACKAGES}:${PYTHONPATH:-}"

: "${ONESCIENCE_MODELS_DIR:?Set ONESCIENCE_MODELS_DIR to the directory containing State}"
: "${ONESCIENCE_DATASETS_DIR:?Set ONESCIENCE_DATASETS_DIR to the directory containing State_dataset}"

STATE_MODEL_ROOT="${ONESCIENCE_MODELS_DIR}/State"
STATE_DATASET_ROOT="${ONESCIENCE_DATASETS_DIR}/State_dataset"
STATE_OUTPUT_ROOT="${STATE_OUTPUT_DIR:-${STATE_EXAMPLE_DIR}/outputs}"
STATE_REPLOGLE_ROOT="${STATE_DATASET_ROOT}/Replogle-Nadig-Preprint"
STATE_SE_SMOKE_ROOT="${STATE_DATASET_ROOT}/SE-167M-Human-smoke"
STATE_SE600M_ROOT="${STATE_MODEL_ROOT}/SE-600M"

state_require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 2
    fi
}

state_require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 2
    fi
}

state_replogle_input() {
    case "${STATE_REPLOGLE_CELL_LINE:-hepg2}" in
        hepg2) printf '%s\n' "${STATE_REPLOGLE_ROOT}/GSE264667_hepg2_raw_singlecell_01.h5ad" ;;
        jurkat) printf '%s\n' "${STATE_REPLOGLE_ROOT}/GSE264667_jurkat_raw_singlecell_01.h5ad" ;;
        k562) printf '%s\n' "${STATE_REPLOGLE_ROOT}/K562_essential_normalized_singlecell_01.h5ad" ;;
        rpe1) printf '%s\n' "${STATE_REPLOGLE_ROOT}/rpe1_normalized_singlecell_01.h5ad" ;;
        *) echo "Unsupported STATE_REPLOGLE_CELL_LINE: ${STATE_REPLOGLE_CELL_LINE}" >&2; exit 2 ;;
    esac
}

state_select_checkpoint() {
    local model_dir="$1"
    local candidate
    if [[ -n "${STATE_ST_CHECKPOINT:-}" ]]; then
        state_require_file "${STATE_ST_CHECKPOINT}"
        printf '%s\n' "${STATE_ST_CHECKPOINT}"
        return
    fi
    for candidate in final.ckpt best.ckpt last.ckpt; do
        if [[ -f "${model_dir}/checkpoints/${candidate}" ]]; then
            printf '%s\n' "${model_dir}/checkpoints/${candidate}"
            return
        fi
    done
    echo "No checkpoint found under ${model_dir}/checkpoints" >&2
    exit 2
}

state_render_replogle_toml() {
    local data_dir="$1"
    local output="$2"
    local cell_line="${STATE_REPLOGLE_CELL_LINE:-hepg2}"
    local split_mode="${STATE_REPLOGLE_SPLIT_MODE:-fewshot}"
    local template
    if [[ "${split_mode}" == "fewshot" ]]; then
        template="${STATE_EXAMPLE_DIR}/configs/datasets/legacy_replogle_splits/${cell_line}.toml"
    elif [[ "${split_mode}" == "zeroshot" ]]; then
        template="${STATE_EXAMPLE_DIR}/configs/datasets/legacy_replogle_splits/${cell_line}_zeroshot.toml"
    else
        echo "STATE_REPLOGLE_SPLIT_MODE must be fewshot or zeroshot" >&2
        exit 2
    fi
    state_require_file "${template}"
    state_require_dir "${data_dir}"
    mkdir -p "$(dirname "${output}")"
    sed "s|/data/replogle_nogwps_v2|${data_dir}|g" "${template}" > "${output}"
}
