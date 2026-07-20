#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

cell_line="${STATE_REPLOGLE_CELL_LINE:-hepg2}"
split_mode="${STATE_REPLOGLE_SPLIT_MODE:-fewshot}"
model_dir="${STATE_MODEL_ROOT}/ST-SE-Replogle/${split_mode}/${cell_line}"
checkpoint="$(state_select_checkpoint "${model_dir}")"
input="${STATE_ST_SE_INPUT:-${STATE_OUTPUT_ROOT}/st_se_replogle/data/$(basename "$(state_replogle_input)" .h5ad)_x_state.h5ad}"
output="${STATE_ST_SE_OUTPUT:-${STATE_OUTPUT_ROOT}/st_se_replogle/predictions/${cell_line}.h5ad}"

state_require_file "${input}"
mkdir -p "$(dirname "${output}")"

exec python "${STATE_RUNNER_DIR}/infer_transition.py" \
    --model-dir "${model_dir}" \
    --checkpoint "${checkpoint}" \
    --adata "${input}" \
    --embed-key X_state \
    --pert-col gene \
    --celltype-col cell_line \
    --batch-col gem_group \
    --control-pert non-targeting \
    --output "${output}" \
    "${@}"
