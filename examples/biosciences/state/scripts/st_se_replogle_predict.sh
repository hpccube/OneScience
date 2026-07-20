#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

cell_line="${STATE_REPLOGLE_CELL_LINE:-hepg2}"
split_mode="${STATE_REPLOGLE_SPLIT_MODE:-fewshot}"
model_dir="${STATE_MODEL_ROOT}/ST-SE-Replogle/${split_mode}/${cell_line}"
data_dir="${STATE_ST_SE_PREDICT_DATA_DIR:-${STATE_OUTPUT_ROOT}/st_se_replogle/data}"
toml="${STATE_OUTPUT_ROOT}/st_se_replogle/configs/${split_mode}_${cell_line}.toml"
checkpoint="$(state_select_checkpoint "${model_dir}")"
state_render_replogle_toml "${data_dir}" "${toml}"

exec python "${STATE_RUNNER_DIR}/predict_transition.py" \
    --output-dir "${model_dir}" \
    --checkpoint "$(basename "${checkpoint}")" \
    --toml "${toml}" \
    "${@}"
