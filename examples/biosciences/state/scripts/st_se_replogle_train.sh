#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

data_dir="${STATE_ST_SE_DATA_DIR:-${STATE_OUTPUT_ROOT}/st_se_replogle/data}"
toml="${STATE_OUTPUT_ROOT}/st_se_replogle/configs/train.toml"
state_render_replogle_toml "${data_dir}" "${toml}"

exec python "${STATE_RUNNER_DIR}/train_transition.py" \
    "data.kwargs.toml_config_path=${toml}" \
    data.kwargs.embed_key=X_state \
    data.kwargs.output_space=all \
    data.kwargs.pert_col=gene \
    data.kwargs.cell_type_key=cell_line \
    data.kwargs.batch_col=gem_group \
    data.kwargs.control_pert=non-targeting \
    model=state \
    use_wandb=false \
    "output_dir=${STATE_OUTPUT_ROOT}/st_se_replogle/runs" \
    "name=${STATE_ST_RUN_NAME:-st_se_replogle}" \
    "${@}"
