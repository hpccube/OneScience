#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

: "${ONESCIENCE_DATASETS_DIR:?Set ONESCIENCE_DATASETS_DIR to the directory containing State_dataset}"
export ONESCIENCE_MODELS_DIR="${ONESCIENCE_MODELS_DIR:-${ONESCIENCE_DATASETS_DIR}}"

STATE_REPO_ROOT="$(cd "${STATE_EXAMPLE_DIR}/../../.." && pwd)"

cell_line="${STATE_REPLOGLE_CELL_LINE:-hepg2}"
split_mode="${STATE_REPLOGLE_SPLIT_MODE:-fewshot}"
model_dir="${STATE_MODEL_ROOT}/ST-HVG-Replogle/${split_mode}/${cell_line}"
predict_input="${STATE_ST_HVG_PREDICT_INPUT:-${STATE_REPLOGLE_ROOT}/replogle.h5ad}"
data_dir="${STATE_ST_HVG_PREDICT_DATA_DIR:-${STATE_OUTPUT_ROOT}/st_hvg_replogle/runtime_data/replogle_2024}"
toml="${STATE_OUTPUT_ROOT}/st_hvg_replogle/configs/${split_mode}_${cell_line}.toml"
checkpoint="$(state_select_checkpoint "${model_dir}")"
state_require_file "${predict_input}"
mkdir -p "${data_dir}"
ln -sfnT "${predict_input}" "${data_dir}/replogle.h5ad"
state_render_replogle_toml "${data_dir}" "${toml}"

runtime_model_dir="${STATE_OUTPUT_ROOT}/st_hvg_replogle/runtime_models/${split_mode}/${cell_line}"
runtime_model_parent="$(dirname "${runtime_model_dir}")"
runtime_model_name="$(basename "${runtime_model_dir}")"
runtime_config="${runtime_model_dir}/config.yaml"
runtime_data_module="${runtime_model_dir}/data_module.torch"

state_require_file "${model_dir}/config.yaml"
state_require_file "${model_dir}/data_module.torch"
mkdir -p "${runtime_model_dir}"

python - \
    "${model_dir}/config.yaml" \
    "${model_dir}/data_module.torch" \
    "${runtime_config}" \
    "${runtime_data_module}" \
    "${runtime_model_parent}" \
    "${runtime_model_name}" \
    "${toml}" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

source_config, source_data_module, runtime_config, runtime_data_module, output_dir, name, toml = sys.argv[1:]

with open(source_config, "r") as config_file:
    config = yaml.safe_load(config_file)
config["output_dir"] = output_dir
config["name"] = name
config["data"]["kwargs"]["toml_config_path"] = toml

Path(runtime_config).parent.mkdir(parents=True, exist_ok=True)
with open(runtime_config, "w") as config_file:
    yaml.safe_dump(config, config_file, sort_keys=False)

data_module = torch.load(source_data_module, map_location="cpu", weights_only=False)
data_module["toml_config_path"] = toml
torch.save(data_module, runtime_data_module)
PY

for artifact in checkpoints var_dims.pkl pert_onehot_map.pt batch_onehot_map.pkl; do
    if [[ -e "${model_dir}/${artifact}" ]]; then
        ln -sfnT "${model_dir}/${artifact}" "${runtime_model_dir}/${artifact}"
    fi
done

exec python "${STATE_RUNNER_DIR}/predict_transition.py" \
    --output-dir "${runtime_model_dir}" \
    --checkpoint "$(basename "${checkpoint}")" \
    --toml "${toml}" \
    "${@}"
