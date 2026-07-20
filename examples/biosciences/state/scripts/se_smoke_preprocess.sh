#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

data_dir="${STATE_SE_SMOKE_ROOT}/19k_human_filtered_scbasecount"
profile_root="${STATE_OUTPUT_ROOT}/se_smoke/profile"
manifest_dir="${STATE_OUTPUT_ROOT}/se_smoke/manifests"
config_file="${STATE_SE_CONFIG:-${STATE_OUTPUT_ROOT}/se_smoke/config.yaml}"
all_embeddings="${STATE_SE_SMOKE_ROOT}/all_embeddings_Preprint-SE-167M-Human.pt"

state_require_dir "${data_dir}"
state_require_file "${all_embeddings}"
mkdir -p "$(dirname "${config_file}")"
if [[ ! -f "${config_file}" ]]; then
    cp "${STATE_EXAMPLE_DIR}/configs/embedding/state-defaults.yaml" "${config_file}"
fi

python "${STATE_RUNNER_DIR}/build_embedding_manifests.py" \
    --data-dir "${data_dir}" \
    --output-dir "${manifest_dir}"

exec python "${STATE_RUNNER_DIR}/preprocess_embedding.py" \
    --profile-name "${STATE_SE_PROFILE:-se_human_smoke}" \
    --train-csv "${manifest_dir}/train.csv" \
    --val-csv "${manifest_dir}/val.csv" \
    --output-dir "${profile_root}" \
    --config-file "${config_file}" \
    --all-embeddings "${all_embeddings}" \
    "${@}"
