#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

config_file="${STATE_SE_CONFIG:-${STATE_OUTPUT_ROOT}/se_smoke/config.yaml}"
profile="${STATE_SE_PROFILE:-se_human_smoke}"
state_require_file "${config_file}"

exec python "${STATE_RUNNER_DIR}/train_embedding.py" \
    --conf "${config_file}" \
    "embeddings.current=${profile}" \
    "dataset.current=${profile}" \
    "${@}"
