#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

input="${STATE_REPLOGLE_INPUT:-$(state_replogle_input)}"
output="${STATE_ST_HVG_INPUT:-${STATE_OUTPUT_ROOT}/st_hvg_replogle/data/$(basename "${input}" .h5ad)_x_hvg.h5ad}"
state_require_file "${input}"
mkdir -p "$(dirname "${output}")"

exec python "${STATE_RUNNER_DIR}/preprocess_transition.py" \
    --adata "${input}" \
    --output "${output}" \
    --num_hvgs "${STATE_NUM_HVGS:-2000}" \
    "${@}"
