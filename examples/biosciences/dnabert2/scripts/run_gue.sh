#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dnabert2_common.sh"

dnabert2_require_model
dnabert2_require_dir "${DNABERT2_GUE_ROOT}"

exec "${DNABERT2_PYTHON}" scripts/run_gue.py \
    --model-dir "${DNABERT2_MODEL_ROOT}" \
    --dataset-root "${DNABERT2_DATASET_ROOT}" \
    --output-root "${DNABERT2_GUE_OUTPUT:-${DNABERT2_OUTPUT_ROOT}/gue}" \
    "${@}"
