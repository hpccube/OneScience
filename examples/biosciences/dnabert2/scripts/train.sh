#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dnabert2_common.sh"

dnabert2_require_model
data_dir="${DNABERT2_TRAIN_DATA:-${DNABERT2_SAMPLE_DATA}}"
dnabert2_require_file "${data_dir}/train.csv"
dnabert2_require_file "${data_dir}/dev.csv"
dnabert2_require_file "${data_dir}/test.csv"

exec "${DNABERT2_PYTHON}" scripts/train.py \
    --config "${DNABERT2_TRAIN_CONFIG:-configs/train.yaml}" \
    --model-dir "${DNABERT2_MODEL_ROOT}" \
    --data-dir "${data_dir}" \
    --output-dir "${DNABERT2_TRAIN_OUTPUT:-${DNABERT2_OUTPUT_ROOT}/train}" \
    "${@}"
