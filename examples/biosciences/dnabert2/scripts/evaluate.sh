#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dnabert2_common.sh"

checkpoint_dir="${DNABERT2_CHECKPOINT_DIR:-${DNABERT2_OUTPUT_ROOT}/train}"
data_file="${DNABERT2_EVAL_DATA:-${DNABERT2_SAMPLE_DATA}/test.csv}"
base_model_dir="${DNABERT2_BASE_MODEL_DIR:-${DNABERT2_MODEL_ROOT}}"
dnabert2_require_dir "${checkpoint_dir}"
dnabert2_require_file "${data_file}"
dnabert2_require_dir "${base_model_dir}"
if ! compgen -G "${base_model_dir}/*.safetensors" >/dev/null \
    && ! compgen -G "${base_model_dir}/pytorch_model*.bin" >/dev/null; then
    echo "No model weights found in ${base_model_dir}" >&2
    exit 2
fi

exec "${DNABERT2_PYTHON}" scripts/evaluate.py \
    --checkpoint-dir "${checkpoint_dir}" \
    --base-model-dir "${base_model_dir}" \
    --data-file "${data_file}" \
    --output-dir "${DNABERT2_EVAL_OUTPUT:-${DNABERT2_OUTPUT_ROOT}/evaluation}" \
    "${@}"
