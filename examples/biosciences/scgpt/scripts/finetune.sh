#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_scgpt_common.sh"

scgpt_require_model
scgpt_require_file "${SCGPT_FINETUNE_DATA}"

scgpt_launch scripts/finetune.py \
    --model-dir "${SCGPT_MODEL_DIR}" \
    --data-file "${SCGPT_FINETUNE_DATA}" \
    --label-column "Celltype" \
    --output-dir "${SCGPT_OUTPUT_ROOT}/pancreas_finetune" \
    --device "${SCGPT_DEVICE}" \
    "$@"
