#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_scgpt_common.sh"

scgpt_require_model
scgpt_require_file "${SCGPT_INFERENCE_DATA}"

scgpt_launch scripts/embed.py \
    --model-dir "${SCGPT_MODEL_DIR}" \
    --data-file "${SCGPT_INFERENCE_DATA}" \
    --output "${SCGPT_OUTPUT_ROOT}/pancreas_embeddings.h5ad" \
    --device "${SCGPT_DEVICE}" \
    "$@"
