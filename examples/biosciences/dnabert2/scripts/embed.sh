#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dnabert2_common.sh"

dnabert2_require_model
input="${DNABERT2_INPUT:-inputs/sequences.fasta}"
dnabert2_require_file "${input}"

exec "${DNABERT2_PYTHON}" scripts/embed.py \
    --config "${DNABERT2_INFERENCE_CONFIG:-configs/inference.yaml}" \
    --model-dir "${DNABERT2_MODEL_ROOT}" \
    --input "${input}" \
    --output "${DNABERT2_EMBEDDING_OUTPUT:-${DNABERT2_OUTPUT_ROOT}/embeddings.npz}" \
    "${@}"
