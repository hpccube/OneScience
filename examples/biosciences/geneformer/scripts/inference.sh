#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_geneformer_common.sh"

geneformer_require_model "${GENEFORMER_V1_CELL_MODEL}"
geneformer_require_dir "${GENEFORMER_CELL_DATA}"

exec "${GENEFORMER_PYTHON}" scripts/extract_embeddings.py \
    --model-dir "${GENEFORMER_V1_CELL_MODEL}" \
    --data-file "${GENEFORMER_CELL_DATA}" \
    --output-dir "${GENEFORMER_OUTPUT_ROOT}/embeddings" \
    --output-prefix cardiomyopathy_cell_embeddings \
    --model-version V1 \
    --model-type CellClassifier \
    --num-classes 3 \
    --emb-mode cell \
    --emb-layer 0 \
    --label disease \
    --label cell_type \
    "$@"
