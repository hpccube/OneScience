#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_geneformer_common.sh"

geneformer_require_model "${GENEFORMER_V1_MODEL}"
geneformer_require_dir "${GENEFORMER_CELL_DATA}"

exec "${GENEFORMER_PYTHON}" scripts/finetune.py \
    --model-dir "${GENEFORMER_V1_MODEL}" \
    --data-file "${GENEFORMER_CELL_DATA}" \
    --output-dir "${GENEFORMER_OUTPUT_ROOT}/finetune" \
    --output-prefix cardiomyopathy_disease \
    --state-column disease \
    --filter-column cell_type \
    --filter-value Cardiomyocyte1 \
    --filter-value Cardiomyocyte2 \
    --filter-value Cardiomyocyte3 \
    --model-version V1 \
    "$@"
