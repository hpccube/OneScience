#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_geneformer_common.sh"

geneformer_require_model "${GENEFORMER_V1_MODEL}"
geneformer_require_dir "${GENEFORMER_CELL_DATA}"

exec "${GENEFORMER_PYTHON}" scripts/perturb.py \
    --model-dir "${GENEFORMER_V1_MODEL}" \
    --data-file "${GENEFORMER_CELL_DATA}" \
    --output-dir "${GENEFORMER_OUTPUT_ROOT}/perturb" \
    --output-prefix cardiomyopathy_delete \
    --model-version V1 \
    --model-type Pretrained \
    --emb-mode cell \
    "$@"
