#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_geneformer_common.sh"

geneformer_require_dir "${GENEFORMER_V1_CORPUS}"
geneformer_require_file "${GENEFORMER_V1_LENGTHS}"

exec "${GENEFORMER_PYTHON}" scripts/pretrain.py \
    --data-file "${GENEFORMER_V1_CORPUS}" \
    --lengths-file "${GENEFORMER_V1_LENGTHS}" \
    --output-dir "${GENEFORMER_OUTPUT_ROOT}/pretrain" \
    "$@"
