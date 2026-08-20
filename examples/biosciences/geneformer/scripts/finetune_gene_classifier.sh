#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_geneformer_common.sh"

geneformer_require_model "${GENEFORMER_V1_MODEL}"
geneformer_require_dir "${GENEFORMER_GENE_DATA}"
geneformer_require_file "${GENEFORMER_GENE_CLASSES}"

exec "${GENEFORMER_PYTHON}" scripts/finetune_gene_classifier.py \
    --model-dir "${GENEFORMER_V1_MODEL}" \
    --data-file "${GENEFORMER_GENE_DATA}" \
    --gene-class-dict "${GENEFORMER_GENE_CLASSES}" \
    --output-dir "${GENEFORMER_OUTPUT_ROOT}/gene_finetune" \
    --output-prefix tf_dosage_sensitivity \
    --model-version V1 \
    "$@"
