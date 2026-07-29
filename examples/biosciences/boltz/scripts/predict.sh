#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_boltz_common.sh"

boltz_require_dir "${BOLTZ_MODEL_ROOT}"
boltz_require_dir "${BOLTZ_DATASET_ROOT}"

input="${1:-inputs/prot_no_msa.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi
boltz_require_file "${input}"

exec "${BOLTZ_PYTHON}" scripts/predict.py "${input}" \
    --out_dir "${BOLTZ_OUTPUT_DIR:-outputs/predict}" \
    --model_dir "${BOLTZ_MODEL_ROOT}" \
    --dataset_dir "${BOLTZ_DATASET_ROOT}" \
    "${@}"
