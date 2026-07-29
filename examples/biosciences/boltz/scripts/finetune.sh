#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_boltz_common.sh"

BOLTZ_TARGET_DIR="${BOLTZ_TARGET_DIR:-${BOLTZ_DEFAULT_TARGET_DIR}}"
BOLTZ_MSA_DIR="${BOLTZ_MSA_DIR:-${BOLTZ_DEFAULT_MSA_DIR}}"
BOLTZ_SYMMETRIES="${BOLTZ_SYMMETRIES:-${BOLTZ_DEFAULT_SYMMETRIES}}"
BOLTZ_CHECKPOINT="${BOLTZ_CHECKPOINT:-${BOLTZ_BOLTZ1_CHECKPOINT}}"

boltz_require_dir "${BOLTZ_TARGET_DIR}"
boltz_require_dir "${BOLTZ_MSA_DIR}"
boltz_require_file "${BOLTZ_SYMMETRIES}"
boltz_require_file "${BOLTZ_CHECKPOINT}"
boltz_require_file "${BOLTZ_TARGET_DIR}/manifest.json"
boltz_require_dir "${BOLTZ_TARGET_DIR}/structures"
config="${BOLTZ_TRAIN_CONFIG:-configs/train/full.yaml}"
boltz_require_file "${config}"

exec "${BOLTZ_PYTHON}" scripts/finetune.py "${config}" "${BOLTZ_CHECKPOINT}" \
    "output=${BOLTZ_OUTPUT_DIR:-outputs/finetune}" \
    "data.datasets.0.target_dir=${BOLTZ_TARGET_DIR}" \
    "data.datasets.0.msa_dir=${BOLTZ_MSA_DIR}" \
    "data.symmetries=${BOLTZ_SYMMETRIES}" \
    "${@}"
