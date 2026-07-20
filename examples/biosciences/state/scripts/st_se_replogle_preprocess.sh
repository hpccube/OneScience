#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

input="${STATE_REPLOGLE_INPUT:-$(state_replogle_input)}"
output="${STATE_ST_SE_INPUT:-${STATE_OUTPUT_ROOT}/st_se_replogle/data/$(basename "${input}" .h5ad)_x_state.h5ad}"
checkpoint="${STATE_SE600M_ROOT}/se600m_epoch16.ckpt"
protein_embeddings="${STATE_SE600M_ROOT}/protein_embeddings.pt"

state_require_file "${input}"
state_require_file "${checkpoint}"
state_require_file "${protein_embeddings}"
mkdir -p "$(dirname "${output}")"

exec python "${STATE_RUNNER_DIR}/transform_embedding.py" \
    --checkpoint "${checkpoint}" \
    --protein-embeddings "${protein_embeddings}" \
    --input "${input}" \
    --output "${output}" \
    --embed-key X_state \
    "${@}"
