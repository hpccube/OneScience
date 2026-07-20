#!/usr/bin/env bash
set -euo pipefail

# Prepare the X_state AnnData produced by st_se_replogle_preprocess.sh for
# ST-SE training with data.kwargs.output_space=all.
#
# This script intentionally delegates the conversion to the project's existing
# preprocess_transition.py runner. That runner directly calls the official
# STATE run_tx_preprocess_train implementation, whose standard action is:
#   scanpy.pp.normalize_total -> scanpy.pp.log1p -> highly_variable_genes
#   -> store the selected matrix in adata.obsm["X_hvg"] -> write the h5ad.
#
# In output_space=all mode, training uses the normalized/log1p adata.X as the
# full-gene target. X_hvg is still generated because it is part of the official
# preprocessing action, but it is not used as the all-space training target.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_state_common.sh"

source_h5ad="${STATE_ST_SE_ALL_INPUT:-${STATE_OUTPUT_ROOT}/st_se_replogle/data/$(basename "$(state_replogle_input)" .h5ad)_x_state.h5ad}"
output_dir="${STATE_ST_SE_ALL_DATA_DIR:-${STATE_OUTPUT_ROOT}/st_se_replogle/data_all}"
output_h5ad="${STATE_ST_SE_ALL_OUTPUT:-${output_dir}/$(basename "${source_h5ad}")}"
num_hvgs="${STATE_NUM_HVGS:-2000}"

state_require_file "${source_h5ad}"

if [[ "$(readlink -m "${source_h5ad}")" == "$(readlink -m "${output_h5ad}")" ]]; then
    echo "Input and output must be different files: ${source_h5ad}" >&2
    exit 2
fi

python - "${source_h5ad}" <<'PY'
import sys
import anndata as ad

path = sys.argv[1]
adata = ad.read_h5ad(path, backed="r")
if "X_state" not in adata.obsm:
    raise KeyError(f"obsm['X_state'] not found in {path}")
print(f"Input: {path}")
print(f"X_state shape: {adata.obsm['X_state'].shape}")
PY

mkdir -p "$(dirname "${output_h5ad}")"

echo "Running official STATE transition preprocessing"
echo "Output: ${output_h5ad}"
echo "Number of HVGs: ${num_hvgs}"

python "${STATE_RUNNER_DIR}/preprocess_transition.py" \
    --adata "${source_h5ad}" \
    --output "${output_h5ad}" \
    --num_hvgs "${num_hvgs}"

python - "${output_h5ad}" <<'PY'
import sys
import anndata as ad

path = sys.argv[1]
adata = ad.read_h5ad(path, backed="r")
missing = [key for key in ("X_state", "X_hvg") if key not in adata.obsm]
if missing:
    raise KeyError(f"Missing required obsm keys after preprocessing: {missing}")
if "log1p" not in adata.uns:
    raise KeyError("Official preprocessing completed without expected uns['log1p'] metadata")

print("Preparation complete")
print(f"AnnData shape: {adata.shape}")
print(f"X_state shape: {adata.obsm['X_state'].shape}")
print(f"X_hvg shape: {adata.obsm['X_hvg'].shape}")
print("Use this directory for ST-SE all-space training:")
print(str(__import__("pathlib").Path(path).parent))
PY
