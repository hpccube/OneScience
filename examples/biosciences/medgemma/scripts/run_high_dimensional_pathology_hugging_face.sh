#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it}
IMAGE_DIR=${IMAGE_DIR:-${ONESCIENCE_DATASETS_DIR}/medgemma/pathology_patches}
H5_PATH=${H5_PATH:-}
MAX_PATCHES=${MAX_PATCHES:-8}
SAMPLE_SEED=${SAMPLE_SEED:-42}
PROMPT=${PROMPT:-$'Analyze only the pathology patch provided. Describe the visible morphology, give the most likely interpretation and a short differential diagnosis, and state the confidence and limitations. Do not assume findings that are not visible in the image.'}
TISSUE_CONTEXT="${TISSUE_CONTEXT:-Lymph node H&E patch from CamelyonPatch; assess for morphology suspicious for metastatic breast carcinoma.}"
INFERENCE_MODE=${INFERENCE_MODE:-per_patch}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/high_dimensional_pathology_hugging_face}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

INPUT_ARGS=()
if [ -n "${H5_PATH}" ]; then
    INPUT_ARGS+=(--h5_path "${H5_PATH}")
else
    INPUT_ARGS+=(--image_dir "${IMAGE_DIR}")
fi

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
python ./notebook_conver/high_dimensional_pathology_hugging_face.py \
    --model_path "${MODEL_PATH}" \
    --max_patches "${MAX_PATCHES}" \
    --seed "${SAMPLE_SEED}" \
    --prompt "${PROMPT}" \
    --tissue_context "${TISSUE_CONTEXT}" \
    --inference_mode "${INFERENCE_MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${INPUT_ARGS[@]}" \
    "$@"