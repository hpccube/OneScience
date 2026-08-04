#!/bin/bash
# set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it}
IMAGE_PATH=${IMAGE_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (89).jpg}
PROMPT=${PROMPT:-Describe this X-ray.}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/quick_start_with_hugging_face}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
python ./notebook_conver/quick_start_with_hugging_face.py \
    --model_path "${MODEL_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
