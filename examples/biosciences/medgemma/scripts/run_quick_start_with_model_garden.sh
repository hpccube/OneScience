#!/bin/bash
# set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

# Local/offline quick start. This script intentionally does not use Vertex AI,
# Google Cloud credentials, PROJECT_ID, REGION, or ENDPOINT_ID.
MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it}
IMAGE_PATH=${IMAGE_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (89).jpg}
PROMPT=${PROMPT:-Describe this X-ray.}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/quick_start_with_model_garden}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

echo "Using local model: ${MODEL_PATH}"
echo "Using local image: ${IMAGE_PATH}"

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
python ./notebook_conver/quick_start_with_model_garden.py \
    --model_path "${MODEL_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --prompt "${PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"