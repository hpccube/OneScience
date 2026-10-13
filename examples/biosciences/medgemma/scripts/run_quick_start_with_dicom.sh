#!/bin/bash
# set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it}
# DICOM_PATH=${DICOM_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/COVID-19-AR/dcm/COVID-19-AR-16406488/1.3.6.1.4.1.14519.5.2.1.9999.103.1990954697724761389044939165366/01-17-2012/1.3.6.1.4.1.14519.5.2.1.9999.103.2772748514240373421475911810044/1.3.6.1.4.1.14519.5.2.1.9999.103.2536342667180710666597970295138.dcm}
DICOM_PATH=${DICOM_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/CHAOS/Test_Sets/demo/i0096,0000b.dcm}
PROMPT="${PROMPT:-You are an expert radiologist. Analyze this chest X-ray. Report only findings visible in the image. Use the sections FINDINGS and IMPRESSION. Keep the impression consistent with the findings; do not say the lungs are clear if you report an opacity. When a finding is uncertain, say indeterminate and do not invent a diagnosis. Before finalizing, check the report for internal contradictions.}"
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-500}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/quick_start_with_dicom}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
python ./notebook_conver/quick_start_with_dicom.py \
    --model_path "${MODEL_PATH}" \
    --dicom_path "${DICOM_PATH}" \
    --prompt "${PROMPT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"