#!/bin/bash
# set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it}
CT_DICOM_DIR=${CT_DICOM_DIR:-${ONESCIENCE_DATASETS_DIR}/medgemma/CTLM}
IMAGE_DIR=${IMAGE_DIR:-}
CT_MAX_SLICES=${CT_MAX_SLICES:-85}
CT_STUDY_INSTANCE_UID=${CT_STUDY_INSTANCE_UID:-1.3.6.1.4.1.14519.5.2.1.9203.8273.982856921320609617394372605436}
CT_SERIES_INSTANCE_UID=${CT_SERIES_INSTANCE_UID:-1.3.6.1.4.1.14519.5.2.1.9203.8273.275179554444442893192427753220}
CT_OUTPUT_DIR=${CT_OUTPUT_DIR:-./outputs/high_dimensional_ct_hugging_face}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}

INPUT_ARGS=()
if [[ -n "${IMAGE_DIR}" ]]; then
    INPUT_ARGS+=(--image_dir "${IMAGE_DIR}")
else
    INPUT_ARGS+=(--dicom_dir "${CT_DICOM_DIR}")
fi

PROMPT_ARGS=()
if [[ -n "${CT_PROMPT:-}" ]]; then
    PROMPT_ARGS+=(--prompt "${CT_PROMPT}")
fi
if [[ -n "${CT_INSTRUCTION:-}" ]]; then
    PROMPT_ARGS+=(--instruction "${CT_INSTRUCTION}")
fi

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
python ./notebook_conver/high_dimensional_ct_hugging_face.py \
    --model_path "${MODEL_PATH}" \
    "${INPUT_ARGS[@]}" \
    --max_slices "${CT_MAX_SLICES}" \
    --study_instance_uid "${CT_STUDY_INSTANCE_UID}" \
    --series_instance_uid "${CT_SERIES_INSTANCE_UID}" \
    --output_dir "${CT_OUTPUT_DIR}" \
    "${PROMPT_ARGS[@]}" \
    "$@"