#!/bin/bash
#set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
source "${REPO_ROOT}/env.sh"
cd "${SCRIPT_DIR}"
pwd

LLM_BACKEND=${LLM_BACKEND:-local}
MODEL_PATH=${MODEL_PATH:-${ONESCIENCE_DATASETS_DIR}/medgemma/navigator/model_garden/google--medgemma-27b-text-it/snapshots/master}
DEVICE_MAP=${DEVICE_MAP:-auto}
TORCH_DTYPE=${TORCH_DTYPE:-auto}
PROJECT_ID=${PROJECT_ID:-hai-cd3-foundations}
REGION=${REGION:-us-central1}
ENDPOINT_ID=${ENDPOINT_ID:-1030}
FHIR_STORE_URL=${FHIR_STORE_URL:-}
DEFAULT_PATIENT_ID=auto
PATIENT_ID=${EHR_PATIENT_ID:-${PATIENT_ID_OVERRIDE:-${DEFAULT_PATIENT_ID}}}
QUESTION=${QUESTION:-What specific medications were administered to the patient during their sepsis encounter?}
OUTPUT_DIR=${OUTPUT_DIR:-./ehr_navigator_outputs}
FHIR_DATA_DIR=${FHIR_DATA_DIR:-${ONESCIENCE_DATASETS_DIR}/medgemma/navigator/fhir}

if [[ "${PATIENT_ID}" == "auto" ]]; then
    echo "Scanning FHIR data for a patient with sepsis-related medication administrations..."
    PATIENT_ID=$(python ./notebook_conver/find_sepsis_medication_patient.py --fhir_data_dir "${FHIR_DATA_DIR}" --fallback-any-medication --first)
fi

echo "Using patient ID: ${PATIENT_ID}"

# ============================================================
# EHR 导航智能体运行脚本
# 
# 用途：根据指定患者的结构化医疗数据（FHIR 格式），
#       利用大语言模型回答用户提出的问题。
#
# 主要参数（通过环境变量或脚本开头设置）：
#   LLM_BACKEND   : 大模型后端 (transformers / vertexai)
#   MODEL_PATH    : 本地模型路径 (仅 transformers 后端需要)
#   DEVICE_MAP    : 设备分配 (auto / cuda:0 / cpu)
#   TORCH_DTYPE   : 模型精度 (float16 / bfloat16 / float32)
#   PROJECT_ID    : 谷歌云项目 ID (仅 Vertex AI 需要)
#   REGION        : 谷歌云区域 (仅 Vertex AI 需要)
#   ENDPOINT_ID   : 端点 ID (仅 Vertex AI 需要)
#   FHIR_STORE_URL: 远程 FHIR 数据仓库地址 (可选)
#   FHIR_DATA_DIR : 本地 FHIR 数据文件夹 (可选，用于离线分析)
#   PATIENT_ID    : 要查询的患者编号
#   QUESTION      : 需要咨询的具体问题
#   OUTPUT_DIR    : 结果保存目录
# ============================================================

python ./notebook_conver/ehr_navigator_agent.py \
    --llm_backend "${LLM_BACKEND}" \
    --model_path "${MODEL_PATH}" \
    --device_map "${DEVICE_MAP}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --project_id "${PROJECT_ID}" \
    --region "${REGION}" \
    --endpoint_id "${ENDPOINT_ID}" \
    --fhir_store_url "${FHIR_STORE_URL}" \
    ${FHIR_DATA_DIR:+--fhir_data_dir "${FHIR_DATA_DIR}"} \
    --patient_id "${PATIENT_ID}" \
    --question "${QUESTION}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
