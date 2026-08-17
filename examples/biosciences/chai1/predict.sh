#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "${REPO_ROOT}/env.sh"
if [[ -n "${ROCM_PATH:-}" && -f "${ROCM_PATH}/cuda/env.sh" ]]; then
    source "${ROCM_PATH}/cuda/env.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CHAI1_MODEL_DIR="${CHAI1_MODEL_DIR:-${ONESCIENCE_MODELS_DIR}/chai-lab}"
CHAI1_OUTPUT_DIR="${CHAI1_OUTPUT_DIR:-${SCRIPT_DIR}/outputs/monomer}"
CHAI1_INPUT="${1:-${SCRIPT_DIR}/inputs/monomer.fasta}"
CHAI1_DEVICE="${CHAI1_DEVICE:-cuda:0}"
CHAI1_PYTHON="${CHAI1_PYTHON:-python}"
CHAI1_DEVICES="${CHAI1_DEVICES:-${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-0}}}"

if [[ "${CHAI1_DEVICES}" == *,* ]]; then
    if [[ "${CHAI1_DEVICE}" != "cuda" && "${CHAI1_DEVICE}" != "cuda:0" ]]; then
        echo "CHAI1_DEVICE must be cuda or cuda:0 in multi-card mode; each worker sees one card." >&2
        exit 2
    fi
    for user_arg in "${@:2}"; do
        case "${user_arg}" in
            --output-dir|--output-dir=*|--model-dir|--model-dir=*|--device|--device=*|\
            --seed|--seed=*|--num-trunk-recycles|--num-trunk-recycles=*|\
            --num-diffusion-timesteps|--num-diffusion-timesteps=*|\
            --num-diffusion-samples|--num-diffusion-samples=*|\
            --num-trunk-samples|--num-trunk-samples=*)
                echo "${user_arg} is controlled by CHAI1_* in multi-card mode; remove it from the command." >&2
                exit 2
                ;;
        esac
    done
fi

if ! command -v "${CHAI1_PYTHON}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${CHAI1_PYTHON}" >&2
    echo "Activate your Conda/virtualenv first, or set CHAI1_PYTHON." >&2
    exit 2
fi

required_assets=(
    "conformers_v1.apkl"
    "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    "models_v2/bond_loss_input_proj.pt"
    "models_v2/confidence_head.pt"
    "models_v2/diffusion_module.pt"
    "models_v2/feature_embedding.pt"
    "models_v2/token_embedder.pt"
    "models_v2/trunk.pt"
)
for asset in "${required_assets[@]}"; do
    if [[ ! -f "${CHAI1_MODEL_DIR}/${asset}" ]]; then
        echo "Required Chai-1 asset not found: ${CHAI1_MODEL_DIR}/${asset}" >&2
        exit 2
    fi
done

if [[ -z "${CHAI1_DEVICES}" || "${CHAI1_DEVICES}" != *,* ]]; then
    export HIP_VISIBLE_DEVICES="${CHAI1_DEVICES:-${HIP_VISIBLE_DEVICES:-0}}"
    export CUDA_VISIBLE_DEVICES="${CHAI1_DEVICES:-${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES}}}"
    exec "${CHAI1_PYTHON}" "${SCRIPT_DIR}/predict.py" "${CHAI1_INPUT}" \
        --output-dir "${CHAI1_OUTPUT_DIR}" \
        --model-dir "${CHAI1_MODEL_DIR}" \
        --device "${CHAI1_DEVICE}" \
        --seed "${CHAI1_SEED:-42}" \
        --num-trunk-recycles "${CHAI1_NUM_TRUNK_RECYCLES:-1}" \
        --num-diffusion-timesteps "${CHAI1_NUM_DIFFUSION_TIMESTEPS:-20}" \
        --num-diffusion-samples "${CHAI1_NUM_DIFFUSION_SAMPLES:-1}" \
        --num-trunk-samples "${CHAI1_NUM_TRUNK_SAMPLES:-1}" \
        --overwrite \
        "${@:2}"
fi

IFS=',' read -r -a chai1_devices <<< "${CHAI1_DEVICES}"
total_samples="${CHAI1_NUM_DIFFUSION_SAMPLES:-1}"
trunk_samples="${CHAI1_NUM_TRUNK_SAMPLES:-1}"
if ! [[ "${total_samples}" =~ ^[1-9][0-9]*$ && "${trunk_samples}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHAI1_NUM_DIFFUSION_SAMPLES and CHAI1_NUM_TRUNK_SAMPLES must be positive integers." >&2
    exit 2
fi
if (( trunk_samples != 1 )); then
    echo "Multi-card inference currently requires CHAI1_NUM_TRUNK_SAMPLES=1." >&2
    echo "Use CHAI1_NUM_DIFFUSION_SAMPLES to control the total number of candidates." >&2
    exit 2
fi

device_count="${#chai1_devices[@]}"
declare -A seen_devices=()
for worker_device in "${chai1_devices[@]}"; do
    if ! [[ "${worker_device}" =~ ^[0-9]+$ ]]; then
        echo "CHAI1_DEVICES must be a comma-separated list of numeric device ids: ${CHAI1_DEVICES}" >&2
        exit 2
    fi
    if [[ -n "${seen_devices[${worker_device}]:-}" ]]; then
        echo "CHAI1_DEVICES contains a duplicate device id: ${worker_device}" >&2
        exit 2
    fi
    seen_devices["${worker_device}"]=1
done
if (( total_samples < device_count )); then
    device_count="${total_samples}"
fi
if (( device_count < 2 )); then
    export HIP_VISIBLE_DEVICES="${chai1_devices[0]}"
    export CUDA_VISIBLE_DEVICES="${chai1_devices[0]}"
    exec "${CHAI1_PYTHON}" "${SCRIPT_DIR}/predict.py" "${CHAI1_INPUT}" \
        --output-dir "${CHAI1_OUTPUT_DIR}" \
        --model-dir "${CHAI1_MODEL_DIR}" \
        --device "cuda:0" \
        --seed "${CHAI1_SEED:-42}" \
        --num-trunk-recycles "${CHAI1_NUM_TRUNK_RECYCLES:-1}" \
        --num-diffusion-timesteps "${CHAI1_NUM_DIFFUSION_TIMESTEPS:-20}" \
        --num-diffusion-samples "${total_samples}" \
        --num-trunk-samples "${trunk_samples}" \
        --overwrite \
        "${@:2}"
fi

mkdir -p "$(dirname "${CHAI1_OUTPUT_DIR}")"
worker_root="$(mktemp -d "${CHAI1_OUTPUT_DIR}.workers.XXXXXX")"
worker_pids=()
cleanup_workers() {
    for worker_pid in "${worker_pids[@]:-}"; do
        kill "${worker_pid}" 2>/dev/null || true
    done
    rm -rf "${worker_root}"
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

base_samples=$((total_samples / device_count))
remainder=$((total_samples % device_count))
base_seed="${CHAI1_SEED:-42}"
if ! [[ "${base_seed}" =~ ^[0-9]+$ ]]; then
    echo "CHAI1_SEED must be a non-negative integer for multi-card inference." >&2
    exit 2
fi

for ((worker_idx = 0; worker_idx < device_count; worker_idx++)); do
    worker_samples="${base_samples}"
    if (( worker_idx < remainder )); then
        worker_samples=$((worker_samples + 1))
    fi
    worker_dir="${worker_root}/worker_${worker_idx}"
    worker_device="${chai1_devices[worker_idx]}"
    worker_seed=$((base_seed + worker_idx))

    env \
        HIP_VISIBLE_DEVICES="${worker_device}" \
        CUDA_VISIBLE_DEVICES="${worker_device}" \
        "${CHAI1_PYTHON}" "${SCRIPT_DIR}/predict.py" "${CHAI1_INPUT}" \
        --output-dir "${worker_dir}" \
        --model-dir "${CHAI1_MODEL_DIR}" \
        --device "cuda:0" \
        --seed "${worker_seed}" \
        --num-trunk-recycles "${CHAI1_NUM_TRUNK_RECYCLES:-1}" \
        --num-diffusion-timesteps "${CHAI1_NUM_DIFFUSION_TIMESTEPS:-20}" \
        --num-diffusion-samples "${worker_samples}" \
        --num-trunk-samples 1 \
        --overwrite \
        "${@:2}" &
    worker_pids+=("$!")
done

worker_failed=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "${worker_pid}"; then
        worker_failed=1
    fi
done
worker_pids=()
if (( worker_failed != 0 )); then
    echo "At least one Chai-1 worker failed; see the worker logs above." >&2
    exit 1
fi

"${CHAI1_PYTHON}" "${SCRIPT_DIR}/merge_predictions.py" \
    --output-dir "${CHAI1_OUTPUT_DIR}" \
    --overwrite \
    "${worker_root}"/worker_*
merge_status=$?
cleanup_workers
trap - EXIT INT TERM
exit "${merge_status}"
