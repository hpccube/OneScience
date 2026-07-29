#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_boltz_common.sh"

predictions_match_references() {
    local predictions="$1"
    local references="$2"
    local testset="$3"
    local folder name reference
    local found=0

    while IFS= read -r folder; do
        if [[ -z "$(find "${folder}" -maxdepth 1 -type f -name '*_model_*.cif' -print -quit)" ]]; then
            continue
        fi
        found=1
        name="$(basename "${folder}")"
        if [[ "${testset}" == "casp" ]]; then
            reference="${references}/${name^}.cif"
        else
            reference="${references}/${name,,}.cif.gz"
        fi
        if [[ ! -f "${reference}" ]]; then
            return 1
        fi
    done < <(find "${predictions}" -mindepth 1 -maxdepth 1 -type d | sort)

    [[ "${found}" -eq 1 ]]
}

stage_evaluation_inputs() {
    local predictions="$1"
    local references="$2"
    local staged_predictions="$3"
    local staged_references="$4"
    local testset="$5"
    local num_samples="$6"
    local folder name reference model_id
    local -a model_files

    while IFS= read -r folder; do
        if [[ -z "$(find "${folder}" -maxdepth 1 -type f -name '*_model_*.cif' -print -quit)" ]]; then
            continue
        fi

        name="$(basename "${folder}")"
        if [[ "${testset}" == "casp" ]]; then
            reference="${references}/${name^}.cif"
        else
            reference="${references}/${name,,}.cif.gz"
        fi
        boltz_require_file "${reference}"

        mkdir -p "${staged_predictions}/${name}"
        cp -- "${reference}" "${staged_references}/"
        for ((model_id = 0; model_id < num_samples; model_id++)); do
            mapfile -t model_files < <(
                find "${folder}" -maxdepth 1 -type f \
                    -name "*_model_${model_id}.cif" | sort
            )
            if [[ "${#model_files[@]}" -ne 1 ]]; then
                echo "Expected one model ${model_id} CIF in ${folder}, found ${#model_files[@]}" >&2
                return 2
            fi
            cp -- "${model_files[0]}" "${staged_predictions}/${name}/"
        done
    done < <(find "${predictions}" -mindepth 1 -maxdepth 1 -type d | sort)
}

prepare_podman_runtime() {
    local runtime_dir="${BOLTZ_PODMAN_RUNTIME_DIR:-/tmp/onescience-podman-runtime-${UID}}"

    mkdir -p "${runtime_dir}"
    chmod 700 "${runtime_dir}"
    export XDG_RUNTIME_DIR="${runtime_dir}"
    echo "Boltz Podman runtime directory: ${XDG_RUNTIME_DIR}"
}

load_container_image_archive() {
    local runtime="$1"
    local image="$2"
    local archive="${BOLTZ_EVAL_IMAGE_ARCHIVE:-}"
    local candidate

    if command "${runtime}" image inspect "${image}" >/dev/null 2>&1; then
        return 0
    fi

    if [[ -z "${archive}" ]]; then
        for candidate in \
            "${BOLTZ_MODEL_ROOT}/openstructure-0.2.8.tar" \
            "${BOLTZ_MODEL_ROOT}/openstructure-0.2.8.tar.gz" \
            "${BOLTZ_DATASET_ROOT}/openstructure-0.2.8.tar" \
            "${BOLTZ_DATASET_ROOT}/openstructure-0.2.8.tar.gz"; do
            if [[ -f "${candidate}" ]]; then
                archive="${candidate}"
                break
            fi
        done
    fi

    if [[ -z "${archive}" ]]; then
        return 1
    fi
    boltz_require_file "${archive}"
    echo "Loading offline OpenStructure image from: ${archive}"
    command "${runtime}" load --input "${archive}"
    if ! command "${runtime}" image inspect "${image}" >/dev/null 2>&1; then
        echo "The archive did not provide the requested image tag: ${image}" >&2
        echo "Set BOLTZ_EVAL_IMAGE to the tag reported by ${runtime} load." >&2
        return 1
    fi
}

copy_precomputed_evaluations() {
    local predictions="$1"
    local evaluations="$2"
    local output="$3"
    local model_file model_name polymer_eval ligand_eval
    local copied=0

    boltz_require_dir "${evaluations}"
    while IFS= read -r model_file; do
        model_name="$(basename "${model_file}" .cif)"
        polymer_eval="${evaluations}/${model_name}.json"
        ligand_eval="${evaluations}/${model_name}_ligand.json"
        boltz_require_file "${polymer_eval}"
        boltz_require_file "${ligand_eval}"
        cp -- "${polymer_eval}" "${ligand_eval}" "${output}/"
        ((copied += 2))
    done < <(find "${predictions}" -mindepth 2 -maxdepth 2 -type f \
        -name '*_model_*.cif' | sort)

    if [[ "${copied}" -eq 0 ]]; then
        echo "No staged prediction models matched precomputed evaluations." >&2
        return 2
    fi
    echo "Copied ${copied} official precomputed OpenStructure JSON files"
}

USE_STAGING=false
if [[ $# -eq 0 || "$1" == -* ]]; then
    EXTRA_ARGS=("${@}")
    RESULT_ROOT="${BOLTZ_RESULTS_ROOT:-${BOLTZ_SHARED_RESULTS_ROOT}}"
    EVAL_SPLIT="${BOLTZ_EVAL_SPLIT:-test}"
    case "${EVAL_SPLIT}" in
        test)
            TESTSET="test"
            ;;
        casp15)
            TESTSET="casp"
            ;;
        *)
            echo "BOLTZ_EVAL_SPLIT must be test or casp15" >&2
            exit 2
            ;;
    esac

    PRED_DIR="${BOLTZ_EVAL_PREDICTIONS:-${BOLTZ_PREDICTIONS_DIR:-${RESULT_ROOT}/outputs/${EVAL_SPLIT}/boltz/predictions}}"
    REF_DIR="${BOLTZ_EVAL_REFERENCES:-${RESULT_ROOT}/targets/${EVAL_SPLIT}}"
    OUT_DIR="${BOLTZ_EVAL_OUTPUT:-${BOLTZ_LOCAL_EVAL_DIR:-${BOLTZ_EXAMPLE_DIR}/outputs/evaluate/${EVAL_SPLIT}/boltz}}"
    NUM_SAMPLES="${BOLTZ_EVAL_NUM_SAMPLES:-1}"

    for ((arg_index = 0; arg_index < ${#EXTRA_ARGS[@]}; arg_index++)); do
        case "${EXTRA_ARGS[arg_index]}" in
            --num-samples=*)
                NUM_SAMPLES="${EXTRA_ARGS[arg_index]#*=}"
                ;;
            --num-samples)
                if ((arg_index + 1 >= ${#EXTRA_ARGS[@]})); then
                    echo "--num-samples requires a value" >&2
                    exit 2
                fi
                NUM_SAMPLES="${EXTRA_ARGS[arg_index + 1]}"
                ((arg_index += 1))
                ;;
        esac
    done
    if [[ ! "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "BOLTZ_EVAL_NUM_SAMPLES/--num-samples must be a positive integer" >&2
        exit 2
    fi

    boltz_require_dir "${PRED_DIR}"
    boltz_require_dir "${REF_DIR}"
    if ! predictions_match_references "${PRED_DIR}" "${REF_DIR}" "${TESTSET}"; then
        echo "Boltz predictions and references are not aligned:" >&2
        echo "  predictions: ${PRED_DIR}" >&2
        echo "  references:  ${REF_DIR}" >&2
        exit 2
    fi

    STAGE_BASE="${BOLTZ_EVAL_STAGE_ROOT:-/tmp/onescience-boltz-eval-${UID}}"
    mkdir -p "${STAGE_BASE}"
    STAGE_DIR="$(mktemp -d "${STAGE_BASE%/}/${EVAL_SPLIT}.XXXXXX")"
    STAGE_PRED_DIR="${STAGE_DIR}/predictions"
    STAGE_REF_DIR="${STAGE_DIR}/references"
    STAGE_OUT_DIR="${STAGE_DIR}/evaluations"
    MOUNT_DIR="${STAGE_DIR}"
    EVAL_CONTAINER_NAME=""
    EVAL_CONTAINER_RUNTIME=""
    EVAL_CONTAINER_CREATED=false
    mkdir -p "${STAGE_PRED_DIR}" "${STAGE_REF_DIR}" "${STAGE_OUT_DIR}"

    sync_and_cleanup_stage() {
        local status=$?
        local container_copy_status=0
        local result_count=0
        local sync_status=0

        trap - EXIT
        set +e
        if [[ "${EVAL_CONTAINER_CREATED}" == "true" ]]; then
            command "${EVAL_CONTAINER_RUNTIME}" cp \
                "${EVAL_CONTAINER_NAME}:${STAGE_OUT_DIR}/." \
                "${STAGE_OUT_DIR}/"
            container_copy_status=$?
            if [[ "${container_copy_status}" -ne 0 ]]; then
                echo "Failed to retrieve Boltz evaluations from ${EVAL_CONTAINER_NAME}" >&2
            fi
            command "${EVAL_CONTAINER_RUNTIME}" rm --force \
                "${EVAL_CONTAINER_NAME}" >/dev/null
        fi

        mkdir -p "${OUT_DIR}"
        cp -R "${STAGE_OUT_DIR}/." "${OUT_DIR}/"
        sync_status=$?
        result_count="$(find "${STAGE_OUT_DIR}" -maxdepth 1 -type f -name '*.json' | wc -l)"
        if [[ "${sync_status}" -eq 0 ]]; then
            if [[ "${result_count}" -gt 0 ]]; then
                echo "Synchronized ${result_count} Boltz evaluation JSON files to: ${OUT_DIR}"
            else
                echo "No Boltz evaluation JSON files were produced." >&2
                if [[ "${status}" -eq 0 ]]; then
                    status=1
                fi
            fi
        else
            echo "Failed to synchronize Boltz evaluation results to ${OUT_DIR}" >&2
        fi

        if [[ "${BOLTZ_EVAL_KEEP_STAGE:-0}" == "1" ]]; then
            echo "Boltz evaluation staging retained at: ${STAGE_DIR}"
        else
            rm -rf -- "${STAGE_DIR}"
        fi
        if [[ "${status}" -eq 0 && "${sync_status}" -ne 0 ]]; then
            status="${sync_status}"
        fi
        if [[ "${status}" -eq 0 && "${container_copy_status}" -ne 0 ]]; then
            status="${container_copy_status}"
        fi
        exit "${status}"
    }
    trap sync_and_cleanup_stage EXIT

    echo "Staging Boltz evaluation inputs under: ${STAGE_DIR}"
    stage_evaluation_inputs \
        "${PRED_DIR}" \
        "${REF_DIR}" \
        "${STAGE_PRED_DIR}" \
        "${STAGE_REF_DIR}" \
        "${TESTSET}" \
        "${NUM_SAMPLES}"
    if [[ -d "${OUT_DIR}" ]]; then
        cp -R "${OUT_DIR}/." "${STAGE_OUT_DIR}/"
    fi

    DEFAULT_ARGS=(
        --predictions "${STAGE_PRED_DIR}"
        --references "${STAGE_REF_DIR}"
        --output "${STAGE_OUT_DIR}"
        --format boltz
        --testset "${TESTSET}"
        --mount "${MOUNT_DIR}"
        --image "${BOLTZ_EVAL_IMAGE:-openstructure-0.2.8}"
        --num-samples "${NUM_SAMPLES}"
        --max-workers "${BOLTZ_EVAL_WORKERS:-16}"
    )
    set -- "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}"
    USE_STAGING=true
fi

# The Python evaluator invokes `docker run --volume` inside a Bash subprocess.
# Keep one container alive and translate that call to exec after copying the
# staged files into it, so restricted clusters never receive a bind mount.
if [[ "${USE_STAGING}" == "true" ]]; then
    requested_runtime="${BOLTZ_EVAL_CONTAINER_RUNTIME:-}"
    case "${requested_runtime}" in
        ""|docker|podman)
            ;;
        *)
            echo "BOLTZ_EVAL_CONTAINER_RUNTIME must be docker or podman" >&2
            exit 2
            ;;
    esac

    if [[ -n "${requested_runtime}" ]]; then
        if ! command -v "${requested_runtime}" >/dev/null 2>&1; then
            echo "Requested container runtime not found: ${requested_runtime}" >&2
            exit 2
        fi
        EVAL_CONTAINER_RUNTIME="${requested_runtime}"
    elif command -v docker >/dev/null 2>&1; then
        EVAL_CONTAINER_RUNTIME="docker"
    elif command -v podman >/dev/null 2>&1; then
        EVAL_CONTAINER_RUNTIME="podman"
    else
        echo "OpenStructure evaluation requires Docker or Podman on PATH." >&2
        exit 2
    fi
    if [[ "${EVAL_CONTAINER_RUNTIME}" == "podman" ]]; then
        prepare_podman_runtime
    fi

    EVAL_IMAGE="${BOLTZ_EVAL_IMAGE:-openstructure-0.2.8}"
    if ! load_container_image_archive \
        "${EVAL_CONTAINER_RUNTIME}" "${EVAL_IMAGE}"; then
        OFFICIAL_PRED_DIR="${RESULT_ROOT}/outputs/${EVAL_SPLIT}/boltz/predictions"
        OFFICIAL_REF_DIR="${RESULT_ROOT}/targets/${EVAL_SPLIT}"
        PRECOMPUTED_EVAL_DIR="${BOLTZ_EVAL_PRECOMPUTED_DIR:-${RESULT_ROOT}/evals/${EVAL_SPLIT}/boltz}"
        if [[ "${BOLTZ_EVAL_ALLOW_PRECOMPUTED:-1}" == "1" \
            && "$(realpath -m "${PRED_DIR}")" == "$(realpath -m "${OFFICIAL_PRED_DIR}")" \
            && "$(realpath -m "${REF_DIR}")" == "$(realpath -m "${OFFICIAL_REF_DIR}")" \
            && -d "${PRECOMPUTED_EVAL_DIR}" ]]; then
            echo "OpenStructure image ${EVAL_IMAGE} is not available locally."
            echo "Using official precomputed evaluations for the unchanged official predictions."
            copy_precomputed_evaluations \
                "${STAGE_PRED_DIR}" \
                "${PRECOMPUTED_EVAL_DIR}" \
                "${STAGE_OUT_DIR}"
            exit 0
        fi
        echo "OpenStructure image is not available locally: ${EVAL_IMAGE}" >&2
        echo "This compute node cannot pull it from the registry." >&2
        echo "Pre-load the image, or set BOLTZ_EVAL_IMAGE_ARCHIVE to an offline docker archive." >&2
        exit 2
    fi

    EVAL_CONTAINER_NAME="onescience-boltz-eval-${UID}-$(basename "${STAGE_DIR}")"
    command "${EVAL_CONTAINER_RUNTIME}" create \
        --name "${EVAL_CONTAINER_NAME}" \
        --entrypoint /bin/sh \
        "${EVAL_IMAGE}" \
        -c 'while :; do sleep 3600; done' >/dev/null
    EVAL_CONTAINER_CREATED=true
    command "${EVAL_CONTAINER_RUNTIME}" start "${EVAL_CONTAINER_NAME}" >/dev/null
    command "${EVAL_CONTAINER_RUNTIME}" exec --user 0:0 \
        "${EVAL_CONTAINER_NAME}" \
        mkdir -p "${STAGE_DIR}"
    command "${EVAL_CONTAINER_RUNTIME}" cp "${STAGE_DIR}/." \
        "${EVAL_CONTAINER_NAME}:${STAGE_DIR}/"

    export EVAL_CONTAINER_NAME EVAL_CONTAINER_RUNTIME
    docker() {
        local image_found=false
        local -a command_args=()

        if [[ "${1:-}" != "run" ]]; then
            command "${EVAL_CONTAINER_RUNTIME}" "$@"
            return
        fi
        shift
        while [[ $# -gt 0 ]]; do
            case "$1" in
                -u|--user|-v|--volume)
                    shift 2
                    ;;
                --user=*|--volume=*|-v=*)
                    shift
                    ;;
                --rm)
                    shift
                    ;;
                --)
                    shift
                    ;;
                -*)
                    echo "Unsupported docker run option in container adapter: $1" >&2
                    return 2
                    ;;
                *)
                    image_found=true
                    shift
                    command_args=("$@")
                    break
                    ;;
            esac
        done
        if [[ "${image_found}" != "true" || "${#command_args[@]}" -eq 0 ]]; then
            echo "Could not extract the OpenStructure command from docker run" >&2
            return 2
        fi
        command "${EVAL_CONTAINER_RUNTIME}" exec --user 0:0 \
            "${EVAL_CONTAINER_NAME}" "${command_args[@]}"
    }
    export -f docker
    echo "Using a persistent ${EVAL_CONTAINER_RUNTIME} container without host bind mounts"
fi

echo "Boltz structure evaluation output: ${OUT_DIR:-custom positional output}"
if [[ -n "${PRED_DIR:-}" ]]; then
    echo "Boltz structure prediction source: ${PRED_DIR}"
    echo "Boltz structure reference source: ${REF_DIR}"
    if [[ -n "${EVAL_CONTAINER_NAME:-}" ]]; then
        echo "Boltz OpenStructure transport: ${EVAL_CONTAINER_RUNTIME} cp/exec (no bind mount)"
    else
        echo "Boltz OpenStructure mount: ${MOUNT_DIR}"
    fi
fi
if [[ "${USE_STAGING}" == "true" ]]; then
    "${BOLTZ_PYTHON}" scripts/evaluate.py "${@}"
else
    exec "${BOLTZ_PYTHON}" scripts/evaluate.py "${@}"
fi
