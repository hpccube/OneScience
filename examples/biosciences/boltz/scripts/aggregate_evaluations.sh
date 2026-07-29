#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_boltz_common.sh"

export MPLBACKEND="${MPLBACKEND:-Agg}"

has_option() {
    local expected="$1" arg
    shift
    for arg in "$@"; do
        if [[ "${arg}" == "${expected}" || "${arg}" == "${expected}="* ]]; then
            return 0
        fi
    done
    return 1
}

# Preserve the legacy positional form:
#   aggregate_evaluations.sh [results_root] [output_root]
if [[ $# -gt 0 && "$1" != -* ]]; then
    RESULT_ROOT="$1"
    shift
    OUTPUT_ROOT="${1:-${BOLTZ_AGGREGATE_OUTPUT:-${RESULT_ROOT}/aggregate_local}}"
    if [[ $# -gt 0 ]]; then
        shift
    fi
    if [[ $# -gt 0 ]]; then
        echo "Usage: bash scripts/aggregate_evaluations.sh [results_root] [output_root]" >&2
        exit 2
    fi
    exec "${BOLTZ_PYTHON}" scripts/aggregate_evaluations.py \
        --shared-root "${RESULT_ROOT}" \
        --local-eval-root "${RESULT_ROOT}/evals" \
        --output "${OUTPUT_ROOT}"
fi

AGGREGATE_SPLIT="${BOLTZ_AGGREGATE_SPLIT:-test}"
previous_arg=""
for arg in "${@}"; do
    if [[ "${previous_arg}" == "--split" ]]; then
        AGGREGATE_SPLIT="${arg}"
        break
    fi
    if [[ "${arg}" == --split=* ]]; then
        AGGREGATE_SPLIT="${arg#*=}"
        break
    fi
    previous_arg="${arg}"
done
case "${AGGREGATE_SPLIT}" in
    test)
        default_physical_checks="${BOLTZ_REPORT_ROOT}/physical_checks_test.csv"
        ;;
    casp15)
        default_physical_checks="${BOLTZ_REPORT_ROOT}/physical_checks_casp.csv"
        ;;
    all)
        default_physical_checks=""
        ;;
    *)
        echo "BOLTZ_AGGREGATE_SPLIT must be test, casp15, or all" >&2
        exit 2
        ;;
esac

boltz_predictions="${BOLTZ_PREDICTIONS_DIR:-${BOLTZ_SHARED_RESULTS_ROOT}/outputs/${AGGREGATE_SPLIT}/boltz/predictions}"
boltz_evaluations="${BOLTZ_LOCAL_EVAL_DIR:-${BOLTZ_EXAMPLE_DIR}/outputs/evaluate/${AGGREGATE_SPLIT}/boltz}"
aggregate_output="${BOLTZ_AGGREGATE_OUTPUT:-${BOLTZ_EXAMPLE_DIR}/outputs/aggregate}"

if [[ "${AGGREGATE_SPLIT}" != "all" ]] && ! has_option --boltz-preds "${@}"; then
    boltz_require_dir "${boltz_predictions}"
fi
if [[ "${AGGREGATE_SPLIT}" != "all" ]] && ! has_option --boltz-evals "${@}"; then
    boltz_require_dir "${boltz_evaluations}"
    if [[ -z "$(find "${boltz_evaluations}" -maxdepth 1 -type f \
        -name '*_model_*.json' -print -quit)" ]]; then
        echo "No Boltz evaluation JSON files found in ${boltz_evaluations}" >&2
        echo "Run bash scripts/evaluate.sh successfully before aggregation," >&2
        echo "or set BOLTZ_LOCAL_EVAL_DIR/pass --boltz-evals to existing evaluations." >&2
        exit 2
    fi
fi

if [[ -n "${BOLTZ_PHYSICAL_OUTPUT:-}" ]]; then
    default_physical_checks="${BOLTZ_PHYSICAL_OUTPUT}"
elif [[ "${AGGREGATE_SPLIT}" != "all" ]]; then
    default_physical_checks="${BOLTZ_EXAMPLE_DIR}/outputs/physical/physical_checks_${AGGREGATE_SPLIT}.csv"
fi

explicit_physical_checks=false
for arg in "${@}"; do
    if [[ "${arg}" == "--physical-checks" || "${arg}" == --physical-checks=* ]]; then
        explicit_physical_checks=true
    fi
done

if [[ "${explicit_physical_checks}" == false && -n "${default_physical_checks}" && -f "${default_physical_checks}" ]]; then
    set -- \
        --physical-checks "${default_physical_checks}" \
        "${@}"
fi

DEFAULT_ARGS=(
    --shared-root "${BOLTZ_SHARED_RESULTS_ROOT}"
    --local-eval-root "${BOLTZ_EXAMPLE_DIR}/outputs/evaluate"
    --output "${aggregate_output}"
    --split "${AGGREGATE_SPLIT}"
    --num-samples "${BOLTZ_AGGREGATE_NUM_SAMPLES:-1}"
)
if [[ "${AGGREGATE_SPLIT}" != "all" ]]; then
    DEFAULT_ARGS+=(--boltz-evals "${boltz_evaluations}")
fi
if [[ "${AGGREGATE_SPLIT}" != "all" && -n "${boltz_predictions}" ]]; then
    DEFAULT_ARGS+=(--boltz-preds "${boltz_predictions}")
fi

echo "Boltz aggregate output: ${aggregate_output}"
if [[ "${AGGREGATE_SPLIT}" != "all" ]]; then
    if ! has_option --boltz-preds "${@}"; then
        echo "Boltz aggregate predictions: ${boltz_predictions}"
    fi
    if ! has_option --boltz-evals "${@}"; then
        echo "Boltz aggregate evaluations: ${boltz_evaluations}"
    fi
fi
exec "${BOLTZ_PYTHON}" scripts/aggregate_evaluations.py \
    "${DEFAULT_ARGS[@]}" \
    "${@}"
