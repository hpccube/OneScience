#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_boltz_common.sh"

has_boltz_input() {
    local arg
    for arg in "$@"; do
        case "${arg}" in
            --boltz1_dir|--boltz1_dir=*|--boltz1-dir|--boltz1-dir=*|--boltz-dir|--boltz-dir=*)
                return 0
                ;;
        esac
    done
    return 1
}

EVAL_SPLIT="${BOLTZ_PHYSICAL_SPLIT:-test}"

case "${EVAL_SPLIT}" in
    test)
        output_name="physical_checks_test.csv"
        ;;
    casp15)
        output_name="physical_checks_casp.csv"
        ;;
    *)
        echo "BOLTZ_PHYSICAL_SPLIT must be test or casp15" >&2
        exit 2
        ;;
esac

boltz1_dir="${BOLTZ_PHYSICAL_BOLTZ_DIR:-${BOLTZ_PREDICTIONS_DIR:-${BOLTZ_SHARED_RESULTS_ROOT}/outputs/${EVAL_SPLIT}/boltz/predictions}}"
chai_dir="${BOLTZ_PHYSICAL_CHAI_DIR:-${BOLTZ_SHARED_RESULTS_ROOT}/outputs/${EVAL_SPLIT}/chai}"
af3_dir="${BOLTZ_PHYSICAL_AF3_DIR:-${BOLTZ_SHARED_RESULTS_ROOT}/outputs/${EVAL_SPLIT}/af3}"
ccd_file="${BOLTZ_PHYSICAL_CCD:-${BOLTZ_CCD_FILE}}"
mols_dir="${BOLTZ_PHYSICAL_MOLS:-${BOLTZ_MOLS_DIR}}"
output_path="${BOLTZ_PHYSICAL_OUTPUT:-${BOLTZ_EXAMPLE_DIR}/outputs/physical/${output_name}}"

boltz_require_file "${ccd_file}"
boltz_require_dir "${mols_dir}"
boltz_require_dir "${chai_dir}"
boltz_require_dir "${af3_dir}"
if ! has_boltz_input "${@}"; then
    boltz_require_dir "${boltz1_dir}"
fi

DEFAULT_ARGS=(
    --ccd "${ccd_file}"
    --mols "${mols_dir}"
    --chai_dir "${chai_dir}"
    --af3_dir "${af3_dir}"
    --output "${output_path}"
)
if ! has_boltz_input "${@}"; then
    DEFAULT_ARGS+=(--boltz1_dir "${boltz1_dir}")
fi

echo "Boltz physical metrics output: ${output_path}"
if ! has_boltz_input "${@}"; then
    echo "Boltz physical metrics predictions: ${boltz1_dir}"
fi
exec "${BOLTZ_PYTHON}" scripts/physical_metrics.py \
    "${DEFAULT_ARGS[@]}" \
    "${@}"
