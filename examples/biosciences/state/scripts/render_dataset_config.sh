#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <template.toml> <dataset-dir> <output.toml>" >&2
  exit 2
fi

template="$1"
dataset_dir="$2"
output="$3"

mkdir -p "$(dirname "${output}")"
sed \
  -e "s|__STATE_DATASET_DIR__|${dataset_dir}|g" \
  -e "s|__STATE_DATA_ROOT__|${dataset_dir}|g" \
  "${template}" > "${output}"
echo "Rendered ${output}"
