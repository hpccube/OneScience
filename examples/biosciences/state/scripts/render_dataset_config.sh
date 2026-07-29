#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_state_common.sh"

exec python "${STATE_RUNNER_DIR}/render_dataset_config.py" "$@"
