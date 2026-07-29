#!/bin/bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=""
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --submit) MODE="submit"; shift ;;
        -h|--help) echo "Usage: bash run.sh --config configs/<name>.yaml [--submit]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$CONFIG" ]] || { echo "Please specify --config configs/<name>.yaml" >&2; exit 2; }
[[ "$CONFIG" = /* ]] || CONFIG="$DEMO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
if [[ -z "${CONDA_PREFIX:-}" || -z "${ONESCIENCE_MODELS_DIR:-}" ]]; then
    export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-onescience-mattergen-source}"
    source "$DEMO_DIR/../../matchem_env.sh"
fi
CMD=$(python3 - "$CONFIG" "$DEMO_DIR/.." <<'PY'
import os, shlex, sys, yaml
import json
cfg = yaml.safe_load(open(sys.argv[1]))
root = os.path.abspath(sys.argv[2])
task = cfg.get("task", "train")
script = {"train": "train.py", "finetune": "finetune.py", "generate": "generate.py"}.get(task)
if not script: raise SystemExit(f"Unsupported task: {task}")
args = []
for key, value in cfg.get("args", {}).items():
    if isinstance(value, bool):
        if value: args.append(f"--{key.replace('_', '-')}")
    elif value is not None:
        value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        value = os.path.expandvars(value)
        args.extend([f"--{key.replace('_', '-')}", value] if task == "generate" else [f"{key}={value}"])
print(" ".join(map(shlex.quote, [sys.executable, os.path.join(root, script), *args])))
PY
)
echo "Running MatterGen: $CMD"
if [[ "$MODE" == "submit" ]]; then
    TASK=$(python3 - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("task", "train"))
PY
)
    [[ "$TASK" == "train" ]] || { echo "--submit currently supports train configs only" >&2; exit 2; }
    DEVICES=$(python3 - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1])).get("args", {}).get("trainer.devices", 8))
PY
)
    sbatch \
        --gres="dcu:$DEVICES" \
        --export="ALL,CONFIG_PATH=$CONFIG,SCRIPT_DIR=$DEMO_DIR/.." \
        "$DEMO_DIR/../submit_train.sh"
    exit 0
fi
cd "$DEMO_DIR/.."
eval "$CMD"
