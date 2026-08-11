#!/bin/bash
# Run native eSEN fine-tuning directly or submit it to Slurm.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESEN_DIR="$(cd "$DEMO_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ESEN_DIR/../../.." && pwd)"
PARSER="$DEMO_DIR/_parse_config.py"
CONFIG=""
SUBMIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --submit) SUBMIT=true; shift ;;
        -h|--help)
            echo "Usage: bash demo/run.sh --config configs/<name>.yaml"
            echo "Set launch.mode: submit in YAML to submit a Slurm job."
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "Please specify --config configs/<name>.yaml" >&2; exit 2; }
[[ "$CONFIG" = /* ]] || CONFIG="$DEMO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

if [[ -z "${CONDA_PREFIX:-}" || -z "${ONESCIENCE_MODELS_DIR:-}" || -z "${ONESCIENCE_DATASETS_DIR:-}" ]]; then
    source "$REPO_ROOT/examples/matchem/matchem_env.sh"
fi

NAME="$(python3 "$PARSER" "$CONFIG" name)"
eval "$(python3 "$PARSER" "$CONFIG" launch)"
eval "$(python3 "$PARSER" "$CONFIG" slurm)"
ENV_EXPORTS="$(python3 "$PARSER" "$CONFIG" env)"

if [[ "$RUN_MODE" == "submit" ]]; then
    SUBMIT=true
fi

if ! $SUBMIT; then
    AUTO_SUBMIT_REASON=""
    if (( NODES > 1 )); then
        AUTO_SUBMIT_REASON="the config requests $NODES nodes"
    else
        AVAILABLE_GPUS="$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)"
        if (( AVAILABLE_GPUS < GPUS_PER_NODE )); then
            AUTO_SUBMIT_REASON="the config requests $GPUS_PER_NODE DCUs but only $AVAILABLE_GPUS are visible"
        fi
    fi
    if [[ -n "$AUTO_SUBMIT_REASON" ]]; then
        if ! command -v sbatch >/dev/null 2>&1; then
            echo "Cannot run locally: $AUTO_SUBMIT_REASON, and sbatch is unavailable." >&2
            exit 2
        fi
        echo "Local resources are insufficient: $AUTO_SUBMIT_REASON. Submitting to Slurm."
        SUBMIT=true
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ESEN_OUTPUT_ROOT:-$ESEN_DIR/outputs}"
OUTPUT_DIR="$OUTPUT_ROOT/${NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/checkpoints"
cp "$CONFIG" "$OUTPUT_DIR/config.yaml"
FINETUNE_CONFIG="$OUTPUT_DIR/finetune.yaml"
python3 "$PARSER" "$CONFIG" finetune-config > "$FINETUNE_CONFIG"

if $SUBMIT; then
    SLURM_SCRIPT="$OUTPUT_DIR/submit.sh"
    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gres=dcu:$GPUS_PER_NODE
#SBATCH --time=$TIME
#SBATCH --output=$OUTPUT_DIR/slurm_%j.out
#SBATCH --error=$OUTPUT_DIR/slurm_%j.err
EOF

    cat >> "$SLURM_SCRIPT" <<EOF

set -euo pipefail
export MATCHEM_CONDA_NAME="\${MATCHEM_CONDA_NAME:-test_pip}"
source "$REPO_ROOT/examples/matchem/matchem_env.sh"
export OMP_NUM_THREADS="$OMP_NUM_THREADS"
# A login/compute shell may carry a single-device selection into sbatch.
# Let Slurm expose all devices requested by this YAML.
if (( $GPUS_PER_NODE > 1 )); then
    unset HIP_VISIBLE_DEVICES
fi
$ENV_EXPORTS
cd "$OUTPUT_DIR"
EOF

    if (( NODES > 1 )); then
        cat >> "$SLURM_SCRIPT" <<EOF

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
export MASTER_ADDR="\${nodes[0]}"
export MASTER_PORT=\$((20000 + SLURM_JOB_ID % 20000))
echo "eSEN multi-node DDP: nodes=\$SLURM_NNODES, devices/node=$GPUS_PER_NODE"
echo "MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT"

srun --nodes="\$SLURM_NNODES" --ntasks="\$SLURM_NNODES" --ntasks-per-node=1 \\
  bash -c 'exec torchrun \\
    --nnodes=$NODES \\
    --node_rank="\${SLURM_NODEID}" \\
    --nproc_per_node=$GPUS_PER_NODE \\
    --rdzv_id="\${SLURM_JOB_ID}" \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    "$ESEN_DIR/finetune.py" --config "$FINETUNE_CONFIG"'
EOF
    elif (( GPUS_PER_NODE > 1 )); then
        cat >> "$SLURM_SCRIPT" <<EOF

torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \\
  "$ESEN_DIR/finetune.py" --config "$FINETUNE_CONFIG"
EOF
    else
        echo "python \"$ESEN_DIR/finetune.py\" --config \"$FINETUNE_CONFIG\"" >> "$SLURM_SCRIPT"
    fi

    chmod u+x "$SLURM_SCRIPT"
    echo "Submitting eSEN job: $SLURM_SCRIPT"
    sbatch "$SLURM_SCRIPT"
    exit 0
fi

eval "$ENV_EXPORTS"
cd "$OUTPUT_DIR"
if (( GPUS_PER_NODE > 1 )); then
    exec torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" \
        "$ESEN_DIR/finetune.py" --config "$FINETUNE_CONFIG"
fi
exec python "$ESEN_DIR/finetune.py" --config "$FINETUNE_CONFIG"
