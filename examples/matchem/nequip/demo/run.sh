#!/bin/bash
# Run NequIP training locally or submit it to Slurm from one YAML file.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEQUIP_DIR="$(cd "$DEMO_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NEQUIP_DIR/../../.." && pwd)"
PARSER="$DEMO_DIR/_parse_config.py"
CONFIG=""
SUBMIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --submit) SUBMIT=true; shift ;;
        -h|--help)
            echo "Usage: bash demo/run.sh --config configs/<name>.yaml [--submit]"
            echo "launch.mode: auto uses matching resources or submits when needed."
            echo "launch.mode: local runs directly; submit always submits to Slurm."
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "Please specify --config configs/<name>.yaml" >&2; exit 2; }
[[ "$CONFIG" = /* ]] || CONFIG="$DEMO_DIR/$CONFIG"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

if [[ -z "${CONDA_PREFIX:-}" || -z "${ONESCIENCE_MODELS_DIR:-}" || -z "${ONESCIENCE_DATASETS_DIR:-}" ]]; then
    export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-onescience311}"
    source "$REPO_ROOT/examples/matchem/matchem_env.sh"
fi
export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-$(basename "$CONDA_PREFIX")}"

NAME="$(python3 "$PARSER" "$CONFIG" name)"
eval "$(python3 "$PARSER" "$CONFIG" launch)"
eval "$(python3 "$PARSER" "$CONFIG" slurm)"
ENV_EXPORTS="$(python3 "$PARSER" "$CONFIG" env)"
if [[ "$RUN_MODE" == "submit" ]]; then
    SUBMIT=true
fi

if [[ "$RUN_MODE" == "auto" ]] && ! $SUBMIT; then
    IN_SLURM_ALLOCATION=false
    AVAILABLE_NODES=1
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        IN_SLURM_ALLOCATION=true
        AVAILABLE_NODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
        if ! [[ "$AVAILABLE_NODES" =~ ^[1-9][0-9]*$ ]]; then
            echo "Cannot determine allocated nodes from Slurm: $AVAILABLE_NODES" >&2
            exit 2
        fi
    fi

    AVAILABLE_GPUS="$(
        python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' \
            2>/dev/null || true
    )"
    if ! [[ "$AVAILABLE_GPUS" =~ ^[0-9]+$ ]]; then
        AVAILABLE_GPUS=0
    fi

    RESOURCE_MISMATCH=""
    if (( AVAILABLE_NODES < NODES )); then
        RESOURCE_MISMATCH="the config requests $NODES nodes but only $AVAILABLE_NODES are available"
    elif (( AVAILABLE_GPUS < GPUS_PER_NODE )); then
        RESOURCE_MISMATCH="the config requests $GPUS_PER_NODE DCUs per node but only $AVAILABLE_GPUS are visible"
    fi

    if [[ -n "$RESOURCE_MISMATCH" ]]; then
        if ! command -v sbatch >/dev/null 2>&1; then
            echo "Current resources are insufficient: $RESOURCE_MISMATCH, and sbatch is unavailable." >&2
            exit 2
        fi
        if $IN_SLURM_ALLOCATION; then
            echo "Current Slurm allocation is insufficient: $RESOURCE_MISMATCH. Submitting a new Slurm job."
        else
            echo "Current resources are insufficient: $RESOURCE_MISMATCH. Submitting to Slurm."
        fi
        SUBMIT=true
    else
        echo "Current resources satisfy the config: nodes=$NODES, DCUs/node=$GPUS_PER_NODE."
    fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${ONESCIENCE_NEQUIP_OUTPUT_ROOT:-$NEQUIP_DIR/outputs}"
OUTPUT_DIR="$OUTPUT_ROOT/${NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/checkpoints"
cp "$CONFIG" "$OUTPUT_DIR/source_config.yaml"
python3 "$PARSER" "$CONFIG" training-config > "$OUTPUT_DIR/config.yaml"

if $SUBMIT; then
    SLURM_SCRIPT="$OUTPUT_DIR/submit.sh"
    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --gres=dcu:$GPUS_PER_NODE
#SBATCH --time=$TIME
#SBATCH --output=$OUTPUT_DIR/slurm_%j.out
#SBATCH --error=$OUTPUT_DIR/slurm_%j.err
EOF
    if [[ -n "$NODELIST" ]]; then
        printf '#SBATCH --nodelist=%s\n' "$NODELIST" >> "$SLURM_SCRIPT"
    fi
    cat >> "$SLURM_SCRIPT" <<EOF

set -euo pipefail
export MATCHEM_CONDA_NAME="$MATCHEM_CONDA_NAME"
source "$REPO_ROOT/examples/matchem/matchem_env.sh"
export HSA_FORCE_FINE_GRAIN_PCIE=1
if (( $GPUS_PER_NODE > 1 )); then
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
fi
$ENV_EXPORTS
cd "$OUTPUT_DIR"
EOF
    if (( WORLD_SIZE > 1 )); then
        cat >> "$SLURM_SCRIPT" <<EOF
exec srun --kill-on-bad-exit=1 \
    --nodes=$NODES \
    --ntasks=$WORLD_SIZE \
    --ntasks-per-node=$GPUS_PER_NODE \
    python "$NEQUIP_DIR/train.py" "hydra.run.dir=$OUTPUT_DIR"
EOF
    else
        echo "exec python \"$NEQUIP_DIR/train.py\" \"hydra.run.dir=$OUTPUT_DIR\"" >> "$SLURM_SCRIPT"
    fi
    chmod u+x "$SLURM_SCRIPT"
    echo "Submitting NequIP job: $SLURM_SCRIPT"
    sbatch "$SLURM_SCRIPT"
    exit 0
fi

eval "$ENV_EXPORTS"
cd "$OUTPUT_DIR"
if (( WORLD_SIZE > 1 )); then
    if (( NODES > 1 )); then
        if [[ "$RUN_MODE" == "auto" && -n "${SLURM_JOB_ID:-}" ]]; then
            exec srun --kill-on-bad-exit=1 \
                --nodes="$NODES" \
                --ntasks="$WORLD_SIZE" \
                --ntasks-per-node="$GPUS_PER_NODE" \
                python "$NEQUIP_DIR/train.py" "hydra.run.dir=$OUTPUT_DIR"
        fi
        echo "Multi-node NequIP training must be launched through Slurm (--submit)." >&2
        exit 2
    fi
    exec torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" \
        "$NEQUIP_DIR/train.py" "hydra.run.dir=$OUTPUT_DIR"
fi
exec python "$NEQUIP_DIR/train.py" "hydra.run.dir=$OUTPUT_DIR"
