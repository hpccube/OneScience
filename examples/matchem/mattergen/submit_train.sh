#!/bin/bash
# MatterGen Slurm training entry. Users may edit the defaults below directly.
# The recommended interface is: cd demo && bash run.sh --config configs/train_8dcu.yaml --submit
# Direct interface: cd examples/matchem/mattergen && sbatch submit_train.sh
#SBATCH --job-name=mattergen_train
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
if [[ ! -f "$SCRIPT_DIR/train.py" ]]; then
    echo "ERROR: MatterGen root not found: $SCRIPT_DIR" >&2
    echo "Run this script from examples/matchem/mattergen or use demo/run.sh." >&2
    exit 1
fi

export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-onescience-mattergen-source}"
source "$SCRIPT_DIR/../matchem_env.sh"

NUM_DEVICES="${NUM_DEVICES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-16}"
SMOKE_TEST="${SMOKE_TEST:-0}"

TRAIN_ARGS=(
    data_module=mp_20
    "trainer.devices=$NUM_DEVICES"
    trainer.num_nodes=1
    "trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES"
    "data_module.batch_size.train=$TRAIN_BATCH_SIZE"
    "data_module.batch_size.val=$VAL_BATCH_SIZE"
    "data_module.batch_size.test=$VAL_BATCH_SIZE"
    data_module.num_workers.train=2
    data_module.num_workers.val=2
    data_module.num_workers.test=2
    '~trainer.logger'
)

if [[ -n "${CONFIG_PATH:-}" ]]; then
    mapfile -t TRAIN_ARGS < <(python3 - "$CONFIG_PATH" <<'PY'
import os
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
if config.get("task", "train") != "train":
    raise SystemExit("submit_train.sh only accepts task: train")
for key, value in config.get("args", {}).items():
    if isinstance(value, bool):
        if value:
            print(key)
    elif value is not None:
        print(f"{key}={os.path.expandvars(str(value))}")
PY
    )
fi

if [[ "$SMOKE_TEST" == "1" ]]; then
    TRAIN_ARGS+=(
        trainer.max_epochs=1
        trainer.check_val_every_n_epoch=1
        +trainer.limit_train_batches=1
        +trainer.limit_val_batches=1
        +trainer.num_sanity_val_steps=0
    )
fi

for arg in "${TRAIN_ARGS[@]}"; do
    case "$arg" in
        trainer.devices=*) NUM_DEVICES="${arg#*=}" ;;
        data_module.batch_size.train=*) TRAIN_BATCH_SIZE="${arg#*=}" ;;
        data_module.batch_size.val=*) VAL_BATCH_SIZE="${arg#*=}" ;;
        trainer.accumulate_grad_batches=*) ACCUMULATE_GRAD_BATCHES="${arg#*=}" ;;
    esac
done

echo "MatterGen DDP training"
echo "  devices: $NUM_DEVICES"
echo "  train batch per device: $TRAIN_BATCH_SIZE"
echo "  validation batch per device: $VAL_BATCH_SIZE"
echo "  gradient accumulation: $ACCUMULATE_GRAD_BATCHES"
echo "  effective global batch: $((NUM_DEVICES * TRAIN_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"

cd "$SCRIPT_DIR"
srun python train.py "${TRAIN_ARGS[@]}"
