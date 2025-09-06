#!/bin/bash

cd /work/home/onescience2025/khren/nemo/bionemo-framework/sub-packages/bionemo-evo2/evo2_7b

DIRS=(
    "./lightning_logs"
    "./results"
)

for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "删除文件夹: $DIR"
        rm -rf "$DIR"
    else
        echo "文件夹不存在: $DIR"
    fi
done

srun --ntasks-per-node=1  python /work/home/onescience2025/biao.liu/onescience-evo2/examples/evo2/example/train.py\
    -d /work/home/onescience2025/biao.liu/onescience-evo2/examples/evo2/config/training_data_config.yaml\
    --dataset-dir /work/home/onescience2025/khren/onescience/data/promoters/pretraining_data_promoters\
    --model-size 7b_arc_longcontext \
    --devices 1 \
    --num-nodes 1 \
    --seq-length 1024 \
    --micro-batch-size 2 \
    --lr 0.0001 \
    --warmup-steps 5 \
    --max-steps 100 \
    --ckpt-dir /work/home/onescience2025/khren/nemo/bionemo-framework/sub-packages/bionemo-evo2/evo2_7b/nemo2_evo2_7b \
    --clip-grad 1 \
    --wd 0.01 \
    --activation-checkpoint-recompute-num-layers 1 \
    --val-check-interval 50 \
    --ckpt-async-save