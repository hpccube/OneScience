#!/bin/bash
#SBATCH --job-name=dp_tf_train
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

if [[ -n "${SCRIPT_DIR}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "ERROR: 未检测到外部 SCRIPT_DIR 变量，也不在 Slurm 任务环境（SLURM_SUBMIT_DIR 为空）"
    exit 1
fi
echo $SCRIPT_DIR

export MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-test_pip}"
source "$SCRIPT_DIR/matchem_env.sh"

# DeepMD 训练环境已由 matchem_env.sh 覆盖

# 限制 batch size 和线程数，规避 ROCm kernel launch 失败
export DP_INFER_BATCH_SIZE=4096
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1
export OMP_NUM_THREADS=1
export TF_XLA_FLAGS=""                                                                                                                                                                                                                                     
export TF_ENABLE_XLA=0    
# TF 后端单卡训练

# 输入 JSON 使用环境变量，运行前展开为临时文件
INPUT_JSON="input_tf.json"
EXPANDED_JSON=".${INPUT_JSON%.json}_expanded_$$.json"
trap 'rm -f "$EXPANDED_JSON"' EXIT
python3 -c "import os; open('$EXPANDED_JSON','w').write(os.path.expandvars(open('$INPUT_JSON').read()))"

dp --tf train "$EXPANDED_JSON"
