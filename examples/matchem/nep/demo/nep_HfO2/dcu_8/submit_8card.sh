#!/bin/bash
#SBATCH --job-name=nep_HfO2_8card
#SBATCH --partition=hx1hdexclu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_8card_%j.out
#SBATCH --error=slurm_8card_%j.err

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

# MatPL 运行时环境（不侵入 matchem_env.sh，各算例自行维护）
# 校验MatPL根路径
if [[ -z "${MATPL_SRC_DIR}" || "${MATPL_SRC_DIR}" == "/path/to/matpl_dcu" || ! -d "${MATPL_SRC_DIR}" ]]; then
    echo "=============================================="
    echo "ERROR: MATPL_SRC_DIR 路径配置错误！"
    echo "当前值：${MATPL_SRC_DIR}"
    echo "请设置为真实有效的MatPL源码目录，示例："
    echo "export MATPL_SRC_DIR=/public/home/xxx/real_matpl_path"
    echo "=============================================="
    exit 1
fi

if [ -f "$MATPL_SRC_DIR/env.sh" ]; then
    source "$MATPL_SRC_DIR/env.sh"
fi
echo "MATPL_SRC_DIR = $MATPL_SRC_DIR"
export LD_LIBRARY_PATH="$MATPL_SRC_DIR/src/op/build/lib:${LD_LIBRARY_PATH:-}"

# 输入 JSON 使用环境变量，运行前展开为临时文件
INPUT_JSON="HfO2_nep_train_8card.json"
EXPANDED_JSON=".${INPUT_JSON%.json}_expanded_$$.json"
trap 'rm -f "$EXPANDED_JSON"' EXIT
python3 -c "import os; open('$EXPANDED_JSON','w').write(os.path.expandvars(open('$INPUT_JSON').read()))"

MatPL train "$EXPANDED_JSON"
