#!/bin/bash
# ==========================================
# OneScience MatChem 环境一键安装脚本
# 用途：创建 conda 环境并安装 OneScience[matchem] 及其 DCU 依赖
# 用法：bash matchem_install.sh
# ==========================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 1. 交互式配置（环境变量可作为默认值/非交互回退） ----------
DEFAULT_MATCHEM_CONDA_NAME="${MATCHEM_CONDA_NAME:-matchem_opt}"
DEFAULT_ONESCIENCE_MAIN_DIR="${ONESCIENCE_MAIN_DIR:-/public/home/easyscience2024/wangrui/onescience}"

if [ -t 0 ]; then
    read -rp "请输入 conda 环境名 [默认: ${DEFAULT_MATCHEM_CONDA_NAME}]: " input_name
    MATCHEM_CONDA_NAME="${input_name:-${DEFAULT_MATCHEM_CONDA_NAME}}"

    read -rp "请输入 OneScience 源码根目录 [默认: ${DEFAULT_ONESCIENCE_MAIN_DIR}]: " input_dir
    ONESCIENCE_MAIN_DIR="${input_dir:-${DEFAULT_ONESCIENCE_MAIN_DIR}}"
else
    MATCHEM_CONDA_NAME="${DEFAULT_MATCHEM_CONDA_NAME}"
    ONESCIENCE_MAIN_DIR="${DEFAULT_ONESCIENCE_MAIN_DIR}"
    echo "[提示] 非交互模式，使用环境变量/默认值:"
    echo "  MATCHEM_CONDA_NAME=${MATCHEM_CONDA_NAME}"
    echo "  ONESCIENCE_MAIN_DIR=${ONESCIENCE_MAIN_DIR}"
fi

# 校验 OneScience 目录
if [ ! -d "${ONESCIENCE_MAIN_DIR}" ]; then
    echo "[错误] OneScience 源码目录不存在: ${ONESCIENCE_MAIN_DIR}"
    exit 1
fi

if [ ! -f "${ONESCIENCE_MAIN_DIR}/install.sh" ]; then
    echo "[错误] 在 ${ONESCIENCE_MAIN_DIR} 下未找到 install.sh，请确认目录正确。"
    exit 1
fi

# ---------- 2. 加载基础模块与环境 ----------
# 先关闭 set -u，避免 /etc/bashrc 中未绑定变量报错
set +u
source ~/.bashrc
set -u
module load sghpcdas/25.6
module load sghpc-mpi-gcc/26.3

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "[错误] 未找到 conda，请确认已正确加载 sghpcdas 模块。"
    exit 1
fi

# ---------- 3. 创建 conda 环境 ----------
# 若环境已存在则跳过，避免误覆盖
if conda env list | grep -qE "^${MATCHEM_CONDA_NAME}\s"; then
    echo "[提示] conda 环境 '${MATCHEM_CONDA_NAME}' 已存在，跳过创建。"
    echo "[提示] 如需重建，请先执行：conda remove -n ${MATCHEM_CONDA_NAME} --all -y"
else
    echo "[步骤 1/4] 创建 conda 环境：${MATCHEM_CONDA_NAME} ..."
    conda create -n "${MATCHEM_CONDA_NAME}" python=3.11 -y
fi

# ---------- 4. 安装 uv 工具 ----------
echo "[步骤 2/4] 激活环境并安装 uv ..."
conda activate "${MATCHEM_CONDA_NAME}"
python -m pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# ---------- 5. 安装 OneScience[matchem] ----------
echo "[步骤 3/4] 安装 OneScience[matchem] ..."
cd "${ONESCIENCE_MAIN_DIR}"
bash install.sh matchem

# ---------- 6. 安装验证 ----------
echo "[步骤 4/4] 验证安装结果 ..."
python -c "import torch; import onescience; print('torch 版本:', torch.__version__); print('onescience: 导入成功')"

# ---------- 7. 将配置写回 matchem_env.sh ----------
MATCHEM_ENV_FILE="${SCRIPT_DIR}/matchem_env.sh"
if [ -f "${MATCHEM_ENV_FILE}" ]; then
    echo "[提示] 更新 ${MATCHEM_ENV_FILE} ..."
    sed -i "s|^export MATCHEM_CONDA_NAME=.*|export MATCHEM_CONDA_NAME=${MATCHEM_CONDA_NAME}|" "${MATCHEM_ENV_FILE}"
    sed -i "s|^export ONESCIENCE_MAIN_DIR=.*|export ONESCIENCE_MAIN_DIR=${ONESCIENCE_MAIN_DIR}|" "${MATCHEM_ENV_FILE}"
else
    echo "[警告] 未找到 ${MATCHEM_ENV_FILE}，跳过写入配置。"
fi

echo ""
echo "============================================"
echo "  MatChem 环境安装完成"
echo "  环境名：${MATCHEM_CONDA_NAME}"
echo "  OneScience 目录：${ONESCIENCE_MAIN_DIR}"
echo "  激活命令：source ${MATCHEM_ENV_FILE}"
echo "============================================"
