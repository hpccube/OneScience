#!/bin/bash
# -----------------------------------------------------------------------------
# LAMMPS + DeepMD C++ 接口 DCU 一键安装脚本
# 流程：交互式配置路径 → 下载预编译包 → 解压 → 安装/更新 TF DCU 库 → 验证 → 写回 matchem_env.sh
# 用法：bash lmp_install.sh
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 加载环境
echo ">>> Step 1: 加载环境"
source "$SCRIPT_DIR/../../matchem_env.sh"

# 2. 交互式配置安装路径
echo ">>> Step 2: 配置安装路径"
DEFAULT_LAMMPS_INSTALL="${LAMMPS_INSTALL_DIR:-/public/home/easyscience2024/wangrui/software/lammps_dcu}"
DEFAULT_DP_CPP="${DP_CPP_DIR:-/public/home/easyscience2024/wangrui/software/dp_cpp_dcu}"

if [ -t 0 ]; then
    read -rp "请输入 LAMMPS 安装路径 [默认: ${DEFAULT_LAMMPS_INSTALL}]: " input_lammps
    LAMMPS_INSTALL="${input_lammps:-${DEFAULT_LAMMPS_INSTALL}}"

    read -rp "请输入 DeepMD C++ 接口安装路径 [默认: ${DEFAULT_DP_CPP}]: " input_dp_cpp
    DP_CPP_INSTALL="${input_dp_cpp:-${DEFAULT_DP_CPP}}"
else
    LAMMPS_INSTALL="${DEFAULT_LAMMPS_INSTALL}"
    DP_CPP_INSTALL="${DEFAULT_DP_CPP}"
fi

echo "[提示] LAMMPS 安装路径: ${LAMMPS_INSTALL}"
echo "[提示] DeepMD C++ 接口安装路径: ${DP_CPP_INSTALL}"

# 3. 下载并安装 LAMMPS
echo ">>> Step 3: 下载并安装 LAMMPS"
LAMMPS_URL="https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/lammps_dcu.tar.gz"
mkdir -p "${LAMMPS_INSTALL}"
cd "${LAMMPS_INSTALL}"
curl -L -o lammps_dcu.tar.gz "${LAMMPS_URL}"
tar -xzf lammps_dcu.tar.gz --strip-components=1
rm -f lammps_dcu.tar.gz
echo ">>> Step 3: LAMMPS 安装完成（${LAMMPS_INSTALL}）"

# 后处理：更新 lib_override 符号链接到当前 conda 环境的 torch.libs
# 预编译包里的软链默认指向 matchem_opt，可能在新环境失效
echo ">>> Step 3: 更新 lib_override 符号链接"
TORCH_LIB_DIR="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [ -n "${TORCH_LIB_DIR}" ] && [ -d "${TORCH_LIB_DIR}" ]; then
    TORCH_LIBS_DIR="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "..", "torch.libs"))' 2>/dev/null || true)"
    cd "${LAMMPS_INSTALL}/lib_override"
    if [ -f "${TORCH_LIBS_DIR}/libnl-3-04364822.so.200.26.0" ]; then
        ln -sf "${TORCH_LIBS_DIR}/libnl-3-04364822.so.200.26.0" libnl-3.so.200
    fi
    if [ -f "${TORCH_LIBS_DIR}/libnl-route-3-9b7e574d.so.200.26.0" ]; then
        ln -sf "${TORCH_LIBS_DIR}/libnl-route-3-9b7e574d.so.200.26.0" libnl-route-3.so.200
    fi
    echo ">>> lib_override 已指向当前 conda 环境"
else
    echo "[警告] 未找到当前 conda 环境的 torch/lib，lib_override 未更新"
fi

# 4. 下载并安装 DeepMD C++ 接口
echo ">>> Step 4: 下载并安装 DeepMD C++ 接口"
DP_CPP_URL="https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz"
mkdir -p "${DP_CPP_INSTALL}"
cd "${DP_CPP_INSTALL}"
curl -L -o dp_cpp_dcu.tar.gz "${DP_CPP_URL}"
tar -xzf dp_cpp_dcu.tar.gz --strip-components=1
rm -f dp_cpp_dcu.tar.gz

# 后处理：创建 dpplugin.so 符号链接
cd "${DP_CPP_INSTALL}/lib"
if [ -f "deepmd_lmp/dpplugin.so" ] && [ ! -e "dpplugin.so" ]; then
    ln -s deepmd_lmp/dpplugin.so ./
fi
echo ">>> Step 4: DeepMD C++ 接口安装完成（${DP_CPP_INSTALL}）"

# 5. 安装/更新 DeepMD 推理所需的 TensorFlow DCU 库（dtk2604）
echo ">>> Step 5: 检查并安装 TensorFlow DCU 推理库"
TF_WHEEL_URL="https://download.sourcefind.cn:65024/file/4/tensorflow/DAS1.8/tensorflow-2.18.0+das.opt1.dtk2604-cp311-cp311-manylinux_2_28_x86_64.whl"

CURRENT_TF_VERSION="$(pip show tensorflow 2>/dev/null | awk '/^Version:/ {print $2}' || true)"
if echo "${CURRENT_TF_VERSION}" | grep -q "dtk2604"; then
    echo ">>> TensorFlow dtk2604 已安装（${CURRENT_TF_VERSION}），跳过"
else
    echo ">>> 当前 TensorFlow 版本：${CURRENT_TF_VERSION:-未安装}"
    echo ">>> 正在安装/更新到 dtk2604 版本 ..."
    pip install --no-cache-dir "${TF_WHEEL_URL}"
    if [ $? -ne 0 ]; then
        echo "[错误] TensorFlow dtk2604 安装失败，请检查网络或 pip 环境"
        exit 1
    fi
    echo ">>> TensorFlow dtk2604 安装完成"
fi

# 6. 验证安装
# 说明：登录节点通常无法直接启动 GPU 版 lmp_mpi，因此仅做依赖检查。
# 完整功能验证请提交 tools/lmp 下各测试算例的 submit.sh。
echo ">>> Step 6: 验证安装"

TORCH_LIB_DIR="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [ -z "${TORCH_LIB_DIR}" ] || [ ! -d "${TORCH_LIB_DIR}" ]; then
    TORCH_LIB_DIR="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib"
fi

TF_LIB_DIR="$(python -c 'import tensorflow, os; print(os.path.dirname(tensorflow.__file__))' 2>/dev/null || true)"
if [ -z "${TF_LIB_DIR}" ] || [ ! -d "${TF_LIB_DIR}" ]; then
    TF_LIB_DIR="${CONDA_PREFIX}/lib/python3.11/site-packages/tensorflow"
fi

# 使用 ROCM_PATH（由 matchem_env.sh 加载的模块设置），回退到固定路径
ROCM_LIB_DIR="${ROCM_PATH:-/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4}"

VERIFY_LD_LIBRARY_PATH="${LAMMPS_INSTALL}/lib64:${LAMMPS_INSTALL}/lib_override"
VERIFY_LD_LIBRARY_PATH="${ROCM_LIB_DIR}/lib:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="${ROCM_LIB_DIR}/lib64:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="${ROCM_LIB_DIR}/dcc/lib:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="${TF_LIB_DIR}:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/comm_libs/hwloc/lib:${VERIFY_LD_LIBRARY_PATH}"
VERIFY_LD_LIBRARY_PATH="${VERIFY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

export LAMMPS_PLUGIN_PATH="${DP_CPP_INSTALL}/lib"

# 检查 lmp_mpi 依赖
MISSING_LIBS=$(LD_LIBRARY_PATH="${VERIFY_LD_LIBRARY_PATH}" ldd "${LAMMPS_INSTALL}/bin/lmp_mpi" | grep "not found" || true)
if [ -z "${MISSING_LIBS}" ]; then
    echo ">>> lmp_mpi 依赖解析正常"
else
    echo "[警告] lmp_mpi 存在未解析依赖："
    echo "${MISSING_LIBS}"
fi

# 检查 DeepMD 插件依赖（需要 TF 库）
if [ -f "${DP_CPP_INSTALL}/lib/dpplugin.so" ]; then
    MISSING_DP_LIBS=$(LD_LIBRARY_PATH="${VERIFY_LD_LIBRARY_PATH}" ldd "${DP_CPP_INSTALL}/lib/dpplugin.so" | grep "not found" || true)
    if [ -z "${MISSING_DP_LIBS}" ]; then
        echo ">>> DeepMD 插件依赖解析正常"
    else
        echo "[警告] DeepMD 插件存在未解析依赖："
        echo "${MISSING_DP_LIBS}"
    fi
fi

echo "[提示] 登录节点不保证能正常启动 GPU 版 lmp_mpi；"
echo "       完整运行验证请使用 tools/lmp 下的测试 SLURM 脚本（如 deepmd/dp_pfd/submit.sh）。"

# 7. 将配置写回 matchem_env.sh
echo ">>> Step 7: 更新 matchem_env.sh"
MATCHEM_ENV_FILE="${SCRIPT_DIR}/../../matchem_env.sh"
if [ -f "${MATCHEM_ENV_FILE}" ]; then
    # 更新安装路径
    sed -i "s|^export LAMMPS_INSTALL_DIR=.*|export LAMMPS_INSTALL_DIR=\"${LAMMPS_INSTALL}\"|" "${MATCHEM_ENV_FILE}"
    sed -i "s|^export DP_CPP_DIR=.*|export DP_CPP_DIR=\"${DP_CPP_INSTALL}\"|" "${MATCHEM_ENV_FILE}"

    # 添加 LAMMPS 运行时环境变量（如果不存在）
    if ! grep -q "# ---------- 6. LAMMPS 运行时环境 ----------" "${MATCHEM_ENV_FILE}"; then
        cat <<EOF >> "${MATCHEM_ENV_FILE}"

# ---------- 6. LAMMPS 运行时环境 ----------
export LD_LIBRARY_PATH=\${LAMMPS_INSTALL_DIR}/lib64:\${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=\${LAMMPS_INSTALL_DIR}/lib_override:\${LD_LIBRARY_PATH:-}
export LAMMPS_PLUGIN_PATH=\${DP_CPP_DIR}/lib
EOF
        echo ">>> 已添加 LAMMPS 运行时环境变量到 ${MATCHEM_ENV_FILE}"
    else
        echo ">>> LAMMPS 运行时环境变量已存在，跳过"
    fi
else
    echo "[警告] 未找到 ${MATCHEM_ENV_FILE}，跳过写入配置。"
fi

echo ""
echo "=========================================="
echo " LAMMPS + DeepMD C++ 接口 DCU 安装完成!"
echo "=========================================="
echo "LAMMPS 安装路径: ${LAMMPS_INSTALL}"
echo "DP_CPP 安装路径: ${DP_CPP_INSTALL}"
echo ""
echo "每次使用前请执行:"
echo "  source ${SCRIPT_DIR}/../../matchem_env.sh"
