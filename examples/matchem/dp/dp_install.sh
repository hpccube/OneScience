#!/bin/bash
# -----------------------------------------------------------------------------
# DeepMD-kit DCU 一键安装脚本
# 流程：源码拉取 → 编译安装 → 安装验证
# 用法：bash dp_install.sh
#       DEEPMD_SRC_DIR=/path/to/src bash dp_install.sh  # 指定源码路径
# -----------------------------------------------------------------------------
set -euo pipefail
trap 'echo "[错误] 脚本被中断"; exit 130' INT
DP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DP_SCRIPT_DIR"

# 1. 环境准备
echo ">>> Step 1: 加载环境"
set +u  # 临时关闭未定义变量检查，兼容 /etc/bashrc 等系统配置
source "$DP_SCRIPT_DIR/../matchem_env.sh"
set -u  # 恢复严格检查

# 1.5 确保 gflags/glog 运行库存在（torch cmake 与 lmp_mpi 的运行时依赖，pip 环境通常缺失）
echo ">>> Step 1.5: 检查 gflags/glog 运行库"
MISSING_PKGS=""
ls "$CONDA_PREFIX"/lib/libgflags.so* >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} gflags"
ls "$CONDA_PREFIX"/lib/libglog.so* >/dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} glog"
if [ -n "${MISSING_PKGS}" ]; then
    echo ">>> 安装缺失的运行库:${MISSING_PKGS}（conda-forge）"
    conda install -y -c conda-forge ${MISSING_PKGS} || {
    echo "[错误] conda 安装 gflags/glog 失败"
    exit 1
    }
else
    echo ">>> gflags/glog 已存在，跳过"
fi

# 下载辅助：优先 curl，失败时回退 wget（部分节点 curl 存在 TLS/代理问题）
download_file() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1 && curl -fL -o "$out" "$url"; then
        return 0
    fi
    echo "[提示] curl 下载失败，改用 wget: $url"
    if command -v wget >/dev/null 2>&1 && wget -O "$out" "$url"; then
        return 0
    fi
    echo "[错误] 下载失败: $url"
    return 1
}

# 2. 交互式配置源码路径
DEEPMD_SRC="${DEEPMD_SRC_DIR:-${DP_SCRIPT_DIR}/deepmd-kit}"
if [ -t 0 ]; then
    read -rp "请输入 DeepMD-kit 源码路径 [默认: ${DEEPMD_SRC}]: " input_src
    DEEPMD_SRC="${input_src:-${DEEPMD_SRC}}"
fi
echo "[提示] 使用 DeepMD-kit 源码路径: ${DEEPMD_SRC}"

# 3. 源码准备
# 说明：
#   - 开发/测试阶段：自动通过 HTTPS + 代理拉取源码
#   - 生产/客户场景：建议提前上传源码到集群，通过 DEEPMD_SRC_DIR 指定
if [ ! -d "$DEEPMD_SRC/.git" ] && [ ! -f "$DEEPMD_SRC/setup.py" ]; then
    echo ">>> Step 2: 拉取 DeepMD-kit 源码"
    # 计算节点需通过 HTTP 代理访问外网，配置 git 代理，如代理实效，可向集群管理员重新申请
    git config --global http.proxy "http://scnethpc2601:sWMtqVS@10.16.1.52:3120"
    git clone --depth 1 "https://gitee.com/wang-rui-sugon/deepmd-kit_dcu.git" "$DEEPMD_SRC" || {
    echo "[错误] 源码拉取失败"
    exit 1
    }
else
    echo ">>> Step 2: 源码已存在，跳过拉取"
fi

# 3. 预先锁定 numpy 版本，避免 deepmd-kit 安装过程中短暂升级到不兼容版本
echo ">>> Step 3: 预先锁定 numpy 版本"
pip install numpy==1.26.3 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn || {
    echo "[错误] numpy 安装失败"
    exit 1
}

# 4. 修复 Torch cmake 硬编码 DTK 路径
echo ">>> Step 4: 修复 Torch cmake 硬编码 DTK 路径"
# 使用 importlib.util.find_spec 避免在登录节点触发 torch 的动态库加载
TORCH_PATH=$(python -c "import importlib.util, os; spec = importlib.util.find_spec('torch'); print(os.path.dirname(spec.origin) if spec and spec.origin else '')")
CAFFE2_CMAKE="${TORCH_PATH}/share/cmake/Caffe2/Caffe2Targets.cmake"
DTK_REAL_PATH="/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4"
if [ -f "$CAFFE2_CMAKE" ] && grep -q '/opt/dtk' "$CAFFE2_CMAKE"; then
    echo ">>> 替换 Caffe2Targets.cmake 中的 /opt/dtk 为实际路径"
    sed -i "s|/opt/dtk|${DTK_REAL_PATH}|g" "$CAFFE2_CMAKE"
fi

# 5. 编译安装（PyTorch + TensorFlow 双后端）
echo ">>> Step 5: 编译安装 Python 包（PyTorch + TensorFlow 双后端）"
cd "$DEEPMD_SRC"
DP_VARIANT=rocm \
ROCM_ROOT="$ROCM_PATH" \
DP_ENABLE_TENSORFLOW=1 \
DP_ENABLE_PYTORCH=1 \
PYTORCH_ROOT="${TORCH_PATH}" \
    pip install . "numpy==1.26.3" -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn || {
    echo "[错误] DeepMD-kit Python 包安装失败"
    exit 1
}

# 6. 验证
echo ">>> Step 6: 验证安装"
if ! dp -h >/dev/null 2>&1; then
    echo "[错误] dp 命令不可用，Python 包安装可能失败"
    exit 1
fi
dp -h | head -n 5

echo "========================================"
echo " DeepMD-kit Python 包安装完成"
echo "========================================"

# 7. C++ 接口安装（含 LAMMPS 插件）
# 说明：默认从预编译包下载解压，快速部署；如需自行源码编译，设置 COMPILE_DP_CPP=1。
DP_CPP_URL="https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz"

# DP_CPP_DIR 默认路径：放在 dp_install.sh 同级目录的 dp_cpp_dcu 下
DP_CPP_DIR_DEFAULT="${DP_SCRIPT_DIR}/dp_cpp_dcu"

# 如果环境变量里还是占位符或未设置，用默认值覆盖
if [ -z "${DP_CPP_DIR:-}" ] || [ "${DP_CPP_DIR}" = "/path/to/dp_cpp_dcu" ]; then
    DP_CPP_DIR="${DP_CPP_DIR_DEFAULT}"
fi

# 交互式输入安装路径
if [ -t 0 ]; then
    read -rp "请输入 DeepMD C++ 接口安装路径 [默认: ${DP_CPP_DIR}]: " input_cpp
    DP_CPP_DIR="${input_cpp:-${DP_CPP_DIR}}"
fi

# 转换为绝对路径，避免 source matchem_env.sh 时相对路径解析出错
DP_CPP_DIR="$(cd "${DP_CPP_DIR}" 2>/dev/null && pwd || echo "${DP_CPP_DIR}")"
export DP_CPP_DIR
echo "[提示] 使用 DeepMD C++ 接口安装路径: ${DP_CPP_DIR}"

if [ "${COMPILE_DP_CPP:-0}" = "1" ]; then
    echo ">>> Step 7: 源码编译 C++ 接口（含 LAMMPS 插件）"

    # 7.1 Patch Gelu op：TensorFlow 2.18+ 已内置 Gelu，与 deepmd-kit 自定义 op 冲突，
    #     需在编译前注释掉 source/op/tf/gelu_multi_device.cc 中的 REGISTER_OP("Gelu")
    #     和 REGISTER_OP("GeluGrad") 及其属性链。（GeluGradGrad / GeluCustom 系列不受影响）
    GELU_FILE="$DEEPMD_SRC/source/op/tf/gelu_multi_device.cc"
    if [ -f "$GELU_FILE" ] && grep -q '^REGISTER_OP("Gelu")' "$GELU_FILE"; then
        echo ">>> Step 7.1: Patch Gelu op 注册，避免 TF 2.18+ 冲突"
        sed -i '/^REGISTER_OP("Gelu")$/,/^);$/{ /^$/!s/^/\/\/ /; }' "$GELU_FILE"
        sed -i '/^REGISTER_OP("GeluGrad")$/,/^);$/{ /^$/!s/^/\/\/ /; }' "$GELU_FILE"
    else
        echo "[警告] 未找到 Gelu 注册代码，跳过 patch"
    fi

    cd "$DEEPMD_SRC/source"
    mkdir -p build && cd build

    cmake -DENABLE_TENSORFLOW=ON \
          -DENABLE_PYTORCH=ON \
          -DUSE_ROCM_TOOLKIT=ON \
          -DTENSORFLOW_ROOT="${CONDA_PREFIX}/lib/python3.11/site-packages/tensorflow" \
          -DTensorFlow_INCLUDE_DIRS="${CONDA_PREFIX}/lib/python3.11/site-packages/tensorflow/include" \
          -DTorch_DIR="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/share/cmake/Torch" \
          -DHIP_ROOT_DIR="${ROCM_PATH}/hip" \
          -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};${ROCM_PATH}/lib/cmake" \
          -DLAMMPS_SOURCE_ROOT="${LAMMPS_SRC_DIR}" \
          -DCMAKE_INSTALL_PREFIX="${DP_CPP_DIR}" \
          ..

    make -j$(nproc)
    make install

    # 后处理：创建 dpplugin.so 符号链接
    cd "${DP_CPP_DIR}/lib"
    if [ -f "deepmd_lmp/dpplugin.so" ] && [ ! -e "dpplugin.so" ]; then
        ln -sf deepmd_lmp/dpplugin.so ./
    fi
else
    echo ">>> Step 7: 下载预编译 C++ 接口包"
    mkdir -p "${DP_CPP_DIR}"
    cd "${DP_CPP_DIR}"
    download_file "${DP_CPP_URL}" dp_cpp_dcu.tar.gz || exit 1
    tar -xzf dp_cpp_dcu.tar.gz --strip-components=1 || {
    echo "[错误] 解压 C++ 接口包失败"
    exit 1
    }
    rm -f dp_cpp_dcu.tar.gz
    echo ">>> Step 7: C++ 接口安装完成（${DP_CPP_DIR}）"
fi

# 8. 将配置写回 matchem_env.sh
MATCHEM_ENV_FILE="${DP_SCRIPT_DIR}/../matchem_env.sh"
if [ -f "${MATCHEM_ENV_FILE}" ]; then
    echo "[提示] 更新 ${MATCHEM_ENV_FILE} "
    sed -i "s|^export DEEPMD_SRC_DIR=.*|export DEEPMD_SRC_DIR=${DEEPMD_SRC}|" "${MATCHEM_ENV_FILE}"
    sed -i "s|^export DP_CPP_DIR=.*|export DP_CPP_DIR=${DP_CPP_DIR}|" "${MATCHEM_ENV_FILE}"
else
    echo "[警告] 未找到 ${MATCHEM_ENV_FILE}，跳过写入配置。"
fi

echo ""
echo "=========================================="
echo " DeePMD-kit DCU 安装完成!"
echo " 支持 Pytorch/Tensorflow后端训练"
echo "=========================================="
echo "源码路径: $DEEPMD_SRC"
echo ""
echo "每次使用前请执行:"
echo "  source $DP_SCRIPT_DIR/../matchem_env.sh"


