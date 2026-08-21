#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# OneScience DeepMD-kit DCU 一键安装脚本
# 流程：加载既有环境 -> 下载依赖 -> 拉取固定源码 -> 编译安装 -> 安装旧 C++ 包
# 用法：bash dp_install.sh
#       DEEPMD_SRC_DIR=/path/to/source bash dp_install.sh  # 可选源码缓存位置
# -----------------------------------------------------------------------------

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# matchem_env.sh 为旧版 DeepMD 定义过 DEEPMD_SRC_DIR。这里只接受用户在
# 执行本脚本前显式传入的覆盖值，避免误用旧源码目录。
DEEPMD_SRC_OVERRIDE="${DEEPMD_SRC_DIR:-}"
DP_CPP_OVERRIDE="${DEEPMD_CPP_DIR:-${DP_CPP_DIR:-}}"

if (($# != 0)); then
    echo "错误：dp_install.sh 不需要参数，直接执行 bash dp_install.sh" >&2
    exit 2
fi

# 固定并验证过的软件版本。
DEEPMD_REPOSITORY_URL="https://gitee.com/wang-rui-sugon/deepmd-kit_dcu.git"
DEEPMD_SOURCE_BRANCH="dpa4-torch251"
DEEPMD_SOURCE_COMMIT="40a7d99fa46c8ff1e75b5be9d64540d95dbac184"
DEEPMD_PACKAGE_VERSION="3.2.0b1.dev194+g40a7d99fa"
TORCH_VERSION="2.5.1+das.opt1.dtk25042"
TENSORFLOW_VERSIONS="2.18.0+das.opt1.dtk25042 2.18.0+das.opt1.dtk2604"
TRITON_VERSION="3.1.0+das.opt1.dtk25042"

# DeepMD 运行时缺失项使用候选环境中验证过的版本。bracex、colorama 和
# pyfiglet 虽然是间接依赖，也显式固定，防止不同账号解析出不同版本。这里
# 不包含 NumPy、Pandas、Pydantic 等 OneScience 基础依赖，避免改动现有环境。
DEEPMD_RUNTIME_PACKAGES=(
    "array-api-compat==1.15.0"
    "dargs==0.5.0.post0"
    "mendeleev==1.0.0"
    "msgpack==1.2.1"
    "wcmatch==11.0"
    "bracex==3.0.1"
    "colorama==0.4.6"
    "pyfiglet==0.8.post1"
)

# vesin 的 Python wheel 使用官方包；vesin-torch 使用已针对 DTK PyTorch
# 2.5.1 编译并验证的固定 wheel，避免每次安装都重复编译。
VESIN_URL="https://files.pythonhosted.org/packages/b6/24/ef992ce7a8491b8f1e223febce36ebe1221f24bf5f60f62aa3dba69ae43b/vesin-0.6.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
VESIN_SHA256="27c4d838bbe315837cf7ca62597b94581235c269dce4d1d3600749f64396e746"
VESIN_TORCH_URL="https://gitee.com/zhyzsj/onescience-deepmd-wheels/raw/a2f7b7661855fb9ba1cdae54d48380965cdbf699/wheels/vesin_torch-0.6.1-py3-none-linux_x86_64.whl"
VESIN_TORCH_SHA256="c162c9d7c080047079a663f03f3d15e65d6ce7b07ca1b83ef9ecf19daa2e0bde"
DP_CPP_URL="https://download.sourcefind.cn:65024/file/9/onesicence/dtk-25.04.2/deep_lammps/dp_cpp_dcu.tar.gz"

# 安装器中的函数只封装重复的日志、失败退出、下载和路径处理；训练算例不依赖
# 这些函数。集中处理可以确保每一步使用相同的错误语义和缓存规则。
log() {
    printf '\n>>> %s\n' "$*"
}

die() {
    echo "错误：$*" >&2
    exit 1
}

download_file() {
    local url="$1"
    local output="$2"
    local expected_sha="${3:-}"

    if [[ ! -f "${output}" ]]; then
        mkdir -p "$(dirname "${output}")"
        local downloaded=0
        if command -v curl >/dev/null 2>&1 \
            && curl -fL --retry 3 -o "${output}.part" "${url}"; then
            downloaded=1
        else
            rm -f "${output}.part"
            echo "提示：curl 下载不可用，改用 wget：${url}"
            if command -v wget >/dev/null 2>&1 \
                && wget -O "${output}.part" "${url}"; then
                downloaded=1
            fi
        fi
        ((downloaded == 1)) || {
            rm -f "${output}.part"
            die "下载失败：${url}"
        }
        mv "${output}.part" "${output}"
    fi

    # 有公开 SHA256 的制品必须校验；旧 C++ 包尚无公开 SHA 清单，只复用
    # 同一套原子下载与缓存逻辑，不伪造校验值。
    if [[ -n "${expected_sha}" ]]; then
        local actual_sha
        actual_sha="$(sha256sum "${output}" | awk '{print $1}')"
        [[ "${actual_sha}" == "${expected_sha}" ]] || {
            rm -f "${output}"
            die "文件校验失败：${output}"
        }
    fi
}

# 将用户输入路径规范为绝对路径；目标尚不存在时也可以正常解析。
absolute_path() {
    python - "$1" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

# 防止源码和 C++ 包落入 Git 管理的 dp/ 目录，污染项目工作树。
reject_project_install_path() {
    local label="$1"
    local path="$2"
    case "${path}/" in
        "${SCRIPT_DIR}/"*)
            die "${label}不能放在项目 dp 目录中：${path}"
            ;;
    esac
}

# DTK 安装位置由当前 hipcc 反向解析，不绑定某个集群的绝对路径。
resolve_dtk_root() {
    local hipcc_path hipcc_dir candidate
    hipcc_path="$(readlink -f "$(command -v hipcc)")"
    hipcc_dir="$(dirname "${hipcc_path}")"
    for candidate in "${hipcc_dir}/.." "${hipcc_dir}/../.."; do
        candidate="$(cd "${candidate}" 2>/dev/null && pwd -P || true)"
        if [[ -f "${candidate}/lib/libamdhip64.so" && -f "${candidate}/lib/libhiprtc.so" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    die "无法根据 hipcc 找到 DTK 根目录：${hipcc_path}"
}

make_torch_cmake_overlay() {
    local build_dir="$1"
    local dtk_root="$2"
    local torch_root="$3"
    local torch_cmake overlay_root overlay_cmake target_file

    # DTK Torch wheel 的 CMake 文件含构建机 ROCm 路径。复制到一次性 overlay
    # 后只修正已审计的 3 处路径，不修改 Conda 环境中的原始 Torch 文件。
    torch_cmake="${torch_root}/share/cmake"
    overlay_root="${build_dir}/torch-overlay"
    overlay_cmake="${overlay_root}/share/cmake"
    mkdir -p "${overlay_root}/share"
    cp -a "${torch_cmake}" "${overlay_cmake}"
    ln -s "${torch_root}/lib" "${overlay_root}/lib"
    ln -s "${torch_root}/include" "${overlay_root}/include"
    target_file="${overlay_cmake}/Caffe2/Caffe2Targets.cmake"

    python - "${target_file}" "${dtk_root}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
dtk = pathlib.Path(sys.argv[2])
text = path.read_text()
text, amd_count = re.subn(
    r'/[^";]*?/lib/libamdhip64\.so', str(dtk / "lib/libamdhip64.so"), text
)
text, rtc_count = re.subn(
    r'/[^";]*?/lib/libhiprtc\.so', str(dtk / "lib/libhiprtc.so"), text
)
if amd_count != 2 or rtc_count != 1:
    raise SystemExit(
        f"Torch CMake 路径数量异常：amdhip64={amd_count}, hiprtc={rtc_count}"
    )
path.write_text(text)
PY
    printf '%s\n' "${overlay_cmake}"
}

# 1. 环境准备
log "Step 1: 加载 OneScience MatChem 环境"
# 环境名的唯一默认值由 matchem_env.sh 管理；调用者可在运行本脚本前设置
# MATCHEM_CONDA_NAME，安装器本身不再覆盖。
# shellcheck source=../matchem_env.sh
source "${SCRIPT_DIR}/../matchem_env.sh"

[[ "${CONDA_DEFAULT_ENV:-}" == "${MATCHEM_CONDA_NAME}" ]] \
    || die "Conda 环境激活失败：${MATCHEM_CONDA_NAME}"
[[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] \
    || die "统一 DeepMD 环境要求 Python 3.11"
command -v git >/dev/null 2>&1 || die "没有找到 git"
command -v cmake >/dev/null 2>&1 || die "没有找到 cmake"
command -v hipcc >/dev/null 2>&1 || die "没有找到 hipcc，请检查 MatChem/DTK 环境"

# 下载、Conda 回退缓存、源码和构建结果统一放在用户缓存目录中。
CACHE_ROOT="${DEEPMD_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/onescience/deepmd-unified}"
CACHE_ROOT="$(absolute_path "${CACHE_ROOT}")"
mkdir -p "${CACHE_ROOT}"

# matchem_env.sh 的历史默认值是相对路径。若用户在运行安装器前已经 source
# 过该文件，不能把这些相对默认值误判成显式安装位置。
if [[ -n "${DEEPMD_SRC_OVERRIDE}" && "${DEEPMD_SRC_OVERRIDE}" != /* && "${DEEPMD_SRC_OVERRIDE}" != "~/"* ]]; then
    echo "提示：忽略相对 DEEPMD_SRC_DIR=${DEEPMD_SRC_OVERRIDE}，改用用户缓存目录"
    DEEPMD_SRC_OVERRIDE=""
fi
if [[ -n "${DP_CPP_OVERRIDE}" && "${DP_CPP_OVERRIDE}" != /* && "${DP_CPP_OVERRIDE}" != "~/"* ]]; then
    echo "提示：忽略相对 DP_CPP_DIR=${DP_CPP_OVERRIDE}，改用用户缓存目录"
    DP_CPP_OVERRIDE=""
fi

DEEPMD_SRC="${DEEPMD_SRC_OVERRIDE:-${CACHE_ROOT}/source}"
DP_CPP_INSTALL_DIR="${DP_CPP_OVERRIDE:-${CACHE_ROOT}/dp_cpp_dcu}"
if [[ -t 0 ]]; then
    read -r -p "请输入 DeepMD-kit 源码目录 [默认: ${DEEPMD_SRC}]: " input_src
    DEEPMD_SRC="${input_src:-${DEEPMD_SRC}}"
    read -r -p "请输入 DeepMD C++ 接口目录 [默认: ${DP_CPP_INSTALL_DIR}]: " input_cpp
    DP_CPP_INSTALL_DIR="${input_cpp:-${DP_CPP_INSTALL_DIR}}"
fi
DEEPMD_SRC="$(absolute_path "${DEEPMD_SRC}")"
DP_CPP_INSTALL_DIR="$(absolute_path "${DP_CPP_INSTALL_DIR}")"
reject_project_install_path "DeepMD-kit 源码目录" "${DEEPMD_SRC}"
reject_project_install_path "DeepMD C++ 接口目录" "${DP_CPP_INSTALL_DIR}"
echo "DeepMD Python 包目录：${CONDA_PREFIX}/lib/python3.11/site-packages/deepmd"
echo "DeepMD-kit 源码目录：${DEEPMD_SRC}"
echo "DeepMD C++ 接口目录：${DP_CPP_INSTALL_DIR}"

# 2. C++ 算子运行时依赖
log "Step 2: 检查 gflags、glog 和 msgpack-c"
MISSING_PKGS=()
compgen -G "${CONDA_PREFIX}/lib/libgflags.so*" >/dev/null || MISSING_PKGS+=(gflags)
compgen -G "${CONDA_PREFIX}/lib/libglog.so*" >/dev/null || MISSING_PKGS+=(glog)
compgen -G "${CONDA_PREFIX}/lib/libmsgpackc.so.2*" >/dev/null || MISSING_PKGS+=(msgpack-c=3.3.0)
if ((${#MISSING_PKGS[@]})); then
    echo "安装缺少的运行库：${MISSING_PKGS[*]}"
    if ! conda install -y -c conda-forge "${MISSING_PKGS[@]}"; then
        echo "提示：当前 Conda 镜像下载失败，改用官方 conda-forge 和独立缓存重试"
        CONDA_FALLBACK_PKGS_DIR="${CACHE_ROOT}/conda-pkgs"
        mkdir -p "${CONDA_FALLBACK_PKGS_DIR}"
        CONDA_PKGS_DIRS="${CONDA_FALLBACK_PKGS_DIR}" \
            conda install -y --override-channels \
            -c https://conda.anaconda.org/conda-forge \
            "${MISSING_PKGS[@]}"
    fi
else
    echo "gflags、glog 和 msgpack-c 已存在"
fi
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# 3. 核心框架版本检查
log "Step 3: 检查 PyTorch、TensorFlow 和 Triton 版本"
python - "${TORCH_VERSION}" "${TENSORFLOW_VERSIONS}" "${TRITON_VERSION}" <<'PY'
import importlib.metadata
import sys

torch_expected, tensorflow_expected, triton_expected = sys.argv[1:]
expected_versions = {
    "torch": {torch_expected},
    "tensorflow": set(tensorflow_expected.split()),
    "triton": {triton_expected},
}
for package, expected in expected_versions.items():
    actual = importlib.metadata.version(package)
    if actual not in expected:
        choices = " 或 ".join(sorted(expected))
        raise SystemExit(f"{package} 版本不匹配：需要 {choices}，当前 {actual}")
    print(f"{package}: {actual}")
PY

# 安装缓存位于用户缓存目录，不会改变 dp/ 的三项目录结构。
DOWNLOAD_DIR="${CACHE_ROOT}/downloads"
BUILD_ROOT="${DEEPMD_BUILD_DIR:-${CACHE_ROOT}/build}"
BUILD_JOBS="${DEEPMD_BUILD_JOBS:-4}"
mkdir -p "${DOWNLOAD_DIR}" "${BUILD_ROOT}"

# 4. 补齐 DeepMD Python 运行依赖
log "Step 4: 安装 DeepMD Python 运行依赖"
python -m pip install --upgrade-strategy only-if-needed \
    "${DEEPMD_RUNTIME_PACKAGES[@]}"

# 5. 下载固定依赖
log "Step 5: 下载并校验 vesin 0.6.1 和 vesin-torch 0.6.1"
VESIN_WHEEL="${DOWNLOAD_DIR}/vesin-0.6.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
VESIN_TORCH_WHEEL="${DOWNLOAD_DIR}/vesin_torch-0.6.1-py3-none-linux_x86_64.whl"
download_file "${VESIN_URL}" "${VESIN_WHEEL}" "${VESIN_SHA256}"
download_file "${VESIN_TORCH_URL}" "${VESIN_TORCH_WHEEL}" "${VESIN_TORCH_SHA256}"

# 6. 拉取并固定 DeepMD-kit 源码
log "Step 6: 准备 DeepMD-kit 源码"
if [[ ! -e "${DEEPMD_SRC}/.git" ]]; then
    mkdir -p "$(dirname "${DEEPMD_SRC}")"
    git clone --branch "${DEEPMD_SOURCE_BRANCH}" --single-branch \
        "${DEEPMD_REPOSITORY_URL}" "${DEEPMD_SRC}" || die \
        "无法拉取 ${DEEPMD_SOURCE_BRANCH}；请先确认该分支已经推送到 Gitee"
fi
[[ -z "$(git -C "${DEEPMD_SRC}" status --porcelain)" ]] \
    || die "源码目录存在未提交修改：${DEEPMD_SRC}"
git -C "${DEEPMD_SRC}" fetch origin "${DEEPMD_SOURCE_BRANCH}" || die \
    "无法获取 Gitee 分支 ${DEEPMD_SOURCE_BRANCH}"
REMOTE_COMMIT="$(git -C "${DEEPMD_SRC}" rev-parse FETCH_HEAD^{commit})"
[[ "${REMOTE_COMMIT}" == "${DEEPMD_SOURCE_COMMIT}" ]] \
    || die "Gitee 分支提交不正确：需要 ${DEEPMD_SOURCE_COMMIT}，当前 ${REMOTE_COMMIT}"
git -C "${DEEPMD_SRC}" checkout --detach "${DEEPMD_SOURCE_COMMIT}"
echo "DeepMD 源码：${DEEPMD_SRC}"
echo "DeepMD 提交：$(git -C "${DEEPMD_SRC}" rev-parse HEAD)"

# 7. 创建本次专用构建目录
log "Step 7: 创建干净构建目录"
BUILD_DIR="$(mktemp -d "${BUILD_ROOT}/build.XXXXXXXX")"
WHEEL_DIR="${BUILD_DIR}/wheelhouse"
mkdir -p "${WHEEL_DIR}"
python -m venv --system-site-packages "${BUILD_DIR}/build-venv"
BUILD_PYTHON="${BUILD_DIR}/build-venv/bin/python"
"${BUILD_PYTHON}" -m pip install \
    "dependency-groups==1.3.1" \
    "scikit-build-core==1.0.3" \
    "hatch-fancy-pypi-readme==25.1.0" \
    "setuptools-scm==10.2.1"

# 8. 编译 DeepMD-kit（PyTorch + TensorFlow 双后端）
log "Step 8: 编译 DeepMD-kit Python wheel"
DTK_ROOT="$(resolve_dtk_root)"
TORCH_ROOT="$(python - <<'PY'
import importlib.util
import pathlib
spec = importlib.util.find_spec("torch")
if spec is None or spec.origin is None:
    raise SystemExit("torch is not installed")
print(pathlib.Path(spec.origin).parent)
PY
)"
TORCH_OVERLAY="$(make_torch_cmake_overlay "${BUILD_DIR}" "${DTK_ROOT}" "${TORCH_ROOT}")"
export SKBUILD_BUILD_DIR="${BUILD_DIR}/deepmd-cmake-build"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"
export DP_VARIANT=rocm
export ROCM_ROOT="${DTK_ROOT}"
export DP_ENABLE_TENSORFLOW=1
export DP_ENABLE_PYTORCH=1
export PYTORCH_ROOT="${TORCH_ROOT}"
# Gitee 只需托管固定源码分支，不要求额外复制上游全部 Git tags。
# 该版本号来自同一提交在完整上游 tag 历史中的 setuptools-scm 结果。
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DEEPMD_KIT="${DEEPMD_PACKAGE_VERSION}"
# 某些账号环境会暴露外部 Intel MKL。若让 Torch CMake 自动发现它，生成的
# deepmd_op_pt 会错误依赖计算节点未提供的 ILP64 .so.2；固定使用 Torch wheel
# 自带的 LP64 MKL 实现，不把账号环境中的外部 MKL 带进 DeepMD wheel。
unset MKLROOT MKL_ROOT MKL_DIR MKL_INTERFACE_LAYER
export CMAKE_ARGS="${CMAKE_ARGS:-} -DTorch_DIR=${TORCH_OVERLAY}/Torch -DCaffe2_DIR=${TORCH_OVERLAY}/Caffe2 -DCMAKE_DISABLE_FIND_PACKAGE_MKL=TRUE"
"${BUILD_PYTHON}" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "${WHEEL_DIR}" "${DEEPMD_SRC}"
DEEPMD_WHEEL="$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'deepmd_kit-*.whl' -print -quit)"
[[ -n "${DEEPMD_WHEEL}" ]] || die "没有生成 DeepMD-kit wheel"

# 在安装前审计 PyTorch 自定义算子的直接动态库依赖，防止构建环境再次把
# 外部 Intel MKL 注入 wheel。已验证构建不应直接依赖任何 libmkl_* 库。
DEEPMD_WHEEL_AUDIT_DIR="${BUILD_DIR}/deepmd-wheel-audit"
mkdir -p "${DEEPMD_WHEEL_AUDIT_DIR}"
"${BUILD_PYTHON}" -m zipfile -e "${DEEPMD_WHEEL}" "${DEEPMD_WHEEL_AUDIT_DIR}"
DEEPMD_OP_PT="$(find "${DEEPMD_WHEEL_AUDIT_DIR}" -type f -name 'libdeepmd_op_pt.so' -print -quit)"
[[ -n "${DEEPMD_OP_PT}" ]] || die "DeepMD wheel 中没有找到 libdeepmd_op_pt.so"
command -v readelf >/dev/null 2>&1 || die "没有找到 readelf，无法审计 DeepMD wheel"
if readelf -d "${DEEPMD_OP_PT}" | grep -q 'Shared library: \[libmkl_'; then
    readelf -d "${DEEPMD_OP_PT}" | grep 'Shared library: \[libmkl_' >&2 || true
    die "DeepMD wheel 错误链接了外部 Intel MKL，请检查构建环境"
fi

# 9. 安装和验证 Python 包
log "Step 9: 安装并验证统一 DeepMD Python 包"
python -m pip install --no-deps --force-reinstall \
    "${VESIN_WHEEL}" "${VESIN_TORCH_WHEEL}" "${DEEPMD_WHEEL}"
python -m pip check

if ! VALIDATION_OUTPUT="$(python - "${TORCH_VERSION}" "${DEEPMD_PACKAGE_VERSION}" <<'PY' 2>&1
import deepmd
import importlib.metadata
import sys
import torch
import vesin.torch
from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP

torch_expected, deepmd_expected = sys.argv[1:]
assert importlib.metadata.version("torch") == torch_expected
assert torch.__version__.startswith(torch_expected.split("+", 1)[0])
assert deepmd.__version__ == deepmd_expected
assert ENABLE_CUSTOMIZED_OP
print("DeepMD:", deepmd.__version__)
print("DeepMD path:", deepmd.__file__)
print("Torch:", torch.__version__)
print("vesin.torch: PASS")
PY
)"; then
    if grep -Eq 'libmsgpackc\.so\.2|libglog\.so\.2' <<<"${VALIDATION_OUTPUT}"; then
        echo "警告：登录节点缺少计算节点运行库，Python 包已安装，设备导入请在作业中验证"
        echo "${VALIDATION_OUTPUT}"
    else
        echo "${VALIDATION_OUTPUT}" >&2
        die "DeepMD Python 包验证失败"
    fi
else
    echo "${VALIDATION_OUTPUT}"
    dp --pt train --help >/dev/null
    dp --tf --help >/dev/null
fi

# 10. 保留旧版预编译 C++/LAMMPS 接口
log "Step 10: 安装既有 DeepMD C++/LAMMPS 包"
mkdir -p "${DP_CPP_INSTALL_DIR}"
DP_CPP_ARCHIVE="${DOWNLOAD_DIR}/dp_cpp_dcu.tar.gz"
if [[ ! -f "${DP_CPP_INSTALL_DIR}/lib/deepmd_lmp/dpplugin.so" ]]; then
    download_file "${DP_CPP_URL}" "${DP_CPP_ARCHIVE}"
    tar -xzf "${DP_CPP_ARCHIVE}" -C "${DP_CPP_INSTALL_DIR}" --strip-components=1
else
    echo "C++/LAMMPS 包已存在，跳过重复解压：${DP_CPP_INSTALL_DIR}"
fi

echo ""
echo "=========================================="
echo " DeepMD-kit DCU 统一环境安装完成"
echo " 环境：${MATCHEM_CONDA_NAME}"
echo " 源码提交：${DEEPMD_SOURCE_COMMIT}"
echo " 支持：传统 PyTorch、DPA3、DPA4 eager"
echo " C++/LAMMPS：继续使用既有预编译包"
echo "=========================================="
echo "Python 包：${CONDA_PREFIX}/lib/python3.11/site-packages/deepmd"
echo "源码目录：${DEEPMD_SRC}"
echo "C++ 接口：${DP_CPP_INSTALL_DIR}"
echo "每次使用前执行："
echo "  export MATCHEM_CONDA_NAME=${MATCHEM_CONDA_NAME}"
echo "  source ${SCRIPT_DIR}/../matchem_env.sh"
