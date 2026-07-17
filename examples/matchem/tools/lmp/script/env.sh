source ~/.bashrc

module load sghpc-mpi-gcc/26.3
export LD_LIBRARY_PATH=/public/software/sghpc_sdk.bak/Linux_x86_64/26.3/dtk/dtk-25.04.4/lib:$LD_LIBRARY_PATH

conda activate "${MATCHEM_CONDA_NAME:-test_pip}"

# 1. 使用安装目录中的 lib_override 符号链接统一 libnl 版本，避免 OpenMPI 系统 libnl 与 torch 自带 libnl 符号冲突导致段错误
# 2. 确保运行时加载与编译时一致的 torch 库，避免 ABI 不匹配
# 动态定位 torch/lib（用 find_spec 避免在登录节点 import torch 触发动态库加载）
TORCH_LIB_DIR="$(python -c "import importlib.util, os; spec = importlib.util.find_spec('torch'); print(os.path.join(os.path.dirname(spec.origin), 'lib') if spec and spec.origin else '')" 2>/dev/null || true)"
export LD_LIBRARY_PATH="${LAMMPS_INSTALL_DIR:-$HOME/lammps_dcu}/lib_override:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH}"

