# NEP 训练示例

本目录包含基于 MatPL 的 NEP（Neural Network Potential）训练示例。NEP 训练需要额外编译安装 MatPL DCU 原生算子。

## 前置条件

1. 已完成 `onescience[matchem]` 基础环境安装。
2. 已加载统一环境：
   ```bash
   # 默认激活 test_pip 环境；若使用其他 conda 环境，请先设置：
   # export MATCHEM_CONDA_NAME=your_env
   source ../matchem_env.sh
   ```
3. **首次使用需安装 MatPL DCU 版**：
   ```bash
   bash matpl_install.sh
   ```
   - 脚本默认从 gitee 拉取 MatPL DCU 源码到 `matpl_dcu/`。
   - 可通过 `MATPL_SRC_DIR` 指定已有源码路径。

## 目录说明

```
nep/
├── matpl_install.sh           # MatPL DCU 安装脚本
└── demo/
    ├── nep_Cu/                # Cu 元素 NEP 训练示例
    │   ├── Cu_nep_train.json
    │   ├── std_input.json
    │   └── submit.sh
    └── nep_LiSiC/             # Li-Si-C 体系 NEP 训练示例
        ├── LiSiC_nep_train.json
        ├── std_input.json
        └── submit.sh
```

## 快速开始

```bash
cd demo/nep_Cu
sbatch submit.sh
```

`submit.sh` 会 inline 维护 MatPL 运行时环境，自动加载 DTK、MPI 与 conda 环境。

## 注意事项

- NEP 算子为 DCU 原生实现，必须使用 DCU 节点运行。
- 所有 `submit.sh` 已通过 `matchem_env.sh` 加载环境。
- 默认 conda 环境为 `test_pip`，可通过 `MATCHEM_CONDA_NAME` 覆盖。
- 队列名 `#SBATCH --partition=hx1hdexclu12` 为本集群配置，迁移时需修改。
