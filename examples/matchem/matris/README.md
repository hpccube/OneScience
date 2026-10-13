# MatRIS 训练与推理示例

本目录包含 MatRIS 材料表示学习与结构预测模型的示例。MatRIS 已包含在 `onescience[matchem]` 基础环境中，无需额外编译安装。

## 前置条件

1. 已完成 `onescience[matchem]` 基础环境安装。
2. 已加载统一环境：
   ```bash
   # 默认激活 test_pip 环境；若使用其他 conda 环境，请先设置：
   # export MATCHEM_CONDA_NAME=your_env
   source ../matchem_env.sh
   ```

## 目录说明

```
matris/
├── data/                      # 示例数据集与数据工具
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE
│   └── pyproject.toml
├── cif_file/
│   └── demo.cif               # 示例 CIF 文件
├── test_relaxation.py         # 结构弛豫测试
└── test_modularization.py     # 模块化测试
```

## 快速开始

### 结构弛豫测试

```bash
python test_relaxation.py
```

### 模块化接口测试

```bash
python test_modularization.py
```

### 使用示例 CIF

```bash
python test_relaxation.py --cif cif_file/demo.cif
```

## 注意事项

- MatRIS 训练与推理脚本会优先从 `${ONESCIENCE_MODELS_DIR}/matris` 读取模型，从 `${ONESCIENCE_DATASETS_DIR}/matris` 读取数据。
- 默认 conda 环境为 `test_pip`，可通过 `MATCHEM_CONDA_NAME` 覆盖。
- 如需使用自有数据，请参考 `data/README.md` 中的数据格式说明。
