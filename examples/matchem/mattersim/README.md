# MatterSim

本目录提供 MatterSim v1.2.3 在 OneScience 中的集成示例。

## 模型简介

MatterSim 是面向材料体系的通用机器学习原子间势模型。它根据原子种类和三维结构预测体系能量、原子力和应力，可替代大量昂贵的第一性原理计算，用于单点计算、批量推理、结构弛豫、分子动力学和下游数据微调。OneScience 中的 MatterSim 示例默认使用预训练 checkpoint，也支持使用用户数据继续训练。

## 环境准备

本示例假设已完成 OneScience `matchem` 安装。若尚未安装，请参考仓库根目录的 `install.sh`：

```bash
bash install.sh matchem
```

### 加载 MatChem 环境变量

所有示例默认通过 `ONESCIENCE_MODELS_DIR` 和 `ONESCIENCE_DATASETS_DIR` 环境变量查找模型和数据。从仓库根目录执行：

```bash
source examples/matchem/matchem_env.sh
```

加载后默认路径为：

- 模型：`$ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth`
- 数据：`$ONESCIENCE_DATASETS_DIR/matchem/mattersim/high_level_water.xyz`

如需覆盖，可通过 `--checkpoint` 或 `--train-data-path` 传入自定义路径。

## 示例运行

先进入本目录：

```bash
cd examples/matchem/mattersim
```

所有示例默认使用 `cuda`，也可通过 `--device cpu` 切换。

### 单点推理

```bash
python single_point.py
```

> 默认使用 `$ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth`，设备默认为 `cuda`。

### 批量推理

```bash
python batch_inference.py
```

> 同样默认使用 `$ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth`。

### 结构弛豫

```bash
python relax.py
```

> 依赖 `$ONESCIENCE_MODELS_DIR` 中的默认 checkpoint。

### 分子动力学

```bash
python md.py
```

> 同样依赖 `$ONESCIENCE_MODELS_DIR` 中的默认 checkpoint。

## 微调

### 方式一：YAML 配置文件（推荐）

直接修改 `finetune_config.yaml` 中的路径和参数（例如 `train_data_path`、`checkpoint` 等）：

```bash
# 编辑 finetune_config.yaml 中的 train_data_path、checkpoint 等字段
```

运行：

```bash
python finetune.py --config finetune_config.yaml
```

命令行参数可以覆盖 YAML 中的值，例如临时改为 2 个 epoch：

```bash
python finetune.py --config finetune_config.yaml --epochs 2
```

### 方式二：命令行参数

```bash
python finetune.py \
  --train-data-path "$ONESCIENCE_DATASETS_DIR/matchem/mattersim/high_level_water.xyz" \
  --checkpoint "$ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth" \
  --epochs 2 --batch-size 16
```

### 多卡 DDP

使用 `torchrun` 启动，参数与单卡完全一致：

```bash
torchrun --nproc_per_node=4 finetune.py \
  --config finetune_config.yaml
```

## Python API

也可以在代码中直接调用：

```python
from onescience.utils.mattersim import FineTuneConfig, MatterSimTrainer

trainer = MatterSimTrainer(FineTuneConfig(
    train_data_path="/path/to/high_level_water.xyz",
    checkpoint="/path/to/mattersim-v1.0.0-1M.pth",
    save_path="results/mattersim",
    epochs=2,
    batch_size=16,
))
result = trainer.fit()
```

加载微调后的 checkpoint 继续推理：

```bash
python single_point.py --checkpoint results/mattersim/last_model.pth
```

