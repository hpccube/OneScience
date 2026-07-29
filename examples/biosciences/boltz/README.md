<p align="center"><strong><span style="font-size: 30px;">Boltz</span></strong></p>

# 模型介绍

Boltz 是面向蛋白质、核酸和配体复合物结构预测的模型家族；Boltz-2 还提供结合亲和力预测。

官方项目：<https://github.com/jwohlwend/boltz>。

# 模型描述

Boltz 使用 PyTorch 扩散结构模块、MSA/Pairformer 表示和本地结构数据管线，输出结构坐标、置信度以及 Boltz-2 亲和力结果。

# 适用场景

| 场景 | 说明 |
| :--- | :--- |
| 结构预测 | 对本地 FASTA/YAML 输入生成 PDB 或 mmCIF |
| 亲和力预测 | 对含 affinity 属性的 Boltz-2 输入预测 binder 概率和相对亲和力 |
| pocket 条件 | 使用本地 pocket YAML 进行条件结构预测 |
| 训练 | 使用本地 manifest 和配置训练 Boltz-1 |
| 微调 | 从本地 Lightning checkpoint 继续训练 |
| 评估 | 运行 OpenStructure、聚合评估或物理几何指标 |

# 使用说明

本模型已接入 OneScience。运行前准备本地 `Boltz` 模型资产和 `Boltz_dataset` 数据资产，并在已初始化的 OneScience 环境中调用 shell 入口。

# 快速开始

以下命令从 `examples/biosciences/boltz` 执行。输入样例位于 `inputs/`，训练配置位于 `configs/train/`。

### 1. `predict.sh`

运行 Boltz-1/2 结构或亲和力预测。

```bash
bash scripts/predict.sh inputs/prot_no_msa.yaml
```

### 2. `train.sh`

按本地 Hydra 配置训练。

```bash
bash scripts/train.sh trainer.devices=1 trainer.max_epochs=1
```

### 3. `finetune.sh`

从本地 checkpoint 继续训练。

```bash
bash scripts/finetune.sh trainer.devices=1 trainer.max_epochs=1
```

### 4. `evaluate.sh`

运行 OpenStructure 评估流程。

```bash
bash scripts/evaluate.sh
```

### 5. `aggregate_evaluations.sh`

聚合本地评估结果。

```bash
bash scripts/aggregate_evaluations.sh
```

### 6. `physical_metrics.sh`

计算结构物理几何指标。

```bash
bash scripts/physical_metrics.sh --workers 1 --num_samples 1
```

# 注意事项

- 模型默认根目录为 `${ONESCIENCE_MODELS_DIR}/Boltz`，数据默认根目录为 `${ONESCIENCE_DATASETS_DIR}/Boltz_dataset`，输出由 `BOLTZ_OUTPUT_DIR` 控制。
- Boltz-1 需要本地 CCD/checkpoint；Boltz-2 需要本地 molecule reference、CCD 和对应 checkpoint。
- `--use_msa_server` 是显式在线 MSA 业务功能；不启用时必须提供本地 MSA 或允许空 MSA 的输入。
- 预测默认使用标准 PyTorch 实现；仅在已准备好 cuEquivariance 加速环境时显式传入 `--use_kernels`。
- OpenStructure 评估需要外部 Docker 和 benchmark 数据；物理指标需要本地预测目录。
- `block` 文件中的 Boltz trunk/atom 复合体保持官方调用闭包，能力导出通过现有 attention、msa、pairformer、diffusion、encoder/decoder 模块提供；未改动 forward、参数名和 checkpoint 键。
- 本次整理未修改 OneScience requirements；PyTorch Lightning、SciPy、scikit-learn 等版本差异需按依赖审计报告处理。
- 官方下载地址只在交付报告中记录，README 不包含模型包获取命令。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- Boltz 官方项目：<https://github.com/jwohlwend/boltz>。
- Boltz 论文、代码、权重和第三方组件的许可分别以对应官方发布版本中的 LICENSE 和模型卡为准。
- 本仓库新增 OneScience 适配代码遵循 OneScience 根目录 LICENSE。
