<p align="center"><strong><span style="font-size: 30px;">DNABERT-2</span></strong></p>

# 模型介绍

DNABERT-2 是面向多物种基因组理解的 DNA 基础模型，提供 DNA 序列 embedding、监督分类和 LoRA 微调。

论文：<https://arxiv.org/abs/2306.15006>；官方仓库：<https://github.com/MAGICS-LAB/DNABERT_2>。

# 模型描述

DNABERT-2 使用 Hugging Face PyTorch 架构、BPE tokenization 和 ALiBi 长序列注意力；本地 HF 架构闭包保持不拆分，以确保 `auto_map`、权重键和自定义代码加载不变。

# 适用场景

| 场景 | 说明 |
| :--- | :--- |
| DNA embedding | 对本地 FASTA 生成 mean/max/CLS pooled embedding |
| 全参数监督训练 | 从本地 DNABERT-2 checkpoint 训练分类器 |
| LoRA 微调 | 使用本地基础模型和 PEFT adapter |
| checkpoint 评估 | 输出 metrics JSON 和 predictions CSV |
| GUE benchmark | 按配置运行 EMP、promoter、splice、virus、mouse 和 TF 任务 |

# 使用说明

本模型已接入 OneScience。运行前准备本地 `DNABERT-2` 模型资产和 `DNABERT-2_dataset` 数据资产，并在已初始化的 OneScience 环境中调用 shell 入口。

# 快速开始

以下命令从 `examples/biosciences/dnabert2` 执行。输入样例位于 `inputs/`，运行配置位于 `configs/`。

### 1. `embed.sh`

生成 DNA pooled embedding。

```bash
bash scripts/embed.sh
```

### 2. `train.sh`

运行全参数监督训练。

```bash
bash scripts/train.sh --num-train-epochs 1
```

### 3. `finetune.sh`

运行 LoRA 参数高效微调。

```bash
bash scripts/finetune.sh --num-train-epochs 1
```

### 4. `evaluate.sh`

评估本地全参数 checkpoint 或 LoRA adapter。

```bash
bash scripts/evaluate.sh
```

### 5. `run_gue.sh`

运行 GUE 配置矩阵

```bash
bash scripts/run_gue.sh --group EMP
```

# 注意事项

- 模型默认根目录为 `${ONESCIENCE_MODELS_DIR}/DNABERT-2`，数据默认根目录为 `${ONESCIENCE_DATASETS_DIR}/DNABERT-2_dataset`。
- HF 自定义代码仅从本地 architecture bundle 加载，默认 `local_files_only=True`；缺失资产会直接报本地路径，不会联网补取。
- `config.json` 的 `auto_map` 与 `bert_layers.py` 的相对导入构成原生架构闭包，本次没有移动或改写其中的 forward、attention、state dict 键。
- GUE 和监督训练输入为 `sequence,label` CSV；embedding 输入为 FASTA。
- Triton 不兼容时使用数学等价的 PyTorch attention fallback，保留 QKV、ALiBi、mask 和梯度语义，但显存和速度可能不同。
- 本次整理未修改 OneScience requirements；官方旧版本与当前 Transformers/PEFT/PyTorch 约束的差异以依赖审计报告为准。
- 官方下载地址只在交付报告中记录，README 不包含模型包获取命令。

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- DNABERT-2 论文：<https://arxiv.org/abs/2306.15006>。
- DNABERT-2 官方仓库：<https://github.com/MAGICS-LAB/DNABERT_2>。
- 预训练权重及 GUE 数据的许可证和访问条款以官方仓库、模型页和数据发布页为准。
- 本仓库新增 OneScience 适配代码遵循 OneScience 根目录 LICENSE；bundled HF architecture 文件保留其 Apache-2.0 SPDX 标识。
