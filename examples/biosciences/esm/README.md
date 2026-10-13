# ESM

本示例将 ESM 蛋白质语言模型集成到 OneScience，提供蛋白质序列表征提取、ESMFold 结构预测、ESM-1v 零样本变异评分、ESM-IF1 反向折叠以及相关 notebook 示例入口。

## 简介

ESM（Evolutionary Scale Modeling）是 Meta Fundamental AI Research Protein Team 发布的 Transformer 蛋白质语言模型系列。当前目录保留并适配了 ESM-2、ESMFold、MSA Transformer、ESM-1v 和 ESM-IF1 等上游能力。

* **ESM-2**：通用蛋白质语言模型，可用于序列表征、接触预测和结构/功能相关任务。
* **ESMFold**：基于 ESM-2 的端到端单序列蛋白质结构预测模型。
* **ESM-1v**：面向零样本突变效应预测的蛋白质语言模型。
* **ESM-IF1**：反向折叠模型，可根据给定骨架结构采样或评分蛋白质序列。
* **MSA Transformer**：基于多序列比对的蛋白质语言模型，可用于接触预测等任务。

当前目录主要提供**预训练 ESM 模型的推理、表征提取、结构预测、变异评分、反向折叠和下游分析示例**，不包含 ESM-2、ESMFold、ESM-1v 或 ESM-IF1 的完整预训练流程。

上游项目背景、模型列表、ESM Atlas、notebook 和完整引用信息见 [UPSTREAM_README.md](UPSTREAM_README.md)。

## 目录

* [目录结构](#目录结构)
* [环境准备](#环境准备)
* [数据与模型权重](#数据与模型权重)
* [功能与入口](#功能与入口)
* [运行随附推理示例](#运行随附推理示例)
* [ESM-2 表征提取](#esm-2-表征提取)
* [ESMFold 结构预测](#esmfold-结构预测)
* [零样本变异评分](#零样本变异评分)
* [反向折叠](#反向折叠)
* [Notebooks 与上游资料](#notebooks-与上游资料)
* [输出与复现](#输出与复现)
* [运行约束](#运行约束)
* [Issues](#issues)
* [许可证与引用](#许可证与引用)

## 目录结构

```text
examples/biosciences/esm/
├── atlas/                                      # ESM Atlas 批量下载说明和文件列表
├── data/                                       # FASTA、MSA 和 notebook 示例数据
│   ├── few_proteins.fasta                      # infer.sh 使用的小型 FASTA 示例
│   ├── some_proteins.fasta                     # 表征提取示例 FASTA
│   ├── P62593.fasta                            # 有监督变异预测 notebook 示例数据
│   └── *.a3m                                   # MSA Transformer / 接触预测示例 MSA
├── inverse_folding/                            # ESM-IF1 反向折叠示例
│   ├── sample_sequences.py                     # 给定结构采样序列
│   ├── score_log_likelihoods.py                # 给定结构和序列计算条件 log-likelihood
│   └── data/                                   # 反向折叠示例 PDB/FASTA
├── lm-design/                                  # 上游 ESM2 蛋白质设计示例
├── protein-programming-language/               # 上游生成式蛋白质设计语言示例
├── scripts/
│   ├── extract.py                              # 从 FASTA 批量提取 ESM embedding
│   ├── fold.py                                 # ESMFold FASTA 批量结构预测
│   └── download_weights.sh                     # ESM 相关模型权重下载脚本
├── variant-prediction/
│   ├── predict.py                              # ESM-1v/ESM-2 零样本变异评分入口
│   └── data/                                   # Deep mutational scanning 示例数据
├── contact_prediction.ipynb                    # 无监督接触预测 notebook
├── esm_structural_dataset.ipynb                # ESM Structural Split Dataset notebook
├── sup_variant_prediction.ipynb                # 有监督变异预测 notebook
├── esm2_infer_fairscale_fsdp_cpu_offloading.py # ESM-2 15B FSDP CPU offloading 示例
├── infer.sh                                    # 表征提取、结构预测和变异评分串联示例
├── README.md                                   # OneScience ESM 使用说明
└── UPSTREAM_README.md                          # 上游 ESM README
```

## 环境准备

在仓库根目录执行以下命令：

```bash
conda create -n onescience-esm python=3.11 -y
conda activate onescience-esm

bash install.sh bio
source env.sh
```

当前示例使用 OneScience 内的 ESM 适配代码。新代码中应使用以下导入方式：

```python
import onescience.models.esm as esm
from onescience.models.esm import pretrained
```

若从仓库根目录直接运行 Python 脚本，应确保 `src` 已加入 `PYTHONPATH`；通常执行：

```bash
source env.sh
```

即可完成相关环境设置。

以下环境变量用于解析默认数据和模型路径：

```bash
export ONESCIENCE_DATASETS_DIR=/path/to/datasets
export ONESCIENCE_MODELS_DIR=/path/to/models
```

本文档后续所有相对路径命令均假定当前工作目录为：

```text
examples/biosciences/esm
```

因此建议首先执行：

```bash
cd examples/biosciences/esm
```

`infer.sh` 内部会执行：

```bash
source ../../../env.sh
```

因此也应从 `examples/biosciences/esm` 目录运行该脚本。

## 数据与模型权重

本目录已包含可用于功能验证的小型 FASTA、MSA、PDB 和 DMS 示例数据。

完整数据集、ESM Atlas 资源以及上游 notebook 数据说明见：

[UPSTREAM_README.md](UPSTREAM_README.md)

模型权重建议统一存放在：

```bash
$ONESCIENCE_MODELS_DIR/esm_models/
```

推荐按照以下结构放置常用模型权重：

```text
$ONESCIENCE_MODELS_DIR/esm_models/
├── esm2_t6_8M_UR50D.pt
├── esm2_t33_650M_UR50D.pt
├── esm2_t33_650M_UR50D-contact-regression.pt
├── esm1v_t33_650M_UR90S_1.pt
├── esm_if1_gvp4_t16_142M_UR50.pt
└── checkpoints/
    └── esmfold_3B_v1.pt
```

通过本地 `.pt` 文件加载 ESM-1、ESM-2 等带接触预测头的模型时，对应的：

```text
*-contact-regression.pt
```

权重应与主权重位于同一目录。

例如：

```text
esm_models/
├── esm2_t33_650M_UR50D.pt
└── esm2_t33_650M_UR50D-contact-regression.pt
```

`scripts/fold.py --model-dir` 会将指定目录传递给 `torch.hub.set_dir()`，因此 ESMFold 权重需要位于：

```text
<model-dir>/checkpoints/
```

例如：

```text
$ONESCIENCE_MODELS_DIR/esm_models/checkpoints/esmfold_3B_v1.pt
```

> **ESM-IF1 权重说明**
>
> ESM-IF1 通过 `esm.pretrained.esm_if1_gvp4_t16_142M_UR50()` 加载模型，当前使用 PyTorch Hub 缓存机制。离线环境下，可将统一保存的权重软链接到 PyTorch Hub 缓存目录：
>
> ```bash
> HUB_DIR=$(python -c "import torch; print(torch.hub.get_dir())")
> mkdir -p "$HUB_DIR/checkpoints"
>
> ln -sf \
>   "$ONESCIENCE_MODELS_DIR/esm_models/esm_if1_gvp4_t16_142M_UR50.pt" \
>   "$HUB_DIR/checkpoints/esm_if1_gvp4_t16_142M_UR50.pt"
> ```

仓库同时提供：

```text
scripts/download_weights.sh
```

用于辅助准备模型权重。

由于不同模型的加载路径存在差异，使用该脚本后建议确认实际下载位置与上述目录布局一致。本文档中的运行示例以上述目录结构为准。

## 功能与入口

| 任务                | 推荐入口                                          | 输入                                  | 输出                        |
| ----------------- | --------------------------------------------- | ----------------------------------- | ------------------------- |
| 串联推理验证            | `infer.sh`                                    | `data/few_proteins.fasta`、本地 ESM 权重 | embedding、PDB、变异评分 CSV    |
| ESM-2 表征提取        | `scripts/extract.py`                          | 模型名或 `.pt` 权重、FASTA                 | 每条序列一个 `.pt` 表征文件         |
| ESMFold 结构预测      | `scripts/fold.py`                             | FASTA、ESMFold 权重目录                  | 每条序列一个 PDB 文件             |
| 零样本变异评分           | `variant-prediction/predict.py`               | ESM-1v/ESM-2 权重、DMS CSV、野生型序列       | 带评分列的 CSV                 |
| 反向折叠采样            | `inverse_folding/sample_sequences.py`         | PDB/mmCIF 结构                        | 采样序列 FASTA                |
| 反向折叠评分            | `inverse_folding/score_log_likelihoods.py`    | PDB/mmCIF 结构、候选序列 FASTA             | 条件 log-likelihood CSV     |
| 大模型 offloading 示例 | `esm2_infer_fairscale_fsdp_cpu_offloading.py` | ESM-2 15B 权重                        | 单卡 FSDP CPU offloading 推理 |
| Notebook 示例       | `*.ipynb`                                     | 示例 FASTA、MSA、embedding 或结构数据        | 交互式分析结果                   |

## 运行随附推理示例

`infer.sh` 依次演示：

1. ESM-2 表征提取；
2. ESMFold 结构预测；
3. ESM-1v 零样本变异评分。

> **注意**
>
> 当前 `infer.sh` 中零样本变异评分步骤的：
>
> ```text
> --sequence WTSEQUENCE_HERE
> ```
>
> 是占位符。
>
> 在完整执行脚本前，应将其替换为与示例 DMS 数据突变编号对应的野生型蛋白质序列。
>
> 如果只希望验证 ESM-2 表征提取或 ESMFold，可直接运行后续章节对应的独立命令。

进入 ESM 示例目录：

```bash
cd examples/biosciences/esm
```

完成占位符修改后运行：

```bash
bash infer.sh
```

脚本内部依次调用：

```bash
python ./scripts/extract.py \
  "$ONESCIENCE_MODELS_DIR/esm_models/esm2_t6_8M_UR50D.pt" \
  ./data/few_proteins.fasta \
  /tmp/esm_extract_out \
  --include mean per_tok \
  --repr_layers 6
```

```bash
python ./scripts/fold.py \
  -i ./data/few_proteins.fasta \
  -o /tmp/esmfold_pdb_out \
  --model-dir "$ONESCIENCE_MODELS_DIR/esm_models/"
```

```bash
python ./variant-prediction/predict.py \
  --model-location "$ONESCIENCE_MODELS_DIR/esm_models/esm1v_t33_650M_UR90S_1.pt" \
  --sequence WTSEQUENCE_HERE \
  --dms-input ./variant-prediction/data/BLAT_ECOLX_Ranganathan2015.csv \
  --mutation-col mutant \
  --dms-output /tmp/esm_variant_prediction.csv \
  --offset-idx 24 \
  --scoring-strategy wt-marginals
```

变异评分中的野生型序列、DMS 数据中的突变编号以及 `--offset-idx` 必须保持一致，否则得到的评分不具有正确的生物学含义。

## ESM-2 表征提取

`scripts/extract.py` 可从 FASTA 文件中批量提取逐 token、均值、BOS 或接触预测相关输出。

例如：

```bash
cd examples/biosciences/esm

mkdir -p outputs

python scripts/extract.py \
  "$ONESCIENCE_MODELS_DIR/esm_models/esm2_t33_650M_UR50D.pt" \
  data/some_proteins.fasta \
  ./outputs/some_proteins_emb_esm2 \
  --repr_layers 0 32 33 \
  --include mean per_tok
```

主要参数如下：

| 参数                        |    默认值 | 说明                                   |
| ------------------------- | -----: | ------------------------------------ |
| `model_location`          |     必填 | 模型名称或本地 `.pt` 权重路径                   |
| `fasta_file`              |     必填 | 输入 FASTA 文件                          |
| `output_dir`              |     必填 | 输出 `.pt` 表征文件目录                      |
| `--repr_layers`           |   `-1` | 提取哪些层的表征，范围为 `0` 到模型层数               |
| `--include`               |     必填 | 可选 `mean`、`per_tok`、`bos`、`contacts` |
| `--toks_per_batch`        | `4096` | 每批最大 token 数                         |
| `--truncation_seq_length` | `1022` | 超长序列截断长度                             |
| `--nogpu`                 |     关闭 | 请求不使用 GPU                            |

输出目录中，每条 FASTA 记录对应一个 `.pt` 文件，可使用：

```python
import torch

result = torch.load("example.pt")
```

读取。

如果在：

```bash
--include
```

中启用：

```text
contacts
```

需要确认对应的 `*-contact-regression.pt` 权重已经放在主模型权重同一目录。

## ESMFold 结构预测

`scripts/fold.py` 使用 `esmfold_v1` 对 FASTA 文件中的蛋白质序列进行结构预测。

建议首先确保输出父目录存在：

```bash
cd examples/biosciences/esm

mkdir -p outputs
```

然后运行：

```bash
python scripts/fold.py \
  -i data/few_proteins.fasta \
  -o ./outputs/esmfold_pdb \
  --model-dir "$ONESCIENCE_MODELS_DIR/esm_models/" \
  --chunk-size 128
```

主要参数如下：

| 参数                       |    默认值 | 说明                                |
| ------------------------ | -----: | --------------------------------- |
| `-i/--fasta`             |     必填 | 输入 FASTA 文件                       |
| `-o/--pdb`               |     必填 | 输出 PDB 目录                         |
| `-m/--model-dir`         | `None` | torch hub 模型缓存父目录                 |
| `--num-recycles`         |  训练默认值 | 结构 refinement recycle 次数          |
| `--max-tokens-per-batch` | `1024` | 每次前向传播允许的最大序列长度总和                 |
| `--chunk-size`           | `None` | 轴向注意力 chunk 大小，常用 `128`、`64`、`32` |
| `--cpu-only`             |     关闭 | 完全在 CPU 上运行                       |
| `--cpu-offload`          |     关闭 | GPU 推理时将部分模型参数 offload 到 CPU      |

输出目录中会为每条 FASTA 序列写出：

```text
{header}.pdb
```

日志中通常包含：

* 序列长度；
* 平均 `pLDDT`；
* `pTM`；
* 推理耗时；
* 整体执行进度。

若 FASTA 记录包含多条链，可在同一序列中使用：

```text
:
```

分隔不同链。

例如：

```text
MKT...AAA:GLY...VVV
```

## 零样本变异评分

`variant-prediction/predict.py` 可使用 ESM-1v 或 ESM-2 对突变进行零样本打分。

示例：

```bash
cd examples/biosciences/esm

mkdir -p outputs

python variant-prediction/predict.py \
  --model-location "$ONESCIENCE_MODELS_DIR/esm_models/esm1v_t33_650M_UR90S_1.pt" \
  --sequence HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW \
  --dms-input variant-prediction/data/BLAT_ECOLX_Ranganathan2015.csv \
  --mutation-col mutant \
  --dms-output ./outputs/esm_variant_prediction.csv \
  --offset-idx 24 \
  --scoring-strategy wt-marginals
```

主要参数如下：

| 参数                   | 说明                                                          |
| -------------------- | ----------------------------------------------------------- |
| `--model-location`   | 一个或多个模型名称或本地权重路径                                            |
| `--sequence`         | 野生型蛋白质序列字符串                                                 |
| `--dms-input`        | 包含突变信息的 CSV                                                 |
| `--mutation-col`     | DMS 文件中的突变列名                                                |
| `--dms-output`       | 输出 CSV 路径                                                   |
| `--offset-idx`       | DMS 突变编号与序列索引之间的偏移                                          |
| `--scoring-strategy` | `wt-marginals`、`masked-marginals` 或 `pseudo-ppl`            |
| `--nogpu`            | 请求使用 CPU 模式； |

突变编号、野生型序列和：

```text
--offset-idx
```

必须保持一致，否则评分结果会失去正确的生物学含义。

示例数据来自 ESM upstream 的 DMS 任务，更多背景见：

[variant-prediction/README.md](variant-prediction/README.md)

## 反向折叠

ESM-IF1 可根据给定蛋白质骨架结构采样序列，也可以为给定结构下的候选序列计算条件 log-likelihood。

### 给定结构采样序列

```bash
cd examples/biosciences/esm

mkdir -p outputs

python inverse_folding/sample_sequences.py \
  inverse_folding/data/5YH2.pdb \
  --chain C \
  --temperature 1 \
  --num-samples 3 \
  --outpath ./outputs/sampled_sequences.fasta
```

### 为候选序列评分

```bash
python inverse_folding/score_log_likelihoods.py \
  inverse_folding/data/5YH2.pdb \
  inverse_folding/data/5YH2_mutated_seqs.fasta \
  --chain C \
  --outpath ./outputs/5YH2_mutated_seqs_scores.csv
```

主要参数如下：

| 参数              | 说明                                                                 |
| --------------- | ------------------------------------------------------------------ |
| `pdbfile`       | 输入 PDB 或 mmCIF 结构文件                                                |
| `seqfile`       | 待评分序列 FASTA，仅评分脚本需要                                                |
| `--chain`       | 需要设计或评分的链 ID                                                       |
| `--temperature` | 采样温度，越高通常意味着更高的序列多样性                                               |
| `--num-samples` | 采样序列数量                                                             |
| `--outpath`     | 输出 FASTA 或 CSV 路径                                                  |
| `--nogpu`       | 请求使用 CPU 模式； |

ESM-IF1 示例通过：

```python
esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
```

加载模型，因此反向折叠脚本本身不提供：

```text
--model-location
```

形式的本地模型路径参数。

反向折叠的详细背景和使用说明见：

[inverse_folding/README.md](inverse_folding/README.md)

## Notebooks 与上游资料

本目录保留了多个 ESM upstream 的交互式示例。

### 有监督变异预测

```text
sup_variant_prediction.ipynb
```

基于 ESM embedding 进行有监督变异预测。

### 无监督接触预测

```text
contact_prediction.ipynb
```

基于 ESM-2 和 MSA Transformer 进行无监督接触预测。

### ESM Structural Split Dataset

```text
esm_structural_dataset.ipynb
```

加载 ESM Structural Split Dataset，并进行相关结构和接触预测实验。

### ESM Atlas

```text
atlas/README.md
```

提供 ESM Metagenomic Atlas 数据和批量下载相关说明。

这些 notebook 大多保留了 ESM upstream 的原始示例语境。

如果在 OneScience 环境中改写或复用 notebook，应优先将 ESM 导入路径调整为：

```python
import onescience.models.esm as esm
```

以及在需要数据组件时使用：

```python
onescience.datapipes.esm
```

避免额外安装的上游 `fair-esm` 覆盖 OneScience 当前实现。

## 运行约束

* ESMFold、较大的 ESM-2 模型以及 ESM-IF1 对 GPU 显存要求较高。
* 对长序列或大模型推理，建议降低 batch 或 token 数量。
* ESMFold 显存不足时可设置 `--chunk-size 128/64/32`。
* ESMFold 也可根据硬件条件尝试 `--cpu-offload`。
* `scripts/fold.py --model-dir` 指向的是 torch hub 的父目录，ESMFold 权重需要位于其 `checkpoints/` 子目录。
* 通过本地 `.pt` 加载 ESM-1/ESM-2 并请求 `contacts` 输出时，需要保证对应 contact-regression 权重与主权重位于同一目录。
* `infer.sh` 中的零样本变异评分命令包含 `WTSEQUENCE_HERE` 占位符，完整执行前必须替换。
* `variant-prediction/predict.py` 和部分 ESM-IF1 上游迁移代码虽然保留 `--nogpu` 参数，但当前实现中仍可能存在显式 CUDA 调用；需要 CPU-only 运行时应先确认对应代码路径。
* `esm2_infer_fairscale_fsdp_cpu_offloading.py` 依赖 Fairscale/FSDP 和分布式初始化，执行前需要确认 GPU、通信端口和相关依赖配置。
* notebook 中部分上游资源链接需要网络访问，离线环境应提前准备数据和模型权重。
* 当前目录主要用于预训练模型推理和下游分析，不提供 ESM 系列模型的完整预训练流程。
* 模型输出不能替代实验验证；涉及蛋白质结构、功能或突变效应结论时，应进行独立复核。

## Issues

### 1. `scripts/extract.py` 提示缺少 `*-contact-regression.pt`

如果请求：

```bash
--include contacts
```

需要将对应的 contact regression 权重放在主模型 `.pt` 文件同一目录。

例如：

```text
esm2_t33_650M_UR50D.pt
esm2_t33_650M_UR50D-contact-regression.pt
```

如果不需要接触预测，则不要请求：

```text
contacts
```

输出。

### 2. `scripts/fold.py` 找不到 `esmfold_3B_v1.pt`

检查：

```bash
--model-dir "$ONESCIENCE_MODELS_DIR/esm_models/"
```

是否正确，并确认权重位于：

```text
$ONESCIENCE_MODELS_DIR/esm_models/checkpoints/esmfold_3B_v1.pt
```

### 3. ESMFold 显存不足

可依次尝试：

```bash
--chunk-size 128
```

或：

```bash
--chunk-size 64
```

或：

```bash
--chunk-size 32
```

同时可以降低：

```text
--max-tokens-per-batch
```

必要时尝试：

```bash
--cpu-offload
```

### 4. 变异评分结果异常

重点检查：

* 野生型序列是否正确；
* DMS CSV 的突变列格式是否正确；
* `--mutation-col` 是否指向正确列；
* `--offset-idx` 是否与数据编号方式一致；
* 模型权重是否加载正确。

### 5. `--nogpu` 后仍出现 CUDA 相关错误

当前部分上游迁移脚本仍可能包含：

```python
.cuda()
```

或显式：

```python
device=torch.device("cuda")
```

调用。

因此 `--nogpu` 当前不能视为所有任务的完整 CPU-only 保证。

若需要纯 CPU 运行，应检查对应脚本中的 device 设置，确保模型和输入 tensor 使用统一的 CPU device。

### 6. 反向折叠无法读取结构

确认：

* 输入为有效的 `.pdb` 或 `.cif` 文件；
* 指定的链 ID 在结构文件中真实存在；
* 输入结构包含模型需要的主链坐标。

### 7. ESMFold 输出目录创建失败

建议提前创建父目录：

```bash
mkdir -p outputs
```

然后再指定：

```text
./outputs/esmfold_pdb
```

作为输出目录。

### 8. 导入 `esm` 失败

在 OneScience 中推荐使用：

```python
import onescience.models.esm as esm
```

并确认已经在仓库根目录执行：

```bash
source env.sh
```

不要在同一环境中额外安装并优先加载其他版本的 `fair-esm`。

### 9. 使用 `download_weights.sh` 后找不到模型

检查脚本实际下载目录，并确认模型文件最终位置与运行命令使用的路径一致。

本文档运行示例默认：

```text
ESM-2 / ESM-1v:
$ONESCIENCE_MODELS_DIR/esm_models/*.pt

ESMFold:
$ONESCIENCE_MODELS_DIR/esm_models/checkpoints/*.pt
```

## 许可证与引用

ESM upstream 代码采用 MIT License。

使用本示例开展研究或开发工作时，应同时遵循：

* ESM upstream 许可证；
* 对应模型权重许可证；
* 所使用数据集的许可证；
* OneScience 项目许可证。

主要相关工作包括：

* Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model*, Science, 2023.
* Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*, PNAS, 2021.
* Meier et al., *Language models enable zero-shot prediction of the effects of mutations on protein function*, 2021.
* Hsu et al., *Learning inverse folding from millions of predicted structures*, 2022.

完整项目背景、BibTeX、模型列表和数据说明见：

[UPSTREAM_README.md](UPSTREAM_README.md)
