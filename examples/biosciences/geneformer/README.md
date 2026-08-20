# Geneformer

本目录提供 Geneformer 合入 OneScience 后的分词、预训练、细胞/基因分类微调、嵌入推理、多任务微调、超参数搜索和 in-silico perturbation 示例。代码基于官方 Hugging Face 仓库提交 `04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5`，没有添加官方仓库不存在的训练任务。

## 代码布局

| OneScience 模块 | 职责 |
| --- | --- |
| `onescience.datapipes.geneformer` | Loom/H5AD/Zarr 分词、V1/V2 基因字典、分类 collator、多任务数据加载 |
| `onescience.models.geneformer` | 官方自定义的多任务分类网络；V1/V2 基础模型直接使用 Transformers BERT |
| `onescience.metrics.geneformer` | 分类评估、混淆矩阵、ROC 和预测可视化 |
| `onescience.utils.geneformer` | 预训练器、单任务/多任务微调、嵌入提取、扰动及统计分析 |

## 环境准备

测试环境固定为 `bio_test`。进入计算节点前先在登录环境加载集群模块和 DTK CUDA 兼容层：

```bash
module load sghpc-mpi-gcc/26.3
source "${ROCM_PATH}/cuda/env.sh"
```

进入计算节点后激活环境并进入本目录：

```bash
conda activate bio_test
cd examples/biosciences/geneformer
```

示例入口会再次加载 `${ROCM_PATH}/cuda/env.sh`，但不会申请计算资源、加载 module 或修改 Python 环境。Geneformer 全精度流程所需依赖已记录在仓库根目录的 `requirements.txt` 和 `setup.py`。官方量化流程还需要 `bitsandbytes`；标准 wheel 与当前 DTK 环境不兼容，因此该可选依赖仅在 `requirements.txt` 中说明，未安装且未纳入本次验证。

## 模型与数据

脚本与其他 OneScience 模型示例一致，先加载仓库根目录的 `env.sh`，再使用以下标准资源目录：

```text
/public/share/sugonhpcapp01/onestore/onemodels/Geneformer
/public/share/sugonhpcapp01/onestore/onedatasets/Geneformer
```

可通过 `GENEFORMER_MODEL_ROOT`、`GENEFORMER_DATASET_ROOT`、`GENEFORMER_V1_MODEL`、`GENEFORMER_V1_CELL_MODEL`、`GENEFORMER_CELL_DATA`、`GENEFORMER_GENE_DATA`、`GENEFORMER_GENE_CLASSES` 和 `GENEFORMER_OUTPUT_ROOT` 覆盖默认值。

## 嵌入推理

该示例对应官方 `extract_and_plot_cell_embeddings.ipynb`，默认使用 V1 心肌病细胞分类模型和疾病分类数据：

```bash
bash scripts/inference.sh --max-cells 64 --batch-size 16
```

结果保存为 `outputs/embeddings/cardiomyopathy_cell_embeddings.csv`。也可直接调用 `extract_embeddings.py` 选择预训练模型、V2 模型、CLS/cell/gene 嵌入或其他标签列。

## 细胞分类微调

该示例对应官方 `cell_classification.ipynb`，默认按 `disease` 列微调 V1 模型，并保留独立测试集：

```bash
bash scripts/finetune.sh \
  --max-cells 1000 \
  --epochs 1 \
  --batch-size 12
```

仅验证数据准备时可增加 `--prepare-only`。官方明确建议针对具体任务进行超参数搜索；脚本提供的参数只用于演示流程，不表示通用最优配置。

官方 `Classifier.validate` 支持 Ray/HyperOpt 超参数搜索；以下参数也适用于基因分类入口：

```bash
bash scripts/finetune.sh --hyperparameter-trials 10
```

启用该可选搜索路径前需安装 `requirements.txt` 中声明的 `hyperopt`；不使用该参数时不会导入它。

## 基因分类微调

该入口对应官方 `gene_classification.ipynb`。默认使用共享目录中的 V1 dosage-sensitive transcription-factor 数据和类别字典执行分层交叉验证：

```bash
bash scripts/finetune_gene_classifier.sh \
  --max-cells 1000 \
  --cross-validation-splits 1 \
  --epochs 1
```

`--train-all-data` 对应官方在全部标注数据上训练最终模型的流程；`--gene-balance` 可用于官方支持的二分类基因平衡。仅准备 gene-labeled Dataset 时可使用 `--prepare-only`。

## 预训练

官方仓库包含 masked-language-model 预训练脚本，因此本合入保留了对应入口。默认配置与官方 V1 6 层、256 隐藏维度配置一致：

```bash
bash scripts/pretrain.sh
```

完整 Genecorpus-30M 训练开销很大。冒烟测试可限制样本和步数：

```bash
bash scripts/pretrain.sh \
  --max-cells 64 \
  --max-steps 1 \
  --batch-size 2 \
  --overwrite-output-dir
```

## 转录组分词

该入口对应官方转录组 tokenizer 示例。输入必须是未经 feature selection 的 raw counts，并提供 Ensembl ID；H5AD/Zarr 默认从 `var["ensembl_id"]` 读取，也可通过 `--use-h5ad-index` 使用 `var_names`：

```bash
source scripts/_geneformer_common.sh
python scripts/tokenize_transcriptomes.py \
  --input-dir /path/to/raw_h5ad_directory \
  --output-dir outputs/tokenized \
  --output-prefix cells_v2 \
  --file-format h5ad \
  --model-version V2 \
  --metadata cell_type=cell_type
```

## In-silico perturbation

该流程对应官方 `in_silico_perturbation.ipynb`。不指定基因时会逐细胞测试所有检测到的基因，开销可能很大；建议先指定少量 Ensembl ID：

```bash
bash scripts/perturb.sh \
  --max-cells 2 \
  --gene ENSG00000141510
```

原始扰动结果可继续交给 `onescience.utils.geneformer.InSilicoPerturberStats` 执行官方支持的 goal-state、null distribution、mixture model 或聚合统计。

## 多任务分类微调

官方仓库还提供单机和分布式多任务分类。输入必须是已经切分的 tokenized Dataset，并包含 `unique_cell_id` 以及每个任务的标签列：

```bash
python scripts/multitask_finetune.py \
  --model-dir "${GENEFORMER_V1_MODEL}" \
  --train-data /path/to/train.dataset \
  --validation-data /path/to/validation.dataset \
  --task-column disease \
  --task-column cell_type
```

共享数据中没有满足这些字段要求的官方多任务样例切分，因此没有设置可能误导用户的默认 Bash 数据入口。

## 来源与引用

- 官方代码与模型：<https://huggingface.co/ctheodoris/Geneformer>
- 官方数据：<https://huggingface.co/datasets/ctheodoris/Genecorpus-30M>
- Theodoris et al., *Nature* (2023), “Transfer learning enables predictions in network biology.”

Geneformer 在官方仓库中声明为 Apache-2.0；许可证文本随代码保存在 `src/onescience/models/geneformer/LICENSE.geneformer`。
