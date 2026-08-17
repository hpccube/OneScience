# scGPT

本目录提供 scGPT 合入 OneScience 后的细胞嵌入推理和细胞类型微调示例。相关实现按照职责拆分到以下模块：

- `onescience.models.scgpt`：模型结构与权重加载。
- `onescience.datapipes.scgpt`：AnnData 预处理、分词、数据整理与采样。
- `onescience.metrics.scgpt`：损失函数与评估指标。
- `onescience.utils.scgpt`：训练、嵌入、日志与其他工具。

## 环境准备

请先申请计算节点，然后加载集群模块并进入已有虚拟环境：

```bash
module load sghpc-mpi-gcc/26.3
conda activate bio_test
```

示例脚本不会申请计算资源，也不会加载模块或激活虚拟环境。与其他 OneScience 生物科学示例一致，脚本要求环境中已经设置 `ROCM_PATH` 和 `CONDA_PREFIX`，随后会加载 `${ROCM_PATH}/cuda/env.sh` 以及仓库根目录下的 `env.sh`。

Python 依赖和验证期间额外安装的两个 conda 软件包记录在 `requirements.txt` 中。验证过程没有替换或升级 DTK 定制的 Python 软件包。

## 模型与数据路径

Bash 入口脚本根据仓库根目录 `env.sh` 配置的 OneScience 标准路径推导默认模型和数据位置：

```text
${ONESCIENCE_MODELS_DIR}/scGPT/scGPT_human
${ONESCIENCE_DATASETS_DIR}/scGPT/annotation_pancreas
```

当前仓库配置对应以下共享目录：

```text
/public/share/sugonhpcapp01/onestore/onemodels
/public/share/sugonhpcapp01/onestore/onedatasets
```

可以通过以下环境变量覆盖具体位置，无需修改脚本：

- `SCGPT_MODEL_ROOT`：scGPT 模型根目录。
- `SCGPT_DATASET_ROOT`：scGPT 数据集根目录。
- `SCGPT_MODEL_DIR`：包含 `args.json`、`best_model.pt` 和 `vocab.json` 的模型目录。
- `SCGPT_INFERENCE_DATA`：用于嵌入推理的 AnnData 文件。
- `SCGPT_FINETUNE_DATA`：用于细胞类型微调的 AnnData 文件。
- `SCGPT_OUTPUT_ROOT`：输出目录，默认为本示例目录下的 `outputs`。
- `SCGPT_DEVICE`：PyTorch 计算设备，默认为 `cuda`。

## 单卡与多卡运行

推理和微调入口会通过 `torch.cuda.device_count()` 自动检测当前环境中 PyTorch 可见的计算设备数量，无需指定卡数：

- 只检测到一张卡时，脚本直接使用单进程运行。
- 检测到多张卡时，脚本自动使用 `torchrun`，每张卡启动一个进程。

可见设备数量由用户申请的计算资源以及 `HIP_VISIBLE_DEVICES` 等环境变量决定。例如，申请并进入一个具有多张可见 DCU 的计算节点后，仍然使用相同命令：

```bash
bash scripts/infer.sh
bash scripts/finetune.sh
```

多卡推理会按连续的细胞区间将数据分配给各张卡，最后由主进程按原始顺序合并嵌入。多卡微调使用 PyTorch DistributedDataParallel 和 DistributedSampler；只有主进程执行 AnnData 预处理和保存最终产物，避免各进程重复进行 Scanpy 预处理或相互覆盖输出文件。

多卡微调中的 `--batch-size` 表示每张卡的批量大小。例如，8 张卡且 `--batch-size 4` 时，有效全局批量大小为 32。`--max-steps` 表示所有 rank 同步执行的优化步数。

## 细胞嵌入推理

进入 `examples/biosciences/scgpt` 目录后运行：

```bash
bash scripts/infer.sh
```

默认输出文件为 `outputs/pancreas_embeddings.h5ad`，归一化后的细胞嵌入保存在 `obsm["X_scGPT"]` 中。

可以在 Bash 命令后追加 Python 脚本支持的参数。例如，仅对 64 个细胞执行推理：

```bash
bash scripts/infer.sh \
  --max-cells 64 \
  --max-length 256 \
  --batch-size 8 \
  --output outputs/pancreas_embeddings_64.h5ad
```

## 细胞类型微调

共享胰腺数据集使用 `Celltype` 作为标签列，使用 `Gene Symbol` 作为基因符号列，示例脚本已将其设为默认值：

```bash
bash scripts/finetune.sh \
  --epochs 5 \
  --batch-size 32
```

默认输出目录为 `outputs/pancreas_finetune`，其中包含以下文件：

- `best_model.pt`：验证指标最优的模型权重。
- `args.json`：模型配置、标签名称和数据处理信息。
- `vocab.json`：模型使用的基因词表。
- `metrics.json`：验证集指标。

进行较长时间的训练前，可以先运行以下短流程验证完整链路：

```bash
bash scripts/finetune.sh \
  --max-cells 64 \
  --n-hvg 200 \
  --max-length 201 \
  --batch-size 8 \
  --epochs 1 \
  --max-steps 2 \
  --freeze-encoder \
  --output-dir outputs/pancreas_finetune_64
```

对于元数据字段不同的数据集，可以使用 `--gene-column` 或 `--label-column` 指定相应列。未指定基因列时，程序会依次检查 `Gene Symbol`、`feature_name`、`gene_name`、`gene_symbols` 和 `symbol`，均不存在时使用 AnnData 的 `var_names`。

微调数据管道会将非负整数矩阵识别为原始计数。共享胰腺数据会被识别为已归一化数据。可以使用 `--data-is-raw` 或 `--data-is-normalized` 显式覆盖自动判断结果。

## 注意力后端

默认使用 PyTorch 注意力实现，以保证检查点兼容性。只有在确认已安装的 Flash Attention 扩展与当前 DTK 运行时匹配时，才建议设置以下环境变量并为嵌入推理传入 `--use-fast-transformer`：

```bash
export ONESCIENCE_SCGPT_ENABLE_FLASH_ATTN=1
bash scripts/infer.sh --use-fast-transformer
```

微调示例默认使用 PyTorch 注意力实现。

合入的 scGPT 源代码沿用上游 MIT 许可证。仓库级许可证和第三方声明请参阅 OneScience 根目录下的 `LICENSE` 和 `NOTICE`。
