# OpenFold

本示例将 OpenFold 蛋白质结构预测模型集成到 OneScience，提供单体结构推理、预计算 MSA/模板特征复用、OpenFold/AlphaFold 参数加载、数据缓存构建、单机/多机训练以及微调入口。

## 简介

OpenFold 是 DeepMind AlphaFold2 的可训练 PyTorch 复现，支持使用 OpenFold checkpoint 或 AlphaFold JAX 参数进行蛋白质结构预测，也支持从 mmCIF 结构数据和预计算 alignment 训练单体模型。OneScience 当前仓库将核心模型、特征管线和工具函数拆分到 `src/onescience` 中，`examples/biosciences/openfold` 保留可直接运行的推理、训练和数据准备脚本。

- 官方文档：[OpenFold documentation](https://openfold.readthedocs.io/en/latest/)
- 训练说明：[Training OpenFold](https://openfold.readthedocs.io/en/latest/Training_OpenFold.html)
- 原始项目：[aqlaboratory/openfold](https://github.com/aqlaboratory/openfold)
- 许可证：[Apache License 2.0](https://github.com/aqlaboratory/openfold/blob/main/LICENSE)

## 目录

- [主要目录与入口](#主要目录与入口)
- [环境准备](#环境准备)
- [数据与模型准备](#数据与模型准备)
- [功能与入口](#功能与入口)
- [单体结构推理](#单体结构推理)
- [Alignment 预计算](#alignment-预计算)
- [训练数据准备](#训练数据准备)
- [模型训练](#模型训练)
- [微调与恢复训练](#微调与恢复训练)
- [输出与复现](#输出与复现)
- [运行约束](#运行约束)
- [Issues](#issues)
- [许可证与引用](#许可证与引用)

## 主要目录与入口

```text
examples/biosciences/openfold/
├── monomer/
│   ├── fasta_dir/
│   │   └── 6kwc.fasta                  # 单体推理 FASTA 示例
│   ├── alignments/
│   │   └── 6KWC_1/                     # 预计算 alignment 示例
│   └── inference.sh                    # 单体推理脚本
├── scripts/
│   ├── download_alphafold_dbs.sh       # 下载 AlphaFold/OpenFold 推理数据库
│   ├── download_openfold_params.sh     # 下载 OpenFold 参数
│   ├── download_alphafold_params.sh    # 下载 AlphaFold 参数
│   ├── precompute_alignments.py        # JackHMMER/HHblits/HHsearch MSA 预计算
│   ├── precompute_alignments_mmseqs.py # MMseqs2/ColabFold 风格 MSA 预计算
│   ├── generate_mmcif_cache.py         # 生成 template release-date cache
│   ├── generate_chain_data_cache.py    # 生成训练链级 cache
│   ├── build_deepspeed_config.py       # 生成 DeepSpeed 配置
│   ├── convert_v1_to_v2_weights.py     # 转换 OpenFold v1 权重命名
│   ├── alignment_db_scripts/           # alignment DB 构建与索引合并工具
│   └── ...                             # 其他 OpenFold 辅助脚本
├── deepspeed_config.json               # DeepSpeed 训练配置示例
├── run_pretrained_openfold.py          # 预训练模型推理入口
├── thread_sequence.py                  # 将序列 thread 到指定模板结构
├── train_openfold.py                   # OpenFold 训练入口
└── setup.py                            # OpenFold CUDA/CPU 扩展构建脚本
```

OneScience 中与 OpenFold 相关的核心代码位于：

```text
src/onescience/configs/bio/openfold/      # 模型与数据配置
src/onescience/models/openfold/           # AlphaFold/OpenFold 模型模块
src/onescience/datapipes/openfold/        # FASTA、MSA、模板与特征处理
src/onescience/utils/openfold/            # 权重导入、relax、loss、metrics、tensor 工具
```

## 环境准备

在仓库根目录执行以下命令：

```bash
conda create -n onescience-openfold python=3.11 -y
conda activate onescience-openfold
bash install.sh bio
source env.sh
```

当前 OneScience `env.sh` 在 SCNET 环境中默认设置：

```bash
export ONESCIENCE_DATASETS_DIR="/public/share/sugonhpcapp01/onestore/onedatasets"
export ONESCIENCE_MODELS_DIR="/public/share/sugonhpcapp01/onestore/onemodels"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export device="gpu"
```

如果在其他集群或工作站运行，请将 `ONESCIENCE_DATASETS_DIR` 和 `ONESCIENCE_MODELS_DIR` 修改为本地实际路径。

OpenFold 推理和训练依赖 PyTorch、OpenMM、Biopython、HH-suite、HMMER、Kalign 等工具。若需要在线生成 MSA，还需安装并确认以下二进制可访问：

```bash
which jackhmmer
which hhblits
which hhsearch
which hmmsearch
which hmmbuild
which kalign
```

如需构建 OpenFold 自带注意力扩展，可在 `examples/biosciences/openfold` 下执行：

```bash
cd examples/biosciences/openfold
python setup.py build_ext --inplace
```

## 数据与模型准备

### 模型参数

OpenFold 参数可通过脚本下载到模型目录：

```bash
cd examples/biosciences/openfold
bash scripts/download_openfold_params.sh ${ONESCIENCE_MODELS_DIR}/OpenFold
```

注意：当前 `download_openfold_params.sh` 会在传入目录下继续创建 `openfold_params/` 子目录。因此上述命令实际将参数下载到：

```text
${ONESCIENCE_MODELS_DIR}/OpenFold/openfold_params/
```

而 `monomer/inference.sh` 默认查找 `${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt`。使用下载脚本后，请将 `--openfold_checkpoint_path` 指向实际 checkpoint，或通过移动/软链接使默认路径可用。

AlphaFold JAX 参数可通过以下脚本准备：

```bash
cd examples/biosciences/openfold
bash scripts/download_alphafold_params.sh ${ONESCIENCE_MODELS_DIR}/AlphaFold
```

推理时可二选一：

- `--openfold_checkpoint_path`：加载 OpenFold `.pt` checkpoint 或 DeepSpeed checkpoint 目录；
- `--jax_param_path`：加载 AlphaFold `.npz` JAX 参数。

`monomer/inference.sh` 默认使用：

```bash
${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt
```

请确认该文件存在，或将脚本中的路径替换为实际 checkpoint。

### 推理数据库

如果使用在线 MSA 和模板搜索，需要准备 UniRef90、MGnify、PDB70、BFD/Small BFD、UniRef30/UniClust30、PDB mmCIF 等数据库：

```bash
cd examples/biosciences/openfold
bash scripts/download_alphafold_dbs.sh ${ONESCIENCE_DATASETS_DIR}/alphafold full_dbs
```

`full_dbs` 数据规模很大，下载前应确认存储容量和网络条件。功能验证可以使用 `reduced_dbs` 或复用已有的预计算 alignment。

### 示例数据

仓库内置了一个单体推理示例：

```text
monomer/fasta_dir/6kwc.fasta
monomer/alignments/6KWC_1/
```

该示例已经包含 `uniref90_hits.sto`、`mgnify_hits.sto`、`hhsearch_output.hhr` 和 `bfd_uniref_hits.a3m`，因此可以通过 `--use_precomputed_alignments` 跳过数据库搜索。

> `--use_precomputed_alignments` 只跳过 MSA/模板搜索计算，不会取消模板结构特征的读取；单体推理仍需要将位置参数 `template_mmcif_dir` 指向包含有效 `.cif` 文件的 mmCIF 目录。

## 功能与入口

| 任务 | 推荐入口 | 输入 | 输出 |
|------|----------|------|------|
| 单体结构推理 | `run_pretrained_openfold.py` 或 `monomer/inference.sh` | FASTA、template mmCIF、alignment、checkpoint | PDB/ModelCIF、pLDDT、可选中间输出 |
| 序列 thread 到模板 | `thread_sequence.py` | 单条 FASTA、mmCIF 模板、checkpoint | thread 后的结构文件 |
| MSA/模板预计算 | `scripts/precompute_alignments.py` | mmCIF/FASTA、序列数据库、模板数据库 | 每条链的 alignment 目录 |
| MMseqs2 预计算 | `scripts/precompute_alignments_mmseqs.py` | FASTA、MMseqs DB、PDB70 | alignment 目录 |
| 训练 cache 生成 | `generate_mmcif_cache.py`、`generate_chain_data_cache.py` | mmCIF/PDB 结构目录 | JSON cache |
| alignment DB 构建 | `alignment_db_scripts/create_alignment_db.py` | alignment 目录 | `.db` 与 `.index` |
| 模型训练 | `train_openfold.py` | mmCIF、alignment、template、cache | Lightning/OpenFold checkpoint |
| 微调 | `train_openfold.py --resume_from_ckpt --resume_model_weights_only true` | 训练数据与预训练 checkpoint | 微调 checkpoint |

所有相对路径命令均假定当前目录为 `examples/biosciences/openfold`。

## 单体结构推理

### 使用内置 monomer 示例

```bash
cd examples/biosciences/openfold
bash monomer/inference.sh
```

该脚本会：

- 读取 `monomer/fasta_dir/6kwc.fasta`；
- 复用 `monomer/alignments` 下的预计算 alignment；
- 从 `${ONESCIENCE_DATASETS_DIR}/alphafold2.3.0/pdb_mmcif/mmcif_files/` 读取模板结构；
- 加载 `${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt`；
- 将预测结果写入 `monomer/`。

### 直接调用推理入口

```bash
cd examples/biosciences/openfold
python run_pretrained_openfold.py \
  ./monomer/fasta_dir \
  ${ONESCIENCE_DATASETS_DIR}/alphafold2.3.0/pdb_mmcif/mmcif_files \
  --output_dir ./monomer \
  --config_preset model_1_ptm \
  --model_device cuda:0 \
  --data_random_seed 42 \
  --use_precomputed_alignments ./monomer/alignments \
  --openfold_checkpoint_path ${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt \
  --skip_relaxation
```

常用参数如下：

| 参数 | 说明 |
|------|------|
| `fasta_dir` | 包含 `.fasta` 或 `.fa` 文件的目录；单体模式下每个文件应包含一条序列 |
| `template_mmcif_dir` | 模板 mmCIF 文件目录 |
| `--use_precomputed_alignments` | 指向已生成 alignment 的目录；设置后跳过数据库搜索 |
| `--config_preset` | 推理配置，如 `model_1`、`model_1_ptm`、`model_3_ptm`、`seq_model_esm1b_ptm` |
| `--model_device` | 运行设备，如 `cuda:0`、`cpu` |
| `--openfold_checkpoint_path` | OpenFold checkpoint 路径，支持逗号分隔多个模型 |
| `--jax_param_path` | AlphaFold JAX 参数路径 |
| `--skip_relaxation` | 跳过 OpenMM relaxation，加快功能验证 |
| `--save_outputs` | 保存完整模型输出字典 |
| `--trace_model` | 对部分模型进行 TorchScript trace，适合大批量同长度推理 |
| `--long_sequence_inference` | 启用长序列低显存推理选项 |
| `--cif_output` | 输出 ModelCIF，而非默认 PDB |
| `--use_deepspeed_evoformer_attention` | 使用 DeepSpeed Evoformer attention，需要对应依赖 |

未提供 `--use_precomputed_alignments` 时，脚本会根据数据库参数自动运行 MSA 和模板搜索。例如：

```bash
cd examples/biosciences/openfold
python run_pretrained_openfold.py \
  /path/to/fasta_dir \
  ${ONESCIENCE_DATASETS_DIR}/alphafold/pdb_mmcif/mmcif_files \
  --output_dir ./outputs \
  --config_preset model_1_ptm \
  --model_device cuda:0 \
  --openfold_checkpoint_path ${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt \
  --uniref90_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/uniref90/uniref90.fasta \
  --mgnify_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/mgnify/mgy_clusters_2018_12.fa \
  --pdb70_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/pdb70/pdb70 \
  --bfd_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --uniref30_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/uniref30/UniRef30_2021_03 \
  --cpus 16
```

## Alignment 预计算

训练前通常需要为每条训练链预计算 alignment。官方文档也建议先完成 MSA/模板特征准备，再调用训练入口。

### JackHMMER/HH-suite 流程

```bash
cd examples/biosciences/openfold
python scripts/precompute_alignments.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  --uniref90_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/uniref90/uniref90.fasta \
  --mgnify_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/mgnify/mgy_clusters_2018_12.fa \
  --pdb70_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/pdb70/pdb70 \
  --uniclust30_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
  --bfd_database_path ${ONESCIENCE_DATASETS_DIR}/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
  --cpus_per_task 16
```

该步骤耗时较长。大规模训练建议在集群中分片执行，并记录每个分片使用的数据库版本。

### MMseqs2 流程

如果使用 ColabFold/MMseqs2 风格数据库，可执行：

```bash
cd examples/biosciences/openfold
python scripts/precompute_alignments_mmseqs.py \
  /path/to/input.fasta \
  ${ONESCIENCE_DATASETS_DIR}/openfold/mmseqs_dbs \
  uniref30_2103_db \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  /path/to/mmseqs \
  --hhsearch_binary_path /path/to/hhsearch \
  --env_db colabfold_envdb_202108_db \
  --pdb70 ${ONESCIENCE_DATASETS_DIR}/alphafold/pdb70/pdb70
```

可使用 `scripts/data_dir_to_fasta.py` 从 mmCIF/PDB 数据目录生成输入 FASTA。

## 训练数据准备

OpenFold 训练需要以下核心输入：

- 训练结构目录：包含 mmCIF 文件，通常为 `${DATA_DIR}/pdb_data/mmcifs`；
- 训练 alignment 目录：每条链一个子目录，如 `6kwc_A/`；
- template mmCIF 目录：可与训练 mmCIF 目录相同，也可使用独立模板库；
- template release-date cache：由 `generate_mmcif_cache.py` 生成；
- chain data cache：由 `generate_chain_data_cache.py` 生成；
- obsolete PDB 列表：用于过滤过时结构；
- 可选 validation/distillation 数据与 alignment。

建议的数据目录如下：

```text
${ONESCIENCE_DATASETS_DIR}/openfold/
├── pdb_data/
│   ├── mmcifs/
│   ├── obsolete.dat
│   └── data_caches/
│       ├── mmcif_cache.json
│       └── chain_data_cache.json
├── alignment_data/
│   ├── alignments/
│   └── alignment_db/
└── val_data/
    ├── mmcifs/
    └── alignments/
```

### 生成 template mmCIF cache

```bash
cd examples/biosciences/openfold
mkdir -p ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches
python scripts/generate_mmcif_cache.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --no_workers 16
```

该 cache 用于按 release date 过滤模板，训练命令中通过 `--template_release_dates_cache_path` 传入。

### 生成 chain data cache

```bash
cd examples/biosciences/openfold
python scripts/generate_chain_data_cache.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --cluster_file ${ONESCIENCE_DATASETS_DIR}/openfold/clusters-by-entity-40.txt \
  --no_workers 16
```

`--cluster_file` 是可选项，用于记录每条链所在聚类大小，训练采样时会用到该信息。若没有聚类文件，也可以先不传该参数完成小规模功能验证。

### 构建 alignment DB

在 I/O 压力较大的文件系统中，可以将大量 alignment 小文件合并成 DB：

```bash
cd examples/biosciences/openfold
mkdir -p ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignment_db
python scripts/alignment_db_scripts/create_alignment_db.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignment_db \
  alignment_db
```

生成文件包括：

```text
alignment_db.db
alignment_db.index
```

训练时将 alignment 位置改为 DB 所在目录，并通过 `--alignment_index_path` 指向 `.index` 文件。

## 模型训练

> **当前适配状态**：`run_pretrained_openfold.py` 已主要迁移到 `onescience.*` 命名空间；但当前仓库中的 `train_openfold.py`、`thread_sequence.py` 以及部分数据准备脚本仍保留上游 `openfold.*` import。若环境中只有 OneScience 而没有兼容的上游 OpenFold 包，这些入口可能出现 `ModuleNotFoundError`。因此下面的训练、微调、DeepSpeed 和多节点命令应视为当前脚本提供的接口示例，运行前需先确认依赖和命名空间兼容性。

### 从头训练

下面命令按当前 `train_openfold.py` 的参数接口给出基础训练示例；运行前请先确认上文所述上游 `openfold.*` 依赖可用：

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/initial_training \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16 \
  --max_epochs 1 \
  --log_every_n_steps 25 \
  --checkpoint_every_epoch
```

位置参数含义如下：

| 位置参数 | 说明 |
|----------|------|
| `train_data_dir` | 训练 mmCIF 文件目录 |
| `train_alignment_dir` | 预计算 alignment 目录，或 alignment DB 目录 |
| `template_mmcif_dir` | template mmCIF 文件目录 |
| `output_dir` | checkpoint、日志和训练输出目录 |
| `max_template_date` | 模板最大发布日期，防止使用晚于训练目标的结构信息 |

常用训练参数如下：

| 参数 | 说明 |
|------|------|
| `--config_preset` | 训练配置，如 `initial_training`、`finetuning`、`finetuning_ptm`、`finetuning_no_templ` |
| `--train_chain_data_cache_path` | 训练链级 cache |
| `--template_release_dates_cache_path` | template release-date cache |
| `--obsolete_pdbs_file_path` | obsolete PDB 列表 |
| `--num_nodes` | 节点数 |
| `--gpus` | 当前脚本主要用该值判断分布式 strategy 并计算有效 batch size；实际可见设备需由运行环境/启动器配置。多卡或多节点时必须设置 `--seed` |
| `--precision` | 训练精度，常见值为 `bf16`、`bf16-mixed`、`32` |
| `--deepspeed_config_path` | 启用 DeepSpeed 训练配置 |
| `--val_data_dir` / `--val_alignment_dir` | 验证集结构与 alignment |
| `--distillation_data_dir` / `--distillation_alignment_dir` | 自蒸馏数据 |
| `--experiment_config_json` | 覆盖配置中的具体键值 |
| `--wandb` | 启用 Weights & Biases 记录 |

### 使用 alignment DB 训练

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignment_db \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/initial_training_db \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --alignment_index_path ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignment_db/alignment_db.index \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16
```

### 使用验证集

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/with_validation \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --val_data_dir ${ONESCIENCE_DATASETS_DIR}/openfold/val_data/mmcifs \
  --val_alignment_dir ${ONESCIENCE_DATASETS_DIR}/openfold/val_data/alignments \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16
```

### DeepSpeed 训练

仓库提供 `deepspeed_config.json` 示例，也提供配置生成脚本。当前训练入口仍保留上游 `openfold.*` 依赖，因此使用 DeepSpeed 前应先确认训练脚本在本地环境可正常导入并启动：

```bash
cd examples/biosciences/openfold
python scripts/build_deepspeed_config.py
```

启动 DeepSpeed 训练：

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/deepspeed \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16 \
  --deepspeed_config_path ./deepspeed_config.json
```

### 多节点训练

`train_openfold.py` 使用 PyTorch Lightning。`--num_nodes` 会传给 Trainer；当前脚本中的 `--gpus` 主要用于选择分布式 strategy 和计算有效 batch size，并不会直接作为 Lightning 的 `devices` 参数传入。实际设备数量和可见设备需由集群启动器/运行环境配置。多节点运行前还需设置主节点地址、端口、rank 和通信后端。

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/multinode \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 2 \
  --gpus 8 \
  --precision bf16 \
  --deepspeed_config_path ./deepspeed_config.json
```

分布式训练必须显式设置 `--seed`，否则训练脚本会报错。若集群使用 MPI，可安装 `mpi4py` 并增加 `--mpi_plugin`。

## 微调与恢复训练

### 从 OpenFold checkpoint 微调

官方训练文档推荐在已有模型参数基础上微调时同时使用 `--resume_from_ckpt` 和 `--resume_model_weights_only`，这样只加载模型权重，训练步数从当前任务重新开始：

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/finetuning_ptm \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset finetuning_ptm \
  --seed 4242022 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16 \
  --resume_from_ckpt ${ONESCIENCE_MODELS_DIR}/OpenFold/finetuning_ptm_2.pt \
  --resume_model_weights_only true
```

### 从训练状态恢复

如果需要恢复 optimizer、scheduler、EMA 等完整训练状态，只传入 `--resume_from_ckpt`：

```bash
cd examples/biosciences/openfold
python train_openfold.py \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_DATASETS_DIR}/openfold/alignment_data/alignments \
  ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/mmcifs \
  ${ONESCIENCE_MODELS_DIR}/OpenFold/train_runs/resume \
  2021-10-10 \
  --train_chain_data_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/chain_data_cache.json \
  --template_release_dates_cache_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/data_caches/mmcif_cache.json \
  --obsolete_pdbs_file_path ${ONESCIENCE_DATASETS_DIR}/openfold/pdb_data/obsolete.dat \
  --config_preset initial_training \
  --seed 42 \
  --num_nodes 1 \
  --gpus 4 \
  --precision bf16 \
  --resume_from_ckpt /path/to/lightning_or_deepspeed_checkpoint
```

`--resume_from_jax_params` 可用于从 AlphaFold `.npz` 参数初始化，但不能与 `--resume_from_ckpt` 同时使用。

### 覆盖模型或数据配置

可通过 JSON 文件覆盖 `model_config` 中的扁平化键：

```json
{
  "data.train.crop_size": 128
}
```

训练时传入：

```bash
--experiment_config_json /path/to/openfold_experiment_config.json
```

## 输出与复现

推理输出通常包含：

- `*_unrelaxed.pdb` 或 `*_unrelaxed.cif`：未 relaxation 的预测结构；
- relaxed PDB/ModelCIF：未使用 `--skip_relaxation` 时生成；
- pLDDT 可写入结构文件的 B-factor 字段；
- 使用 `--save_outputs` 时额外保存完整模型输出 pickle，其中可包含 pTM 等置信度与中间结果。

训练输出通常包含：

- PyTorch Lightning 日志；
- OpenFold/DeepSpeed checkpoint；
- EMA 参数；
- 可选 WandB 记录；
- validation metrics，如 lDDT-Ca、dRMSD、GDT-TS、GDT-HA。

## 运行约束

- OpenFold 训练不支持纯 CPU；推理可在 CPU 上运行但速度很慢。
- 训练必须预先准备 alignment；在线为大规模训练集生成 MSA 会非常耗时。
- 多 GPU 或多节点训练必须设置 `--seed`；实际设备可见性和进程映射应由启动环境配置，不能仅依赖 `--gpus`。
- `template_mmcif_dir` 可以与训练 mmCIF 目录相同，但必须通过 `max_template_date` 和 release-date cache 控制模板泄漏。
- `full_dbs` 数据量很大，下载和解压前应评估磁盘容量、inode 数量和文件系统吞吐。
- alignment 目录包含大量小文件时，建议构建 alignment DB 以降低 I/O 压力。
- `--precision bf16` 或 `bf16-mixed` 对硬件和 PyTorch 后端有要求；不支持时应改用 `32`。
- `--deepspeed_config_path` 与 `--precision 16` 不兼容，脚本会直接报错。
- 结构预测结果是计算模型输出，涉及生物学解释时必须结合实验数据和专业分析验证。

## Issues

- 如果报 `Could not find CIFs in ...`，请先确认 `ONESCIENCE_DATASETS_DIR` 已正确加载，并检查传入的 `template_mmcif_dir` 目录中确实存在 `.cif` 文件；即使使用 `--use_precomputed_alignments`，该 mmCIF 目录仍然是必需的。
- 如果 `run_pretrained_openfold.py` 找不到 `jackhmmer`、`hhblits`、`hhsearch` 或 `kalign`，请确认相关二进制在 `PATH` 中，或显式传入 `--jackhmmer_binary_path` 等参数。
- 如果推理阶段找不到 `params_model_*.npz`，说明未提供 `--openfold_checkpoint_path` 且默认 AlphaFold 参数目录不存在；请下载参数或显式指定 checkpoint。
- 如果 relaxation 报 OpenMM 相关错误，可先使用 `--skip_relaxation` 完成结构预测功能验证。
- 如果长序列推理显存不足，可尝试 `--long_sequence_inference`、降低模板/MSA 数量，或切换更大显存设备。
- 如果训练出现大量小文件 I/O 瓶颈，可使用 `scripts/alignment_db_scripts/` 将 alignment 转换为 DB 格式。
- 如果旧 OpenFold v1 checkpoint 无法加载，可先运行 `scripts/convert_v1_to_v2_weights.py`。
- 如“模型训练”章节前的适配状态说明所述，`train_openfold.py`、`thread_sequence.py` 及部分数据预处理脚本仍保留上游 `openfold.*` import；若出现 `ModuleNotFoundError`，需安装兼容的 OpenFold 包，或将相应 import 迁移到 OneScience 已提供的 `onescience.configs.bio.openfold`、`onescience.datapipes.openfold`、`onescience.models.openfold` 和 `onescience.utils.openfold` 命名空间。

## 许可证与引用

OpenFold 源码采用 Apache License 2.0。AlphaFold 预训练参数和 OpenFold 参数可能采用不同许可证；下载、分发和商用前应分别确认对应条款，并同时遵守 OneScience 仓库许可证。

使用本示例开展研究时，请引用 OpenFold、AlphaFold；若使用 OpenProteinSet 或 OpenFold 训练数据，也应引用 OpenProteinSet。

```bibtex
@article {Ahdritz2022.11.20.517210,
    author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O'Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolo and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
    title = {OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
    elocation-id = {2022.11.20.517210},
    year = {2022},
    doi = {10.1101/2022.11.20.517210},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
    journal = {bioRxiv}
}
```
