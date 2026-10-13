# La-Proteina 

本示例将 La-Proteina 集成到 OneScience 生物信息（AI for Biology）组件中，提供训练、蛋白质结构生成、生成结果评估与自编码器推理的统一入口。

## 简介

La-Proteina 是一种基于**部分隐变量流匹配（Partially Latent Flow Matching）**的蛋白质结构生成模型，能够直接生成全原子蛋白质结构及其对应的氨基酸序列。模型将蛋白质的骨架（backbone CA）显式建模，而序列和原子级细节则通过每个残基的固定维度隐变量来捕捉，从而有效避免显式侧链表示带来的挑战。

论文：_La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching_（arXiv 2025）。

- [论文链接](https://arxiv.org/abs/2507.09466)
- [项目主页](https://research.nvidia.com/labs/genair/la-proteina/)

<div align="center">
    <img width="600" alt="teaser" src="assets/samples_visual.png"/>
</div>

La-Proteina 在多个生成基准上取得了领先性能，包括全原子协同可设计性（co-designability）、多样性、结构有效性以及原子级 motif 支架（motif scaffolding）。模型可生成最长约 800 个残基的蛋白质结构。

---

## 目录

- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [数据与模型权重](#数据与模型权重)
- [数据格式与预处理](#数据格式与预处理)
- [功能定位](#功能定位)
- [模型架构与配置关系](#模型架构与配置关系)
- [标准处理流程](#标准处理流程)
- [脚本索引](#脚本索引)
- [详细使用说明](#详细使用说明)
  - [0. 训练 AutoEncoder（`training_ae.yaml`）](#0-训练-autoencodertraining_aeyaml)
  - [1. 训练 La-Proteina 主模型（`run_train.sh`）](#1-训练-la-proteina-主模型run_trainsh)
  - [2. 蛋白质结构生成（`run_generate.sh`）](#2-蛋白质结构生成run_generatesh)
  - [3. 生成结果评估（`run_evaluate.sh`）](#3-生成结果评估run_evaluatesh)
  - [4. 自编码器推理（`run_ae_infer.sh`）](#4-自编码器推理run_ae_infersh)
- [配置说明](#配置说明)
- [运行约束](#运行约束)
- [Issues](#issues)
- [许可证与引用](#许可证与引用)

---

## 目录结构

```
examples/biosciences/laproteina/
├── scripts/                          # 可执行脚本
│   ├── run_train.sh                  # 训练 La-Proteina 主模型
│   ├── run_generate.sh               # 蛋白质结构生成
│   ├── run_evaluate.sh               # 生成结果评估
│   └── run_ae_infer.sh               # 自编码器推理/重建
├── train_laproteina.py               # 训练入口（Hydra 配置）
├── infer_laproteina.py               # 生成入口
├── evaluate_laproteina.py            # 评估入口
├── train_laproteina_ae.py            # 自编码器训练入口
├── infer_laproteina_ae.py            # 自编码器推理入口
└── README.md                         
```

对应源码模块位于 `src/onescience/models/laproteina/`。

---

## 环境准备

1. 参照项目根目录 [README.md](../../../README.md) 完成 OneScience（bio 领域）安装：

    ```bash
    bash install.sh bio
    ```

2. 激活环境：

    ```bash
    conda activate onescience311
    ```

3. 确保 `ONESCIENCE_DATASETS_DIR` 环境变量已设置（通常由项目根目录 `env.sh` 自动配置）：

    ```bash
    source /path/to/onescience/env.sh
    ```

4. 可选：通过环境变量覆盖默认路径：

    ```bash
    export LAPROTEINA_ROOT=/path/to/la-proteina
    export LAPROTEINA_DATASET_DIR=/path/to/dataset
    export LAPROTEINA_CHECKPOINTS_DIR=/path/to/checkpoints_laproteina
    export DATA_PATH=/path/to/dataset
    ```

---

## 数据与模型权重

### 1. 数据集

脚本默认读取以下路径：

```
${ONESCIENCE_DATASETS_DIR}/la-proteina/dataset/pdb_train
```

- 训练/推理使用 PDB 结构数据集。
- 官方论文中模型主要在 AFDB 子集上训练，相关 ID 列表可从 [NVIDIA NGC - la_proteina_afdb_ids.zip](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/resources/la_proteina_afdb_ids.zip/files) 下载。
- 当前 OneScience 示例使用 `pdb_train` 作为训练和评估数据集。

### 2. 模型权重

请预先下载 checkpoint 文件，默认存储路径如下：

```
${ONESCIENCE_DATASETS_DIR}/la-proteina/checkpoints_laproteina/
```

官方权重下载地址：[NVIDIA NGC - La-Proteina Weights & Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/collections/laproteina_weights_data/artifacts)

#### 隐变量扩散模型（Latent Diffusion）

| 模型 | 配置文件 | 说明 | 默认对应 AE |
|------|----------|------|-------------|
| LD1 | `inference_ucond_notri` | 无条件生成，无三角注意力层，最长 500 残基 | AE1 |
| LD2 | `inference_ucond_tri` | 无条件生成，含三角乘法更新层，最长 500 残基 | AE1 |
| LD3 | `inference_ucond_notri_long` | 无条件生成，无三角注意力层，300-800 残基 | AE2 |
| LD4 | `inference_motif_idx_aa` | 索引式（indexed）全原子 motif 支架 | AE3 |
| LD5 | `inference_motif_idx_tip` | 索引式（indexed） tip-原子 motif 支架 | AE3 |
| LD6 | `inference_motif_uidx_aa` | 非索引式（unindexed）全原子 motif 支架 | AE3 |
| LD7 | `inference_motif_uidx_tip` | 非索引式（unindexed） tip-原子 motif 支架 | AE3 |

#### 自编码器（Autoencoder）

| 模型 | 文件名 | 说明 |
|------|--------|------|
| AE1 | `AE1_ucond_512.ckpt` | 无条件生成最长 500 残基，与 LD1/LD2 配对 |
| AE2 | `AE2_ucond_800.ckpt` | 无条件生成 300-800 残基，与 LD3 配对 |
| AE3 | `AE3_motif.ckpt` | 原子级 motif 支架，与 LD4/LD5/LD6/LD7 配对 |

### 3. 默认路径汇总

| 用途 | 默认路径 |
|------|----------|
| 数据根目录 | `${ONESCIENCE_DATASETS_DIR}/la-proteina` |
| 训练数据集 | `${ONESCIENCE_DATASETS_DIR}/la-proteina/dataset/pdb_train` |
| 自编码器权重 | `${ONESCIENCE_DATASETS_DIR}/la-proteina/checkpoints_laproteina/AE1_ucond_512.ckpt` |
| 扩散模型权重 | `${ONESCIENCE_DATASETS_DIR}/la-proteina/checkpoints_laproteina/LD2_ucond_tri_512.ckpt` |

---

## 数据格式与预处理

使用官方预训练 AE 和 LD 权重进行结构生成时，无需重新预处理训练数据，可直接执行蛋白质结构生成流程。

从零训练或继续训练时，需要先准备 PDB 数据目录。默认数据根目录为：

```text
$ONESCIENCE_DATASETS_DIR/la-proteina/dataset/pdb_train/
```

数据配置位于 `src/onescience/configs/bio/laproteina/dataset/`。不同配置负责不同任务：

| 数据配置 | 用途 |
|----------|------|
| `pdb/pdb_train_ucond` | 普通无条件 AE/LD 训练 |
| `pdb/pdb_train_motif_aa` | 全原子 motif 条件训练 |
| `pdb/pdb_train_motif_tip` | tip-atom motif 条件训练 |

训练前必须确认数据目录、数据配置和网络配置相互匹配。数据预处理产生的结构、序列和 motif 特征会被 AE 训练和 LD 训练共同使用。

---
## 功能定位

- **蛋白质结构生成**：基于流匹配生成蛋白质主链（backbone CA）与局部隐变量（local latents）。
- **Motif 约束生成**：支持 motif 位置与序列约束，实现功能 motif 的骨架设计。
- **训练扩散模型**：在 PDB 数据集上训练 La-Proteina 主模型。
- **训练自编码器**：训练局部隐变量自编码器（local-latent autoencoder）。
- **评估生成结果**：计算生成结构的 RMSD、序列恢复率、(co-)designability 等指标。
- **自编码器推理**：对 PDB 结构进行编码-解码重建并评估重建质量。

---

## 模型架构与配置关系

La-Proteina 采用两阶段架构。`training_*.yaml` 是训练模板，`inference_*.yaml` 是推理时的模型装配配置；两者不是按文件名一一对应。

```text
training_ae.yaml
    │ 训练 AutoEncoder
    ▼
AE checkpoint
    │ 通过 autoencoder_ckpt_path 提供给第二阶段
    ▼
training_local_latents.yaml
    │ 训练 backbone CA + local latent 的 Flow Matching 模型
    ▼
LD checkpoint + 配套 AE checkpoint
    │
    ▼
inference_xxx.yaml
    │ 选择生成任务和采样参数
    ▼
生成蛋白质结构
```

### 训练配置的职责

| 配置 | 训练内容 | 主要产物 | 是否依赖其他 checkpoint |
|------|----------|----------|--------------------------|
| `training_ae.yaml` | 局部隐变量 AutoEncoder | AE checkpoint | 否，可通过 `pretrain_ckpt_path` 续训 |
| `training_local_latents.yaml` | Local Latents Flow Matching 主模型 | LD checkpoint | 是，必须通过 `autoencoder_ckpt_path` 加载 AE |

这两个训练文件是可通过 Hydra 覆盖的通用模板。发布的 AE1/AE2/AE3 和 LD1～LD7 代表不同数据范围、网络结构或条件任务，不能认为运行一次默认配置就会同时得到全部模型。

### 发布模型与训练模板的对应关系

| 目标模型 | training_local_latents.yaml 的关键选择 | 推理配置 | 必须配套的 AE |
|----------|------------------------------------------|----------|---------------|
| LD1 | `dataset=pdb/pdb_train_ucond`，`nn=local_latents_score_nn_160M` | `inference_ucond_notri` | AE1 |
| LD2 | `dataset=pdb/pdb_train_ucond`，`nn=local_latents_score_nn_160M_tri` | `inference_ucond_tri` | AE1 |
| LD3 | 长序列训练数据/长度设置，非 triangular 网络 | `inference_ucond_notri_long` | AE2 |
| LD4 | `dataset=pdb/pdb_train_motif_aa`，`nn=local_latents_score_nn_160M_motif_idx_aa` | `inference_motif_idx_aa` | AE3 |
| LD5 | `dataset=pdb/pdb_train_motif_tip`，`nn=local_latents_score_nn_160M_motif_idx_tip` | `inference_motif_idx_tip` | AE3 |
| LD6 | `dataset=pdb/pdb_train_motif_aa`，`nn=local_latents_score_nn_160M_motif_uidx` | `inference_motif_uidx_aa` | AE3 |
| LD7 | `dataset=pdb/pdb_train_motif_tip`，`nn=local_latents_score_nn_160M_motif_uidx` | `inference_motif_uidx_tip` | AE3 |

- `tri` / `notri`：是否启用 triangular multiplication 层。
- `idx` / `uidx`：motif 残基是否映射到指定序列索引。
- `aa` / `tip`：使用 motif 的全原子信息还是侧链 tip 原子信息。
- `inference_ae.yaml`：仅用于评估 AE 编码-解码重建，不加载 LD。

### 推理配置装配关系

所有完整生成配置都继承 `inference_base.yaml`，关键字段如下：

```yaml
ckpt_path: /path/to/local-latents/checkpoints
ckpt_name: last-EMA.ckpt
autoencoder_ckpt_path: /path/to/autoencoder/checkpoints/last-EMA.ckpt
```

`ckpt_path` 和 `ckpt_name` 指向第二阶段 LD 权重；`autoencoder_ckpt_path` 指向训练该 LD 时使用的第一阶段 AE 权重。两者必须配套，推理任务也必须与训练时的数据和网络结构一致。

使用自训练模型时，应复制任务最接近的 `inference_xxx.yaml`，并仅替换上述三个 checkpoint 字段。非 triangular 模型不适用于 `inference_ucond_tri`；不同长度或 motif 类型的 AE 与 LD 不得混用。

---
## 标准处理流程：
数据处理 → AE 训练 → LD 训练 → 推理 → 评估

```text
PDB 结构数据
    │ 数据配置与特征处理
    ▼
training_ae.yaml
    │ 训练局部隐变量 AutoEncoder
    ▼
AE checkpoint
    │ 写入 training_local_latents.yaml 的 autoencoder_ckpt_path
    ▼
training_local_latents.yaml
    │ 训练 Local Latents Flow Matching 模型
    ▼
LD checkpoint + 配套 AE checkpoint
    │ 写入 inference_xxx.yaml
    ▼
run_generate.sh
    │ 生成 PDB 结构
    ▼
run_evaluate.sh
    │ 计算 designability、RMSD 等指标
    ▼
评估 CSV
```

从零训练必须先训练 AE，再使用该 AE 训练 LD;
LD 训练依赖 AE，推理时也必须使用与 LD 兼容的 AE;
使用官方 AE/LD 权重时，无需执行两阶段训练，可直接进入“蛋白质结构生成”章节。

---
## 脚本索引

以下 4 个 `bash` 脚本为本示例的官方入口，均可直接运行。

| 脚本 | 功能 | 推荐运行方式 | 默认输出 |
|------|------|--------------|----------|
| `scripts/run_train.sh` | 训练 La-Proteina 主模型 | `bash scripts/run_train.sh` | `./store/<run_name>/` |
| `scripts/run_generate.sh` | 蛋白质结构生成 | `bash scripts/run_generate.sh` | `./inference/inference_ucond_tri/` |
| `scripts/run_evaluate.sh` | 生成结构评估 | `bash scripts/run_evaluate.sh` | `./inference/results_inference_ucond_tri_0.csv` |
| `scripts/run_ae_infer.sh` | 自编码器编码-解码推理 | `bash scripts/run_ae_infer.sh` | `./inference/inference_ae/` |

以上脚本均位于 `examples/biosciences/laproteina/scripts/` 目录。

---

## 详细使用说明

建议在 `examples/biosciences/laproteina` 目录下运行脚本，以便输出集中管理。

### 0. 训练 AutoEncoder（`training_ae.yaml`）

完整生成模型需要两个阶段。第一阶段训练 AutoEncoder，把结构和序列编码为 local latent，并支持从 latent 解码重建结构。

配置文件：`src/onescience/configs/bio/laproteina/training_ae.yaml`

```bash
cd examples/biosciences/laproteina
python train_laproteina_ae.py
```

默认 checkpoint 目录：

```text
./store/test_release_ae/checkpoints/
```

训练完成后，将选定的 AE checkpoint 写入 `training_local_latents.yaml` 的 `autoencoder_ckpt_path`，再训练第二阶段 LD 模型。
### 1. 训练 La-Proteina 主模型（`run_train.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_train.sh
```

脚本会检查 `DATA_PATH/pdb_train` 与 `LAPROTEINA_CHECKPOINTS_DIR/AE1_ucond_512.ckpt` 是否存在，并自动设置 `PYTHONPATH` 与相关环境变量。

默认配置：`src/onescience/configs/bio/laproteina/training_local_latents.yaml`

**常用 Hydra 参数覆盖：**

```bash
# 单卡调试
bash scripts/run_train.sh hardware.ngpus_per_node_=1 single=true

# 四卡调试
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_train.sh \
  run_name_=laproteina_4gpu \
  opt.max_epochs=10 \
  hardware.ngpus_per_node_=4 \
  opt.dist_strategy=ddp

# 指定运行名称
bash scripts/run_train.sh run_name=my_laproteina_run

# 覆盖数据集或网络配置（对应 motif 训练场景）
bash scripts/run_train.sh dataset=pdb/pdb_train_motif_aa nn=local_latents_score_nn_160M_motif_idx_aa
```

脚本内部调用：

```bash
python train_laproteina.py "+CK_PATH=$LAPROTEINA_ROOT" "$@"
```

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LAPROTEINA_ROOT` | `${ONESCIENCE_DATASETS_DIR}/la-proteina` | 数据与权重根目录 |
| `LAPROTEINA_CHECKPOINTS_DIR` | `${LAPROTEINA_ROOT}/checkpoints_laproteina` | 自编码器权重目录 |
| `DATA_PATH` | `${LAPROTEINA_ROOT}/dataset` | 数据集目录 |

输出：

- `./store/<run_name>/`：训练日志、检查点与 Hydra 配置

---

### 2. 蛋白质结构生成（`run_generate.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_generate.sh
```

默认使用 `inference_ucond_tri` 配置进行无条件生成（LD2 模型 + AE1）。

脚本内部调用：

```bash
python infer_laproteina.py --config_name inference_ucond_tri "$@"
```

**切换生成配置：**

```bash
# 无条件生成（无三角注意力）
bash scripts/run_generate.sh --config_name inference_ucond_notri

# 无条件生成长链（300-800 残基）
bash scripts/run_generate.sh --config_name inference_ucond_notri_long

# 索引式全原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_idx_aa

# 索引式 tip-原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_idx_tip

# 非索引式全原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_uidx_aa

# 非索引式 tip-原子 motif 支架
bash scripts/run_generate.sh --config_name inference_motif_uidx_tip
```

**各配置说明：**

| 配置名 | 对应模型 | 任务类型 | 默认生成长度 |
|--------|----------|----------|--------------|
| `inference_ucond_tri` | LD2 + AE1 | 无条件生成 | [100, 200, 300, 400, 500]，每种 100 个样本 |
| `inference_ucond_notri` | LD1 + AE1 | 无条件生成 | [100, 200, 300, 400, 500]，每种 100 个样本 |
| `inference_ucond_notri_long` | LD3 + AE2 | 无条件长链生成 | [300, 400, 500, 600, 700, 800]，每种 100 个样本 |
| `inference_motif_idx_aa` | LD4 + AE3 | 索引式全原子 motif 支架 | 由 `configs/generation/motif.yaml` 指定 |
| `inference_motif_idx_tip` | LD5 + AE3 | 索引式 tip-原子 motif 支架 | 由 `configs/generation/motif.yaml` 指定 |
| `inference_motif_uidx_aa` | LD6 + AE3 | 非索引式全原子 motif 支架 | 由 `configs/generation/motif.yaml` 指定 |
| `inference_motif_uidx_tip` | LD7 + AE3 | 非索引式 tip-原子 motif 支架 | 由 `configs/generation/motif.yaml` 指定 |

**OneScience 移植版本与官方采样参数**

OneScience 当前配置偏向快速验证，默认值可能小于官方完整评估规模：

```yaml
nres_lens: [100]
nsamples: 2
nsteps: 20
self_cond: False
```

官方完整评估通常使用多种长度、更多样本和更多采样步数，例如：

```yaml
nres_lens: [100, 200, 300, 400, 500]
nsamples: 100
nsteps: 400
self_cond: True
```

正式评估时，应根据可用显存和计算时间调整 `configs/generation/uncond_codes.yaml`，并同步确认 checkpoint 与配置的兼容性。
**关键推理参数说明：**

- `ckpt_name`：隐变量扩散模型权重文件名（仅文件名，不需要完整路径）。
- `autoencoder_ckpt_path`：自编码器权重完整路径。
- `self_cond`：是否使用自条件采样，官方评估默认开启。
- `sc_scale_noise`：alpha 碳原子与隐变量的噪声尺度。

输出：

- `./inference/<config_name>/`：生成的蛋白质结构文件与元数据

---

### 3. 生成结果评估（`run_evaluate.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_evaluate.sh
```

默认评估 `inference_ucond_tri` 配置对应的生成结果。

脚本内部调用：

```bash
python evaluate_laproteina.py --config_name inference_ucond_tri "$@"
```

**切换评估配置：**

```bash
bash scripts/run_evaluate.sh --config_name inference_motif_idx_aa
```

评估指标包括：

- 全原子 RMSD
- 序列恢复率（sequence recovery）
- (Co-)designability（需 ProteinMPNN 权重）
- Motif RMSD（motif 支架任务）
- Motif 序列恢复率（motif 支架任务）

**ProteinMPNN 权重准备：**

评估前需下载 ProteinMPNN 权重，可在项目根目录执行：

```bash
bash script_utils/download_pmpnn_weights.sh
```

输出：

- `./inference/<config_name>/results_<config_name>_<job_id>.csv`：评估结果文件

---

### 4. 自编码器推理（`run_ae_infer.sh`）

```bash
cd examples/biosciences/laproteina
bash scripts/run_ae_infer.sh --config_name inference_ae   --output-dir /path/to/onescience/examples/biosciences/laproteina/inference_ae
```

对 PDB 数据集执行编码-解码重建，评估重建指标（如全原子 RMSD、序列恢复率等）。

脚本会检查 `DATA_PATH/pdb_train` 与 `AE1_ucond_512.ckpt` 是否存在。

脚本内部调用：

```bash
python infer_laproteina_ae.py "$@"
```

常用参数覆盖：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LAPROTEINA_ROOT` | `${ONESCIENCE_DATASETS_DIR}/la-proteina` | 数据与权重根目录 |
| `LAPROTEINA_CHECKPOINTS_DIR` | `${LAPROTEINA_ROOT}/checkpoints_laproteina` | 自编码器权重目录 |
| `DATA_PATH` | `${LAPROTEINA_ROOT}/dataset` | 数据集目录 |

输出：

- `./inference_ae/`：重建结构与评估指标

---

## 配置说明

所有推理配置基于 `src/onescience/configs/bio/laproteina/inference_base.yaml`，各实验配置通过覆盖部分参数得到。

### 配置组合链

```text
training_ae.yaml
    └─训练并输出 AE checkpoint
                   │
                   ▼ autoencoder_ckpt_path
training_local_latents.yaml
    └─训练并输出 LD checkpoint

inference_xxx.yaml
    ├─继承 inference_base.yaml        通用模型与采样设置
    ├─引入 generation/xxx.yaml        数据集、生成数量和评估设置
    ├─指定 ckpt_path + ckpt_name      加载 LD checkpoint
    └─指定 autoencoder_ckpt_path      加载与 LD 配套的 AE checkpoint
```

### 训练相关配置

| 配置 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `training_ae.yaml` | 训练局部隐变量 AutoEncoder | 结构和序列训练数据 | AE checkpoint |
| `training_local_latents.yaml` | 训练 Local Latents Flow Matching 主模型 | 训练数据 + 已训练的 AE checkpoint | LD checkpoint |

两者是前后依赖关系，不是两种可互换的训练方式。第二阶段通过 `autoencoder_ckpt_path` 使用第一阶段的 AE；推理时必须继续使用同一个或严格兼容的 AE。

### 完整生成配置

具体 `inference_xxx.yaml` 使用 Hydra `defaults` 合并三层配置：

1. `inference_base.yaml`：公共默认值，例如 checkpoint 根目录、采样步数、self-conditioning、噪声尺度和模型组件。
2. `generation/*.yaml`：生成数据集、长度、样本数以及评估指标。
3. `inference_xxx.yaml`：最终覆盖层，指定 LD 权重、AE 权重、运行名称和具体任务。

例如：

```yaml
defaults:
  - inference_base
  - generation: uncond_codes
  - _self_
```

`_self_` 放在最后，表示当前 `inference_xxx.yaml` 中的字段优先级最高，可以覆盖前面继承的值。

| 推理配置 | 引入的 generation 配置 | checkpoint 组合 | 用途 |
|----------|--------------------------|-----------------|------|
| `inference_ucond_notri.yaml` | `generation/uncond_codes.yaml` | LD1 + AE1 | 无 triangular 的无条件生成 |
| `inference_ucond_tri.yaml` | `generation/uncond_codes.yaml` | LD2 + AE1 | 有 triangular 的无条件生成 |
| `inference_ucond_notri_long.yaml` | `generation/uncond_codes_800.yaml` | LD3 + AE2 | 300～800 残基长链生成 |
| `inference_motif_idx_aa.yaml` | `generation/motif.yaml` | LD4 + AE3 | indexed 全原子 motif |
| `inference_motif_idx_tip.yaml` | `generation/motif.yaml` | LD5 + AE3 | indexed tip-atom motif |
| `inference_motif_uidx_aa.yaml` | `generation/motif.yaml` | LD6 + AE3 | unindexed 全原子 motif |
| `inference_motif_uidx_tip.yaml` | `generation/motif.yaml` | LD7 + AE3 | unindexed tip-atom motif |

`inference_ae.yaml` 不属于上述完整生成链。该配置仅通过 `ckpt_file` 加载一个 AE checkpoint，对 PDB 结构执行编码、解码和重建指标计算，不加载 LD checkpoint。

### Motif 任务配置

Motif 配置还包含一层组合关系：

```text
generation/motif_dict.yaml
    └─定义任务名称、contig_string、motif PDB 和长度范围
                         │
                         ▼
generation/motif.yaml
    └─继承任务字典，并定义样本数量和 motif 评估指标
                         │
                         ▼
inference_motif_xx.yaml
    └─通过 generation.dataset.motif_task_name 选择一个具体任务
```

- `generation/motif_dict.yaml`：定义 motif 任务及各任务约束的残基。
- `generation/motif.yaml`：定义各任务的生成样本数和评估指标。
- `inference_motif_xx.yaml`：指定 LD/AE 模型及当前执行的 motif 任务。
> 任务命名约定：
> - 名称含 `_TIP` 后缀的任务用于 tip-原子模型（LD5/LD7）。
> - 名称不含 `_TIP` 后缀的任务用于全原子模型（LD4/LD6）。

---

## 运行约束

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 训练脚本默认检查 `DATA_PATH/pdb_train` 与 `AE1_ucond_512.ckpt` 是否存在，缺失会报错退出。
- 当前集成中 `dataset=pdb` 已可用，`dataset=genie2` 与 `dataset=pdb_multimer` 在当前仓库快照中未打包，运行会显式报错。
- 脚本会自动设置 ROCm/DCU 相关的 `LD_LIBRARY_PATH`，在海光 DCU 平台可直接运行；在 CUDA 平台可忽略或按需调整。
- 生成 motif 支架结构时，请确保 LD 模型与对应的 AE 模型配对正确，否则可能因长度/任务不匹配导致失败。
- 评估 (co-)designability 需要 ProteinMPNN 权重，请提前运行 `script_utils/download_pmpnn_weights.sh` 下载。
- 所有脚本建议在 `examples/biosciences/laproteina` 目录下执行，以便输出目录统一。
- README 中的 `training_*.yaml`、`inference_*.yaml` 和 `generation/*.yaml` 是 Hydra 配置名，实际文件位于 `src/onescience/configs/bio/laproteina/`，不是示例目录下的同名文件。
- `scripts/run_*.sh` 是实际路径；文中省略 `scripts/` 的名称表示脚本 basename。
- 通过 Hydra 覆盖配置时，可以使用 `+CK_PATH=...` 指定 checkpoint 根路径，脚本在未提供时会自动设置为 `LAPROTEINA_ROOT`。

---

## Issues

- 在 SCNET 平台运行时，若出现 `dlerror: libamdocl64.so: cannot open shared object file` 错误，可设置环境变量解决：  
  ```bash
  `export LD_LIBRARY_PATH=${ROCM_PATH}/opencl/lib:$LD_LIBRARY_PATH`
   ```
- 确认 `huggingface-hub` 版本满足 `>=0.34.0,<1.0`。缺失或版本不兼容时，执行：
    ```bash
    pip install "huggingface-hub>=0.34.0,<1.0"
    ```

---

## 许可证与引用

示例代码采用 Apache 2.0 许可证。La-Proteina 模型权重采用 [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)，其他材料采用 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)。

如果您在研究中使用了 La-Proteina，请引用原始论文：

```bibtex
@article{geffner2025laproteina,
  title={La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching},
  author={Geffner, Tomas and Didi, Kieran and Cao, Zhonglin and Reidenbach, Danny and Zhang, Zuobai and Dallago, Christian and Kucukbenli, Emine and Kreis, Karsten and Vahdat, Arash},
  journal={arXiv preprint arXiv:2507.09466},
  year={2025}
}
```