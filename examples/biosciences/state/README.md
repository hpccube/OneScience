<p align="center">
  <strong>
    <span style="font-size: 30px;">STATE</span>
  </strong>
</p>

# 模型介绍

STATE 是 Arc Institute 提出的单细胞虚拟细胞模型体系，用于学习细胞状态表示并预测细胞在基因、药物或细胞因子扰动后的响应。STATE 包含两条可以独立使用、也可以串联的模型主线：

- **State Embedding（SE）**：把单细胞基因表达数据编码为低维细胞表征，默认写入 `adata.obsm["X_state"]`。
- **State Transition（ST）**：根据对照细胞、扰动条件、细胞类型和批次等协变量，预测扰动后的细胞状态或基因表达。

ST 又分为两种输入空间：

```text
原始单细胞 h5ad
├── HVG 预处理 → X_hvg   → ST-HVG → 扰动响应预测
└── SE-600M    → X_state → ST-SE  → 扰动响应预测
```

本模型包优先整理并验证以下三条运行主线：

1. `SE-600M + SE-167M-Human-smoke → X_state`
2. `Replogle + ST-HVG-Replogle → 训练/预测/推理`
3. `Replogle + SE-600M + ST-SE-Replogle → 完整联合推理`

原始论文：Predicting cellular responses to perturbation across diverse contexts with State
https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2

# 仓库说明

当前支持能力：

- 使用 `SE-167M-Human-smoke` 完成 SE profile、短训练和 SE-600M transform 验证
- 使用 `State-Tahoe-Filtered-smoke` 设计 Tahoe 数据加载、ST-HVG/ST-SE 推理和短训练验证
- 使用全量 `Replogle-Nadig-Preprint` 执行 ST-HVG 预处理、训练、预测和推理
- 使用全量 `Replogle-Nadig-Preprint` 执行 SE-600M → ST-SE 联合训练、预测和推理
- 使用全量 `State-Parse-Filtered` 和官方 split TOML 规划 Parse 论文训练与预测复现
- 使用 SE-600M 为新 h5ad 生成 `X_state`
- 通过通用 runner 替换 Tahoe、Parse、Replogle 权重和数据
- few-shot / zero-shot Replogle 官方划分
- ST checkpoint 的 `final.ckpt → best.ckpt → last.ckpt` fallback

这里的“全量复现”表示数据规模和官方划分资源已具备；严格复现论文还必须使用对应权重目录保存的模型结构、训练参数、随机种子和 split 配置，并记录实际硬件与依赖版本。

当前不支持能力：

- `State-Tahoe-Filtered-smoke` 只有三个 smoke 文件，不能复现 Tahoe 全量训练和完整官方划分。
- `SE-167M-Human-smoke` 只用于 profile 和训练连通性验证，不等价于 167M 完整预训练数据。
- Parse 数据约 343 GB，不应直接使用当前内存式 HVG 预处理脚本处理全量文件。
- 当前没有面向 Tahoe 和 Parse 的一键式专用 Shell；需要使用通用 runner 和本地渲染的 TOML。
- 当前没有在本仓库中重新完成 SE-600M、Replogle 或 Parse 的全规模收敛训练。
- 当前不保证用户自定义数据可直接复用官方 ST 权重；输入维度、基因顺序、扰动和协变量映射必须与权重一致。
- ST 推理必须保留整个模型 run 目录，不能只提供单独的 `.ckpt`。

# 使用场景

| 场景 | 说明 |
| :---: | :--- |
| SE 细胞表征 | 使用 SE-600M 将任意兼容 h5ad 转换为 `X_state` |
| SE 训练验证 | 使用 SE-167M-Human-smoke 构建 profile 并完成短训练 |
| CRISPR 基因扰动预测 | 使用 Replogle 数据和 ST-HVG/ST-SE 权重预测基因敲低响应 |
| 药物扰动预测 | 使用 Tahoe 数据和 Tahoe 权重预测药物×剂量响应 |
| 免疫刺激预测 | 使用 Parse PBMC 数据和 Parse 权重预测 cytokine 响应 |
| few-shot 评估 | 在同一细胞类型中留出部分扰动进行测试 |
| zero-shot 评估 | 留出完整细胞类型、细胞系或 donor 进行测试 |
| 新数据推理 | 使用 `infer_transition.py` 对不在训练 TOML 中的新 h5ad 推理 |
| 配置数据预测 | 使用 `predict_transition.py` 在 TOML 定义的数据划分上预测并运行 cell-eval |

# 文件说明

| 路径 | 功能 | 备注 |
| :--- | :--- | :--- |
| `README.md` | OneScience STATE 使用说明 | 三条主线和资产替换说明 |
| `STATE_README_ZH.md` | 原 STATE 仓库、数据和权重说明 | 包含 7 条官方标准路线 |
| `configs/paths.yaml` | 外部资产根路径 | 使用 `State` 和 `State_dataset` |
| `configs/weights.yaml` | 权重 registry | 当前登记 SE-600M、ST-HVG-Replogle、ST-SE-Replogle |
| `configs/datasets/assets.yaml` | 数据 registry | 登记 Replogle、SE smoke、Parse 和 Tahoe smoke |
| `configs/transition/` | ST Hydra 配置 | 模型、训练、数据和 wandb 配置 |
| `configs/embedding/state-defaults.yaml` | SE 默认配置 | profile 构建和训练入口 |
| `configs/datasets/legacy_replogle_splits/` | Replogle 官方划分 | hepg2、jurkat、k562、rpe1 的 few/zero-shot |
| `configs/datasets/legacy_preprint_splits/` | Parse 官方划分 | cell type、donor 和 split 配置 |
| `configs/datasets/legacy_tahoe_splits/` | Tahoe 官方划分 | generalization 配置 |
| `scripts/_state_common.sh` | Shell 公共路径和 checkpoint 解析 | 所有主线共享 |
| `scripts/runner/` | Python 原子入口 | 所有 `.py` 文件统一放置于此 |
| `scripts/se_smoke_*.sh` | SE smoke profile 和训练 | 主线一 |
| `scripts/se600m_smoke_transform.sh` | SE-600M transform | 主线一 |
| `scripts/st_hvg_replogle_*.sh` | Replogle ST-HVG 四阶段入口 | 主线二 |
| `scripts/st_se_replogle_*.sh` | Replogle ST-SE 五阶段入口 | 主线三 |
| `examples/random.h5ad` | 小型示例 AnnData | 仅用于轻量接口检查 |
| `outputs/` | 默认输出目录 | 不提交 Git |
| `licenses/state/` | STATE 许可证副本 | 位于 OneScience 仓库根目录 |

# 使用说明
## 1. 使用环境

在安装onescience库环境下执行。

## 2.默认数据库

```bash
ONESCIENCE_MODELS_DIR = /public/share/sugonhpcapp01/onestore/onemodels/
ONESCIENCE_DATASETS_DIR= /public/share/sugonhpcapp01/onestore/onedatasets/
```

## 3. 快速开始

### 3.1 设置资产路径

以下命令均从 `examples/biosciences/state` 目录执行。仓库根目录的 `env.sh` 提供站点默认模型和数据路径，输出无需设置环境变量，默认写入当前示例的 `outputs/`：

```bash
cd /path/to/onescience/examples/biosciences/state
```

仅当模型、数据或输出位于其他目录时才覆盖对应变量：

```bash
export ONESCIENCE_MODELS_DIR=/path/to/onemodels
export ONESCIENCE_DATASETS_DIR=/path/to/onedatasets
export STATE_OUTPUT_DIR=/path/to/state_outputs
```

所有 shell 入口都由 `_state_common.sh` 将输出默认设为 STATE 示例目录下的 `outputs/`；从本文要求的工作目录执行时，等价于 `${STATE_OUTPUT_DIR:-$PWD/outputs}`。直接调用 Python runner 时不会加载 `_state_common.sh`，因此本文的直接 Python 示例在参数中显式使用这个等价值。

### 3.2 默认训练数据来源

当前 README 中的默认训练均使用 `${ONESCIENCE_DATASETS_DIR}/State_dataset` 下的数据或由这些数据生成的中间文件，不使用 `examples/random.h5ad` 训练：

| 训练入口 | 原始数据来源 | 训练实际读取位置 | 类型 |
| :--- | :--- | :--- | :--- |
| `scripts/se_smoke_train.sh` | `State_dataset/SE-167M-Human-smoke/19k_human_filtered_scbasecount/*.h5ad` | `${STATE_OUTPUT_DIR:-$PWD/outputs}/se_smoke/profile` 及生成的 train/val CSV | smoke 训练验证 |
| `scripts/st_hvg_replogle_train.sh` | `State_dataset/Replogle-Nadig-Preprint/<cell-line>.h5ad` | `${STATE_OUTPUT_DIR:-$PWD/outputs}/st_hvg_replogle/data` 中预处理后的 `X_hvg` h5ad | 全量 Replogle 数据的单细胞系/划分训练 |
| `scripts/st_se_replogle_train.sh` | `State_dataset/Replogle-Nadig-Preprint/<cell-line>.h5ad` | `${STATE_OUTPUT_DIR:-$PWD/outputs}/st_se_replogle/data` 或 `data_all` 中 SE-600M 生成并预处理的 `X_state` h5ad | 全量 Replogle 数据的单细胞系/划分训练 |

输出目录中的文件是从 `State_dataset` 预处理得到的训练中间产物，不是另外的示例数据。运行训练前必须先执行对应主线的 preprocess 脚本。

另外两组现有数据默认不由专用 Shell 自动训练：

- `State-Tahoe-Filtered-smoke`：smoke 数据，使用通用 runner 和自定义 Tahoe smoke TOML 做短训练验证。
- `State-Parse-Filtered`：全量数据，使用通用 runner、官方 Parse split TOML 和对应完整配置做论文复现训练。

### 3.3 主线一：SE-600M + SE smoke

该主线验证 SE profile、SE 训练入口和预训练 SE-600M transform。

#### 步骤一：生成 train/val manifest 并构建 profile

```bash
bash scripts/se_smoke_preprocess.sh
```

默认输入：

```text
${ONESCIENCE_DATASETS_DIR}/State_dataset/SE-167M-Human-smoke/
```

默认输出：

```text
${STATE_OUTPUT_DIR:-$PWD/outputs}/se_smoke/manifests/
${STATE_OUTPUT_DIR:-$PWD/outputs}/se_smoke/profile/
${STATE_OUTPUT_DIR:-$PWD/outputs}/se_smoke/config.yaml
```

#### 步骤二：训练 SE smoke 模型

```bash
bash scripts/se_smoke_train.sh \
  experiment.num_epochs=1 \
  experiment.val_check_interval=1 \
  experiment.limit_val_batches=1 \
  model.batch_size=2
```

这一步用于验证训练接口，不用于复现官方 SE-600M。

#### 步骤三：使用 SE-600M 生成 `X_state`

```bash
bash scripts/se600m_smoke_transform.sh
```

默认输入和输出：

```text
输入：${ONESCIENCE_DATASETS_DIR}/State_dataset/SE-167M-Human-smoke/19k_human_filtered_scbasecount/SRX10188960.h5ad
输出：${STATE_OUTPUT_DIR:-$PWD/outputs}/se600m_smoke/SRX10188960_x_state.h5ad
```

替换输入输出：

```bash
export STATE_SE_INPUT=/path/to/input.h5ad
export STATE_SE_OUTPUT=/path/to/output_x_state.h5ad
bash scripts/se600m_smoke_transform.sh --batch-size 8
```

如需替换 SE 权重，直接使用通用 runner：

```bash
python scripts/runner/transform_embedding.py \
  --checkpoint /path/to/se_checkpoint.ckpt \
  --protein-embeddings /path/to/protein_embeddings.pt \
  --input /path/to/input.h5ad \
  --output /path/to/output_x_state.h5ad \
  --embed-key X_state
```

### 3.4 主线二：Replogle + ST-HVG

该主线使用 `X_hvg` 预测 CRISPR/基因扰动后的表达变化。

选择默认细胞系和划分：

```bash
export STATE_REPLOGLE_CELL_LINE=hepg2
export STATE_REPLOGLE_SPLIT_MODE=fewshot
```

这两个变量可不设置，默认分别为 `hepg2` 和 `fewshot`。可选细胞系：`hepg2`、`jurkat`、`k562`、`rpe1`；可选划分：`fewshot`、`zeroshot`。

#### 步骤一：生成 `X_hvg`

```bash
bash scripts/st_hvg_replogle_preprocess.sh
```

替换输入、输出和 HVG 数量：

```bash
export STATE_REPLOGLE_INPUT=/path/to/input.h5ad
export STATE_ST_HVG_INPUT=/path/to/input_x_hvg.h5ad
export STATE_NUM_HVGS=2000
bash scripts/st_hvg_replogle_preprocess.sh
```

#### 步骤二：短训练

```bash
bash scripts/st_hvg_replogle_train.sh \
  training.max_steps=10 \
  training.batch_size=2 \
  training.val_freq=5
```

#### 步骤三：使用下载权重预测并评估

```bash
bash scripts/st_hvg_replogle_predict.sh --profile minimal
```

#### 步骤四：对新 h5ad 推理

```bash
bash scripts/st_hvg_replogle_infer.sh
```

### 3.5 主线三：Replogle + SE-600M + ST-SE

该主线先用 SE-600M 生成 `X_state`，再使用 ST-SE-Replogle 预测基因扰动响应。

#### 步骤一：为 Replogle 生成 `X_state`

```bash
bash scripts/st_se_replogle_preprocess.sh
```

#### 步骤二：可选 ST-SE 短训练

```bash
bash scripts/st_se_replogle_prepare_all.sh

STATE_ST_SE_DATA_DIR="${STATE_OUTPUT_DIR:-$PWD/outputs}/st_se_replogle/data_all" \
bash scripts/st_se_replogle_train.sh \
  training.max_steps=10 \
  training.batch_size=2 \
  training.val_freq=5
```

#### 步骤三：使用 ST-SE-Replogle 预测并评估

使用下载的官方权重和上述流程生成的数据：

```bash
bash scripts/st_se_replogle_predict.sh --profile minimal
```

如果步骤二已经生成本地训练 run，也可以直接评估该 run。即使未设置 `STATE_OUTPUT_DIR`，以下命令仍会读取当前示例目录下的 `outputs/`：

```bash
python scripts/runner/predict_transition.py \
  --output-dir "${STATE_OUTPUT_DIR:-$PWD/outputs}/st_se_replogle/runs/st_se_replogle" \
  --checkpoint last.ckpt \
  --toml "${STATE_OUTPUT_DIR:-$PWD/outputs}/st_se_replogle/configs/train.toml" \
  --profile minimal
```

本地训练预测要求 `--output-dir` 指向完整 run 目录，其中必须存在 `config.yaml`、`data_module.torch`、映射文件和 `checkpoints/last.ckpt`。官方权重预测应使用 shell wrapper；wrapper 会在默认输出目录中创建运行时配置，避免官方 `config.yaml` 中的旧机器绝对路径影响当前用户。

#### 步骤四：完整联合推理

```bash
bash scripts/st_se_replogle_infer.sh
```

### 3.6 修改数据、权重和任务参数

替换 Replogle、Tahoe、Parse 或自有资产时，不需要修改 Python runner，统一在以下配置中调整：

| 要修改的内容 | 配置位置 | 主要字段 |
| :--- | :--- | :--- |
| 数据和权重根目录 | `configs/paths.yaml` | `State`、`State_dataset` |
| 数据集目录和文件 | `configs/datasets/assets.yaml` | 对应数据集的路径与文件名 |
| SE/ST 权重 | `configs/weights.yaml` | 模型 run 目录、checkpoint、protein embedding |
| ST 输入空间和数据字段 | `configs/transition/` 下的数据配置 | `embed_key`、`pert_col`、`cell_type_key`、`batch_col`、`control_pert` |
| train/val/test 划分 | `configs/datasets/legacy_*_splits/*.toml` | `[datasets]`、`[training]`、`[fewshot]`、`[zeroshot]` |
| 模型与训练参数 | `configs/transition/` 下的模型和训练配置 | batch size、训练步数、模型层数、随机种子等 |

三组数据对应的字段如下：

| 数据 | `pert_col` | `cell_type_key` | `batch_col` | `control_pert` |
| :--- | :--- | :--- | :--- | :--- |
| Replogle | `gene` | `cell_line` | `gem_group` | `non-targeting` |
| Tahoe | `drugname_drugconc` | `cell_name` | `plate` | `[('DMSO_TF', 0.0, 'uM')]` |
| Parse | `cytokine` | `cell_type_clean` | `donor` | `PBS` |

使用 ST-HVG 时将 `embed_key` 设为 `X_hvg`，使用 ST-SE 时设为 `X_state`。TOML 中的数据路径必须改为本机实际路径；ST 推理必须保留完整模型 run 目录，不能只替换单个 checkpoint。

### 3.7 smoke 验证与全量复现配置

smoke 验证和全量复现使用相同 runner，主要区别是数据、划分和训练规模。按下表修改配置即可：

| 场景 | 修改位置 | 需要确认的内容 |
| :--- | :--- | :--- |
| SE smoke | `configs/embedding/state-defaults.yaml`、`configs/datasets/assets.yaml` | smoke 数据目录、profile 输出、batch size、epoch |
| Tahoe smoke | `configs/datasets/assets.yaml`、本地 Tahoe smoke TOML、`configs/transition/` | `X_hvg`/`X_state`、实际 `cell_name`、train/val/test 划分、短训练步数 |
| Replogle 全量 | `configs/datasets/assets.yaml`、`configs/datasets/legacy_replogle_splits/`、`configs/transition/` | 四个细胞系文件、few-shot/zero-shot 划分、训练参数 |
| Parse 全量 | `configs/datasets/assets.yaml`、`configs/datasets/legacy_preprint_splits/`、`configs/transition/` | 全量数据路径、目标 split、`X_hvg`/`X_state`、训练参数 |

论文级复现时，再以对应权重 run 目录中的 `config.yaml` 为准，对齐模型结构、训练步数、随机种子和输出空间。Parse 全量数据较大，运行前需确认内存、显存和存储空间，并避免直接使用内存式脚本重新预处理整个文件。

# 数据格式

STATE 主要使用 AnnData：

```text
*.h5ad
```

ST 输入至少需要：

```text
adata.X                         基因表达矩阵
adata.obs[pert_col]             扰动标签
adata.obs[cell_type_key]        细胞类型或细胞系
adata.obs[batch_col]            donor、plate、gem_group 等批次
adata.obsm["X_hvg"]             ST-HVG 输入
adata.obsm["X_state"]           ST-SE 输入
```

不同数据的默认字段：

| 数据集 | 级别 | 生物领域 | 典型文件 | 关键字段与用途 |
| :--- | :--- | :--- | :--- | :--- |
| `Replogle-Nadig-Preprint` | 全量 | CRISPR/基因扰动 | HepG2、Jurkat、K562、RPE1 h5ad | `gene`、`cell_line`、`gem_group`；Replogle 论文复现 |
| `State-Tahoe-Filtered-smoke` | smoke | 药物×剂量扰动 | `c36.h5ad`、`c39.h5ad`、`c44.h5ad` | `drugname_drugconc`、`cell_name`、`plate`；训练/推理连通性验证 |
| `State-Parse-Filtered` | 全量 | PBMC cytokine 刺激 | `parse_concat_full.h5ad` | `cytokine`、`cell_type_clean`、`donor`；Parse 论文复现 |
| `SE-167M-Human-smoke` | smoke | 通用人类细胞表示学习 | 4 个 SRX h5ad | 基因表达和可识别基因名；SE 训练验证 |

SE preprocess 使用 `species,path,names` 三列 CSV：

```csv
species,path,names
human,/path/to/train_1.h5ad,train_1
human,/path/to/train_2.h5ad,train_2
```

SE 会从以下字段或 `var.index` 自动识别基因名：

```text
_index
gene_name
gene_symbols
feature_name
gene_id
symbol
```

ST 模型目录不能只包含 checkpoint，至少需要：

```text
<run>/
├── config.yaml
├── data_module.torch
├── var_dims.pkl
├── pert_onehot_map.pt
├── batch_onehot_map.pkl 或 batch_onehot_map.torch
├── cell_type_onehot_map.pkl 或 cell_type_onehot_map.torch
└── checkpoints/
    ├── final.ckpt
    ├── best.ckpt
    └── last.ckpt
```

# OneScience 官方信息

| 平台 | OneScience 主仓库 | Skills 仓库 |
| --- | --- | --- |
| Gitee | https://gitee.com/onescience-ai/onescience | https://gitee.com/onescience-ai/oneskills |
| GitHub | https://github.com/onescience-ai/OneScience | https://github.com/onescience-ai/oneskills |

# 引用与许可证

- STATE 原始论文：Predicting cellular responses to perturbation across diverse contexts with State。
- 论文地址：https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2
- 原始项目：https://github.com/ArcInstitute/state
- 原始 STATE 源码采用 CC BY-NC-SA 4.0 许可。
- STATE 模型权重和输出受 Arc Research Institute State Model Non-Commercial License 和 Acceptable Use Policy 约束。
- Parse 数据的部分用途可能需要 Parse Biosciences 许可，使用前应同时检查数据目录中的许可证文件。
- 在科研工作中使用本模型、权重或数据时，应引用 STATE 原始论文、对应数据集来源和 OneScience 项目信息。
