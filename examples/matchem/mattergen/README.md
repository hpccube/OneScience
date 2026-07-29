# MatterGen 训练与推理示例

本目录提供 MatterGen 晶体生成模型的训练、属性微调和推理示例。MatterGen 已集成到 OneScience MatChem，模型实现、数据处理和扩散模块位于 `src/onescience/`，示例入口和用户配置保留在本目录。

## 模型简介

MatterGen 是面向无机晶体材料设计的扩散生成模型。模型联合生成元素组成、晶胞和周期性原子坐标，可进行无条件晶体生成，也可通过属性条件生成满足化学体系、空间群、带隙、磁密度、体积模量等目标的候选材料。生成结构可以继续使用 MatterSim 做结构弛豫和稳定性评估。

## 前置条件

1. 已安装 OneScience MatChem，并能加载 `matchem_env.sh`。
2. 已准备 DTK/DCU 运行时和对应的 PyG 扩展。
3. 已准备预训练模型和 MP-20 数据集。模型和数据通过 `ONESCIENCE_MODELS_DIR`、`ONESCIENCE_DATASETS_DIR` 查找。
4. 训练和多卡推理需要在 Slurm 计算节点执行；单卡任务可通过 `HIP_VISIBLE_DEVICES` 选择 DCU。

从仓库根目录加载统一环境变量：

```bash
source examples/matchem/matchem_env.sh
```

## 目录说明

```
mattergen/
├── train.py                  # 从头训练入口
├── finetune.py               # 属性微调入口
├── generate.py               # 晶体生成入口
├── csv_to_dataset.py         # CSV 转 MatterGen cache
├── submit_train.sh           # 直接提交 Slurm 训练的高级入口
├── demo/run.sh               # YAML 统一运行和 Slurm 提交入口
├── conf/                     # Hydra 训练、微调和扩散配置
└── outputs/                  # 本目录下生成的训练、微调和推理结果
```

## 预训练模型和数据

预训练模型目录包含：

```text
mattergen/
├── mattergen_base/
├── mp_20_base/
├── chemical_system/
├── space_group/
├── dft_mag_density/
├── dft_band_gap/
├── ml_bulk_modulus/
├── dft_mag_density_hhi_score/
└── chemical_system_energy_above_hull/
```

MP-20 数据建议使用已经生成的 cache：

```text
mattergen/
├── raw/mp_20/{train,val,test}.csv
└── cache/mp_20/{train,val,test}/
```

训练和微调直接读取 `cache/mp_20`。只有新增属性或准备自定义 CSV 数据时，才需要运行 `csv_to_dataset.py`。

## 快速开始

所有任务通过 `demo/run.sh` 读取 YAML 配置，用户不需要手动编写较长的 Hydra 命令：

```bash
cd examples/matchem/mattergen/demo
```

### 晶体生成

配置文件：`configs/generate_base.yaml`

```bash
bash run.sh --config configs/generate_base.yaml
```

可修改 `checkpoint`、`output`、`batch_size` 和 `num_batches`。输出目录包含 `generated_crystals.extxyz` 和 `generated_crystals_cif.zip`。属性条件模型需要同时配置匹配的 `properties`，可参考下面的磁密度示例。

例如磁密度条件生成：

```bash
bash run.sh --config configs/generate_dft_mag_density.yaml
```

条件值在配置文件的 `properties` 中修改。

### 从头训练

配置文件：`configs/train_8dcu.yaml`

```bash
# 提交 Slurm 训练任务；设备数从 YAML 的 trainer.devices 读取
bash run.sh --config configs/train_8dcu.yaml --submit
```

该配置默认使用 8 张 DCU、每卡 batch 4、梯度累积 16，等效全局 batch 为 512。显存不足时减小 `data_module.batch_size.*`，并相应增大 `trainer.accumulate_grad_batches`。

也可以直接修改 `submit_train.sh` 顶部的资源参数或脚本中的训练默认值后提交：

```bash
cd examples/matchem/mattergen
sbatch submit_train.sh
```

通过 `run.sh --submit` 提交时，YAML 参数会覆盖 `submit_train.sh` 中对应的训练参数；直接执行 `sbatch submit_train.sh` 时，则使用脚本中的默认值。

### 属性微调

配置文件：`configs/finetune_dft_mag_density_smoke.yaml`

```bash
bash run.sh --config configs/finetune_dft_mag_density_smoke.yaml
```

该配置执行一个训练 batch 和一个验证 batch，用于确认微调流程。正式微调时复制配置文件，修改 `data_module.properties` 和属性 embedding，并移除 `trainer.limit_train_batches`、`trainer.limit_val_batches`，再设置所需的 `trainer.max_epochs`。

训练、微调和生成结果统一写入 MatterGen 示例目录的 `outputs/`。

### Demo YAML 可修改项

| 字段 | 作用 |
| --- | --- |
| `task` | 任务类型：`train`、`finetune` 或 `generate`。 |
| `checkpoint` | 生成或微调使用的模型目录。 |
| `data_module` | 数据集配置名称，例如 `mp_20`。 |
| `data_module.properties` | 微调目标属性。 |
| `trainer.devices` | 使用的 DCU 数量。 |
| `data_module.batch_size.*` | 每张 DCU 的训练、验证和测试 batch size。 |
| `trainer.accumulate_grad_batches` | 梯度累积步数。 |
| `trainer.max_epochs` | 最大训练轮数。 |
| `output` | 晶体生成结果目录。 |

配置中的 `${ONESCIENCE_MODELS_DIR}` 和 `${ONESCIENCE_DATASETS_DIR}` 由统一环境脚本提供。

`demo/configs/` 面向普通用户，只保留常用任务和关键参数；`conf/` 是 Hydra 的完整内部配置，适合需要修改扩散过程、GemNet 结构、优化器或数据变换的高级用户。两者不应同时修改同一项参数作为日常使用方式。

### 训练参数说明

`configs/train_8dcu.yaml` 中的主要字段对应关系如下：

| 配置字段 | 含义 | 常见修改方式 |
| --- | --- | --- |
| `data_module` | 数据集配置名称 | `mp_20` 或 `alex_mp_20`；自定义数据集需要先增加对应的 `conf/data_module/<name>.yaml`。 |
| `trainer.devices` | 每个节点使用的 DCU 数量 | 必须与可申请的 DCU 数量一致，提交入口会同步 Slurm 资源申请。 |
| `trainer.num_nodes` | 训练节点数 | 当前示例为单节点。多节点还需要额外配置分布式网络和 Slurm 资源。 |
| `data_module.batch_size.train` | 每张 DCU 的训练 batch size | 显存不足时减小，例如从 4 改为 2。 |
| `data_module.batch_size.val` / `test` | 每张 DCU 的验证/测试 batch size | 显存不足时一起减小。 |
| `trainer.accumulate_grad_batches` | 梯度累积次数 | 减小 batch 后增大它，以维持近似的全局 batch。 |
| `data_module.num_workers.*` | DataLoader CPU worker 数 | CPU 资源充足时可以增大，当前示例为 2。 |
| `trainer.max_epochs` | 最大训练轮数 | 训练时间和收敛目标的主要控制项。 |
| `~trainer.logger` | 删除 WandB logger | 保持为 `true` 可避免训练依赖 WandB 服务。 |

默认训练的有效全局 batch 计算为：

```text
每卡 batch × DCU 数量 × 节点数 × 梯度累积次数
 = 4 × 8 × 1 × 16 = 512
```

微调配置 `configs/finetune_dft_mag_density_smoke.yaml` 的关键字段含义如下：

| 配置字段 | 含义 |
| --- | --- |
| `data_module.properties` | 数据集中需要读取的目标属性，例如 `[dft_mag_density]`。 |
| `adapter.model_path` | 微调使用的预训练 checkpoint 目录。 |
| `+...property_embeddings@...` | 为该属性加载对应的 embedding 配置，名称必须与属性一致。 |
| `trainer.max_epochs` | 微调轮数；smoke test 使用 1。 |
| `trainer.limit_train_batches` / `limit_val_batches` | 每个 epoch 使用的 batch 数；设置为 1 只用于流程验证。 |
| `data_module.batch_size.train` / `val` | 微调的每卡 batch size。 |

正式微调时，应复制该 YAML，修改属性、checkpoint 和训练轮数，并移除两个 `limit_*_batches` 字段。

### 数据集支持范围

当前示例默认使用 MP-20 cache：

```text
data_module: mp_20
root_dir: ${ONESCIENCE_DATASETS_DIR}/matchem/mattergen/cache/mp_20
```

当前已经提供的第二个数据配置是 `alex_mp_20`。因此用户可以直接把训练 YAML 中的 `data_module: mp_20` 改为：

```yaml
data_module: alex_mp_20
```

目前不能仅通过填写任意 CSV 路径就直接训练。自定义数据需要先转换为 MatterGen cache，并至少包含：

```text
my_dataset/
├── train/
├── val/
└── test/
```

每个 split 需要由 `csv_to_dataset.py` 生成，且 CSV 中必须包含晶体结构字段以及训练目标属性。完成转换后有两种使用方式：

1. 复制 `conf/data_module/mp_20.yaml` 为 `conf/data_module/my_dataset.yaml`，修改 `root_dir`，然后将 Demo YAML 的 `data_module` 改为 `my_dataset`。
2. 对结构完全兼容 MP-20 的 cache，可继续使用 `data_module: mp_20`，并在 Demo YAML 增加：
   ```yaml
   data_module.root_dir: ${ONESCIENCE_DATASETS_DIR}/matchem/mattergen/cache/my_dataset
   ```

因此，当前版本支持“符合 MatterGen cache 契约的数据集”，但还没有提供用户只给一个任意目录就自动识别字段和生成 Hydra 配置的功能。

### 自定义数据

将 CSV 数据转换为 MatterGen cache：

```bash
python csv_to_dataset.py \
  --csv-folder /path/to/csv_folder \
  --dataset-name my_dataset \
  --cache-folder /path/to/mattergen/cache
```

然后在 `conf/data_module/` 中配置数据集路径、属性字段和 batch size。

## 配置文件

```text
conf/
├── default.yaml                         # 标准训练入口
├── finetune.yaml                        # 属性微调入口
├── csp.yaml                             # 固定组成的 CSP 训练入口
├── adapter/default.yaml                 # 微调 checkpoint 和 adapter
├── data_module/mp_20.yaml               # MP-20 数据集和 batch 配置
├── data_module/alex_mp_20.yaml          # Alex-MP-20 数据集配置
├── trainer/default.yaml                 # 设备、epoch、DDP 和 checkpoint
├── lightning_module/default.yaml        # 优化器和学习率调度器
├── lightning_module/diffusion_module/   # 扩散过程和 GemNet 配置
└── lightning_module/diffusion_module/model/property_embeddings/
    └── *.yaml                           # 各属性条件 embedding
```

常见修改位置：数据路径和 batch size 修改 `data_module/*.yaml`；设备数、epoch 和梯度累积修改 `trainer/default.yaml`；学习率修改 `lightning_module/default.yaml`；GemNet 规模和 cutoff 修改 `diffusion_module/model/mattergen.yaml`。

## 注意事项

- `run.sh --submit` 会申请 YAML 中配置的 DCU 数量；直接提交 `submit_train.sh` 时默认申请 8 张 DCU。提交前请确认队列和节点资源满足要求。
- 属性条件生成必须使用与属性 embedding 匹配的预训练模型。
- 训练结果、微调结果和生成结构均建议保存在本目录 `outputs/` 下，便于复现实验。
