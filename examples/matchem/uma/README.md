# UMA

UMA（Universal Materials Interaction Model）是面向材料与催化体系的通用机器学习原子间势示例模型，基于等变图神经网络构建，可用于原子结构的能量、受力预测，并支持 OC20、OC22、OC25、OMat、OMOL、ODAC、OMC 等多种材料与催化任务的微调训练与推理。

---

## 快速开始

**进入示例目录**

本示例位于 OneScience 仓库的 `examples/matchem/uma`，进入该目录后所有命令均相对于该目录执行：

```bash
cd examples/matchem/uma
```

**准备数据**

本仓库不内置训练数据。以下以 **OC20 微调**为例说明流程，OC22、OC25、OMat、OMOL、ODAC、OMC 等其他任务流程相同，只需替换 `--uma-task` 和数据路径即可。

方式一：从 ModelScope 下载（示例）

```bash
modelscope download --dataset OneScience/oc20 --local_dir ./data
```

方式二：使用集群共享数据（如已存在）

```bash
mkdir -p data/oc20
cp -r /path/to/s2ef_200k_uncompressed data/oc20/
cp -r /path/to/s2ef_val_id_uncompressed data/oc20/
```

**数据格式转换**

下载的原始数据通常是 `.extxyz` 文件，需要先用 `scripts/create_uma_finetune_dataset.py` 转换为 ASE-lmdb 格式，并计算 `elem_refs` 和 `normalizer_rmsd`。该脚本支持以下任务：

| 任务 | 说明 |
| --- | --- |
| `oc20` | 催化（示例） |
| `oc22` | 氧化物催化（仅限 1P2） |
| `oc25` | （电）催化（仅 1P2） |
| `omat` | 无机材料 |
| `omol` | 分子 + 聚合物 |
| `odac` | MOFs |
| `omc` | 分子晶体 |

以 OC20 为例：

```bash
python scripts/create_uma_finetune_dataset.py \
    --train-dir data/oc20/s2ef_200k_uncompressed \
    --val-dir data/oc20/s2ef_val_id_uncompressed \
    --uma-task oc20 \
    --regression-tasks ef \
    --output-dir data/oc20_finetune \
    --num-workers 8
```

转换后生成：

```text
data/oc20_finetune/
├── train/                 # ASE-lmdb 训练数据
├── val/                   # ASE-lmdb 验证数据
└── data/                  # 生成的数据配置 yaml
    └── uma_conserving_data_task_energy_force.yaml
```

然后用 `scripts/update_demo_config.py` 把生成的 `elem_refs`、`normalizer_rmsd` 和数据路径更新到 demo 配置文件：

```bash
python scripts/update_demo_config.py --demo-config demo/configs/oc20_ef_4dcu.yaml
```

`demo/run.sh` 会自动将仓库根目录作为 `ONESCIENCE_DATASETS_DIR`，因此配置文件中的相对路径会自动匹配。

**准备权重**

本仓库已包含旋转基文件 `weight/Jd.pt`。UMA 预训练 checkpoint（如 `uma-s-1p1_converted.pt`）需从 fairchem 官方仓库下载并按 UMA 格式转换后放到：

```text
weight/uma-s-1p1_converted.pt
```

- fairchem 官方仓库：https://github.com/facebookresearch/fairchem

`demo/run.sh` 和 `inference/` 下的示例脚本都会自动检测 `weight/Jd.pt` 并设置 `ONESCIENCE_UMA_JD_PATH`。

**预检（不启动训练）**

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml --dry-run
```

**运行样例训练**

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml
```

SLURM 提交：

```bash
bash demo/run.sh --config demo/configs/oc20_ef_4dcu.yaml --submit
```

训练完成后，输出目录中会生成实验子目录：

```text
demo/outputs/
├── oc20_ef_4dcu_YYYYmmdd_HHMMSS/
│   ├── config.yaml
│   ├── hydra_config.yaml
│   ├── train_merged.out
│   └── uma_finetune_runs/
```

## 常用训练参数

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--config` | `run.sh` 使用的 YAML 配置文件 | `demo/configs/oc20_ef_4dcu.yaml` |
| `--dry-run` | 仅生成训练命令和 Hydra 配置预览 | 调试用 |
| `--submit` | 生成并提交 SLURM 作业 | 集群训练使用 |
| `launch.num_gpus` | 单节点使用的 GPU/DCU 数量 | `4` |
| `data.dataset_name` | UMA 任务数据集名 | `oc20`、`oc22`、`oc25`、`omat`、`omol`、`odac`、`omc` |
| `data.train_dataset.splits.train.src` | 训练集 ASE-lmdb 目录 | `data/oc20_finetune/train` |
| `data.val_dataset.splits.val.src` | 验证集 ASE-lmdb 目录 | `data/oc20_finetune/val` |
| `runner.train_eval_unit.model.checkpoint_location` | 微调 checkpoint 路径 | `weight/uma-s-1p1_converted.pt` |
| `epochs` | 训练轮数 | `1` |
| `batch_size` | 每卡 batch 大小 | `2` |
| `evaluate_every_n_steps` | 验证间隔步数 | `100` |

---

## 数据格式

UMA 微调支持以 `.extxyz` 作为多种任务的原始输入，但需要先用 `scripts/create_uma_finetune_dataset.py` 转换为 ASE-lmdb 格式。支持的 `--uma-task` 包括 `oc20`、`oc22`、`oc25`、`omat`、`omol`、`odac`、`omc`。

以 OC20 为例：

```bash
python scripts/create_uma_finetune_dataset.py \
    --train-dir data/oc20/s2ef_200k_uncompressed \
    --val-dir data/oc20/s2ef_val_id_uncompressed \
    --uma-task oc20 \
    --regression-tasks ef \
    --output-dir data/oc20_finetune \
    --num-workers 8
```

转换后的目录结构如下：

```text
data/oc20_finetune/
├── train/
│   ├── data.0000.aselmdb
│   ├── ...
├── val/
│   ├── data.0000.aselmdb
│   ├── ...
└── data/
    └── uma_conserving_data_task_energy_force.yaml
```

每个 lmdb 目录包含 ASE 原子的序列化数据。`data.elem_refs` 和 `data.normalizer_rmsd` 需与数据生成脚本输出一致，可通过 `scripts/update_demo_config.py` 自动同步到 demo 配置文件。

---

## 引用与许可证

- UMA 示例代码来自 OneScience 项目中的 matchem 示例实现，并参考了上游 fairchem 项目（https://github.com/facebookresearch/fairchem）。上游 fairchem 仓库软件以 [MIT License](https://fair-chem.github.io/core/install.html#license) 发布；fairchem 各模型 checkpoint 和数据集可能带有各自独立的许可证，使用时请遵循对应说明。
- 如果在科研工作中使用 UMA 微调或推理结果，建议引用 UMA/相关通用材料相互作用模型方法、fairchem/OneScience 相关项目信息和实际使用的数据集来源。
