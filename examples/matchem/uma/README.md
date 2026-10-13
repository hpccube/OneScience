# UMA 训练与推理示例

本目录包含 UMA（Universal Multi-task Approximator）多任务图神经网络模型的训练与推理示例。UMA 已包含在 `onescience[matchem]` 基础环境中，无需额外编译安装。

## 前置条件

1. 已完成 `onescience[matchem]` 基础环境安装。
2. 已加载统一环境：
   ```bash
   # 默认激活 test_pip 环境；若使用其他 conda 环境，请先设置：
   # export MATCHEM_CONDA_NAME=your_env
   source ../matchem_env.sh
   ```

## 目录说明

```
uma/
├── train.py                   # UMA 训练入口
├── demo/                      # 训练 demo 算例
│   ├── run.sh                 # 统一提交脚本
│   └── configs/               # 各算例 YAML 配置
│       ├── oc20_ef_4dcu.yaml
│       ├── oc20_ef_8dcu.yaml
│       └── oc20_ef_16dcu.yaml
├── inference/                 # 推理示例
│   ├── _common.py             # 默认模型、数据和通用参数
│   ├── batch_inference_with_dataloader.py
│   ├── inference_heterogeneous_batches.py
│   ├── relax_adsorbate_on_slab.py
│   ├── relax_inorganic_crystal.py
│   ├── run_molecular_md.py
│   └── calculate_spin_gap.py
├── configs/                   # 数据任务与微调模板
│   └── data/
│       ├── uma_conserving_data_task_energy.yaml
│       ├── uma_conserving_data_task_energy_force.yaml
│       └── uma_conserving_data_task_energy_force_stress.yaml
├── scripts/                   # 数据转换与模型转换工具
│   ├── create_finetune_dataset.py
│   ├── create_uma_finetune_dataset.py
│   └── convert_model.py
└── models/
    └── pretrained_models.json # 预训练模型索引
```

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
├── data/
│   └── uma_conserving_data_task_energy_force.yaml
└── uma_sm_finetune_template.yaml   # 可直接用于微调的配置文件
```

`create_uma_finetune_dataset.py` 会同时完成格式转换、计算 `elem_refs` 和
`normalizer_rmsd`，并生成已填充的数据配置和微调配置；通过
`scripts/update_demo_config.py` 可将数据配置同步到 demo 配置文件。

> **提示**：数据处理完成后，生成的 `uma_sm_finetune_template.yaml` 可直接拿来微调，无需再手动编写配置。

如果训练集和验证集已经是 ASE-LMDB 格式，则无需再次执行转换脚本，直接使用
配套 YAML 即可；如果没有配套 YAML，则需要补齐 `dataset_name`、`elem_refs`、
`normalizer_rmsd` 和 train/val 路径。当前转换脚本不能直接以 ASE-LMDB 作为输入。

## 快速开始

### 训练

```bash
cd demo
bash run.sh --config configs/oc20_ef_4dcu.yaml
bash run.sh --config configs/oc20_ef_8dcu.yaml
bash run.sh --config configs/oc20_ef_16dcu.yaml --submit
```

4 卡和 8 卡命令在当前节点直接运行；16 卡为两节点任务，通过 `--submit`
提交 Slurm 作业。4 卡和 8 卡也可以添加 `--submit` 申请 Slurm 资源。

### 推理：弛豫无机晶体

```bash
cd inference
python relax_inorganic_crystal.py
```

默认构造 Fe 体相，并从
`${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/uma-s-1p1_converted.pt` 加载模型。
传入自己的周期结构和 checkpoint：

```bash
python relax_inorganic_crystal.py \
  --model-path /path/to/uma_checkpoint.pt \
  --structure /path/to/structure.cif \
  --output-dir outputs/relax
```

### 推理：吸附物在 slab 上弛豫

```bash
cd inference
python relax_adsorbate_on_slab.py
```

默认构造 CO/Cu(100)；也可以传入自己的 slab、吸附物和模型：

```bash
python relax_adsorbate_on_slab.py \
  --model-path /path/to/uma_checkpoint.pt \
  --slab /path/to/slab.xyz \
  --adsorbate /path/to/adsorbate.xyz \
  --output outputs/relaxed_slab.xyz
```

其余推理示例也都使用相同的 `--model-path` 和 `--device` 参数：

```bash
python generate_batches_at_runtime.py
python inference_heterogeneous_batches.py
python calculate_spin_gap.py
python run_molecular_md.py
python batch_inference_with_dataloader.py
```

批量推理默认读取共享 OMat24 验证样例
`${ONESCIENCE_DATASETS_DIR}/matchem/omat24/val/rattled-300-subsampled/val.aselmdb`；
可用 `--data-path` 替换为其他 ASE-LMDB 文件或目录。

未传入结构或数据参数时，这些脚本使用内置示例；用户可以按需覆盖：

```bash
python generate_batches_at_runtime.py \
  --structure /path/to/structure_1.cif \
  --structure /path/to/structure_2.xyz \
  --task-name omat

python run_molecular_md.py \
  --structure /path/to/molecule.xyz \
  --steps 100

python batch_inference_with_dataloader.py \
  --data-path /path/to/ase_lmdb_or_directory \
  --batch-size 8 \
  --max-batches 2
```

## 注意事项

- UMA 推理脚本默认从 `${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/` 读取
  `uma-s-1p1_converted.pt`；`--model-path` 可以覆盖默认位置。
- 所有 `run.sh` 已通过 `matchem_env.sh` 加载环境。
- 默认 conda 环境为 `test_pip`，可通过 `MATCHEM_CONDA_NAME` 覆盖。
- 队列名 `#SBATCH --partition=hx1hdexclu12` 为本集群配置，迁移时需修改。
