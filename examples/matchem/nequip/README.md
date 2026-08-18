# NequIP

## 模型介绍

NequIP 是基于 E(3) 等变图神经网络的原子间势模型，可用于材料结构的能量、原子力和应力预测。OneScience 提供从头训练、预训练模型微调和 ASE calculator 接口，并包含单点计算、能量-体积曲线和结构弛豫示例。

OAM-L 模型位于 `${ONESCIENCE_MODELS_DIR}/NequIP/`：

| 文件 | 作用 |
| --- | --- |
| `NequIP-OAM-L-0.1.nequip.zip` | 原始 NequIP package，用于 OAM-L 微调和微调 checkpoint 加载 |
| `NequIP-OAM-L-0.1.nequip.pth` | 编译后的 OAM-L 模型，用于 ASE 推理 |

从仓库根目录加载 MatChem 环境并进入示例目录：

```bash
source examples/matchem/matchem_env.sh
cd examples/matchem/nequip
```

## 目录结构

```text
examples/matchem/nequip/
├── README.md
├── train.py
├── single_point.py
├── energy_volume.py
├── structure_relaxation.py
└── demo/
    ├── run.sh
    ├── _parse_config.py
    ├── download_tutorial_data.py
    ├── reference_data/smoke.xyz
    └── configs/
        ├── tutorial_smoke_8dcu.yaml
        ├── tutorial_fcu_8dcu.yaml
        ├── oam_l_finetune_smoke.yaml
        └── oam_l_finetune.yaml
```

| 文件 | 作用 |
| --- | --- |
| `train.py` | NequIP 训练和测试入口 |
| `single_point.py` | ASE 单点能量、原子力和应力计算 |
| `energy_volume.py` | Si 能量-体积曲线计算 |
| `structure_relaxation.py` | ASE 原子位置和晶胞弛豫 |
| `demo/run.sh` | YAML 驱动的本地或 Slurm 训练入口 |
| `demo/download_tutorial_data.py` | 下载并校验官方 `fcu.xyz` 教程数据 |
| `tutorial_smoke_8dcu.yaml` | 单节点 8 DCU DDP smoke 配置 |
| `tutorial_fcu_8dcu.yaml` | 单节点 8 DCU 官方教程训练配置 |
| `oam_l_finetune_smoke.yaml` | OAM-L 单 batch 微调验证配置 |
| `oam_l_finetune.yaml` | OAM-L 正式微调模板 |

## 训练

官方教程数据默认位于：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/NequIP/fcu.xyz
```

数据不存在时可下载并校验：

```bash
python demo/download_tutorial_data.py
```

提交单节点 8 DCU DDP smoke 测试：

```bash
bash demo/run.sh \
  --config configs/tutorial_smoke_8dcu.yaml
```

使用完整 `fcu.xyz` 数据提交单节点 8 DCU 训练：

```bash
bash demo/run.sh \
  --config configs/tutorial_fcu_8dcu.yaml
```

两个 8 DCU 配置使用 `launch.mode: auto`：当前资源满足配置时直接运行，否则自动提交新的 Slurm 作业。训练输出保存在 `outputs/<name>_<timestamp>/`，包括配置快照、Slurm 日志以及 `checkpoints/best.ckpt` 和 `checkpoints/last.ckpt`。多卡配置中的 DataLoader `batch_size` 是每个 rank 的 batch size。

## 微调

OAM-L smoke 配置使用仓库内置 Cu 数据，完成一个 train、validation 和 test batch，只用于验证微调链路：

```bash
bash demo/run.sh \
  --config configs/oam_l_finetune_smoke.yaml \
  --submit
```

正式微调不附带通用数据集，用户必须提供与目标体系和计算设置一致的真实标注数据。配置默认读取：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/NequIP/oam_l_finetune.xyz
```

数据应为 ASE 可读的 extxyz 轨迹，每帧至少包含 `energy` 和 `forces` 标签，且所有元素均属于 OAM-L 支持的元素集合。提交正式微调：

```bash
bash demo/run.sh \
  --config configs/oam_l_finetune.yaml \
  --submit
```

可在 YAML 中调整数据切分、batch size、学习率、训练轮数和 Slurm 资源。获取最近一次 smoke 微调目录并检查 checkpoint：

```bash
RUN_DIR=$(find outputs -maxdepth 1 -type d \
  -name 'nequip_oam_l_finetune_smoke_*' | sort | tail -n 1)

test -s "$RUN_DIR/checkpoints/best.ckpt"
test -s "$RUN_DIR/checkpoints/last.ckpt"
```

## 推理

推理脚本默认加载 `${ONESCIENCE_MODELS_DIR}/NequIP/NequIP-OAM-L-0.1.nequip.pth`。单点计算默认使用周期性 bulk Cu：

```bash
python single_point.py
```

读取 ASE 支持的结构文件：

```bash
python single_point.py \
  --input structure.cif \
  --output outputs/single_point.json
```

Si 能量-体积曲线：

```bash
python energy_volume.py
```

Si 结构弛豫：

```bash
python structure_relaxation.py \
  --fmax 0.05 \
  --steps 100 \
  --output-dir outputs/oam_l_relax
```

结构弛豫默认同时优化原子位置和晶胞；添加 `--fixed-cell` 可固定晶胞。三个脚本均支持通过 `--compiled-model` 指定其他兼容的编译模型。

使用微调产生的 checkpoint 执行单点推理：

```bash
python single_point.py \
  --checkpoint "$RUN_DIR/checkpoints/best.ckpt" \
  --package "$ONESCIENCE_MODELS_DIR/NequIP/NequIP-OAM-L-0.1.nequip.zip" \
  --output "$RUN_DIR/single_point.json"
```
