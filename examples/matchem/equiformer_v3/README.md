# Equiformer V3

## 模型介绍

Equiformer V3 是用于材料结构能量、原子力和应力预测的 SE(3) 等变图注意力势模型。OneScience 提供 ASE calculator 接口，因此共享 checkpoint 可用于单点计算、形成能、弹性张量、声子以及独立数据集评估；本目录同时提供一个 OC20 S2EF 从头训练示例。

预训练权重位于 `${ONESCIENCE_MODELS_DIR}/EquiformerV3/`：

| checkpoint | 训练域与模型类型 | 适用场景 |
| --- | --- | --- |
| `mptrj_gradient.pt` | MPtrj gradient | 与 MPtrj 标注域匹配的评估或推理 |
| `omat24_direct.pt` | OMat24 direct + DeNS | 与 OMat24 direct 模型匹配的评估或推理 |
| `omat24_gradient.pt` | OMat24 gradient | OMat24 验证集评估；默认评估示例使用该权重 |
| `omat24-mptrj-salex_gradient.pt` | OMat24 + MPtrj + sAlex gradient | 通用材料 ASE 推理；四个推理脚本默认使用该权重 |

所有权重都依赖 Wigner 旋转基文件 `${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/Jd.pt`。如文件位于其他位置，可设置：

```bash
export ONESCIENCE_EQUIFORMER_V3_JD_PATH=/path/to/Jd.pt
```

从 OneScience 仓库根目录加载 MatChem 环境并进入示例目录：

```bash
export MATCHEM_CONDA_NAME=onescience311  # 按实际环境修改
source examples/matchem/matchem_env.sh
cd examples/matchem/equiformer_v3
```

## 目录结构

```text
examples/matchem/equiformer_v3/
├── README.md
├── single_point.py
├── formation_energy.py
├── elastic.py
├── phonons.py
├── evaluate.py
├── train.py
├── finetune.py
├── fit_element_references.py
└── demo/
    ├── run.sh
    ├── _parse_config.py
    ├── reference_data/
    │   └── oc20_subset_energy_element_references.npz
    └── configs/
        ├── oc20_scratch_8dcu.yaml
        └── oc20_scratch_8dcu_smoke.yaml
```

| 文件 | 作用 |
| --- | --- |
| `single_point.py` | 对单个 ASE 结构预测能量、原子力和应力 |
| `formation_energy.py` | 使用显式元素参考能计算未修正形成能 |
| `elastic.py` | 通过多组应变和模型应力拟合弹性张量 |
| `phonons.py` | 通过 ASE 有限位移法计算声子能带和态密度 |
| `evaluate.py` | 在 ASE DB/ASE-LMDB 上报告能量、原子力及可选应力指标 |
| `train.py` | 从头训练、checkpoint 初始化或恢复训练状态的底层入口 |
| `finetune.py` | 与现有调用方式兼容的训练入口 |
| `fit_element_references.py` | 从目标训练集拟合 energy 元素参考系数 |
| `demo/run.sh` | YAML 驱动的本地/Slurm 训练入口，并创建本次运行目录 |

## 推理

四个推理脚本默认加载 `${ONESCIENCE_MODELS_DIR}/EquiformerV3/omat24-mptrj-salex_gradient.pt`。单点、弹性和声子默认使用代码内置的周期性 bulk Cu；形成能默认使用代码内置的 MgO，因此复制下面的命令即可运行，不需要预先准备结构文件。

### 单点计算

对结构执行一次能量、原子力和应力预测，不更新原子位置或晶胞：

```bash
python single_point.py \
  --device cuda \
  --output outputs/single_point.json
```

JSON 输出包含能量（eV）、力（eV/Angstrom）、ASE Voigt 顺序应力（eV/Angstrom³）、晶胞和输入来源。

### 形成能

默认示例计算 MgO，并由同一模型计算 Mg(hcp) 与 O₂ 参考能：

```bash
python formation_energy.py \
  --device cuda \
  --output outputs/formation_energy.json
```

输出是未加 DFT 基准、气体或经验修正的形成能。添加 `--relax` 可在计算前弛豫化合物和模型参考相。

### 弹性张量

```bash
python elastic.py \
  --relax \
  --device cuda \
  --output outputs/elastic.json
```

脚本对周期结构施加法向和剪切应变，由模型应力拟合二阶弹性张量，并报告 Voigt、Reuss 和 Hill 模量。添加 `--relax-positions` 可在每个固定变形晶胞内弛豫原子位置。

### 声子

```bash
python phonons.py \
  --supercell 3 3 3 \
  --bandpath GXWKGL \
  --device cuda \
  --output-dir outputs/phonons
```

脚本使用 ASE 有限位移法计算声子能带和态密度。添加 `--relax` 可在有限位移前弛豫晶胞与原子位置。

### 使用自己的结构或权重

四个脚本均可通过 `--input` 读取 ASE 支持的结构文件，例如 CIF、POSCAR、XYZ 或 trajectory：

```bash
python single_point.py --input structure.cif
python elastic.py --input POSCAR --output outputs/elastic.json
python phonons.py --input structure.cif --supercell 2 2 2
```

计算自有化合物形成能时，还应提供覆盖结构中全部元素的可信参考能：

```bash
python formation_energy.py \
  --input compound.cif \
  --reference-energies element_references.json \
  --output outputs/formation_energy.json
```

`element_references.json` 是元素到每原子参考能（eV/atom）的 JSON/YAML 映射。内置 Mg/O 参考结构只适用于默认 MgO 示例。所有脚本均可用 `--checkpoint /path/to/checkpoint.pt` 替换默认权重；结构、参考能与 checkpoint 的训练域必须相互匹配。

## 模型评估

`evaluate.py` 可评估本地或共享目录中的任意兼容 checkpoint。它在线累计总能量与每原子能量 MAE/RMSE、自由原子 force 分量 MAE/RMSE、逐原子 force cosine similarity、force magnitude error，以及可选的 stress MAE/RMSE，不会把完整验证集预测保存在内存中。

以下示例使用 `omat24_gradient.pt` 评估匹配的 OMat24 验证集：

```bash
python evaluate.py \
  --checkpoint "${ONESCIENCE_MODELS_DIR}/EquiformerV3/omat24_gradient.pt" \
  --data "${ONESCIENCE_DATASETS_DIR}/matchem/omat24/val/rattled-300-subsampled" \
  --batch-size 1 \
  --max-samples 128 \
  --device cuda \
  --include-stress \
  --output outputs/omat24_gradient_eval_128.json
```

默认只统计 `fixed == 0` 的自由原子；使用 `--no-free-atoms-only` 可关闭过滤。OC20 S2EF 评估可添加 `--include-oc20-threshold`，报告 0.02 eV energy / 0.03 eV/Angstrom force 联合阈值成功率。

## 训练

### 示例数据集

当前训练示例使用处理好的 OC20 S2EF ASE-LMDB 数据：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/oc20/uma_oc20_finetune/
├── train/
└── val/
```

训练监督包含 energy 和 forces，不训练 stress。配置依赖 `demo/reference_data/oc20_subset_energy_element_references.npz` 中随当前训练数据拟合的 energy 元素参考系数；该文件是配置的一部分，不是原始数据集。

替换训练数据后，必须重新拟合参考系数：

```bash
python fit_element_references.py \
  --config demo/configs/oc20_scratch_8dcu.yaml \
  --output demo/reference_data/oc20_subset_energy_element_references.npz
```

### 训练执行

先运行 8 DCU smoke，检查数据、分布式通信、前向、反向、优化器更新和 checkpoint 保存链路：

```bash
bash demo/run.sh --config configs/oc20_scratch_8dcu_smoke.yaml
```

smoke 只使用 8 个训练样本和 8 个验证样本，采用一层缩小模型并执行一次更新，只用于链路验证，不代表正式模型规模、训练收敛或论文指标。

完整训练配置保留 Equiformer V3 OC20 模型规模，在 8 个 DCU 上运行 12 个 epoch：

```bash
bash demo/run.sh --config configs/oc20_scratch_8dcu.yaml
```

训练会在 `outputs/` 下创建带时间戳的运行目录，保存原始 YAML、解析后的训练配置、Slurm 脚本、日志和 checkpoint。完整配置不设置 `max_steps`、`max_train_samples` 或 `max_val_samples`，会遍历配置中的全部数据。

训练完成后，可评估本次 checkpoint：

```bash
RUN_DIR=$(find outputs -maxdepth 1 -type d \
  -name 'equiformer_v3_oc20_scratch_8dcu_*' | sort | tail -n 1)

python evaluate.py \
  --checkpoint "$RUN_DIR/checkpoints/equiformer_v3_oc20_scratch_8dcu.pt" \
  --data "${ONESCIENCE_DATASETS_DIR}/matchem/oc20/uma_oc20_finetune/val" \
  --include-oc20-threshold \
  --output "$RUN_DIR/val_metrics.json"
```

### 资源配置

当前 DCU 稳定路径为 FP32。控制实验中，相同完整模型的 20-step FP16/BF16 训练出现 DCU kernel VMFault，而 FP32 完成并保存 checkpoint，因此正式 YAML 设置 `amp: false`。

| 配置 | 资源 | 精度 | 模型 | epoch/step | train batch/DCU | 累积步数 | 有效全局 batch |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `oc20_scratch_8dcu.yaml` | 1 节点 × 8 DCU | FP32 | 完整 OC20 模型 | 12 epoch | 1 | 8 | 64 |
| `oc20_scratch_8dcu_smoke.yaml` | 1 节点 × 8 DCU | FP32 | 一层缩小模型 | 1 step | 1 | 1 | 8 |

所有 YAML 使用同一个入口；`launch.mode: local` 在当前节点执行，`launch.mode: submit` 提交 Slurm 作业。当前可见 DCU 不足或配置请求多节点时，`demo/run.sh` 也会自动提交。多卡训练使用 `torchrun`，只有 rank 0 写入 checkpoint。

| YAML 字段 | 作用 |
| --- | --- |
| `launch.mode` | `local` 直接执行；`submit` 生成并提交 Slurm 作业 |
| `launch.num_nodes`、`launch.num_gpus` | 节点数和每节点 DCU 数 |
| `launch.omp_num_threads` | 每个训练进程的 CPU 线程数 |
| `slurm.partition`、`slurm.time`、`slurm.cpus_per_task` | 队列、作业时限和每节点 CPU 资源 |
| `nccl` | 集群需要时配置网卡、IB HCA 和通信协议 |

`demo/run.sh` 在交互式 shell 中复用已经加载的环境；提交 Slurm 后，生成的作业脚本会在计算节点的新 shell 中重新加载 `examples/matchem/matchem_env.sh`。