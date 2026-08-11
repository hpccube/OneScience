# eSEN

## 模型介绍

eSEN（equivariant Smooth Energy Network）是用于材料结构能量、原子力和应力预测的等变图神经网络势。OneScience 提供 ASE calculator 接口，因此同一模型可用于单点计算、结构弛豫、分子动力学和 checkpoint 微调。

预训练权重位于 `${ONESCIENCE_MODELS_DIR}/eSEN/`：

| 权重 | 训练域 | 适用场景 |
| --- | --- | --- |
| `esen_30m_mptrj.pt` | MPTrj | 无机晶体 PBE/PBE+U 能量、力、应力；默认微调权重 |
| `esen_30m_omat.pt` | OMat24 | 更广的无机非平衡结构 |
| `esen_30m_oam.pt` | OAM | 通用材料预训练权重 |

所有 eSEN 权重都依赖共享旋转基文件 `${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/Jd.pt`。该文件与 UMA 共用，不需要复制到 `eSEN/` 目录。

从仓库根目录加载 MatChem 环境并进入示例目录：

```bash
source examples/matchem/matchem_env.sh
cd examples/matchem/esen
```


## 目录结构

```text
examples/matchem/esen/
├── README.md
├── single_point.py
├── relax.py
├── md.py
├── finetune.py
├── evaluate.py
├── prepare_oxide_dataset.py
└── demo/
    ├── run.sh
    ├── _parse_config.py
    ├── configs/finetune_1dcu.yaml
    ├── configs/finetune_2dcu.yaml
    ├── configs/finetune_16dcu.yaml
    └── configs/finetune_16dcu_smoke.yaml
```

| 文件 | 作用 |
| --- | --- |
| `single_point.py` | eSEN 单点能量、力、应力计算示例 |
| `relax.py` | 使用 ASE BFGS 弛豫周期结构 |
| `md.py` | 使用 ASE Langevin 运行 NVT 分子动力学 |
| `finetune.py` | 从预训练 checkpoint 开始微调的底层执行入口 |
| `evaluate.py` | 在独立 ASE DB/ASE-LMDB 上报告物理单位的能量、力和应力误差 |
| `prepare_oxide_dataset.py` | 将 FairChem 官方 oxide JSON 转为 ASE DB 训练集、验证集和测试集 |
| `demo/run.sh` | YAML 驱动的本地/Slurm 微调入口；负责创建本次训练输出目录 |
| `demo/configs/finetune_1dcu.yaml` | 单 DCU 的默认 oxide PBE 微调配置 |
| `demo/configs/finetune_2dcu.yaml` | 1 节点 x 2 DCU 的 YAML 启动配置 |
| `demo/configs/finetune_16dcu.yaml` | 2 节点 x 8 DCU 的 Slurm 多节点微调配置 |
| `demo/configs/finetune_16dcu_smoke.yaml` | 2 节点 x 8 DCU 的端到端验证配置，不用于正式训练 |

## 推理

三个推理脚本默认加载 `${ONESCIENCE_MODELS_DIR}/eSEN/esen_30m_mptrj.pt`。单点计算和结构弛豫默认使用金刚石 Si 晶胞，MD 默认使用其 `2 x 2 x 2` 超胞。

单点计算：

```bash
python single_point.py
```

结构弛豫：

```bash
python relax.py \
  --fmax 0.05 --steps 100 --output relaxed.cif
```

弛豫默认同时优化原子位置与晶胞；添加 `--fixed-cell` 可固定晶胞，只优化原子位置。

NVT 分子动力学：

```bash
python md.py \
  --steps 100 --temperature 300 --timestep 1.0 --output md.traj
```

三个脚本均可通过 `--input` 读取 ASE 支持的结构文件，例如：

```bash
python single_point.py --input structure.cif
python relax.py --input POSCAR --output relaxed.cif
python md.py --input structure.cif --repeat 2 2 2 --steps 1000
```

可用 `--checkpoint /path/to/checkpoint.pt` 替换为兼容的预训练或微调 checkpoint。对于其他工作流，可直接使用 `onescience.utils.esen.eSENCalculator` 绑定 ASE `Atoms.calc`。

## 微调

### 示例数据集

默认微调示例使用已处理的 FairChem 官方 oxide PBE 数据，位于：

```text
${ONESCIENCE_DATASETS_DIR}/matchem/esen/oxides/
├── train.db      # 238 个结构
├── val.db        # 28 个结构
├── test.db       # 29 个结构
└── manifest.json
```

该数据包含 5 种氧化物、295 个结构，均带 energy、forces 和 stress 标签；默认 YAML 已指向其中的 `train.db` 与 `val.db`。示例只使用 energy 和 forces 监督，stress 标签保留用于独立诊断。`prepare_oxide_dataset.py` 用于从官方原始 JSON 可复现地生成这套数据。

### 微调执行

默认配置使用 `esen_30m_mptrj.pt`，训练 energy 和 forces：

```bash
bash demo/run.sh \
  --config configs/finetune_1dcu.yaml
```

在 [demo/configs/finetune_1dcu.yaml](demo/configs/finetune_1dcu.yaml) 中可直接修改：

| 字段 | 作用 |
| --- | --- |
| `checkpoint` | 初始化权重路径 |
| `train`、`val` | ASE DB 或 ASE-LMDB 训练、验证数据路径 |
| `output` | 微调 checkpoint 输出路径 |
| `epochs`、`batch_size`、`workers` | 训练轮数、批大小、DataLoader worker 数 |
| `max_train_samples`、`max_val_samples` | 固定 seed 随机抽取的训练和验证样本数；不设置时使用完整数据 |
| `lr` | AdamW 学习率 |
| `energy_weight`、`force_weight` | energy 与 forces MSE 损失项的权重 |
| `stress_weight` | stress 损失权重；当前氧化物示例为 `0.0`，不训练应力 |
| `fit_element_references` | 是否用训练集重新拟合 energy 元素参考能；跨数据集时应在独立验证集上比较后再开启 |

训练会逐 epoch 输出 JSON 格式的训练和验证损失，并在 `outputs/` 下创建带时间戳的运行目录，其中包含本次配置和最终 checkpoint。日志中的 energy 和 forces 分项均为 checkpoint 归一化空间中的 MSE，其中 energy 使用逐原子误差，避免不同晶胞大小造成权重偏差。训练使用预训练 checkpoint 的统一 normalizer；元素参考能是否写入微调 checkpoint 由 `fit_element_references` 控制。

训练完成后，将 `RUN_DIR` 设置为本次生成的运行目录，即可用微调权重执行推理：

```bash
RUN_DIR=$(find outputs -maxdepth 1 -type d -name 'esen_oxides_1dcu_*' | sort | tail -n 1)
python single_point.py \
  --checkpoint "$RUN_DIR/checkpoints/esen_oxides_1dcu_finetuned.pt"
```

使用未参与训练的 `test.db` 评估微调 checkpoint：

```bash
python evaluate.py \
  --checkpoint "$RUN_DIR/checkpoints/esen_oxides_1dcu_finetuned.pt" \
  --data "${ONESCIENCE_DATASETS_DIR}/matchem/esen/oxides/test.db" \
  --output "$RUN_DIR/test_metrics.json"
```

该示例 checkpoint 只针对 energy 和 forces 微调。它适合单点计算、固定晶胞结构弛豫和 NVT MD；变胞弛豫、弹性常数和 NPT MD 应使用包含充分应变覆盖与可靠 stress 标签的数据另行微调。

### 资源配置

所有 YAML 使用同一个入口；只需替换配置文件名即可切换单卡、多卡或多节点资源。`local` 在当前计算节点直接运行，`submit` 自动提交 Slurm 作业；若当前可见 DCU 数不足或配置请求多节点，脚本也会自动提交。多卡训练使用 `torchrun`，只有 rank 0 写入 checkpoint。

| YAML 字段 | 作用 |
| --- | --- |
| `launch.mode` | `local` 直接执行；`submit` 自动生成并提交 Slurm 作业 |
| `launch.num_nodes`、`launch.num_gpus` | 节点数和每节点 DCU 数 |
| `launch.omp_num_threads` | 每个训练进程的 CPU 线程数 |
| `slurm.partition`、`slurm.time`、`slurm.cpus_per_task` | 队列、作业时限和每节点 CPU 资源 |
| `nccl` | 集群需要时配置网卡、IB HCA 和通信协议 |

## 注意事项

1. `Jd.pt` 对推理、弛豫、MD 和微调都必需。加载 `matchem_env.sh` 后，源码会自动从 `${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/Jd.pt` 查找它；若文件位于其他位置，设置 `ONESCIENCE_ESEN_JD_PATH=/path/to/Jd.pt`。

2. 当前微调入口支持带 energy、forces、可选 stress 标签的 ASE DB 和 ASE-LMDB。原始 `.xyz`、`.extxyz` 应先转换为 ASE DB/ASE-LMDB，转换操作可见uma目录下的readme。

3. 单节点多卡微调使用 `torchrun` 和 NCCL；多节点多卡要求 YAML 中设置 `launch.mode: submit`。训练节点应已正确加载 DTK/DCU 运行时。

4. 开启 `fit_element_references` 后，微调 checkpoint 的绝对能量参考针对训练集中出现的元素重新拟合；该 checkpoint 应用于相同元素域，跨元素体系需准备相应训练数据或关闭该选项。

5. 微调前先判断预训练模型与数据集的关系：同一材料类型和相近 DFT 标注域属于推荐场景；相近但不同分布的数据可以微调，但应增加验证集并检查目标物理量的误差；分子、表面吸附、催化或有机体系不应仅靠拟合元素参考能来适配，应优先选择对应领域的 eSEN checkpoint，或准备足够数据进行专门训练。
