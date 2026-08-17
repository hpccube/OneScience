# Chai-1

Chai-1 是面向蛋白质、核酸、配体和糖链复合物的全原子结构预测模型。本目录提供 OneScience 集成后的本地推理入口；模型源码分别位于 `datapipes/chai1`、`models/chai1`、`metrics/chai1` 和 `utils/chai1`，示例不依赖外部 `chai_lab` Python 包。

## 目录结构

```text
examples/biosciences/chai1/
├── inputs/monomer.fasta   # 短蛋白示例输入
├── predict.py             # Python 推理入口
├── merge_predictions.py   # 多卡候选合并工具
└── predict.sh             # OneScience 环境与共享路径包装
```

## 模型资产

默认从 `${ONESCIENCE_MODELS_DIR}/chai-lab` 读取，不会复制权重到代码仓库。目录需要包含：

```text
chai-lab/
├── conformers_v1.apkl
├── esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt
└── models_v2/
    ├── bond_loss_input_proj.pt
    ├── confidence_head.pt
    ├── diffusion_module.pt
    ├── feature_embedding.pt
    ├── token_embedder.pt
    └── trunk.pt
```

可通过 `CHAI1_MODEL_DIR` 指定另一套完整资产目录。Python API 也支持显式传入 `model_dir`。

## 推理

请先由用户自行申请包含 GPU/DCU 的计算节点，并在节点上激活已安装 OneScience 依赖的 Conda 或 virtualenv 环境。`predict.sh` 不会替用户申请资源、加载 module、激活 Conda 环境，也不会调用 `sbatch` 或 `srun`。然后在 OneScience 根目录执行：

```bash
# conda activate <your-environment>
bash examples/biosciences/chai1/predict.sh
```

如果当前环境中的解释器不是 `python`，可通过 `CHAI1_PYTHON=/path/to/python` 指定。集群所需的 `module load` 命令也应由用户按本地集群规则执行。

Chai-1 推理核心本身是单进程单设备。需要多卡时，在同一计算节点申请多张卡，并设置 `CHAI1_DEVICES`；脚本会为每张卡启动独立进程，将总候选数分摊到各卡，最后合并并重新排序结果：

```bash
CHAI1_DEVICES=0,1,2,3 \
CHAI1_NUM_DIFFUSION_SAMPLES=8 \
bash examples/biosciences/chai1/predict.sh
```

这里的 `CHAI1_NUM_DIFFUSION_SAMPLES` 是所有卡合计生成的候选数，不是每卡数量。多卡模式当前要求 `CHAI1_NUM_TRUNK_SAMPLES=1`；需要增加 trunk samples 时，可分批运行或使用单卡模式。每个进程只看到自己对应的一张卡，因此同时兼容 `CUDA_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES`。

多卡模式中的输出目录、模型目录、设备、随机种子和采样参数由 `CHAI1_*` 环境变量统一控制，不要再追加同名 Python 参数；MSA、模板、约束和服务开关等其他参数仍可追加在输入文件之后。

也可以直接设置 `CUDA_VISIBLE_DEVICES=0,1,2,3` 或 `HIP_VISIBLE_DEVICES=0,1,2,3`，脚本会将其作为可用卡列表；显式设置 `CHAI1_DEVICES` 时优先使用显式列表。

脚本默认运行适合验证集成正确性的配置：1 次 trunk recycle、20 个扩散步、1 个候选。输出写入 `examples/biosciences/chai1/outputs/monomer`，包括预测 CIF、置信度 NPZ 和按 aggregate score 排序的 `ranking.json`。

使用标准精度配置可设置：

```bash
CHAI1_NUM_TRUNK_RECYCLES=3 \
CHAI1_NUM_DIFFUSION_TIMESTEPS=200 \
CHAI1_NUM_DIFFUSION_SAMPLES=5 \
bash examples/biosciences/chai1/predict.sh
```

输入采用 Chai-1 FASTA-like 格式，每个实体必须有唯一名称：

```text
>protein|name=target
MKT...
>ligand|name=ligand
CC(=O)O
>rna|name=rna
AGUC
>dna|name=dna
ATGC
```

完整参数可查看：

```bash
PYTHONPATH=src python examples/biosciences/chai1/predict.py --help
```

本地 MSA、模板命中和约束可分别通过 `--msa-directory`、`--template-hits-path` 和 `--constraint-path` 传入。`--use-msa-server` 与 `--use-templates-server` 会访问外部服务，默认关闭。

## Python API

```python
from pathlib import Path

from onescience.models.chai1 import run_inference

candidates = run_inference(
    fasta_file=Path("input.fasta"),
    output_dir=Path("outputs/example"),
    model_dir=Path("/public/share/sugonhpcapp01/onestore/onemodels/chai-lab"),
    device="cuda:0",
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_diffn_samples=5,
    seed=42,
)
```

## 许可证

迁移的 Chai-Lab 源码保留原项目 Apache License 2.0 版权头。模型、权重及第三方依赖的使用需同时遵守各自许可。
