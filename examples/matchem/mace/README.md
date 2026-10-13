# MACE 训练与推理示例

本目录包含 MACE 等变图神经网络势函数的训练、微调和推理示例。MACE 已包含在 `onescience[matchem]` 基础环境中，无需额外编译安装。

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
mace/
├── train.py                   # MACE 训练入口脚本
├── run_md.py                  # 分子动力学推理示例
├── eval_configs.py            # 配置评估工具
├── demo/                      # 训练/微调 demo 算例
│   ├── run.sh                 # 统一提交脚本
│   └── configs/               # 各算例 YAML 配置
│       ├── water_1dcu.yaml
│       ├── water_4dcu.yaml
│       ├── water_8dcu.yaml
│       ├── DMC.yaml
│       └── ...
└── fasteq/                    # FastEq-hip 推理加速（可选）
    ├── install.sh
    └── fasteq_install.sh
```

## 快速开始

### 训练

```bash
cd demo
bash run.sh --config configs/DMC.yaml
```

`run.sh` 会根据配置文件自动解析卡数、环境并生成 SLURM 提交脚本。

### 微调

基于预训练 MACE-MP0 在 Water 数据集上微调：

```bash
cd demo
bash run.sh --config configs/water_finetune_mp0_8dcu.yaml
```

### 评估

评估训练好的碳纳米管模型：

```bash
cd demo
bash run.sh --config configs/nanotube_eval_l0_1dcu.yaml
```

> **提示**：`demo/configs/` 下的 YAML 配置文件已经覆盖了训练、微调和评估三类任务。你可以复制其中任意一个配置文件，修改数据集路径、模型路径和超参数后运行，无需改动 `run.sh` 等 shell 脚本。

### 使用训练好的模型跑 MD

```bash
python run_md.py --model_path <your_model.pt> --structure <your_structure.xyz>
```

也可以通过 `demo/configs/nanotube_run_md_l0_1dcu.yaml` 直接启动 MD 示例：

```bash
cd demo
bash run.sh --config configs/nanotube_run_md_l0_1dcu.yaml
```

> 你可以复制并修改 `nanotube_run_md_l0_1dcu.yaml` 中的模型路径、初始结构和模拟参数来运行自己的 MD 任务。

### FastEq 推理加速（可选）

```bash
cd fasteq
bash install.sh
```

FastEq 当前支持 MACE 在 ASE 中的推理加速，暂不支持训练加速和 LAMMPS 推理加速。

## 注意事项

- 所有 `run.sh` 已通过 `matchem_env.sh` 加载环境。
- 默认 conda 环境为 `test_pip`，可通过 `MATCHEM_CONDA_NAME` 覆盖。
- 队列名 `#SBATCH --partition=hx1hdexclu12` 为本集群配置，迁移时需修改。
