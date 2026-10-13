# DeepMD (DP) 训练示例

本目录包含基于 DeepMD-kit PyTorch 后端的分子动力学势函数训练示例。

## 前置条件

1. 已完成 `onescience[matchem]` 基础环境安装。
2. 已加载统一环境：
   ```bash
   # 默认激活 test_pip 环境；若使用其他 conda 环境，请先设置：
   # export MATCHEM_CONDA_NAME=your_env
   source ../matchem_env.sh
   ```
3. **首次使用需安装 DeepMD-kit DCU 版**：
   ```bash
   bash dp_install.sh
   ```
   - 脚本默认拉取 DCU 适配源码到 `deepmd-kit/`。
   - 可通过 `DEEPMD_SRC_DIR` 指定已有源码路径。
   - **请勿**自行 `git clone` 上游 `deepmodeling/deepmd-kit`，上游源码不含 DCU 支持。

## 目录说明

```
dp/
├── dp_install.sh              # DeepMD-kit DCU 安装脚本
└── demo/
    ├── water_se_e2_a_pt/      # 水分子 se_e2_a 模型（PyTorch 后端）
    ├── water_se_atten_pt/     # 水分子 se_atten 模型（PyTorch 后端）
    ├── water_se_e2_a_tf/      # 水分子 se_e2_a 模型（TensorFlow 后端，不推荐）
    └── dpa3_finetune_CH/      # DPA-3 微调示例
```

## 快速开始

### 单卡训练

```bash
cd demo/water_se_e2_a_pt
dp --pt train input_torch.json
```

### 多卡训练

```bash
cd demo/water_se_e2_a_pt
torchrun --nproc_per_node=4 -m deepmd --pt train input_torch.json
```

或直接使用示例提交脚本：

```bash
sbatch submit_1card.sh   # 单卡
sbatch submit_4card.sh   # 四卡
sbatch submit_8card.sh   # 八卡
```

## 注意事项

- DCU 平台 TensorFlow 后端存在 MLIR kernel_gen JIT 编译缺陷，**推荐优先使用 PyTorch 后端**。
- 所有 `submit.sh` 已通过 `matchem_env.sh` 加载环境，默认 conda 环境为 `test_pip`，可通过 `MATCHEM_CONDA_NAME` 覆盖。
- 队列名 `#SBATCH --partition=hx1hdexclu12` 为本集群配置，迁移时需修改。
