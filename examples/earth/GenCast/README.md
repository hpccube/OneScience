# GenCast

GenCast 是 Google DeepMind 提出的概率性全球天气预报模型。模型将图神经网络与扩散模型结合，用集合采样表达未来天气的不确定性，可用于中期全球天气集合预报。

> 论文：[GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)



## 数据准备

真实数据的存储格式参照 ../era5_dataset_prepare/README.md，在 `conf/config.yaml` 中配置数据路径和年份：

```yaml
data:
  data_dir: ./data
  static_dir: ./data/static
  stats_dir: ./data/stats
  train_years: [2000, 2001]
  test_years: [2003]
```


无真实数据时可生成确定性小网格数据：

```bash
python fake_data.py
```

## 运行

单卡训练（无 JAX 多主机初始化）：

```bash
source ../earth_env.sh

# 单卡训练
python train.py

# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py --config conf/config.yaml --parallel-mode pmap --num-devices 2 --global-batch-size 2
# CUDA_VISIBLE_DEVICES 使用的显卡索引
# --num-devices 使用的显卡数量
# --global-batch-size batch_size大小（可被显卡数量整除）

# 推理
python inference.py

# 评估 & 可视化
python result.py
```



## 集群训练

提交前根据集群实际情况调整 `work_slurm.sh` 中的分区、module模块、conda环境和设备资源，要求样本数能够被卡数量（节点数 * 单节点卡数）整除：

```bash
sbatch work_slurm.sh
```


## 许可证
Apache 2.0，可免费用于学术研究和商业用途。
