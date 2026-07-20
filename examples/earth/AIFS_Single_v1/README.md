# AIFS_Single_v1


AIFS (Artificial Intelligence Forecasting System) 是 ECMWF 发布的基于图神经网络（GNN）的全球中期天气预报模型。通过 Encoder-Processor-Decoder 架构在 N320 高斯网格（542,080 节点）与 o96 简化网格之间进行消息传递，使用 16 层滑动窗口 Transformer 实现全球大气状态的 10 天预报。

论文：*AIFS — ECMWF's data-driven forecasting system*, arXiv:2406.01465

https://arxiv.org/abs/2406.01465


## 数据准备
真实数据的存储格式参照 `../era5_dataset_prepare/README.md`，在 `conf/config.yaml` 中修改：

```yaml
data_dir: 存放ERA5年度数据、均值/标准差文件、静态文件，存放方式参考'../era5_dataset_prepare/README.md'
train_years:                      # 训练年份
  - 2005
  - 2006
val_years:                        # 验证年份
  - 2007
test_years:                       # 测试年份
  - 2008
```
无真实数据时，可生成虚拟数据快速验证流程(若快速验证，则需将conf/config.yaml中training.max_steps设为2)：

```bash
source ../earth_env.sh
python fake_data.py
```

## 运行

```bash
source ../earth_env.sh

# 1、下载网格数据，执行后网格数据(.npz)将下载至src/onescience/models/aifs/目录下
bash download.sh

# 2、单卡训练，训练权重保存至 `weights/model_bak.ckpt`，训练前计算得到的归一化文件保存至`weights/era5_stats.npz`
python train.py

# 3、单机多卡。使用多卡时，需修改batch_size参数，否则默认batch_size=1，多卡加速不生效
torchrun --nproc_per_node=8 train.py

# 4、模型推理，使用后训练后的权重进行推理，预报步数通过 `conf/config.yaml` 中 `test_lead_time` 控制（小时数，默认 24 = 1 天）。推理结果将保存至 `output`目录
python inference.py

# 5、计算 ACC / RMSE 指标并绘图。指标保存至 `metrics/`，图片保存至 `plots/`。
python result.py
```

## 集群训练

```bash
sbatch work_slurm.sh   # 提交前检查分区、节点数等配置
```


## 许可证

Apache 2.0，可免费用于学术研究和商业用途。
