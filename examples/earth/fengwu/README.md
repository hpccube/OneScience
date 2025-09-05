# FourCastNet

该项目包含用于 [FourCastNet：使用自适应傅里叶神经算子的全球数据驱动高分辨率天气模型](https://arxiv.org/abs/2202.11214) 的代码。

## 模型简介

FourCastNet 是傅立叶预测神经网络的缩写，是一种全球数据驱动的天气预报模型，可提供 0.25° 分辨率的精确短期至中期全球预测。FourCastNet 可准确预测高分辨率、快速时间尺度变量，例如地面风速、降水和大气水蒸气。它对规划风能资源、预测热带气旋、温带气旋和大气河流等极端天气事件具有重要意义。FourCastNet 在短时间内即可达到 ECMWF 综合预报系统 (IFS)（一种最先进的数值天气预报 (NWP) 模型）的预测精度，同时在具有复杂精细结构（包括降水）的变量方面优于 IFS。FourCastNet 可在不到 2 秒的时间内生成为期一周的预报，比 IFS 快几个数量级。 FourCastNet 的速度使我们能够快速、廉价地创建具有数千个集合成员的大型集合预报，从而改进概率预报。

![Comparison between the FourCastNet and the ground truth (ERA5) for $u-10$ for
different lead times.](../../../doc/FourCastNet.gif)

## 数据集

该模型在 ERA5 再分析数据上进行训练，这些大气变量在单层和压力层上经过预处理并存储到 HDF5 文件中。还使用了其他静态通道，例如陆地-海洋掩模和位势。数据集通过[预处理训练数据](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F)链接可以下载。

数据集目录组织方式如下：

```
FCN_ERA5_data_v0
│   README.md
└───train
│   │   1979.h5
│   │   1980.h5
│   │   ...
│   │   ...
│   │   2015.h5
│   
└───test
│   │   2016.h5
│   │   2017.h5
│
└───out_of_sample
│   │   2018.h5
│
└───static
│   │   orography.h5
│
└───precip
│   │   train/
│   │   test/
│   │   out_of_sample/
```

预先计算的统计数据通过[stats](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fadditional%2F)提供，并具有如下目录结构：

```
stats_v0
│   global_means.npy  
│   global_stds.npy  
│   land_sea_mask.npy  
│   latitude.npy  
│   longitude.npy  
│   time_means.npy
│   time_means_daily.h5
└───precip
│   │   time_means.npy
```

## 环境构建与配置

运行如下命令安装依赖：

```
pip install -r requirements.txt
```

训练配置可以在 [config/AFNO.yaml](config/AFNO.yaml) 中设置。以下路径需要自行设置。这些路径指向在上述步骤中下载的数据和统计数据：

```
afno_backbone: &backbone
  <<: *FULL_FIELD
  ...
  ...
  orography: !!bool False 
  orography_path: None # provide path to orography.h5 file if set to true, 
  exp_dir:             # directory path to store training checkpoints and other output
  train_data_path:     # full path to /train/
  valid_data_path:     # full path to /test/
  inf_data_path:       # full path to /out_of_sample. Will not be used while training.
  time_means_path:     # full path to time_means.npy
  global_means_path:   # full path to global_means.npy
  global_stds_path:    # full path to global_stds.npy
```

如果要训练降水模型，需要进行如下配置：

```
precip: &precip
  <<: *backbone
  ...
  ...
  precip:              # full path to precipitation data files 
  time_means_path_tp:  # full path to time means for precipitation
  model_wind_path:     # full path to backbone model weights ckpt
```

## 模型训练

单卡训练：

```bash
python train_fourcastnet.py --enable_amp --yaml_config=./conf/AFNO.yaml --config='afno_backbone' --run_num='check_exp'
```

多卡并行训练(集群)可参考run_mpi.sh脚本设置。

```bash
sbatch run_mpi.sh
```

## 参考

[FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)
[Adaptive Fourier Neural Operators:
Efficient Token Mixers for Transformers](https://openreview.net/pdf?id=EXHG-A3jlM)
