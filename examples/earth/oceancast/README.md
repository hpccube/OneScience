# OceanCast

该项目实现海浪(海流/洋流)细时空分辨率预测，模型参照 [FourCastNet：使用自适应傅里叶神经算子的全球数据驱动高分辨率天气模型](https://arxiv.org/abs/2202.11214) 的代码构建。

## 模型简介

OceanCast 是一种全球数据驱动的海浪/海流/洋流预报模型，可提供1° 分辨率的精确超短期全球预测。
OceanCast 可准确预测高分辨率、快速时间尺度变量，例如海浪高度、海浪周期、海浪方向(海表温、海表盐、海面高度或洋流)等。
通过修改conf/oceancast.yaml中待预测的input/output_type，可切换海浪/海流/洋流预报模型


## 环境构建与配置

conf/oceancast.yaml中包含数据路径、模型训练参数等，需要注意或修改的内容如下：

    data_path为包含训练集、验证集、测试集的路径，默认为当前路径
    
    global_means_path为数据集不同变量的均值，用于归一化
    
    global_stds_path为数据集不同变量的标准差，用于归一化 
    
    maskpath为海洋陆地掩码
    
    目前本文件夹没有数据，具体数据构造加载参照oceancast-pytorch.ipynb中构建方法
    

//目前没有风的预测值，因此在data_loader中将预测值临时替换为了真实值


## 参考

[FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214)
[Adaptive Fourier Neural Operators:
Efficient Token Mixers for Transformers](https://openreview.net/pdf?id=EXHG-A3jlM)
