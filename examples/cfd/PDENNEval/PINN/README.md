# PINN

### 配置文件位于 ./config/ 目录下
配置文件中的参数：

* model_name：模型名称（PINN）
* scenario：PDE 类型
* data_path：数据集存储目录
* filename：数据集文件名
* model_update：打印损失信息的间隔
* learning_rate：训练时使用的学习率。PINN 所有场景都使用 1e-3。
* aux_params：PDE 中使用的参数。
* val_batch_idx：整数，表示使用数据集中的哪部分索引。在某些场景下使用 "seed"，对应数据集中的字符串索引。
* seed：随机种子

### 训练和测试：
```bash
CUDA_VISIBLE_DEVICES=0 python run.py ./config/${config_filename}
# 示例：CUDA_VISIBLE_DEVICES=0 python run.py ./config/config_pinn_darcy.yaml
```
其中 ${config_filename} 是配置文件名，例如 config_pinn_darcy.yaml。