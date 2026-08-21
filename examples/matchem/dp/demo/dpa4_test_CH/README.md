# DPA4 `dp test` 直接推理算例

本算例使用 PyTorch 后端的 `dp test` 直接加载 DPA4 checkpoint，对 DeepMD
数据系统进行前向推理并与数据中的 DFT 标签比较，不需要先执行 `dp freeze`。

默认配置：

- 模型：相邻 `dpa4_finetune_CH/DPA4-beta.pt`
- 模型分支：`OMat24`
- 数据：`${ONESCIENCE_DATASETS_DIR}/matchem/dp/dpa4_finetune/val_CH/sys_100`
- 测试数量：1 帧
- 加速卡：1 张 DCU

## 提交运行

```bash
cd examples/matchem/dp/demo/dpa4_test_CH
sbatch submit.sh
```

结果写入 `run_<作业号>/`：

- `dp_test.log`：MAE/RMSE 汇总和运行日志；
- `detail.e.out`、`detail.e_peratom.out`：能量预测与标签；
- `detail.f.out`：力预测与标签；
- `detail.v.out`、`detail.v_peratom.out`：virial 预测与标签。

## 参数覆盖

```bash
DPA4_BASE_MODEL=/path/to/model.pt \
SYSTEM=/path/to/deepmd/system \
NUMB_TEST=10 \
MODEL_BRANCH=OMat24 \
sbatch submit.sh
```

`NUMB_TEST=0` 表示测试该数据系统的全部帧。对于不需要模型分支的普通
checkpoint，可设置 `MODEL_BRANCH=`。

此算例主要验证模型加载和直接推理流程。默认 DPA4-beta 与 CH 数据的能量零点
可能不同，因此基础模型的 Energy MAE 不应当作微调后精度结论；判断微调模型
精度时，应通过 `DPA4_BASE_MODEL` 指定微调后的 checkpoint，并使用独立测试集。
