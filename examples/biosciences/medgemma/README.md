# MedGemma

本示例将 MedGemma 集成到 OneScience 生物信息（AI for Biology）组件中，提供面向医学场景的统一推理、微调与评估入口。

## 简介

MedGemma 是 Google 开源的医学多模态大语言模型，基于 [Gemma 3](https://ai.google.dev/gemma/docs/core) 架构，针对医学文本与医学影像理解进行训练。MedGemma 提供两种变体：

- **MedGemma 4B**：多模态模型，支持医学文本与医学图像联合输入。
- **MedGemma 27B**：纯文本模型，专注于医学文本理解与问答。

MedGemma 4B 使用 [SigLIP](https://arxiv.org/abs/2303.15343) 图像编码器，已在多种去标识化医学数据上预训练，包括胸片（CXR）、皮肤科图像、眼科图像和组织病理学切片；其语言模型组件在放射学图像、病理学图像、眼科图像、皮肤科图像和医学文本上进行了训练。

MedGemma 已在多项临床相关基准上评估，涵盖开放基准数据集和专家人工评估任务。更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡（Model Card）](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Hugging Face 模型](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)

当前示例默认基于 `google/medgemma-1.5-4b-it`（4B 多模态指令模型），支持文本与医学图像联合输入。

---

## 目录

- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [数据与模型权重](#数据与模型权重)
- [数据预处理与格式要求](#数据预处理与格式要求)
- [功能定位](#功能定位)
- [任务选择与基础流程](#任务选择与基础流程)
- [详细使用说明](#详细使用说明)
  - [1. 集成测试](#1-集成测试)
  - [2. 医学问答评估（`run_evaluate_on_medqa.sh`）](#2-医学问答评估run_evaluate_on_medqash)
  - [3. 胸片解剖结构定位（`run_cxr_anatomy.sh`）](#3-胸片解剖结构定位run_cxr_anatomysh)
  - [4. 胸片纵向对比分析（`run_cxr_longitudinal_comparison.sh`）](#4-胸片纵向对比分析run_cxr_longitudinal_comparisonsh)
  - [5. 病理图像 LoRA 微调（`run_fine_tune.sh`）](#5-病理图像-lora-微调run_fine_tunesh)
  - [6. 本地图像基础推理](#6-本地图像基础推理)
  - [7. 单个 DICOM 推理](#7-单个-dicom-推理)
  - [8. CT 多切片推理](#8-ct-多切片推理)
  - [9. 病理图像块推理](#9-病理图像块推理)
  - [10. MedQA 本地监督微调（SFT）](#10-medqa-本地监督微调sft)
  - [11. EHR/FHIR 导航（暂不可用）](#11-ehrfhir-导航暂不可用)
  - [12. 使用推理运行器](#12-使用推理运行器)
  - [13. Python API 调用](#13-python-api-调用)
- [运行约束](#运行约束)
- [Issues](#issues)
- [许可证与引用](#许可证与引用)

---

## 目录结构

```
examples/biosciences/medgemma/
├── configs/                          # 配置目录
│   ├── inference_config.yaml         # 推理配置示例
│   └── configs_base.py               # 基础配置定义
├── runner/
│   └── medical_inference_runner.py   # 统一医学推理运行器
├── scripts/                          # 可执行脚本
│   ├── notebook_conver/              # 脚本调用的 Python 实现
│   │   ├── cxr_anatomy_localization_with_hugging_face.py
│   │   ├── cxr_longitudinal_comparison.py
│   │   ├── evaluate_on_medqa.py
│   │   ├── fine_tune_with_hugging_face.py
│   │   ├── quick_start_with_hugging_face.py
│   │   ├── quick_start_with_model_garden.py
│   │   ├── quick_start_with_dicom.py
│   │   ├── high_dimensional_ct_hugging_face.py
│   │   ├── high_dimensional_pathology_hugging_face.py
│   │   ├── reinforcement_learning_with_hugging_face.py
│   │   ├── find_sepsis_medication_patient.py
│   │   ├── medgemma_script_utils.py
│   │   └── detect_image_token.py
│   ├── run_cxr_anatomy.sh            # 胸片解剖结构定位
│   ├── run_cxr_longitudinal_comparison.sh  # 胸片前后对比分析
│   ├── run_evaluate_on_medqa.sh      # MedQA 医学问答评估
│   ├── run_fine_tune.sh              # 病理图像 LoRA 微调
│   ├── run_quick_start_with_hugging_face.sh
│   ├── run_quick_start_with_model_garden.sh
│   ├── run_quick_start_with_dicom.sh
│   ├── run_high_dimensional_ct_hugging_face.sh
│   ├── run_high_dimensional_pathology_hugging_face.sh
│   ├── run_reinforcement_learning_with_hugging_face.sh
│   └── run_ehr_navigator_agent.sh    # 当前缺少对应 Python 主程序
├── tests/
│   └── test_integration.py           # 集成测试脚本
└── README.md                         
```

模型实现位于 `src/onescience/models/medgemma`。

---

## 环境准备

1. 参照项目根目录 [README.md](../../../README.md) 完成 OneScience（bio 领域）安装：

    ```bash
    bash install.sh bio
    ```

2. 激活环境：

    ```bash
    conda activate onescience311
    ```

3. 确认 `boto3`、`botocore` 和 `transformers` 版本与脚本要求一致。需要更新时，可执行：

    ```bash
    pip install --upgrade boto3==1.43.36 botocore==1.43.36
    pip install --upgrade transformers==5.12.1
    ```

4. 根据任务安装以下附加依赖；基础 PNG/JPG 图像推理无需安装全部依赖：

    | 任务 | 额外依赖 |
    |------|----------|
    | DICOM / CT | `pydicom`、`numpy` |
    | HDF5 病理图像块 | `h5py`、`numpy` |
    | MedQA SFT | `datasets`；启用 LoRA 时还需 `peft` |

5. 确保 `ONESCIENCE_DATASETS_DIR` 环境变量已设置（通常由项目根目录 `env.sh` 自动配置）：

    ```bash
    source /path/to/onescience/env.sh
    ```

---

## 数据与模型权重

### 1. 模型权重

脚本默认从以下路径加载模型：

```
${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it
```

请提前下载模型并放置到该目录，或通过 `model_path` 环境变量覆盖。模型可通过以下渠道获取：

- [Hugging Face - google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- [ModelScope](https://modelscope.cn/)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)

### 2. 数据集

| 任务 | 数据 | 默认路径 |
|------|------|----------|
| MedQA 评估 | MedQA parquet 数据 | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` |
| 胸片解剖定位 | 胸片图像 | `${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/...` |
| 胸片纵向对比 | 前后两次胸片 | `${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/...` |
| 病理图像微调 | NCT-CRC-HE-100K / CRC-VAL-HE-7K | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/...` |
| DICOM 单图报告 | `.dcm` 文件 | `${ONESCIENCE_DATASETS_DIR}/medgemma/COVID-19-AR/...` |
| CT 序列分析 | DICOM 序列目录或预渲染图片目录 | `${ONESCIENCE_DATASETS_DIR}/medgemma/CTLM` |
| 病理图像块推理 | PNG/JPG 目录或 HDF5 文件 | `${ONESCIENCE_DATASETS_DIR}/medgemma/pathology_patches` |
| MedQA SFT | MedQA parquet | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` |
| EHR 导航代理 | 本地 FHIR JSON 或远端 FHIR store | `${ONESCIENCE_DATASETS_DIR}/medgemma/fhir` |

数据集可通过以下方式获取：

- **MedQA**：https://github.com/jind11/MedQA
- **Chest X-ray**：推荐使用公开胸片数据集，如 COVID-19 Chest X-Ray Dataset 或 MIMIC-CXR
- **NCT-CRC-HE-100K / CRC-VAL-HE-7K**：https://zenodo.org/records/1214456

---

## 数据预处理与格式要求

使用预训练模型推理不需要重新训练或生成特征文件，但输入数据必须符合各任务格式：

| 任务 | 输入格式 | 是否需要手工预处理 |
|------|----------|--------------------|
| MedQA | parquet 目录 | 通常不需要，脚本直接读取 |
| 胸片定位 | PNG/JPG 单图或图片目录 | 建议确认方向和尺寸；脚本可填充为正方形 |
| 胸片纵向对比 | 同一患者的 before/after 两张图 | 必须保证先后顺序正确，建议统一方向 |
| NCT LoRA 微调 | NCT-CRC-HE-100K.zip 和 CRC-VAL-HE-7K.zip | 脚本负责读取/解压，但压缩包结构必须完整 |
| DICOM 单图 | 单个 `.dcm` 文件 | 脚本读取像素、应用窗宽窗位并保存模型输入预览 |
| CT 序列 | DICOM 目录或 PNG/JPG 目录 | 用 UID 选择序列，限制切片数以控制显存 |
| 病理图像块 | PNG/JPG 目录或包含图像块的 HDF5 | 支持逐图像块推理或聚合推理 |
| EHR 导航代理 | FHIR Bundle/Resource JSON | Patient、Encounter、MedicationAdministration 引用需保持一致 |
| MedicalInferenceRunner | JSON/JSONL 或输入目录 | 字段必须与 inference_config.yaml 的数据配置一致 |

使用预训练模型执行 MedQA 或单图推理时，无需准备 LoRA 训练数据。微调前应检查类别目录、标签数量及训练/验证划分，以排除数据格式问题。

---
## 功能定位

- **医学问答**：基于 MedQA 等医学知识基准评估模型问答能力。
- **医学影像分析**：支持胸片（CXR）解剖结构定位、多期影像对比分析等任务。
- **领域微调**：基于 NCT 结肠组织病理图像等数据，使用 LoRA 进行参数高效微调。
- **统一推理接口**：通过 `MedicalInferenceRunner` 提供交互式与批量文件推理能力。
- **本地推理**：提供普通图像、DICOM、CT 序列和病理图像块的离线推理入口。
- **MedQA 监督微调**：保留上游“reinforcement learning”脚本名称，但当前实现是本地 SFT，可选 PEFT LoRA。
- **EHR 导航**：提供 FHIR 患者筛选脚本。

---

## 任务选择与基础流程

### 入口选择

| 目标 | 推荐入口 | 输入 | 输出 |
|--------------|----------|------|------|
| 环境和模块能否导入 | python tests/test_integration.py | 无模型推理 | 测试日志 |
| 医学文本问答 | bash scripts/run_evaluate_on_medqa.sh | MedQA parquet | medqa_results |
| 单张胸片解剖定位 | bash scripts/run_cxr_anatomy.sh | 胸片图像 + 结构名称 | 标注图 + JSON |
| 两次胸片变化对比 | bash scripts/run_cxr_longitudinal_comparison.sh | before/after 两张图 | TXT + JSON |
| 病理图像领域微调 | bash scripts/run_fine_tune.sh | 两个 NCT zip 数据集 | LoRA adapter + eval_metrics.json |
| 自定义文本/图像批处理 | MedicalInferenceRunner | inference_config.yaml + JSON/目录 | JSON/JSONL |
| 最小图像问答 | bash scripts/run_quick_start_with_hugging_face.sh | PNG/JPG | JSON + TXT |
| DICOM 放射学报告 | bash scripts/run_quick_start_with_dicom.sh | DICOM | 预览图 + JSON/TXT/Markdown |
| CT 序列分析 | bash scripts/run_high_dimensional_ct_hugging_face.sh | DICOM/图片序列 | JSON + TXT |
| 病理图像块分析 | bash scripts/run_high_dimensional_pathology_hugging_face.sh | 图片目录/HDF5 | 选中图像块 + JSON/Markdown |
| MedQA 监督微调（SFT） | bash scripts/run_reinforcement_learning_with_hugging_face.sh | MedQA + 27B 文本模型 | Trainer checkpoint |

### 各步骤之间的关系

各任务共享同一个基础模型，除明确标注的依赖关系外，可独立执行：

```text
本地 MedGemma 基础模型
        │
        ├── MedQA 数据 ─────────────► 问答评估
        ├── 单张 CXR ───────────────► 解剖定位
        ├── 前后两张 CXR ───────────► 纵向对比
        ├── inference_config.yaml ──► 通用交互/批量推理
        └── NCT 病理数据 ───────────► LoRA 微调
                                          │
                                          ▼
                                      adapter 目录
```

LoRA 微调不会覆盖基础模型。
后续使用 adapter 时，需要以原始基础模型为 base，再通过 PEFT 加载 adapter；不能把 adapter 目录当成一个完全独立的基础模型随意替换 model_path。

---

## 详细使用说明

所有脚本默认在 `examples/biosciences/medgemma/scripts/` 目录下执行，并自动定位项目根目录加载 `env.sh`。

### 1. 集成测试

验证 MedGemma 在 OneScience 中的模块、配置、数据适配器与图像处理组件是否可正常导入：

```bash
cd examples/biosciences/medgemma
python tests/test_integration.py
```

---

### 2. 医学问答评估（`run_evaluate_on_medqa.sh`）

在 MedQA 数据集上评估模型医学问答能力，默认处理 10 条样本用于快速验证。

```bash
cd examples/biosciences/medgemma
bash scripts/run_evaluate_on_medqa.sh
```

脚本内部调用：

```bash
python ./notebook_conver/evaluate_on_medqa.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --parquet_dir ${ONESCIENCE_DATASETS_DIR}/medgemma/medqa \
    --output_dir ./medqa_results \
    --max_samples 10
```

常用参数覆盖（通过修改脚本或环境变量）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it` | 模型目录 |
| `--parquet_dir` | `${ONESCIENCE_DATASETS_DIR}/medgemma/medqa` | MedQA parquet 数据目录 |
| `--output_dir` | `./medqa_results` | 结果输出目录 |
| `--max_samples` | `10` | 评估样本数，设置为 `-1` 可评估全部 |
| `--device` / `HIP_VISIBLE_DEVICES` | `0` | GPU 设备 |

输出：

- `scripts/medqa_results/medqa_results.json`：每条样本的详细结果
- `scripts/medqa_results/summary.txt`：准确率等汇总指标

---

### 3. 胸片解剖结构定位（`run_cxr_anatomy.sh`）

对单张或多张胸片进行解剖部位定位。脚本内部同时运行单图模式和批量模式：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_anatomy.sh
```

脚本内部调用：

```bash
# 单图模式
python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --image_path "${ONESCIENCE_DATASETS_DIR}/medgemma/Chest_Xray/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (89).jpg" \
    --object_name "right clavicle" \
    --num_gpus 2

# 多图模式
python ./notebook_conver/cxr_anatomy_localization_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --input_dir "${ONESCIENCE_DATASETS_DIR}/medgemma/test_images" \
    --object_name "right clavicle" \
    --num_gpus 2
```

常用参数：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--model_path` | 是 | 本地模型目录 |
| `--image_path` | 单图模式必填 | 单张胸片路径 |
| `--input_dir` | 批量模式必填 | 批量胸片目录 |
| `--object_name` | 是 | 待定位的解剖结构，例如 `"right clavicle"` |
| `--num_gpus` | 否 | 使用的 GPU 数量，默认 `1` |
| `--output_dir` | 否 | 结果输出目录，默认 `./outputs` |

输出：

- `scripts/outputs/result_*.json`：定位坐标与标签
- `scripts/outputs/result_*.png`：带边界框标注的可视化图像
- `scripts/outputs/batch_summary.json`：批量模式汇总结果

---

### 4. 胸片纵向对比分析（`run_cxr_longitudinal_comparison.sh`）

对同一患者的前后两次胸片进行对比分析：

```bash
cd examples/biosciences/medgemma
bash scripts/run_cxr_longitudinal_comparison.sh
```

脚本内部调用：

```bash
python ./notebook_conver/cxr_longitudinal_comparison.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --image1 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_before.png \
    --image2 ${ONESCIENCE_DATASETS_DIR}/medgemma/test_compare/longitudinal_cxr_after.png \
    --output_dir ./compare_outputs
```

常用参数：

| 参数 | 是否必填 | 说明 |
|------|----------|------|
| `--model_path` | 是 | 本地模型目录 |
| `--image1` | 是 | 第一张图像路径（如治疗前） |
| `--image2` | 是 | 第二张图像路径（如治疗后） |
| `--output_dir` | 否 | 结果输出目录，默认 `./compare_outputs` |
| `--prompt` | 否 | 自定义对比提示词 |
| `--preprocess` | 否 | 图像非正方形时是否填充为正方形 |
| `--num_gpus` | 否 | 使用的 GPU 数量 |

输出：

- `scripts/compare_outputs/compare_<image1>_vs_<image2>.txt`：文本对比报告
- `scripts/compare_outputs/compare_<image1>_vs_<image2>.json`：结构化 JSON 结果

---

### 5. 病理图像 LoRA 微调（`run_fine_tune.sh`）

基于 NCT 结肠组织病理图像数据集进行 LoRA 微调：

```bash
cd examples/biosciences/medgemma
bash scripts/run_fine_tune.sh
```

脚本内部调用：

```bash
python ./notebook_conver/fine_tune_with_hugging_face.py \
    --model_path ${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it \
    --train_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/NCT-CRC-HE-100K.zip \
    --test_zip ${ONESCIENCE_DATASETS_DIR}/medgemma/nct/CRC-VAL-HE-7K.zip \
    --output_dir ./medgemma-nct-lora \
    --max_train_samples 9000 \
    --max_val_samples 1000 \
    --max_test_samples 1000
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `${ONESCIENCE_DATASETS_DIR}/medgemma/modelscope/google/medgemma-1.5-4b-it` | 本地模型目录 |
| `--train_zip` | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/NCT-CRC-HE-100K.zip` | 训练集 zip |
| `--test_zip` | `${ONESCIENCE_DATASETS_DIR}/medgemma/nct/CRC-VAL-HE-7K.zip` | 测试集 zip |
| `--output_dir` | `./medgemma-nct-lora` | LoRA 输出目录 |
| `--max_train_samples` | `9000` | 训练样本数 |
| `--max_val_samples` | `1000` | 验证样本数 |
| `--max_test_samples` | `1000` | 测试样本数 |

输出：

- `scripts/medgemma-nct-lora/`：LoRA 权重、训练日志与评估结果

> 依赖管理：脚本会检查 `transformers==5.12.1` 和 `accelerate>=0.29.0`，缺失或版本不符时会调用 pip 升级。生产环境应预先固定依赖，避免在作业运行期间修改环境。

---

以下推理与微调脚本均先加载仓库根目录的 `env.sh`，再从 `examples/biosciences/medgemma/scripts/` 执行 Python 主程序。路径类环境变量可在命令前覆盖，附加命令行参数将传递给 Python 入口。

### 6. 本地图像基础推理

```bash
bash scripts/run_quick_start_with_hugging_face.sh

MODEL_PATH=/path/to/medgemma-1.5-4b-it \
IMAGE_PATH=/path/to/cxr.png \
PROMPT="Describe the visible findings." \
OUTPUT_DIR=./outputs/cxr_quick_start \
bash scripts/run_quick_start_with_hugging_face.sh
```

`run_quick_start_with_model_garden.sh` 使用相同的本地 Hugging Face 加载方式。它不会访问 Vertex AI，也不需要 `PROJECT_ID`、`REGION` 或 endpoint；名称仅用于对应上游 notebook 场景。两个入口都会写出任务 JSON 和 TXT。

### 7. 单个 DICOM 推理

```bash
DICOM_PATH=/path/to/image.dcm \
OUTPUT_DIR=./outputs/dicom_report \
bash scripts/run_quick_start_with_dicom.sh
```

常用变量为 `MODEL_PATH`、`DICOM_PATH`、`PROMPT`、`MAX_NEW_TOKENS`、`OUTPUT_DIR` 和 `HIP_VISIBLE_DEVICES`。输出包括作为模型输入的 `quick_start_with_dicom_input.png`，以及 JSON、TXT 和 Markdown 结果。可追加 `--revision_pass`，在初稿存在明显矛盾时执行第二次校订；这仍不替代放射科医师复核。

### 8. CT 多切片推理

```bash
CT_DICOM_DIR=/path/to/ct/dicom \
CT_MAX_SLICES=32 \
CT_OUTPUT_DIR=./outputs/ct \
bash scripts/run_high_dimensional_ct_hugging_face.sh

# 已转换为图片时
IMAGE_DIR=/path/to/ct/png \
bash scripts/run_high_dimensional_ct_hugging_face.sh
```

DICOM 模式可用 `CT_STUDY_INSTANCE_UID` 和 `CT_SERIES_INSTANCE_UID` 选择序列，并用 `CT_PROMPT`、`CT_INSTRUCTION` 自定义任务。`CT_MAX_SLICES` 直接影响显存和上下文长度，首次运行建议从 8 到 32 张开始。

### 9. 病理图像块推理

```bash
IMAGE_DIR=/path/to/pathology_patches \
MAX_PATCHES=8 \
INFERENCE_MODE=per_patch \
OUTPUT_DIR=./outputs/pathology \
bash scripts/run_high_dimensional_pathology_hugging_face.sh

# HDF5 输入
H5_PATH=/path/to/patches.h5 \
INFERENCE_MODE=aggregate \
bash scripts/run_high_dimensional_pathology_hugging_face.sh
```

`INFERENCE_MODE` 可取 `per_patch` 或 `aggregate`。脚本将选中的图像保存到 `selected_patches/`，并写出 JSON 和 Markdown。`TISSUE_CONTEXT` 仅提供上下文提示，不能替代切片元数据或病理标注。

### 10. MedQA 本地监督微调（SFT）

```bash
# 参数解析验证：不加载模型或训练依赖
bash scripts/run_reinforcement_learning_with_hugging_face.sh --dry_run

MAX_TRAIN_SAMPLES=256 \
MAX_EVAL_SAMPLES=64 \
USE_LORA=1 \
bash scripts/run_reinforcement_learning_with_hugging_face.sh
```

该脚本沿用 `reinforcement_learning_with_hugging_face` 文件名，但当前 Python 入口为 `local_sft_no_trl`。其实现使用 `transformers.Trainer` 执行因果语言模型监督微调，不包含奖励函数、奖励模型、PPO、DPO 或 GRPO，因此不属于强化学习。

该入口默认加载 27B 文本模型，使用本地 MedQA parquet，并禁用在线实验上报。主要变量包括 `MODEL_PATH`、`PARQUET_DIR`、`RL_OUTPUT_DIR`、`MAX_TRAIN_SAMPLES`、`MAX_EVAL_SAMPLES`、`MAX_SEQ_LENGTH` 和 `USE_LORA`。启用 LoRA 需要安装兼容版本的 PEFT；当前脚本没有量化加载参数。

### 11. EHR/FHIR 导航

`run_ehr_navigator_agent.sh` 支持本地 FHIR JSON ，并可自动调用 `find_sepsis_medication_patient.py` 选择示例患者。主要变量包括 `LLM_BACKEND`、`MODEL_PATH`、`FHIR_DATA_DIR`、`FHIR_STORE_URL`、`EHR_PATIENT_ID`、`QUESTION` 和 `OUTPUT_DIR`。

---
### 12. 使用推理运行器

`runner/medical_inference_runner.py` 提供统一的推理入口，支持交互式与批量文件推理。

#### 交互式推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --interactive
```

#### 批量文件推理

```bash
cd examples/biosciences/medgemma
export PYTHONPATH=../../../src:$PYTHONPATH
python runner/medical_inference_runner.py \
    --config configs/inference_config.yaml \
    --input /path/to/input.json
```

---

### 13. Python API 调用

```python
from onescience.models.medgemma import MedGemma
from onescience.models.medgemma.config import load_config

configs = load_config("configs/inference_config.yaml")
model = MedGemma(configs)

messages = [
    {"role": "system", "content": "You are an expert medical AI assistant."},
    {"role": "user", "content": "What are the common causes of hypertension?"}
]
result = model.forward(messages, max_tokens=500)
print(result["choices"][0]["message"]["content"])
```

---

## 运行约束

- 运行脚本前需确保 `ONESCIENCE_DATASETS_DIR` 环境变量已正确设置。
- 脚本默认使用 `HIP_VISIBLE_DEVICES=0`，在海光 DCU 平台可直接运行；在 CUDA 平台可替换为 `CUDA_VISIBLE_DEVICES=0` 或根据设备调整。
- 如需使用 vLLM 加速推理，请确保已安装对应版本的 vLLM 并配置 `use_vllm: true`。
- `run_fine_tune.sh` 会检查并可能升级 `transformers` 和 `accelerate`；在离线节点或受控环境中应提前安装并固定版本。
- 所有 sh 脚本内部自动加载项目根目录 `env.sh`，无需手动 source。
- 基础图像、DICOM、CT 和病理推理脚本使用本地模型，不会自动下载缺失权重。
- `run_ehr_navigator_agent.sh` 当前缺少 `ehr_navigator_agent.py`，需补齐实现后才能运行。
- 4B 多模态模型推理显存需求较大，建议至少单卡 24GB 显存；多卡可通过 `num_gpus` 或外部 `CUDA_VISIBLE_DEVICES` 控制。

---

## Issues

| 现象 | 处理 |
|------|------|
| 模型目录下载不完整 | 确认 config、tokenizer/processor 和所有权重分片都存在 |
| 4B 图像脚本提示不支持图片 | 检查是否错误使用 27B 文本模型，或 is_multimodal 配置不正确 |
| 显存不足 | 减小 batch_size、default_max_tokens，使用单样本；必要时调整 tensor parallel |
| 模型尝试联网 | 使用完整本地目录，并根据环境设置离线模式 |
| 定位 JSON 无法解析 | 模型输出未遵循 bounding-box 格式；保留原始文本并调整 prompt/temperature |
| LoRA 训练后效果没有变化 | 推理时可能仍加载基础模型，需要显式加载 output_dir 中的 PEFT adapter |
| 医学回答缺少可验证依据 | 模型可能产生事实性错误或幻觉，结果必须由专业人员复核 |

---

## 许可证与引用

MedGemma 模型采用 [Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms) 许可，本仓库示例代码采用 Apache 2.0 许可。

更多信息请参阅：

- [开发者文档](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [模型卡](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [社区准则](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
- [Hugging Face](https://huggingface.co/models?other=medgemma)
- [Google Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)
