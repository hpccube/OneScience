# MedGemma 模型配置
# 此文件定义了 MedGemma 模型的基础配置结构

from onescience.models.protenix.config.extend_types import (
    DefaultNoneWithType,
    RequiredValue,
)

# MedGemma 基础配置
MEDGEMMA_BASE_CONFIG = {
    "variant": RequiredValue(str),  # "4b" or "27b"
    "model_path": RequiredValue(str),
    "tokenizer_path": DefaultNoneWithType(str),
    "prompt_format": "chat",
    "is_multimodal": True,
}

# 推理配置
MEDGEMMA_INFERENCE_CONFIG = {
    "gpu_memory_utilization": 0.9,
    "max_model_len": DefaultNoneWithType(int),
    "tensor_parallel_size": 1,
    "default_max_tokens": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "batch_size": 1,
}
