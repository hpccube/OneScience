"""Local model loading and embedding interfaces for DNABERT-2."""

import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import load_sharded_checkpoint, load_state_dict
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from onescience.modules.attention.dnabert2attention import torch_qkvpacked_attention
from onescience.modules.pooling.dnabert2pooling import pool_hidden_states


LOGGER = logging.getLogger(__name__)
_ATTENTION_FALLBACK_WARNED = False


def _legacy_triton_dot_supported() -> bool:
    """Return whether Triton's dot API accepts the bundled kernel arguments."""
    try:
        import triton.language as tl

        parameters = inspect.signature(tl.dot).parameters
    except (ImportError, TypeError, ValueError):
        return False
    return "trans_a" in parameters and "trans_b" in parameters


def _configure_attention_backend(model):
    """Replace only the incompatible bundled Triton attention entry point."""
    global _ATTENTION_FALLBACK_WARNED

    if _legacy_triton_dot_supported():
        return model

    patched_modules = set()
    for layer in model.modules():
        if layer.__class__.__name__ != "BertUnpadSelfAttention":
            continue
        remote_module = sys.modules.get(layer.__class__.__module__)
        if remote_module is None or not hasattr(remote_module, "flash_attn_qkvpacked_func"):
            continue
        if remote_module in patched_modules:
            continue
        remote_module.flash_attn_qkvpacked_func = torch_qkvpacked_attention
        patched_modules.add(remote_module)

    if patched_modules and not _ATTENTION_FALLBACK_WARNED:
        LOGGER.warning(
            "The bundled DNABERT-2 Triton attention kernel uses a legacy tl.dot API; "
            "using the equivalent PyTorch scaled dot-product implementation."
        )
        _ATTENTION_FALLBACK_WARNED = True
    return model


def validate_model_directory(model_dir: str | Path) -> Path:
    """Validate local DNABERT-2 weights and tokenizer assets."""
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DNABERT-2 model directory not found: {path}")
    tokenizer_files = ["tokenizer.json", "vocab.txt", "tokenizer.model"]
    if not any((path / name).is_file() for name in tokenizer_files):
        raise FileNotFoundError(f"No tokenizer asset found in DNABERT-2 model directory: {path}")
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No DNABERT-2 model weights found in: {path}")
    return path


def validate_architecture_directory(architecture_dir: str | Path) -> Path:
    """Validate the reusable local architecture bundle used by all entry points."""
    path = Path(architecture_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DNABERT-2 architecture directory not found: {path}")
    required_files = (
        "config.json",
        "configuration_bert.py",
        "bert_layers.py",
        "bert_padding.py",
        "flash_attn_triton.py",
    )
    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DNABERT-2 architecture directory {path} is missing: {', '.join(missing)}"
        )
    return path


def load_tokenizer(model_dir: str | Path, model_max_length: int = 512):
    """Load the tokenizer without accessing the Hugging Face Hub."""
    path = validate_model_directory(model_dir)
    return AutoTokenizer.from_pretrained(
        path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )


def _load_config(architecture_dir: str | Path, num_labels: int | None = None):
    architecture_path = validate_architecture_directory(architecture_dir)
    config = AutoConfig.from_pretrained(
        architecture_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if num_labels is not None:
        config.num_labels = num_labels
    return config


def _load_pretrained_model(
    model_dir: str | Path,
    architecture_dir: str | Path,
    auto_class: str,
    num_labels: int | None = None,
):
    """Load weights with model code resolved from a separate local bundle."""
    path = validate_model_directory(model_dir)
    architecture_path = validate_architecture_directory(architecture_dir)
    config = _load_config(architecture_path, num_labels=num_labels)
    class_reference = config.auto_map.get(auto_class)
    if not class_reference:
        raise ValueError(
            f"DNABERT-2 architecture config does not define auto_map[{auto_class!r}]"
        )
    model_class = get_class_from_dynamic_module(
        class_reference,
        architecture_path,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        path,
        config=config,
        local_files_only=True,
    )
    return _configure_attention_backend(model)


def load_sequence_classifier(
    model_dir: str | Path,
    num_labels: int,
    architecture_dir: str | Path | None = None,
):
    """Load a local DNABERT-2 classifier using a reusable architecture bundle."""
    return _load_pretrained_model(
        model_dir,
        architecture_dir or model_dir,
        "AutoModelForSequenceClassification",
        num_labels=num_labels,
    )


def _load_training_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "dnabert2_training.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_full_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"DNABERT-2 checkpoint directory not found: {checkpoint_dir}")
    weight_files = (
        SAFE_WEIGHTS_NAME,
        WEIGHTS_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
    )
    if not any((checkpoint_dir / name).is_file() for name in weight_files):
        expected = ", ".join(weight_files)
        raise FileNotFoundError(
            f"No full DNABERT-2 checkpoint weights found in {checkpoint_dir}; "
            f"expected one of: {expected}"
        )
    return checkpoint_dir


def _build_sequence_classifier(architecture_dir: str | Path, num_labels: int):
    """Build the classifier architecture without loading base checkpoint weights."""
    architecture_path = validate_architecture_directory(architecture_dir)
    config = _load_config(architecture_path, num_labels=num_labels)
    class_reference = config.auto_map.get("AutoModelForSequenceClassification")
    if not class_reference:
        raise ValueError(
            "DNABERT-2 architecture config does not define "
            "auto_map['AutoModelForSequenceClassification']"
        )
    model_class = get_class_from_dynamic_module(
        class_reference,
        architecture_path,
        local_files_only=True,
    )
    model = model_class(config)
    return _configure_attention_backend(model)


def _load_full_checkpoint_weights(model, checkpoint_dir: Path):
    """Load full trained weights while keeping architecture code in the base model."""
    safe_weights = checkpoint_dir / SAFE_WEIGHTS_NAME
    torch_weights = checkpoint_dir / WEIGHTS_NAME
    if safe_weights.is_file() or torch_weights.is_file():
        checkpoint_file = safe_weights if safe_weights.is_file() else torch_weights
        state_dict = load_state_dict(str(checkpoint_file))
        model.load_state_dict(state_dict, strict=True)
        return model

    load_sharded_checkpoint(model, checkpoint_dir, strict=True, prefer_safe=True)
    return model


def load_trained_classifier(
    model_dir: str | Path,
    base_model_dir: str | Path | None = None,
    architecture_dir: str | Path | None = None,
    num_labels: int | None = None,
):
    """Load trained weights with architecture code kept outside weight folders."""
    path = Path(model_dir).expanduser().resolve()
    metadata = _load_training_metadata(path)
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(path, local_files_only=True)
        base_path = (
            base_model_dir
            or metadata.get("base_model_dir")
            or peft_config.base_model_name_or_path
        )
        if not base_path:
            raise ValueError("A base model directory is required for the PEFT adapter")
        effective_num_labels = num_labels or metadata.get("num_labels")
        if effective_num_labels is None:
            raise ValueError("num_labels is required when loading a PEFT adapter")
        architecture_path = architecture_dir or base_path
        base_model = load_sequence_classifier(
            base_path,
            num_labels=effective_num_labels,
            architecture_dir=architecture_path,
        )
        model = PeftModel.from_pretrained(base_model, path, local_files_only=True)
        return _configure_attention_backend(model)

    _validate_full_checkpoint(path)
    architecture_path = architecture_dir or base_model_dir or metadata.get("base_model_dir")
    if not architecture_path:
        raise ValueError(
            "An architecture directory is required to load DNABERT-2 model code"
        )
    effective_num_labels = num_labels or metadata.get("num_labels")
    if effective_num_labels is None:
        raise ValueError(
            "num_labels is required when checkpoint metadata does not define it"
        )
    model = _build_sequence_classifier(
        architecture_path,
        num_labels=effective_num_labels,
    )
    return _load_full_checkpoint_weights(model, path)


class DNABert2Encoder:
    """High-level batched embedding interface for a local DNABERT-2 checkpoint."""

    def __init__(
        self,
        model_dir: str | Path,
        architecture_dir: str | Path | None = None,
        device: str | None = None,
        model_max_length: int = 512,
    ):
        self.model_dir = validate_model_directory(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = load_tokenizer(self.model_dir, model_max_length=model_max_length)
        model = _load_pretrained_model(
            self.model_dir,
            architecture_dir or self.model_dir,
            "AutoModel",
        )
        self.model = _configure_attention_backend(model).to(self.device)
        self.model.eval()
        self.model_max_length = model_max_length

    @torch.inference_mode()
    def embed(
        self,
        sequences: Sequence[str],
        batch_size: int = 8,
        pooling: Literal["mean", "max", "cls"] = "mean",
    ) -> np.ndarray:
        """Return one pooled embedding per DNA sequence."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not sequences:
            raise ValueError("At least one DNA sequence is required")
        outputs = []
        for start in range(0, len(sequences), batch_size):
            batch = list(sequences[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.model_max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            model_output = self.model(**encoded)
            hidden_states = (
                model_output.last_hidden_state
                if hasattr(model_output, "last_hidden_state")
                else model_output[0]
            )
            pooled = pool_hidden_states(hidden_states, encoded["attention_mask"], pooling)
            outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)
