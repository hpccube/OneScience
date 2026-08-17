"""Model construction and checkpoint loading for released scGPT weights."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from onescience.datapipes.scgpt.tokenizer import GeneVocab
from onescience.utils.scgpt.util import load_pretrained

from .model import TransformerModel


PathLike = Union[str, Path]


def validate_model_directory(model_dir: PathLike) -> Path:
    """Validate a released scGPT checkpoint directory."""
    model_dir = Path(model_dir).expanduser().resolve()
    required = ("args.json", "vocab.json", "best_model.pt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid scGPT model directory {model_dir}; missing: {', '.join(missing)}"
        )
    return model_dir


def load_model_config(model_dir: PathLike) -> Dict[str, Any]:
    """Load the model arguments stored with a released checkpoint."""
    model_dir = validate_model_directory(model_dir)
    with (model_dir / "args.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_model_and_vocab(
    model_dir: PathLike,
    *,
    n_cls: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    use_fast_transformer: bool = False,
    dropout: Optional[float] = None,
    strict: bool = False,
    load_weights: bool = True,
) -> Tuple[TransformerModel, GeneVocab, Dict[str, Any]]:
    """Build scGPT and load compatible parameters from a released checkpoint."""
    model_dir = validate_model_directory(model_dir)
    config = load_model_config(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    pad_token = config.get("pad_token", "<pad>")
    vocab.set_default_index(vocab[pad_token])

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nheads"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        nlayers_cls=config.get("n_layers_cls", 3),
        n_cls=n_cls,
        vocab=vocab,
        dropout=config.get("dropout", 0.2) if dropout is None else dropout,
        pad_token=pad_token,
        pad_value=config.get("pad_value", -2),
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=config.get("pre_norm", False),
    )
    map_location = torch.device(device) if device is not None else torch.device("cpu")
    if load_weights:
        checkpoint = torch.load(
            model_dir / "best_model.pt",
            map_location=map_location,
            weights_only=True,
        )
        load_pretrained(model, checkpoint, strict=strict, verbose=False)
    model.to(map_location)
    return model, vocab, config


__all__ = [
    "load_model_and_vocab",
    "load_model_config",
    "validate_model_directory",
]
