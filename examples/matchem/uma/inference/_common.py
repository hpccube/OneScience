"""Shared command-line and path helpers for UMA inference examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_RELATIVE = Path("UMA/checkpoint/uma-s-1p1_converted.pt")
DEFAULT_BATCH_DATA_RELATIVE = Path(
    "matchem/omat24/val/rattled-300-subsampled/val.aselmdb"
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add options shared by all standalone inference examples."""
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        default=None,
        help=(
            "UMA checkpoint. Defaults to "
            "${ONESCIENCE_MODELS_DIR}/UMA/checkpoint/"
            "uma-s-1p1_converted.pt."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for inference (default: cuda).",
    )


def resolve_model_path(model_path: str | None) -> str:
    """Resolve a user-supplied checkpoint or the shared UMA default."""
    if model_path:
        path = Path(model_path).expanduser()
    else:
        models_root = os.environ.get("ONESCIENCE_MODELS_DIR")
        if not models_root:
            raise RuntimeError(
                "ONESCIENCE_MODELS_DIR is not set. Source matchem_env.sh or "
                "pass --model-path explicitly."
            )
        path = Path(models_root) / DEFAULT_MODEL_RELATIVE

    if not path.is_file():
        raise FileNotFoundError(
            f"UMA checkpoint not found: {path}. Pass --model-path to a valid "
            "checkpoint or check ONESCIENCE_MODELS_DIR."
        )
    return str(path.resolve())


def resolve_dataset_path(data_path: str | None) -> str:
    """Resolve a user dataset path or the shared OMat24 validation example."""
    if data_path:
        path = Path(data_path).expanduser()
    else:
        datasets_root = os.environ.get("ONESCIENCE_DATASETS_DIR")
        if not datasets_root:
            raise RuntimeError(
                "ONESCIENCE_DATASETS_DIR is not set. Source matchem_env.sh or "
                "pass --data-path explicitly."
            )
        path = Path(datasets_root) / DEFAULT_BATCH_DATA_RELATIVE

    if not path.exists():
        raise FileNotFoundError(
            f"Inference dataset not found: {path}. Pass --data-path to a "
            "valid ASE-LMDB path."
        )
    return str(path.resolve())


def output_path(output_dir: str, filename: str) -> str:
    """Create an output directory and return a path inside it."""
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / filename)
