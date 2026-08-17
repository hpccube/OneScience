#!/usr/bin/env python3
"""Fit OC20 energy element references before starting model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from onescience.utils.uma.normalization.element_references import (
    fit_linear_references,
)

from train import _expand_path, _loader


def fit_references(
    config_path: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
    num_batches: int | None = None,
) -> Path:
    """Fit and save legacy ``coeff`` references consumed by the OC20 YAML."""

    if batch_size < 2:
        raise ValueError("reference fitting batch_size must be at least 2")
    if num_batches is not None and num_batches < 1:
        raise ValueError("num_batches must be positive")

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    train_path = _expand_path(config.get("train"))
    if not train_path:
        raise ValueError(f"training data is not configured in {config_path}")

    loader = _loader(
        train_path,
        batch_size=batch_size,
        workers=int(config.get("workers", 0)),
        max_samples=config.get("max_train_samples"),
        train=False,
        seed=int(config.get("seed", 0)),
        max_atoms=config.get("max_atoms"),
        load_balancing=False,
    )
    references = fit_linear_references(
        targets=["energy"],
        dataset=loader.dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        num_workers=int(config.get("workers", 0)),
        shuffle=False,
    )

    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix != ".npz":
        raise ValueError("output path must use the .npz extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients = (
        references["energy"].element_references.detach().cpu().numpy()
    )
    np.savez(output_path, coeff=coefficients)
    print(f"Saved energy element references to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OC20 training YAML")
    parser.add_argument("--output", required=True, help="output .npz path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fit_references(
        args.config,
        args.output,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
    )


if __name__ == "__main__":
    main()
