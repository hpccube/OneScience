#!/usr/bin/env python3
"""可视化 GenCast 集合预报结果。"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray

warnings.filterwarnings("ignore", message="Changing the sparsity structure")

PROJECT_ROOT = Path(__file__).resolve().parent

from common import load_config, resolve_path


def load_predictions(path: Path) -> xarray.Dataset:
    """Load predictions from either a single NetCDF or a chunked directory."""
    # Auto-detect: if path.nc doesn't exist but path/ directory does, use chunks
    if not path.exists() and path.suffix == ".nc":
        chunk_dir = path.with_suffix("")
        if chunk_dir.is_dir():
            path = chunk_dir
    if path.is_dir():
        files = sorted(path.glob("member_*_lead_*.nc"))
        if not files:
            raise FileNotFoundError(f"No prediction chunks found in {path}")
        datasets = [xarray.load_dataset(f) for f in files]
        members = sorted(set(int(f.stem.split("_")[1]) for f in files))
        member_parts = []
        for m in members:
            member_ds = [ds for ds, f in zip(datasets, files)
                         if f"member_{m:03d}_lead_" in str(f)]
            member_parts.append(xarray.concat(member_ds, dim="time"))
        return xarray.concat(member_parts, dim="sample")
    else:
        return xarray.load_dataset(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf/config.yaml"))
    parser.add_argument("--prediction")
    parser.add_argument("--variable", default="2m_temperature")
    parser.add_argument("--output")
    parser.add_argument("--lead", type=int, default=0, help="lead time index (0-based)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prediction_path = resolve_path(args.prediction or config["output"]["prediction"])
    output_path = resolve_path(args.output or config["output"]["plot"])
    prediction = load_predictions(prediction_path)[args.variable]

    sample_dim = "sample" if "sample" in prediction.dims else None
    mean = prediction.mean(sample_dim) if sample_dim else prediction
    spread = prediction.std(sample_dim) if sample_dim else xarray.zeros_like(mean)

    # Select lead time index (0=first, -1=last)
    lead_idx = args.lead
    field = mean.isel(batch=0, time=lead_idx).values
    spread_field = spread.isel(batch=0, time=lead_idx).values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(field, origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_title(f"GenCast ensemble mean: {args.variable}")
    fig.colorbar(im0, ax=axes[0], orientation="horizontal")
    im1 = axes[1].imshow(spread_field, origin="lower", cmap="magma", aspect="auto")
    axes[1].set_title("Ensemble spread")
    fig.colorbar(im1, ax=axes[1], orientation="horizontal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
