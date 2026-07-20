#!/usr/bin/env python
# coding: utf-8
"""
Generate AIFS-only ERA5 fake H5 dataset for training & inference.

Follows the onescience fake-data generation pattern (chunked H5 with
fillvalue=0, embedded global_means/global_stds).  Contains exactly the
106 ERA5 variables required by AIFS v1.1 — no more, no less.

Each year file has 60 timesteps (6h × 60 = 360h = 15 days), matching
AIFS's maximum inference lead time.

Usage::

    python fake_data_all.py                          # 2005, 60 steps
    python fake_data_all.py --years 2005,2006        # two full years
    python fake_data_all.py -y 2005 -o ./my_era5      # custom output dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ============================================================================
# AIFS-required ERA5 variables  (106 total, no extras)
# ============================================================================
# Order: surface (12) → soil (4) → pressure-levels (78) → diagnostic (12)
# These exactly match the keys/values in train.py's ERA5_*_MAP dictionaries.

AIFS_ERA5_VARIABLES: List[str] = [
    # ── surface prognostic (12) ─────────────────────────────────────────
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "skin_temperature",
    "surface_pressure",
    "total_column_water",
    "land_sea_mask",
    "geopotential",
    "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography",

    # ── soil prognostic (4) ─────────────────────────────────────────────
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",

    # ── pressure levels: geopotential (13) ──────────────────────────────
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",

    # ── pressure levels: temperature (13) ───────────────────────────────
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",

    # ── pressure levels: u wind (13) ────────────────────────────────────
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",

    # ── pressure levels: v wind (13) ────────────────────────────────────
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",

    # ── pressure levels: vertical velocity (13) ─────────────────────────
    "vertical_velocity_50",
    "vertical_velocity_100",
    "vertical_velocity_150",
    "vertical_velocity_200",
    "vertical_velocity_250",
    "vertical_velocity_300",
    "vertical_velocity_400",
    "vertical_velocity_500",
    "vertical_velocity_600",
    "vertical_velocity_700",
    "vertical_velocity_850",
    "vertical_velocity_925",
    "vertical_velocity_1000",

    # ── pressure levels: specific humidity (13) ─────────────────────────
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",

    # ── diagnostic (12, output-only) ────────────────────────────────────
    "total_precipitation",
    "convective_precipitation",
    "snowfall_water_equivalent",
    "total_cloud_cover",
    "high_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "runoff",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]

# ===========================================================================
# Dataset dimensions
# ===========================================================================
# 60 timesteps @ 6h = 360h = 15 days  (covers AIFS max inference lead time)
_DIMS = {
    "T": 60,
    "H": 721,
    "W": 1440,
    "time_step": 6,
}


# ===========================================================================
# Core generation
# ===========================================================================

def generate_fake_h5(
    output_dir: str,
    var_names: List[str],
    years: List[int],
    dims: dict,
) -> None:
    """Generate one H5 file per year with the correct schema.

    Uses HDF5 chunked datasets with ``fillvalue=0.0`` — unwritten chunks
    return zero, keeping files tiny while preserving the logical shape.
    """
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    T, C, H, W = dims["T"], len(var_names), dims["H"], dims["W"]
    means = np.zeros((1, C, 1, 1), dtype=np.float32)
    stds = np.ones((1, C, 1, 1), dtype=np.float32)
    logical_gib = T * C * H * W * 4 / 1024**3

    for year in years:
        path = os.path.join(data_dir, f"{year}.h5")
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                shape=(T, C, H, W),
                dtype="float32",
                chunks=(1, C, H, W),
                fillvalue=0.0,
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]
            f.create_dataset("global_means", data=means)
            f.create_dataset("global_stds", data=stds)

        size_mb = os.path.getsize(path) / 1024**2
        print(f"  {year}.h5  shape=({T},{C},{H},{W})  "
              f"logical={logical_gib:.1f} GiB  actual={size_mb:.1f} MiB")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate AIFS-only ERA5 fake H5 files",
    )
    p.add_argument("--config", "-c", type=str,
                   default=str(ROOT / "conf" / "config.yaml"),
                   help="Path to config.yaml")
    p.add_argument("--output_dir", "-o", type=str,
                   default=str(ROOT / "fake_era5"),
                   help="Output root directory")
    p.add_argument("--years", "-y", type=str, default=None,
                   help="Comma-separated years (overrides config).  "
                        "Each year = 60 timesteps (15 days @ 6h).")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    # Resolve years: CLI > config train+val+test > default
    if args.years:
        years = [int(y.strip()) for y in args.years.replace("，", ",").split(",")
                 if y.strip()]
    elif os.path.exists(args.config):
        import yaml
        cfg = yaml.safe_load(open(args.config))
        data = cfg.get("data", {})
        raw = (data.get("train_years", [])
               + data.get("val_years", [])
               + data.get("test_years", []))
        years = sorted(set(raw))
        if not years:
            years = [2005]
    else:
        years = [2005]

    dims = _DIMS

    print("=" * 60)
    print("  AIFS v1.1 — Fake ERA5 Dataset Generator")
    print("=" * 60)
    print(f"  Output dir : {os.path.abspath(args.output_dir)}")
    print(f"  Years      : {years}  ({dims['T']} steps = "
          f"{dims['T'] * dims['time_step'] // 24} days each)")
    print(f"  Variables  : {len(AIFS_ERA5_VARIABLES)}  (AIFS-only, no extras)")
    print(f"  Shape      : ({dims['T']}, {len(AIFS_ERA5_VARIABLES)}, "
          f"{dims['H']}, {dims['W']})")
    print(f"  Timestep   : {dims['time_step']}h")
    if not args.years:
        print(f"  (years auto-collected from config: train+val+test)")
    print()

    print("[1/1] Generating per-year H5 files ...")
    generate_fake_h5(args.output_dir, AIFS_ERA5_VARIABLES, years, dims)

    # Quick verification
    diag_count = sum(
        1 for v in AIFS_ERA5_VARIABLES
        if v in {"total_precipitation", "convective_precipitation",
                 "snowfall_water_equivalent", "total_cloud_cover",
                 "high_cloud_cover", "low_cloud_cover", "medium_cloud_cover",
                 "runoff", "surface_solar_radiation_downwards",
                 "surface_thermal_radiation_downwards",
                 "100m_u_component_of_wind", "100m_v_component_of_wind"}
    )
    pl_count = sum(1 for v in AIFS_ERA5_VARIABLES
                   if any(v.endswith(f"_{lvl}") for lvl in
                          [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]))
    sfc_count = len(AIFS_ERA5_VARIABLES) - pl_count - diag_count

    print(f"\n  Variables: {sfc_count} surface/soil + {pl_count} PL + "
          f"{diag_count} diagnostic = {len(AIFS_ERA5_VARIABLES)}")
    print(f"\n✅ Done.  Use with train.py:")
    print(f"   python train.py --dataset_path {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
