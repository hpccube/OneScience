#!/usr/bin/env python3
"""生成确定性的 ERA5、GenCast 统计量和静态场。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import xarray
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
SURFACE = (
    "2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
    "10m_u_component_of_wind", "sea_surface_temperature", "total_precipitation",
)
ATMOSPHERIC = (
    "temperature", "geopotential", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "specific_humidity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf/config.yaml"))
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stat_dataset(value: float, *, minimum: bool = False) -> xarray.Dataset:
    data_vars = {}
    surface_targets = (
        "2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
        "10m_u_component_of_wind", "sea_surface_temperature", "total_precipitation_12hr",
    )
    for name in surface_targets:
        scalar = -3.0 if minimum and name == "sea_surface_temperature" else value
        data_vars[name] = xarray.DataArray(np.float32(scalar))
    for name in ATMOSPHERIC:
        data_vars[name] = xarray.DataArray(
            np.full(len(PRESSURE_LEVELS), value, dtype=np.float32),
            dims=("level",),
            coords={"level": np.asarray(PRESSURE_LEVELS, dtype=np.int32)},
        )
    # Generated forcings already in [-1, 1] — include so normalization is silent
    for name in ("year_progress_sin", "year_progress_cos",
                 "day_progress_sin", "day_progress_cos"):
        data_vars[name] = xarray.DataArray(np.float32(value))
    for name in ("geopotential_at_surface", "land_sea_mask"):
        data_vars[name] = xarray.DataArray(np.float32(value))
    return xarray.Dataset(data_vars)


def main() -> None:
    args = parse_args()
    with Path(args.config).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    root = PROJECT_ROOT / config["data"]["data_dir"]
    height = int(args.height or config["fake_data"]["height"])
    width = int(args.width or config["fake_data"]["width"])
    timesteps = int(args.timesteps or config["fake_data"]["timesteps"])
    if width != 2 * (height - 1):
        raise ValueError("Synthetic GenCast grid must satisfy width=2*(height-1)")
    variables = list(SURFACE) + [f"{name}_{level}" for name in ATMOSPHERIC for level in PRESSURE_LEVELS]
    years = sorted(set(config["data"]["train_years"] + config["data"]["test_years"]))
    rng = np.random.default_rng(args.seed)
    (root / "data").mkdir(parents=True, exist_ok=True)
    for year in years:
        path = root / "data" / f"{year}.h5"
        time = np.arange(timesteps, dtype=np.float32)[:, None, None, None]
        channel = np.arange(len(variables), dtype=np.float32)[None, :, None, None]
        lat = np.linspace(1.0, -1.0, height, dtype=np.float32)[None, None, :, None]
        lon = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False, dtype=np.float32)[None, None, None, :]
        values = 0.01 * time + 0.001 * channel + 0.1 * lat + 0.05 * np.sin(lon)
        values += rng.normal(0.0, 1e-4, values.shape).astype(np.float32)
        precip_index = variables.index("total_precipitation")
        values[:, precip_index] = np.maximum(values[:, precip_index], 0.0)
        sst_index = variables.index("sea_surface_temperature")
        values[:, sst_index, : height // 4] = np.nan
        with h5py.File(path, "w") as output:
            fields = output.create_dataset("fields", data=values, chunks=(1, len(variables), height, width))
            fields.attrs["variables"] = variables
            fields.attrs["time_step"] = 6
        print(f"Generated {path} shape={values.shape}")

    static_dir = root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        static_dir / "geopotential_at_surface.npy",
        np.zeros((height, width), dtype=np.float32),
    )
    land = np.zeros((height, width), dtype=np.float32)
    land[: height // 4] = 1.0
    np.save(static_dir / "land_mask.npy", land)
    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    _stat_dataset(0.0).to_netcdf(stats_dir / "mean_by_level.nc")
    _stat_dataset(1.0).to_netcdf(stats_dir / "stddev_by_level.nc")
    _stat_dataset(1.0).to_netcdf(stats_dir / "diffs_stddev_by_level.nc")
    _stat_dataset(0.0, minimum=True).to_netcdf(stats_dir / "min_by_level.nc")
    print(f"Generated GenCast statistics and static fields under {root}")


if __name__ == "__main__":
    main()
