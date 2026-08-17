"""将 ERA5 HDF5 严格适配为官方 GenCast xarray 数据协议。"""

from __future__ import annotations

import bisect
import datetime as dt
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import xarray

try:
    from onescience.datapipes.climate import ERA5Dataset as _ERA5Dataset
except ModuleNotFoundError as error:
    if error.name not in ("torch", "onescience"):
        raise

    class _ERA5Dataset:
        """Minimal discovery fallback for JAX-only OneScience environments."""

        def __init__(self, dataset_dir, used_years, used_variables, **_):
            self.dataset_dir = dataset_dir
            self.used_years = used_years
            self.used_variables = used_variables
            self._init_avail_samples()
            self._init_normalized_files()

        def _init_avail_samples(self):
            data_dir = Path(self.dataset_dir) / "data"
            available = {int(path.stem): path for path in data_dir.glob("*.h5")}
            missing_years = sorted(set(self.used_years) - set(available))
            if missing_years:
                raise ValueError(f"Years not found in dataset: {missing_years}")
            first = available[self.used_years[0]]
            with h5py.File(first, "r") as source:
                fields = source["fields"]
                variables = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in fields.attrs["variables"]
                ]
                self.T, self.C, self.H, self.W = fields.shape
                self.time_step = int(fields.attrs["time_step"])
            missing_variables = sorted(set(self.used_variables) - set(variables))
            if missing_variables:
                raise ValueError(f"Variables not found in dataset: {missing_variables}")
            self.file_map = {year: str(available[year]) for year in self.used_years}

        def _init_normalized_files(self):
            pass

from onescience.models.gencast import data_utils
from onescience.models.gencast import gencast_core as gencast
from onescience.models.gencast import graphcast_core as graphcast


PRESSURE_LEVELS = tuple(graphcast.PRESSURE_LEVELS_WEATHERBENCH_13)
SURFACE_VARIABLES = tuple(gencast.TARGET_SURFACE_NO_PRECIP_VARS)
ATMOSPHERIC_VARIABLES = tuple(graphcast.TARGET_ATMOSPHERIC_VARS)
STATIC_VARIABLES = tuple(graphcast.STATIC_VARS)
RAW_PRECIPITATION = "total_precipitation"
TARGET_PRECIPITATION = "total_precipitation_12hr"
MODEL_TARGET_CHANNELS = 6 + 6 * len(PRESSURE_LEVELS)
ERA5_VARIABLES = (
    *SURFACE_VARIABLES,
    RAW_PRECIPITATION,
    *(f"{name}_{level}" for name in ATMOSPHERIC_VARIABLES for level in PRESSURE_LEVELS),
)


def expected_target_channel_names() -> tuple[str, ...]:
    channels: list[str] = []
    atmospheric = set(ATMOSPHERIC_VARIABLES)
    for name in sorted(gencast.TASK.target_variables):
        if name in atmospheric:
            channels.extend(f"{name}_{level}" for level in PRESSURE_LEVELS)
        else:
            channels.append(name)
    return tuple(channels)


class GenCastERA5Dataset(_ERA5Dataset):
    """Reuse ERA5Dataset discovery while enforcing GenCast's named protocol."""

    def __init__(
        self,
        dataset_dir: str | Path,
        used_years: list[int],
        *,
        static_dir: str | Path | None = None,
        prediction_steps: int = 1,
        stride: int = 1,
        task_config: Any = gencast.TASK,
        precipitation_interval_hours: int = 6,
        load_future_targets: bool = True,
    ) -> None:
        super().__init__(
            dataset_dir=str(dataset_dir),
            used_years=used_years,
            used_variables=list(ERA5_VARIABLES),
            input_steps=1,
            output_steps=1,
            normalize=False,
        )
        self.static_dir = Path(static_dir or Path(dataset_dir) / "static")
        self.prediction_steps = int(prediction_steps)
        self.stride = int(stride)
        self.task_config = task_config
        self.precipitation_interval_hours = int(precipitation_interval_hours)
        self.load_future_targets = bool(load_future_targets)
        self._validate_task_config()
        if self.prediction_steps < 1 or self.stride < 1:
            raise ValueError("prediction_steps and stride must be positive")
        self._inspect_years()

    def _init_normalized_files(self) -> None:
        # GenCast uses named by-level NetCDF statistics in the model wrapper.
        self.mu = self.sd = None

    def _inspect_years(self) -> None:
        self._year_meta: list[dict[str, Any]] = []
        self._cumulative: list[int] = []
        total = 0
        for year in self.used_years:
            path = Path(self.file_map[year])
            with h5py.File(path, "r") as source:
                fields = source["fields"]
                variables = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in fields.attrs["variables"]
                ]
                time_step = int(fields.attrs["time_step"])
                shape = tuple(fields.shape)
            if time_step not in (6, 12):
                raise ValueError(f"{path}: GenCast requires 6h or 12h ERA5, got {time_step}h")
            if self.precipitation_interval_hours != time_step:
                raise ValueError(
                    f"{path}: total_precipitation must be an accumulation over each "
                    f"{time_step}h source interval; configured "
                    f"{self.precipitation_interval_hours}h"
                )
            missing = sorted(set(ERA5_VARIABLES) - set(variables))
            if missing:
                raise ValueError(f"{path}: missing GenCast ERA5 variables: {missing}")
            frame_stride = 12 // time_step
            # The -12h input also needs a complete 12h precipitation window.
            first_reference = 2 * frame_stride - 1
            last_reference = (
                shape[0] - frame_stride * self.prediction_steps - 1
                if self.load_future_targets else shape[0] - 1
            )
            references = list(range(first_reference, last_reference + 1, self.stride))
            meta = {
                "year": year,
                "path": path,
                "shape": shape,
                "time_step": time_step,
                "frame_stride": frame_stride,
                "variables": variables,
                "references": references,
            }
            self._year_meta.append(meta)
            total += len(references)
            self._cumulative.append(total)
        self.total_samples = total
        if not total:
            raise ValueError("No complete GenCast samples are available")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        year_index = bisect.bisect_right(self._cumulative, index)
        start = 0 if year_index == 0 else self._cumulative[year_index - 1]
        meta = self._year_meta[year_index]
        reference_index = meta["references"][index - start]
        dataset = self._read_dataset(meta, reference_index)
        return data_utils.extract_inputs_targets_forcings(
            dataset,
            target_lead_times=slice("12h", f"{12 * self.prediction_steps}h"),
            input_variables=self.task_config.input_variables,
            target_variables=self.task_config.target_variables,
            forcing_variables=self.task_config.forcing_variables,
            pressure_levels=self.task_config.pressure_levels,
            input_duration=self.task_config.input_duration,
        )

    def _read_dataset(self, meta: dict[str, Any], reference_index: int) -> xarray.Dataset:
        frame_stride = meta["frame_stride"]
        frame_indices = [
            reference_index - frame_stride,
            reference_index,
            *(reference_index + frame_stride * step for step in range(1, self.prediction_steps + 1)),
        ]
        variable_index = {name: i for i, name in enumerate(meta["variables"])}
        selected_names = list(SURFACE_VARIABLES) + [
            f"{name}_{level}"
            for name in ATMOSPHERIC_VARIABLES
            for level in PRESSURE_LEVELS
        ]
        selected_indices = [variable_index[name] for name in selected_names]
        order = np.argsort(selected_indices)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        read_count = len(frame_indices) if self.load_future_targets else 2
        with h5py.File(meta["path"], "r") as source:
            fields = source["fields"]
            loaded = np.stack([
                fields[t, np.asarray(selected_indices)[order], :, :][inverse]
                for t in frame_indices[:read_count]
            ]).astype(np.float32)
            values = np.full(
                (len(frame_indices), *loaded.shape[1:]), np.nan, dtype=np.float32
            )
            values[:read_count] = loaded
            precipitation = np.full(
                (len(frame_indices), *loaded.shape[-2:]), np.nan, dtype=np.float32
            )
            if self.load_future_targets:
                precipitation[:] = np.stack([
                    self._precipitation_12h(
                        fields, variable_index[RAW_PRECIPITATION], t, frame_stride
                    )
                    for t in frame_indices
                ]).astype(np.float32)

        # OneScience ERA5 uses north-to-south storage; GenCast spherical noise requires ascending lat.
        values = values[..., ::-1, :]
        precipitation = precipitation[..., ::-1, :]
        height, width = values.shape[-2:]
        lat = np.linspace(-90.0, 90.0, height, dtype=np.float32)
        lon = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
        reference_time = dt.datetime(meta["year"], 1, 1) + dt.timedelta(
            hours=reference_index * meta["time_step"]
        )
        datetimes = np.asarray([
            np.datetime64(reference_time + dt.timedelta(hours=(t - reference_index) * meta["time_step"]))
            for t in frame_indices
        ], dtype="datetime64[ns]")
        times = np.asarray([
            np.timedelta64((t - reference_index) * meta["time_step"], "h")
            for t in frame_indices
        ], dtype="timedelta64[ns]")

        data_vars: dict[str, Any] = {}
        cursor = 0
        for name in SURFACE_VARIABLES:
            data_vars[name] = (("batch", "time", "lat", "lon"), values[:, cursor][None])
            cursor += 1
        for name in ATMOSPHERIC_VARIABLES:
            data_vars[name] = (
                ("batch", "time", "level", "lat", "lon"),
                values[:, cursor:cursor + len(PRESSURE_LEVELS)][None],
            )
            cursor += len(PRESSURE_LEVELS)
        data_vars[TARGET_PRECIPITATION] = (
            ("batch", "time", "lat", "lon"), precipitation[None]
        )
        data_vars.update(self._load_static(height, width))
        dataset = xarray.Dataset(
            data_vars=data_vars,
            coords={
                "batch": np.arange(1),
                "time": times,
                "datetime": (("batch", "time"), datetimes[None]),
                "level": np.asarray(PRESSURE_LEVELS, dtype=np.int32),
                "lat": lat,
                "lon": lon,
            },
        )
        dataset.attrs["forecast_reference_time"] = np.datetime_as_string(
            np.datetime64(reference_time), unit="h"
        )
        self.validate_dataset(dataset)
        return dataset

    @staticmethod
    def _precipitation_12h(fields, channel: int, end: int, frame_stride: int):
        start = end - frame_stride + 1
        if start < 0:
            raise IndexError("Insufficient precipitation history for 12h accumulation")
        return np.sum(fields[start:end + 1, channel], axis=0)

    def _load_static(self, height: int, width: int) -> dict[str, Any]:
        paths = {
            "geopotential_at_surface": self.static_dir / "geopotential_at_surface.npy",
            "land_sea_mask": self.static_dir / "land_mask.npy",
        }
        result = {}
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing GenCast static field: {path}")
            values = np.load(path).astype(np.float32)
            if values.shape != (height, width):
                raise ValueError(f"{path}: expected {(height, width)}, got {values.shape}")
            result[name] = (("lat", "lon"), values[::-1])
        return result

    def _validate_task_config(self) -> None:
        expected = gencast.TASK
        for field in (
            "input_variables", "target_variables", "forcing_variables",
            "pressure_levels", "input_duration",
        ):
            if getattr(self.task_config, field) != getattr(expected, field):
                raise ValueError(
                    "This ERA5 adapter supports the official WB13 GenCast task "
                    f"only; checkpoint field {field} differs"
                )

    @staticmethod
    def validate_dataset(dataset: xarray.Dataset) -> None:
        missing = sorted(
            set(gencast.TASK.input_variables + gencast.TASK.target_variables)
            - set(dataset.data_vars)
            - set(graphcast.GENERATED_FORCING_VARS)
        )
        if missing:
            raise ValueError(f"Missing GenCast variables: {missing}")
        if tuple(int(level) for level in dataset.level.values) != PRESSURE_LEVELS:
            raise ValueError("GenCast WB13 pressure-level order changed")
        if not np.all(np.diff(dataset.lat.values) > 0):
            raise ValueError("GenCast latitude must be strictly ascending")
        height, width = dataset.sizes["lat"], dataset.sizes["lon"]
        if width != 2 * (height - 1):
            raise ValueError(
                "GenCast equiangular grids with poles require lon=2*(lat-1), "
                f"got lat={height}, lon={width}"
            )
        if len(expected_target_channel_names()) != MODEL_TARGET_CHANNELS:
            raise AssertionError("The official GenCast target contract must contain 84 channels")


def batch_iterator(
    dataset: GenCastERA5Dataset,
    *,
    shuffle: bool,
    seed: int,
    batch_size: int = 1,
    global_batch_size: int | None = None,
    process_index: int = 0,
    process_count: int = 1,
) -> Iterator:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if process_count < 1 or process_index < 0 or process_index >= process_count:
        raise ValueError("Invalid process index/count")
    global_batch_size = int(global_batch_size or batch_size)
    if global_batch_size != batch_size * process_count:
        raise ValueError(
            "global_batch_size must equal batch_size * process_count: "
            f"{global_batch_size} != {batch_size} * {process_count}"
        )
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices) - global_batch_size + 1, global_batch_size):
        local_start = start + process_index * batch_size
        local_indices = indices[local_start:local_start + batch_size]
        samples = [dataset[int(index)] for index in local_indices]
        if batch_size == 1:
            yield samples[0]
            continue
        yield tuple(
            xarray.concat(values, dim="batch", data_vars="minimal", coords="minimal")
            for values in zip(*samples)
        )
