#!/usr/bin/env python3
"""Lightweight validation for the OneScience GenCast model."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path


ONESCIENCE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(ONESCIENCE_SRC))

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from onescience.models.gencast import (
    GenCastModel,
    gencast_core,
    graphcast_core,
    parameter_count,
    xarray_jax,
)


warnings.filterwarnings("ignore", message="Changing the sparsity structure")

BATCH_SIZE = 1
HEIGHT = 9
WIDTH = 16
LEVELS = tuple(graphcast_core.PRESSURE_LEVELS_WEATHERBENCH_13)
ATMOSPHERIC_VARIABLES = tuple(graphcast_core.TARGET_ATMOSPHERIC_VARS)

MODEL_CONFIG = {
    "data": {"reintroduce_sst_nans": False},
    "model": {
        "mesh_size": 1,
        "latent_size": 32,
        "hidden_layers": 1,
        "radius_query_fraction_edge_length": 0.6,
        "attention_k_hop": 1,
        "attention_type": "triblockdiag_mha",
        "mask_type": "full",
        "num_layers": 1,
        "num_heads": 4,
        "ffw_hidden": 128,
    },
    "sampler": {
        "max_noise_level": 80.0,
        "min_noise_level": 0.03,
        "num_noise_levels": 2,
        "rho": 7.0,
        "stochastic_churn_rate": 0.0,
        "churn_min_noise_level": 0.75,
        "churn_max_noise_level": float("inf"),
        "noise_level_inflation_factor": 1.0,
    },
}


def _field(shape: tuple[int, ...], offset: float) -> jax.Array:
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    return offset + values / max(int(values.size), 1)


def _make_inputs_targets_forcings():
    lat = np.linspace(-90.0, 90.0, HEIGHT, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, WIDTH, endpoint=False, dtype=np.float32)
    input_time = np.asarray([-12, 0], dtype="timedelta64[h]")
    target_time = np.asarray([12], dtype="timedelta64[h]")

    input_vars = {}
    target_vars = {}
    for index, name in enumerate(gencast_core.TARGET_SURFACE_NO_PRECIP_VARS):
        input_vars[name] = (
            ("batch", "time", "lat", "lon"),
            _field((BATCH_SIZE, 2, HEIGHT, WIDTH), index * 0.01),
        )
    for index, name in enumerate(ATMOSPHERIC_VARIABLES):
        input_vars[name] = (
            ("batch", "time", "level", "lat", "lon"),
            _field(
                (BATCH_SIZE, 2, len(LEVELS), HEIGHT, WIDTH),
                index * 0.01,
            ),
        )

    for index, name in enumerate(gencast_core.TARGET_SURFACE_VARS):
        target_vars[name] = (
            ("batch", "time", "lat", "lon"),
            _field((BATCH_SIZE, 1, HEIGHT, WIDTH), 0.1 + index * 0.01),
        )
    for index, name in enumerate(ATMOSPHERIC_VARIABLES):
        target_vars[name] = (
            ("batch", "time", "level", "lat", "lon"),
            _field(
                (BATCH_SIZE, 1, len(LEVELS), HEIGHT, WIDTH),
                0.1 + index * 0.01,
            ),
        )

    input_vars.update(
        {
            "year_progress_sin": (
                ("batch", "time"),
                jnp.zeros((BATCH_SIZE, 2), dtype=jnp.float32),
            ),
            "year_progress_cos": (
                ("batch", "time"),
                jnp.ones((BATCH_SIZE, 2), dtype=jnp.float32),
            ),
            "day_progress_sin": (
                ("batch", "time", "lon"),
                jnp.zeros((BATCH_SIZE, 2, WIDTH), dtype=jnp.float32),
            ),
            "day_progress_cos": (
                ("batch", "time", "lon"),
                jnp.ones((BATCH_SIZE, 2, WIDTH), dtype=jnp.float32),
            ),
            "geopotential_at_surface": (
                ("lat", "lon"),
                jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32),
            ),
            "land_sea_mask": (
                ("lat", "lon"),
                jnp.zeros((HEIGHT, WIDTH), dtype=jnp.float32),
            ),
        }
    )

    forcing_vars = {
        "year_progress_sin": (
            ("batch", "time"),
            jnp.zeros((BATCH_SIZE, 1), dtype=jnp.float32),
        ),
        "year_progress_cos": (
            ("batch", "time"),
            jnp.ones((BATCH_SIZE, 1), dtype=jnp.float32),
        ),
        "day_progress_sin": (
            ("batch", "time", "lon"),
            jnp.zeros((BATCH_SIZE, 1, WIDTH), dtype=jnp.float32),
        ),
        "day_progress_cos": (
            ("batch", "time", "lon"),
            jnp.ones((BATCH_SIZE, 1, WIDTH), dtype=jnp.float32),
        ),
    }

    common_coords = {
        "batch": np.arange(BATCH_SIZE),
        "level": np.asarray(LEVELS, dtype=np.int32),
        "lat": lat,
        "lon": lon,
    }
    inputs = xarray_jax.Dataset(
        input_vars,
        coords={**common_coords, "time": input_time},
    )
    targets = xarray_jax.Dataset(
        target_vars,
        coords={**common_coords, "time": target_time},
    )
    forcings = xarray_jax.Dataset(
        forcing_vars,
        coords={
            "batch": common_coords["batch"],
            "time": target_time,
            "lon": lon,
        },
    )
    return inputs, targets, forcings


def _stat_dataset(names: set[str], value: float) -> xarray.Dataset:
    data_vars = {}
    for name in sorted(names):
        if name in ATMOSPHERIC_VARIABLES:
            data_vars[name] = xarray.DataArray(
                np.full(len(LEVELS), value, dtype=np.float32),
                dims=("level",),
                coords={"level": np.asarray(LEVELS, dtype=np.int32)},
            )
        else:
            data_vars[name] = xarray.DataArray(np.float32(value))
    return xarray.Dataset(data_vars)


def _make_stats() -> dict[str, xarray.Dataset]:
    input_names = set(gencast_core.TASK.input_variables)
    target_names = set(gencast_core.TASK.target_variables)
    return {
        "mean_by_level": _stat_dataset(input_names | target_names, 0.0),
        "stddev_by_level": _stat_dataset(input_names | target_names, 1.0),
        "diffs_stddev_by_level": _stat_dataset(input_names & target_names, 1.0),
        "min_by_level": _stat_dataset({"sea_surface_temperature"}, 0.0),
    }


def main() -> None:
    inputs, targets, forcings = _make_inputs_targets_forcings()
    model = GenCastModel.from_config_and_stats(MODEL_CONFIG, _make_stats())

    init_rng, loss_rng = jax.random.split(jax.random.PRNGKey(42))
    params, state = model.init(init_rng, inputs, targets, forcings)
    (loss, diagnostics), _ = model.loss(
        params, state, loss_rng, inputs, targets, forcings
    )
    loss_value = float(jax.device_get(loss))
    num_params = parameter_count(params)
    diagnostics_finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(diagnostics)
    )

    assert num_params > 0
    assert targets.sizes["batch"] == BATCH_SIZE
    assert targets.sizes["time"] == 1
    assert np.isfinite(loss_value)
    assert diagnostics_finite

    print(f"JAX devices: {jax.devices()}")
    print(f"GenCast parameter count: {num_params}")
    print("Function: GenCast Model Loss Forward")
    print(f"loss: {loss_value:.8f}")
    print(f"target sizes: {dict(targets.sizes)}")
    print("GenCast validation passed\n")


if __name__ == "__main__":
    main()
