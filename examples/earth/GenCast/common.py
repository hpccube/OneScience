from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def configure_jax(platform: str) -> None:
    if platform != "auto" and "JAX_PLATFORM_NAME" not in os.environ:
        os.environ["JAX_PLATFORM_NAME"] = platform


def load_stats(stats_dir: str | Path) -> dict[str, xarray.Dataset]:
    directory = resolve_path(stats_dir)
    names = (
        "diffs_stddev_by_level",
        "mean_by_level",
        "stddev_by_level",
        "min_by_level",
    )
    stats = {}
    for name in names:
        path = directory / f"{name}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing GenCast statistic: {path}")
        stats[name] = xarray.load_dataset(path).compute()
    from onescience.models.gencast import gencast_core, graphcast_core

    inputs = set(gencast_core.TASK.input_variables) - set(
        graphcast_core.GENERATED_FORCING_VARS
    )
    targets = set(gencast_core.TASK.target_variables)
    required_by_stat = {
        "mean_by_level": inputs | (targets - inputs),
        "stddev_by_level": inputs | (targets - inputs),
        "diffs_stddev_by_level": targets & inputs,
        "min_by_level": {"sea_surface_temperature"},
    }
    for stat_name, dataset in stats.items():
        missing = sorted(required_by_stat[stat_name] - set(dataset.data_vars))
        if missing:
            raise ValueError(f"{stat_name} is missing GenCast variables: {missing}")
        for name, values in dataset.data_vars.items():
            array = np.asarray(values)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{stat_name}.{name} contains non-finite values")
            if "level" in values.dims and tuple(values.level.values) != tuple(
                gencast_core.TASK.pressure_levels
            ):
                raise ValueError(f"{stat_name}.{name} does not use GenCast WB13 order")
            if stat_name in ("stddev_by_level", "diffs_stddev_by_level") and np.any(array <= 0):
                raise ValueError(f"{stat_name}.{name} must be strictly positive")
    return stats


def save_trainer_checkpoint(
    path: str | Path,
    *,
    params: Any,
    state: Any,
    optimizer_state: Any,
    step: int,
    config: dict[str, Any],
) -> None:
    import jax

    destination = resolve_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(
        {"params": params, "state": state, "optimizer_state": optimizer_state}
    )
    arrays = {f"leaf_{i}": np.asarray(value) for i, value in enumerate(leaves)}
    arrays["treedef"] = np.array([treedef], dtype=object)
    arrays["step"] = np.asarray(step, dtype=np.int64)
    arrays["config_json"] = np.asarray(json.dumps(config, sort_keys=True))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **arrays)
    os.replace(temporary, destination)


def load_trainer_checkpoint(
    path: str | Path,
) -> tuple[Any, Any, Any, int, dict[str, Any]]:
    import jax

    source_path = resolve_path(path)
    with np.load(source_path, allow_pickle=True) as source:
        treedef = source["treedef"].item()
        leaves = [source[f"leaf_{i}"] for i in range(len(source.files) - 3)]
        tree = jax.tree_util.tree_unflatten(treedef, leaves)
        saved_config = json.loads(str(source["config_json"]))
        return (
            tree["params"], tree["state"], tree["optimizer_state"],
            int(source["step"]), saved_config,
        )


def validate_checkpoint_config(
    current: dict[str, Any],
    saved: dict[str, Any],
    *,
    scope: str = "resume",
) -> None:
    """Validate checkpoint compatibility for training resume or inference."""
    if scope not in ("resume", "inference"):
        raise ValueError("scope must be 'resume' or 'inference'")

    inference_paths = (
        ("model",), ("sampler",), ("data", "stats_dir"),
        ("data", "static_dir"), ("data", "precipitation_interval_hours"),
    )
    resume_only_paths = (
        ("training", "learning_rate"),
        ("training", "betas"), ("training", "epsilon"),
        ("training", "seed"), ("data", "data_dir"),
        ("data", "train_years"), ("data", "train_stride"),
        ("parallel", "mode"), ("parallel", "num_devices"),
        ("parallel", "num_processes"),
        ("parallel", "global_batch_size"), ("parallel", "axis_name"),
    )
    if scope == "resume":
        if "parallel" not in saved:
            saved = dict(saved)
            saved["parallel"] = {
                "mode": "single",
                "num_devices": 1,
                "global_batch_size": 1,
                "axis_name": "devices",
            }
        else:
            saved = dict(saved)
            saved["parallel"] = dict(saved["parallel"])
            saved["parallel"].setdefault("num_processes", 1)
        paths = inference_paths + resume_only_paths
    else:
        paths = inference_paths

    for path in paths:
        current_value: Any = current
        saved_value: Any = saved
        for key in path:
            current_value = current_value[key]
            saved_value = saved_value[key]
        if current_value != saved_value:
            name = ".".join(path)
            raise ValueError(
                f"Trainer checkpoint configuration mismatch for {name} "
                f"during {scope}: "
                f"saved={saved_value!r}, current={current_value!r}"
            )
