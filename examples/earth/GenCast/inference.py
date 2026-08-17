#!/usr/bin/env python3
"""使用官方 GenCast DPM-Solver++ 执行集合自回归推理。"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Changing the sparsity structure")

PROJECT_ROOT = Path(__file__).resolve().parent

from common import configure_jax, load_config, load_stats, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf/config.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-members", type=int)
    parser.add_argument("--prediction-steps", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_jax(config["runtime"].get("platform", "auto"))

    import jax
    import numpy as np
    import xarray

    from onescience.models.gencast import (
        GenCastModel,
        load_model_checkpoint,
        rollout,
    )
    from common import (
        load_trainer_checkpoint, validate_checkpoint_config,
    )
    from data_loader import GenCastERA5Dataset

    prediction_steps = int(args.prediction_steps or config["inference"]["prediction_steps"])
    num_members = int(args.num_members or config["inference"]["num_members"])
    stats = load_stats(config["data"]["stats_dir"])
    checkpoint_path = args.checkpoint or config["inference"].get("official_checkpoint")
    if checkpoint_path:
        official = load_model_checkpoint(resolve_path(checkpoint_path))
        model = GenCastModel.from_checkpoint_and_stats(
            official,
            stats,
            attention_type=config["inference"].get("attention_type_override"),
        )
        params, state = official.params, {}
        task_config = official.task_config
    else:
        model = GenCastModel.from_config_and_stats(config, stats)
        params, state, _, _, saved_config = load_trainer_checkpoint(
            config["checkpoint"]["trainer"]
        )
        validate_checkpoint_config(config, saved_config, scope="inference")
        task_config = model.task_config

    dataset = GenCastERA5Dataset(
        resolve_path(config["data"]["data_dir"]),
        list(config["data"]["test_years"]),
        static_dir=resolve_path(config["data"]["static_dir"]),
        prediction_steps=prediction_steps,
        stride=int(config["data"].get("test_stride", 1)),
        task_config=task_config,
        precipitation_interval_hours=int(
            config["data"]["precipitation_interval_hours"]
        ),
        load_future_targets=False,
    )
    inputs, targets, forcings = dataset[args.sample_index]

    def forward(rng, inputs, targets_template, forcings):
        return model.predict(
            params, state, rng, inputs, targets_template, forcings
        )[0]

    forward = jax.jit(forward)
    seed = int(config["inference"]["seed"])
    rngs = np.stack([jax.random.fold_in(jax.random.PRNGKey(seed), i) for i in range(num_members)])
    chunks = rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=forward,
        rngs=rngs,
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
        num_steps_per_chunk=1,
        num_samples=num_members,
        pmap_devices=None,
    )
    output = resolve_path(args.output or config["output"]["prediction"])
    if bool(config["inference"].get("stream_chunks", True)):
        output_dir = output.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
        for chunk_index, chunk in enumerate(chunks):
            host_chunk = jax.device_get(chunk)
            member = int(host_chunk.coords["sample"])
            lead = int(host_chunk.time.values[0] / np.timedelta64(1, "h"))
            host_chunk = host_chunk.drop_vars("sample").assign_coords(time=[lead])
            host_chunk.coords["time"].attrs = {"long_name": "forecast lead time hours"}
            host_chunk.attrs.update(
                model="GenCast", target_channel_count=84,
                forecast_reference_time=inputs.attrs["forecast_reference_time"],
            )
            path = output_dir / f"member_{member:03d}_lead_{lead:04d}h.nc"
            host_chunk.to_netcdf(path)
            print(f"Saved prediction chunk to {path}")
        return

    chunks = list(chunks)
    member_chunks: list[list[xarray.Dataset]] = [[] for _ in range(num_members)]
    for chunk in chunks:
        host_chunk = jax.device_get(chunk)
        member = int(host_chunk.coords["sample"])
        member_chunks[member].append(host_chunk.drop_vars("sample"))
    members = [
        xarray.concat(parts, dim="time").expand_dims(sample=[member])
        for member, parts in enumerate(member_chunks)
    ]
    predictions = xarray.concat(members, dim="sample")
    predictions.attrs.update(
        model="GenCast",
        target_channel_count=84,
        ensemble_members=num_members,
        step_hours=12,
        forecast_reference_time=inputs.attrs["forecast_reference_time"],
    )
    # Store lead time as plain hours; xarray_jax's internal dtype attribute is
    # not valid CF metadata and conflicts with decoding after NetCDF round-trip.
    lead_hours = (
        predictions.coords["time"].values / np.timedelta64(1, "h")
    ).astype(np.int32)
    predictions = predictions.assign_coords(time=("time", lead_hours))
    predictions.coords["time"].attrs = {
        "long_name": "forecast lead time",
        "units": "hours",
    }
    reference_time = np.datetime64(inputs.attrs["forecast_reference_time"])
    predictions = predictions.assign_coords(
        valid_time=("time", reference_time + lead_hours.astype("timedelta64[h]"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    predictions.to_netcdf(temporary)
    temporary.replace(output)
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
