#!/usr/bin/env python3
"""使用官方单步 EDM 去噪目标训练 GenCast。"""

from __future__ import annotations

import argparse
import itertools
import os
import socket
import warnings
from pathlib import Path

import xarray

# Mesh adjacency construction triggers one-time scipy CSR restructure warning.
warnings.filterwarnings("ignore", message="Changing the sparsity structure")

PROJECT_ROOT = Path(__file__).resolve().parent

from common import configure_jax, load_config, load_stats, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf/config.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--parallel-mode", choices=("single", "pmap"))
    parser.add_argument("--num-devices", type=int)
    parser.add_argument(
        "--num-processes", type=int,
        help="JAX process count; one process is normally launched per host",
    )
    parser.add_argument(
        "--process-index", type=int,
        help="JAX process index; defaults to SLURM_PROCID/JAX_PROCESS_INDEX",
    )
    parser.add_argument(
        "--coordinator-address",
        help="host:port address used by jax.distributed.initialize",
    )
    parser.add_argument("--coordinator-port", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _adam_init(params):
    import jax
    import jax.numpy as jnp

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"count": jnp.asarray(0, dtype=jnp.int32), "mu": zeros, "nu": zeros}


def _adam_update(params, grads, state, learning_rate, beta1, beta2, eps):
    import jax
    import jax.numpy as jnp

    count = state["count"] + 1
    mu = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state["mu"], grads,
    )
    nu = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state["nu"], grads,
    )
    mu_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta1**count), mu)
    nu_hat = jax.tree_util.tree_map(lambda value: value / (1.0 - beta2**count), nu)
    params = jax.tree_util.tree_map(
        lambda value, first, second: value - learning_rate * first / (jnp.sqrt(second) + eps),
        params, mu_hat, nu_hat,
    )
    return params, {"count": count, "mu": mu, "nu": nu}


def _replicate(tree, devices):
    import jax

    return jax.device_put_replicated(tree, devices)


def _unreplicate(tree):
    import jax

    return jax.tree_util.tree_map(lambda value: value[0], tree)


def _device_batch(batch, device_count):
    """Add a leading device dimension to each GenCast xarray input."""
    result = []
    for value in batch:
        if not isinstance(value, xarray.Dataset):
            raise TypeError("GenCast batches must contain xarray.Dataset values")
        value = value.transpose("batch", ...)
        if "batch" not in value.dims:
            value = value.expand_dims("batch")
        if value.sizes["batch"] % device_count:
            raise ValueError("Batch size must be divisible by the device count")
        local_batch = value.sizes["batch"] // device_count
        shards = [
            value.isel(batch=slice(index * local_batch, (index + 1) * local_batch))
            for index in range(device_count)
        ]
        result.append(xarray.concat(shards, dim="device"))
    return tuple(result)


def _parallel_runtime(args, parallel):
    """Resolve the multi-host process topology without requiring SLURM."""
    configured_processes = int(parallel.get("num_processes", 0) or 0)
    process_count = int(
        args.num_processes
        or configured_processes
        or os.environ.get("JAX_NUM_PROCESSES", "")
        or os.environ.get("SLURM_NTASKS", "1")
    )
    process_index = int(
        args.process_index
        if args.process_index is not None
        else os.environ.get("JAX_PROCESS_INDEX", os.environ.get("SLURM_PROCID", "0"))
    )
    coordinator_address = (
        args.coordinator_address
        or os.environ.get("JAX_COORDINATOR_ADDRESS")
        or parallel.get("coordinator_address")
    )
    coordinator_port = int(
        args.coordinator_port
        or os.environ.get("JAX_COORDINATOR_PORT", "")
        or parallel.get("coordinator_port", 12355)
    )
    if process_count < 1:
        raise ValueError("parallel.num_processes must be positive")
    if process_index < 0 or process_index >= process_count:
        raise ValueError(
            f"process index {process_index} is outside [0, {process_count})"
        )
    if process_count > 1 and not coordinator_address:
        raise ValueError(
            "Multi-host training requires --coordinator-address or "
            "JAX_COORDINATOR_ADDRESS"
        )
    return process_count, process_index, coordinator_address, coordinator_port


def _initialize_jax_distributed(
    jax,
    *,
    process_count,
    process_index,
    coordinator_address,
    coordinator_port,
):
    """Initialize JAX's cross-host collective runtime exactly once."""
    if process_count == 1:
        return
    if ":" not in coordinator_address:
        coordinator_address = f"{coordinator_address}:{coordinator_port}"
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=process_count,
        process_id=process_index,
        initialization_timeout=120,
    )


def _sync_global_devices(jax, name):
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    parallel = config.setdefault("parallel", {})
    if args.parallel_mode is not None:
        parallel["mode"] = args.parallel_mode
    if args.num_devices is not None:
        parallel["num_devices"] = args.num_devices
    if args.global_batch_size is not None:
        parallel["global_batch_size"] = args.global_batch_size
    if args.checkpoint is not None:
        config["checkpoint"]["trainer"] = args.checkpoint
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    configure_jax(config["runtime"].get("platform", "auto"))

    import jax
    import jax.numpy as jnp

    process_count, process_index, coordinator_address, coordinator_port = (
        _parallel_runtime(args, parallel)
    )
    _initialize_jax_distributed(
        jax,
        process_count=process_count,
        process_index=process_index,
        coordinator_address=coordinator_address,
        coordinator_port=coordinator_port,
    )
    if jax.process_count() != process_count or jax.process_index() != process_index:
        raise RuntimeError(
            "JAX initialized with an unexpected process topology: "
            f"expected {process_index}/{process_count}, got "
            f"{jax.process_index()}/{jax.process_count()}"
        )
    parallel["num_processes"] = process_count
    print(
        f"JAX process {process_index}/{process_count} initialized on "
        f"{socket.gethostname()}"
    )

    from onescience.models.gencast import GenCastModel, parameter_count, xarray_jax
    from common import (
        load_trainer_checkpoint, save_trainer_checkpoint,
        validate_checkpoint_config,
    )
    from data_loader import GenCastERA5Dataset, batch_iterator

    mode = str(parallel.get("mode", "single")).lower()
    if mode not in ("single", "pmap"):
        raise ValueError("parallel.mode must be 'single' or 'pmap'")
    local_devices = list(jax.local_devices())
    requested_devices = int(parallel.get("num_devices", 1))
    if requested_devices < 1:
        raise ValueError("parallel.num_devices must be positive")
    if mode == "pmap":
        if requested_devices > len(local_devices):
            raise ValueError(
                f"Requested {requested_devices} local devices, only "
                f"{len(local_devices)} available on process {process_index}"
            )
        devices = local_devices[:requested_devices]
    else:
        requested_devices = 1
        devices = local_devices[:1]
    if process_count > 1 and mode != "pmap":
        raise ValueError("Multi-host training requires parallel.mode='pmap'")
    if process_count > 1 and len(local_devices) != len(jax.local_devices()):
        raise AssertionError("JAX local device discovery changed during setup")
    if process_count > 1:
        pmap_devices = []
        for host_index in range(process_count):
            host_devices = [
                device
                for device in jax.devices()
                if device.process_index == host_index
            ]
            if len(host_devices) < requested_devices:
                raise ValueError(
                    f"Process {host_index} exposes only {len(host_devices)} devices; "
                    f"{requested_devices} requested per process"
                )
            pmap_devices.extend(host_devices[:requested_devices])
    else:
        pmap_devices = devices
    global_batch_size = int(parallel.get("global_batch_size", requested_devices))
    global_device_count = process_count * requested_devices
    if global_batch_size < 1 or global_batch_size % global_device_count:
        raise ValueError(
            "global_batch_size must be divisible by all devices: "
            f"global_batch_size={global_batch_size}, "
            f"processes={process_count}, local_devices={requested_devices}"
        )
    local_batch_size = global_batch_size // process_count
    stats = load_stats(config["data"]["stats_dir"])
    model = GenCastModel.from_config_and_stats(config, stats)
    dataset = GenCastERA5Dataset(
        resolve_path(config["data"]["data_dir"]),
        list(config["data"]["train_years"]),
        static_dir=resolve_path(config["data"]["static_dir"]),
        prediction_steps=1,
        stride=int(config["data"].get("train_stride", 1)),
        precipitation_interval_hours=int(
            config["data"]["precipitation_interval_hours"]
        ),
    )
    first_batch = dataset[0]
    seed = int(config["training"]["seed"])
    start_step = 0
    resume = args.resume or config["checkpoint"].get("resume")
    if resume:
        params, state, optimizer_state, start_step, saved_config = \
            load_trainer_checkpoint(resume)
        validate_checkpoint_config(config, saved_config)
    else:
        params, state = model.init(
            jax.random.fold_in(jax.random.PRNGKey(seed), -1), *first_batch
        )
        optimizer_state = _adam_init(params)

    learning_rate = float(config["training"]["learning_rate"])
    beta1, beta2 = (float(value) for value in config["training"]["betas"])
    epsilon = float(config["training"].get("epsilon", 1e-8))

    def train_step(params, state, optimizer_state, rng, inputs, targets, forcings):
        def objective(current_params, current_state):
            (loss, diagnostics), next_state = model.loss(
                current_params, current_state, rng, inputs, targets, forcings
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            objective, has_aux=True
        )(params, state)
        finite = jnp.logical_and(
            jnp.isfinite(loss),
            jnp.all(jnp.asarray([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(grads)])),
        )
        new_params, new_optimizer_state = _adam_update(
            params, grads, optimizer_state, learning_rate, beta1, beta2, epsilon
        )
        params = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old), new_params, params
        )
        next_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old), next_state, state
        )
        new_optimizer_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(finite, new, old),
            new_optimizer_state,
            optimizer_state,
        )
        return params, next_state, new_optimizer_state, loss, diagnostics, finite

    if mode == "pmap":
        axis_name = str(parallel.get("axis_name", "devices"))

        def parallel_train_step(
            params, state, optimizer_state, rng, inputs, targets, forcings
        ):
            rng = jax.random.fold_in(rng, jax.lax.axis_index(axis_name))

            def objective(current_params, current_state):
                (loss, diagnostics), next_state = model.loss(
                    current_params, current_state, rng, inputs, targets, forcings
                )
                return loss, (diagnostics, next_state)

            (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
                objective, has_aux=True
            )(params, state)
            grads = jax.lax.pmean(grads, axis_name)
            loss = jax.lax.pmean(loss, axis_name)
            diagnostics = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, axis_name), diagnostics
            )
            next_state = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, axis_name), next_state
            )
            finite = jnp.logical_and(
                jnp.isfinite(loss),
                jnp.all(jnp.asarray([
                    jnp.all(jnp.isfinite(x))
                    for x in jax.tree_util.tree_leaves(grads)
                ])),
            )
            finite = jax.lax.pmin(finite, axis_name)
            new_params, new_optimizer_state = _adam_update(
                params, grads, optimizer_state, learning_rate, beta1, beta2, epsilon
            )
            params = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), new_params, params
            )
            next_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old), next_state, state
            )
            new_optimizer_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(finite, new, old),
                new_optimizer_state,
                optimizer_state,
            )
            return params, next_state, new_optimizer_state, loss, diagnostics, finite

        train_step = xarray_jax.pmap(
            parallel_train_step,
            dim="device",
            axis_name=axis_name,
            devices=pmap_devices,
        )
    else:
        train_step = jax.jit(train_step)
    max_steps = int(args.max_steps or config["training"]["max_steps"])
    save_interval = int(config["training"].get("save_interval", max_steps))
    checkpoint_path = config["checkpoint"]["trainer"]
    if process_index == 0:
        print(
            f"Training samples: {len(dataset)}; parameters: "
            f"{parameter_count(params):,}"
        )
    if mode == "pmap":
        params = _replicate(params, devices)
        state = _replicate(state, devices)
        optimizer_state = _replicate(optimizer_state, devices)
        if process_index == 0:
            print(
                f"Parallel mode: pmap; processes: {process_count}; "
                f"local devices: {requested_devices}; "
                f"global devices: {global_device_count}; "
                f"global batch: {global_batch_size}"
            )

    step = start_step
    batches_per_epoch = len(dataset) // global_batch_size
    if batches_per_epoch < 1:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than global_batch_size "
            f"{global_batch_size}"
        )
    while step < max_steps:
        epoch = step // batches_per_epoch
        offset = step % batches_per_epoch
        epoch_batches = batch_iterator(
            dataset,
            shuffle=True,
            seed=seed + epoch,
            batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            process_index=process_index,
            process_count=process_count,
        )
        for batch in itertools.islice(epoch_batches, offset, None):
            if step >= max_steps:
                break
            step_rng = jax.random.fold_in(jax.random.PRNGKey(seed), step)
            step_rng = jax.random.fold_in(step_rng, process_index)
            if mode == "pmap":
                batch = _device_batch(batch, requested_devices)
                step_rng = jax.numpy.broadcast_to(
                    step_rng, (requested_devices, *step_rng.shape)
                )
                params, state, optimizer_state, loss, _, finite = train_step(
                    params, state, optimizer_state, step_rng, *batch
                )
                loss, finite = loss[0], finite[0]
            else:
                params, state, optimizer_state, loss, _, finite = train_step(
                    params, state, optimizer_state, step_rng, *batch
                )
            step += 1
            if process_index == 0:
                print(f"step={step} loss={float(loss):.8f} finite={bool(finite)}")
            if not bool(finite):
                raise FloatingPointError(f"Non-finite GenCast loss at step {step}")
            if step % save_interval == 0 or step == max_steps:
                _sync_global_devices(jax, f"gencast-checkpoint-{step}-before")
                checkpoint_trees = (params, state, optimizer_state)
                if mode == "pmap":
                    checkpoint_trees = tuple(map(_unreplicate, checkpoint_trees))
                if process_index == 0:
                    save_trainer_checkpoint(
                        checkpoint_path,
                        params=checkpoint_trees[0],
                        state=checkpoint_trees[1],
                        optimizer_state=checkpoint_trees[2],
                        step=step,
                        config=config,
                    )
                    print(f"Saved checkpoint to {resolve_path(checkpoint_path)}")
                _sync_global_devices(jax, f"gencast-checkpoint-{step}-after")


if __name__ == "__main__":
    main()
