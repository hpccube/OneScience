"""Compatibility entrypoint for Equiformer V3 checkpoint fine-tuning.

New training configurations are handled by :mod:`train`, which supports
scratch training, checkpoint initialization, and full-state resume. The
loader remains here for compatibility with ``evaluate.py`` and external code.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.materials.custom_stack import data_list_collater
from onescience.datapipes.materials.custom_stack.storage.ase_datasets import AseDBDataset
from onescience.utils.equiformer_v3 import (
    EquiformerV3CheckpointTransforms,
    load_equiformer_v3_checkpoint,
)
from onescience.utils.uma.normalization.element_references import (
    fit_linear_references,
)


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for a normal Python process or a torchrun worker."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed(device_name: str, backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if device_name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(0)
        return DistributedContext()

    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for multi-device fine-tuning.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torchrun requested multiple CUDA/DCU devices, but CUDA is unavailable."
            )
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank=rank, world_size=world_size, local_rank=local_rank)


def _close_distributed(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _loader(
    path: str | list[str],
    batch_size: int,
    workers: int,
    max_samples: int | None = None,
    context: DistributedContext | None = None,
    train: bool = False,
    seed: int = 0,
) -> DataLoader:
    dataset = AseDBDataset(
        {
            "src": path,
            "a2g_args": {
                "r_edges": False,
                "r_energy": True,
                "r_forces": True,
                "r_stress": True,
            },
        }
    )
    if max_samples is not None:
        sample_count = min(max_samples, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        dataset = Subset(dataset, indices)
    context = context or DistributedContext()
    sampler = None
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and train,
        sampler=sampler,
        num_workers=workers,
        collate_fn=lambda items: data_list_collater(items, otf_graph=True),
    )


def _loss(
    pred,
    batch,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    transforms: EquiformerV3CheckpointTransforms,
):
    losses = {}
    if energy_weight:
        energy_target = transforms.normalize_target(
            "energy", batch.energy, pred["energy"], batch
        )
        energy_error = pred["energy"] - energy_target
        natoms_shape = (-1,) + (1,) * (energy_error.ndim - 1)
        natoms = batch.natoms.to(energy_error).reshape(natoms_shape)
        losses["energy"] = (energy_error / natoms).square().mean()
    if force_weight:
        force_target = transforms.normalize_target(
            "forces", batch.forces, pred["forces"], batch
        )
        losses["forces"] = (pred["forces"] - force_target).square().mean()
    if stress_weight:
        if not hasattr(batch, "stress"):
            raise ValueError("stress_weight is nonzero, but the batch has no stress labels")
        if "stress" not in pred:
            raise ValueError("stress_weight is nonzero, but the model returned no stress")
        stress_target = transforms.normalize_target(
            "stress", batch.stress, pred["stress"], batch
        )
        losses["stress"] = (pred["stress"] - stress_target).square().mean()
    total = energy_weight * losses.get("energy", 0.0)
    total = total + force_weight * losses.get("forces", 0.0)
    total = total + stress_weight * losses.get("stress", 0.0)
    return total, {key: float(value.detach()) for key, value in losses.items()}


def _run_epoch(
    model,
    loader,
    device,
    optimizer,
    weights,
    transforms: EquiformerV3CheckpointTransforms,
    context: DistributedContext,
):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0
    metric_names = tuple(
        name
        for name, weight in zip(("energy", "forces", "stress"), weights)
        if weight
    )
    metrics = {name: 0.0 for name in metric_names}
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss, batch_metrics = _loss(prediction, batch, *weights, transforms)
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        batches += 1
        for key, value in batch_metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value
    if batches == 0:
        raise RuntimeError("The dataset contains no samples.")
    values = torch.tensor(
        [total, *metrics.values(), float(batches)],
        dtype=torch.float64,
        device=device,
    )
    if context.enabled:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    global_batches = values[-1].item()
    return {
        "loss": values[0].item() / global_batches,
        **{
            key: values[index].item() / global_batches
            for index, key in enumerate(metrics, start=1)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train", help="ASE DB or ASE-LMDB training path")
    parser.add_argument("--val", help="ASE DB or ASE-LMDB validation path")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--energy-weight", type=float)
    parser.add_argument("--force-weight", type=float)
    parser.add_argument("--stress-weight", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--backend", help="torch.distributed backend for torchrun")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--fit-element-references",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fit energy element references on the training data",
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required; use a YAML file from demo/configs")
    config_path = args.config
    with Path(config_path).expanduser().open() as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in config.items():
        if getattr(args, key.replace("-", "_"), None) is None:
            setattr(args, key.replace("-", "_"), value)
    for key in ("checkpoint", "train", "val", "output"):
        value = getattr(args, key)
        if value is not None:
            if isinstance(value, list):
                value = [
                    os.path.expandvars(os.path.expanduser(str(item)))
                    for item in value
                ]
            else:
                value = os.path.expandvars(os.path.expanduser(str(value)))
            setattr(args, key, value)
    required = ("checkpoint", "train", "val", "output")
    missing = [key for key in required if not getattr(args, key)]
    if missing:
        parser.error("missing required config fields: " + ", ".join(missing))
    args.backend = args.backend or "nccl"
    args.seed = 0 if args.seed is None else args.seed
    args.fit_element_references = bool(args.fit_element_references)
    if not any((args.energy_weight, args.force_weight, args.stress_weight)):
        parser.error(
            "at least one of energy_weight, force_weight, or stress_weight "
            "must be nonzero"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false.")
    context = _init_distributed(args.device, args.backend)
    try:
        if args.device.startswith("cuda"):
            device = torch.device(f"cuda:{context.local_rank}")
        else:
            device = torch.device(args.device)
        torch.manual_seed(args.seed + context.rank)

        model = load_equiformer_v3_checkpoint(args.checkpoint).to(device)
        transforms = EquiformerV3CheckpointTransforms.from_checkpoint(
            args.checkpoint
        )
        if args.fit_element_references:
            reference_dataset = _loader(
                args.train, args.batch_size, args.workers
            ).dataset
            fitted_references = fit_linear_references(
                targets=["energy"],
                dataset=reference_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                log_metrics=False,
                shuffle=False,
            )
            transforms.elementrefs["energy"] = fitted_references["energy"]
            if context.is_main:
                print("fitted energy element references from training data", flush=True)
        transforms = transforms.to(device)
        if context.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_loader = _loader(
            args.train,
            args.batch_size,
            args.workers,
            args.max_train_samples,
            context=context,
            train=True,
            seed=args.seed,
        )
        val_loader = _loader(
            args.val,
            args.batch_size,
            args.workers,
            args.max_val_samples,
            context=context,
            train=False,
            seed=args.seed + 1,
        )
        weights = (args.energy_weight, args.force_weight, args.stress_weight)

        history = []
        for epoch in range(args.epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = _run_epoch(
                model, train_loader, device, optimizer, weights, transforms, context
            )
            # Force/stress outputs are gradients of the energy, so validation also
            # needs autograd even though model parameters are not updated.
            val_metrics = _run_epoch(
                model, val_loader, device, None, weights, transforms, context
            )
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            if context.is_main:
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

        if context.is_main:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            base_model = model.module if context.enabled else model
            checkpoint = {
                "config": source["config"],
                "normalizers": source.get("normalizers", {}),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in base_model.state_dict().items()
                },
                "metadata": dict(source.get("metadata", {})),
            }
            checkpoint["elementrefs"] = {
                name: {
                    key: value.detach().cpu()
                    for key, value in elementref.state_dict().items()
                }
                for name, elementref in transforms.elementrefs.items()
            }
            checkpoint["metadata"].update(
                {
                    "onescience_equiformer_v3_history": history,
                    "source_checkpoint": args.checkpoint,
                    "world_size": context.world_size,
                    "loss_space": "checkpoint_normalized",
                    "element_references": (
                        "fitted_from_training_data"
                        if args.fit_element_references
                        else "source_checkpoint"
                    ),
                }
            )
            del source
            torch.save(checkpoint, output)
            print(f"saved checkpoint: {output}", flush=True)
    finally:
        _close_distributed(context)


if __name__ == "__main__":
    from train import main as training_main

    training_main()
