"""Train, fine-tune, or resume an Equiformer V3 atomistic model.

The input splits must be ASE DB or ASE-LMDB datasets containing calculator
results. Checkpoints remain compatible with ``EquiformerV3Calculator`` while
also carrying optimizer, scheduler, EMA, and progress state for resuming.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.materials.custom_stack import data_list_collater
from onescience.datapipes.materials.custom_stack.base_dataset import (
    Subset as MetadataSubset,
)
from onescience.datapipes.materials.custom_stack.storage.ase_datasets import (
    AseDBDataset,
)
from onescience.modules.loss.uma_loss import DDPLoss
from onescience.utils.equiformer_v3 import (
    EquiformerV3CheckpointTransforms,
    load_equiformer_v3_checkpoint,
)
from onescience.utils.uma.common.data_parallel import BalancedBatchSampler
from onescience.utils.uma.common.registry import registry
from onescience.utils.uma.normalization.element_references import (
    LinearReferences,
    create_element_references,
    fit_linear_references,
)
from onescience.utils.uma.normalization.normalizer import (
    create_normalizer,
    fit_normalizers,
)
from onescience.utils.uma.scheduler import CosineLRLambda


MODE_ALIASES = {
    "scratch": "train_from_scratch",
    "train": "train_from_scratch",
    "train_from_scratch": "train_from_scratch",
    "finetune": "init_from_checkpoint",
    "fine_tune": "init_from_checkpoint",
    "init_from_checkpoint": "init_from_checkpoint",
    "resume": "resume_training",
    "resume_training": "resume_training",
}


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information for a normal process or a torchrun worker."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class LossSpec:
    """One target loss from the YAML contract."""

    name: str
    function: str
    coefficient: float
    free_atoms_only: bool = False


@dataclass(frozen=True)
class DenoisingPosParams:
    """Official Equiformer V3 DeNS position-corruption contract."""

    enabled: bool = False
    prob: float = 0.0
    fixed_noise_std: bool = True
    std: float = 0.025
    corrupt_ratio: float | None = None
    all_atoms: bool = False
    min_num_atoms: int | None = None
    strict_max_ratio: float | None = None
    max_force_norm: float | None = None
    max_stress_norm: float | None = None
    max_mean_force_norm: float | None = None
    coefficient: float = 1.0


class ModelEMA:
    """Exponential moving average of trainable model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must be between zero and one")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            value.lerp_(parameters[name].detach(), 1.0 - self.decay)

    @contextmanager
    def apply(self, model: torch.nn.Module):
        parameters = dict(model.named_parameters())
        backup = {
            name: parameters[name].detach().clone() for name in self.shadow
        }
        with torch.no_grad():
            for name, value in self.shadow.items():
                parameters[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": {
                name: value.detach().cpu() for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device) -> None:
        self.decay = float(state["decay"])
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA parameters do not match the resumed model")
        self.shadow = {
            name: value.to(device=device) for name, value in state["shadow"].items()
        }


def _init_distributed(device_name: str, backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if device_name.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(0)
        return DistributedContext()
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is required for torchrun training")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested CUDA/DCU, but it is unavailable")
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=backend, rank=rank, world_size=world_size
    )
    return DistributedContext(rank, world_size, local_rank)


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
    max_atoms: int | None = None,
    load_balancing: str | bool | None = "atoms",
    load_balancing_on_error: str = "raise",
    device: torch.device | None = None,
) -> DataLoader:
    """Build the shared OneScience FairChem-style ASE data loader."""

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
    indices = list(range(len(dataset)))
    if max_atoms is not None:
        if not dataset.metadata_hasattr("natoms"):
            raise ValueError("max_atoms requires dataset metadata.npz with natoms")
        natoms = dataset.get_metadata("natoms", indices)
        indices = [
            index for index, count in zip(indices, natoms) if int(count) <= max_atoms
        ]
        if not indices:
            raise ValueError(f"max_atoms={max_atoms} filtered every sample")
    if max_samples is not None:
        sample_count = min(max_samples, len(indices))
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(indices), generator=generator)[:sample_count]
        indices = [indices[index] for index in order.tolist()]
    if len(indices) != len(dataset):
        dataset = MetadataSubset(dataset, indices, metadata={})

    context = context or DistributedContext()
    if load_balancing:
        batch_sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=context.world_size,
            rank=context.rank,
            device=device,
            seed=seed,
            mode=load_balancing,
            shuffle=train,
            on_error=load_balancing_on_error,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            collate_fn=lambda items: data_list_collater(items, otf_graph=True),
            generator=torch.Generator().manual_seed(seed),
        )

    sampler = None
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=train,
            drop_last=False,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and train,
        sampler=sampler,
        num_workers=workers,
        collate_fn=lambda items: data_list_collater(items, otf_graph=True),
        generator=torch.Generator().manual_seed(seed),
    )


def _expand_path(value: str | list[str] | None) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [_expand_path(item) for item in value]
    return os.path.expandvars(os.path.expanduser(str(value)))


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(raw)
    mode = config.get("mode")
    if mode is None:
        mode = "resume_training" if config.get("resume") else None
        mode = mode or (
            "init_from_checkpoint"
            if config.get("initialization_checkpoint") or config.get("checkpoint")
            else "train_from_scratch"
        )
    try:
        config["mode"] = MODE_ALIASES[str(mode)]
    except KeyError as error:
        raise ValueError(f"unsupported training mode: {mode!r}") from error

    if config.get("checkpoint") and not config.get("initialization_checkpoint"):
        config["initialization_checkpoint"] = config["checkpoint"]
    for key in ("initialization_checkpoint", "resume", "train", "val", "output"):
        config[key] = _expand_path(config.get(key))
    config["transforms_checkpoint"] = _expand_path(
        config.get("transforms_checkpoint")
    )

    optimizer = config.setdefault("optimizer", {})
    optimizer.setdefault("name", "AdamW")
    optimizer.setdefault("lr", config.get("lr", 5.0e-5))
    optimizer.setdefault("weight_decay", config.get("weight_decay", 1.0e-3))
    optimizer.setdefault("betas", config.get("betas", [0.9, 0.98]))
    optimizer.setdefault("eps", config.get("eps", 1.0e-6))

    scheduler = config.setdefault("scheduler", {})
    scheduler.setdefault("name", "cosine")
    scheduler.setdefault("warmup_factor", 0.0)
    scheduler.setdefault("warmup_epochs", 0.1)
    scheduler.setdefault("lr_min_factor", 0.01)

    config.setdefault("device", "cuda")
    config.setdefault("backend", "nccl")
    config.setdefault("seed", 0)
    config.setdefault("epochs", 1)
    config.setdefault("batch_size", 1)
    config.setdefault("eval_batch_size", config["batch_size"])
    config.setdefault("workers", 0)
    config.setdefault("grad_accumulation_steps", 1)
    config.setdefault("log_every_n_steps", 0)
    config.setdefault("log_every_n_validation_batches", 0)
    config.setdefault("clip_grad_norm", 100.0)
    config.setdefault("ema_decay", 0.999)
    config.setdefault("amp", False)
    config.setdefault("amp_dtype", "float16")
    config.setdefault("amp_init_scale", 65536.0)
    config.setdefault("load_balancing", "atoms")
    config.setdefault("load_balancing_on_error", "raise")
    config.setdefault("ddp_find_unused_parameters", False)
    return config


def _loss_specs(config: dict[str, Any]) -> list[LossSpec]:
    raw = config.get("losses") or config.get("loss_functions")
    if raw is None:
        raw = {
            "energy": {
                "fn": "per_atom_mae",
                "coefficient": config.get("energy_weight", 1.0),
            },
            "forces": {
                "fn": "l2mae",
                "coefficient": config.get("force_weight", 10.0),
            },
            "stress": {
                "fn": "mae",
                "coefficient": config.get("stress_weight", 0.0),
            },
        }
    if isinstance(raw, list):
        merged = {}
        for item in raw:
            merged.update(item)
        raw = merged
    specs = []
    for name, values in raw.items():
        coefficient = float(values.get("coefficient", values.get("weight", 1.0)))
        if coefficient == 0.0:
            continue
        specs.append(
            LossSpec(
                name=name,
                function=str(values.get("fn", values.get("function", "mae"))),
                coefficient=coefficient,
                free_atoms_only=bool(values.get("free_atoms_only", False)),
            )
        )
    if not specs:
        raise ValueError("at least one nonzero target loss is required")
    unknown = {spec.name for spec in specs} - {"energy", "forces", "stress"}
    if unknown:
        raise ValueError(f"unsupported loss targets: {sorted(unknown)}")
    return specs


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _denoising_pos_params(config: dict[str, Any]) -> DenoisingPosParams:
    values = config.get("denoising_pos_params") or {}
    return DenoisingPosParams(
        enabled=bool(config.get("use_denoising_pos", False)),
        prob=float(values.get("prob", 0.0)),
        fixed_noise_std=bool(values.get("fixed_noise_std", True)),
        std=float(values.get("std", 0.025)),
        corrupt_ratio=_optional_float(values.get("corrupt_ratio")),
        all_atoms=bool(values.get("all_atoms", False)),
        min_num_atoms=(
            None
            if values.get("min_num_atoms") is None
            else int(values["min_num_atoms"])
        ),
        strict_max_ratio=_optional_float(values.get("strict_max_ratio")),
        max_force_norm=_optional_float(values.get("max_force_norm")),
        max_stress_norm=_optional_float(values.get("max_stress_norm")),
        max_mean_force_norm=_optional_float(values.get("max_mean_force_norm")),
        coefficient=float(config.get("denoising_pos_coefficient", 1.0)),
    )


def _transforms_from_config(config: dict[str, Any]) -> EquiformerV3CheckpointTransforms:
    transform_config = config.get("transforms", {})
    normalizers = {}
    for name, values in transform_config.get("normalizers", {}).items():
        values = copy.deepcopy(values)
        if "file" in values:
            values["file"] = _expand_path(values["file"])
        normalizers[name] = create_normalizer(**values)
    elementrefs = {}
    for name, values in transform_config.get("element_references", {}).items():
        values = copy.deepcopy(values)
        if "values" in values:
            elementrefs[name] = LinearReferences(
                torch.as_tensor(values["values"], dtype=torch.float32)
            )
        else:
            if "file" in values:
                values["file"] = _expand_path(values["file"])
            elementrefs[name] = create_element_references(**values)
    return EquiformerV3CheckpointTransforms(normalizers, elementrefs)


def _training_transforms(
    config: dict[str, Any], checkpoint_path: str | Path | None
) -> EquiformerV3CheckpointTransforms:
    """Resolve target transforms independently from model initialization weights.

    Initialization checkpoints carry the statistics used by their training
    dataset.  Fine-tuning may instead point at a target-domain checkpoint or
    override individual entries in ``transforms``.  Resume deliberately keeps
    the source checkpoint transforms unchanged and is validated separately.
    """

    source = config.get("transforms_checkpoint") or checkpoint_path
    if source:
        if config.get("clear_checkpoint_transforms"):
            transforms = EquiformerV3CheckpointTransforms()
        else:
            transforms = EquiformerV3CheckpointTransforms.from_checkpoint(source)
    else:
        transforms = EquiformerV3CheckpointTransforms()

    overrides = _transforms_from_config(config)
    for name, module in overrides.normalizers.items():
        transforms.normalizers[name] = module
    for name, module in overrides.elementrefs.items():
        transforms.elementrefs[name] = module
    return transforms


def _construct_model(model_config: dict[str, Any]) -> torch.nn.Module:
    import onescience.models.equiformer_v3  # noqa: F401

    kwargs = copy.deepcopy(model_config)
    name = kwargs.pop("name", None)
    if name not in {"equiformer_v3", "equiformer_v3_dens"}:
        raise ValueError(f"unsupported Equiformer V3 model name: {name!r}")
    return registry.get_model_class(name)(**kwargs)


def _checkpoint_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    document = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in document or "state_dict" not in document:
        raise ValueError(f"invalid Equiformer V3 checkpoint: {path}")
    return document


def _reset_module(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def _initialize_model(
    config: dict[str, Any],
) -> tuple[
    torch.nn.Module,
    EquiformerV3CheckpointTransforms,
    dict[str, Any],
    dict[str, Any] | None,
]:
    mode = config["mode"]
    if mode == "train_from_scratch":
        if not config.get("model"):
            raise ValueError("train_from_scratch requires a model mapping")
        model_config = copy.deepcopy(config["model"])
        return (
            _construct_model(model_config),
            _training_transforms(config, None),
            model_config,
            None,
        )

    path = config.get("resume") if mode == "resume_training" else config.get(
        "initialization_checkpoint"
    )
    if not path:
        required = "resume" if mode == "resume_training" else "initialization_checkpoint"
        raise ValueError(f"{mode} requires {required}")
    document = _checkpoint_document(path)
    source_model_config = copy.deepcopy(document["config"]["model"])
    model_config = source_model_config | copy.deepcopy(config.get("model", {}))

    if mode == "resume_training":
        model = _construct_model(model_config)
        state = document.get("training_state_dict", document["state_dict"])
        model.load_state_dict(state, strict=True)
    elif model_config == source_model_config:
        model = load_equiformer_v3_checkpoint(path)
    else:
        model = _construct_model(model_config)
        source_state = document["state_dict"]
        excluded = tuple(config.get("exclude_initialization_prefixes", []))
        compatible = {
            key.removeprefix("_orig_mod."): value
            for key, value in source_state.items()
            if not key.removeprefix("_orig_mod.").startswith(excluded)
            and key.removeprefix("_orig_mod.") in model.state_dict()
            and model.state_dict()[key.removeprefix("_orig_mod.")].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(
            f"initialized {len(compatible)}/{len(model.state_dict())} model tensors "
            f"from {path}",
            flush=True,
        )
    if config.get("reset_energy_head"):
        _reset_module(model.energy_block)

    if mode == "resume_training":
        transforms = EquiformerV3CheckpointTransforms.from_checkpoint(path)
    else:
        transforms = _training_transforms(config, path)
    return model, transforms, model_config, document


def _target_tensor(name: str, batch) -> torch.Tensor:
    if not hasattr(batch, name):
        raise ValueError(f"the batch does not contain required {name} labels")
    return getattr(batch, name)


def _masked_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: LossSpec,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.name == "forces" and spec.free_atoms_only:
        if not hasattr(batch, "fixed"):
            raise ValueError("free_atoms_only requires a fixed atom mask")
        mask = batch.fixed.reshape(-1) == 0
        prediction = prediction[mask]
        target = target[mask]
    return prediction, target


def _loss_functions(specs: list[LossSpec]) -> dict[str, DDPLoss]:
    """Build the same DDP-aware reductions used by the official trainer."""

    return {
        spec.name: DDPLoss(spec.function, reduction="mean") for spec in specs
    }


def _loss(
    prediction: dict[str, torch.Tensor],
    batch,
    specs: list[LossSpec],
    transforms: EquiformerV3CheckpointTransforms,
    loss_functions: dict[str, DDPLoss] | None = None,
    dens_params: DenoisingPosParams | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_functions = loss_functions or _loss_functions(specs)
    total = next(iter(prediction.values())).new_zeros(())
    components = {}
    for spec in specs:
        if spec.name not in prediction:
            raise ValueError(f"the model did not return required {spec.name} output")
        target = _target_tensor(spec.name, batch)
        normalized = transforms.normalize_target(
            spec.name, target, prediction[spec.name], batch
        )
        if spec.name == "forces" and _is_dens_batch(batch):
            if dens_params is None:
                raise RuntimeError("DeNS batch requires denoising parameters")
            pred = prediction[spec.name]
            noise_mask = batch.noise_mask.reshape(-1, 1).bool()
            denoising_target = batch.noise_vec.to(pred) / dens_params.std
            hybrid_target = torch.where(noise_mask, denoising_target, normalized)
            selection = torch.ones(
                pred.shape[0], dtype=torch.bool, device=pred.device
            )
            if spec.free_atoms_only:
                if not hasattr(batch, "fixed"):
                    raise ValueError("free_atoms_only requires a fixed atom mask")
                selection = batch.fixed.reshape(-1) == 0
                if dens_params.all_atoms:
                    selection = selection | noise_mask.reshape(-1)
            if not bool(selection.any()):
                raise RuntimeError("DeNS batch has no atoms selected for force loss")

            atomwise = torch.linalg.vector_norm(pred - hybrid_target, dim=-1)
            coefficients = torch.where(
                noise_mask.reshape(-1),
                atomwise.new_full(atomwise.shape, dens_params.coefficient),
                atomwise.new_full(atomwise.shape, spec.coefficient),
            )
            value = (atomwise[selection] * coefficients[selection]).mean()
            total = total + value
            force_mask = selection & ~noise_mask.reshape(-1)
            dens_mask = selection & noise_mask.reshape(-1)
            if bool(force_mask.any()):
                components[f"{spec.name}_{spec.function}"] = float(
                    atomwise[force_mask].mean().detach()
                )
            if bool(dens_mask.any()):
                components["denoising_pos_l2mae"] = float(
                    atomwise[dens_mask].mean().detach()
                )
            components["forces_dens_hybrid_l2mae"] = float(value.detach())
            continue
        pred, normalized = _masked_tensors(
            prediction[spec.name], normalized, spec, batch
        )
        value = loss_functions[spec.name](pred, normalized, natoms=batch.natoms)
        total = total + spec.coefficient * value
        components[f"{spec.name}_{spec.function}"] = float(value.detach())
    return total, components


@torch.no_grad()
def _physical_metrics(
    prediction: dict[str, torch.Tensor],
    batch,
    specs: list[LossSpec],
    transforms: EquiformerV3CheckpointTransforms,
    dens_params: DenoisingPosParams | None = None,
) -> dict[str, tuple[float, int]]:
    metrics = {}
    for spec in specs:
        if spec.name == "forces" and _is_dens_batch(batch):
            if dens_params is None:
                raise RuntimeError("DeNS batch requires denoising parameters")
            prediction_tensor = prediction[spec.name].detach()
            noise_mask = batch.noise_mask.reshape(-1).bool()
            selection = torch.ones_like(noise_mask)
            if spec.free_atoms_only:
                selection = batch.fixed.reshape(-1) == 0
                if dens_params.all_atoms:
                    selection = selection | noise_mask
            force_mask = selection & ~noise_mask
            dens_mask = selection & noise_mask

            physical_force = transforms.denormalize_prediction(
                spec.name, prediction_tensor, batch
            )
            force_error = physical_force[force_mask] - batch.forces[force_mask]
            dens_prediction = prediction_tensor[dens_mask] * dens_params.std
            dens_error = dens_prediction - batch.noise_vec[dens_mask]
            metrics["denoising_force_mae"] = (
                float(force_error.abs().sum()),
                force_error.numel(),
            )
            metrics["denoising_force_l2mae"] = (
                float(torch.linalg.vector_norm(force_error, dim=-1).sum()),
                force_error.shape[0],
            )
            metrics["denoising_pos_mae"] = (
                float(dens_error.abs().sum()),
                dens_error.numel(),
            )
            metrics["denoising_pos_l2mae"] = (
                float(torch.linalg.vector_norm(dens_error, dim=-1).sum()),
                dens_error.shape[0],
            )
            metrics["dens_corrupted_atom_fraction"] = (
                float(dens_mask.sum()),
                int(selection.sum()),
            )
            continue
        physical = transforms.denormalize_prediction(
            spec.name, prediction[spec.name].detach(), batch
        )
        target = _target_tensor(spec.name, batch).reshape_as(physical)
        physical, target = _masked_tensors(physical, target, spec, batch)
        error = physical - target
        absolute_error = error.abs()
        metrics[f"{spec.name}_mae"] = (
            float(absolute_error.sum()),
            absolute_error.numel(),
        )
        if spec.name == "energy":
            shape = (-1,) + (1,) * (error.ndim - 1)
            per_atom = error / batch.natoms.to(error).reshape(shape)
            metrics["energy_per_atom_mae"] = (
                float(per_atom.abs().sum()),
                per_atom.numel(),
            )
        elif error.ndim >= 2:
            vector_error = torch.linalg.vector_norm(error, dim=-1)
            metrics[f"{spec.name}_l2mae"] = (
                float(vector_error.sum()),
                vector_error.numel(),
            )
    return metrics


def _reduce_metrics(
    sums: dict[str, tuple[float, int]],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, float]:
    if not sums:
        raise RuntimeError("the dataset contains no samples")
    names = sorted(sums)
    if context.enabled:
        rank_names: list[list[str] | None] = [None] * context.world_size
        torch.distributed.all_gather_object(rank_names, names)
        names = sorted(
            {
                name
                for gathered_names in rank_names
                if gathered_names is not None
                for name in gathered_names
            }
        )
    values = torch.tensor(
        [
            *(sums.get(name, (0.0, 0))[0] for name in names),
            *(sums.get(name, (0.0, 0))[1] for name in names),
        ],
        device=device,
        dtype=torch.float64,
    )
    if context.enabled:
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    split = len(names)
    reduced = {}
    for index, name in enumerate(names):
        count = values[split + index].item()
        if count <= 0:
            continue
        reduced[name] = values[index].item() / count
    return reduced


def _collect_batch_metrics(
    sums: dict[str, tuple[float, int]],
    loss: torch.Tensor,
    components: dict[str, float],
    physical: dict[str, tuple[float, int]],
) -> None:
    values = {
        "loss": (float(loss.detach()), 1),
        **{
            f"normalized_{name}": (value, 1) for name, value in components.items()
        },
        **physical,
    }
    for name, (total, count) in values.items():
        previous_total, previous_count = sums.get(name, (0.0, 0))
        sums[name] = previous_total + total, previous_count + count


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _is_dens_batch(batch) -> bool:
    value = getattr(batch, "denoising_pos_forward", False)
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def _graph_max(
    values: torch.Tensor, batch_index: torch.Tensor, graph_count: int
) -> torch.Tensor:
    result = values.new_full((graph_count,), float("-inf"))
    return result.scatter_reduce_(
        0, batch_index, values, reduce="amax", include_self=True
    )


def _apply_graph_filter(
    graph_mask: torch.Tensor,
    dens_batch_mask: torch.Tensor,
    noise_mask: torch.Tensor,
    noise_vec: torch.Tensor,
    batch_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dens_batch_mask = dens_batch_mask & graph_mask
    atom_mask = graph_mask[batch_index]
    noise_mask = noise_mask & atom_mask
    noise_vec = noise_vec * atom_mask.reshape(-1, 1)
    return dens_batch_mask, noise_mask, noise_vec


def _add_gaussian_noise_to_position(batch, params: DenoisingPosParams):
    """Apply the official Equiformer V3 DeNS corruption to one collated batch."""

    graph_count = int(batch.natoms.numel())
    batch_index = batch.batch.long()
    noise_vec = torch.empty_like(batch.pos).normal_(mean=0.0, std=params.std)
    if params.corrupt_ratio is None:
        noise_mask = torch.ones(
            batch.pos.shape[0], dtype=torch.bool, device=batch.pos.device
        )
    else:
        noise_mask = (
            torch.rand(
                batch.pos.shape[0], dtype=batch.pos.dtype, device=batch.pos.device
            )
            < params.corrupt_ratio
        )
    noise_vec = noise_vec * noise_mask.reshape(-1, 1)
    dens_batch_mask = torch.ones(
        graph_count, dtype=torch.bool, device=batch.pos.device
    )

    if hasattr(batch, "skip_dens"):
        graph_mask = ~batch.skip_dens.reshape(-1).bool()
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )
    if params.min_num_atoms is not None:
        graph_mask = batch.natoms >= params.min_num_atoms
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )
    if params.strict_max_ratio is not None:
        corrupted = batch.pos.new_zeros(graph_count)
        corrupted.index_add_(0, batch_index, noise_mask.to(batch.pos.dtype))
        graph_mask = corrupted <= batch.natoms.to(corrupted) * params.strict_max_ratio
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )
    if params.max_force_norm is not None:
        graph_mask = _graph_max(
            torch.linalg.vector_norm(batch.forces, dim=-1),
            batch_index,
            graph_count,
        ) <= params.max_force_norm
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )
    if params.max_stress_norm is not None:
        graph_mask = (
            torch.linalg.vector_norm(batch.stress.reshape(graph_count, -1), dim=-1)
            <= params.max_stress_norm
        )
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )
    if params.max_mean_force_norm is not None:
        force_sum = batch.forces.new_zeros((graph_count, batch.forces.shape[-1]))
        force_sum.index_add_(0, batch_index, batch.forces)
        graph_mask = (
            torch.linalg.vector_norm(force_sum, dim=-1)
            <= params.max_mean_force_norm
        )
        dens_batch_mask, noise_mask, noise_vec = _apply_graph_filter(
            graph_mask, dens_batch_mask, noise_mask, noise_vec, batch_index
        )

    if params.all_atoms:
        position_mask = torch.ones_like(noise_mask)
    else:
        if not hasattr(batch, "fixed"):
            raise ValueError("DeNS with all_atoms=false requires a fixed atom mask")
        position_mask = batch.fixed.reshape(-1) == 0
    batch.pos = batch.pos + noise_vec * position_mask.reshape(-1, 1)
    batch.noise_vec = noise_vec
    batch.noise_mask = noise_mask
    batch.denoising_pos_forward = True
    batch.dens_batch_mask = dens_batch_mask
    return batch


def _should_apply_dens(
    params: DenoisingPosParams,
    device: torch.device,
    context: DistributedContext,
) -> bool:
    if not params.enabled or params.prob <= 0.0:
        return False
    decision = torch.rand((), device=device) < params.prob
    if context.enabled:
        torch.distributed.broadcast(decision, src=0)
    return bool(decision.item())


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema: ModelEMA | None,
    specs: list[LossSpec],
    loss_functions: dict[str, DDPLoss],
    transforms: EquiformerV3CheckpointTransforms,
    context: DistributedContext,
    scaler: torch.GradScaler | None,
    amp_dtype: torch.dtype,
    grad_accumulation_steps: int,
    clip_grad_norm: float | None,
    global_step: int,
    max_steps: int | None,
    epoch: int,
    log_every_n_steps: int,
    dens_params: DenoisingPosParams,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, tuple[float, int]] = {}
    progress_sums: dict[str, tuple[float, int]] = {}
    progress_start_step = global_step
    pending = 0
    completed_updates = 0
    skipped_optimizer_steps = 0
    updates_per_epoch = _updates_per_epoch(len(loader), grad_accumulation_steps)
    batches_to_process = updates_per_epoch * grad_accumulation_steps
    for index, batch in enumerate(loader):
        if index >= batches_to_process:
            break
        batch = batch.to(device)
        if _should_apply_dens(dens_params, device, context):
            batch = _add_gaussian_noise_to_position(batch, dens_params)
        synchronize_gradients = pending + 1 == grad_accumulation_steps
        with _gradient_sync_context(model, synchronize_gradients):
            with torch.autocast(
                device_type=device.type,
                enabled=scaler is not None,
                dtype=amp_dtype,
            ):
                prediction = model(batch)
                loss, components = _loss(
                    prediction,
                    batch,
                    specs,
                    transforms,
                    loss_functions,
                    dens_params,
                )
            physical = _physical_metrics(
                prediction, batch, specs, transforms, dens_params
            )
            backward_loss = loss / grad_accumulation_steps
            if scaler is None:
                backward_loss.backward()
            else:
                scaler.scale(backward_loss).backward()
        _collect_batch_metrics(sums, loss, components, physical)
        _collect_batch_metrics(progress_sums, loss, components, physical)
        pending += 1
        if pending != grad_accumulation_steps:
            continue
        if clip_grad_norm:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        if scaler is None:
            optimizer.step()
            optimizer_step_succeeded = True
        else:
            previous_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_succeeded = _amp_step_succeeded(
                previous_scale, float(scaler.get_scale())
            )
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        if not optimizer_step_succeeded:
            skipped_optimizer_steps += 1
            continue
        scheduler.step()
        if ema is not None:
            ema.update(_unwrap(model))
        global_step += 1
        completed_updates += 1
        reached_max_steps = max_steps is not None and global_step >= max_steps
        should_log = _should_log_progress(
            global_step,
            completed_updates,
            updates_per_epoch,
            log_every_n_steps,
            reached_max_steps,
        )
        if should_log:
            window_metrics = _reduce_metrics(progress_sums, device, context)
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "epoch_step": completed_updates,
                            "epoch_steps": updates_per_epoch,
                            "global_step": global_step,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "window_steps": global_step - progress_start_step,
                            "window_metrics": window_metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            progress_sums = {}
            progress_start_step = global_step
        if reached_max_steps:
            break
    metrics = _reduce_metrics(sums, device, context)
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
    metrics["skipped_optimizer_steps"] = float(skipped_optimizer_steps)
    return metrics, global_step


def _amp_step_succeeded(previous_scale: float, current_scale: float) -> bool:
    """A decreasing GradScaler scale means optimizer.step was skipped."""

    return current_scale >= previous_scale


@contextmanager
def _gradient_sync_context(
    model: torch.nn.Module, synchronize_gradients: bool
):
    """Delay DDP reduction until the final microbatch in an update."""

    if synchronize_gradients or not hasattr(model, "no_sync"):
        yield
        return
    with model.no_sync():
        yield


def _updates_per_epoch(loader_batches: int, grad_accumulation_steps: int) -> int:
    """Return the upstream trainer's number of complete optimizer updates."""

    updates = loader_batches // grad_accumulation_steps
    if updates < 1:
        raise ValueError(
            "grad_accumulation_steps exceeds the number of training batches; "
            "reduce it or provide more training samples"
        )
    return updates


def _should_log_progress(
    global_step: int,
    completed: int,
    total: int,
    interval: int,
    reached_limit: bool = False,
) -> bool:
    """Log periodic progress plus the final update or batch in a phase."""

    if interval <= 0:
        return False
    return global_step % interval == 0 or completed == total or reached_limit


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    specs: list[LossSpec],
    loss_functions: dict[str, DDPLoss],
    transforms: EquiformerV3CheckpointTransforms,
    context: DistributedContext,
    amp: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    log_every_n_batches: int,
) -> dict[str, float]:
    # Gradient models derive forces/stress from energy, so validation must keep
    # autograd enabled even though no parameter update is performed.
    model.eval()
    sums: dict[str, tuple[float, int]] = {}
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            enabled=amp,
            dtype=amp_dtype,
        ):
            prediction = model(batch)
            loss, components = _loss(
                prediction, batch, specs, transforms, loss_functions
            )
        physical = _physical_metrics(prediction, batch, specs, transforms)
        _collect_batch_metrics(sums, loss, components, physical)
        if context.is_main and _should_log_progress(
            batch_index,
            batch_index,
            total_batches,
            log_every_n_batches,
        ):
            print(
                json.dumps(
                    {
                        "event": "validation_progress",
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": total_batches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return _reduce_metrics(sums, device, context)


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
    max_steps: int | None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if scheduler_config.get("name", "cosine").lower() not in {
        "cosine",
        "lambdalr",
    }:
        raise ValueError("only the official cosine LambdaLR scheduler is supported")
    del max_steps
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(
        float(scheduler_config.get("warmup_epochs", 0.0)) * steps_per_epoch
    )
    # Official full runs always have at least one warmup update. Keep bounded
    # smoke configurations away from the upstream zero-step division edge case.
    warmup_steps = max(1, min(warmup_steps, total_steps))
    warmup_factor = float(scheduler_config.get("warmup_factor", 0.0))
    minimum = float(scheduler_config.get("lr_min_factor", 0.01))
    lr_lambda = CosineLRLambda(
        warmup_epochs=warmup_steps,
        warmup_factor=warmup_factor,
        epochs=total_steps,
        lr_min_factor=minimum,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _state_dict_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


def _inference_state_dict(model: torch.nn.Module, ema: ModelEMA | None) -> dict:
    state = _state_dict_cpu(model.state_dict())
    if ema is not None:
        for name, value in ema.shadow.items():
            state[name] = value.detach().cpu()
    return state


def _save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    transforms: EquiformerV3CheckpointTransforms,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema: ModelEMA | None,
    epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    source_document: dict[str, Any] | None,
    scaler: torch.GradScaler | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_metadata = copy.deepcopy((source_document or {}).get("metadata", {}))
    source_metadata.update(
        {
            "onescience_equiformer_v3_history": history,
            "training_mode": training_config["mode"],
            "source_checkpoint": training_config.get("initialization_checkpoint"),
            "resume_checkpoint": training_config.get("resume"),
            "global_step": global_step,
            "ema_decay": ema.decay if ema is not None else None,
            "amp": scaler is not None,
        }
    )
    document = {
        "config": {"model": copy.deepcopy(model_config), "training": training_config},
        "normalizers": {
            name: _state_dict_cpu(module.state_dict())
            for name, module in transforms.normalizers.items()
        },
        "elementrefs": {
            name: _state_dict_cpu(module.state_dict())
            for name, module in transforms.elementrefs.items()
        },
        "state_dict": _inference_state_dict(model, ema),
        "training_state_dict": _state_dict_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "amp_state_dict": scaler.state_dict() if scaler is not None else None,
        "training_state": {"epoch": epoch, "global_step": global_step},
        "metadata": source_metadata,
    }
    torch.save(document, output)
    history_path = output.with_name(output.name + ".history.json")
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")


def _fit_transforms(
    config: dict[str, Any],
    transforms: EquiformerV3CheckpointTransforms,
    dataset,
) -> None:
    if config.get("fit_element_references"):
        fitted = fit_linear_references(
            targets=["energy"],
            dataset=dataset,
            batch_size=config["batch_size"],
            num_batches=config.get("fit_statistics_batches"),
            num_workers=config["workers"],
            log_metrics=False,
            shuffle=False,
        )
        transforms.elementrefs["energy"] = fitted["energy"]
    requested = config.get("fit_normalizers")
    if requested:
        targets = (
            [spec.name for spec in _loss_specs(config)]
            if requested is True
            else list(requested)
        )
        fitted = fit_normalizers(
            targets=targets,
            dataset=dataset,
            batch_size=config["batch_size"],
            num_batches=config.get("fit_statistics_batches"),
            num_workers=config["workers"],
            shuffle=False,
            element_references=dict(transforms.elementrefs),
        )
        for name, normalizer in fitted.items():
            transforms.normalizers[name] = normalizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML path")
    parser.add_argument("--mode", choices=sorted(MODE_ALIASES))
    parser.add_argument("--checkpoint", dest="initialization_checkpoint")
    parser.add_argument("--transforms-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-atoms", type=int)
    parser.add_argument("--log-every-n-steps", type=int)
    parser.add_argument("--log-every-n-validation-batches", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"))
    parser.add_argument(
        "--clear-checkpoint-transforms",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).expanduser().open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            raw[key] = value
    return _normalize_config(raw)


def _validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in ("train", "val", "output") if not config.get(key)]
    if missing:
        raise ValueError("missing required config fields: " + ", ".join(missing))
    for name in ("epochs", "batch_size", "eval_batch_size", "grad_accumulation_steps"):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if config.get("max_steps") is not None and int(config["max_steps"]) < 1:
        raise ValueError("max_steps must be positive")
    if config.get("max_atoms") is not None and int(config["max_atoms"]) < 1:
        raise ValueError("max_atoms must be positive")
    for name in ("log_every_n_steps", "log_every_n_validation_batches"):
        if int(config.get(name, 0)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if config["amp_dtype"] not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if config["mode"] == "resume_training" and (
        config.get("transforms_checkpoint")
        or config.get("transforms")
        or config.get("clear_checkpoint_transforms")
    ):
        raise ValueError(
            "resume_training restores transforms from the resume checkpoint; "
            "remove transforms_checkpoint/transforms overrides"
        )
    specs = _loss_specs(config)
    dens_params = _denoising_pos_params(config)
    if dens_params.enabled:
        if not dens_params.fixed_noise_std:
            raise ValueError("the official DeNS trainer requires fixed_noise_std=true")
        if not 0.0 <= dens_params.prob <= 1.0:
            raise ValueError("denoising_pos_params.prob must be between zero and one")
        if dens_params.std <= 0.0:
            raise ValueError("denoising_pos_params.std must be positive")
        for name, value in (
            ("corrupt_ratio", dens_params.corrupt_ratio),
            ("strict_max_ratio", dens_params.strict_max_ratio),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"denoising_pos_params.{name} must be between zero and one"
                )
        if dens_params.min_num_atoms is not None and dens_params.min_num_atoms < 1:
            raise ValueError("denoising_pos_params.min_num_atoms must be positive")
        if dens_params.coefficient <= 0.0:
            raise ValueError("denoising_pos_coefficient must be positive")
        force_specs = [spec for spec in specs if spec.name == "forces"]
        if len(force_specs) != 1 or force_specs[0].function != "l2mae":
            raise ValueError("DeNS requires one forces loss using l2mae")
        if config["mode"] == "train_from_scratch":
            model_config = config.get("model") or {}
            if model_config.get("name") != "equiformer_v3_dens":
                raise ValueError("DeNS requires model.name=equiformer_v3_dens")
            if not model_config.get("direct_prediction", False):
                raise ValueError("DeNS pre-training requires direct_prediction=true")


def main() -> None:
    args = _parse_args()
    try:
        config = _load_config(args)
        _validate_config(config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but torch.cuda.is_available() is false")

    context = _init_distributed(config["device"], config["backend"])
    try:
        device = (
            torch.device(f"cuda:{context.local_rank}")
            if config["device"].startswith("cuda")
            else torch.device(config["device"])
        )
        torch.manual_seed(int(config["seed"]) + context.rank)
        dens_params = _denoising_pos_params(config)
        model, transforms, model_config, source_document = _initialize_model(config)
        model = model.to(device)

        train_loader = _loader(
            config["train"],
            config["batch_size"],
            config["workers"],
            config.get("max_train_samples"),
            context,
            train=True,
            seed=config["seed"],
            max_atoms=config.get("max_atoms"),
            load_balancing=config.get("load_balancing"),
            load_balancing_on_error=config["load_balancing_on_error"],
            device=device,
        )
        val_loader = _loader(
            config["val"],
            config["eval_batch_size"],
            config["workers"],
            config.get("max_val_samples"),
            context,
            train=False,
            seed=config["seed"] + 1,
            max_atoms=config.get("eval_max_atoms"),
            load_balancing=config.get("load_balancing"),
            load_balancing_on_error=config["load_balancing_on_error"],
            device=device,
        )
        if config["mode"] != "resume_training":
            _fit_transforms(config, transforms, train_loader.dataset)
        transforms = transforms.to(device)

        optimizer_config = config["optimizer"]
        if optimizer_config["name"].lower() != "adamw":
            raise ValueError("only the official AdamW optimizer is supported")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["lr"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(value) for value in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]),
        )
        updates_per_epoch = _updates_per_epoch(
            len(train_loader), int(config["grad_accumulation_steps"])
        )
        scheduler = _cosine_scheduler(
            optimizer,
            config["scheduler"],
            updates_per_epoch,
            int(config["epochs"]),
            config.get("max_steps"),
        )
        if config["amp"] and device.type != "cuda":
            raise ValueError("amp requires a CUDA/DCU device")
        amp_dtype = getattr(torch, config["amp_dtype"])
        scaler = (
            torch.GradScaler(
                "cuda", init_scale=float(config["amp_init_scale"])
            )
            if config["amp"]
            else None
        )
        ema = (
            ModelEMA(model, float(config["ema_decay"]))
            if config.get("ema_decay")
            else None
        )

        start_epoch = 0
        global_step = 0
        history: list[dict[str, Any]] = []
        if config["mode"] == "resume_training":
            state = source_document.get("training_state", {})
            start_epoch = int(state.get("epoch", -1)) + 1
            global_step = int(state.get("global_step", 0))
            optimizer.load_state_dict(source_document["optimizer_state_dict"])
            scheduler.load_state_dict(source_document["scheduler_state_dict"])
            if scaler is not None and source_document.get("amp_state_dict") is not None:
                scaler.load_state_dict(source_document["amp_state_dict"])
            if ema is not None and source_document.get("ema_state_dict") is not None:
                ema.load_state_dict(source_document["ema_state_dict"], device)
            history = list(
                source_document.get("metadata", {}).get(
                    "onescience_equiformer_v3_history", []
                )
            )

        if context.enabled:
            find_unused_parameters = bool(
                config["ddp_find_unused_parameters"]
            )
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "ddp_setup",
                            "find_unused_parameters": find_unused_parameters,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )
        specs = _loss_specs(config)
        loss_functions = _loss_functions(specs)
        output = Path(config["output"])

        for epoch in range(start_epoch, int(config["epochs"])):
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            elif isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_metrics, global_step = _run_train_epoch(
                model,
                train_loader,
                device,
                optimizer,
                scheduler,
                ema,
                specs,
                loss_functions,
                transforms,
                context,
                scaler,
                amp_dtype,
                int(config["grad_accumulation_steps"]),
                config.get("clip_grad_norm"),
                global_step,
                config.get("max_steps"),
                epoch,
                int(config["log_every_n_steps"]),
                dens_params,
            )
            base_model = _unwrap(model)
            if ema is None:
                val_metrics = _run_validation(
                    model,
                    val_loader,
                    device,
                    specs,
                    loss_functions,
                    transforms,
                    context,
                    scaler is not None,
                    amp_dtype,
                    epoch,
                    int(config["log_every_n_validation_batches"]),
                )
            else:
                with ema.apply(base_model):
                    val_metrics = _run_validation(
                        model,
                        val_loader,
                        device,
                        specs,
                        loss_functions,
                        transforms,
                        context,
                        scaler is not None,
                        amp_dtype,
                        epoch,
                        int(config["log_every_n_validation_batches"]),
                    )
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "val": val_metrics,
            }
            if context.is_main:
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                _save_checkpoint(
                    output,
                    base_model,
                    transforms,
                    model_config,
                    config,
                    optimizer,
                    scheduler,
                    ema,
                    epoch,
                    global_step,
                    history,
                    source_document,
                    scaler,
                )
                print(f"saved checkpoint: {output}", flush=True)
            if config.get("max_steps") is not None and global_step >= int(
                config["max_steps"]
            ):
                break
    finally:
        _close_distributed(context)


if __name__ == "__main__":
    main()
