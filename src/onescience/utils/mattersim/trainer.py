"""OneScience fine-tuning interface for the MatterSim integration."""

from __future__ import annotations

import json
import os
import pickle
import random
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from ase.units import GPa

from onescience.models.mattersim import resolve_checkpoint


@dataclass
class FineTuneConfig:
    """Configuration accepted by MatterSim's native fine-tuning loop."""

    train_data_path: str
    checkpoint: str | os.PathLike | None = None
    save_path: str | os.PathLike = "./results"
    valid_data_path: str | None = None
    run_name: str = "onescience-mattersim"
    device: str | None = None
    epochs: int = 1000
    batch_size: int = 16
    lr: float = 2e-4
    seed: int = 42
    cutoff: float = 5.0
    threebody_cutoff: float = 4.0
    step_size: int = 10
    include_forces: bool = True
    include_stresses: bool = False
    force_loss_ratio: float = 1.0
    stress_loss_ratio: float = 0.1
    early_stop_patience: int = 10
    save_checkpoint: bool = True
    ckpt_interval: int = 10
    re_normalize: bool = False
    scale_key: str = "per_species_forces_rms"
    shift_key: str = "per_species_energy_mean_linear_reg"
    init_scale: float | None = None
    init_shift: float | None = None
    trainable_scale: bool = False
    trainable_shift: bool = False
    wandb: bool = False
    wandb_api_key: str | None = None
    wandb_project: str = "onescience-mattersim"


def _distributed_environment() -> None:
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _result(save_path: Path) -> dict:
    result = {
        "save_path": str(save_path),
        "best_checkpoint": str(save_path / "best_model.pth"),
        "last_checkpoint": str(save_path / "last_model.pth"),
    }
    checkpoint = save_path / "last_model.pth"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["last_epoch"] = state.get("last_epoch")
        result["metrics"] = state.get("validation_metrics", {})
    return result


def _native_finetune_main(args) -> None:
    """Run the integrated MatterSim fine-tuning loop."""
    from onescience.datapipes.materials.mattersim import build_dataloader
    from onescience.modules.func_utils.mattersim_scaling import AtomScaling

    from .atoms import AtomsAdaptor
    from .logger import get_logger
    from .potential import Potential

    logger = get_logger()
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if args.device == "cuda" else "gloo")
    args_dict = vars(args)
    wandb_module = None
    if args.wandb:
        import wandb as wandb_module

        if local_rank == 0:
            wandb_api_key = args.wandb_api_key or os.getenv("WANDB_API_KEY")
            wandb_module.login(key=wandb_api_key)
            wandb_module.init(
                project=args.wandb_project, name=args.run_name, config=args
            )
        args_dict["wandb"] = wandb_module

    dist.barrier()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)

    if args.train_data_path.endswith(".pkl"):
        with open(args.train_data_path, "rb") as stream:
            atoms_train = pickle.load(stream)
    else:
        atoms_train = AtomsAdaptor.from_file(filename=args.train_data_path)

    energies = [atoms.get_potential_energy() for atoms in atoms_train]
    forces = [atoms.get_forces() for atoms in atoms_train] if args.include_forces else None
    stresses = (
        [atoms.get_stress(voigt=False) / GPa for atoms in atoms_train]
        if args.include_stresses
        else None
    )
    logger.info("Processing training datasets...")
    dataloader = build_dataloader(
        atoms_train,
        energies,
        forces,
        stresses,
        shuffle=True,
        pin_memory=args.device == "cuda",
        is_distributed=True,
        **args_dict,
    )

    if args.re_normalize:
        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=energies,
            forces=forces,
            verbose=True,
            **args_dict,
        ).to(args.device)

    if args.valid_data_path is not None:
        if args.valid_data_path.endswith(".pkl"):
            with open(args.valid_data_path, "rb") as stream:
                atoms_val = pickle.load(stream)
        else:
            atoms_val = AtomsAdaptor.from_file(filename=args.valid_data_path)
        val_energies = [atoms.get_potential_energy() for atoms in atoms_val]
        val_forces = (
            [atoms.get_forces() for atoms in atoms_val]
            if args.include_forces
            else None
        )
        val_stresses = (
            [atoms.get_stress(voigt=False) / GPa for atoms in atoms_val]
            if args.include_stresses
            else None
        )
        logger.info("Processing validation datasets...")
        val_dataloader = build_dataloader(
            atoms_val,
            val_energies,
            val_forces,
            val_stresses,
            pin_memory=args.device == "cuda",
            is_distributed=True,
            **args_dict,
        )
    else:
        val_dataloader = None

    potential = Potential.from_checkpoint(
        load_path=args.load_model_path,
        load_training_state=False,
        **args_dict,
    )
    if args.re_normalize:
        potential.model.set_normalizer(scale)
    if args.device == "cuda":
        potential.model = torch.nn.parallel.DistributedDataParallel(potential.model)
    dist.barrier()
    potential.train_model(
        dataloader,
        val_dataloader,
        loss=torch.nn.HuberLoss(delta=0.01),
        is_distributed=True,
        **args_dict,
    )
    if local_rank == 0 and args.save_checkpoint and wandb_module is not None:
        wandb_module.save(os.path.join(args.save_path, "best_model.pth"))


def finetune(config: FineTuneConfig | None = None, **kwargs) -> dict:
    """Fine-tune MatterSim through its validated native DDP training loop.

    The function can be called directly for a single process or from ``torchrun``
    for multi-process training. Only rank zero returns checkpoint metadata.
    """
    if config is not None and kwargs:
        raise TypeError("Pass either FineTuneConfig or keyword arguments, not both")
    config = config or FineTuneConfig(**kwargs)

    if config.device not in {None, "cpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'cuda', or None")
    if config.threebody_cutoff > config.cutoff:
        raise ValueError("threebody_cutoff must not exceed cutoff")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    values = asdict(config)
    values["train_data_path"] = str(Path(config.train_data_path).expanduser())
    values["valid_data_path"] = (
        str(Path(config.valid_data_path).expanduser())
        if config.valid_data_path is not None
        else None
    )
    save_path = Path(config.save_path).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    values["save_path"] = str(save_path)
    values["load_model_path"] = resolve_checkpoint(config.checkpoint)
    values["device"] = config.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    values.pop("checkpoint")

    _distributed_environment()
    try:
        _native_finetune_main(Namespace(**values))
        if dist.is_initialized():
            dist.barrier()
        result = _result(save_path) if int(os.environ["RANK"]) == 0 else {}
        if result:
            print(json.dumps(result, indent=2))
        return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class MatterSimTrainer:
    """Reusable trainer object matching other OneScience material APIs."""

    def __init__(self, config: FineTuneConfig):
        self.config = config

    def fit(self) -> dict:
        return finetune(self.config)


__all__ = ["FineTuneConfig", "MatterSimTrainer", "finetune"]
