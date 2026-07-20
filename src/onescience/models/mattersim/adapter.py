"""Checkpoint and model factories for the OneScience MatterSim integration."""

import os
from pathlib import Path

import torch

DEFAULT_CHECKPOINT = "mattersim-v1.0.0-1M.pth"


def resolve_checkpoint(checkpoint: str | os.PathLike | None = None) -> str:
    """Resolve a MatterSim checkpoint without changing its native format.

    Explicit checkpoint values are passed through so MatterSim model aliases keep
    working. When no value is supplied, the shared OneScience model store is
    checked before falling back to MatterSim's native alias and download logic.
    """
    if checkpoint is not None:
        return str(Path(checkpoint).expanduser())

    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if models_dir:
        shared_checkpoint = Path(models_dir).expanduser() / "mattersim" / DEFAULT_CHECKPOINT
        if shared_checkpoint.is_file():
            return str(shared_checkpoint)

    return DEFAULT_CHECKPOINT


def _device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_potential(
    checkpoint: str | os.PathLike | None = None,
    device: str | None = None,
    load_training_state: bool = False,
    **kwargs,
):
    """Load a MatterSim ``Potential`` from a resolved checkpoint."""
    from onescience.utils.mattersim.potential import Potential

    return Potential.from_checkpoint(
        load_path=resolve_checkpoint(checkpoint),
        device=_device(device),
        load_training_state=load_training_state,
        **kwargs,
    )


def load_calculator(
    checkpoint: str | os.PathLike | None = None,
    device: str | None = None,
    **kwargs,
):
    """Create an ASE-compatible ``MatterSimCalculator``."""
    from onescience.utils.mattersim.calculator import MatterSimCalculator

    return MatterSimCalculator.from_checkpoint(
        resolve_checkpoint(checkpoint), device=_device(device), **kwargs
    )


def predict_structures(
    atoms,
    checkpoint: str | os.PathLike | None = None,
    device: str | None = None,
    batch_size: int = 16,
    include_forces: bool = True,
    include_stresses: bool = False,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
):
    """Predict ASE structures with the MatterSim potential."""
    from onescience.datapipes.materials.mattersim import build_dataloader

    potential = load_potential(checkpoint=checkpoint, device=device)
    dataloader = build_dataloader(
        atoms=list(atoms),
        batch_size=batch_size,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        only_inference=True,
    )
    energies, forces, stresses = potential.predict_properties(
        dataloader,
        include_forces=include_forces,
        include_stresses=include_stresses,
    )
    return {
        "energies": energies,
        "forces": forces,
        "stresses": stresses,
    }
