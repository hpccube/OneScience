"""Build NequIP ASE calculators from compiled models or checkpoints."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import torch
from ase import Atoms

from onescience.datapipes.materials.nequip.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from onescience.utils.nequip.integrations.ase import NequIPCalculator
from onescience.models.nequip.model.nequip_models import NequIPGNNModel
from onescience.models.nequip.model.saved_models import load_saved_model
from onescience.utils.nequip.internal.global_state import set_global_state


def _default_type_names(model_type_names: Optional[Sequence[str]]) -> List[str]:
    return list(model_type_names) if model_type_names is not None else ["H", "C", "O", "Cu"]


def _guess_r_max(model: torch.nn.Module, fallback: float = 4.0) -> float:
    return float(getattr(model, "r_max", fallback))


def _type_map_from_names(type_names: Sequence[str]) -> Union[bool, Dict[str, str]]:
    """Return an identity mapping when all type names are chemical symbols."""
    import ase.data
    if all(name in ase.data.atomic_numbers for name in type_names):
        return True
    return {name: name for name in type_names}


def load_nequip_model_for_ase(
    checkpoint: str,
    device: Union[str, torch.device] = "cuda",
) -> torch.nn.Module:
    """Load a NequIP model from a checkpoint or package file.

    The loaded model is moved to ``device`` and set to eval mode. Legacy
    ``nequip.`` Hydra targets embedded in the checkpoint are rewritten to the
    OneScience package namespace.
    """
    set_global_state()
    model = load_saved_model(input_path=checkpoint)
    if isinstance(device, str):
        device = torch.device(device)
    return model.to(device).eval()


def build_nequip_calculator(
    model: Optional[torch.nn.Module] = None,
    checkpoint: Optional[str] = None,
    compiled_model: Optional[str] = None,
    device: Union[str, torch.device] = "cuda",
    model_type_names: Optional[Sequence[str]] = None,
    r_max: Optional[float] = None,
    chemical_species_to_atom_type_map: Optional[Union[bool, Dict[str, str]]] = None,
    energy_units_to_eV: float = 1.0,
    length_units_to_A: float = 1.0,
) -> NequIPCalculator:
    """Build a NequIP ASE calculator.

    Exactly one of ``model``, ``checkpoint`` or ``compiled_model`` must be
    provided. The official NequIP workflow recommends ``compiled_model`` for
    inference (see https://nequip.readthedocs.io/en/latest/integrations/ase.html).
    """
    provided = sum(x is not None for x in [model, checkpoint, compiled_model])
    if provided != 1:
        raise ValueError(
            "Exactly one of model, checkpoint, or compiled_model must be provided"
        )

    set_global_state()

    if compiled_model is not None:
        if chemical_species_to_atom_type_map is None:
            chemical_species_to_atom_type_map = _type_map_from_names(
                _default_type_names(model_type_names)
            )
        return NequIPCalculator.from_compiled_model(
            compile_path=compiled_model,
            device=device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
        )

    if checkpoint is not None:
        model = load_nequip_model_for_ase(checkpoint, device=device)

    assert model is not None
    model.eval()

    type_names = _default_type_names(model_type_names)
    if r_max is None:
        r_max = _guess_r_max(model)

    transforms: List[Any] = [
        ChemicalSpeciesToAtomTypeMapper(model_type_names=type_names),
        NeighborListTransform(r_max=r_max),
    ]
    return NequIPCalculator(
        model=model,
        device=device,
        transforms=transforms,
        energy_units_to_eV=energy_units_to_eV,
        length_units_to_A=length_units_to_A,
    )


# Backwards-compatible alias.
NequIPCalculatorBuilder = build_nequip_calculator
