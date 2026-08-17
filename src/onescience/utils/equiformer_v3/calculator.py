"""ASE calculator for OneScience Equiformer V3 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from onescience.datapipes.materials.custom_stack import data_list_collater
from onescience.datapipes.materials.custom_stack.core.atomic_data import AtomicData
from onescience.utils.equiformer_v3.checkpoint import (
    EquiformerV3CheckpointTransforms,
    load_equiformer_v3_checkpoint,
)

if TYPE_CHECKING:
    from ase import Atoms
    from torch import nn


class EquiformerV3Calculator(Calculator):
    """Run an official Equiformer V3 checkpoint through ASE.

    Energies are returned in eV, forces in eV/Angstrom, and stresses in
    ASE's six-component Voigt convention (eV/Angstrom^3).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device | None = None,
        transforms: EquiformerV3CheckpointTransforms | None = None,
    ):
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.transforms = (
            transforms or EquiformerV3CheckpointTransforms()
        ).to(self.device)
        self.backbone = getattr(self.model, "backbone", self.model)

        if not hasattr(self.backbone, "max_neighbors"):
            raise TypeError("Equiformer V3 model is missing max_neighbors")
        if not hasattr(self.backbone, "cutoff"):
            raise TypeError("Equiformer V3 model is missing cutoff")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        jd_path: str | Path | None = None,
    ) -> EquiformerV3Calculator:
        """Load a native Equiformer V3 checkpoint without installing FairChem."""

        model = load_equiformer_v3_checkpoint(checkpoint_path, jd_path=jd_path)
        transforms = EquiformerV3CheckpointTransforms.from_checkpoint(checkpoint_path)
        return cls(model=model, device=device, transforms=transforms)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        if atoms is None or len(atoms) == 0:
            raise ValueError(
                "EquiformerV3Calculator requires a non-empty ASE Atoms object."
            )

        Calculator.calculate(self, atoms, properties, system_changes)
        data = AtomicData.from_ase(
            atoms,
            max_neigh=self.backbone.max_neighbors,
            radius=self.backbone.cutoff,
            r_edges=False,
            r_energy=False,
            r_forces=False,
            r_stress=False,
        )
        batch = data_list_collater([data], otf_graph=True).to(self.device)

        # Gradient checkpoints obtain forces and stress through autograd even
        # while the model is in evaluation mode.
        with torch.enable_grad():
            prediction = self.model(batch)

        if "energy" not in prediction:
            raise RuntimeError("Equiformer V3 model did not return an energy")

        energy_prediction = self.transforms.denormalize_prediction(
            "energy", prediction["energy"], batch
        )
        energy = float(energy_prediction.detach().cpu().reshape(-1)[0])
        self.results = {"energy": energy, "free_energy": energy}

        if "forces" in prediction:
            forces = self.transforms.denormalize_prediction(
                "forces", prediction["forces"], batch
            )
            self.results["forces"] = forces.detach().cpu().reshape(-1, 3).numpy()
        if "stress" in prediction:
            stress_prediction = self.transforms.denormalize_prediction(
                "stress", prediction["stress"], batch
            )
            stress = stress_prediction.detach().cpu().reshape(3, 3).numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
