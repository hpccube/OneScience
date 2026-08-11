"""ASE calculator for OneScience eSEN checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from onescience.datapipes.materials.custom_stack import data_list_collater
from onescience.datapipes.materials.custom_stack.core.atomic_data import AtomicData
from onescience.utils.esen.checkpoint import ESENCheckpointTransforms
from onescience.utils.uma.common.utils import load_model_and_weights_from_checkpoint

if TYPE_CHECKING:
    from ase import Atoms
    from onescience.models.UMA.base import HydraModel


class eSENCalculator(Calculator):
    """Run an eSEN checkpoint through ASE.

    The calculator returns energies in eV, forces in eV/Angstrom, and stresses
    in ASE's six-component Voigt convention (eV/Angstrom^3).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: HydraModel,
        device: str | torch.device | None = None,
        transforms: ESENCheckpointTransforms | None = None,
    ):
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.transforms = (transforms or ESENCheckpointTransforms()).to(self.device)
        self.backbone = self.model.backbone

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: str | torch.device | None = None
    ) -> eSENCalculator:
        """Load a native FairChem-v1 eSEN checkpoint without FairChem."""
        # Registers the eSEN backbone and heads used by checkpoint configs.
        import onescience.models.esen  # noqa: F401

        model = load_model_and_weights_from_checkpoint(checkpoint_path)
        transforms = ESENCheckpointTransforms.from_checkpoint(checkpoint_path)
        return cls(model=model, device=device, transforms=transforms)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        if atoms is None or len(atoms) == 0:
            raise ValueError("eSENCalculator requires a non-empty ASE Atoms object.")

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
        prediction = self.model(batch)

        energy_prediction = self.transforms.denormalize_prediction(
            "energy", prediction["energy"], batch
        )
        energy = float(energy_prediction.detach().cpu().reshape(-1)[0])
        self.results = {"energy": energy, "free_energy": energy}

        if "forces" in prediction:
            forces = self.transforms.denormalize_prediction(
                "forces", prediction["forces"], batch
            )
            self.results["forces"] = forces.detach().cpu().numpy()
        if "stress" in prediction:
            stress_prediction = self.transforms.denormalize_prediction(
                "stress", prediction["stress"], batch
            )
            stress = stress_prediction.detach().cpu().reshape(3, 3).numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
