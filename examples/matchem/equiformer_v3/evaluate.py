"""Evaluate an Equiformer V3 checkpoint on an independent ASE database."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from onescience.utils.equiformer_v3 import (
    EquiformerV3CheckpointTransforms,
    load_equiformer_v3_checkpoint,
)

from finetune import _loader


@dataclass
class ErrorAccumulator:
    """Accumulate MAE/RMSE inputs without retaining the full dataset."""

    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    count: int = 0

    def update(self, error: torch.Tensor) -> None:
        error = error.detach().reshape(-1).double()
        self.absolute_sum += float(error.abs().sum())
        self.squared_sum += float(error.square().sum())
        self.count += error.numel()

    def result(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("cannot compute metrics for an empty tensor")
        return {
            "mae": self.absolute_sum / self.count,
            "rmse": (self.squared_sum / self.count) ** 0.5,
        }


@dataclass
class MeanAccumulator:
    total: float = 0.0
    count: int = 0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1).double()
        self.total += float(values.sum())
        self.count += values.numel()

    def result(self) -> float:
        if self.count == 0:
            raise ValueError("cannot compute a mean for an empty tensor")
        return self.total / self.count


def _force_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.detach().float().reshape(-1, 3)
    target = target.detach().float().reshape(-1, 3)
    if prediction.shape != target.shape or prediction.shape[0] == 0:
        raise ValueError(
            "cannot compute force cosine similarity for an empty/mismatched tensor"
        )
    return torch.cosine_similarity(prediction, target, dim=1).detach().cpu()


def _force_magnitude_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    prediction = prediction.detach().float().reshape(-1, 3)
    target = target.detach().float().reshape(-1, 3)
    if prediction.shape != target.shape or prediction.shape[0] == 0:
        raise ValueError(
            "cannot compute force magnitude error for an empty/mismatched tensor"
        )
    return (
        torch.linalg.vector_norm(prediction, dim=1)
        - torch.linalg.vector_norm(target, dim=1)
    ).abs().cpu()


def _energy_force_success(
    energy_error: torch.Tensor,
    force_error: torch.Tensor,
    selected_natoms: torch.Tensor,
) -> torch.Tensor:
    """Match FairChem's OC20 energy/force threshold metric per structure."""

    energy_error = energy_error.detach().reshape(-1).abs()
    force_error = force_error.detach().float().reshape(-1, 3).abs()
    selected_natoms = selected_natoms.detach().reshape(-1).long()
    if energy_error.numel() != selected_natoms.numel():
        raise ValueError("energy count and per-structure atom counts differ")
    if int(selected_natoms.sum()) != force_error.shape[0]:
        raise ValueError("force count and per-structure atom counts differ")
    if bool((selected_natoms == 0).any()):
        raise ValueError("a structure contains no selected atoms for force evaluation")

    successes = []
    for structure_energy_error, structure_force_error in zip(
        energy_error, torch.split(force_error, selected_natoms.tolist())
    ):
        successes.append(
            (structure_energy_error < 0.02)
            & (structure_force_error.max() < 0.03)
        )
    return torch.stack(successes).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Independent ASE DB/ASE-LMDB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--free-atoms-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate forces only on atoms not constrained by FixAtoms (default: true).",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Also report stress when both checkpoint and dataset provide it.",
    )
    parser.add_argument(
        "--include-oc20-threshold",
        action="store_true",
        help="Report FairChem's OC20 energy/force threshold success rate.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but is unavailable")

    model = load_equiformer_v3_checkpoint(args.checkpoint).to(device)
    model.eval()
    transforms = EquiformerV3CheckpointTransforms.from_checkpoint(
        args.checkpoint
    ).to(device)
    loader = _loader(
        args.data,
        args.batch_size,
        args.workers,
        max_samples=args.max_samples,
        train=False,
        seed=args.seed,
    )

    energy_errors = ErrorAccumulator()
    energy_per_atom_errors = ErrorAccumulator()
    force_errors = ErrorAccumulator()
    force_cosines = MeanAccumulator()
    force_magnitude_errors = MeanAccumulator()
    energy_force_successes = MeanAccumulator()
    stress_errors = ErrorAccumulator()
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            pred_energy = transforms.denormalize_prediction(
                "energy", prediction["energy"], batch
            )
            target_energy = batch.energy.reshape_as(pred_energy)
            energy_error = pred_energy - target_energy
            energy_errors.update(energy_error)
            natoms = batch.natoms.to(pred_energy).reshape(
                (-1,) + (1,) * (pred_energy.ndim - 1)
            )
            energy_per_atom_errors.update(energy_error / natoms)

            pred_forces = transforms.denormalize_prediction(
                "forces", prediction["forces"], batch
            )
            target_forces = batch.forces.reshape_as(pred_forces)
            selected_natoms = batch.natoms
            if args.free_atoms_only and hasattr(batch, "fixed"):
                free_mask = batch.fixed.reshape(-1) == 0
                selected_natoms = torch.stack(
                    [
                        structure_mask.sum()
                        for structure_mask in torch.split(
                            free_mask, batch.natoms.tolist()
                        )
                    ]
                )
                pred_forces = pred_forces.reshape(-1, 3)[free_mask]
                target_forces = target_forces.reshape(-1, 3)[free_mask]
            if not pred_forces.numel():
                selection = "free atoms" if args.free_atoms_only else "atoms"
                raise ValueError(f"an evaluation batch contains no selected {selection}")
            force_error = pred_forces - target_forces
            force_errors.update(force_error)
            force_cosines.update(_force_cosine(pred_forces, target_forces))
            force_magnitude_errors.update(
                _force_magnitude_error(pred_forces, target_forces)
            )
            if args.include_oc20_threshold:
                energy_force_successes.update(
                    _energy_force_success(
                        energy_error, force_error, selected_natoms
                    )
                )
            if (
                args.include_stress
                and "stress" in prediction
                and hasattr(batch, "stress")
            ):
                pred_stress = transforms.denormalize_prediction(
                    "stress", prediction["stress"], batch
                )
                target_stress = batch.stress.reshape_as(pred_stress)
                stress_errors.update(pred_stress - target_stress)

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "data": str(Path(args.data).expanduser()),
        "samples": len(loader.dataset),
        "force_atoms": force_cosines.count,
        "free_atoms_only": args.free_atoms_only,
        "energy_total_eV": energy_errors.result(),
        "energy_per_atom_eV": energy_per_atom_errors.result(),
    }
    if force_errors.count == 0:
        selection = "free atoms" if args.free_atoms_only else "atoms"
        raise ValueError(f"the evaluation dataset contains no selected {selection}")
    result["forces_eV_per_A"] = force_errors.result()
    result["forces_cosine_similarity"] = {"mean": force_cosines.result()}
    result["forces_magnitude_error_eV_per_A"] = {
        "mean": force_magnitude_errors.result()
    }
    if args.include_oc20_threshold:
        result["energy_forces_within_threshold"] = {
            "fraction": energy_force_successes.result(),
            "energy_threshold_eV": 0.02,
            "force_threshold_eV_per_A": 0.03,
        }
    if stress_errors.count:
        result["stress_eV_per_A3"] = stress_errors.result()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
