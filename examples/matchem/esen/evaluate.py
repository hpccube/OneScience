"""Evaluate an eSEN checkpoint in physical units on an independent ASE DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from onescience.utils.esen.checkpoint import ESENCheckpointTransforms
from onescience.utils.uma.common.utils import load_model_and_weights_from_checkpoint

from finetune import _loader


def _stats(error: torch.Tensor) -> dict[str, float]:
    error = error.detach().reshape(-1).double()
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Independent ASE DB/ASE-LMDB")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/DCU was requested but is unavailable")
    import onescience.models.esen  # noqa: F401

    model = load_model_and_weights_from_checkpoint(args.checkpoint).to(device)
    model.eval()
    transforms = ESENCheckpointTransforms.from_checkpoint(args.checkpoint).to(device)
    loader = _loader(
        args.data,
        args.batch_size,
        args.workers,
        max_samples=args.max_samples,
        train=False,
        seed=args.seed,
    )

    energy_errors = []
    energy_per_atom_errors = []
    force_errors = []
    stress_errors = []
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch)
            pred_energy = transforms.denormalize_prediction("energy", prediction["energy"], batch)
            target_energy = batch.energy.reshape_as(pred_energy)
            energy_errors.append((pred_energy - target_energy).detach().cpu())
            natoms = batch.natoms.to(pred_energy).reshape((-1,) + (1,) * (pred_energy.ndim - 1))
            energy_per_atom_errors.append(((pred_energy - target_energy) / natoms).detach().cpu())

            pred_forces = transforms.denormalize_prediction("forces", prediction["forces"], batch)
            target_forces = batch.forces.reshape_as(pred_forces)
            force_errors.append((pred_forces - target_forces).detach().cpu())
            if "stress" in prediction and hasattr(batch, "stress"):
                pred_stress = transforms.denormalize_prediction("stress", prediction["stress"], batch)
                target_stress = batch.stress.reshape_as(pred_stress)
                stress_errors.append((pred_stress - target_stress).detach().cpu())

    result = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "data": str(Path(args.data).expanduser()),
        "samples": len(loader.dataset),
        "energy_total_eV": _stats(torch.cat(energy_errors)),
        "energy_per_atom_eV": _stats(torch.cat(energy_per_atom_errors)),
        "forces_eV_per_A": _stats(torch.cat(force_errors)),
    }
    if stress_errors:
        result["stress_eV_per_A3"] = _stats(torch.cat(stress_errors))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
