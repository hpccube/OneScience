"""Run one Equiformer V3 energy, force, and stress prediction through ASE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ase.build import bulk
from ase.io import read

from onescience.utils.equiformer_v3 import (
    EquiformerV3Calculator,
    write_workflow_result,
)


def default_checkpoint() -> str:
    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if not models_dir:
        raise RuntimeError(
            "Set ONESCIENCE_MODELS_DIR or pass --checkpoint explicitly"
        )
    return os.path.join(
        models_dir,
        "EquiformerV3",
        "omat24-mptrj-salex_gradient.pt",
    )


def load_structure(path: str | None):
    if path:
        return read(path)
    # Keep the example runnable without requiring a separate structure file.
    return bulk("Cu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--input",
        help=(
            "CIF, POSCAR, XYZ, trajectory, or another ASE-readable structure; "
            "defaults to the built-in periodic Cu example"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/single_point.json")
    args = parser.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    atoms = load_structure(args.input)
    atoms.calc = EquiformerV3Calculator.from_checkpoint(
        checkpoint,
        device=args.device,
    )

    result = {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "input": str(Path(args.input).expanduser()) if args.input else None,
        "input_source": args.input or "ASE bulk Cu default",
        "checkpoint": str(Path(checkpoint).expanduser()),
        "pbc": atoms.pbc.tolist(),
        "cell_angstrom": atoms.cell.array.tolist(),
        "energy_ev": float(atoms.get_potential_energy()),
        "forces_ev_per_angstrom": atoms.get_forces().tolist(),
        "stress_ev_per_angstrom_cubed_voigt": atoms.get_stress().tolist(),
    }
    output = write_workflow_result(result, args.output)

    print("formula:", result["formula"])
    print("atoms:", result["natoms"])
    print("energy (eV):", result["energy_ev"])
    print("result:", output)


if __name__ == "__main__":
    main()
