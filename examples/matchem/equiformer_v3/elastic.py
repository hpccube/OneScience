"""Fit an elastic tensor from Equiformer V3 stress predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ase.build import bulk
from ase.io import read, write

from onescience.utils.equiformer_v3 import (
    EquiformerV3Calculator,
    relax_structure,
    run_elastic_workflow,
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
    return bulk("Cu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--input",
        help="periodic structure readable by ASE; defaults to built-in bulk Cu",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="relax the cell and atomic positions before applying strains",
    )
    parser.add_argument(
        "--normal-strains", type=float, nargs="+", default=(-0.01, 0.01)
    )
    parser.add_argument(
        "--shear-strains", type=float, nargs="+", default=(-0.02, 0.02)
    )
    parser.add_argument(
        "--relax-positions",
        action="store_true",
        help="relax atomic positions at fixed cell for every deformation",
    )
    parser.add_argument("--relax-fmax", type=float, default=0.02)
    parser.add_argument("--relax-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/elastic.json")
    args = parser.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    calculator = EquiformerV3Calculator.from_checkpoint(
        checkpoint, device=args.device
    )
    structure = load_structure(args.input)
    initial_relaxation = None
    if args.relax:
        structure, initial_relaxation = relax_structure(
            structure,
            calculator,
            relax_cell=True,
            fmax=args.relax_fmax,
            steps=args.relax_steps,
        )
        relaxed_output = Path(args.output).with_name("elastic_relaxed.cif")
        relaxed_output.parent.mkdir(parents=True, exist_ok=True)
        write(relaxed_output, structure)
    result = run_elastic_workflow(
        structure,
        calculator,
        normal_strains=args.normal_strains,
        shear_strains=args.shear_strains,
        relax_positions=args.relax_positions,
        relax_fmax=args.relax_fmax,
        relax_steps=args.relax_steps,
    )
    result["checkpoint"] = str(checkpoint)
    result["initial_relaxation"] = initial_relaxation
    output = write_workflow_result(result, args.output)

    print("formula:", result["formula"])
    print("bulk modulus, Hill (GPa):", result["bulk_modulus_gpa"]["hill"])
    print("result:", output)


if __name__ == "__main__":
    main()
