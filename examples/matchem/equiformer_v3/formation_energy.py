"""Calculate an uncorrected formation energy with Equiformer V3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ase.build import bulk, molecule
from ase.io import read

from onescience.utils.equiformer_v3 import (
    EquiformerV3Calculator,
    calculate_element_reference_energies,
    calculate_formation_energy,
    load_element_reference_energies,
    relax_structure,
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


def load_compound(path: str | None):
    if path:
        return read(path)
    return bulk("MgO", "rocksalt", a=4.21)


def model_reference_structures():
    oxygen = molecule("O2")
    oxygen.center(vacuum=8.0)
    oxygen.pbc = True
    return {
        "Mg": bulk("Mg", "hcp", a=3.21, c=5.21),
        "O": oxygen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--input",
        help="compound structure readable by ASE; defaults to built-in MgO",
    )
    parser.add_argument(
        "--reference-energies",
        help="JSON/YAML mapping of element to trusted reference energy in eV/atom",
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="relax the compound and model-evaluated reference phases",
    )
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/formation_energy.json")
    args = parser.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    calculator = EquiformerV3Calculator.from_checkpoint(
        checkpoint, device=args.device
    )
    compound = load_compound(args.input)
    compound_relaxation = None
    if args.relax:
        compound, compound_relaxation = relax_structure(
            compound,
            calculator,
            relax_cell=bool(compound.pbc.all()),
            fmax=args.fmax,
            steps=args.steps,
        )

    reference_relaxations = {}
    if args.reference_energies:
        reference_energies = load_element_reference_energies(
            args.reference_energies
        )
        reference_source = str(Path(args.reference_energies))
    else:
        reference_structures = model_reference_structures()
        if args.relax:
            relaxed_references = {}
            for element, atoms in reference_structures.items():
                relaxed, metadata = relax_structure(
                    atoms,
                    calculator,
                    relax_cell=element != "O",
                    fmax=args.fmax,
                    steps=args.steps,
                )
                relaxed_references[element] = relaxed
                reference_relaxations[element] = metadata
            reference_structures = relaxed_references
        reference_energies = calculate_element_reference_energies(
            reference_structures, calculator
        )
        reference_source = "Equiformer V3 evaluation of Mg(hcp) and O2"

    result = calculate_formation_energy(
        compound,
        calculator,
        reference_energies,
    )
    result["reference_source"] = reference_source
    result["checkpoint"] = str(checkpoint)
    result["compound_relaxation"] = compound_relaxation
    result["reference_relaxations"] = reference_relaxations
    output = write_workflow_result(result, args.output)

    print("formula:", result["formula"])
    print("formation energy (eV/atom):", result["formation_energy_ev_per_atom"])
    print("corrections applied:", result["corrections_applied"])
    print("result:", output)


if __name__ == "__main__":
    main()
