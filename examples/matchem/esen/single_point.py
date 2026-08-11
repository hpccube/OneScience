"""Run one eSEN single-point calculation through ASE."""

from __future__ import annotations

import argparse
import os

from ase.build import bulk
from ase.io import read

from onescience.utils.esen import eSENCalculator


def default_checkpoint() -> str:
    return os.path.join(
        os.environ["ONESCIENCE_MODELS_DIR"], "eSEN", "esen_30m_mptrj.pt"
    )


def load_structure(path: str | None):
    if path:
        return read(path)
    return bulk("Si", "diamond", a=5.43)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--input", help="CIF, POSCAR, XYZ, or another ASE-readable structure")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    atoms = load_structure(args.input)
    atoms.calc = eSENCalculator.from_checkpoint(args.checkpoint, device=args.device)

    print("formula:", atoms.get_chemical_formula())
    print("atoms:", len(atoms))
    print("energy (eV):", atoms.get_potential_energy())
    print("forces (eV/Angstrom):\n", atoms.get_forces())
    print("stress (eV/Angstrom^3):", atoms.get_stress())


if __name__ == "__main__":
    main()
