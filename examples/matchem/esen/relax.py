"""Relax a periodic structure with an eSEN ASE calculator."""

from __future__ import annotations

import argparse
import os

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS

from onescience.utils.esen import eSENCalculator


def default_checkpoint() -> str:
    return os.path.join(
        os.environ["ONESCIENCE_MODELS_DIR"], "eSEN", "esen_30m_mptrj.pt"
    )


def load_structure(path: str | None):
    if path:
        return read(path)
    return bulk("Si", "diamond", a=5.50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--input", help="CIF, POSCAR, XYZ, or another ASE-readable structure")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="relaxed.cif")
    parser.add_argument(
        "--fixed-cell",
        action="store_true",
        help="relax atomic positions only; by default the cell is relaxed too",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    atoms = load_structure(args.input)
    if not args.fixed_cell and not atoms.pbc.all():
        parser.error("cell relaxation requires periodic boundaries; use --fixed-cell")
    atoms.calc = eSENCalculator.from_checkpoint(args.checkpoint, device=args.device)
    target = atoms if args.fixed_cell else FrechetCellFilter(atoms)
    optimizer = BFGS(target, logfile="relax.log", trajectory="relax.traj")
    optimizer.run(fmax=args.fmax, steps=args.steps)
    write(args.output, atoms)
    print("formula:", atoms.get_chemical_formula())
    print("atoms:", len(atoms))
    print("steps:", optimizer.nsteps)
    print("energy (eV):", atoms.get_potential_energy())
    print("maximum force (eV/Angstrom):", max((atoms.get_forces() ** 2).sum(1) ** 0.5))


if __name__ == "__main__":
    main()
