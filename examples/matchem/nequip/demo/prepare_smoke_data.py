"""Generate a tiny extxyz smoke dataset for NequIP demo training."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write


def main() -> None:
    out_dir = Path(__file__).parent / "reference_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "smoke.xyz"

    rng = np.random.default_rng(123)
    structures = []
    # A few small Cu clusters with random displacements.
    base = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    for i in range(8):
        atoms = base.copy()
        atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
        atoms.info["energy"] = float(-len(atoms) * 3.5 + rng.normal(scale=0.5))
        atoms.arrays["forces"] = rng.normal(scale=0.1, size=atoms.positions.shape)
        structures.append(atoms)

    write(out_file, structures, format="extxyz")
    print(f"Wrote {len(structures)} structures to {out_file}")


if __name__ == "__main__":
    main()
