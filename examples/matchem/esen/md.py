"""Run a short NVT molecular-dynamics trajectory with eSEN."""

from __future__ import annotations

import argparse
import os

import numpy as np
from ase import units
from ase.build import bulk
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from onescience.utils.esen import eSENCalculator


def default_checkpoint() -> str:
    return os.path.join(
        os.environ["ONESCIENCE_MODELS_DIR"], "eSEN", "esen_30m_mptrj.pt"
    )


def load_structure(path: str | None, repeat: tuple[int, int, int] | None):
    if path:
        atoms = read(path)
    else:
        atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
    return atoms.repeat(repeat) if repeat else atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--input", help="CIF, POSCAR, XYZ, or another ASE-readable structure")
    parser.add_argument(
        "--repeat",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="repeat the input structure along its three cell vectors",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep", type=float, default=1.0, help="time step in fs")
    parser.add_argument(
        "--friction", type=float, default=0.01, help="Langevin friction in 1/fs"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="md.traj")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repeat = tuple(args.repeat) if args.repeat else None
    atoms = load_structure(args.input, repeat)
    atoms.calc = eSENCalculator.from_checkpoint(args.checkpoint, device=args.device)
    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    dynamics = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        friction=args.friction / units.fs,
    )
    trajectory = Trajectory(args.output, "w", atoms)
    dynamics.attach(trajectory.write, interval=1)
    dynamics.run(args.steps)
    trajectory.close()
    print("formula:", atoms.get_chemical_formula())
    print("atoms:", len(atoms))
    print("steps:", dynamics.nsteps)
    print("energy (eV):", atoms.get_potential_energy())


if __name__ == "__main__":
    main()
