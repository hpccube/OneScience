"""Relax an ASE structure with a compiled NequIP model.

This follows the official NequIP ASE relaxation example: it supports atomic
and cell relaxation, tracks forces at every ionic step, and aborts exploding
relaxations before they can hang indefinitely.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.io import read, write
import ase.optimize as opt

from onescience.utils.nequip.integrations.ase import NequIPCalculator


OPTIMIZERS = {
    "BFGS": opt.BFGS,
    "BFGSLineSearch": opt.BFGSLineSearch,
    "FIRE": opt.FIRE,
    "FIRE2": opt.FIRE2,
    "GOQN": opt.GoodOldQuasiNewton,
    "GPMin": opt.GPMin,
    "LBFGS": opt.LBFGS,
    "LBFGSLineSearch": opt.LBFGSLineSearch,
}
CELL_FILTERS = {
    "exp": ExpCellFilter,
    "frechet": FrechetCellFilter,
}


def default_compiled_model() -> str | None:
    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if not models_dir:
        return None
    return str(Path(models_dir) / "NequIP" / "NequIP-OAM-L-0.1.nequip.pth")


def load_structure(
    input_path: str | None,
    index: int,
    element: str,
    crystal_structure: str,
    lattice_constant: float,
    displacement: float,
) -> tuple[Atoms, str]:
    if input_path:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input structure not found: {path}")
        return read(path, index=index), f"{path}[{index}]"

    atoms = bulk(
        element,
        crystalstructure=crystal_structure,
        a=lattice_constant,
        cubic=True,
    )
    if displacement:
        atoms.positions[0, 0] += displacement
    return atoms, (
        f"ASE bulk {element} {crystal_structure}, a={lattice_constant} Angstrom, "
        f"atom-0 displacement={displacement} Angstrom"
    )


def _max_vector_norm(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.reshape(-1, 3), axis=1).max())


def relaxation_snapshot(atoms: Atoms, target: Any, step: int) -> dict[str, Any]:
    forces = atoms.get_forces()
    optimizer_forces = target.get_forces()
    stress = atoms.get_stress()
    return {
        "step": step,
        "energy_ev": float(atoms.get_potential_energy()),
        "energy_ev_per_atom": float(atoms.get_potential_energy() / len(atoms)),
        "volume_angstrom3": float(atoms.get_volume()),
        "max_atomic_force_ev_per_angstrom": _max_vector_norm(forces),
        "max_optimizer_force": _max_vector_norm(optimizer_forces),
        "stress_ev_per_angstrom3_voigt": np.asarray(stress).tolist(),
        "max_abs_stress_ev_per_angstrom3": float(np.abs(stress).max()),
    }


def relax_structure(
    atoms: Atoms,
    *,
    optimizer_name: str,
    cell_filter_name: str,
    fixed_cell: bool,
    fmax: float,
    steps: int,
    force_limit: float,
    logfile: Path,
    trajectory: Path,
) -> tuple[bool, int, list[dict[str, Any]]]:
    if not fixed_cell and not atoms.pbc.all():
        raise ValueError("cell relaxation requires periodic boundaries; use --fixed-cell")

    target = atoms if fixed_cell else CELL_FILTERS[cell_filter_name](atoms)
    optimizer_cls = OPTIMIZERS[optimizer_name]
    history: list[dict[str, Any]] = []
    converged = False

    with optimizer_cls(
        target,
        logfile=str(logfile),
        trajectory=str(trajectory),
    ) as optimizer:
        for converged in optimizer.irun(fmax=fmax, steps=steps):
            snapshot = relaxation_snapshot(atoms, target, optimizer.nsteps)
            history.append(snapshot)
            if max(
                snapshot["max_atomic_force_ev_per_angstrom"],
                snapshot["max_optimizer_force"],
            ) > force_limit:
                raise RuntimeError(
                    f"relaxation force exceeded safety limit {force_limit:g}"
                )

    return bool(converged), int(optimizer.nsteps), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-model", default=default_compiled_model())
    parser.add_argument(
        "--input",
        help="CIF, POSCAR, XYZ, trajectory, or another ASE-readable structure",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="GOQN")
    parser.add_argument(
        "--cell-filter", choices=sorted(CELL_FILTERS), default="frechet"
    )
    parser.add_argument(
        "--fixed-cell",
        action="store_true",
        help="relax atomic positions only; the default also relaxes the cell",
    )
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--force-limit", type=float, default=1.0e6)
    parser.add_argument("--element", default="Si")
    parser.add_argument("--crystal-structure", default="diamond")
    parser.add_argument("--lattice-constant", type=float, default=5.65)
    parser.add_argument("--displacement", type=float, default=0.08)
    parser.add_argument("--output-dir", default="outputs/structure_relaxation")
    parser.add_argument("--output-structure", default="relaxed.cif")
    parser.add_argument("--result", default="result.json")
    parser.add_argument("--trajectory", default="relax.traj")
    parser.add_argument("--log", default="relax.log")
    args = parser.parse_args()

    if not args.compiled_model:
        parser.error("--compiled-model is required when ONESCIENCE_MODELS_DIR is unset")
    compiled_model = Path(args.compiled_model).expanduser().resolve()
    if not compiled_model.is_file():
        parser.error(f"compiled model not found: {compiled_model}")
    if args.fmax <= 0:
        parser.error("--fmax must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.force_limit <= 0:
        parser.error("--force-limit must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_structure = output_dir / args.output_structure
    result_path = output_dir / args.result
    trajectory_path = output_dir / args.trajectory
    log_path = output_dir / args.log

    try:
        atoms, input_source = load_structure(
            args.input,
            args.index,
            args.element,
            args.crystal_structure,
            args.lattice_constant,
            args.displacement,
        )
    except (FileNotFoundError, IndexError, ValueError) as error:
        parser.error(str(error))
    if len(atoms) == 0:
        parser.error("input structure has no atoms")

    species = sorted(set(atoms.get_chemical_symbols()))
    atoms.calc = NequIPCalculator.from_compiled_model(
        compile_path=str(compiled_model),
        chemical_species_to_atom_type_map={symbol: symbol for symbol in species},
        device=args.device,
    )

    try:
        converged, nsteps, history = relax_structure(
            atoms,
            optimizer_name=args.optimizer,
            cell_filter_name=args.cell_filter,
            fixed_cell=args.fixed_cell,
            fmax=args.fmax,
            steps=args.steps,
            force_limit=args.force_limit,
            logfile=log_path,
            trajectory=trajectory_path,
        )
    except ValueError as error:
        parser.error(str(error))

    write(output_structure, atoms)
    result = {
        "compiled_model": str(compiled_model),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu",
        "input_source": input_source,
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "chemical_species_to_atom_type_map": {
            symbol: symbol for symbol in species
        },
        "optimizer": args.optimizer,
        "cell_filter": None if args.fixed_cell else args.cell_filter,
        "fixed_cell": args.fixed_cell,
        "fmax_ev_per_angstrom": args.fmax,
        "max_steps": args.steps,
        "force_safety_limit": args.force_limit,
        "converged": converged,
        "steps": nsteps,
        "initial": history[0],
        "final": history[-1],
        "energy_change_ev": history[-1]["energy_ev"] - history[0]["energy_ev"],
        "volume_change_angstrom3": (
            history[-1]["volume_angstrom3"] - history[0]["volume_angstrom3"]
        ),
        "history": history,
        "relaxed_structure": str(output_structure),
        "trajectory": str(trajectory_path),
        "log": str(log_path),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("formula:", result["formula"])
    print("atoms:", result["num_atoms"])
    print("converged:", converged)
    print("steps:", nsteps)
    print("initial energy (eV):", result["initial"]["energy_ev"])
    print("final energy (eV):", result["final"]["energy_ev"])
    print(
        "final max force (eV/Angstrom):",
        result["final"]["max_atomic_force_ev_per_angstrom"],
    )
    print("result:", result_path)


if __name__ == "__main__":
    main()
