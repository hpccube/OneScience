"""Run finite-displacement phonons with Equiformer V3 and ASE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ase.build import bulk
from ase.io import read, write

from onescience.utils.equiformer_v3 import (
    EquiformerV3Calculator,
    relax_structure,
    run_phonon_workflow,
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
        help="relax the cell and atomic positions before finite displacements",
    )
    parser.add_argument("--relax-fmax", type=float, default=0.01)
    parser.add_argument("--relax-steps", type=int, default=200)
    parser.add_argument("--supercell", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--bandpath", help="ASE band path, for example GXWKGL")
    parser.add_argument("--band-points", type=int, default=100)
    parser.add_argument("--dos-kpts", type=int, nargs=3, default=(10, 10, 10))
    parser.add_argument("--dos-points", type=int, default=400)
    parser.add_argument("--dos-width", type=float, default=0.001, help="eV")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/phonons")
    args = parser.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    calculator = EquiformerV3Calculator.from_checkpoint(
        checkpoint, device=args.device
    )
    output_dir = Path(args.output_dir)
    structure = load_structure(args.input)
    relaxation = None
    if args.relax:
        structure, relaxation = relax_structure(
            structure,
            calculator,
            relax_cell=True,
            fmax=args.relax_fmax,
            steps=args.relax_steps,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write(output_dir / "relaxed.cif", structure)
    result = run_phonon_workflow(
        structure,
        calculator,
        output_dir,
        supercell=args.supercell,
        delta=args.delta,
        bandpath=args.bandpath,
        band_points=args.band_points,
        dos_kpts=args.dos_kpts,
        dos_points=args.dos_points,
        dos_width_ev=args.dos_width,
    )
    result["checkpoint"] = str(checkpoint)
    result["relaxation"] = relaxation
    output = write_workflow_result(result, output_dir / "phonons.json")

    print("formula:", result["formula"])
    print("minimum phonon energy (eV):", result["minimum_band_energy_ev"])
    print("result:", output)


if __name__ == "__main__":
    main()
