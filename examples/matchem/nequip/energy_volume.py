"""Compute the ASE energy-volume curve from the official NequIP example."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

from onescience.utils.nequip.integrations.ase import NequIPCalculator


def default_compiled_model() -> str | None:
    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if not models_dir:
        return None
    return str(Path(models_dir) / "NequIP" / "NequIP-OAM-L-0.1.nequip.pth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-model", default=default_compiled_model())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--element", default="Si")
    parser.add_argument("--crystal-structure", default="diamond")
    parser.add_argument("--lattice-constant", type=float, default=5.43)
    parser.add_argument("--supercell", type=int, default=3)
    parser.add_argument("--scale-min", type=float, default=0.95)
    parser.add_argument("--scale-max", type=float, default=1.05)
    parser.add_argument("--num-points", type=int, default=10)
    parser.add_argument("--output", default="outputs/energy_volume.json")
    parser.add_argument("--plot", default="outputs/energy_volume.png")
    args = parser.parse_args()

    if not args.compiled_model:
        parser.error("--compiled-model is required when ONESCIENCE_MODELS_DIR is unset")
    compiled_model = Path(args.compiled_model).expanduser().resolve()
    if not compiled_model.is_file():
        parser.error(f"compiled model not found: {compiled_model}")
    if args.num_points < 2:
        parser.error("--num-points must be at least 2")
    if args.supercell < 1:
        parser.error("--supercell must be positive")

    calculator = NequIPCalculator.from_compiled_model(
        compile_path=str(compiled_model),
        chemical_species_to_atom_type_map={args.element: args.element},
        device=args.device,
    )

    points = []
    for scale in np.linspace(args.scale_min, args.scale_max, args.num_points):
        atoms = bulk(
            args.element,
            crystalstructure=args.crystal_structure,
            a=args.lattice_constant * float(scale),
            cubic=True,
        )
        atoms *= (args.supercell,) * 3
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        points.append(
            {
                "scale": float(scale),
                "volume_angstrom3": float(atoms.get_volume()),
                "energy_ev": energy,
                "energy_ev_per_atom": energy / len(atoms),
                "max_force_ev_per_angstrom": float(
                    np.linalg.norm(forces, axis=1).max()
                ),
            }
        )

    energies = np.asarray([point["energy_ev"] for point in points])
    volumes = np.asarray([point["volume_angstrom3"] for point in points])
    minimum_index = int(np.argmin(energies))
    result = {
        "compiled_model": str(compiled_model),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu",
        "element": args.element,
        "crystal_structure": args.crystal_structure,
        "base_lattice_constant_angstrom": args.lattice_constant,
        "supercell": [args.supercell] * 3,
        "num_atoms": len(atoms),
        "points": points,
        "sampled_minimum": points[minimum_index],
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = Path(args.plot).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(volumes, energies, marker="o", label="E-V Curve")
        plt.xlabel("Volume (Angstrom^3)", fontsize=14)
        plt.ylabel("Energy (eV)", fontsize=14)
        plt.title(f"Energy-Volume Curve for Cubic {args.element}", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        result["plot"] = str(plot_path)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"points: {len(points)}")
    print(f"atoms per point: {result['num_atoms']}")
    print(f"sampled minimum: {result['sampled_minimum']}")
    print(f"result: {output_path}")


if __name__ == "__main__":
    main()
