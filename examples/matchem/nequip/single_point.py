"""Run one NequIP energy, force, and stress prediction through ASE.

This script follows the official NequIP ASE integration style:
https://nequip.readthedocs.io/en/latest/integrations/ase.html
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")

from ase.build import bulk
from ase.io import read

from onescience.models.nequip.model import ModelTypeNamesFromPackage
from onescience.models.nequip.model.nequip_models import NequIPGNNModel
from onescience.utils.nequip.internal.global_state import set_global_state
from onescience.utils.nequip import build_nequip_calculator


def default_paths() -> Dict[str, str | None]:
    """Return default compiled model / checkpoint paths if env var is set."""
    models_dir = os.environ.get("ONESCIENCE_MODELS_DIR")
    if not models_dir:
        return {"compiled_model": None, "checkpoint": None}
    nequip_dir = Path(models_dir) / "NequIP"
    return {
        "compiled_model": str(nequip_dir / "NequIP-OAM-L-0.1.nequip.pth"),
        "checkpoint": None,
    }


def resolve_model_paths(
    compiled_model: str | None, checkpoint: str | None
) -> Dict[str, str | None]:
    """Prefer an explicitly selected model source over environment defaults."""
    if compiled_model or checkpoint:
        return {"compiled_model": compiled_model, "checkpoint": checkpoint}
    return default_paths()


def load_structure(path: str | None, index: int):
    """Load an ASE structure or use the built-in Cu bulk example."""
    if path:
        return read(path, index=index)
    return bulk("Cu")


def write_workflow_result(result: Dict[str, Any], output_path: str) -> str:
    """Write a workflow result dictionary to a JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compiled-model",
        help="Path to a compiled NequIP model (.nequip.pth or .nequip.pt2).",
    )
    group.add_argument(
        "--checkpoint",
        help="Path to a NequIP checkpoint (.ckpt) or packaged model (.nequip.zip).",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="Use a small built-in demo model instead of a real checkpoint.",
    )
    parser.add_argument(
        "--package",
        help=(
            "Original .nequip.zip package for a fine-tuned checkpoint; its atom "
            "types are read automatically."
        ),
    )
    parser.add_argument(
        "--input",
        help=(
            "CIF, POSCAR, XYZ, trajectory, or another ASE-readable structure; "
            "defaults to the built-in periodic Cu example"
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based frame index for trajectory inputs (default: 0).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/single_point.json")
    parser.add_argument(
        "--model-type-names",
        nargs="+",
        default=["C", "H", "O", "Cu"],
        help="Chemical species the model knows about (used for demo/checkpoint).",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=4.0,
        help="Neighbor-list cutoff in Angstrom (used for demo/checkpoint models).",
    )
    args = parser.parse_args()

    for label, path in (
        ("compiled model", args.compiled_model),
        ("checkpoint", args.checkpoint),
        ("package", args.package),
    ):
        if path and not Path(path).expanduser().is_file():
            parser.error(f"{label} not found: {path}")

    model_paths = resolve_model_paths(args.compiled_model, args.checkpoint)
    compiled_model = model_paths["compiled_model"]
    checkpoint = model_paths["checkpoint"]
    if args.package and not checkpoint:
        parser.error("--package requires --checkpoint")

    model_type_names = list(args.model_type_names)
    package_for_types = args.package
    if package_for_types is None and checkpoint and checkpoint.endswith(".nequip.zip"):
        package_for_types = checkpoint
    if package_for_types:
        model_type_names = list(ModelTypeNamesFromPackage(package_for_types))

    atoms = load_structure(args.input, args.index)
    calc_kwargs: Dict[str, Any] = {"device": args.device}

    if args.demo:
        set_global_state()
        calc_kwargs["model"] = NequIPGNNModel(
            seed=123,
            model_dtype="float32",
            type_names=model_type_names,
            num_layers=2,
            l_max=1,
            num_features=32,
            r_max=args.r_max,
            parity=False,
            avg_num_neighbors=10.0,
        )
    elif compiled_model and Path(compiled_model).exists():
        calc_kwargs["compiled_model"] = compiled_model
    elif checkpoint and Path(checkpoint).exists():
        calc_kwargs["checkpoint"] = checkpoint
        calc_kwargs["model_type_names"] = model_type_names
    else:
        parser.error(
            "no model found; pass --compiled-model, --checkpoint, or --demo"
        )

    atoms.calc = build_nequip_calculator(**calc_kwargs)

    result = {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "input": str(Path(args.input).expanduser()) if args.input else None,
        "input_index": args.index if args.input else None,
        "input_source": args.input or "ASE bulk Cu default",
        "compiled_model": str(Path(compiled_model).expanduser()) if compiled_model else None,
        "checkpoint": str(Path(checkpoint).expanduser()) if checkpoint else None,
        "package": str(Path(package_for_types).expanduser()) if package_for_types else None,
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
