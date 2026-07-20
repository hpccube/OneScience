import argparse

import torch
from ase.build import bulk
from ase.units import GPa

from onescience.models.mattersim import load_calculator


def main() -> None:
    parser = argparse.ArgumentParser(description="MatterSim single-point inference")
    parser.add_argument(
        "--checkpoint",
        help="Path to MatterSim checkpoint. Defaults to $ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.calc = load_calculator(checkpoint=args.checkpoint, device=device)

    energy = atoms.get_potential_energy()
    print(f"Energy (eV)                 = {energy}")
    print(f"Energy per atom (eV/atom)   = {energy / len(atoms)}")
    print(f"Forces of first atom (eV/A) = {atoms.get_forces()[0]}")
    stress = atoms.get_stress(voigt=False)
    print(f"Stress[0][0] (eV/A^3)       = {stress[0][0]}")
    print(f"Stress[0][0] (GPa)          = {stress[0][0] / GPa}")


if __name__ == "__main__":
    main()
