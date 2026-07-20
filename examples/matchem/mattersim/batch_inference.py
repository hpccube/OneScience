import argparse

from ase.build import bulk

from onescience.models.mattersim import predict_structures


def main() -> None:
    parser = argparse.ArgumentParser(description="MatterSim batch inference")
    parser.add_argument(
        "--checkpoint",
        help="Path to MatterSim checkpoint. Defaults to $ONESCIENCE_MODELS_DIR/mattersim/mattersim-v1.0.0-1M.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    structures = [bulk("Si", "diamond", a=a) for a in (5.40, 5.43, 5.46)]
    result = predict_structures(
        structures,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    for index, (energy, forces) in enumerate(
        zip(result["energies"], result["forces"], strict=True)
    ):
        print(
            f"Structure {index}: energy={energy:.6f} eV, "
            f"max_force={abs(forces).max():.6e} eV/angstrom"
        )


if __name__ == "__main__":
    main()
