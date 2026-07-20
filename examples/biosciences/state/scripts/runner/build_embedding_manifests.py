"""Build deterministic train/validation CSV manifests for State Embedding."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing h5ad files")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv and val.csv")
    parser.add_argument("--species", default="human", help="Species value written to the manifests")
    parser.add_argument("--val-count", type=int, default=1, help="Number of sorted files assigned to validation")
    return parser.parse_args()


def write_manifest(path: Path, files: list[Path], species: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "path", "names"])
        writer.writeheader()
        for dataset_path in files:
            writer.writerow(
                {
                    "species": species,
                    "path": str(dataset_path.resolve()),
                    "names": dataset_path.stem,
                }
            )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.h5ad"))
    if len(files) < 2:
        raise ValueError(f"Expected at least two h5ad files in {data_dir}, found {len(files)}")
    if args.val_count < 1 or args.val_count >= len(files):
        raise ValueError("--val-count must be at least 1 and smaller than the number of h5ad files")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "train.csv", files[: -args.val_count], args.species)
    write_manifest(output_dir / "val.csv", files[-args.val_count :], args.species)


if __name__ == "__main__":
    main()
