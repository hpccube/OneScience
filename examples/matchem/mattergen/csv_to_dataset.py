"""Convert MatterGen CSV splits into cached CrystalDataset directories."""

import argparse
from pathlib import Path

from onescience.datapipes.materials.mattergen import CrystalDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert every CSV file in a directory to MatterGen cache format."
    )
    parser.add_argument(
        "--csv-folder",
        type=Path,
        required=True,
        help="Directory containing split files such as train.csv and val.csv.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset directory name created below --cache-folder.",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        required=True,
        help="Parent directory in which the dataset cache is created.",
    )
    args = parser.parse_args()

    if not args.csv_folder.is_dir():
        parser.error(f"CSV directory does not exist: {args.csv_folder}")

    csv_files = sorted(args.csv_folder.glob("*.csv"))
    if not csv_files:
        parser.error(f"No CSV files found in: {args.csv_folder}")

    dataset_root = args.cache_folder / args.dataset_name
    for csv_path in csv_files:
        cache_path = dataset_root / csv_path.stem
        print(f"Processing {csv_path} -> {cache_path}")
        CrystalDataset.from_csv(csv_path=csv_path, cache_path=cache_path)


if __name__ == "__main__":
    main()
