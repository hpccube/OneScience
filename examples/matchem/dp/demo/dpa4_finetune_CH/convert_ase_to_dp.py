#!/usr/bin/env python3
"""Convert H/C ASE databases to DeepMD NumPy systems for this example.

The normal fine-tuning run uses the prepared OneScience dataset. This utility
is only needed when rebuilding that dataset from ASE database files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.db import connect


# These are local data type indices: H -> 0 and C -> 1. During fine-tuning,
# DeepMD maps these element names to the full 118-element DPA4 model type map.
TYPE_MAP = ["H", "C"]
MAX_CUTOFF = 6.0
BOX_MARGIN = 2 * MAX_CUTOFF + 4.0


def make_cubic_box(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a non-periodic cubic box and a shift that centers the molecule."""
    min_pos = coords.min(axis=0)
    max_pos = coords.max(axis=0)
    extent = (max_pos - min_pos).max()
    box_length = extent + BOX_MARGIN
    shift = box_length / 2.0 - (min_pos + max_pos) / 2.0
    return np.eye(3) * box_length, shift


def convert_database(input_file: Path, output_base: Path) -> None:
    frames_by_natoms: dict[int, list[dict[str, np.ndarray | float]]] = {}
    for row in connect(input_file).select():
        atoms = row.toatoms()
        symbols = atoms.get_chemical_symbols()
        unsupported = sorted(set(symbols) - set(TYPE_MAP))
        if unsupported:
            raise ValueError(
                f"{input_file} contains unsupported elements: {unsupported}"
            )

        coords = atoms.get_positions()
        box, shift = make_cubic_box(coords)
        frame = {
            "coords": coords + shift,
            "types": np.asarray([TYPE_MAP.index(symbol) for symbol in symbols]),
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "box": box,
        }
        frames_by_natoms.setdefault(len(atoms), []).append(frame)

    frame_count = 0
    for natoms, frames in sorted(frames_by_natoms.items()):
        system_dir = output_base / f"sys_{natoms}"
        set_dir = system_dir / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        reference_types = np.asarray(frames[0]["types"], dtype=int)
        if any(
            not np.array_equal(reference_types, np.asarray(frame["types"], dtype=int))
            for frame in frames[1:]
        ):
            raise ValueError(
                f"{input_file}: frames with {natoms} atoms do not share one "
                "type ordering and must be split into separate DeepMD systems"
            )

        np.save(
            set_dir / "coord.npy",
            np.stack([f["coords"] for f in frames]).reshape(len(frames), -1),
        )
        np.save(
            set_dir / "force.npy",
            np.stack([f["forces"] for f in frames]).reshape(len(frames), -1),
        )
        np.save(set_dir / "energy.npy", np.asarray([f["energy"] for f in frames]))
        np.save(
            set_dir / "box.npy",
            np.stack([f["box"] for f in frames]).reshape(len(frames), 9),
        )

        (system_dir / "type.raw").write_text(
            " ".join(map(str, reference_types)) + "\n", encoding="utf-8"
        )
        (system_dir / "type_map.raw").write_text("H\nC\n", encoding="utf-8")

        frame_count += len(frames)
        print(f"{system_dir}: {len(frames)} frames, {natoms} atoms")

    print(
        f"Converted {input_file} to {output_base}: "
        f"{frame_count} frames in {len(frames_by_natoms)} systems"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-db", type=Path, required=True)
    parser.add_argument("--validation-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_database(args.train_db, args.output_root / "train_CH")
    convert_database(args.validation_db, args.output_root / "val_CH")


if __name__ == "__main__":
    main()
