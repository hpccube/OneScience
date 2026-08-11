"""Build reproducible ASE DB splits from FairChem's oxide tutorial data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect


def load_records(path: Path):
    with path.open() as handle:
        source = json.load(handle)
    records = []
    for oxide, polymorphs in source.items():
        for polymorph, calculations in polymorphs.items():
            for calculation in calculations.get("PBE", {}).get("EOS", {}).get(
                "calculations", []
            ):
                records.append((oxide, polymorph, calculation))
    return records


def split_groups(records, seed: int):
    groups = sorted({(oxide, polymorph) for oxide, polymorph, _ in records})
    random.Random(seed).shuffle(groups)
    n_train = int(0.8 * len(groups))
    n_val = int(0.1 * len(groups))
    split_for = {group: "train" for group in groups[:n_train]}
    split_for.update({group: "val" for group in groups[n_train : n_train + n_val]})
    split_for.update({group: "test" for group in groups[n_train + n_val :]})
    return split_for


def to_atoms(oxide: str, polymorph: str, calculation: dict) -> Atoms:
    structure = calculation["atoms"]
    results = calculation["data"]
    atoms = Atoms(
        symbols=structure["symbols"],
        positions=structure["positions"],
        cell=structure["cell"],
        pbc=structure["pbc"],
    )
    atoms.set_tags(np.ones(len(atoms), dtype=int))
    # ASE stress uses eV/Angstrom^3. The source JSON documents stress in GPa.
    stress = np.asarray(results["stress"], dtype=float) * units.GPa
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(results["total_energy"]),
        forces=np.asarray(results["forces"], dtype=float),
        stress=stress,
    )
    atoms.info.update({"oxide": oxide, "polymorph": polymorph, "xc": "PBE"})
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing split databases"
    )
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise ValueError(f"No PBE EOS structures found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {split: args.output / f"{split}.db" for split in ("train", "val", "test")}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output exists; pass --overwrite to replace: " + ", ".join(existing))
    for path in paths.values():
        if path.exists():
            path.unlink()

    split_for = split_groups(records, args.seed)
    counts = Counter()
    databases = {split: connect(path) for split, path in paths.items()}
    for oxide, polymorph, calculation in records:
        split = split_for[(oxide, polymorph)]
        databases[split].write(
            to_atoms(oxide, polymorph, calculation),
            oxide=oxide,
            polymorph=polymorph,
            xc="PBE",
        )
        counts[split] += 1

    manifest = {
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "split_strategy": "oxide-polymorph grouped 80/10/10",
        "stress_source_unit": "GPa",
        "stress_output_unit": "eV/Angstrom^3",
        "counts": dict(counts),
        "groups": {f"{oxide}/{polymorph}": split for (oxide, polymorph), split in sorted(split_for.items())},
    }
    with (args.output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"saved dataset: {args.output.resolve()}")


if __name__ == "__main__":
    main()
