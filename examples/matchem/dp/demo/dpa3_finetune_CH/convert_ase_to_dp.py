#!/usr/bin/env python3
"""Convert ASE LMDB database to DeepMD-kit numpy format, grouped by natoms.
Non-periodic molecules are placed in a cubic box large enough to avoid
self-images across the cutoff."""

import os
import numpy as np
from ase.db import connect

TYPE_MAP = ["H", "C"]
# Largest cutoff in the DPA-3 repflow config; add margin to avoid self-images
MAX_CUTOFF = 6.0
BOX_MARGIN = 2 * MAX_CUTOFF + 4.0  # ~16 Å margin on each side

def make_cubic_box(coords):
    """Build a cubic box that contains the molecule with enough margin."""
    min_pos = coords.min(axis=0)
    max_pos = coords.max(axis=0)
    extent = (max_pos - min_pos).max()
    box_len = extent + BOX_MARGIN
    # Place molecule centered in the box
    shift = box_len / 2.0 - (min_pos + max_pos) / 2.0
    return np.eye(3) * box_len, shift

def convert_aselmdb_to_deepmd(input_file, output_base):
    db = connect(input_file)
    frames_by_natoms = {}
    for row in db.select():
        atoms = row.toatoms()
        natoms = len(atoms)
        atoms_type = [TYPE_MAP.index(sym) for sym in atoms.get_chemical_symbols()]
        coords = atoms.get_positions()
        box, shift = make_cubic_box(coords)
        coords = coords + shift
        frames_by_natoms.setdefault(natoms, []).append({
            "coords": coords,
            "types": atoms_type,
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "box": box,
        })

    system_dirs = []
    for natoms, frames in sorted(frames_by_natoms.items()):
        nframes = len(frames)
        system_dir = os.path.join(output_base, f"sys_{natoms}")
        set_dir = os.path.join(system_dir, "set.000")
        os.makedirs(set_dir, exist_ok=True)

        coord = np.stack([f["coords"] for f in frames]).reshape(nframes, -1)
        force = np.stack([f["forces"] for f in frames]).reshape(nframes, -1)
        energy = np.array([f["energy"] for f in frames])
        box = np.stack([f["box"] for f in frames])
        types = np.array(frames[0]["types"])

        np.save(os.path.join(set_dir, "coord.npy"), coord)
        np.save(os.path.join(set_dir, "force.npy"), force)
        np.save(os.path.join(set_dir, "energy.npy"), energy)
        np.save(os.path.join(set_dir, "box.npy"), box)
        with open(os.path.join(system_dir, "type.raw"), "w") as f:
            f.write(" ".join(map(str, types)) + "\n")

        system_dirs.append(system_dir)
        print(f"  {system_dir}: {nframes} frames, {natoms} atoms")

    print(f"Converted {input_file} -> {output_base}: {sum(len(v) for v in frames_by_natoms.values())} frames in {len(frames_by_natoms)} systems")
    return system_dirs

if __name__ == "__main__":
    convert_aselmdb_to_deepmd("my_data_CH_3787_train.aselmdb", "train_CH")
    convert_aselmdb_to_deepmd("my_data_CH_3787_val.aselmdb", "val_CH")
