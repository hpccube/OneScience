"""Physical geometry metrics for Boltz predictions."""

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from onescience.datapipes.boltz import const
from onescience.datapipes.boltz.parse.mmcif_with_constraints import parse_mmcif


ccd = None
moldir = None
prediction_dirs = {}
prediction_targets = {}


def compute_torsion_angles(coords, torsion_index):
    r_ij = coords[..., torsion_index[0], :] - coords[..., torsion_index[1], :]
    r_kj = coords[..., torsion_index[2], :] - coords[..., torsion_index[1], :]
    r_kl = coords[..., torsion_index[2], :] - coords[..., torsion_index[3], :]
    n_ijk = np.cross(r_ij, r_kj, axis=-1)
    n_jkl = np.cross(r_kj, r_kl, axis=-1)
    r_kj_norm = np.linalg.norm(r_kj, axis=-1)
    n_ijk_norm = np.linalg.norm(n_ijk, axis=-1)
    n_jkl_norm = np.linalg.norm(n_jkl, axis=-1)
    sign_phi = np.sign(
        r_kj[..., None, :] @ np.cross(n_ijk, n_jkl, axis=-1)[..., None]
    ).squeeze(axis=(-1, -2))
    phi = sign_phi * np.arccos(
        np.clip(
            (n_ijk[..., None, :] @ n_jkl[..., None]).squeeze(axis=(-1, -2))
            / (n_ijk_norm * n_jkl_norm),
            -1 + 1e-8,
            1 - 1e-8,
        )
    )
    return phi


def check_ligand_distance_geometry(
    structure, constraints, bond_buffer=0.25, angle_buffer=0.25, clash_buffer=0.2
):
    coords = structure.coords["coords"]
    rdkit_bounds_constraints = constraints.rdkit_bounds_constraints
    pair_index = rdkit_bounds_constraints["atom_idxs"].copy().astype(np.int64).T
    bond_mask = rdkit_bounds_constraints["is_bond"].copy().astype(bool)
    angle_mask = rdkit_bounds_constraints["is_angle"].copy().astype(bool)
    upper_bounds = rdkit_bounds_constraints["upper_bound"].copy().astype(np.float32)
    lower_bounds = rdkit_bounds_constraints["lower_bound"].copy().astype(np.float32)
    dists = np.linalg.norm(coords[pair_index[0]] - coords[pair_index[1]], axis=-1)
    bond_length_violations = (
        dists[bond_mask] <= lower_bounds[bond_mask] * (1.0 - bond_buffer)
    ) + (dists[bond_mask] >= upper_bounds[bond_mask] * (1.0 + bond_buffer))
    bond_angle_violations = (
        dists[angle_mask] <= lower_bounds[angle_mask] * (1.0 - angle_buffer)
    ) + (dists[angle_mask] >= upper_bounds[angle_mask] * (1.0 + angle_buffer))
    internal_clash_violations = dists[~bond_mask * ~angle_mask] <= lower_bounds[
        ~bond_mask * ~angle_mask
    ] * (1.0 - clash_buffer)
    num_ligands = sum(
        [
            int(const.chain_types[chain["mol_type"]] == "NONPOLYMER")
            for chain in structure.chains
        ]
    )
    return {
        "num_ligands": num_ligands,
        "num_bond_length_violations": bond_length_violations.sum(),
        "num_bonds": bond_mask.sum(),
        "num_bond_angle_violations": bond_angle_violations.sum(),
        "num_angles": angle_mask.sum(),
        "num_internal_clash_violations": internal_clash_violations.sum(),
        "num_non_neighbors": (~bond_mask * ~angle_mask).sum(),
    }


def check_ligand_stereochemistry(structure, constraints):
    coords = structure.coords["coords"]
    chiral_atom_constraints = constraints.chiral_atom_constraints
    stereo_bond_constraints = constraints.stereo_bond_constraints

    chiral_atom_index = chiral_atom_constraints["atom_idxs"].T
    true_chiral_atom_orientations = chiral_atom_constraints["is_r"]
    chiral_atom_ref_mask = chiral_atom_constraints["is_reference"]
    chiral_atom_index = chiral_atom_index[:, chiral_atom_ref_mask]
    true_chiral_atom_orientations = true_chiral_atom_orientations[chiral_atom_ref_mask]
    pred_chiral_atom_orientations = (
        compute_torsion_angles(coords, chiral_atom_index) > 0
    )
    chiral_atom_violations = (
        pred_chiral_atom_orientations != true_chiral_atom_orientations
    )

    stereo_bond_index = stereo_bond_constraints["atom_idxs"].T
    true_stereo_bond_orientations = stereo_bond_constraints["is_e"]
    stereo_bond_ref_mask = stereo_bond_constraints["is_reference"]
    stereo_bond_index = stereo_bond_index[:, stereo_bond_ref_mask]
    true_stereo_bond_orientations = true_stereo_bond_orientations[stereo_bond_ref_mask]
    pred_stereo_bond_orientations = (
        np.abs(compute_torsion_angles(coords, stereo_bond_index)) > np.pi / 2
    )
    stereo_bond_violations = (
        pred_stereo_bond_orientations != true_stereo_bond_orientations
    )

    return {
        "num_chiral_atom_violations": chiral_atom_violations.sum(),
        "num_chiral_atoms": chiral_atom_index.shape[1],
        "num_stereo_bond_violations": stereo_bond_violations.sum(),
        "num_stereo_bonds": stereo_bond_index.shape[1],
    }


def check_ligand_flatness(structure, constraints, buffer=0.25):
    coords = structure.coords["coords"]

    planar_ring_5_index = constraints.planar_ring_5_constraints["atom_idxs"]
    ring_5_coords = coords[planar_ring_5_index, :]
    centered_ring_5_coords = ring_5_coords - ring_5_coords.mean(axis=-2, keepdims=True)
    ring_5_vecs = np.linalg.svd(centered_ring_5_coords)[2][..., -1, :, None]
    ring_5_dists = np.abs((centered_ring_5_coords @ ring_5_vecs).squeeze(axis=-1))
    ring_5_violations = np.any(ring_5_dists >= buffer, axis=-1)

    planar_ring_6_index = constraints.planar_ring_6_constraints["atom_idxs"]
    ring_6_coords = coords[planar_ring_6_index, :]
    centered_ring_6_coords = ring_6_coords - ring_6_coords.mean(axis=-2, keepdims=True)
    ring_6_vecs = np.linalg.svd(centered_ring_6_coords)[2][..., -1, :, None]
    ring_6_dists = np.abs((centered_ring_6_coords @ ring_6_vecs)).squeeze(axis=-1)
    ring_6_violations = np.any(ring_6_dists >= buffer, axis=-1)

    planar_bond_index = constraints.planar_bond_constraints["atom_idxs"]
    bond_coords = coords[planar_bond_index, :]
    centered_bond_coords = bond_coords - bond_coords.mean(axis=-2, keepdims=True)
    bond_vecs = np.linalg.svd(centered_bond_coords)[2][..., -1, :, None]
    bond_dists = np.abs((centered_bond_coords @ bond_vecs)).squeeze(axis=-1)
    bond_violations = np.any(bond_dists >= buffer, axis=-1)

    return {
        "num_planar_5_ring_violations": ring_5_violations.sum(),
        "num_planar_5_rings": ring_5_violations.shape[0],
        "num_planar_6_ring_violations": ring_6_violations.sum(),
        "num_planar_6_rings": ring_6_violations.shape[0],
        "num_planar_double_bond_violations": bond_violations.sum(),
        "num_planar_double_bonds": bond_violations.shape[0],
    }


def check_steric_clash(structure, molecules, buffer=0.25):
    result = {}
    for type_i in const.chain_types:
        out_type_i = type_i.lower()
        out_type_i = out_type_i if out_type_i != "nonpolymer" else "ligand"
        result[f"num_chain_pairs_sym_{out_type_i}"] = 0
        result[f"num_chain_clashes_sym_{out_type_i}"] = 0
        for type_j in const.chain_types:
            out_type_j = type_j.lower()
            out_type_j = out_type_j if out_type_j != "nonpolymer" else "ligand"
            result[f"num_chain_pairs_asym_{out_type_i}_{out_type_j}"] = 0
            result[f"num_chain_clashes_asym_{out_type_i}_{out_type_j}"] = 0

    connected_chains = set()
    for bond in structure.bonds:
        if bond["chain_1"] != bond["chain_2"]:
            connected_chains.add(tuple(sorted((bond["chain_1"], bond["chain_2"]))))

    vdw_radii = []
    for res in structure.residues:
        mol = molecules[res["name"]]
        token_atoms = structure.atoms[
            res["atom_idx"] : res["atom_idx"] + res["atom_num"]
        ]
        atom_name_to_ref = {a.GetProp("name"): a for a in mol.GetAtoms()}
        token_atoms_ref = [atom_name_to_ref[a["name"]] for a in token_atoms]
        vdw_radii.extend(
            [const.vdw_radii[a.GetAtomicNum() - 1] for a in token_atoms_ref]
        )
    vdw_radii = np.array(vdw_radii, dtype=np.float32)

    np.array([a.GetAtomicNum() for a in token_atoms_ref])
    for i, chain_i in enumerate(structure.chains):
        for j, chain_j in enumerate(structure.chains):
            if (
                chain_i["atom_num"] == 1
                or chain_j["atom_num"] == 1
                or j <= i
                or (i, j) in connected_chains
            ):
                continue
            coords_i = structure.coords["coords"][
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            coords_j = structure.coords["coords"][
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            dists = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=-1)
            radii_i = vdw_radii[
                chain_i["atom_idx"] : chain_i["atom_idx"] + chain_i["atom_num"]
            ]
            radii_j = vdw_radii[
                chain_j["atom_idx"] : chain_j["atom_idx"] + chain_j["atom_num"]
            ]
            radii_sum = radii_i[:, None] + radii_j[None, :]
            is_clashing = np.any(dists < radii_sum * (1.00 - buffer))
            type_i = const.chain_types[chain_i["mol_type"]].lower()
            type_j = const.chain_types[chain_j["mol_type"]].lower()
            type_i = type_i if type_i != "nonpolymer" else "ligand"
            type_j = type_j if type_j != "nonpolymer" else "ligand"
            is_symmetric = (
                chain_i["entity_id"] == chain_j["entity_id"]
                and chain_i["atom_num"] == chain_j["atom_num"]
            )
            if is_symmetric:
                key = "sym_" + type_i
            else:
                key = "asym_" + type_i + "_" + type_j
            result["num_chain_pairs_" + key] += 1
            result["num_chain_clashes_" + key] += int(is_clashing)
    return result


tools = ["boltz1", "chai", "af3"]


def initialize_worker(ccd_path, mols_dir, prediction_paths, target_names):
    global ccd, moldir, prediction_dirs, prediction_targets
    with Path(ccd_path).open("rb") as file:
        ccd = pickle.load(file)
    moldir = Path(mols_dir)
    prediction_dirs = {name: Path(path) for name, path in prediction_paths.items()}
    prediction_targets = target_names


def process_fn(key):
    tool, pdb_id, model_idx = key
    target_name = prediction_targets[tool][pdb_id]
    target_dir = prediction_dirs[tool] / target_name
    if tool == "boltz1":
        cif_path = target_dir / f"{target_name}_model_{model_idx}.cif"
        if not cif_path.is_file():
            matches = sorted(target_dir.glob(f"*_model_{model_idx}.cif"))
            if len(matches) == 1:
                cif_path = matches[0]
    elif tool == "chai":
        cif_path = target_dir / f"pred.model_idx_{model_idx}.cif"
    else:
        cif_path = target_dir / f"seed-1_sample-{model_idx}" / "model.cif"

    if not cif_path.is_file():
        raise FileNotFoundError(
            f"{tool} prediction not found for {pdb_id} model {model_idx}: {cif_path}"
        )

    parsed_structure = parse_mmcif(
        cif_path,
        ccd,
        moldir,
    )
    structure = parsed_structure.data
    constraints = parsed_structure.residue_constraints

    record = {
        "tool": tool,
        "pdb_id": pdb_id,
        "model_idx": model_idx,
    }
    record.update(check_ligand_distance_geometry(structure, constraints))
    record.update(check_ligand_stereochemistry(structure, constraints))
    record.update(check_ligand_flatness(structure, constraints))
    record.update(check_steric_clash(structure, molecules=ccd))
    return record


def process_fn_safe(key):
    try:
        return process_fn(key), None
    except Exception as exc:  # noqa: BLE001
        tool, pdb_id, model_idx = key
        return None, {
            "tool": tool,
            "pdb_id": pdb_id,
            "model_idx": model_idx,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Boltz ligand geometry")
    parser.add_argument("--ccd", type=Path, required=True)
    parser.add_argument("--mols", type=Path, required=True)
    parser.add_argument(
        "--boltz1_dir",
        "--boltz1-dir",
        "--boltz-dir",
        dest="boltz1_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--chai_dir",
        "--chai-dir",
        dest="chai_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--af3_dir",
        "--af3-dir",
        dest="af3_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_samples", "--num-samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not args.ccd.is_file():
        raise FileNotFoundError(f"CCD file not found: {args.ccd}")
    if not args.mols.is_dir():
        raise NotADirectoryError(f"Molecule directory not found: {args.mols}")

    paths = {tool: getattr(args, f"{tool}_dir").resolve() for tool in tools}
    for tool, directory in paths.items():
        if not directory.is_dir():
            raise NotADirectoryError(
                f"{tool} prediction directory not found: {directory}"
            )

    target_names = {
        tool: {
            path.name.lower(): path.name
            for path in directory.iterdir()
            if path.is_dir()
        }
        for tool, directory in paths.items()
    }
    pdb_id_sets = [set(names) for names in target_names.values()]
    common_pdb_ids = set.intersection(*pdb_id_sets)
    if not common_pdb_ids:
        counts = ", ".join(
            f"{tool}={len(names)}" for tool, names in target_names.items()
        )
        raise ValueError(
            "No common prediction targets were found across boltz1, chai, and af3 "
            f"({counts})"
        )

    keys = [
        (tool, pdb_id, model_idx)
        for tool in tools
        for pdb_id in sorted(common_pdb_ids)
        for model_idx in range(args.num_samples)
    ]
    records = []
    errors = []
    initargs = (args.ccd.resolve(), args.mols.resolve(), paths, target_names)
    if args.workers == 1:
        initialize_worker(*initargs)
        iterator = map(process_fn_safe, keys)
        with tqdm(total=len(keys)) as pbar:
            for record, error in iterator:
                if record is not None:
                    records.append(record)
                if error is not None:
                    errors.append(error)
                pbar.update(1)
    else:
        with Pool(
            args.workers,
            initializer=initialize_worker,
            initargs=initargs,
        ) as pool:
            with tqdm(total=len(keys)) as pbar:
                for record, error in pool.imap_unordered(process_fn_safe, keys):
                    if record is not None:
                        records.append(record)
                    if error is not None:
                        errors.append(error)
                    pbar.update(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        error_path = args.output.with_name(f"{args.output.stem}_errors.csv")
        pd.DataFrame.from_records(errors).to_csv(error_path, index=False)
        print(  # noqa: T201
            f"{len(errors)} physical metric evaluations failed; "
            f"details were written to {error_path}",
            flush=True,
        )
    if not records:
        raise RuntimeError(
            "All physical metric evaluations failed; "
            "see the generated errors CSV for details"
        )

    df = pd.DataFrame.from_records(records)
    df["num_chain_clashes_all"] = df[
        [key for key in df.columns if "chain_clash" in key]
    ].sum(axis=1)
    df["num_pairs_all"] = df[
        [key for key in df.columns if "chain_pair" in key]
    ].sum(axis=1)
    df["clash_free"] = df["num_chain_clashes_all"] == 0
    df["valid_ligand"] = (
        df[[key for key in df.columns if "violation" in key]].sum(axis=1) == 0
    )
    df["valid"] = df["clash_free"] & df["valid_ligand"]
    df.to_csv(args.output, index=False)
    print(  # noqa: T201
        f"Wrote {len(df)} physical metric records to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
