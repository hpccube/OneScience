"""Material-property workflows driven by an Equiformer V3 ASE calculator."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def relax_structure(
    atoms: Atoms,
    calculator: Calculator,
    *,
    relax_cell: bool = True,
    fmax: float = 0.02,
    steps: int = 200,
) -> tuple[Atoms, dict[str, Any]]:
    """Relax a copied structure and return it with convergence metadata."""

    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    if fmax <= 0 or steps < 1:
        raise ValueError("fmax and steps must be positive")
    if relax_cell and not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError("Cell relaxation requires three-dimensional periodicity")

    structure = atoms.copy()
    structure.calc = calculator
    target = FrechetCellFilter(structure) if relax_cell else structure
    optimizer = FIRE(target, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    forces = np.asarray(structure.get_forces(), dtype=float)
    maximum_force = float(np.max(np.linalg.norm(forces, axis=1)))
    metadata = {
        "cell_relaxed": bool(relax_cell),
        "converged": converged,
        "optimizer": "FIRE",
        "steps": int(optimizer.nsteps),
        "target_fmax_ev_per_angstrom": float(fmax),
        "maximum_atomic_force_ev_per_angstrom": maximum_force,
        "energy_ev": float(structure.get_potential_energy()),
        "cell_angstrom": np.asarray(structure.cell).tolist(),
    }
    if np.asarray(structure.pbc, dtype=bool).all():
        metadata["stress_ev_per_angstrom_cubed_voigt"] = np.asarray(
            structure.get_stress(), dtype=float
        ).tolist()
    return structure, metadata


def _validate_element_reference_energies(
    reference_energies: Mapping[str, float],
) -> dict[str, float]:
    references = {
        str(element): float(value) for element, value in reference_energies.items()
    }
    if not references:
        raise ValueError("At least one elemental reference energy is required")
    invalid = [
        element for element, value in references.items() if not math.isfinite(value)
    ]
    if invalid:
        raise ValueError(f"Non-finite elemental reference energies: {invalid}")
    return references


def load_element_reference_energies(path: str | Path) -> dict[str, float]:
    """Load an element-to-energy mapping from JSON or YAML.

    Values must be elemental reference energies in eV/atom. A previous
    workflow result can also be supplied because its
    ``element_reference_energies_ev_per_atom`` mapping is recognized.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as error:
                raise ImportError(
                    "PyYAML is required to read YAML references"
                ) from error
            payload = yaml.safe_load(handle)
        else:
            raise ValueError(
                "Reference energies must use a .json, .yaml, or .yml suffix"
            )

    if not isinstance(payload, Mapping):
        raise ValueError(f"Reference file {path} must contain a mapping")
    nested_key = "element_reference_energies_ev_per_atom"
    if nested_key in payload:
        payload = payload[nested_key]
    if not isinstance(payload, Mapping):
        raise ValueError(f"{nested_key} in {path} must be a mapping")
    return _validate_element_reference_energies(payload)


def calculate_element_reference_energies(
    reference_structures: Mapping[str, Atoms],
    calculator: Calculator,
) -> dict[str, float]:
    """Evaluate pure-element reference phases and return energies in eV/atom."""

    energies: dict[str, float] = {}
    for element, atoms in reference_structures.items():
        symbols = set(atoms.get_chemical_symbols())
        if symbols != {element}:
            raise ValueError(
                f"Reference structure {element!r} contains elements {sorted(symbols)}"
            )
        structure = atoms.copy()
        structure.calc = calculator
        energies[element] = float(structure.get_potential_energy()) / len(structure)
    return _validate_element_reference_energies(energies)


def formation_energy_from_references(
    atoms: Atoms,
    total_energy_ev: float,
    reference_energies: Mapping[str, float],
) -> dict[str, Any]:
    """Calculate an uncorrected formation energy from explicit references."""

    references = _validate_element_reference_energies(reference_energies)
    composition = Counter(atoms.get_chemical_symbols())
    missing = sorted(set(composition) - set(references))
    if missing:
        raise ValueError(f"Missing elemental reference energies for: {missing}")

    reference_energy = sum(
        composition[element] * references[element] for element in composition
    )
    formation_energy = float(total_energy_ev) - reference_energy
    return {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "composition": dict(sorted(composition.items())),
        "total_energy_ev": float(total_energy_ev),
        "reference_energy_ev": float(reference_energy),
        "formation_energy_ev": float(formation_energy),
        "formation_energy_ev_per_atom": float(formation_energy / len(atoms)),
        "element_reference_energies_ev_per_atom": dict(sorted(references.items())),
        "corrections_applied": False,
    }


def calculate_formation_energy(
    atoms: Atoms,
    calculator: Calculator,
    reference_energies: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate a structure and calculate its uncorrected formation energy."""

    structure = atoms.copy()
    structure.calc = calculator
    total_energy = float(structure.get_potential_energy())
    return formation_energy_from_references(structure, total_energy, reference_energies)


def run_phonon_workflow(
    atoms: Atoms,
    calculator: Calculator,
    workdir: str | Path,
    *,
    supercell: Sequence[int] = (3, 3, 3),
    delta: float = 0.01,
    bandpath: str | None = None,
    band_points: int = 100,
    dos_kpts: Sequence[int] = (10, 10, 10),
    dos_points: int = 400,
    dos_width_ev: float = 0.001,
) -> dict[str, Any]:
    """Run ASE finite-displacement phonons and return bands and a DOS grid."""

    from ase.phonons import Phonons

    if not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError(
            "Phonon workflow requires three-dimensional periodic boundaries"
        )
    supercell = tuple(int(value) for value in supercell)
    dos_kpts = tuple(int(value) for value in dos_kpts)
    if len(supercell) != 3 or any(value < 1 for value in supercell):
        raise ValueError("supercell must contain three positive integers")
    if len(dos_kpts) != 3 or any(value < 1 for value in dos_kpts):
        raise ValueError("dos_kpts must contain three positive integers")
    if delta <= 0 or band_points < 2 or dos_points < 2 or dos_width_ev <= 0:
        raise ValueError("delta, point counts, and DOS width must be positive")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    structure = atoms.copy()
    structure.calc = calculator
    phonons = Phonons(
        structure,
        calculator,
        supercell=supercell,
        delta=delta,
        name=str(workdir / "displacements"),
    )
    phonons.run()
    phonons.read(acoustic=True)

    path = structure.cell.bandpath(path=bandpath, npoints=band_points)
    bands = phonons.get_band_structure(path, verbose=False)
    raw_dos = phonons.get_dos(kpts=dos_kpts, verbose=False)
    dos = raw_dos.sample_grid(npts=dos_points, width=dos_width_ev)

    band_energies = np.asarray(bands.energies[0])
    imaginary_threshold_ev = -1e-6
    return {
        "formula": structure.get_chemical_formula(),
        "supercell": list(supercell),
        "displacement_angstrom": float(delta),
        "band_path": path.path,
        "band_kpoints": np.asarray(path.kpts).tolist(),
        "band_energies_ev": band_energies.tolist(),
        "minimum_band_energy_ev": float(np.min(band_energies)),
        "imaginary_band_sample_count": int(
            np.count_nonzero(band_energies < imaginary_threshold_ev)
        ),
        "imaginary_threshold_ev": imaginary_threshold_ev,
        "dos_kpoints": list(dos_kpts),
        "dos_energy_ev": np.asarray(dos.get_energies()).tolist(),
        "dos_states_per_ev": np.asarray(dos.get_weights()).tolist(),
        "cache_directory": str(workdir / "displacements"),
    }


def run_elastic_workflow(
    atoms: Atoms,
    calculator: Calculator,
    *,
    normal_strains: Sequence[float] = (-0.01, 0.01),
    shear_strains: Sequence[float] = (-0.02, 0.02),
    relax_positions: bool = False,
    relax_fmax: float = 0.02,
    relax_steps: int = 100,
) -> dict[str, Any]:
    """Fit a second-order elastic tensor with pymatgen strain states."""

    from ase.optimize import FIRE
    from pymatgen.core.elasticity import (
        DeformedStructureSet,
        ElasticTensor,
        Stress,
    )
    from pymatgen.io.ase import AseAtomsAdaptor

    if not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError(
            "Elastic workflow requires three-dimensional periodic boundaries"
        )
    normal_strains = tuple(float(value) for value in normal_strains)
    shear_strains = tuple(float(value) for value in shear_strains)
    if len(set(normal_strains)) < 2 or len(set(shear_strains)) < 2:
        raise ValueError("At least two distinct normal and shear strains are required")
    if relax_fmax <= 0 or relax_steps < 1:
        raise ValueError("relax_fmax and relax_steps must be positive")

    adaptor = AseAtomsAdaptor()
    equilibrium = atoms.copy()
    equilibrium.calc = calculator
    equilibrium_stress = np.asarray(
        equilibrium.get_stress(voigt=False), dtype=float
    ) / units.GPa

    structure = adaptor.get_structure(equilibrium)
    deformed_set = DeformedStructureSet(
        structure,
        norm_strains=normal_strains,
        shear_strains=shear_strains,
        symmetry=False,
    )
    strains = []
    stresses = []
    samples = []
    for deformation, deformed_structure in zip(
        deformed_set.deformations, deformed_set.deformed_structures, strict=True
    ):
        deformed_atoms = adaptor.get_atoms(deformed_structure)
        deformed_atoms.calc = calculator
        if relax_positions:
            optimizer = FIRE(deformed_atoms, logfile=None)
            optimizer.run(fmax=relax_fmax, steps=relax_steps)
        stress = (
            np.asarray(deformed_atoms.get_stress(voigt=False), dtype=float)
            / units.GPa
        )
        strain = deformation.green_lagrange_strain
        strains.append(strain)
        stresses.append(Stress(stress))
        samples.append(
            {
                "strain_voigt": np.asarray(strain.voigt).tolist(),
                "stress_gpa_voigt": np.asarray(Stress(stress).voigt).tolist(),
            }
        )

    tensor = ElasticTensor.from_independent_strains(
        strains,
        stresses,
        eq_stress=Stress(equilibrium_stress),
    )
    return {
        "formula": equilibrium.get_chemical_formula(),
        "normal_strains": list(normal_strains),
        "shear_strains": list(shear_strains),
        "internal_positions_relaxed": bool(relax_positions),
        "equilibrium_stress_gpa_voigt": np.asarray(
            Stress(equilibrium_stress).voigt
        ).tolist(),
        "elastic_tensor_gpa_voigt": np.asarray(tensor.voigt).tolist(),
        "bulk_modulus_gpa": {
            "voigt": float(tensor.k_voigt),
            "reuss": float(tensor.k_reuss),
            "hill": float(tensor.k_vrh),
        },
        "shear_modulus_gpa": {
            "voigt": float(tensor.g_voigt),
            "reuss": float(tensor.g_reuss),
            "hill": float(tensor.g_vrh),
        },
        "universal_anisotropy": float(tensor.universal_anisotropy),
        "samples": samples,
    }


def write_workflow_result(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write a workflow result as stable, human-readable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
