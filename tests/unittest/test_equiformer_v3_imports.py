from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_jd(path: Path, max_l: int = 8) -> None:
    import torch

    torch.save([torch.eye(2 * degree + 1) for degree in range(max_l + 1)], path)


def test_equiformer_v3_public_imports():
    from onescience.models.equiformer_v3 import (
        EquiformerV3,
        EquiformerV3DeNS,
    )
    from onescience.utils.equiformer_v3 import (
        EquiformerV3Calculator,
        EquiformerV3CheckpointTransforms,
        load_equiformer_v3_checkpoint,
        relax_structure,
    )

    assert EquiformerV3.__module__.startswith("onescience.models.equiformer_v3")
    assert EquiformerV3DeNS.__module__.startswith(
        "onescience.models.equiformer_v3"
    )
    assert EquiformerV3Calculator.__module__ == (
        "onescience.utils.equiformer_v3.calculator"
    )
    assert EquiformerV3CheckpointTransforms.__module__ == (
        "onescience.utils.equiformer_v3.checkpoint"
    )
    assert load_equiformer_v3_checkpoint.__module__ == (
        "onescience.utils.equiformer_v3.checkpoint"
    )
    assert relax_structure.__module__ == "onescience.utils.equiformer_v3.workflows"


def test_equiformer_v3_strips_torch_compile_prefix():
    import torch

    from onescience.utils.equiformer_v3.checkpoint import (
        strip_torch_compile_prefix,
    )

    state_dict = {
        "_orig_mod.module.weight": torch.ones(1),
        "module.bias": torch.zeros(1),
    }

    stripped = strip_torch_compile_prefix(state_dict)

    assert set(stripped) == {"module.weight", "module.bias"}


def test_equiformer_v3_checkpoint_loader_handles_compiled_state(
    monkeypatch, tmp_path
):
    import torch

    from onescience.utils.equiformer_v3.checkpoint import (
        load_equiformer_v3_checkpoint,
    )
    from onescience.utils.uma.common.registry import registry

    checkpoint_path = tmp_path / "compiled.pt"
    checkpoint_path.touch()
    checkpoint = {
        "config": {"model": {"name": "equiformer_v3", "marker": 7}},
        "state_dict": {
            "_orig_mod.module.weight": torch.ones(1),
            "_orig_mod.module.bias": torch.zeros(1),
        },
    }
    constructed = {}

    class Model(torch.nn.Module):
        def __init__(self, marker):
            super().__init__()
            constructed["marker"] = marker
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.bias = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(registry, "get_model_class", lambda name: Model)

    model = load_equiformer_v3_checkpoint(checkpoint_path)

    assert constructed == {"marker": 7}
    assert torch.equal(model.weight, torch.ones(1))
    assert torch.equal(model.bias, torch.zeros(1))
    assert checkpoint["config"]["model"]["name"] == "equiformer_v3"


def test_equiformer_v3_registry_and_small_model(monkeypatch, tmp_path):
    from onescience.models.equiformer_v3 import EquiformerV3
    from onescience.utils.uma.common.registry import registry

    jd_path = tmp_path / "Jd.pt"
    _write_jd(jd_path)
    monkeypatch.setenv("ONESCIENCE_EQUIFORMER_V3_JD_PATH", str(jd_path))

    model = EquiformerV3(
        num_layers=1,
        num_channels=16,
        attn_hidden_channels=8,
        num_heads=2,
        attn_alpha_channels=4,
        attn_value_channels=4,
        ffn_hidden_channels=16,
        lmax=2,
        mmax=2,
        attn_grid_resolution_list=[8, 4],
        ffn_grid_resolution_list=[8, 8],
        edge_channels=16,
        num_radial_basis=8,
        max_neighbors=8,
        max_radius=4.0,
        drop_path_rate=0.0,
        attn_weights_drop=0.0,
    )

    assert model.num_params > 0
    assert registry.get_model_class("equiformer_v3") is EquiformerV3
    assert registry.get_model_class("equiformer_v3_dens").__name__ == (
        "EquiformerV3DeNS_OC"
    )


def test_equiformer_v3_wigner_blocks_match_rotation_dtype(monkeypatch):
    import torch

    from onescience.modules.layer.equiformer_v3 import so3

    def half_precision_wigner(degree, alpha, beta, gamma):
        del beta, gamma
        width = 2 * degree + 1
        return torch.ones(
            len(alpha), width, width, device=alpha.device, dtype=torch.float16
        )

    monkeypatch.setattr(so3, "wigner_D", half_precision_wigner)
    rotation = so3.SO3Rotation(lmax=1, mmax=1, use_rotation_mask=True)
    edge_rot_mat = torch.stack(
        (
            torch.eye(3),
            torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )
    )

    wigner = rotation._rotation_to_wigner_matrix(edge_rot_mat, 0, 1)

    assert wigner.dtype == edge_rot_mat.dtype
    assert torch.count_nonzero(wigner) == 20


def test_equiformer_v3_wigner_uses_active_autocast_dtype(monkeypatch):
    import torch

    from onescience.modules.layer.equiformer_v3 import so3

    rotation = so3.SO3Rotation(lmax=1, mmax=1)
    monkeypatch.setattr(
        rotation,
        "_rotation_to_wigner_matrix",
        lambda edge_rot_mat, start_lmax, end_lmax: torch.eye(4).repeat(
            edge_rot_mat.shape[0], 1, 1
        ),
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda device_type: torch.bfloat16)

    rotation.set_wigner(torch.eye(3).unsqueeze(0))

    assert rotation.wigner.dtype == torch.bfloat16
    assert rotation.wigner_inv.dtype == torch.bfloat16


def test_equiformer_v3_dens_small_model(monkeypatch, tmp_path):
    from onescience.models.equiformer_v3 import EquiformerV3DeNS

    jd_path = tmp_path / "Jd.pt"
    _write_jd(jd_path)
    monkeypatch.setenv("ONESCIENCE_EQUIFORMER_V3_JD_PATH", str(jd_path))

    model = EquiformerV3DeNS(
        num_layers=1,
        num_channels=16,
        attn_hidden_channels=8,
        num_heads=2,
        attn_alpha_channels=4,
        attn_value_channels=4,
        ffn_hidden_channels=16,
        lmax=2,
        mmax=2,
        attn_grid_resolution_list=[8, 4],
        ffn_grid_resolution_list=[8, 8],
        edge_channels=16,
        num_radial_basis=8,
        max_neighbors=8,
        max_radius=4.0,
        drop_path_rate=0.0,
        attn_weights_drop=0.0,
    )

    assert model.num_params > 0
    assert model.dens_block is not None


def test_equiformer_v3_jd_resolution_prefers_model_override(monkeypatch, tmp_path):
    from onescience.modules.func_utils.equiformer_v3_path_utils import (
        resolve_equiformer_v3_jd_path,
    )

    eqv3_jd = tmp_path / "equiformer_v3_Jd.pt"
    uma_jd = tmp_path / "uma_Jd.pt"
    eqv3_jd.touch()
    uma_jd.touch()
    monkeypatch.setenv("ONESCIENCE_EQUIFORMER_V3_JD_PATH", str(eqv3_jd))
    monkeypatch.setenv("ONESCIENCE_UMA_JD_PATH", str(uma_jd))

    assert resolve_equiformer_v3_jd_path() == str(eqv3_jd)


def test_equiformer_v3_jd_resolution_uses_shared_model_store(monkeypatch, tmp_path):
    from onescience.modules.func_utils.equiformer_v3_path_utils import (
        resolve_equiformer_v3_jd_path,
    )

    jd_path = tmp_path / "UMA" / "checkpoint" / "Jd.pt"
    jd_path.parent.mkdir(parents=True)
    jd_path.touch()
    monkeypatch.delenv("ONESCIENCE_EQUIFORMER_V3_JD_PATH", raising=False)
    monkeypatch.setenv("ONESCIENCE_MODELS_DIR", str(tmp_path))

    assert resolve_equiformer_v3_jd_path() == str(jd_path)


def test_equiformer_v3_calculator_accepts_direct_models():
    import torch

    from onescience.utils.equiformer_v3 import EquiformerV3Calculator

    class DirectModel(torch.nn.Module):
        max_neighbors = 8
        cutoff = 4.0

        def forward(self, batch):
            return {}

    model = DirectModel()
    calculator = EquiformerV3Calculator(model=model, device="cpu")

    assert calculator.backbone is model


def test_equiformer_v3_calculator_runs_ase_contract():
    import torch
    from ase import Atoms

    from onescience.utils.equiformer_v3 import EquiformerV3Calculator

    class DirectModel(torch.nn.Module):
        max_neighbors = 8
        cutoff = 4.0

        def forward(self, batch):
            assert torch.is_grad_enabled()
            return {
                "energy": torch.tensor([2.5], device=batch.pos.device),
                "forces": torch.zeros_like(batch.pos),
                "stress": torch.diag(
                    torch.tensor([1.0, 2.0, 3.0], device=batch.pos.device)
                ).reshape(1, 9),
            }

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = EquiformerV3Calculator(model=DirectModel(), device="cpu")

    assert atoms.get_potential_energy() == 2.5
    np.testing.assert_allclose(atoms.get_forces(), np.zeros((1, 3)))
    np.testing.assert_allclose(atoms.get_stress(), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])


def test_equiformer_v3_formation_energy_contract(tmp_path):
    import json

    from ase import Atoms

    from onescience.utils.equiformer_v3 import (
        formation_energy_from_references,
        load_element_reference_energies,
    )

    atoms = Atoms("MgO")
    result = formation_energy_from_references(
        atoms,
        total_energy_ev=-12.0,
        reference_energies={"Mg": -2.0, "O": -4.0},
    )

    assert result["formation_energy_ev"] == -6.0
    assert result["formation_energy_ev_per_atom"] == -3.0
    assert result["corrections_applied"] is False

    references_path = tmp_path / "references.json"
    references_path.write_text(json.dumps(result), encoding="utf-8")
    assert load_element_reference_energies(references_path) == {
        "Mg": -2.0,
        "O": -4.0,
    }


def test_equiformer_v3_relaxation_contract():
    from ase.build import bulk
    from ase.calculators.calculator import Calculator

    from onescience.utils.equiformer_v3 import relax_structure

    class ZeroCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

    atoms = bulk("Cu")
    relaxed, metadata = relax_structure(
        atoms, ZeroCalculator(), relax_cell=True, fmax=0.01, steps=2
    )

    assert relaxed is not atoms
    assert metadata["converged"] is True
    assert metadata["steps"] == 0
    assert metadata["cell_relaxed"] is True
    assert metadata["stress_ev_per_angstrom_cubed_voigt"] == [0.0] * 6


def test_equiformer_v3_formation_energy_requires_all_elements():
    import pytest
    from ase import Atoms

    from onescience.utils.equiformer_v3 import formation_energy_from_references

    with pytest.raises(ValueError, match="Missing elemental reference energies"):
        formation_energy_from_references(
            Atoms("MgO"), total_energy_ev=-12.0, reference_energies={"Mg": -2.0}
        )


def test_equiformer_v3_phonon_output_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ase.build import bulk

    from onescience.utils.equiformer_v3 import run_phonon_workflow

    class Grid:
        def get_energies(self):
            return np.array([0.0, 0.1])

        def get_weights(self):
            return np.array([1.0, 2.0])

    class RawDOS:
        def sample_grid(self, npts, width):
            assert npts == 2
            assert width == 0.001
            return Grid()

    class FakePhonons:
        def __init__(self, atoms, calculator, supercell, delta, name):
            assert supercell == (1, 1, 1)
            assert delta == 0.01

        def run(self):
            pass

        def read(self, acoustic):
            assert acoustic

        def get_band_structure(self, path, verbose):
            assert not verbose
            energies = np.arange(len(path.kpts) * 3, dtype=float).reshape(
                1, len(path.kpts), 3
            )
            return SimpleNamespace(energies=energies)

        def get_dos(self, kpts, verbose):
            assert kpts == (1, 1, 1)
            assert not verbose
            return RawDOS()

    monkeypatch.setattr("ase.phonons.Phonons", FakePhonons)
    result = run_phonon_workflow(
        bulk("Cu"),
        calculator=None,
        workdir=tmp_path,
        supercell=(1, 1, 1),
        band_points=2,
        dos_kpts=(1, 1, 1),
        dos_points=2,
    )

    assert result["formula"] == "Cu"
    assert result["supercell"] == [1, 1, 1]
    assert len(result["band_energies_ev"]) == len(result["band_kpoints"])
    assert result["dos_states_per_ev"] == [1.0, 2.0]
    assert result["imaginary_band_sample_count"] == 0


def test_equiformer_v3_elastic_workflow_fits_linear_response():
    from ase import units
    from ase.build import bulk
    from ase.calculators.calculator import Calculator
    from ase.stress import voigt_6_to_full_3x3_stress
    from pymatgen.core.elasticity import Strain

    from onescience.utils.equiformer_v3 import run_elastic_workflow

    stiffness = np.diag([200.0, 200.0, 200.0, 80.0, 80.0, 80.0])
    reference_cell = np.asarray(bulk("Cu").cell)

    class LinearElasticCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            deformation = (np.linalg.inv(reference_cell) @ np.asarray(atoms.cell)).T
            strain = Strain.from_deformation(deformation)
            stress_voigt = stiffness @ np.asarray(strain.voigt)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": voigt_6_to_full_3x3_stress(stress_voigt * units.GPa),
            }

    result = run_elastic_workflow(
        bulk("Cu"),
        LinearElasticCalculator(),
        normal_strains=(-0.01, 0.01),
        shear_strains=(-0.02, 0.02),
    )

    np.testing.assert_allclose(
        result["elastic_tensor_gpa_voigt"], stiffness, atol=1e-8
    )
    np.testing.assert_allclose(result["bulk_modulus_gpa"]["hill"], 200.0 / 3.0)


def test_equiformer_v3_has_no_external_fairchem_imports():
    repo_root = Path(__file__).resolve().parents[2]
    source_roots = (
        repo_root / "src" / "onescience" / "models" / "equiformer_v3",
        repo_root / "src" / "onescience" / "modules" / "layer" / "equiformer_v3",
        repo_root / "src" / "onescience" / "utils" / "equiformer_v3",
    )

    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "from fairchem" not in source
            assert "import fairchem" not in source


def test_equiformer_v3_does_not_import_esen():
    repo_root = Path(__file__).resolve().parents[2]
    source_roots = (
        repo_root / "src" / "onescience" / "models" / "equiformer_v3",
        repo_root / "src" / "onescience" / "utils" / "equiformer_v3",
        repo_root / "examples" / "matchem" / "equiformer_v3",
    )

    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            assert "onescience.models.esen" not in source
            assert "onescience.utils.esen" not in source


def test_esen_graph_mixin_compatibility_import():
    from onescience.models.esen.graph import GraphModelMixin as CompatMixin
    from onescience.modules.func_utils.uma_graph.mixin import GraphModelMixin

    assert CompatMixin is GraphModelMixin


def test_equiformer_v3_examples_expose_inference_workflows():
    repo_root = Path(__file__).resolve().parents[2]
    example_root = repo_root / "examples" / "matchem" / "equiformer_v3"

    for example in (
        "single_point.py",
        "formation_energy.py",
        "elastic.py",
        "phonons.py",
    ):
        assert (example_root / example).is_file()

    for removed_example in ("relax.py", "md.py"):
        assert not (example_root / removed_example).exists()


def test_equiformer_v3_single_point_example_writes_json(monkeypatch, tmp_path):
    import importlib.util
    import json
    import sys

    from ase.calculators.calculator import Calculator

    example = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "matchem"
        / "equiformer_v3"
        / "single_point.py"
    )
    spec = importlib.util.spec_from_file_location(
        "equiformer_v3_single_point_example", example
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ConstantCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": -3.25,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.arange(6, dtype=float),
            }

    class FakeFactory:
        @staticmethod
        def from_checkpoint(checkpoint, device=None):
            assert checkpoint == "model.pt"
            assert device == "cpu"
            return ConstantCalculator()

    output = tmp_path / "single_point.json"
    monkeypatch.setattr(module, "EquiformerV3Calculator", FakeFactory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_point.py",
            "--checkpoint",
            "model.pt",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    module.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["formula"] == "Cu"
    assert result["energy_ev"] == -3.25
    assert result["forces_ev_per_angstrom"] == [[0.0, 0.0, 0.0]]
    assert result["stress_ev_per_angstrom_cubed_voigt"] == list(range(6))


def test_equiformer_v3_source_revision_is_recorded():
    repo_root = Path(__file__).resolve().parents[2]
    source_notice = (
        repo_root
        / "src"
        / "onescience"
        / "models"
        / "equiformer_v3"
        / "SOURCE.md"
    ).read_text(encoding="utf-8")

    assert "a7300c58df683dc99cb48027d5bfd4c887486c48" in source_notice
    assert "977a80328f2be44649b414a9907a1d6ef2f81e95" in source_notice


if __name__ == "__main__":
    import pytest

    exit_code = pytest.main([str(Path(__file__).resolve())])
    if exit_code == pytest.ExitCode.OK:
        print("Equiformer V3 模块导入测试成功。")
    raise SystemExit(exit_code)
