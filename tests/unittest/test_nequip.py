"""Unit tests for the NequIP OneScience integration and examples."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
import pytest
import torch


warnings.filterwarnings("ignore", category=FutureWarning, module="e3nn")


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "matchem" / "nequip"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ddp_demo_config_matches_launcher_contract():
    parser = load_module("nequip_demo_parser", EXAMPLE_DIR / "demo/_parse_config.py")
    config_path = EXAMPLE_DIR / "demo/configs/tutorial_smoke_8dcu.yaml"
    config = parser._config(str(config_path))

    output = StringIO()
    with redirect_stdout(output):
        parser._print_launch(config)

    assignments = dict(line.split("=", 1) for line in output.getvalue().splitlines())
    assert assignments == {
        "RUN_MODE": "auto",
        "NODES": "1",
        "GPUS_PER_NODE": "8",
        "WORLD_SIZE": "8",
    }
    assert config["trainer"]["devices"] == config["launch"]["num_gpus"]
    assert config["trainer"]["num_nodes"] == config["launch"]["num_nodes"]
    assert config["trainer"]["strategy"]["_target_"].endswith("DDPStrategy")


def _fake_launcher_environment(tmp_path: Path, visible_dcus: int) -> tuple[dict, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launch_log = tmp_path / "launch.log"

    python3 = bin_dir / "python3"
    python3.write_text(
        "#!/bin/bash\n"
        "if [[ \"${1:-}\" == '-c' && \"${2:-}\" == *'torch.cuda.device_count'* ]]; then\n"
        "  echo \"$FAKE_VISIBLE_DCUS\"\n"
        "  exit 0\n"
        "fi\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    python3.chmod(0o755)

    for command in ("python", "torchrun", "srun", "sbatch"):
        executable = bin_dir / command
        executable.write_text(
            "#!/bin/bash\n"
            f"printf '{command} %s\\n' \"$*\" >> \"$LAUNCH_LOG\"\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    env = os.environ.copy()
    for name in ("SLURM_JOB_ID", "SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        env.pop(name, None)
    env.update(
        {
            "CONDA_PREFIX": str(tmp_path / "conda"),
            "ONESCIENCE_MODELS_DIR": str(tmp_path / "models"),
            "ONESCIENCE_DATASETS_DIR": str(tmp_path / "datasets"),
            "ONESCIENCE_NEQUIP_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "FAKE_VISIBLE_DCUS": str(visible_dcus),
            "REAL_PYTHON": sys.executable,
            "LAUNCH_LOG": str(launch_log),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    return env, launch_log


def _run_auto_launcher(tmp_path: Path, visible_dcus: int, *, in_slurm: bool = False):
    env, launch_log = _fake_launcher_environment(tmp_path, visible_dcus)
    if in_slurm:
        env.update({"SLURM_JOB_ID": "12345", "SLURM_NNODES": "1"})
    result = subprocess.run(
        [
            "bash",
            str(EXAMPLE_DIR / "demo/run.sh"),
            "--config",
            str(EXAMPLE_DIR / "demo/configs/tutorial_smoke_8dcu.yaml"),
        ],
        cwd=EXAMPLE_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = launch_log.read_text(encoding="utf-8") if launch_log.exists() else ""
    return result, log


def test_auto_launcher_uses_matching_compute_node_resources(tmp_path):
    result, log = _run_auto_launcher(tmp_path, visible_dcus=8, in_slurm=True)

    assert result.returncode == 0, result.stderr
    assert "Current resources satisfy the config" in result.stdout
    assert log.startswith("torchrun ")


def test_auto_launcher_submits_when_resources_are_insufficient(tmp_path):
    result, log = _run_auto_launcher(tmp_path, visible_dcus=0)

    assert result.returncode == 0, result.stderr
    assert "Submitting to Slurm" in result.stdout
    assert log.startswith("sbatch ")


def test_auto_launcher_submits_from_small_allocation(tmp_path):
    result, log = _run_auto_launcher(tmp_path, visible_dcus=1, in_slurm=True)

    assert result.returncode == 0, result.stderr
    assert "Submitting a new Slurm job" in result.stdout
    assert log.startswith("sbatch ")


def test_example_launchers_do_not_override_pythonpath():
    run_script = (EXAMPLE_DIR / "demo/run.sh").read_text(encoding="utf-8")
    train_script = (EXAMPLE_DIR / "train.py").read_text(encoding="utf-8")

    assert "export PYTHONPATH=" not in run_script
    assert "sys.path.insert" not in train_script


def test_full_ddp_config_preserves_official_data_contract():
    parser = load_module("nequip_demo_parser_full", EXAMPLE_DIR / "demo/_parse_config.py")
    config = parser._config(
        str(EXAMPLE_DIR / "demo/configs/tutorial_fcu_8dcu.yaml")
    )

    assert config["data"]["split_dataset"]["train"] == 0.8
    assert config["data"]["split_dataset"]["val"] == 0.1
    assert config["data"]["split_dataset"]["test"] == 0.1
    assert config["data"]["train_dataloader"]["batch_size"] == 5
    assert config["trainer"]["devices"] == 8
    assert config["trainer"]["logger"]["_target_"].endswith("CSVLogger")
    assert config["trainer"]["enable_progress_bar"] is True
    assert config["trainer"]["callbacks"][0]["_target_"].endswith(
        "PlainTextMetricsLogger"
    )
    assert config["training_module"]["optimizer"]["lr"] == 0.01


def test_user_finetune_configs_enable_metrics_and_progress():
    parser = load_module("nequip_demo_parser_finetune", EXAMPLE_DIR / "demo/_parse_config.py")

    for name in ("oam_l_finetune_smoke.yaml", "oam_l_finetune.yaml"):
        config = parser._config(str(EXAMPLE_DIR / "demo/configs" / name))
        assert config["trainer"]["logger"]["_target_"].endswith("CSVLogger")
        assert config["trainer"]["enable_progress_bar"] is True
        assert config["trainer"]["callbacks"][0]["_target_"].endswith(
            "PlainTextMetricsLogger"
        )


def test_plain_text_metrics_logger_writes_epoch_line(capsys):
    from onescience.utils.nequip.train.callbacks import PlainTextMetricsLogger

    trainer = SimpleNamespace(
        callback_metrics={
            "train_loss_epoch/weighted_sum": torch.tensor(0.25),
            "val0_epoch/weighted_sum": torch.tensor(0.5),
            "train_loss_step/weighted_sum": torch.tensor(0.75),
        },
        current_epoch=2,
        global_step=30,
        is_global_zero=True,
        max_epochs=100,
        sanity_checking=False,
    )

    PlainTextMetricsLogger().on_validation_end(trainer, None)

    output = capsys.readouterr().out
    assert "[progress] epoch=3/100 global_step=30" in output
    assert "train_loss_epoch/weighted_sum=0.25" in output
    assert "val0_epoch/weighted_sum=0.5" in output
    assert "train_loss_step" not in output


def test_single_point_explicit_checkpoint_overrides_environment_default(
    tmp_path, monkeypatch
):
    single_point = load_module(
        "nequip_single_point_paths", EXAMPLE_DIR / "single_point.py"
    )
    default_model = tmp_path / "NequIP" / "NequIP-OAM-L-0.1.nequip.pth"
    default_model.parent.mkdir()
    default_model.touch()
    monkeypatch.setenv("ONESCIENCE_MODELS_DIR", str(tmp_path))

    assert single_point.resolve_model_paths(None, None) == {
        "compiled_model": str(default_model),
        "checkpoint": None,
    }

    selected = single_point.resolve_model_paths(None, "/checkpoints/best.ckpt")

    assert selected == {
        "compiled_model": None,
        "checkpoint": "/checkpoints/best.ckpt",
    }


def test_relaxation_history_contract(tmp_path):
    relaxation = load_module(
        "nequip_structure_relaxation", EXAMPLE_DIR / "structure_relaxation.py"
    )
    atoms = bulk("Cu", "fcc", a=3.7, cubic=True)
    atoms.positions[0, 0] += 0.02
    atoms.calc = EMT()

    converged, steps, history = relaxation.relax_structure(
        atoms,
        optimizer_name="BFGS",
        cell_filter_name="frechet",
        fixed_cell=True,
        fmax=10.0,
        steps=2,
        force_limit=1.0e6,
        logfile=tmp_path / "relax.log",
        trajectory=tmp_path / "relax.traj",
    )

    assert converged
    assert steps == 0
    assert len(history) == 1
    assert set(history[0]) == {
        "step",
        "energy_ev",
        "energy_ev_per_atom",
        "volume_angstrom3",
        "max_atomic_force_ev_per_angstrom",
        "max_optimizer_force",
        "stress_ev_per_angstrom3_voigt",
        "max_abs_stress_ev_per_angstrom3",
    }


def test_cell_relaxation_rejects_nonperiodic_structure(tmp_path):
    relaxation = load_module(
        "nequip_structure_relaxation_nonperiodic",
        EXAMPLE_DIR / "structure_relaxation.py",
    )

    atoms = Atoms("H")
    try:
        relaxation.relax_structure(
            atoms,
            optimizer_name="BFGS",
            cell_filter_name="frechet",
            fixed_cell=False,
            fmax=0.05,
            steps=1,
            force_limit=1.0e6,
            logfile=tmp_path / "relax.log",
            trajectory=tmp_path / "relax.traj",
        )
    except ValueError as error:
        assert "periodic boundaries" in str(error)
    else:
        raise AssertionError("nonperiodic cell relaxation should fail")


def test_import():
    """The vendored package imports under the OneScience namespace."""
    import onescience.models.nequip as nequip

    assert nequip.__version__


def test_training_entry_can_be_imported_first():
    """The split packages must not depend on model-first import ordering."""
    code = """
from onescience.utils.nequip.cli.train import main
from onescience.models.nequip.model import NequIPGNNModel
from onescience.datapipes.materials.nequip.datamodule import ASEDataModule
assert callable(main)
assert NequIPGNNModel is not None
assert ASEDataModule is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_set_global_state():
    """Global state initializes without error on e3nn 0.4.4."""
    from onescience.utils.nequip.internal.global_state import set_global_state

    set_global_state()


def test_build_model():
    """A small NequIP model can be instantiated."""
    from onescience.models.nequip.model.nequip_models import NequIPGNNModel
    from onescience.utils.nequip.internal.global_state import set_global_state

    set_global_state()
    model = NequIPGNNModel(
        seed=123,
        model_dtype="float32",
        type_names=["Cu"],
        num_layers=2,
        l_max=1,
        num_features=8,
        r_max=4.0,
        parity=False,
        avg_num_neighbors=10.0,
    )
    assert model is not None
    assert any(p.numel() > 0 for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_cuda():
    """The model runs a forward pass on CUDA."""
    from onescience.datapipes.materials.nequip import AtomicDataDict, from_ase
    from onescience.datapipes.materials.nequip._nl import compute_neighborlist_
    from onescience.datapipes.materials.nequip.transforms import (
        ChemicalSpeciesToAtomTypeMapper,
    )
    from onescience.models.nequip.model.nequip_models import NequIPGNNModel
    from onescience.utils.nequip.internal.global_state import set_global_state

    set_global_state()
    model = NequIPGNNModel(
        seed=123,
        model_dtype="float32",
        type_names=["Cu"],
        num_layers=2,
        l_max=1,
        num_features=8,
        r_max=4.0,
        parity=False,
        avg_num_neighbors=10.0,
    )

    atoms = bulk("Cu", "fcc", a=3.6)
    data = from_ase(atoms)
    data = ChemicalSpeciesToAtomTypeMapper(model_type_names=["Cu"])(data)
    data = compute_neighborlist_(data, r_max=4.0)

    device = torch.device("cuda")
    out = model.to(device)(AtomicDataDict.to_(data, device))
    assert AtomicDataDict.TOTAL_ENERGY_KEY in out
    assert AtomicDataDict.FORCE_KEY in out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ase_calculator():
    """The NequIP ASE calculator returns energy, forces, and stress on CUDA."""
    from onescience.models.nequip.model.nequip_models import NequIPGNNModel
    from onescience.utils.nequip import build_nequip_calculator
    from onescience.utils.nequip.internal.global_state import set_global_state

    set_global_state()
    model = NequIPGNNModel(
        seed=123,
        model_dtype="float32",
        type_names=["Cu"],
        num_layers=2,
        l_max=1,
        num_features=8,
        r_max=4.0,
        parity=False,
        avg_num_neighbors=10.0,
    )

    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.calc = build_nequip_calculator(
        model=model,
        r_max=4.0,
        device="cuda",
    )
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    assert isinstance(energy, float)
    assert forces.shape == (1, 3)
    assert stress.shape == (6,)


def test_rewrite_targets():
    """The target-rewriting helper maps upstream targets to OneScience."""
    from onescience.utils.nequip.internal.compat import rewrite_nequip_targets

    config = {
        "model": {"_target_": "nequip.model.NequIPGNNModel"},
        "transforms": [
            {"_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper"},
        ],
        "other_key": "nequip.value",
    }
    rewritten = rewrite_nequip_targets(config)
    assert rewritten["model"]["_target_"] == (
        "onescience.models.nequip.model.NequIPGNNModel"
    )
    assert rewritten["transforms"][0]["_target_"] == (
        "onescience.datapipes.materials.nequip.transforms.ChemicalSpeciesToAtomTypeMapper"
    )
    assert rewritten["other_key"] == "nequip.value"


def test_rewrite_pre_refactor_onescience_targets():
    """Checkpoints made before the layout refactor use the canonical paths."""
    from onescience.utils.nequip.internal.compat import rewrite_nequip_targets

    config = {
        "data": {
            "_target_": "onescience.models.nequip.data.datamodule.ASEDataModule"
        },
        "training_module": {
            "_target_": "onescience.models.nequip.train.EMALightningModule"
        },
    }

    rewritten = rewrite_nequip_targets(config)

    assert rewritten["data"]["_target_"] == (
        "onescience.datapipes.materials.nequip.datamodule.ASEDataModule"
    )
    assert rewritten["training_module"]["_target_"] == (
        "onescience.utils.nequip.train.EMALightningModule"
    )


def test_training_resume_rewrites_checkpoint_targets():
    """Training resume rewrites legacy targets before Lightning restoration."""
    from onescience.utils.nequip.cli.train import (
        _compatible_checkpoint_hyper_parameters,
    )

    checkpoint = {
        "hyper_parameters": {
            "model": {"_target_": "nequip.model.NequIPGNNModel"},
            "info_dict": {
                "training_module": {
                    "_target_": "onescience.models.nequip.train.EMALightningModule"
                }
            },
        }
    }

    compatible = _compatible_checkpoint_hyper_parameters(checkpoint)

    assert compatible["model"]["_target_"] == (
        "onescience.models.nequip.model.NequIPGNNModel"
    )
    assert compatible["info_dict"]["training_module"]["_target_"] == (
        "onescience.utils.nequip.train.EMALightningModule"
    )


def test_trainer_call_compatibility_for_weights_only():
    """Pass weights_only only when the installed Trainer API accepts it."""
    from onescience.utils.nequip.cli.train import (
        _run_with_suppressed_leafspec_warning,
    )

    captured = {}

    def legacy_trainer_call(model, datamodule, ckpt_path):
        captured["legacy"] = (model, datamodule, ckpt_path)

    _run_with_suppressed_leafspec_warning(
        legacy_trainer_call,
        model="model",
        datamodule="datamodule",
        ckpt_path=None,
        weights_only=False,
    )
    assert captured["legacy"] == ("model", "datamodule", None)

    def current_trainer_call(model, datamodule, ckpt_path, weights_only):
        captured["current"] = (model, datamodule, ckpt_path, weights_only)

    _run_with_suppressed_leafspec_warning(
        current_trainer_call,
        model="model",
        datamodule="datamodule",
        ckpt_path=None,
        weights_only=False,
    )
    assert captured["current"] == ("model", "datamodule", None, False)


def test_legacy_checkpoint_targets_are_used_for_loading(tmp_path, monkeypatch):
    """Legacy upstream targets are overridden during Lightning restoration."""
    from onescience.models.nequip.model.saved_models import checkpoint as checkpoint_module

    checkpoint_path = tmp_path / "upstream.ckpt"
    checkpoint_path.touch()
    checkpoint = {
        "hyper_parameters": {
            "model": {"_target_": "nequip.model.NequIPGNNModel"},
            "info_dict": {
                "versions": {},
                "training_module": {
                    "_target_": "nequip.train.EMALightningModule"
                },
            },
        }
    }
    captured = {}
    expected_model = object()

    class FakeLightningModule:
        evaluation_model = expected_model

        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(checkpoint_module.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        checkpoint_module.hydra.utils,
        "get_class",
        lambda target: FakeLightningModule,
    )

    model = checkpoint_module.ModelFromCheckpoint(str(checkpoint_path))

    assert model is expected_model
    assert captured["path"] == str(checkpoint_path)
    assert captured["weights_only"] is False
    assert captured["model"]["_target_"] == (
        "onescience.models.nequip.model.NequIPGNNModel"
    )
    assert captured["info_dict"]["training_module"]["_target_"] == (
        "onescience.utils.nequip.train.EMALightningModule"
    )


def test_package_dependency_namespaces():
    """PackageExporter rules cover vendored model classes and e3nn 0.4.x."""
    import e3nn

    from onescience.utils.nequip.cli._package_utils import (
        _EXTERNAL_MODULES,
        _INTERNAL_MODULES,
    )

    assert "onescience.models.nequip" in _INTERNAL_MODULES
    assert "onescience.datapipes.materials.nequip" in _INTERNAL_MODULES
    assert "onescience.utils.nequip" in _INTERNAL_MODULES
    assert "nequip" not in _INTERNAL_MODULES
    assert "sympy" in _EXTERNAL_MODULES
    if e3nn.__version__.startswith("0.4."):
        assert "e3nn" in _EXTERNAL_MODULES
        assert "e3nn" not in _INTERNAL_MODULES
