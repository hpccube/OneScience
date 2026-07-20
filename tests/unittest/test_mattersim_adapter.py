from pathlib import Path

import pytest
import torch
from setuptools import find_packages

from onescience.models.mattersim import (
    DEFAULT_CHECKPOINT,
    MATTERSIM_INTEGRATION_VERSION,
    MATTERSIM_SOURCE_VERSION,
    resolve_checkpoint,
)
from onescience.utils.mattersim import FineTuneConfig, MatterSimTrainer, finetune


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_mattersim_model_and_versions_use_onescience_paths():
    model_path = REPOSITORY_ROOT / "src/onescience/models/mattersim/mattersim.py"
    assert model_path.is_file()
    assert "class M3Gnet" in model_path.read_text()
    assert MATTERSIM_SOURCE_VERSION == "1.2.3"
    assert MATTERSIM_INTEGRATION_VERSION == "dcu2"


def test_setup_package_discovery_includes_mattersim_datapipe():
    packages = find_packages(str(REPOSITORY_ROOT / "src"))
    assert "mattersim" not in packages
    assert "onescience.models.mattersim" in packages
    assert "onescience.datapipes.materials.mattersim" in packages
    assert (
        REPOSITORY_ROOT
        / "src/onescience/datapipes/materials/mattersim/threebody_indices.pyx"
    ).is_file()
    package_config = (
        REPOSITORY_ROOT
        / "src/onescience/datapipes/materials/mattersim/package_config.py"
    ).read_text()
    assert "onescience.datapipes.materials.mattersim.threebody_indices" in package_config


def test_mattersim_runtime_exports_resolve_from_onescience():
    pytest.importorskip("torch_runstats")
    from onescience.utils.mattersim import MatterSimCalculator, Potential, Relaxer

    assert MatterSimCalculator.__module__ == "onescience.utils.mattersim.calculator"
    assert Potential.__module__ == "onescience.utils.mattersim.potential"
    assert Relaxer.__module__ == "onescience.utils.mattersim.relax"


def test_explicit_checkpoint_is_preserved(tmp_path):
    checkpoint = tmp_path / "custom.pth"
    assert resolve_checkpoint(checkpoint) == str(checkpoint)


def test_shared_checkpoint_is_preferred(monkeypatch, tmp_path):
    checkpoint = tmp_path / "mattersim" / DEFAULT_CHECKPOINT
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.setenv("ONESCIENCE_MODELS_DIR", str(tmp_path))

    assert resolve_checkpoint() == str(checkpoint)


def test_native_checkpoint_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("ONESCIENCE_MODELS_DIR", str(tmp_path))
    assert resolve_checkpoint() == DEFAULT_CHECKPOINT


def test_finetune_maps_onescience_config_to_native_entry(monkeypatch, tmp_path):
    checkpoint = tmp_path / "input.pth"
    output = tmp_path / "output"
    captured = {}

    def fake_main(args):
        captured.update(vars(args))
        torch.save(
            {"last_epoch": 1, "validation_metrics": {"loss": 0.25}},
            Path(args.save_path) / "last_model.pth",
        )

    monkeypatch.setattr(
        "onescience.utils.mattersim.trainer._native_finetune_main", fake_main
    )
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    result = MatterSimTrainer(
        FineTuneConfig(
            train_data_path="train.xyz",
            checkpoint=checkpoint,
            save_path=output,
            epochs=2,
            batch_size=16,
            device="cpu",
        )
    ).fit()

    assert captured["load_model_path"] == str(checkpoint)
    assert captured["train_data_path"] == "train.xyz"
    assert captured["save_checkpoint"] is True
    assert "checkpoint" not in captured
    assert result["last_epoch"] == 1
    assert result["metrics"]["loss"] == 0.25
    assert result["last_checkpoint"] == str(output / "last_model.pth")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "cuda:0"}, "device"),
        ({"cutoff": 4.0, "threebody_cutoff": 5.0}, "threebody_cutoff"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_finetune_rejects_invalid_config(overrides, message):
    config = FineTuneConfig(train_data_path="train.xyz", **overrides)
    with pytest.raises(ValueError, match=message):
        finetune(config)


def test_finetune_rejects_config_and_keyword_arguments():
    config = FineTuneConfig(train_data_path="train.xyz")
    with pytest.raises(TypeError, match="either FineTuneConfig"):
        finetune(config, epochs=2)
