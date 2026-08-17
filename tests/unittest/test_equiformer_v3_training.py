"""Focused tests for the standalone Equiformer V3 training contract."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from onescience.utils.equiformer_v3 import EquiformerV3CheckpointTransforms
from onescience.utils.uma.normalization.normalizer import create_normalizer


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO_ROOT / "examples" / "matchem" / "equiformer_v3" / "train.py"
EQUIFORMER_V3_EXAMPLE_DIR = TRAIN_PATH.parent
CONFIG_DIR = (
    REPO_ROOT / "examples" / "matchem" / "equiformer_v3" / "demo" / "configs"
)
FULL_SMOKE_CONFIG_PAIRS = {
    "oc20_scratch_8dcu.yaml": "oc20_scratch_8dcu_smoke.yaml",
}
SPEC = importlib.util.spec_from_file_location("equiformer_v3_train", TRAIN_PATH)
training = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)

sys.path.insert(0, str(EQUIFORMER_V3_EXAMPLE_DIR))
EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "equiformer_v3_evaluate", EQUIFORMER_V3_EXAMPLE_DIR / "evaluate.py"
)
evaluation = importlib.util.module_from_spec(EVALUATE_SPEC)
assert EVALUATE_SPEC.loader is not None
sys.modules[EVALUATE_SPEC.name] = evaluation
EVALUATE_SPEC.loader.exec_module(evaluation)

CONVERTER_PATH = (
    REPO_ROOT
    / "examples"
    / "matchem"
    / "equiformer_v3"
    / "prepare_mptrj_dataset.py"
)
CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "equiformer_v3_mptrj_converter", CONVERTER_PATH
)
converter = importlib.util.module_from_spec(CONVERTER_SPEC)
assert CONVERTER_SPEC.loader is not None
sys.modules[CONVERTER_SPEC.name] = converter
CONVERTER_SPEC.loader.exec_module(converter)


def _ddp_loss_worker(
    rank: int, world_size: int, init_file: str, result_file: str
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from onescience.modules.loss.uma_loss import DDPLoss

        atom_count = (2, 5)[rank]
        scale = torch.nn.Parameter(torch.tensor(1.0))
        prediction = scale * torch.ones(atom_count, 3)
        target = torch.zeros_like(prediction)
        loss = DDPLoss("l2mae", reduction="mean")(
            prediction, target, torch.tensor([atom_count])
        )
        loss.backward()
        averaged_gradient = scale.grad.detach().clone()
        torch.distributed.all_reduce(averaged_gradient)
        averaged_gradient /= world_size
        if rank == 0:
            torch.save(
                {
                    "local_loss": loss.detach(),
                    "averaged_gradient": averaged_gradient,
                },
                result_file,
            )
    finally:
        torch.distributed.destroy_process_group()


def _metric_reduction_worker(
    rank: int, world_size: int, init_file: str, result_file: str
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        sums = {"loss": ((2.0, 4.0)[rank], 1)}
        if rank == 1:
            sums["normalized_denoising_pos_l2mae"] = (0.25, 1)
        metrics = training._reduce_metrics(
            sums,
            torch.device("cpu"),
            training.DistributedContext(rank=rank, world_size=world_size),
        )
        if rank == 0:
            torch.save(metrics, result_file)
    finally:
        torch.distributed.destroy_process_group()


def _yaml_config(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_evaluation_error_accumulator_matches_concatenated_metrics() -> None:
    accumulator = evaluation.ErrorAccumulator()
    accumulator.update(torch.tensor([-1.0, 2.0]))
    accumulator.update(torch.tensor([3.0]))

    assert accumulator.result() == {
        "mae": pytest.approx(2.0),
        "rmse": pytest.approx((14.0 / 3.0) ** 0.5),
    }


def test_evaluation_force_cosine_is_per_atom() -> None:
    prediction = torch.tensor([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])

    cosine = evaluation._force_cosine(prediction, target)

    assert torch.equal(cosine, torch.tensor([1.0, -1.0]))


def test_evaluation_force_magnitude_error_is_per_atom() -> None:
    prediction = torch.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    target = torch.tensor([[0.0, 3.0, 4.0], [0.0, -1.0, 0.0]])

    error = evaluation._force_magnitude_error(prediction, target)

    assert torch.equal(error, torch.tensor([0.0, 1.0]))


def test_evaluation_energy_force_threshold_matches_official_metric() -> None:
    successes = evaluation._energy_force_success(
        energy_error=torch.tensor([0.019, 0.021, 0.001]),
        force_error=torch.tensor(
            [
                [0.029, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.031, 0.0, 0.0],
            ]
        ),
        selected_natoms=torch.tensor([2, 1, 1]),
    )

    assert torch.equal(successes, torch.tensor([True, False, False]))


def test_evaluation_accumulators_reject_empty_metrics() -> None:
    with pytest.raises(ValueError, match="empty tensor"):
        evaluation.ErrorAccumulator().result()
    with pytest.raises(ValueError, match="empty tensor"):
        evaluation.MeanAccumulator().result()


@pytest.mark.parametrize(
    ("full_name", "smoke_name"), FULL_SMOKE_CONFIG_PAIRS.items()
)
def test_full_training_configs_have_bounded_smoke_pairs(
    full_name: str, smoke_name: str
) -> None:
    full = _yaml_config(full_name)
    smoke = _yaml_config(smoke_name)

    assert full["mode"] in training.MODE_ALIASES.values()
    assert smoke["mode"] in training.MODE_ALIASES.values()
    assert full["train"] and full["val"]
    assert smoke["train"] and smoke["val"]
    assert full["losses"] == smoke["losses"]
    assert full["optimizer"] == smoke["optimizer"]
    assert full["scheduler"] == smoke["scheduler"]
    assert full["amp"] is smoke["amp"]
    assert full["load_balancing"] == smoke["load_balancing"] == "atoms"
    assert full["load_balancing_on_error"] == "raise"
    assert smoke["load_balancing_on_error"] == "raise"
    assert full["launch"]["num_nodes"] == smoke["launch"]["num_nodes"]
    assert full["launch"]["num_gpus"] == smoke["launch"]["num_gpus"]
    assert full["slurm"]["partition"] == "hx1hdnormal01"
    assert smoke["slurm"]["partition"] == "hx1hdnormal01"

    assert "max_steps" not in full
    assert "max_train_samples" not in full
    assert "max_val_samples" not in full
    assert smoke["max_steps"] >= 1
    world_size = smoke["launch"]["num_nodes"] * smoke["launch"]["num_gpus"]
    assert smoke["max_train_samples"] >= world_size
    assert smoke["max_val_samples"] >= world_size
    assert not str(full["output"]).startswith("/tmp/")

def test_config_directory_contains_only_supported_oc20_pair() -> None:
    paired_configs = set(FULL_SMOKE_CONFIG_PAIRS) | set(
        FULL_SMOKE_CONFIG_PAIRS.values()
    )
    present_configs = {path.name for path in CONFIG_DIR.glob("*.yaml")}
    assert present_configs == paired_configs


def test_oc20_full_config_preserves_effective_batch_contract() -> None:
    oc20 = _yaml_config("oc20_scratch_8dcu.yaml")
    assert oc20["launch"]["num_gpus"] == 8
    assert oc20["seed"] == 0
    assert oc20["batch_size"] == 1
    assert oc20["eval_batch_size"] == 1
    assert oc20["grad_accumulation_steps"] == 8
    assert oc20["model"]["gradient_checkpointing_block_list"] == [1] * 8
    assert (
        oc20["launch"]["num_nodes"]
        * oc20["launch"]["num_gpus"]
        * oc20["batch_size"]
        * oc20["grad_accumulation_steps"]
        == 64
    )

def _mptrj_record(energy: float) -> dict:
    return {
        "structure": {
            "lattice": {
                "matrix": [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            },
            "sites": [
                {"species": [{"element": "H"}], "xyz": [0.0, 0.0, 0.0]},
                {"species": [{"element": "H"}], "xyz": [1.0, 0.0, 0.0]},
            ],
        },
        "uncorrected_total_energy": energy,
        "force": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        "stress": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_config_modes_and_legacy_checkpoint_alias() -> None:
    scratch = training._normalize_config({"model": {"name": "equiformer_v3"}})
    assert scratch["mode"] == "train_from_scratch"
    assert scratch["amp_dtype"] == "float16"
    assert scratch["log_every_n_steps"] == 0
    assert scratch["log_every_n_validation_batches"] == 0
    assert scratch["ddp_find_unused_parameters"] is False

    finetune = training._normalize_config({"checkpoint": "/models/model.pt"})
    assert finetune["mode"] == "init_from_checkpoint"
    assert finetune["initialization_checkpoint"] == "/models/model.pt"

    resume = training._normalize_config({"mode": "resume", "resume": "/run.pt"})
    assert resume["mode"] == "resume_training"

    target = training._normalize_config(
        {
            "initialization_checkpoint": "/models/omat24.pt",
            "transforms_checkpoint": "/models/target.pt",
        }
    )
    assert target["transforms_checkpoint"] == "/models/target.pt"

    bf16 = training._normalize_config({"amp": True, "amp_dtype": "bfloat16"})
    assert bf16["amp_dtype"] == "bfloat16"


def test_finetune_can_separate_model_and_target_transforms(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    torch.save(
        {
            "normalizers": {
                "energy": create_normalizer(mean=0.0, rmsd=4.98).state_dict()
            },
            "elementrefs": {},
        },
        source,
    )
    torch.save(
        {
            "normalizers": {
                "energy": create_normalizer(mean=0.0, rmsd=0.627).state_dict()
            },
            "elementrefs": {},
        },
        target,
    )
    transforms = training._training_transforms(
        {
            "transforms_checkpoint": str(target),
            "transforms": {
                "normalizers": {"forces": {"mean": 0.0, "rmsd": 0.5}},
            },
        },
        source,
    )
    assert float(transforms.normalizers["energy"].rmsd) == pytest.approx(0.627)
    assert float(transforms.normalizers["forces"].rmsd) == pytest.approx(0.5)


def test_resume_rejects_transform_overrides() -> None:
    with pytest.raises(ValueError, match="restores transforms"):
        training._validate_config(
            training._normalize_config(
                {
                    "mode": "resume_training",
                    "train": "train",
                    "val": "val",
                    "output": "out.pt",
                    "epochs": 1,
                    "batch_size": 1,
                    "eval_batch_size": 1,
                    "grad_accumulation_steps": 1,
                    "transforms_checkpoint": "target.pt",
                    "losses": {"energy": {"fn": "mae", "coefficient": 1}},
                }
            )
        )


def test_official_oc20_loss_and_free_atom_mask() -> None:
    batch = SimpleNamespace(
        energy=torch.tensor([0.0]),
        forces=torch.zeros(2, 3),
        natoms=torch.tensor([2]),
        fixed=torch.tensor([0, 1]),
    )
    prediction = {
        "energy": torch.tensor([2.0]),
        "forces": torch.tensor([[3.0, 4.0, 0.0], [100.0, 0.0, 0.0]]),
    }
    specs = training._loss_specs(
        {
            "losses": {
                "energy": {"fn": "mae", "coefficient": 4},
                "forces": {
                    "fn": "l2mae",
                    "coefficient": 100,
                    "free_atoms_only": True,
                },
            }
        }
    )
    loss, components = training._loss(
        prediction, batch, specs, EquiformerV3CheckpointTransforms()
    )
    assert loss.item() == pytest.approx(508.0)
    assert components == {"energy_mae": 2.0, "forces_l2mae": 5.0}


def test_loss_modules_are_the_shared_fairchem_ddp_reductions() -> None:
    specs = training._loss_specs(
        {
            "losses": {
                "energy": {"fn": "per_atom_mae", "coefficient": 20},
                "forces": {"fn": "l2mae", "coefficient": 20},
            }
        }
    )
    losses = training._loss_functions(specs)
    assert type(losses["energy"]).__name__ == "DDPLoss"
    assert type(losses["energy"].loss_fn).__name__ == "PerAtomMAELoss"
    assert type(losses["forces"].loss_fn).__name__ == "L2NormLoss"


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="Gloo distributed support is unavailable",
)
def test_ddp_loss_uses_global_atom_count_for_uneven_ranks(tmp_path: Path) -> None:
    result_file = tmp_path / "ddp_loss_result.pt"
    torch.multiprocessing.spawn(
        _ddp_loss_worker,
        args=(2, str(tmp_path / "gloo_init"), str(result_file)),
        nprocs=2,
        join=True,
    )
    result = torch.load(result_file, weights_only=True)
    expected_global_mean = 3.0**0.5
    assert result["local_loss"].item() == pytest.approx(
        2 * 2 * expected_global_mean / 7
    )
    assert result["averaged_gradient"].item() == pytest.approx(
        expected_global_mean
    )


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="Gloo distributed support is unavailable",
)
def test_metric_reduction_unifies_conditional_keys_across_ranks(
    tmp_path: Path,
) -> None:
    result_file = tmp_path / "metric_reduction_result.pt"
    torch.multiprocessing.spawn(
        _metric_reduction_worker,
        args=(2, str(tmp_path / "metric_gloo_init"), str(result_file)),
        nprocs=2,
        join=True,
    )
    result = torch.load(result_file, weights_only=True)
    assert result == {
        "loss": pytest.approx(3.0),
        "normalized_denoising_pos_l2mae": pytest.approx(0.25),
    }


def test_cosine_scheduler_has_warmup_and_minimum_factor() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = training._cosine_scheduler(
        optimizer,
        {
            "name": "cosine",
            "warmup_factor": 0.2,
            "warmup_epochs": 1.0,
            "lr_min_factor": 0.01,
        },
        steps_per_epoch=2,
        epochs=2,
        max_steps=None,
    )
    official_lambda = training.CosineLRLambda(2, 0.2, 4, 0.01)
    for step in range(6):
        assert scheduler.lr_lambdas[0](step) == pytest.approx(official_lambda(step))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.6)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_gradient_accumulation_uses_complete_official_windows() -> None:
    assert training._updates_per_epoch(2500, 4) == 625
    assert training._updates_per_epoch(33, 32) == 1
    with pytest.raises(ValueError, match="exceeds the number of training batches"):
        training._updates_per_epoch(31, 32)


def test_gradient_accumulation_defers_ddp_reduction_until_update_boundary() -> None:
    class FakeDDP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.no_sync_calls = 0

        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield

    model = FakeDDP()
    for microbatch in range(16):
        with training._gradient_sync_context(model, microbatch == 15):
            pass
    assert model.no_sync_calls == 15

    with training._gradient_sync_context(torch.nn.Linear(1, 1), False):
        pass


def test_progress_logging_is_periodic_and_includes_phase_end() -> None:
    assert not training._should_log_progress(9, 9, 156, 10)
    assert training._should_log_progress(10, 10, 156, 10)
    assert training._should_log_progress(156, 156, 156, 10)
    assert training._should_log_progress(3, 3, 156, 10, reached_limit=True)
    assert not training._should_log_progress(10, 10, 10, 0)


def test_progress_window_metrics_use_the_existing_metric_reduction() -> None:
    window = {}
    training._collect_batch_metrics(
        window,
        torch.tensor(2.0),
        {"energy_mae": 1.5},
        {"forces_l2mae": (3.0, 2)},
    )
    training._collect_batch_metrics(
        window,
        torch.tensor(4.0),
        {"energy_mae": 2.5},
        {"forces_l2mae": (5.0, 3)},
    )

    metrics = training._reduce_metrics(
        window, torch.device("cpu"), training.DistributedContext()
    )

    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["normalized_energy_mae"] == pytest.approx(2.0)
    assert metrics["forces_l2mae"] == pytest.approx(8.0 / 5.0)


def test_all_training_configs_enable_periodic_progress_logging() -> None:
    for name in set(FULL_SMOKE_CONFIG_PAIRS) | set(
        FULL_SMOKE_CONFIG_PAIRS.values()
    ):
        config = _yaml_config(name)
        assert config["log_every_n_steps"] == 10
        assert config["log_every_n_validation_batches"] == 100


def test_amp_overflow_does_not_count_as_an_optimizer_update() -> None:
    assert training._amp_step_succeeded(1024.0, 1024.0)
    assert training._amp_step_succeeded(1024.0, 2048.0)
    assert not training._amp_step_succeeded(1024.0, 512.0)


def test_metric_reduction_uses_total_numel_instead_of_batch_means() -> None:
    sums = {}
    training._collect_batch_metrics(
        sums,
        torch.tensor(2.0),
        {"energy_mae": 2.0},
        {"forces_mae": (3.0, 3)},
    )
    training._collect_batch_metrics(
        sums,
        torch.tensor(4.0),
        {"energy_mae": 4.0},
        {"forces_mae": (100.0, 1)},
    )
    metrics = training._reduce_metrics(
        sums, torch.device("cpu"), training.DistributedContext()
    )
    assert metrics["loss"] == pytest.approx(3.0)
    assert metrics["normalized_energy_mae"] == pytest.approx(3.0)
    assert metrics["forces_mae"] == pytest.approx(103.0 / 4.0)


def test_full_state_checkpoint_keeps_inference_and_resume_weights(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    transforms = EquiformerV3CheckpointTransforms()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = training._cosine_scheduler(
        optimizer,
        {"name": "cosine", "warmup_epochs": 0.0, "lr_min_factor": 0.01},
        steps_per_epoch=1,
        epochs=2,
        max_steps=None,
    )
    ema = training.ModelEMA(model, 0.9)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    output = tmp_path / "checkpoint.pt"
    config = {
        "mode": "train_from_scratch",
        "initialization_checkpoint": None,
        "resume": None,
    }
    training._save_checkpoint(
        output,
        model,
        transforms,
        {"name": "equiformer_v3"},
        config,
        optimizer,
        scheduler,
        ema,
        epoch=0,
        global_step=1,
        history=[{"epoch": 0}],
        source_document=None,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["training_state"] == {"epoch": 0, "global_step": 1}
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "ema_state_dict" in checkpoint
    assert not torch.equal(
        checkpoint["state_dict"]["weight"],
        checkpoint["training_state_dict"]["weight"],
    )
    assert output.with_name(output.name + ".history.json").is_file()


def test_mptrj_conversion_matches_upstream_order_and_metadata(tmp_path: Path) -> None:
    import json

    from onescience.datapipes.materials.custom_stack.storage.ase_datasets import (
        AseDBDataset,
    )

    source = tmp_path / "mptrj.json"
    source.write_text(
        json.dumps(
            {
                "outer-group-1": {
                    "sample-1": _mptrj_record(-1.0),
                    "sample-2": _mptrj_record(-2.0),
                },
                "outer-group-2": {
                    "sample-3": _mptrj_record(-3.0),
                    "sample-4": _mptrj_record(-4.0),
                    "sample-5": _mptrj_record(-5.0),
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "converted"
    converter.convert(
        source,
        output,
        cutoff=6.0,
        max_neighbors=1000,
        shards=3,
        progress_every=0,
    )

    shard_paths = sorted(output.glob("data_*.aselmdb"))
    assert [path.name for path in shard_paths] == [
        "data_00000.aselmdb",
        "data_00001.aselmdb",
        "data_00002.aselmdb",
    ]
    assert [len(converter.connect(str(path))) for path in shard_paths] == [2, 2, 1]

    dataset = AseDBDataset(
        {
            "src": str(output),
            "a2g_args": {
                "r_edges": False,
                "r_energy": True,
                "r_forces": True,
                "r_stress": True,
            },
        }
    )
    assert len(dataset) == 5
    atoms = [dataset.get_atoms(index) for index in range(len(dataset))]
    assert [item.info["sid"] for item in atoms] == [
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
        "sample-5",
    ]
    assert [item.get_potential_energy() for item in atoms] == pytest.approx(
        [-1.0, -2.0, -3.0, -4.0, -5.0]
    )
    metadata = converter.np.load(output / "metadata.npz")
    assert metadata["natoms"].tolist() == [len(item) for item in atoms]

    sample = dataset[0]
    assert sample.energy.shape == (1,)
    assert sample.forces.shape == (2, 3)
    assert sample.stress.shape == (1, 3, 3)


def test_mptrj_parallel_conversion_matches_serial_with_filtering(
    tmp_path: Path,
) -> None:
    import json

    from onescience.datapipes.materials.custom_stack.storage.ase_datasets import (
        AseDBDataset,
    )

    isolated = _mptrj_record(-99.0)
    isolated["structure"]["lattice"]["matrix"] = [
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 100.0],
    ]
    isolated["structure"]["sites"] = [
        {"species": [{"element": "H"}], "xyz": [50.0, 50.0, 50.0]}
    ]
    isolated["force"] = [[0.0, 0.0, 0.0]]

    source = tmp_path / "mptrj_parallel.json"
    source.write_text(
        json.dumps(
            {
                "group-a": {
                    "sample-1": _mptrj_record(-1.0),
                    "filtered": isolated,
                    "sample-2": _mptrj_record(-2.0),
                },
                "group-b": {
                    "sample-3": _mptrj_record(-3.0),
                    "sample-4": _mptrj_record(-4.0),
                    "sample-5": _mptrj_record(-5.0),
                },
            }
        ),
        encoding="utf-8",
    )

    outputs = []
    for workers in (1, 2):
        output = tmp_path / f"converted-{workers}"
        converter.convert(
            source,
            output,
            cutoff=6.0,
            max_neighbors=1000,
            shards=3,
            progress_every=0,
            workers=workers,
            batch_size=2,
        )
        outputs.append(output)

    serial_shards = sorted(outputs[0].glob("data_*.aselmdb"))
    parallel_shards = sorted(outputs[1].glob("data_*.aselmdb"))
    assert [path.name for path in serial_shards] == [
        path.name for path in parallel_shards
    ]
    assert [len(converter.connect(str(path))) for path in serial_shards] == [
        len(converter.connect(str(path))) for path in parallel_shards
    ] == [1, 2, 2]

    snapshots = []
    for output in outputs:
        dataset = AseDBDataset({"src": str(output), "a2g_args": {}})
        atoms = [dataset.get_atoms(index) for index in range(len(dataset))]
        snapshots.append(atoms)

    assert [atoms.info["sid"] for atoms in snapshots[0]] == [
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
        "sample-5",
    ]
    assert [atoms.info["sid"] for atoms in snapshots[1]] == [
        atoms.info["sid"] for atoms in snapshots[0]
    ]
    for serial, parallel in zip(*snapshots, strict=True):
        assert parallel.get_potential_energy() == pytest.approx(
            serial.get_potential_energy()
        )
        converter.np.testing.assert_allclose(
            parallel.get_forces(), serial.get_forces()
        )
        converter.np.testing.assert_allclose(
            parallel.calc.results["stress"], serial.calc.results["stress"]
        )

    serial_metadata = converter.np.load(outputs[0] / "metadata.npz")["natoms"]
    parallel_metadata = converter.np.load(outputs[1] / "metadata.npz")["natoms"]
    converter.np.testing.assert_array_equal(parallel_metadata, serial_metadata)


def test_mptrj_labels_match_upstream_conventions() -> None:
    atoms = converter._to_atoms("inner-id", _mptrj_record(-2.5))

    assert atoms.info == {"sid": "inner-id"}
    assert atoms.get_potential_energy() == pytest.approx(-2.5)
    converter.np.testing.assert_allclose(
        atoms.get_forces(),
        [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
    )
    assert atoms.calc.results["free_energy"] == pytest.approx(-2.5)
    converter.np.testing.assert_allclose(
        atoms.calc.results["stress"],
        converter.np.eye(3) * (-0.1 * converter.ase.units.GPa),
    )


def test_mptrj_conversion_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        converter.convert(
            tmp_path / "unused.json",
            output,
            cutoff=6.0,
            max_neighbors=1000,
            shards=1,
        )


if __name__ == "__main__":
    exit_code = pytest.main([str(Path(__file__).resolve())])
    if exit_code == pytest.ExitCode.OK:
        print("Equiformer V3 训练单元测试成功。")
    raise SystemExit(exit_code)
