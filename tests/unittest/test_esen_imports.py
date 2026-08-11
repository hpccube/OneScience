def test_esen_public_imports():
    from onescience.models.esen import eSEN_Backbone, eSEN_DeNS_Backbone
    from onescience.utils.esen import ESENCheckpointTransforms, eSENCalculator

    assert eSEN_Backbone.__module__.startswith("onescience.models.esen")
    assert eSEN_DeNS_Backbone.__module__.startswith("onescience.models.esen")
    assert eSENCalculator.__module__ == "onescience.utils.esen.calculator"
    assert ESENCheckpointTransforms.__module__ == "onescience.utils.esen.checkpoint"


def test_esen_checkpoint_transforms_round_trip():
    from types import SimpleNamespace

    import torch

    from onescience.utils.esen import ESENCheckpointTransforms
    from onescience.utils.uma.normalization.element_references import LinearReferences
    from onescience.utils.uma.normalization.normalizer import Normalizer

    references = torch.zeros(119)
    references[8] = -4.0
    references[14] = -5.0
    transforms = ESENCheckpointTransforms(
        normalizers={
            "energy": Normalizer(mean=0.0, rmsd=2.0),
            "forces": Normalizer(mean=0.0, rmsd=0.5),
        },
        elementrefs={"energy": LinearReferences(references)},
    )
    batch = SimpleNamespace(
        natoms=torch.tensor([2]),
        atomic_numbers=torch.tensor([8, 14]),
        batch=torch.tensor([0, 0]),
    )

    physical_energy = torch.tensor([-7.0])
    model_energy = torch.tensor([[1.0]])
    normalized_energy = transforms.normalize_target(
        "energy", physical_energy, model_energy, batch
    )
    assert normalized_energy.shape == model_energy.shape
    assert torch.equal(normalized_energy, model_energy)
    assert torch.equal(
        transforms.denormalize_prediction("energy", normalized_energy, batch),
        physical_energy.reshape_as(model_energy),
    )

    physical_forces = torch.tensor([[0.5, -0.5, 0.0], [1.0, 0.0, -1.0]])
    model_forces = physical_forces / 0.5
    normalized_forces = transforms.normalize_target(
        "forces", physical_forces, model_forces, batch
    )
    assert torch.equal(normalized_forces, model_forces)
    assert torch.equal(
        transforms.denormalize_prediction("forces", model_forces, batch),
        physical_forces,
    )


def test_uma_jd_resolution_ignores_esen_override(monkeypatch, tmp_path):
    from onescience.modules.func_utils.uma_path_utils import resolve_jd_path

    esen_jd = tmp_path / "esen_Jd.pt"
    uma_jd = tmp_path / "uma_Jd.pt"
    esen_jd.touch()
    uma_jd.touch()
    monkeypatch.setenv("ONESCIENCE_ESEN_JD_PATH", str(esen_jd))
    monkeypatch.setenv("ONESCIENCE_UMA_JD_PATH", str(uma_jd))

    assert resolve_jd_path() == str(uma_jd)
