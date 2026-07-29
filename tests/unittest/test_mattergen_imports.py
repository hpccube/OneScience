from pathlib import Path


def test_mattergen_public_imports():
    from onescience.models.mattergen import CrystalGenerator
    from onescience.utils.mattergen import generate_structures

    assert CrystalGenerator.__module__.startswith("onescience.models.mattergen")
    assert generate_structures.__module__ == "onescience.utils.mattergen.generate"


def test_checkpoint_config_targets_are_rewritten():
    from onescience.models.mattergen.common.utils.data_classes import MatterGenCheckpointInfo

    checkpoint = Path("/public/home/yuxiaodong/mattergen/checkpoints/mattergen_base")
    if not checkpoint.exists():
        return
    cfg = MatterGenCheckpointInfo(checkpoint, "last").config
    assert cfg.lightning_module._target_.startswith("onescience.models.mattergen.")
