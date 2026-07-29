from pathlib import Path
from typing import Any

from pymatgen.core import Structure

from onescience.models.mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from onescience.models.mattergen.generator import CrystalGenerator


def generate_structures(
    checkpoint: str,
    output_dir: str = "outputs/mattergen",
    batch_size: int = 1,
    num_batches: int = 1,
    device: str | None = None,
    record_trajectories: bool = False,
    properties_to_condition_on: dict[str, Any] | None = None,
    target_compositions: list[dict[str, int]] | None = None,
) -> list[Structure]:
    """Generate crystal structures from a local MatterGen checkpoint.

    ``device`` is accepted for API consistency. MatterGen resolves the active
    OneScience DTK device from the runtime; callers should select DCUs with
    ``HIP_VISIBLE_DEVICES`` before starting the process.
    """
    del device
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(checkpoint).expanduser().resolve(),
        load_epoch="last",
    )
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=batch_size,
        num_batches=num_batches,
        properties_to_condition_on=properties_to_condition_on or {},
        target_compositions_dict=target_compositions or [],
        record_trajectories=record_trajectories,
    )
    return generator.generate(output_dir=resolved_output_dir)
