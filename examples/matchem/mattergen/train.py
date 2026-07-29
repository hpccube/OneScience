"""Train MatterGen with the OneScience materials stack."""

import os
from datetime import datetime
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import OmegaConf

from onescience.models.mattergen.diffusion.config import Config
from onescience.models.mattergen.diffusion.run import main as run_training

EXAMPLE_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "OUTPUT_DIR",
    str(EXAMPLE_DIR / "outputs" / "train" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
)


@hydra.main(
    config_path=str(EXAMPLE_DIR / "conf"),
    config_name="default",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig) -> None:
    """Resolve the Hydra config and run MatterGen training."""
    torch.set_float32_matmul_precision("high")
    config = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    OmegaConf.set_readonly(config, True)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    run_training(config)


if __name__ == "__main__":
    main()
