"""Fine-tune MatterGen with a pretrained checkpoint and property adapter."""

import json
import os
from datetime import datetime
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.cli import SaveConfigCallback

from onescience.models.mattergen.diffusion.run import (
    AddConfigCallback,
    SimpleParser,
    maybe_instantiate,
)
from onescience.utils.mattergen.finetune import (
    init_adapter_lightningmodule_from_pretrained,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "OUTPUT_DIR",
    str(EXAMPLE_DIR / "outputs" / "finetune" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
)


@hydra.main(
    config_path=str(EXAMPLE_DIR / "conf"),
    config_name="finetune",
    version_base="1.1",
)
def main(cfg: omegaconf.DictConfig) -> None:
    """Build the data, adapter model, and Trainer, then start fine-tuning."""
    torch.set_float32_matmul_precision("high")
    trainer: pl.Trainer = maybe_instantiate(cfg.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(
        cfg.data_module, pl.LightningDataModule
    )

    model, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(
        cfg.adapter, cfg.lightning_module
    )
    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(resolved_config, indent=4))
    trainer.callbacks.append(
        SaveConfigCallback(
            parser=SimpleParser(),
            config=resolved_config,
            overwrite=True,
        )
    )
    trainer.callbacks.append(AddConfigCallback(resolved_config))
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
