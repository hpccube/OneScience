"""MatterGen crystal dataset and datamodule."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import CrystDataModule
    from .dataset import CrystalDataset

__all__ = ["CrystalDataset", "CrystDataModule"]


def __getattr__(name: str):
    if name == "CrystalDataset":
        from .dataset import CrystalDataset

        return CrystalDataset
    if name == "CrystDataModule":
        from .datamodule import CrystDataModule

        return CrystDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
