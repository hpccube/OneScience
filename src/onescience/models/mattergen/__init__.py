# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
__version__ = "1.0.0"
"""MatterGen crystal structure generation model."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import CrystalGenerator

__all__ = ["CrystalGenerator"]


def __getattr__(name: str):
    if name == "CrystalGenerator":
        from .generator import CrystalGenerator

        return CrystalGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
