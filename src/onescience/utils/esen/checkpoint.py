"""Normalization and element-reference transforms stored in eSEN checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from onescience.utils.uma.normalization.element_references import (
    create_element_references,
)
from onescience.utils.uma.normalization.normalizer import create_normalizer

if TYPE_CHECKING:
    from onescience.datapipes.materials.custom_stack.core.atomic_data import AtomicData


class ESENCheckpointTransforms(nn.Module):
    """Move targets into and predictions out of an eSEN model's value space."""

    def __init__(self, normalizers: dict | None = None, elementrefs: dict | None = None):
        super().__init__()
        self.normalizers = nn.ModuleDict(normalizers or {})
        self.elementrefs = nn.ModuleDict(elementrefs or {})

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> ESENCheckpointTransforms:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )
        normalizers = {
            name: create_normalizer(state_dict=state)
            for name, state in checkpoint.get("normalizers", {}).items()
        }
        elementrefs = {
            name: create_element_references(state_dict=state)
            for name, state in checkpoint.get("elementrefs", {}).items()
        }
        return cls(normalizers=normalizers, elementrefs=elementrefs)

    @staticmethod
    def _structure_shape(batch: AtomicData) -> tuple[int, int]:
        return int(batch.natoms.numel()), -1

    def normalize_target(
        self,
        name: str,
        target: torch.Tensor,
        prediction: torch.Tensor,
        batch: AtomicData,
    ) -> torch.Tensor:
        """Apply checkpoint element references and normalization to one target."""
        if target.numel() != prediction.numel():
            raise ValueError(
                f"{name} target and prediction sizes differ: "
                f"{tuple(target.shape)} != {tuple(prediction.shape)}"
            )
        value = target.reshape_as(prediction)
        if name in self.elementrefs:
            prediction_shape = value.shape
            value = value.reshape(self._structure_shape(batch))
            value = self.elementrefs[name].dereference(value, batch)
            value = value.reshape(prediction_shape)
        if name in self.normalizers:
            value = self.normalizers[name].norm(value)
        return value

    def denormalize_prediction(
        self, name: str, prediction: torch.Tensor, batch: AtomicData
    ) -> torch.Tensor:
        """Convert one model prediction to physical checkpoint units."""
        value = prediction
        if name in self.normalizers:
            value = self.normalizers[name].denorm(value)
        if name in self.elementrefs:
            prediction_shape = value.shape
            value = value.reshape(self._structure_shape(batch))
            value = self.elementrefs[name](value, batch)
            value = value.reshape(prediction_shape)
        return value
