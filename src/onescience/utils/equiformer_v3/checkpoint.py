"""Checkpoint loading and transforms for Equiformer V3."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from onescience.utils.uma.common.registry import registry
from onescience.utils.uma.common.utils import load_state_dict, match_state_dict
from onescience.utils.uma.normalization.element_references import (
    create_element_references,
)
from onescience.utils.uma.normalization.normalizer import create_normalizer

if TYPE_CHECKING:
    from onescience.datapipes.materials.custom_stack.core.atomic_data import AtomicData


_TORCH_COMPILE_PREFIX = "_orig_mod."


def strip_torch_compile_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict:
    """Remove the wrapper prefix stored by ``torch.compile`` checkpoints."""

    return {
        key.removeprefix(_TORCH_COMPILE_PREFIX): value
        for key, value in state_dict.items()
    }


def load_equiformer_v3_checkpoint(
    checkpoint_path: str | Path,
    *,
    jd_path: str | Path | None = None,
) -> torch.nn.Module:
    """Construct an Equiformer V3 model and strictly load official weights."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # Ensure both official model names are registered for direct API callers.
    import onescience.models.equiformer_v3  # noqa: F401

    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    try:
        model_config = copy.deepcopy(checkpoint["config"]["model"])
        state_dict = checkpoint["state_dict"]
    except KeyError as error:
        raise ValueError(
            f"Invalid Equiformer V3 checkpoint {checkpoint_path}: missing {error!s}"
        ) from error

    model_name = model_config.pop("name", None)
    if model_name not in {"equiformer_v3", "equiformer_v3_dens"}:
        raise ValueError(
            f"Unsupported Equiformer V3 model name in {checkpoint_path}: "
            f"{model_name!r}"
        )
    if jd_path is not None:
        model_config["jd_path"] = str(jd_path)

    model = registry.get_model_class(model_name)(**model_config)
    state_dict = strip_torch_compile_prefix(state_dict)
    state_dict = match_state_dict(model.state_dict(), state_dict)
    load_state_dict(model, state_dict, strict=True)
    return model


class EquiformerV3CheckpointTransforms(nn.Module):
    """Move targets into and predictions out of Equiformer V3 value space."""

    def __init__(
        self, normalizers: dict | None = None, elementrefs: dict | None = None
    ):
        super().__init__()
        self.normalizers = nn.ModuleDict(normalizers or {})
        self.elementrefs = nn.ModuleDict(elementrefs or {})

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path
    ) -> EquiformerV3CheckpointTransforms:
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
        """Apply checkpoint element references and normalization to a target."""

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
