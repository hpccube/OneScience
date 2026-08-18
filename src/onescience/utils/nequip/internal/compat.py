"""Compatibility helpers for vendored NequIP in OneScience."""

from __future__ import annotations

from typing import Any, Dict


_TARGET_PREFIXES = (
    ("onescience.models.nequip.data", "onescience.datapipes.materials.nequip"),
    ("onescience.models.nequip.train", "onescience.utils.nequip.train"),
    ("onescience.models.nequip.scripts", "onescience.utils.nequip.cli"),
    ("onescience.models.nequip.integrations", "onescience.utils.nequip.integrations"),
    ("onescience.models.nequip.ase", "onescience.utils.nequip.ase"),
    ("onescience.models.nequip.utils", "onescience.utils.nequip.internal"),
    ("nequip.data", "onescience.datapipes.materials.nequip"),
    ("nequip.train", "onescience.utils.nequip.train"),
    ("nequip.scripts", "onescience.utils.nequip.cli"),
    ("nequip.integrations", "onescience.utils.nequip.integrations"),
    ("nequip.ase", "onescience.utils.nequip.ase"),
    ("nequip.utils", "onescience.utils.nequip.internal"),
    ("nequip", "onescience.models.nequip"),
)


def _rewrite_target(target: str) -> str:
    for old_prefix, new_prefix in _TARGET_PREFIXES:
        if target == old_prefix or target.startswith(f"{old_prefix}."):
            return new_prefix + target[len(old_prefix) :]
    return target


def rewrite_nequip_targets(obj: Any) -> Any:
    """Recursively rewrite ``_target_`` values in Hydra/OmegaConf structures.

    This allows checkpoints and YAML files produced by the upstream ``nequip``
    package to be loaded through the OneScience vendored package namespace.
    """
    if isinstance(obj, dict):
        result: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "_target_" and isinstance(value, str):
                value = _rewrite_target(value)
            else:
                value = rewrite_nequip_targets(value)
            result[key] = value
        return result
    elif isinstance(obj, list):
        return [rewrite_nequip_targets(item) for item in obj]
    return obj
