"""MatterSim model and integration entry points."""

MATTERSIM_SOURCE_VERSION = "1.2.3"
MATTERSIM_INTEGRATION_VERSION = "dcu2"

from .adapter import (
    DEFAULT_CHECKPOINT,
    load_calculator,
    load_potential,
    predict_structures,
    resolve_checkpoint,
)
__all__ = [
    "DEFAULT_CHECKPOINT",
    "M3Gnet",
    "MATTERSIM_INTEGRATION_VERSION",
    "MATTERSIM_SOURCE_VERSION",
    "MatterSim",
    "load_calculator",
    "load_potential",
    "predict_structures",
    "resolve_checkpoint",
]


def __getattr__(name):
    if name in {"M3Gnet", "MatterSim"}:
        from .mattersim import M3Gnet, MatterSim

        return {"M3Gnet": M3Gnet, "MatterSim": MatterSim}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
