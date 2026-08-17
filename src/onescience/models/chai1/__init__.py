"""OneScience integration for Chai-1 biomolecular structure prediction."""

from importlib import import_module
from typing import Any

__all__ = [
    "StructureCandidates",
    "UnsupportedInputError",
    "make_all_atom_feature_context",
    "run_folding_on_context",
    "run_inference",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module(".chai1", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
