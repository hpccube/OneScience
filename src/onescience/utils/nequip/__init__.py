"""NequIP training, integration, and checkpoint utilities for OneScience."""

__all__ = [
    "NequIPCalculatorBuilder",
    "build_nequip_calculator",
    "load_nequip_model_for_ase",
]


def __getattr__(name):
    if name in __all__:
        from .calculator import (
            NequIPCalculatorBuilder,
            build_nequip_calculator,
            load_nequip_model_for_ase,
        )

        return {
            "NequIPCalculatorBuilder": NequIPCalculatorBuilder,
            "build_nequip_calculator": build_nequip_calculator,
            "load_nequip_model_for_ase": load_nequip_model_for_ase,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
