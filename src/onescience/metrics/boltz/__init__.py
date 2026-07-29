"""Evaluation utilities for Boltz structure and affinity predictions."""

from importlib import import_module


_EXPORTS = {
    "compute_boltz_metrics": (
        "onescience.metrics.boltz.aggregate_evals",
        "compute_boltz_metrics",
    ),
    "evaluate_structure": (
        "onescience.metrics.boltz.run_evals",
        "evaluate_structure",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
