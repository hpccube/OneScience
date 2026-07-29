"""Boltz-1 and Boltz-2 models integrated into OneScience."""

from importlib import import_module


_EXPORTS = {
    "Boltz1": ("onescience.models.boltz.boltz1", "Boltz1"),
    "Boltz2": ("onescience.models.boltz.boltz2", "Boltz2"),
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
