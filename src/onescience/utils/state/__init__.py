"""State training, checkpoint, and callback utilities."""

from importlib import import_module


_EXPORTS = {
    "get_checkpoint_callbacks": "onescience.utils.state.transition",
    "get_loggers": "onescience.utils.state.transition",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
