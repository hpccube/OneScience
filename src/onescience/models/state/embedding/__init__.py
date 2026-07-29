"""Public State Embedding model exports."""

from importlib import import_module


_EXPORTS = {
    "StateEmbeddingModel": "onescience.models.state.embedding.model",
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
