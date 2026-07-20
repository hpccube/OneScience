"""Reusable State neural building blocks."""

from importlib import import_module


_EXPORTS = {
    "FlashTransformerEncoder": "onescience.modules.state.embedding.flash_transformer",
    "FlashTransformerEncoderLayer": "onescience.modules.state.embedding.flash_transformer",
    "FinetuneVCICountsDecoder": "onescience.modules.state.transition.decoders",
    "KLDivergenceLoss": "onescience.modules.state.embedding.loss",
    "MMDLoss": "onescience.modules.state.embedding.loss",
    "StateEmbeddingModel": "onescience.modules.state.embedding.model",
    "TabularLoss": "onescience.modules.state.embedding.loss",
    "WassersteinLoss": "onescience.modules.state.embedding.loss",
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
