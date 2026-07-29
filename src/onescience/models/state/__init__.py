"""State virtual-cell models integrated into OneScience."""

from importlib import import_module


_EXPORTS = {
    "StateEmbeddingModel": "onescience.models.state.embedding.model",
    "ContextMeanPerturbationModel": "onescience.models.state.transition.implementations.context_mean",
    "DecoderOnlyPerturbationModel": "onescience.models.state.transition.implementations.decoder_only",
    "EmbedSumPerturbationModel": "onescience.models.state.transition.implementations.embed_sum",
    "OldNeuralOTPerturbationModel": "onescience.models.state.transition.implementations.old_neural_ot",
    "PerturbationModel": "onescience.models.state.transition.implementations.base",
    "PerturbMeanPerturbationModel": "onescience.models.state.transition.implementations.perturb_mean",
    "PseudobulkPerturbationModel": "onescience.models.state.transition.implementations.pseudobulk",
    "StateTransitionPerturbationModel": "onescience.models.state.transition.implementations.state_transition",
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
