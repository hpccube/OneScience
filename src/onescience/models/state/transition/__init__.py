"""Public State Transition model exports."""

from importlib import import_module


_EXPORTS = {
    "ContextMeanPerturbationModel": "onescience.modules.state.transition.context_mean",
    "DecoderOnlyPerturbationModel": "onescience.modules.state.transition.decoder_only",
    "EmbedSumPerturbationModel": "onescience.modules.state.transition.embed_sum",
    "OldNeuralOTPerturbationModel": "onescience.modules.state.transition.old_neural_ot",
    "PerturbationModel": "onescience.modules.state.transition.base",
    "PerturbMeanPerturbationModel": "onescience.modules.state.transition.perturb_mean",
    "PseudobulkPerturbationModel": "onescience.modules.state.transition.pseudobulk",
    "StateTransitionPerturbationModel": "onescience.modules.state.transition.state_transition",
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
