"""State single-cell data loading and preprocessing exports."""

from importlib import import_module


_EXPORTS = {
    "H5adSentenceDataset": "onescience.datapipes.state.embedding.loader",
    "VCIDatasetSentenceCollator": "onescience.datapipes.state.embedding.loader",
    "create_dataloader": "onescience.datapipes.state.embedding.loader",
    "scGPTPerturbationDataset": "onescience.datapipes.state.transition.scgpt_perturbation_dataset",
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
