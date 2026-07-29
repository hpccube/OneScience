"""Boltz structure-data parsing, featurization, and loading pipelines."""

from importlib import import_module


_EXPORTS = {
    "BoltzInferenceDataModule": (
        "onescience.datapipes.boltz.module.inference",
        "BoltzInferenceDataModule",
    ),
    "Boltz2InferenceDataModule": (
        "onescience.datapipes.boltz.module.inferencev2",
        "Boltz2InferenceDataModule",
    ),
    "BoltzTrainingDataModule": (
        "onescience.datapipes.boltz.module.trainingv2",
        "BoltzTrainingDataModule",
    ),
    "parse_a3m": ("onescience.datapipes.boltz.parse.a3m", "parse_a3m"),
    "parse_boltz_schema": (
        "onescience.datapipes.boltz.parse.schema",
        "parse_boltz_schema",
    ),
    "parse_fasta": ("onescience.datapipes.boltz.parse.fasta", "parse_fasta"),
    "parse_mmcif": ("onescience.datapipes.boltz.parse.mmcif", "parse_mmcif"),
    "parse_pdb": ("onescience.datapipes.boltz.parse.pdb", "parse_pdb"),
    "parse_yaml": ("onescience.datapipes.boltz.parse.yaml", "parse_yaml"),
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
