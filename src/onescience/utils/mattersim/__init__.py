"""MatterSim force-field, application, and training utilities."""

from onescience.models.mattersim import load_calculator

from .trainer import FineTuneConfig, MatterSimTrainer, finetune


def relax(atoms, checkpoint=None, device=None, **kwargs):
    """Attach MatterSim and run its native structure relaxer.

    ``optimizer``, ``filter``, ``constrain_symmetry``, and ``fix_axis`` configure
    the relaxer. Remaining keyword arguments are forwarded to ``Relaxer.relax``.
    """
    from .relax import Relaxer

    relaxer_keys = {"optimizer", "filter", "constrain_symmetry", "fix_axis"}
    relaxer_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in relaxer_keys}
    atoms.calc = load_calculator(checkpoint=checkpoint, device=device)
    return Relaxer(**relaxer_kwargs).relax(atoms, **kwargs)


def molecular_dynamics(atoms, checkpoint=None, device=None, **kwargs):
    """Attach MatterSim and create its native ASE molecular-dynamics wrapper."""
    from .molecular_dynamics import MolecularDynamics

    atoms.calc = load_calculator(checkpoint=checkpoint, device=device)
    return MolecularDynamics(atoms, **kwargs)


__all__ = [
    "AtomsAdaptor",
    "BatchRelaxer",
    "FineTuneConfig",
    "MatterSimCalculator",
    "MatterSimTrainer",
    "MolecularDynamics",
    "Potential",
    "Relaxer",
    "batch_to_dict",
    "download_checkpoint",
    "download_file",
    "finetune",
    "load_calculator",
    "molecular_dynamics",
    "relax",
]


def __getattr__(name):
    exports = {
        "AtomsAdaptor": (".atoms", "AtomsAdaptor"),
        "BatchRelaxer": (".relax", "BatchRelaxer"),
        "MatterSimCalculator": (".calculator", "MatterSimCalculator"),
        "MolecularDynamics": (".molecular_dynamics", "MolecularDynamics"),
        "Potential": (".potential", "Potential"),
        "Relaxer": (".relax", "Relaxer"),
        "batch_to_dict": (".potential", "batch_to_dict"),
        "download_checkpoint": (".checkpoint", "download_checkpoint"),
        "download_file": (".checkpoint", "download_file"),
    }
    if name in exports:
        from importlib import import_module

        module_name, attribute = exports[name]
        return getattr(import_module(module_name, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
