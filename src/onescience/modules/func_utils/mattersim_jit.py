# -*- coding: utf-8 -*-
"""jit.py is used to compile model with jit and modified from
    https://github.com/e3nn/e3nn/blob/main/e3nn/util/jit.py
"""

import copy
import inspect
import logging
import pathlib
import sys
import warnings
from typing import Dict, Final, Optional, Tuple, Union

import ase.data
import numpy as np
import torch
from opt_einsum_fx import jitable
from torch import fx

_RL4CSP_COMPILE_MODE = "__rl4csp_compile_mode__"
_VALID_MODES = ("trace", "script", "unsupported", None)
_MAKE_TRACING_INPUTS = "_make_tracing_inputs"


def compile_mode(mode: str):
    """Decorator to set the compile mode of a module.

    Parameters
    ----------
        mode : str
            'script', 'trace', or None
    """
    if mode not in _VALID_MODES:
        raise ValueError("Invalid compile mode")

    def decorator(obj):
        if not (inspect.isclass(obj) and issubclass(obj, torch.nn.Module)):
            raise TypeError(
                "@onescience.modules.func_utils.mattersim_jit.compile_mode can only "
                "decorate classes derived from torch.nn.Module"
            )
        setattr(obj, _RL4CSP_COMPILE_MODE, mode)
        return obj

    return decorator


def get_compile_mode(mod: torch.nn.Module) -> str:
    """Get the compilation mode of a module.

    Parameters
    ----------
        mod : torch.nn.Module

    Returns
    -------
    'script', 'trace', or None if the module was not decorated with
                       @compile_mode
    """
    if hasattr(mod, _RL4CSP_COMPILE_MODE):
        mode = getattr(mod, _RL4CSP_COMPILE_MODE)
    else:
        mode = getattr(type(mod), _RL4CSP_COMPILE_MODE, None)
    if mode is None and isinstance(mod, fx.GraphModule):
        mode = "script"
    assert mode in _VALID_MODES, "Invalid compile mode `%r`" % mode
    return mode


def compile(
    mod: torch.nn.Module,
    n_trace_checks: int = 1,
    script_options: dict = None,
    trace_options: dict = None,
    in_place: bool = True,
):
    """Recursively compile a module and all submodules according
       to their decorators.

    (Sub)modules without decorators will be unaffected.

    Parameters
    ----------
        mod : torch.nn.Module
            The module to compile. The module will have its submodules
            compiled replaced in-place.
        n_trace_checks : int, default = 1
            How many random example inputs to generate when tracing a module.
            Must be at least one in order to have a tracing input.
            Extra example inputs will be passed to ``torch.jit.trace``
            to confirm that the traced copmute graph doesn't change.
        script_options : dict, default = {}
            Extra kwargs for ``torch.jit.script``.
        trace_options : dict, default = {}
            Extra kwargs for ``torch.jit.trace``.

    Returns
    -------
    Returns the compiled module.
    :param trace_options:
    :param script_options:
    :param n_trace_checks:
    :param mod:
    :param in_place:
    """
    script_options = script_options or {}
    trace_options = trace_options or {}

    mode = get_compile_mode(mod)
    if mode == "unsupported":
        raise NotImplementedError(
            f"{type(mod).__name__} does not support TorchScript compilation"
        )

    if not in_place:
        mod = copy.deepcopy(mod)
    # TODO: debug logging
    assert n_trace_checks >= 1
    # == recurse to children ==
    # This allows us to trace compile submodules of modules we are going to
    # script
    for submod_name, submod in mod.named_children():
        setattr(
            mod,
            submod_name,
            compile(
                submod,
                n_trace_checks=n_trace_checks,
                script_options=script_options,
                trace_options=trace_options,
                in_place=True,
                # since we deepcopied the module above, we can do inplace
            ),
        )
    # == Compile this module now ==
    if mode == "script":
        if isinstance(mod, fx.GraphModule):
            mod = jitable(mod)
        mod = torch.jit.script(mod, **script_options)
    elif mode == "trace":
        # These are always modules, so we're always using trace_module
        # We need tracing inputs:
        check_inputs = get_tracing_inputs(
            mod,
            n_trace_checks,
        )
        assert len(check_inputs) >= 1, "Must have at least one tracing input."
        # Do the actual trace
        mod = torch.jit.trace_module(
            mod,
            inputs=check_inputs[0],
            check_inputs=check_inputs,
            **trace_options,  # noqa: E501
        )
    return mod


def get_tracing_inputs(
    mod: torch.nn.Module,
    n: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Get random tracing inputs for ``mod``.

    First checks if ``mod`` has a ``_make_tracing_inputs`` method.
    If so, calls it with ``n`` as the single argument and returns its results.

    Otherwise, attempts to infer the input signature of the module using
    ``e3nn.util._argtools._get_io_irreps``.

    Parameters
    ----------
        mod : torch.nn.Module
        n : int, default = 1
            A hint for how many inputs are wanted. Usually n will be returned,
            but modules don't necessarily have to.
        device : torch.device
            The device to do tracing on. If `None` (default), will be guessed.
        dtype : torch.dtype
            The dtype to trace with. If `None` (default), will be guessed.

    Returns
    -------
    list of dict
        Tracing inputs in the format of ``torch.jit.trace_module``:
        dicts mapping method names like ``'forward'`` to tuples of arguments.
    """
    # Avoid circular imports
    from e3nn.util._argtools import (
        _get_device,
        _get_floating_dtype,
        _get_io_irreps,
        _rand_args,
        _to_device_dtype,
    )

    # - Get inputs -
    if hasattr(mod, _MAKE_TRACING_INPUTS):
        # This returns a trace_module style dict of method names to test inputs
        trace_inputs = mod._make_tracing_inputs(n)
        assert isinstance(trace_inputs, list)
        for d in trace_inputs:
            assert isinstance(
                d, dict
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
            assert all(
                isinstance(k, str) and isinstance(v, tuple)
                for k, v in d.items()  # noqa: E501
            ), "_make_tracing_inputs must return a list of dict[str, tuple]"
    else:
        # Try to infer. This will throw if it can't.
        irreps_in, _ = _get_io_irreps(
            mod, irreps_out=[None]
        )  # we're only trying to infer inputs
        trace_inputs = [{"forward": _rand_args(irreps_in)} for _ in range(n)]
    # - Put them on the right device -
    if device is None:
        device = _get_device(mod)
    if dtype is None:
        dtype = _get_floating_dtype(mod)
    # Move them
    trace_inputs = _to_device_dtype(trace_inputs, device, dtype)
    return trace_inputs


def trace_module(
    mod: torch.nn.Module,
    inputs: dict = None,
    check_inputs: list = None,
    in_place: bool = True,
):
    """Trace a module.

    Identical signature to ``torch.jit.trace_module``, but first recursively
    compiles ``mod`` using ``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        inputs : dict
        check_inputs : list of dict
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []

    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _RL4CSP_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "trace":
        warnings.warn(
            f"Trying to trace a module of type {type(mod).__name__} marked "
            "with @compile_mode != 'trace', expect errors!"
        )
    setattr(mod, _RL4CSP_COMPILE_MODE, "trace")

    # If inputs are provided, set make_tracing_input temporarily
    old_make_tracing_input = None
    if inputs is not None:
        old_make_tracing_input = getattr(mod, _MAKE_TRACING_INPUTS, None)
        setattr(
            mod,
            _MAKE_TRACING_INPUTS,
            lambda num: ([inputs] + check_inputs),  # noqa: E501
        )

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _RL4CSP_COMPILE_MODE, old_mode)
    if old_make_tracing_input is not None:
        setattr(mod, _MAKE_TRACING_INPUTS, old_make_tracing_input)
    return out


def trace(
    mod: torch.nn.Module,
    example_inputs: tuple = None,
    check_inputs: list = None,
    in_place: bool = True,
):
    """Trace a module.

    Identical signature to ``torch.jit.trace``, but first recursively compiles
    ``mod`` using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
        example_inputs : tuple
        check_inputs : list of tuple
    Returns
    -------
    Traced module.
    """
    check_inputs = check_inputs or []

    return trace_module(
        mod=mod,
        inputs=(
            {"forward": example_inputs}
            if example_inputs is not None
            else None  # noqa: E501
        ),
        check_inputs=[{"forward": c} for c in check_inputs],
        in_place=in_place,
    )


def script(mod, in_place: bool = True):
    """Script a module.

    Like ``torch.jit.script``, but first recursively compiles ``mod``
    using :func:``compile``.

    Parameters
    ----------
        mod : torch.nn.Module
    Returns
    -------
    Scripted module.
    """
    # Set the compile mode for mod, temporarily
    old_mode = getattr(mod, _RL4CSP_COMPILE_MODE, None)
    if old_mode is not None and old_mode != "script":
        warnings.warn(
            f"Trying to script a module of type {type(mod).__name__} marked "
            "with @compile_mode != 'script', expect errors! "
        )
    setattr(mod, _RL4CSP_COMPILE_MODE, "script")

    # Compile
    out = compile(mod, in_place=in_place)

    # Restore old values, if we had them
    if old_mode is not None:
        setattr(mod, _RL4CSP_COMPILE_MODE, old_mode)

    return out
TWO_BODY_CUTOFF: Final[str] = "two_body_cutoff"
HAS_THREE_BODY: Final[str] = "has_three_body"
THREE_BODY_CUTOFF: Final[str] = "three_body_cutoff"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"

_ALL_METADATA_KEYS = [
    TWO_BODY_CUTOFF,
    HAS_THREE_BODY,
    THREE_BODY_CUTOFF,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_BAILOUT_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
]


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        print("Non TorchScript model detected,JIT  compiling the model ....")
        model = script(model)
    else:
        print(
            "Model provided is already a TorchScript model, "
            "return as it is."  # noqa: E501
        )
    return model


def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.
    Args:
        model_path: the path to the deployed model's ``.pth`` file.
    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    try:
        model = torch.jit.load(
            model_path, map_location=device, _extra_files=metadata
        )  # noqa: E501
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed RL4CSP model file. "
            f"Did you forget to deploy it? \n\n(Underlying error: {e})"
        )

    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)

    # Make sure we're in eval mode
    model.eval()
    # Freeze on load:
    if freeze and hasattr(model, "training"):
        # hasattr is how torch checks whether model is unfrozen
        # only freeze if already unfrozen
        model = torch.jit.freeze(model)

    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}

    # JIT strategy
    strategy = metadata.get(JIT_FUSION_STRATEGY, "")

    if strategy != "":
        strategy = [e.split(",") for e in strategy.split(";")]
        strategy = [(e[0], int(e[1])) for e in strategy]
    else:
        print(
            "Missing information: JIT strategy, "
            "loading deployed model fails !"  # noqa: E501
        )
        exit()

    # JIT bailout
    jit_bailout: int = metadata.get(JIT_BAILOUT_KEY, "")
    if jit_bailout == "":
        print(
            "Missing information: JIT_BAILOUT_KEY, "
            "loading deployed model fails !"  # noqa: E501
        )
        exit()

    # JIT allow_tf32
    jit_allow_tf32: int = metadata.get(TF32_KEY, "")
    if jit_allow_tf32 == "":
        print("Missing information: TF32_KEY, loading deployed model fails !")
        exit()

    return model, metadata


def deploy(
    model,
    is_m3gnet_pretrained=False,
    is_m3gnet_multi_head_pretrained=False,
    metadata=None,
    deployed_model_name="deployed.pth",
    device="cpu",
):
    # Compile model
    complied_model = _compile_for_deploy(model)

    # Use default metadata dictionary for pretrained models
    if is_m3gnet_pretrained:
        metadata = {}

        # Do set differences get atomic numbers
        full_atomic_numbers = set(np.arange(1, 95, 1))
        discard_atomic_numbers = set(np.arange(84, 89, 1))
        covered_atomic_numbers = list(
            full_atomic_numbers.difference(discard_atomic_numbers)
        )
        type_names = []
        for atomic_num in covered_atomic_numbers:
            type_names.append(ase.data.chemical_symbols[atomic_num])
        metadata[TWO_BODY_CUTOFF] = str(5.0)
        metadata[HAS_THREE_BODY] = str(True)
        metadata[THREE_BODY_CUTOFF] = str(4.0)
        metadata[N_SPECIES_KEY] = str(89)
        metadata[TYPE_NAMES_KEY] = " ".join(type_names)
        metadata[JIT_BAILOUT_KEY] = str(2)
        metadata[JIT_FUSION_STRATEGY] = ";".join(
            "%s,%i" % e for e in [("DYNAMIC", 3)]  # noqa: E501
        )
        metadata[TF32_KEY] = str(int(0))

    # TODO: Add default meta keys for m3gent_multi_head models
    # elif is_m3gnet_multi_head_pretrained:

    else:
        # Missing fields in meta data triggers failing compilation
        metadata_keys = metadata.keys
        for _ALL_METADATA_KEY in _ALL_METADATA_KEYS:
            if _ALL_METADATA_KEY not in metadata_keys:
                logging.info(
                    "Miss metadata key: "
                    + _ALL_METADATA_KEY
                    + " model deploying fails!"
                )
                exit()
        # Missing metadata values, other than JIT compile information,
        # triggers failing compilation
        for i in range(len(metadata_keys) - 3):
            if metadata[metadata_keys[i]].empty():
                logging.info(
                    "metadata with key "
                    + metadata_keys
                    + "not set, model deploying fails!"
                )
                exit()
        # Set default JIT compile information is values are not set
        if (
            metadata["JIT_BAILOUT_KEY"].empty()
            or metadata[JIT_FUSION_STRATEGY].empty()
            or metadata[TF32_KEY].empty()
        ):
            metadata[JIT_BAILOUT_KEY] = str(2)
            metadata[JIT_FUSION_STRATEGY] = ";".join(
                "%s,%i" % e for e in [("DYNAMIC", 3)]
            )
            metadata[TF32_KEY] = str(int(0))

    # Deploy model with full information
    # Confirm its TorchScript
    assert isinstance(complied_model, torch.jit.ScriptModule)
    if device != "cuda":
        complied_model = complied_model.cpu()

    torch.jit.save(complied_model, deployed_model_name, _extra_files=metadata)

    return complied_model, metadata
