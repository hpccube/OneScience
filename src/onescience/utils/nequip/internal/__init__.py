"""Internal NequIP helpers with cycle-safe lazy exports."""

from importlib import import_module


_EXPORTS = {
    "find_first_of_type": (".modules", "find_first_of_type"),
    "dtype_to_name": (".dtype", "dtype_to_name"),
    "dtype_from_name": (".dtype", "dtype_from_name"),
    "torch_default_dtype": (".dtype", "torch_default_dtype"),
    "floating_point_tolerance": (".dtype", "floating_point_tolerance"),
    "download_url": (".file_utils", "download_url"),
    "extract_zip": (".file_utils", "extract_zip"),
    "extract_tar": (".file_utils", "extract_tar"),
    "get_project_root": (".file_utils", "get_project_root"),
    "RankedLogger": (".logger", "RankedLogger"),
    "conditional_torchscript_mode": (
        ".compile",
        "conditional_torchscript_mode",
    ),
    "conditional_torchscript_jit": (".compile", "conditional_torchscript_jit"),
    "get_current_code_versions": (".versions", "get_current_code_versions"),
}

_MODULE_EXPORTS = {
    "model_repository": ".model_repository",
}

__all__ = [*_EXPORTS, *_MODULE_EXPORTS]


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        value = import_module(_MODULE_EXPORTS[name], __name__)
        globals()[name] = value
        return value

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
