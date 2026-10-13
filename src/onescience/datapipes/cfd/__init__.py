"""CFD datapipes with lazy optional-dependency imports."""

from importlib import import_module


_EXPORTS = {
    "AirfRANSDataset": (".AirfRANS", "AirfRANSDataset"),
    "AirfRANSDatapipe": (".AirfRANS", "AirfRANSDatapipe"),
    "ShapeNetCarDataset": (".ShapeNetCar", "ShapeNetCarDataset"),
    "ShapeNetCarDatapipe": (".ShapeNetCar", "ShapeNetCarDatapipe"),
    "DeepMind_CylinderFlowDataset": (
        ".deepmind_cylinderflow",
        "DeepMind_CylinderFlowDataset",
    ),
    "DeepMind_CylinderFlowDatapipe": (
        ".deepmind_cylinderflow",
        "DeepMind_CylinderFlowDatapipe",
    ),
    "EagleDataset": (".eagle", "EagleDataset"),
    "EagleDatapipe": (".eagle", "EagleDatapipe"),
    "CFDBenchDataset": (".cfdbench", "CFDBenchDataset"),
    "CFDBenchDatapipe": (".cfdbench", "CFDBenchDatapipe"),
    "DeepCFDDataset": (".deepcfd", "DeepCFDDataset"),
    "DeepCFDDatapipe": (".deepcfd", "DeepCFDDatapipe"),
    "DeepMindLagrangianDataset": (
        ".deepmind_lagrangian",
        "DeepMindLagrangianDataset",
    ),
    "DeepMindLagrangianDatapipe": (
        ".deepmind_lagrangian",
        "DeepMindLagrangianDatapipe",
    ),
    "graph_update": (".deepmind_lagrangian", "graph_update"),
    "BENODataset": (".beno", "BENODataset"),
    "BENODatapipe": (".beno", "BENODatapipe"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
