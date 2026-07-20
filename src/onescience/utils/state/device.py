from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    else:
        normalized = str(requested or "auto").strip().lower()
        if normalized == "auto":
            normalized = "cuda" if torch.cuda.is_available() else "cpu"
        elif normalized == "gpu":
            normalized = "cuda"
        elif normalized == "dcu":
            normalized = "cuda"
        elif normalized.startswith("dcu:"):
            normalized = f"cuda:{normalized.split(':', 1)[1]}"
        device = torch.device(normalized)

    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device
