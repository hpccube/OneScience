from __future__ import annotations

import numpy as np
import torch

from onescience.utils.state.device import resolve_device

__all__ = ["normalize_batch_labels", "resolve_device"]


def normalize_batch_labels(values, batch_size: int) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        if values.ndim == 2:
            if values.shape[0] != batch_size:
                return None
            if values.shape[1] == 1:
                flat = values.reshape(batch_size)
                return [str(x) for x in flat.tolist()]
            indices = values.argmax(axis=1)
            return [str(int(x)) for x in indices.tolist()]
        if values.ndim == 1:
            if values.shape[0] != batch_size:
                return None
            return [str(x) for x in values.tolist()]
        if values.ndim == 0:
            return [str(values.item())] * batch_size
        return None
    if isinstance(values, (list, tuple)):
        if len(values) != batch_size:
            return None
        normalized = []
        for item in values:
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray):
                if item.ndim == 0:
                    normalized.append(str(item.item()))
                    continue
                if item.ndim == 1:
                    if item.size == 1:
                        normalized.append(str(item.item()))
                    elif np.count_nonzero(item) == 1:
                        normalized.append(str(int(item.argmax())))
                    else:
                        normalized.append(str(item.tolist()))
                    continue
            normalized.append(str(item))
        return normalized
    return [str(values)] * batch_size
