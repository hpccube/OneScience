"""Backward-compatible Boltz scalar, augmentation, and EMA utilities."""

import torch

from onescience.utils.boltz.augmentation import (
    _copysign,
    center_random_augmentation,
    quaternion_to_matrix,
    random_quaternions,
    random_rotations,
    randomly_rotate,
)
from onescience.utils.boltz.optim.simple_ema import ExponentialMovingAverage


def autocast_device_type(device_type: str) -> str:
    """Return a device type accepted by ``torch.autocast``."""
    from torch.amp.autocast_mode import is_autocast_available

    return device_type if is_autocast_available(device_type) else "cpu"


def exists(value):
    return value is not None


def default(value, fallback):
    return value if exists(value) else fallback


def log(tensor, eps=1e-20):
    return torch.log(tensor.clamp(min=eps))


def center(atom_coords, atom_mask):
    atom_mean = torch.sum(
        atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
    ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
    return atom_coords - atom_mean


def compute_random_augmentation(
    multiplicity,
    s_trans=1.0,
    device=None,
    dtype=torch.float32,
):
    rotations = random_rotations(multiplicity, dtype=dtype, device=device)
    translation = torch.randn(
        (multiplicity, 1, 3),
        dtype=dtype,
        device=device,
    ) * s_trans
    return rotations, translation


__all__ = [
    "ExponentialMovingAverage",
    "autocast_device_type",
    "center",
    "center_random_augmentation",
    "compute_random_augmentation",
    "default",
    "exists",
    "log",
    "quaternion_to_matrix",
    "random_quaternions",
    "random_rotations",
    "randomly_rotate",
]
