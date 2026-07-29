"""Coordinate augmentation utilities shared by Boltz data and model code."""

from typing import Optional

import torch
from torch.types import Device


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    values = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return values.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Generate uniformly distributed scalar-first unit quaternions."""
    if isinstance(device, str):
        device = torch.device(device)
    values = torch.randn((n, 4), dtype=dtype, device=device)
    squared_norm = (values * values).sum(1)
    return values / _copysign(torch.sqrt(squared_norm), values[:, 0])[:, None]


def random_rotations(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Generate random 3-by-3 rotation matrices."""
    return quaternion_to_matrix(random_quaternions(n, dtype=dtype, device=device))


def randomly_rotate(
    coords: torch.Tensor,
    return_second_coords: bool = False,
    second_coords: Optional[torch.Tensor] = None,
):
    """Apply the same random rotation to one or two coordinate tensors."""
    rotations = random_rotations(len(coords), coords.dtype, coords.device)
    rotated = torch.einsum("bmd,bds->bms", coords, rotations)
    if return_second_coords:
        rotated_second = (
            torch.einsum("bmd,bds->bms", second_coords, rotations)
            if second_coords is not None
            else None
        )
        return rotated, rotated_second
    return rotated


def center_random_augmentation(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    s_trans: float = 1.0,
    augmentation: bool = True,
    centering: bool = True,
    return_second_coords: bool = False,
    second_coords: Optional[torch.Tensor] = None,
):
    """Center coordinates and optionally apply a shared rigid augmentation."""
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean
        if second_coords is not None:
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords,
            return_second_coords=True,
            second_coords=second_coords,
        )
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans
        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords
    return atom_coords
