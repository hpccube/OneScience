"""Broadcast-aware geometry utilities used by Boltz-2."""

import torch
from einops import einsum


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Align target coordinates to predictions over arbitrary batch dimensions."""
    out_shape = torch.broadcast_shapes(true_coords.shape, pred_coords.shape)
    *batch_size, num_points, dim = out_shape
    weights = (mask * weights).unsqueeze(-1)

    true_centroid = (true_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2, keepdim=True
    )
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if torch.any(mask.sum(dim=-1) < (dim + 1)):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    cov_matrix = einsum(
        weights * pred_coords_centered,
        true_coords_centered,
        "... n i, ... n j -> ... i j",
    )
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    rot_matrix = torch.einsum("... i j, ... k j -> ... i k", U, V).to(
        dtype=torch.float32
    )
    orientation = torch.eye(
        dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device
    )[None].repeat(*batch_size, 1, 1)
    orientation[..., -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(
        U, orientation, V, "... i j, ... j k, ... l k -> ... i l"
    )
    rot_matrix = rot_matrix.to(dtype=original_dtype)
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "... n i, ... j i -> ... n j")
        + pred_centroid
    )
    aligned_coords.detach_()
    return aligned_coords


__all__ = ["weighted_rigid_align"]
