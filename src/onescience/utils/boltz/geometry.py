"""Geometry metrics shared by Boltz data and model code."""

import torch
from einops import einsum


def lddt_dist(
    dmat_predicted: torch.Tensor,
    dmat_true: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float | torch.Tensor = 15.0,
    per_atom: bool = False,
):
    """Compute lDDT from predicted and target distance matrices."""
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)
    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )
    if per_atom:
        valid = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, valid.float()

    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
    score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
    total = torch.sum(dists_to_score, dim=(-2, -1))
    return score, total


def weighted_rigid_align(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Align target coordinates to predictions with weighted Kabsch alignment."""
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid

    if num_points < dim + 1:
        print(
            "Warning: WeightedRigidAlign cannot return a unique rotation "
            "for a point cloud with at most dim+1 points."
        )

    covariance = einsum(
        weights * pred_centered,
        true_centered,
        "b n i, b n j -> b i j",
    )
    original_dtype = covariance.dtype
    covariance_32 = covariance.to(dtype=torch.float32)
    u, singular_values, v = torch.linalg.svd(
        covariance_32,
        driver="gesvd" if covariance_32.is_cuda else None,
    )
    v = v.mH
    if (singular_values.abs() <= 1e-15).any() and num_points >= dim + 1:
        print(
            "Warning: WeightedRigidAlign cannot return a unique rotation "
            "for an excessively low-rank covariance matrix."
        )

    rotation = torch.einsum("b i j, b k j -> b i k", u, v).to(torch.float32)
    orientation = torch.eye(
        dim,
        dtype=covariance_32.dtype,
        device=covariance.device,
    )[None].repeat(batch_size, 1, 1)
    orientation[:, -1, -1] = torch.det(rotation)
    rotation = einsum(u, orientation, v, "b i j, b j k, b l k -> b i l")
    rotation = rotation.to(dtype=original_dtype)
    aligned = (
        einsum(true_centered, rotation, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned.detach_()
    return aligned
