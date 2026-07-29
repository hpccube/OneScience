"""Pooling functions for DNABERT-2 token representations."""

from typing import Literal

import torch


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: Literal["mean", "max", "cls"] = "mean",
) -> torch.Tensor:
    """Pool ``[batch, tokens, hidden]`` states while ignoring padded tokens."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match hidden_states batch/token axes")
    if mode == "cls":
        return hidden_states[:, 0]

    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
    if mode == "mean":
        weights = mask.to(dtype=hidden_states.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denominator
    if mode == "max":
        minimum = torch.finfo(hidden_states.dtype).min
        return hidden_states.masked_fill(~mask, minimum).max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {mode}")
