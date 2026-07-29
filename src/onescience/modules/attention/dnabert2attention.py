"""Attention compatibility helpers for DNABERT-2 remote model code."""

import math

import torch


def torch_qkvpacked_attention(
    qkv: torch.Tensor,
    bias: torch.Tensor | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Compute packed QKV attention with the legacy DNABERT-2 kernel contract.

    The official DNABERT-2 checkpoint bundles an experimental Triton kernel whose
    ``tl.dot(..., trans_a/trans_b=...)`` calls are not accepted by current Triton.
    This implementation preserves the same scaled dot-product attention operation
    using public PyTorch tensor operations.
    """
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(
            "qkv must have shape (batch, sequence, 3, heads, head_dimension)"
        )

    input_dtype = qkv.dtype
    compute_dtype = (
        torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
    )
    query = qkv[:, :, 0].permute(0, 2, 1, 3).to(compute_dtype)
    key = qkv[:, :, 1].permute(0, 2, 3, 1).to(compute_dtype)
    value = qkv[:, :, 2].permute(0, 2, 1, 3).to(compute_dtype)

    scale = softmax_scale
    if scale is None:
        scale = 1.0 / math.sqrt(qkv.shape[-1])
    attention_scores = torch.matmul(query, key) * scale
    if bias is not None:
        attention_scores = attention_scores + bias.to(
            device=attention_scores.device,
            dtype=attention_scores.dtype,
        )
    if causal:
        causal_mask = torch.ones(
            attention_scores.shape[-2:],
            dtype=torch.bool,
            device=attention_scores.device,
        ).triu(diagonal=1)
        attention_scores = attention_scores.masked_fill(causal_mask, float("-inf"))

    attention_probs = torch.softmax(attention_scores, dim=-1)
    attention = torch.matmul(attention_probs, value).permute(0, 2, 1, 3)
    return attention.to(input_dtype)
