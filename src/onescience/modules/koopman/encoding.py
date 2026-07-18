from __future__ import annotations

import torch
from torch import nn


def _validate_dimensions(input_channels: int, latent_channels: int) -> tuple[int, int]:
    input_channels = int(input_channels)
    latent_channels = int(latent_channels)
    if min(input_channels, latent_channels) < 1:
        raise ValueError("input_channels and latent_channels must be positive")
    return input_channels, latent_channels


class EncoderMLP(nn.Module):
    """Project the last input dimension into a Koopman latent space."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Linear(t_len, op_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DecoderMLP(nn.Module):
    """Project a Koopman latent representation back to output channels."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Linear(op_size, t_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class EncoderConv1D(nn.Module):
    """Pointwise 1D encoder accepting channel-last tensors."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Conv1d(t_len, op_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.permute(0, 2, 1)).permute(0, 2, 1)


class DecoderConv1D(nn.Module):
    """Pointwise 1D decoder returning channel-last tensors."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Conv1d(op_size, t_len, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.permute(0, 2, 1)).permute(0, 2, 1)


class EncoderConv2D(nn.Module):
    """Pointwise 2D encoder accepting channel-last tensors."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Conv2d(t_len, op_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class DecoderConv2D(nn.Module):
    """Pointwise 2D decoder returning channel-last tensors."""

    def __init__(self, t_len: int, op_size: int) -> None:
        super().__init__()
        t_len, op_size = _validate_dimensions(t_len, op_size)
        self.layer = nn.Conv2d(op_size, t_len, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


__all__ = [
    "EncoderMLP",
    "DecoderMLP",
    "EncoderConv1D",
    "DecoderConv1D",
    "EncoderConv2D",
    "DecoderConv2D",
]
