from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.koopman import (
    DecoderConv1D,
    DecoderConv2D,
    DecoderMLP,
    EncoderConv1D,
    EncoderConv2D,
    EncoderMLP,
    KoopmanOperator1D,
    KoopmanOperator2D,
)


class KNO1D(nn.Module):
    """One-dimensional Koopman neural operator."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        op_size: int,
        modes_x: int = 16,
        decompose: int = 4,
        linear_type: bool = True,
        normalization: bool = False,
    ) -> None:
        super().__init__()
        self.op_size = int(op_size)
        self.decompose = int(decompose)
        if self.decompose < 1:
            raise ValueError("decompose must be positive")
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = KoopmanOperator1D(self.op_size, modes_x=modes_x)
        self.w0 = nn.Conv1d(self.op_size, self.op_size, kernel_size=1)
        self.linear_type = bool(linear_type)
        self.normalization = bool(normalization)
        if self.normalization:
            self.norm_layer = nn.BatchNorm1d(self.op_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.tanh(self.enc(x))
        x_reconstruct = self.dec(encoded)
        latent = encoded.permute(0, 2, 1)
        shortcut = latent
        for _ in range(self.decompose):
            evolved = self.koopman_layer(latent)
            latent = latent + evolved if self.linear_type else torch.tanh(latent + evolved)
        shortcut = self.w0(shortcut)
        if self.normalization:
            shortcut = self.norm_layer(shortcut)
        latent = torch.tanh(shortcut + latent).permute(0, 2, 1)
        return self.dec(latent), x_reconstruct


class KNO2D(nn.Module):
    """Two-dimensional Koopman neural operator."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        op_size: int,
        modes_x: int = 12,
        modes_y: int = 12,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
    ) -> None:
        super().__init__()
        self.op_size = int(op_size)
        self.decompose = int(decompose)
        if self.decompose < 1:
            raise ValueError("decompose must be positive")
        self.enc = encoder
        self.dec = decoder
        self.koopman_layer = KoopmanOperator2D(
            self.op_size,
            modes_x=modes_x,
            modes_y=modes_y,
        )
        self.w0 = nn.Conv2d(self.op_size, self.op_size, kernel_size=1)
        self.linear_type = bool(linear_type)
        self.normalization = bool(normalization)
        if self.normalization:
            self.norm_layer = nn.BatchNorm2d(self.op_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.tanh(self.enc(x))
        x_reconstruct = self.dec(encoded)
        latent = encoded.permute(0, 3, 1, 2)
        shortcut = latent
        for _ in range(self.decompose):
            evolved = self.koopman_layer(latent)
            latent = latent + evolved if self.linear_type else torch.tanh(latent + evolved)
        shortcut = self.w0(shortcut)
        if self.normalization:
            shortcut = self.norm_layer(shortcut)
        latent = torch.tanh(shortcut + latent).permute(0, 2, 3, 1)
        return self.dec(latent), x_reconstruct


class KNO2DNavierStokes(nn.Module):
    """KNO2D adapter for flattened Navier-Stokes batches.

    ``pos`` is accepted for API parity with other neural operators. ``fx`` has
    shape ``(B, H*W, input_channels)`` and the output has shape
    ``(B, H*W, output_channels)``.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        spatial_shape: Sequence[int],
        op_size: int = 32,
        modes_x: int = 12,
        modes_y: int = 12,
        decompose: int = 6,
        linear_type: bool = True,
        normalization: bool = False,
    ) -> None:
        super().__init__()
        if int(output_channels) != 1:
            raise ValueError("KNO2DNavierStokes currently expects output_channels=1")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.spatial_shape = tuple(int(value) for value in spatial_shape)
        if len(self.spatial_shape) != 2 or min(self.spatial_shape) < 1:
            raise ValueError("spatial_shape must contain two positive dimensions")
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.kno = KNO2D(
            encoder=EncoderConv2D(self.input_channels, int(op_size)),
            decoder=DecoderConv2D(self.output_channels, int(op_size)),
            op_size=int(op_size),
            modes_x=int(modes_x),
            modes_y=int(modes_y),
            decompose=int(decompose),
            linear_type=bool(linear_type),
            normalization=bool(normalization),
        )

    def forward(self, pos: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, channels = fx.shape
        height, width = self.spatial_shape
        if pos.shape != (batch_size, point_count, 2):
            raise ValueError(
                f"Expected pos [B, {point_count}, 2], got {tuple(pos.shape)}"
            )
        if point_count != height * width:
            raise ValueError(f"Expected {height * width} points, got {point_count}")
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {channels}"
            )
        field = fx.reshape(batch_size, height, width, channels)
        prediction, _ = self.kno(field)
        return prediction.reshape(batch_size, point_count, self.output_channels)


__all__ = ["KNO1D", "KNO2D", "KNO2DNavierStokes"]
