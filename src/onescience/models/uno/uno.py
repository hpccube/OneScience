from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.modules.fourier.fno_layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)
from onescience.modules.layer.unet_layer import (
    DoubleConv1D,
    DoubleConv2D,
    DoubleConv3D,
    Down1D,
    Down2D,
    Down3D,
    OutConv1D,
    OutConv2D,
    OutConv3D,
    Up1D,
    Up2D,
    Up3D,
)
from onescience.modules.mlp.MLP import StandardMLP


_CONV_BLOCKS = {1: DoubleConv1D, 2: DoubleConv2D, 3: DoubleConv3D}
_DOWN_BLOCKS = {1: Down1D, 2: Down2D, 3: Down3D}
_UP_BLOCKS = {1: Up1D, 2: Up2D, 3: Up3D}
_OUT_BLOCKS = {1: OutConv1D, 2: OutConv2D, 3: OutConv3D}
_POINTWISE_CONVS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_SPECTRAL_CONVS = {1: SpectralConv1d, 2: SpectralConv2d, 3: SpectralConv3d}


class UNO(nn.Module):
    """Structured-grid U-NO model built from OneScience modular components.

    The forward API matches the CFD_Benchmark neural-operator examples:
    ``pos`` has shape ``(B, N, space_dim)`` and ``fx`` has shape
    ``(B, N, in_dim)``. The output has shape ``(B, N, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_shape: Sequence[int],
        hidden_dim: int = 64,
        modes: int = 12,
        space_dim: int | None = None,
        include_pos: bool = True,
        normtype: str = "in",
        bilinear: bool = True,
        activation: str = "gelu",
        pad_to_multiple: int = 16,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.modes = int(modes)
        self.space_dim = int(space_dim or len(self.spatial_shape))
        self.include_pos = bool(include_pos)
        self.pad_to_multiple = int(pad_to_multiple)

        dim = len(self.spatial_shape)
        if dim not in (1, 2, 3):
            raise ValueError(f"UNO supports 1D/2D/3D structured grids, got {dim}D.")
        if self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be positive.")

        input_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = StandardMLP(
            input_dim=input_dim,
            hidden_dims=[self.hidden_dim * 2],
            output_dim=self.hidden_dim,
            activation=activation,
            use_bias=True,
        )

        conv_block = _CONV_BLOCKS[dim]
        down_block = _DOWN_BLOCKS[dim]
        up_block = _UP_BLOCKS[dim]
        out_block = _OUT_BLOCKS[dim]

        factor = 2 if bilinear else 1
        self.factor = factor

        self.inc = conv_block(self.hidden_dim, self.hidden_dim, normtype=normtype)
        self.down1 = down_block(self.hidden_dim, self.hidden_dim * 2, normtype=normtype)
        self.down2 = down_block(self.hidden_dim * 2, self.hidden_dim * 4, normtype=normtype)
        self.down3 = down_block(self.hidden_dim * 4, self.hidden_dim * 8, normtype=normtype)
        self.down4 = down_block(
            self.hidden_dim * 8,
            self.hidden_dim * 16 // factor,
            normtype=normtype,
        )

        self.up1 = up_block(
            self.hidden_dim * 16,
            self.hidden_dim * 8 // factor,
            bilinear,
            normtype=normtype,
        )
        self.up2 = up_block(
            self.hidden_dim * 8,
            self.hidden_dim * 4 // factor,
            bilinear,
            normtype=normtype,
        )
        self.up3 = up_block(
            self.hidden_dim * 4,
            self.hidden_dim * 2 // factor,
            bilinear,
            normtype=normtype,
        )
        self.up4 = up_block(
            self.hidden_dim * 2,
            self.hidden_dim,
            bilinear,
            normtype=normtype,
        )
        self.outc = out_block(self.hidden_dim, self.hidden_dim)

        augmented_shape = self._augmented_shape(self.spatial_shape)
        self.padding = tuple(a - s for a, s in zip(augmented_shape, self.spatial_shape))

        self.process1_down = self._fno_layer(self.hidden_dim, self.hidden_dim, augmented_shape, 2)
        self.process2_down = self._fno_layer(self.hidden_dim * 2, self.hidden_dim * 2, augmented_shape, 4)
        self.process3_down = self._fno_layer(self.hidden_dim * 4, self.hidden_dim * 4, augmented_shape, 8)
        self.process4_down = self._fno_layer(self.hidden_dim * 8, self.hidden_dim * 8, augmented_shape, 16)
        self.process5_down = self._fno_layer(
            self.hidden_dim * 16 // factor,
            self.hidden_dim * 16 // factor,
            augmented_shape,
            32,
        )

        self.process5_up = self._fno_layer(
            self.hidden_dim * 16 // factor,
            self.hidden_dim * 16 // factor,
            augmented_shape,
            32,
        )
        self.process4_up = self._fno_layer(
            self.hidden_dim * 8 // factor,
            self.hidden_dim * 8 // factor,
            augmented_shape,
            16,
        )
        self.process3_up = self._fno_layer(
            self.hidden_dim * 4 // factor,
            self.hidden_dim * 4 // factor,
            augmented_shape,
            8,
        )
        self.process2_up = self._fno_layer(
            self.hidden_dim * 2 // factor,
            self.hidden_dim * 2 // factor,
            augmented_shape,
            4,
        )
        self.process1_up = self._fno_layer(self.hidden_dim, self.hidden_dim, augmented_shape, 2)

        self.w1_down = self._pointwise(self.hidden_dim, self.hidden_dim)
        self.w2_down = self._pointwise(self.hidden_dim * 2, self.hidden_dim * 2)
        self.w3_down = self._pointwise(self.hidden_dim * 4, self.hidden_dim * 4)
        self.w4_down = self._pointwise(self.hidden_dim * 8, self.hidden_dim * 8)
        self.w5_down = self._pointwise(self.hidden_dim * 16 // factor, self.hidden_dim * 16 // factor)
        self.w5_up = self._pointwise(self.hidden_dim * 16 // factor, self.hidden_dim * 16 // factor)
        self.w4_up = self._pointwise(self.hidden_dim * 8 // factor, self.hidden_dim * 8 // factor)
        self.w3_up = self._pointwise(self.hidden_dim * 4 // factor, self.hidden_dim * 4 // factor)
        self.w2_up = self._pointwise(self.hidden_dim * 2 // factor, self.hidden_dim * 2 // factor)
        self.w1_up = self._pointwise(self.hidden_dim, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.out_dim)

    def _augmented_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            size + (self.pad_to_multiple - size % self.pad_to_multiple) % self.pad_to_multiple
            for size in shape
        )

    def _fno_layer(
        self,
        in_channels: int,
        out_channels: int,
        augmented_shape: Sequence[int],
        divisor: int,
    ) -> nn.Module:
        dim = len(augmented_shape)
        modes = [max(1, min(self.modes, max(1, size // divisor))) for size in augmented_shape]
        kwargs = {"in_channels": in_channels, "out_channels": out_channels}
        for index, mode in enumerate(modes, start=1):
            kwargs[f"modes{index}"] = mode
        return _SPECTRAL_CONVS[dim](**kwargs)

    def _pointwise(self, in_channels: int, out_channels: int) -> nn.Module:
        conv = _POINTWISE_CONVS[len(self.spatial_shape)]
        return conv(in_channels, out_channels, kernel_size=1)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if not any(self.padding):
            return x
        pad_arg: list[int] = []
        for padding in reversed(self.padding):
            pad_arg.extend([0, padding])
        return F.pad(x, pad_arg)

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        if not any(self.padding):
            return x
        slices = [slice(None), slice(None)]
        for padding in self.padding:
            stop = -padding if padding else None
            slices.append(slice(None, stop))
        return x[tuple(slices)]

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, point_count, _ = pos.shape
        expected_points = 1
        for size in self.spatial_shape:
            expected_points *= size
        if point_count != expected_points:
            raise ValueError(
                f"Expected {expected_points} grid points for spatial_shape={self.spatial_shape}, "
                f"got {point_count}."
            )

        if fx is None:
            if self.in_dim != 0:
                raise ValueError("fx is required when in_dim > 0.")
            features = pos if self.include_pos else pos.new_zeros(batch_size, point_count, 0)
        elif self.include_pos:
            features = torch.cat((pos, fx), dim=-1)
        else:
            features = fx

        x = self.preprocess(features)
        x = x.permute(0, 2, 1).reshape(batch_size, self.hidden_dim, *self.spatial_shape)
        x = self._pad(x)

        x1 = self.inc(x)
        x1 = F.gelu(self.process1_down(x1) + self.w1_down(x1))

        x2 = self.down1(x1)
        x2 = F.gelu(self.process2_down(x2) + self.w2_down(x2))

        x3 = self.down2(x2)
        x3 = F.gelu(self.process3_down(x3) + self.w3_down(x3))

        x4 = self.down3(x3)
        x4 = F.gelu(self.process4_down(x4) + self.w4_down(x4))

        x5 = self.down4(x4)
        x5 = F.gelu(self.process5_down(x5) + self.w5_down(x5))
        x5 = F.gelu(self.process5_up(x5) + self.w5_up(x5))

        x = self.up1(x5, x4)
        x = F.gelu(self.process4_up(x) + self.w4_up(x))

        x = self.up2(x, x3)
        x = F.gelu(self.process3_up(x) + self.w3_up(x))

        x = self.up3(x, x2)
        x = F.gelu(self.process2_up(x) + self.w2_up(x))

        x = self.up4(x, x1)
        x = F.gelu(self.process1_up(x) + self.w1_up(x))
        x = self._unpad(self.outc(x))

        x = x.reshape(batch_size, self.hidden_dim, -1).permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)
