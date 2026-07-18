from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.modules.equivariant.group_conv import GroupEquivariantConv2d
from onescience.modules.fourier.group_spectral import GSpectralConv2d
from onescience.modules.mlp.GMLP import GroupEquivariantMLP2d
from onescience.modules.mlp.MLP import StandardMLP


class GNorm(nn.Module):
    def __init__(self, width: int, group_size: int) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        return x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])


class GFNO(nn.Module):
    """Group equivariant Fourier neural operator for structured 2D grids.

    The forward API follows the CFD neural-operator examples:
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
        num_layers: int = 4,
        space_dim: int = 2,
        include_pos: bool = True,
        activation: str = "gelu",
        reflection: bool = False,
        pad_to_multiple: int = 16,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.modes = int(modes)
        self.num_layers = int(num_layers)
        self.space_dim = int(space_dim)
        self.include_pos = bool(include_pos)
        self.reflection = bool(reflection)
        self.pad_to_multiple = int(pad_to_multiple)

        if len(self.spatial_shape) != 2:
            raise ValueError("GFNO currently supports 2D structured grids.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.pad_to_multiple < 1:
            raise ValueError("pad_to_multiple must be positive.")

        feature_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = StandardMLP(
            input_dim=feature_dim,
            hidden_dims=[self.hidden_dim * 2],
            output_dim=self.hidden_dim,
            activation=activation,
            use_bias=True,
        )

        self.group_size = 4 * (1 + int(self.reflection))
        self.padding = tuple(
            (self.pad_to_multiple - size % self.pad_to_multiple) % self.pad_to_multiple
            for size in self.spatial_shape
        )

        self.p = GroupEquivariantConv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=1,
            reflection=self.reflection,
            first_layer=True,
        )

        self.spectral_layers = nn.ModuleList(
            [
                GSpectralConv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    modes=self.modes,
                    reflection=self.reflection,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.mlp_layers = nn.ModuleList(
            [
                GroupEquivariantMLP2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    mid_channels=self.hidden_dim,
                    reflection=self.reflection,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.residual_layers = nn.ModuleList(
            [
                GroupEquivariantConv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=1,
                    reflection=self.reflection,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = GNorm(self.hidden_dim, self.group_size)
        self.q = GroupEquivariantMLP2d(
            in_channels=self.hidden_dim,
            out_channels=self.out_dim,
            mid_channels=self.hidden_dim * 4,
            reflection=self.reflection,
            last_layer=True,
        )

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if not any(self.padding):
            return x
        return F.pad(x, [0, self.padding[1], 0, self.padding[0]])

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        if not any(self.padding):
            return x
        h_stop = -self.padding[0] if self.padding[0] else None
        w_stop = -self.padding[1] if self.padding[1] else None
        return x[..., :h_stop, :w_stop]

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, point_count, _ = pos.shape
        expected_points = self.spatial_shape[0] * self.spatial_shape[1]
        if point_count != expected_points:
            raise ValueError(
                f"Expected {expected_points} grid points for spatial_shape={self.spatial_shape}, got {point_count}."
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
        x = self.p(x)

        for layer_id, (spectral, mlp, residual) in enumerate(
            zip(self.spectral_layers, self.mlp_layers, self.residual_layers)
        ):
            x1 = self.norm(spectral(self.norm(x)))
            x1 = mlp(x1)
            x = x1 + residual(x)
            if layer_id != self.num_layers - 1:
                x = F.gelu(x)

        x = self._unpad(x)
        x = self.q(x)
        return x.reshape(batch_size, self.out_dim, -1).permute(0, 2, 1)
