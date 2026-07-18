from __future__ import annotations

import torch
from torch import nn


class KoopmanOperator1D(nn.Module):
    """Learned Koopman evolution on low one-dimensional Fourier modes."""

    def __init__(self, op_size: int, modes_x: int = 16) -> None:
        super().__init__()
        self.op_size = int(op_size)
        self.modes_x = int(modes_x)
        if min(self.op_size, self.modes_x) < 1:
            raise ValueError("op_size and modes_x must be positive")
        scale = 1.0 / (self.op_size * self.op_size)
        self.koopman_matrix = nn.Parameter(
            scale
            * torch.rand(
                self.op_size,
                self.op_size,
                self.modes_x,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def time_marching(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btx,tfx->bfx", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.op_size:
            raise ValueError(
                f"Expected x [B, {self.op_size}, X], got {tuple(x.shape)}"
            )
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)
        modes_x = min(self.modes_x, x_ft.shape[-1])
        out_ft[:, :, :modes_x] = self.time_marching(
            x_ft[:, :, :modes_x],
            self.koopman_matrix[:, :, :modes_x],
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class KoopmanOperator2D(nn.Module):
    """Learned Koopman evolution on low two-dimensional Fourier modes."""

    def __init__(
        self,
        op_size: int,
        modes_x: int = 12,
        modes_y: int = 12,
    ) -> None:
        super().__init__()
        self.op_size = int(op_size)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        if min(self.op_size, self.modes_x, self.modes_y) < 1:
            raise ValueError("op_size, modes_x and modes_y must be positive")
        scale = 1.0 / (self.op_size * self.op_size)
        self.koopman_matrix = nn.Parameter(
            scale
            * torch.rand(
                self.op_size,
                self.op_size,
                self.modes_x,
                self.modes_y,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def time_marching(input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btxy,tfxy->bfxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.op_size:
            raise ValueError(
                f"Expected x [B, {self.op_size}, X, Y], got {tuple(x.shape)}"
            )
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        modes_x = min(self.modes_x, max(1, x_ft.shape[-2] // 2))
        modes_y = min(self.modes_y, x_ft.shape[-1])
        weights = self.koopman_matrix[:, :, :modes_x, :modes_y]
        out_ft[:, :, :modes_x, :modes_y] = self.time_marching(
            x_ft[:, :, :modes_x, :modes_y],
            weights,
        )
        out_ft[:, :, -modes_x:, :modes_y] = self.time_marching(
            x_ft[:, :, -modes_x:, :modes_y],
            weights,
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


__all__ = ["KoopmanOperator1D", "KoopmanOperator2D"]
