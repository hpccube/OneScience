"""SwiGLU activation used by Boltz transformer blocks."""

import torch.nn.functional as F
from torch.nn import Module


class SwiGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


__all__ = ["SwiGLU"]
