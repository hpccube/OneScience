from __future__ import annotations

import torch
import torch.nn as nn

from onescience.modules.mlp.MLP import StandardMLP
from onescience.modules.transformer.orthogonal_neural_block import OrthogonalNeuralBlock


class ONO(nn.Module):
    """Orthogonal Neural Operator with an explicit, config-driven API.

    The forward API follows the CFD neural-operator examples:
    ``pos`` has shape ``(B, N, space_dim)`` and ``fx`` has shape
    ``(B, N, in_dim)``. The output has shape ``(B, N, out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 8,
        num_heads: int = 8,
        space_dim: int = 2,
        include_pos: bool = True,
        dropout: float = 0.0,
        activation: str = "gelu",
        mlp_ratio: int = 1,
        attn_type: str = "nystrom",
        psi_dim: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.space_dim = int(space_dim)
        self.include_pos = bool(include_pos)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        feature_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess_x = StandardMLP(
            input_dim=feature_dim,
            hidden_dims=[self.hidden_dim * 2],
            output_dim=self.hidden_dim,
            activation=activation,
            use_bias=True,
        )
        self.preprocess_z = StandardMLP(
            input_dim=feature_dim,
            hidden_dims=[self.hidden_dim * 2],
            output_dim=self.hidden_dim,
            activation=activation,
            use_bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                OrthogonalNeuralBlock(
                    num_heads=self.num_heads,
                    hidden_dim=self.hidden_dim,
                    dropout=float(dropout),
                    act=activation,
                    attn_type=attn_type,
                    mlp_ratio=int(mlp_ratio),
                    last_layer=(layer_id == self.num_layers - 1),
                    psi_dim=int(psi_dim),
                    out_dim=self.out_dim,
                )
                for layer_id in range(self.num_layers)
            ]
        )
        self.placeholder = nn.Parameter((1.0 / self.hidden_dim) * torch.rand(self.hidden_dim))
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor | None = None) -> torch.Tensor:
        if fx is None:
            if self.in_dim != 0:
                raise ValueError("fx is required when in_dim > 0.")
            features = pos if self.include_pos else pos.new_zeros(pos.shape[0], pos.shape[1], 0)
        elif self.include_pos:
            features = torch.cat((pos, fx), dim=-1)
        else:
            features = fx

        x = self.preprocess_x(features)
        z = self.preprocess_z(features) + self.placeholder[None, None, :]
        for block in self.blocks:
            x, z = block(x, z)
        return z
