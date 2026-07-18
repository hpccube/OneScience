from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from onescience.modules.mlp.MLP import StandardMLP
from onescience.modules.transformer.factformer_block import Factformer_block


class FactFormer2D(nn.Module):
    """FactFormer for structured 2D CFD fields built from OneScience modules.

    The model follows the neural-operator batch API used by the CFD examples:
    ``pos`` is shaped ``(B, N, 2)``, ``fx`` is shaped ``(B, N, t_in*out_dim)``,
    and the output is ``(B, N, latent_steps*out_dim)``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_shape: Sequence[int],
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        include_pos: bool = True,
        space_dim: int = 2,
        latent_multiplier: float = 2.0,
        max_latent_steps: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.include_pos = bool(include_pos)
        self.space_dim = int(space_dim)
        self.latent_dim = int(self.hidden_dim * float(latent_multiplier))
        self.max_latent_steps = int(max_latent_steps)

        if len(self.spatial_shape) != 2:
            raise ValueError(f"FactFormer2D expects a 2D structured grid, got {self.spatial_shape}.")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for FactAttention2D.")
        if self.max_latent_steps < 1:
            raise ValueError("max_latent_steps must be positive.")

        input_dim = self.in_dim + (self.space_dim if self.include_pos else 0)
        self.preprocess = StandardMLP(
            input_dim=input_dim,
            output_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim * 2],
            activation=activation,
            use_bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                Factformer_block(
                    num_heads=self.heads,
                    hidden_dim=self.hidden_dim,
                    dropout=float(dropout),
                    act=activation,
                    mlp_ratio=int(mlp_ratio),
                    out_dim=self.hidden_dim,
                    last_layer=False,
                    geotype="structured_2D",
                    shapelist=list(self.spatial_shape),
                )
                for _ in range(self.depth)
            ]
        )

        self.expand_latent = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.latent_time_embedding = nn.Parameter(
            torch.randn(1, self.max_latent_steps, 1, self.latent_dim) * 0.02
        )
        self.propagator = StandardMLP(
            input_dim=self.latent_dim,
            output_dim=self.latent_dim,
            hidden_dims=[self.hidden_dim],
            activation=activation,
            use_bias=True,
        )
        self.to_out = StandardMLP(
            input_dim=self.latent_dim,
            output_dim=self.out_dim,
            hidden_dims=[self.hidden_dim],
            activation=activation,
            use_bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pos: torch.Tensor, fx: torch.Tensor, latent_steps: int = 1) -> torch.Tensor:
        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(fx.shape[0], -1, -1)
        if pos.shape[0] == 1 and fx.shape[0] > 1:
            pos = pos.expand(fx.shape[0], -1, -1)
        if fx.ndim != 3:
            raise ValueError(f"fx must be shaped (B, N, C), got {tuple(fx.shape)}.")
        if pos.shape[:2] != fx.shape[:2]:
            raise ValueError(f"pos shape {tuple(pos.shape)} is incompatible with fx shape {tuple(fx.shape)}.")
        if latent_steps < 1 or latent_steps > self.max_latent_steps:
            raise ValueError(f"latent_steps must be in [1, {self.max_latent_steps}], got {latent_steps}.")

        hidden = torch.cat((pos, fx), dim=-1) if self.include_pos else fx
        hidden = self.preprocess(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        latent = self.expand_latent(hidden)
        outputs = []
        for step in range(int(latent_steps)):
            latent = latent + self.latent_time_embedding[:, step]
            latent = self.propagator(latent) + latent
            outputs.append(self.to_out(latent))
        return torch.cat(outputs, dim=-1)
