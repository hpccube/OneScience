from types import SimpleNamespace

import torch

from onescience.models.transolver import Transolver2D


torch.manual_seed(0)

model = Transolver2D(
    space_dim=2,
    fun_dim=1,
    out_dim=2,
    n_hidden=16,
    n_layers=1,
    n_head=4,
    mlp_ratio=1,
    slice_num=8,
    unified_pos=False,
)
model.eval()

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

data = SimpleNamespace(
    x=torch.randn(32, 3),
    pos=torch.randn(32, 2),
)

with torch.no_grad():
    out = model(data)

print("Function: CFD Transolver2D Forward")
print(f"output shape: {out.shape}")
print("target shape: torch.Size([32, 2])\n")
