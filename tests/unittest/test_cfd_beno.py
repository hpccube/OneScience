from types import SimpleNamespace
import warnings

import torch


warnings.filterwarnings("ignore", message="An issue occurred while importing.*")
torch.manual_seed(0)

try:
    from onescience.models.beno.BE_MPNN import HeteroGNS
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: BENO HeteroGNS optional dependency ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

graph = SimpleNamespace(
    x=torch.randn(4, 3),
    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
    edge_features=torch.randn(4, 2),
    boundary=torch.randn(5, 3),
)
data = {"G1": graph, "G2": graph}
model = HeteroGNS(
    nnode_in_features=3,
    nnode_out_features=1,
    nedge_in_features=2,
    latent_dim=8,
    nmessage_passing_steps=1,
    nmlp_layers=1,
    mlp_hidden_dim=8,
    boundary_dim=8,
    trans_layer=1,
)
model.eval()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

with torch.no_grad():
    out = model(data)

print("Function: BENO HeteroGNS Forward")
print(f"trainable parameters: {trainable_params}")
print(f"output shape: {out.shape}")
print("target shape: torch.Size([4, 1])\n")

