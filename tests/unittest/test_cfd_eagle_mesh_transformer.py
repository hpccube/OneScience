import torch


try:
    from onescience.models.graphvit import GraphViT
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: Eagle GraphViT optional dependency ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

try:
    model = GraphViT(state_size=4, w_size=16, n_attention=1, nb_gn=1, n_heads=1)
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: Eagle GraphViT optional dependency ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

torch.manual_seed(0)
num_params = sum(param.numel() for param in model.parameters())
print(f"Eagle GraphViT parameters: {num_params}")
model.eval()

mesh_pos = torch.randn(1, 2, 6, 2)
edges = torch.tensor(
    [
        [
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 2], [3, 5]],
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 2], [3, 5]],
        ]
    ],
    dtype=torch.long,
)
state = torch.randn(1, 2, 6, 4)
node_type = torch.zeros(1, 2, 6, 9)
node_type[..., 0] = 1
clusters = torch.tensor([[[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]]]])
clusters_mask = torch.ones(1, 2, 2, 3, dtype=torch.bool)

try:
    with torch.no_grad():
        state_hat, output, target = model(
            mesh_pos, edges, state, node_type, clusters, clusters_mask
        )
except (ImportError, ModuleNotFoundError, OSError) as exc:
    print(f"SKIP: Eagle GraphViT optional dependency ({type(exc).__name__}: {exc})")
    raise SystemExit(0)

assert state_hat.shape == torch.Size([1, 2, 6, 4])
assert output.shape == torch.Size([1, 1, 6, 4])
assert target.shape == torch.Size([1, 1, 6, 4])
print("Function: Eagle GraphViT Forward")
print(f"state_hat shape: {state_hat.shape}")
print(f"output shape: {output.shape}")
print("target shape: torch.Size([1, 1, 6, 4])\n")
