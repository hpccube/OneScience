import torch


try:
    import dgl
    from onescience.models.meshgraphnet import MeshGraphNet
except Exception as exc:
    print("SKIP: MeshGraphNet requires the optional DGL/torchdata stack")
    print(f"reason: {type(exc).__name__}: {exc}")
    raise SystemExit(0)


torch.manual_seed(0)

model = MeshGraphNet(
    input_dim_nodes=4,
    input_dim_edges=3,
    output_dim=2,
    processor_size=1,
    hidden_dim_processor=16,
    hidden_dim_node_encoder=16,
    hidden_dim_edge_encoder=16,
    hidden_dim_node_decoder=16,
    num_layers_node_processor=1,
    num_layers_edge_processor=1,
    num_layers_node_encoder=1,
    num_layers_edge_encoder=1,
    num_layers_node_decoder=1,
)
model.eval()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

src = torch.tensor([0, 1, 2, 3, 0, 2])
dst = torch.tensor([1, 2, 3, 0, 2, 0])
graph = dgl.graph((src, dst), num_nodes=4)
node_features = torch.randn(4, 4)
edge_features = torch.randn(graph.num_edges(), 3)

with torch.no_grad():
    out = model(node_features, edge_features, graph)

print("Function: CFD MeshGraphNet Forward")
print(f"trainable parameters: {trainable_params}")
print(f"output shape: {out.shape}")
print("target shape: torch.Size([4, 2])\n")
