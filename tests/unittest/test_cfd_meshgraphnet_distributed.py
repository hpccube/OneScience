import os

import dgl
import torch
import torch.distributed as dist

from onescience.distributed.megatron.core import mpu
from onescience.distributed.megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from onescience.distributed.megatron.core.transformer.transformer_config import (
    TransformerConfig,
)
from onescience.models.meshgraphnet_distributed import MeshGraphNetDistributedStage


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=dist.get_world_size(),
        pipeline_model_parallel_size=1,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(0)

    try:
        torch.manual_seed(0)
        device = torch.device("cuda", local_rank)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
        )
        model = MeshGraphNetDistributedStage(
            config=config,
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
        ).to(device)
        model.eval()

        src = torch.tensor([0, 1, 2, 3, 0, 2], device=device)
        dst = torch.tensor([1, 2, 3, 0, 2, 0], device=device)
        graph = dgl.graph((src, dst), num_nodes=4, device=device)
        node_features = torch.randn(4, 4, device=device)
        edge_features = torch.randn(graph.num_edges(), 3, device=device)
        model.set_graph_info(graph, graph.num_nodes(), graph.num_edges())

        with torch.no_grad():
            out = model((node_features, edge_features))

        local_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sharded_params = torch.tensor(local_params, device=device)
        dist.all_reduce(sharded_params)

        if dist.get_rank() == 0:
            print("Function: CFD MeshGraphNet Distributed Forward")
            print(f"local trainable parameters: {local_params}")
            print(f"summed sharded parameters: {sharded_params.item()}")
            print(f"output shape: {out.shape}")
            print("target shape: torch.Size([4, 2])\n")
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
