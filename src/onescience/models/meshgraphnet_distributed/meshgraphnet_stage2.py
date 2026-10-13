"""
MeshGraphNet Stage 2: Decoder

This stage decodes processed node features back to physical space.
"""

import torch
import torch.nn as nn
from typing import Tuple

import onescience
from onescience.modules.mlp.mesh_graph_distributed_mlp import DistributedMeshGraphMLP
from onescience.modules.mlp.mesh_graph_mlp import MeshGraphMLP
from onescience.modules.layer.activations import get_activation


class MeshGraphNetStage2(nn.Module):
    """
    MeshGraphNet Stage 2: Decoder

    This stage decodes processed node features back to physical space.
    It reads from input_tensor and returns final output.

    Args:
        output_dim (int): Output feature dimension
        hidden_dim_processor (int): Hidden dimension for processor
        hidden_dim_node_decoder (int): Hidden dimension for node decoder
        num_layers_node_decoder (int): Number of layers in node decoder
        mlp_activation_fn (str): Activation function for MLP
        recompute_activation (bool): Whether to recompute activation
        config: Megatron config object (for tensor parallel)
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim_processor: int,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        mlp_activation_fn: str = "relu",
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()
        self.input_tensor = None

        activation_fn = get_activation(mlp_activation_fn)

        use_distributed = config is not None and config.tensor_model_parallel_size > 1

        def _create_mlp(**kwargs):
            if use_distributed:
                return DistributedMeshGraphMLP(config=config, **kwargs)
            return MeshGraphMLP(**kwargs)

        # Node Decoder
        mlp_kwargs = {
            "input_dim": hidden_dim_processor,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim_node_decoder,
            "hidden_layers": num_layers_node_decoder,
            "activation_fn": activation_fn,
            "norm_type": None,
            "recompute_activation": recompute_activation,
        }
        self.node_decoder = _create_mlp(**mlp_kwargs)

    def set_input_tensor(self, input_tensor):
        """Megatron pipeline scheduling hook"""
        self.input_tensor = input_tensor

    def forward(self, node_features, edge_features) -> torch.Tensor:
        """
        Args:
            node_features: Node features (num_nodes, hidden_dim)
            edge_features: Edge features (num_edges, hidden_dim)

        Returns:
            Decoded node features (num_nodes, output_dim)
        """
        # Decode node features
        output = self.node_decoder(node_features)

        return output
