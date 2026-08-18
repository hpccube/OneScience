import torch
from typing import Union, Sequence, Dict
from math import sqrt
from onescience.datapipes.materials.nequip import AtomicDataDict


class AvgNumNeighborsNorm(torch.nn.Module):
    def __init__(
        self,
        type_names: Sequence[str],
        avg_num_neighbors: Union[float, Dict[str, float]],
    ) -> None:
        """
        Module to normalize features during training using per type edge sum normalization.

        Args:
            type_names (Sequence[str]): list of atom type names
            avg_num_neighbors (float/Dict[str, float]): used to normalize edge sums for better numerics
        """
        super().__init__()
        assert avg_num_neighbors is not None, "avg_num_neighbors must be specified"

        self.in_field = self.out_field = AtomicDataDict.NODE_FEATURES_KEY
        self.norm_key = AtomicDataDict.FEATURE_NORM_FACTOR_KEY

        # Put avg_num_neighbors in a list (global or per type)
        if isinstance(avg_num_neighbors, (float, int)):
            avg_num_neighbors = [avg_num_neighbors]
        elif isinstance(avg_num_neighbors, dict):
            assert set(type_names) == set(avg_num_neighbors.keys())
            avg_num_neighbors = [avg_num_neighbors[k] for k in type_names]
        else:
            raise RuntimeError(
                "Unrecognized format for `avg_num_neighbors`, only floats or dicts allowed."
            )
        assert isinstance(avg_num_neighbors, list)

        # Tensorize avg_num_neighbors and register as buffer
        norm_const = torch.tensor([(1.0 / sqrt(N)) for N in avg_num_neighbors])
        norm_const = norm_const.reshape(-1, 1)
        # Persistent=False to ensure backwards compatibility of FMs.
        # TODO remove this once we're sure FMs are not using this anymore
        self.register_buffer("norm_const", norm_const, persistent=False)

        # If global avg_num_neighbors or only one type, no need to do embedding lookup in forward
        self.norm_shortcut = self.norm_const.numel() == 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.in_field]
        norm_size = features.size(0)

        if self.norm_key in data and data[self.norm_key].size(0) == norm_size:
            norm_factor = data[self.norm_key]
        else:
            # Compute norm factor for the first time
            if self.norm_shortcut:
                # No need to do embedding lookup in forward
                norm_factor = self.norm_const.expand(norm_size, -1)
            else:
                # Embed each avg_num_neighbors value per type
                norm_factor = torch.nn.functional.embedding(
                    data[AtomicDataDict.ATOM_TYPE_KEY][:norm_size],
                    self.norm_const,
                )
            data[self.norm_key] = norm_factor  # shape: (num_local_nodes, 1)

        data[self.out_field] = norm_factor * features
        return data
