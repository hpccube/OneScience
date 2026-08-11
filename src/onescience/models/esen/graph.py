"""Graph helpers shared by the OneScience eSEN backbones."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from onescience.modules.func_utils.uma_graph.compute import generate_graph


class GraphModelMixin:
    """Provide FairChem-v1 compatible graph output using OneScience graph code."""

    def generate_graph(self, data, cutoff=None, max_neighbors=None, **kwargs):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        pbc = kwargs.pop("pbc", None)
        if pbc is None:
            pbc = getattr(data, "pbc", None)
        if pbc is None:
            pbc = torch.ones(
                (data.natoms.numel(), 3), dtype=torch.bool, device=data.pos.device
            )
        elif pbc.ndim == 1:
            pbc = pbc.view(1, 3).expand(data.natoms.numel(), -1)

        graph = generate_graph(
            data,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            enforce_max_neighbors_strictly=getattr(
                self, "enforce_max_neighbors_strictly", False
            ),
            radius_pbc_version=getattr(self, "radius_pbc_version", 1),
            pbc=pbc,
        )
        graph.update(
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers,
            node_offset=0,
        )
        return SimpleNamespace(**graph)
