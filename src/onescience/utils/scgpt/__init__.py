"""Training, inference, and checkpoint utilities for scGPT."""

from .logging import logger
from .distributed import (
    DistributedContext,
    broadcast_module,
    contiguous_shard_bounds,
    distributed_barrier,
    finalize_distributed,
    initialize_distributed,
)
from .util import load_pretrained, set_seed


def embed_data(*args, **kwargs):
    """Lazily import the AnnData embedding helper."""
    from .cell_emb import embed_data as _embed_data

    return _embed_data(*args, **kwargs)


def get_batch_cell_embeddings(*args, **kwargs):
    """Lazily import the batched cell embedding helper."""
    from .cell_emb import get_batch_cell_embeddings as _get_batch_cell_embeddings

    return _get_batch_cell_embeddings(*args, **kwargs)

__all__ = [
    "DistributedContext",
    "broadcast_module",
    "contiguous_shard_bounds",
    "distributed_barrier",
    "embed_data",
    "finalize_distributed",
    "get_batch_cell_embeddings",
    "initialize_distributed",
    "load_pretrained",
    "logger",
    "set_seed",
]
