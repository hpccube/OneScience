"""Small distributed-runtime helpers for scGPT examples."""

import os
from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Process information initialized by ``torchrun``."""

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(
    requested_device: Union[str, torch.device],
) -> DistributedContext:
    """Initialize NCCL when launched by ``torchrun`` and select the local device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistributedContext(device=torch.device(requested_device))

    requested_device = torch.device(requested_device)
    if requested_device.type != "cuda":
        raise ValueError("Multi-device scGPT execution requires a CUDA/DTK device")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("The active PyTorch build does not provide NCCL support")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return DistributedContext(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def finalize_distributed(context: DistributedContext) -> None:
    """Release the process group."""
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_module(module: torch.nn.Module, source: int = 0) -> None:
    """Broadcast parameters and buffers from one distributed rank."""
    if not dist.is_initialized():
        return
    tensors = list(module.state_dict().values())
    dist._broadcast_coalesced(
        dist.group.WORLD,
        tensors,
        buffer_size=256 * 1024 * 1024,
        src=source,
    )


def distributed_barrier(context: DistributedContext) -> None:
    """Synchronize ranks while explicitly binding the collective to a device."""
    if context.enabled and dist.is_initialized():
        dist.barrier(device_ids=[context.local_rank])


def contiguous_shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return a balanced, contiguous half-open interval for one rank."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return total * rank // world_size, total * (rank + 1) // world_size


__all__ = [
    "DistributedContext",
    "broadcast_module",
    "contiguous_shard_bounds",
    "distributed_barrier",
    "finalize_distributed",
    "initialize_distributed",
]
