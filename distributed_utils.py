import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sized

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main_process: bool
    backend: Optional[str]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def setup_distributed(
    *,
    allow_cpu_distributed: bool = False,
    backend: Optional[str] = None,
) -> DistributedContext:
    """Initialize env:// DDP when torchrun exported WORLD_SIZE > 1."""
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size < 1:
        raise RuntimeError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"RANK={rank} is invalid for WORLD_SIZE={world_size}")
    if local_rank < 0:
        raise RuntimeError(f"LOCAL_RANK must be >= 0, got {local_rank}")

    if world_size == 1:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return DistributedContext(
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
            is_main_process=True,
            backend=None,
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")

    if torch.cuda.is_available():
        selected_backend = backend or "nccl"
        if selected_backend == "nccl" and not dist.is_nccl_available():
            raise RuntimeError("WORLD_SIZE > 1 requires NCCL, but NCCL is unavailable")
        device_count = torch.cuda.device_count()
        if world_size > device_count:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} exceeds visible CUDA device count={device_count}"
            )
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is invalid for {device_count} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        selected_backend = backend or "gloo"
        if not allow_cpu_distributed:
            raise RuntimeError(
                "WORLD_SIZE > 1 was requested without CUDA; main training requires CUDA/NCCL"
            )
        if selected_backend != "gloo":
            raise RuntimeError("CPU distributed tests require the gloo backend")
        device = torch.device("cpu")

    init_kwargs = {}
    if device.type == "cuda":
        init_kwargs["device_id"] = device
    try:
        dist.init_process_group(
            backend=selected_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            **init_kwargs,
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

    context = DistributedContext(
        distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main_process=rank == 0,
        backend=selected_backend,
    )
    print(
        f"rank={rank} local_rank={local_rank} device={device}",
        flush=True,
    )
    return context


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(context: DistributedContext) -> None:
    if context.distributed:
        if not dist.is_initialized():
            raise RuntimeError("distributed barrier requested without an initialized process group")
        if context.backend == "nccl":
            dist.barrier(device_ids=[context.local_rank])
        else:
            dist.barrier()


def unwrap_model(module):
    """Remove DDP and the local operation adapter without changing state-dict keys."""
    current = module
    while current is not None:
        if isinstance(current, DistributedDataParallel):
            current = current.module
            continue
        if getattr(current, "_is_cfm_ddp_adapter", False):
            current = current.wrapped_model
            continue
        break
    return current


def wrap_ddp(
    module: torch.nn.Module,
    context: DistributedContext,
    *,
    find_unused_parameters: bool,
) -> torch.nn.Module:
    if not context.distributed:
        return module
    cuda_kwargs = {}
    if context.device.type == "cuda":
        cuda_kwargs = {
            "device_ids": [context.local_rank],
            "output_device": context.local_rank,
        }
    return DistributedDataParallel(
        module,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
        **cuda_kwargs,
    )


def reduce_mean(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    reduced = tensor.detach().float().clone()
    if context.distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= context.world_size
    return reduced


def reduce_scalar_dict(
    metrics: Mapping[str, torch.Tensor],
    context: DistributedContext,
) -> Dict[str, torch.Tensor]:
    keys = list(metrics)
    if not keys:
        return {}
    values = torch.stack(
        [metrics[key].detach().float().reshape(()) for key in keys]
    )
    if context.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return {key: values[index] for index, key in enumerate(keys)}


def reduce_max(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    reduced = tensor.detach().float().clone()
    if context.distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return reduced


def assert_equal_across_ranks(
    value: int,
    context: DistributedContext,
    *,
    name: str,
) -> None:
    if not context.distributed:
        return
    local = torch.tensor(value, device=context.device, dtype=torch.int64)
    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        raise RuntimeError(
            f"rank-local {name} differs: min={int(minimum.item())}, "
            f"max={int(maximum.item())}"
        )


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DistributedEvalSampler(Sampler[int]):
    """Non-padding eval sampler: every dataset index is evaluated exactly once."""

    def __init__(
        self,
        dataset: Sized,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"invalid rank={rank} for world_size={world_size}")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(len(self.dataset) - self.rank, 0)
        return math.ceil(remaining / self.world_size)
