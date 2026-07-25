import json
import os
import socket
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from distributed_utils import (
    DistributedContext,
    DistributedEvalSampler,
    cleanup_distributed,
    distributed_barrier,
    reduce_mean,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from main import (
    DDPCompatibleTrainingModel,
    build_parser,
    json_safe_config,
    normalize_args,
    save_checkpoint,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gloo_worker(
    rank: int,
    world_size: int,
    port: int,
    result_path: str,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    context = setup_distributed(
        allow_cpu_distributed=True,
        backend="gloo",
    )
    payload = None
    try:
        sampler = DistributedSampler(
            list(range(8)),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
        local_indices = list(iter(sampler))
        gathered_indices = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_indices, local_indices)

        torch.manual_seed(9)
        endpoint = wrap_ddp(
            nn.Linear(2, 2, bias=False),
            context,
            find_unused_parameters=False,
        )
        source = wrap_ddp(
            nn.Linear(2, 2, bias=False),
            context,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.SGD(
            list(endpoint.parameters()) + list(source.parameters()),
            lr=0.1,
        )
        for micro_step in range(2):
            with ExitStack() as stack:
                if micro_step == 0:
                    stack.enter_context(endpoint.no_sync())
                    stack.enter_context(source.no_sync())
                inputs = torch.full(
                    (3, 2),
                    float(rank + micro_step + 1),
                )
                loss = endpoint(source(inputs)).square().mean() / 2.0
                loss.backward()
        optimizer.step()

        endpoint_weight = unwrap_model(endpoint).weight.detach().clone()
        source_weight = unwrap_model(source).weight.detach().clone()
        endpoint_weights = [torch.zeros_like(endpoint_weight) for _ in range(world_size)]
        source_weights = [torch.zeros_like(source_weight) for _ in range(world_size)]
        torch.distributed.all_gather(endpoint_weights, endpoint_weight)
        torch.distributed.all_gather(source_weights, source_weight)
        reduced_loss = reduce_mean(
            torch.tensor(float(rank + 1)),
            context,
        )

        if context.is_main_process:
            flattened = [index for part in gathered_indices for index in part]
            payload = {
                "indices": gathered_indices,
                "unique_indices": sorted(set(flattened)),
                "no_overlap": len(flattened) == len(set(flattened)),
                "endpoint_synced": all(
                    torch.equal(endpoint_weights[0], weight)
                    for weight in endpoint_weights[1:]
                ),
                "source_synced": all(
                    torch.equal(source_weights[0], weight)
                    for weight in source_weights[1:]
                ),
                "reduced_mean": float(reduced_loss.item()),
                "writer_rank": rank,
            }
        distributed_barrier(context)
    finally:
        cleanup_distributed(context)

    if rank == 0:
        payload["cleanup_complete"] = not torch.distributed.is_initialized()
        Path(result_path).write_text(json.dumps(payload), encoding="utf-8")


def test_single_process_context_does_not_initialize_process_group(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    context = setup_distributed()
    try:
        assert context.distributed is False
        assert context.rank == 0
        assert context.world_size == 1
        assert context.is_main_process is True
        assert not torch.distributed.is_initialized()
    finally:
        cleanup_distributed(context)


def test_distributed_eval_sampler_has_no_padding_duplicates():
    rank0 = list(DistributedEvalSampler(range(5), rank=0, world_size=2))
    rank1 = list(DistributedEvalSampler(range(5), rank=1, world_size=2))
    assert set(rank0).isdisjoint(rank1)
    assert sorted(rank0 + rank1) == list(range(5))


def test_config_records_global_batch_sizes():
    args = normalize_args(
        build_parser().parse_args(
            [
                "--result_dir",
                "/tmp/test",
                "--batch_size",
                "2",
                "--grad_accum_steps",
                "3",
            ]
        )
    )
    context = DistributedContext(
        distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        is_main_process=True,
        backend="gloo",
    )
    config = json_safe_config(args, distributed_context=context)
    assert config["local_batch_size"] == 2
    assert config["global_batch_size"] == 4
    assert config["effective_global_batch_size"] == 12
    assert config["effective_batch_size"] == 12


def test_checkpoint_uses_unwrapped_model_keys(tmp_path):
    raw_model = nn.Linear(2, 2)
    adapter = DDPCompatibleTrainingModel(raw_model)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    args = type("Args", (), {"num_classes": 2})()
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(
        path,
        adapter,
        None,
        optimizer,
        scheduler,
        None,
        0,
        args,
        1.0,
        {"loss": [1.0]},
        {},
        global_step=3,
        optimizer_step=2,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload["model"]) == set(raw_model.state_dict())
    assert all(not key.startswith("module.") for key in payload["model"])
    assert payload["global_step"] == 3
    assert payload["optimizer_step"] == 2


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(),
    reason="Gloo backend is unavailable",
)
def test_two_process_gloo_ddp_sampler_sync_reduce_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result_path = tmp_path / "rank0_result.json"
    mp.spawn(
        _gloo_worker,
        args=(2, _free_port(), str(result_path)),
        nprocs=2,
        join=True,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["indices"] == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert result["unique_indices"] == list(range(8))
    assert result["no_overlap"] is True
    assert result["endpoint_synced"] is True
    assert result["source_synced"] is True
    assert result["reduced_mean"] == pytest.approx(1.5)
    assert result["writer_rank"] == 0
    assert result["cleanup_complete"] is True
