# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed IPC weight transfer with the buffer shared as a CUDA VMM fd."""

import multiprocessing

import pytest
import torch

from vllm.platforms import current_platform

NAMES = [f"layer{i}.weight" for i in range(10)]


def _weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    return [
        (n, torch.randn(200 + i, 384, device="cuda", generator=g, dtype=torch.bfloat16))
        for i, n in enumerate(NAMES)
    ]


def _trainer(chunks_q, acks_q):
    from vllm.distributed.weight_transfer.packed_tensor import packed_ipc_producer

    torch.cuda.set_device(0)
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    # A small buffer forces several chunks through the same shared buffer.
    for chunk in packed_ipc_producer(
        iter(_weights()), uuid, lambda nt: nt[1], buffer_size_bytes=1 << 20, vmm_fd=True
    ):
        chunks_q.put(chunk)
        assert acks_q.get(timeout=120) == "ok"
    chunks_q.put(None)


def _worker(chunks_q, acks_q, results_q):
    from vllm.distributed.weight_transfer.packed_tensor import (
        PackedBufferImporter,
        packed_ipc_consumer,
    )

    torch.cuda.set_device(0)
    importer = PackedBufferImporter()
    received = {}
    handles = set()
    while (chunk := chunks_q.get(timeout=120)) is not None:
        handles.add(next(iter(chunk.ipc_handle.values()))[0])
        for name, t in packed_ipc_consumer(
            ipc_handle=chunk.ipc_handle,
            names=chunk.names,
            shapes=chunk.shapes,
            dtype_names=chunk.dtype_names,
            tensor_sizes=chunk.tensor_sizes,
            device_index=0,
            importer=importer,
        ):
            received[name] = t
        torch.cuda.synchronize()
        acks_q.put("ok")
    importer.close()
    expected = dict(_weights())
    results_q.put(
        (
            sorted(handles),
            all(torch.equal(received[n], expected[n]) for n in NAMES),
            len(received),
        )
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="vmm_fd needs CUDA VMM")
def test_packed_vmm_fd_transfer_between_sibling_processes():
    """Trainer and worker are siblings, where PyTorch's own sharing of VMM
    memory (pidfd_getfd) is refused under the default ptrace policy."""
    ctx = multiprocessing.get_context("spawn")
    chunks_q, acks_q, results_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    trainer = ctx.Process(target=_trainer, args=(chunks_q, acks_q))
    worker = ctx.Process(target=_worker, args=(chunks_q, acks_q, results_q))
    trainer.start()
    worker.start()
    handles, all_equal, count = results_q.get(timeout=300)
    trainer.join(timeout=60)
    worker.join(timeout=60)
    assert trainer.exitcode == 0 and worker.exitcode == 0
    assert handles == ["vmm_fd"]
    assert count == len(NAMES)
    assert all_equal


def test_vmm_fd_requires_packed():
    from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

    with pytest.raises(ValueError, match="requires packed=True"):
        IPCTrainerInitInfo(rank=0, packed=False, packed_transport="vmm_fd")
    with pytest.raises(ValueError, match="Unknown packed_transport"):
        IPCTrainerInitInfo(rank=0, packed=True, packed_transport="nvlink")
    assert IPCTrainerInitInfo(rank=0, packed=True, packed_transport="vmm_fd").packed
