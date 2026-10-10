# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The weight cache's vmm_fd transport: packing, layout and lifetime."""

import multiprocessing
import os
import pickle
import socket

import pytest
import torch
from torch import nn

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="vmm_fd needs CUDA VMM"
)


class _Model(nn.Module):
    """Tied, transposed, fused-view and buffer tensors in one module."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.head = nn.Linear(32, 64, bias=False)
        self.head.weight = self.embed.weight  # tied
        # A non-contiguous parameter, as transposing post-processing leaves.
        self.wt = nn.Parameter(torch.randn(48, 32).t(), requires_grad=False)
        # Two parameters viewing one storage, as fused projections do.
        fused = torch.randn(2, 16, 32)
        self.q = nn.Parameter(fused[0], requires_grad=False)
        self.k = nn.Parameter(fused[1], requires_grad=False)
        self.register_buffer("scale", torch.randn(7, dtype=torch.float16))
        self.register_buffer("cpu_only", torch.arange(3), persistent=False)


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    model = _Model()
    for name, buf in list(model._buffers.items()):
        if name != "cpu_only":
            model._buffers[name] = buf.cuda()
    for p in model.parameters():
        p.data = p.data.cuda()
    # Re-establish the views on CUDA, as the fused layer would allocate them.
    fused = torch.randn(2, 16, 32, device="cuda")
    model.q.data, model.k.data = fused[0], fused[1]
    model.wt.data = torch.randn(48, 32, device="cuda").t()
    model._buffers["cpu_only"] = torch.arange(3)
    return model


def _snapshot(model):
    return {
        name: (t.detach().clone(), t.stride(), t.storage_offset())
        for name, t in list(model.named_parameters()) + list(model.named_buffers())
    }


def test_pack_keeps_values_layout_ties_and_shared_storage():
    from vllm.model_executor.model_loader.weight_cache.vmm import WeightArena

    model = _make_model()
    before = _snapshot(model)
    tied = model.embed.weight
    arena = WeightArena.pack(model, torch.cuda.current_device())
    try:
        assert model.head.weight is model.embed.weight is tied
        for name, t in list(model.named_parameters()) + list(model.named_buffers()):
            value, stride, offset = before[name]
            assert torch.equal(t, value.to(t.device)), name
            assert t.stride() == stride and t.storage_offset() == offset, name
            if t.is_cuda:
                ptr = t.untyped_storage().data_ptr()
                assert arena.allocation.va <= ptr < arena.allocation.va + arena.size
        assert not model.wt.is_contiguous()
        assert (
            model.q.untyped_storage().data_ptr() == model.k.untyped_storage().data_ptr()
        )
        assert not model.cpu_only.is_cuda
    finally:
        arena.close()


def _exporter(conn: socket.socket):
    from vllm.device_allocator.cuda_vmm import send_fd
    from vllm.model_executor.model_loader.weight_cache.daemon import export_entries
    from vllm.model_executor.model_loader.weight_cache.vmm import WeightArena

    torch.cuda.set_device(0)
    model = _make_model()
    snapshot = {k: v[0].cpu() for k, v in _snapshot(model).items()}
    arena = WeightArena.pack(model, 0)
    entries, aliases = export_entries(model, arena)
    payload = pickle.dumps((entries, aliases, arena.size, snapshot))
    conn.sendall(len(payload).to_bytes(8, "big") + payload)
    fd = arena.export_fd()
    send_fd(conn, fd)
    os.close(fd)
    conn.recv(1)  # wait until the parent has mapped the arena
    os._exit(0)  # die without releasing anything


def _recv_exact(conn, n):
    data = b""
    while len(data) < n:
        data += conn.recv(n - len(data))
    return data


def test_engine_mapping_survives_daemon_exit():
    from vllm.device_allocator.cuda_vmm import ShareableAllocation, recv_fd

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    proc = multiprocessing.get_context("spawn").Process(target=_exporter, args=(child,))
    proc.start()
    child.close()
    try:
        size = int.from_bytes(_recv_exact(parent, 8), "big")
        entries, aliases, arena_size, snapshot = pickle.loads(_recv_exact(parent, size))
        fd = recv_fd(parent)
        torch.cuda.set_device(0)
        allocation = ShareableAllocation.import_fd(fd, arena_size, 0)
        os.close(fd)
        tensors = {name: e.rebuild(0, allocation) for name, e in entries.items()}
        parent.send(b"x")
        proc.join(timeout=60)
        assert proc.exitcode == 0
    finally:
        parent.close()

    assert aliases == {"head.weight": "embed.weight"}
    # Fill the rest of the GPU so freed memory would be reused and overwritten.
    free, _ = torch.cuda.mem_get_info()
    filler = torch.full(
        ((free - (1 << 30)) // 4,), -1, dtype=torch.int32, device="cuda"
    )
    torch.cuda.synchronize()
    for name, t in tensors.items():
        assert torch.equal(t.cpu(), snapshot[name]), name
    tensors["scale"].add_(1)  # still writable
    torch.cuda.synchronize()
    del filler
    allocation.close()
