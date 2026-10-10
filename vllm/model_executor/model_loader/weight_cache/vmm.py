# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``vmm_fd`` weight cache transport.

The daemon packs every CUDA storage of its model into one shareable CUDA VMM
allocation, re-points the model at it, and sends each engine one file
descriptor plus per-tensor references. The engine maps the allocation and
rebuilds each tensor with its original shape, stride and storage offset.

Unlike CUDA IPC (``cuda_ipc``), each engine holds its own reference to the
physical memory, ownership is documented by the driver, and only a Unix
socket is needed (no shared PID namespace or ``pidfd_getfd``).
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm.device_allocator.cuda_vmm import ShareableAllocation

CUDA_IPC = "cuda_ipc"
VMM_FD = "vmm_fd"
WEIGHT_CACHE_TRANSPORTS = (CUDA_IPC, VMM_FD)

# Storage alignment inside the arena; matches the caching allocator's.
STORAGE_ALIGNMENT = 512


@dataclass(frozen=True)
class VmmTensorRef:
    """Where a tensor lives inside the shared allocation."""

    storage_offset_bytes: int
    """Byte offset of the tensor's storage from the start of the allocation."""
    storage_nbytes: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    """Element offset of the tensor within its storage."""

    def rebuild(self, allocation: ShareableAllocation) -> torch.Tensor:
        dtype = getattr(torch, self.dtype)
        storage = allocation.tensor(self.storage_offset_bytes, self.storage_nbytes)
        return storage.view(dtype).as_strided(
            self.shape, self.stride, self.storage_offset
        )


class WeightArena:
    """The daemon's shareable copy of its model's CUDA tensors."""

    def __init__(self, allocation: ShareableAllocation):
        self.allocation = allocation

    @classmethod
    def pack(cls, model: nn.Module, device_index: int) -> "WeightArena":
        """Copy every CUDA storage of ``model``'s parameters and buffers into
        one shareable allocation and re-point the model at the copies.

        Views that share a storage (tied or fused weights) keep sharing it.
        Peak device memory is twice the model until the originals are freed.
        """
        tensors = _model_tensors(model)
        layout: dict[int, int] = {}  # storage data_ptr -> arena offset
        sizes: dict[int, int] = {}
        offset = 0
        for tensor in tensors.values():
            if not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key in layout:
                continue
            offset = -(-offset // STORAGE_ALIGNMENT) * STORAGE_ALIGNMENT
            layout[key] = offset
            sizes[key] = storage.nbytes()
            offset += storage.nbytes()

        allocation = ShareableAllocation.create(max(offset, 1), device_index)
        arena = cls(allocation)
        try:
            copied: dict[int, torch.Tensor] = {}
            for tensor in tensors.values():
                if not tensor.is_cuda:
                    continue
                storage = tensor.untyped_storage()
                key = storage.data_ptr()
                if key not in copied:
                    dst = allocation.tensor(layout[key], sizes[key])
                    src = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    src.set_(storage, 0, (sizes[key],))
                    dst.copy_(src)
                    copied[key] = dst
            torch.cuda.synchronize(device_index)
            replacements = {
                id(tensor): VmmTensorRef(
                    storage_offset_bytes=layout[tensor.untyped_storage().data_ptr()],
                    storage_nbytes=sizes[tensor.untyped_storage().data_ptr()],
                    dtype=str(tensor.dtype).removeprefix("torch."),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                    storage_offset=tensor.storage_offset(),
                ).rebuild(allocation)
                for tensor in tensors.values()
                if tensor.is_cuda
            }
            _rebind(model, replacements)
        except BaseException:
            allocation.close()
            raise
        return arena

    def ref(self, tensor: torch.Tensor) -> VmmTensorRef:
        """Reference for a tensor whose storage lies in this arena."""
        storage = tensor.untyped_storage()
        offset = storage.data_ptr() - self.allocation.va
        if not 0 <= offset < self.allocation.size:
            raise ValueError("tensor storage is outside the weight arena")
        return VmmTensorRef(
            storage_offset_bytes=offset,
            storage_nbytes=storage.nbytes(),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
        )

    @property
    def size(self) -> int:
        return self.allocation.size

    def export_fd(self) -> int:
        return self.allocation.export_fd()

    def close(self) -> None:
        self.allocation.close()


def _model_tensors(model: nn.Module) -> dict[int, torch.Tensor]:
    tensors: dict[int, torch.Tensor] = {}
    for _, param in model.named_parameters(remove_duplicate=False):
        tensors.setdefault(id(param), param)
    for _, buffer in model.named_buffers(remove_duplicate=False):
        tensors.setdefault(id(buffer), buffer)
    return tensors


def _rebind(model: nn.Module, replacements: dict[int, torch.Tensor]) -> None:
    """Point every parameter and buffer at its arena copy, keeping tied
    parameters the same object."""
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and id(param) in replacements:
                # In place, so every module sharing this Parameter sees it.
                param.data = replacements[id(param)]
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and id(buffer) in replacements:
                module._buffers[name] = replacements[id(buffer)]
