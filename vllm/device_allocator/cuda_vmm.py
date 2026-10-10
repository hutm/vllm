# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shareable CUDA virtual-memory allocations, via the CUDA driver API.

A pure-ctypes wrapper (like ``cuda_wrapper.py`` for cudart) around the few
driver calls needed to create a device allocation that other processes can
map: ``cuMemCreate`` with a POSIX file-descriptor handle, exported with
``cuMemExportToShareableHandle`` and imported with
``cuMemImportFromShareableHandle``. An importer holds its own reference to
the physical memory, so its mapping stays valid after the exporter exits.
"""

import ctypes
import os
from functools import cache

import torch

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_ulonglong
CUmemGenericAllocationHandle = ctypes.c_ulonglong

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", CUmemAllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


_SIGNATURES = {
    "cuGetErrorString": [CUresult, ctypes.POINTER(ctypes.c_char_p)],
    "cuMemGetAllocationGranularity": [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(CUmemAllocationProp),
        ctypes.c_int,
    ],
    "cuMemCreate": [
        ctypes.POINTER(CUmemGenericAllocationHandle),
        ctypes.c_size_t,
        ctypes.POINTER(CUmemAllocationProp),
        ctypes.c_ulonglong,
    ],
    "cuMemRelease": [CUmemGenericAllocationHandle],
    "cuMemAddressReserve": [
        ctypes.POINTER(CUdeviceptr),
        ctypes.c_size_t,
        ctypes.c_size_t,
        CUdeviceptr,
        ctypes.c_ulonglong,
    ],
    "cuMemAddressFree": [CUdeviceptr, ctypes.c_size_t],
    "cuMemMap": [
        CUdeviceptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        CUmemGenericAllocationHandle,
        ctypes.c_ulonglong,
    ],
    "cuMemUnmap": [CUdeviceptr, ctypes.c_size_t],
    "cuMemSetAccess": [
        CUdeviceptr,
        ctypes.c_size_t,
        ctypes.POINTER(CUmemAccessDesc),
        ctypes.c_size_t,
    ],
    "cuMemExportToShareableHandle": [
        ctypes.c_void_p,
        CUmemGenericAllocationHandle,
        ctypes.c_int,
        ctypes.c_ulonglong,
    ],
    "cuMemImportFromShareableHandle": [
        ctypes.POINTER(CUmemGenericAllocationHandle),
        ctypes.c_void_p,
        ctypes.c_int,
    ],
}


@cache
def _libcuda() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    for name, argtypes in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = CUresult
        fn.argtypes = argtypes
    return lib


def _check(result: int, what: str) -> None:
    if result != 0:
        msg = ctypes.c_char_p()
        _libcuda().cuGetErrorString(result, ctypes.byref(msg))
        detail = msg.value.decode() if msg.value else "unknown error"
        raise RuntimeError(f"{what} failed: {detail} ({result})")


def _device_prop(device_index: int) -> CUmemAllocationProp:
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_index
    return prop


def allocation_granularity(device_index: int) -> int:
    gran = ctypes.c_size_t()
    prop = _device_prop(device_index)
    _check(
        _libcuda().cuMemGetAllocationGranularity(
            ctypes.byref(gran), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_MINIMUM
        ),
        "cuMemGetAllocationGranularity",
    )
    return gran.value


def _ensure_context(device_index: int) -> None:
    # Driver calls use the calling thread's current context; let torch make
    # the device's primary context current.
    torch.cuda.set_device(device_index)
    torch.cuda.synchronize(device_index)


def _map(handle: int, size: int, device_index: int) -> int:
    lib = _libcuda()
    va = CUdeviceptr()
    _check(
        lib.cuMemAddressReserve(ctypes.byref(va), size, 0, 0, 0),
        "cuMemAddressReserve",
    )
    try:
        _check(lib.cuMemMap(va, size, 0, handle, 0), "cuMemMap")
        access = CUmemAccessDesc()
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        try:
            _check(
                lib.cuMemSetAccess(va, size, ctypes.byref(access), 1),
                "cuMemSetAccess",
            )
        except BaseException:
            lib.cuMemUnmap(va, size)
            raise
    except BaseException:
        lib.cuMemAddressFree(va, size)
        raise
    return va.value


class ShareableAllocation:
    """One device allocation mapped into this process, exportable by fd.

    Create one with :meth:`create` (exporter) or :meth:`import_fd`
    (importer). Each process holds its own reference: the physical memory is
    freed once every process that created or imported it has called
    :meth:`close` or exited.
    """

    def __init__(self, handle: int, va: int, size: int, device_index: int):
        self.handle = handle
        self.va = va
        self.size = size
        self.device_index = device_index

    @classmethod
    def create(cls, nbytes: int, device_index: int) -> "ShareableAllocation":
        _ensure_context(device_index)
        gran = allocation_granularity(device_index)
        size = max(gran, -(-nbytes // gran) * gran)
        handle = CUmemGenericAllocationHandle()
        prop = _device_prop(device_index)
        _check(
            _libcuda().cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), 0),
            "cuMemCreate",
        )
        try:
            va = _map(handle.value, size, device_index)
        except BaseException:
            _libcuda().cuMemRelease(handle)
            raise
        return cls(handle.value, va, size, device_index)

    @classmethod
    def import_fd(cls, fd: int, size: int, device_index: int) -> "ShareableAllocation":
        """Import an allocation exported by another process. Does not close
        ``fd``."""
        _ensure_context(device_index)
        handle = CUmemGenericAllocationHandle()
        _check(
            _libcuda().cuMemImportFromShareableHandle(
                ctypes.byref(handle),
                ctypes.c_void_p(fd),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            ),
            "cuMemImportFromShareableHandle",
        )
        try:
            va = _map(handle.value, size, device_index)
        except BaseException:
            _libcuda().cuMemRelease(handle)
            raise
        return cls(handle.value, va, size, device_index)

    def export_fd(self) -> int:
        """A new POSIX fd for this allocation; the caller closes it."""
        fd = ctypes.c_int(-1)
        _check(
            _libcuda().cuMemExportToShareableHandle(
                ctypes.byref(fd),
                self.handle,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            ),
            "cuMemExportToShareableHandle",
        )
        return fd.value

    def tensor(self, offset: int, nbytes: int) -> torch.Tensor:
        """A non-owning uint8 view of ``[offset, offset + nbytes)``. Keep this
        allocation open while the view is in use."""
        if offset < 0 or offset + nbytes > self.size:
            raise ValueError(
                f"range [{offset}, {offset + nbytes}) outside the "
                f"{self.size}-byte allocation"
            )
        device = torch.device("cuda", self.device_index)
        storage = torch._C._construct_storage_from_data_pointer(
            self.va + offset, device, nbytes
        )
        return torch.empty(0, dtype=torch.uint8, device=device).set_(
            storage, 0, (nbytes,)
        )

    def close(self) -> None:
        if self.va:
            torch.cuda.synchronize(self.device_index)
            lib = _libcuda()
            lib.cuMemUnmap(self.va, self.size)
            lib.cuMemAddressFree(self.va, self.size)
            lib.cuMemRelease(self.handle)
            self.va = 0


def send_fd(sock, fd: int) -> None:
    """Send one fd over a connected Unix socket (SCM_RIGHTS)."""
    import socket

    socket.send_fds(sock, [b"F"], [fd])


def recv_fd(sock) -> int:
    import socket

    data, fds, _, _ = socket.recv_fds(sock, 1, 1)
    if data != b"F" or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError("expected one file descriptor on the socket")
    return fds[0]
