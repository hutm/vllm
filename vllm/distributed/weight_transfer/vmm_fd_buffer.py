# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A packed IPC staging buffer shared as a CUDA VMM allocation by fd.

The trainer allocates the buffer with ``cuMemCreate`` (POSIX fd handle) and
serves its fd on a private Unix socket. The worker connects, receives the fd
(SCM_RIGHTS), and maps the buffer, holding its own reference to the memory.

Compared with ``reduce_tensor``:

- No PyTorch cross-process refcount, so a worker can keep or drop its mapping
  in any order.
- Works between unrelated or sibling processes. PyTorch's own sharing of VMM
  (expandable-segment) memory uses ``pidfd_getfd``, which the default Yama
  ptrace policy only allows from an ancestor of the exporter.
"""

import os
import socket
import struct
import tempfile
import threading
import uuid

from vllm.device_allocator.cuda_vmm import ShareableAllocation, recv_fd, send_fd
from vllm.logger import init_logger

logger = init_logger(__name__)

VMM_FD_HANDLE = "vmm_fd"
_PEERCRED = struct.Struct("3i")  # pid, uid, gid


def is_vmm_fd_handle(args) -> bool:
    return len(args) > 0 and args[0] == VMM_FD_HANDLE


def _peer_uid(conn: socket.socket) -> int:
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
    return _PEERCRED.unpack(creds)[1]


class SharedPackedBuffer:
    """Trainer side: the staging buffer and the socket that hands out its fd.

    Only processes of the same user may connect. The handle tuple
    ``("vmm_fd", socket_path, size, transfer_id)`` replaces the
    ``reduce_tensor`` arguments in the update info.
    """

    def __init__(self, nbytes: int, device_index: int):
        self.allocation = ShareableAllocation.create(nbytes, device_index)
        self.transfer_id = uuid.uuid4().hex
        self._dir = tempfile.mkdtemp(prefix="vllm_wt_")
        os.chmod(self._dir, 0o700)
        self.socket_path = os.path.join(self._dir, "buffer.sock")
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server.listen(64)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="vllm-weight-transfer-fd", daemon=True
        )
        self._thread.start()

    def handle(self) -> tuple:
        return (VMM_FD_HANDLE, self.socket_path, self.allocation.size, self.transfer_id)

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                try:
                    if _peer_uid(conn) != os.getuid():
                        logger.warning("Rejected weight transfer peer of another user")
                        continue
                    fd = self.allocation.export_fd()
                    try:
                        send_fd(conn, fd)
                    finally:
                        os.close(fd)
                except OSError as e:
                    logger.warning("Weight transfer fd handoff failed: %s", e)

    def close(self) -> None:
        """Stop serving and drop the trainer's reference; workers that mapped
        the buffer keep theirs until they close it."""
        self._closed.set()
        self._server.close()
        self._thread.join(timeout=5)
        try:
            os.unlink(self.socket_path)
            os.rmdir(self._dir)
        except OSError:
            pass
        self.allocation.close()


def import_shared_buffer(args, device_index: int) -> ShareableAllocation:
    """Worker side: fetch the buffer's fd from the trainer and map it."""
    _, socket_path, size, _ = args
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(60)
        conn.connect(socket_path)
        fd = recv_fd(conn)
    try:
        return ShareableAllocation.import_fd(fd, int(size), device_index)
    finally:
        os.close(fd)
