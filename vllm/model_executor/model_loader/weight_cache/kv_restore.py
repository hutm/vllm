# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV restore through the weight cache daemon.

With ``--load-format ipc_cache --model-loader-extra-config '{"kv_restore":
true}'`` an engine keeps its KV cache, and its prefix cache, across restarts
of the engine process, as long as the ``vllm preload`` daemon keeps running:

- Each worker allocates its KV cache as one shareable CUDA VMM allocation
  (POSIX fd) and deposits the fd with its GPU's daemon over a persistent
  *KV session*. The open session is the writer lease. The daemon holds only
  the fd, which keeps the memory alive after the engine exits.
- The scheduler streams its prefix-cache directory to the daemon of worker 0:
  a block is committed only once the step that wrote it has completed, and an
  evicted block is retracted before its memory can be rewritten.
- A restarted engine whose KV layout matches attaches to the same memory
  (no zero-fill) and imports the directory into its prefix cache. If the
  layout changed, the daemon drops the retained pool and the engine starts
  with a fresh one.

Only one engine at a time holds a pool; a second one runs without restore.

Lifecycle: the daemon cannot tell a clean engine exit from a crash, and
treats both the same: the lease ends and the pool and its directory are kept
for the next engine. They are freed when the daemon exits, when an engine
with a different KV layout attaches, or with ``vllm preload --kv-drop``.
"""

import glob
import hashlib
import os
import socket
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    SOCKET_NAME_TEMPLATE,
    get_socket_dir,
    recv_msg,
    send_msg,
)
from vllm.v1.core.block_pool import BlockPool

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.device_allocator.cuda_vmm import ShareableAllocation
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

KV_RESTORE_BLOCK_POOL = (
    "vllm.model_executor.model_loader.weight_cache.kv_restore.KVRestoreBlockPool"
)


def kv_restore_enabled(vllm_config: "VllmConfig") -> bool:
    load_config = vllm_config.load_config
    extra = load_config.model_loader_extra_config or {}
    return load_config.load_format == "ipc_cache" and bool(extra.get("kv_restore"))


def _digest(parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def kv_layout_summary(vllm_config: "VllmConfig") -> dict[str, Any]:
    """The configuration a retained KV pool depends on, readable in logs."""
    cache_config = vllm_config.cache_config
    return {
        "model": vllm_config.model_config.model,
        "dtype": str(vllm_config.model_config.dtype),
        "kv_cache_dtype": cache_config.cache_dtype,
        "kv_cache_layout": str(cache_config.get_resolved_kv_cache_layout()),
        "block_size": cache_config.block_size,
    }


def kv_layout_key(
    kv_cache_config: "KVCacheConfig", nbytes: int, vllm_config: "VllmConfig"
) -> str:
    """Everything that must match for a worker to reuse KV bytes as-is."""
    return _digest(
        (
            sorted(kv_layout_summary(vllm_config).items()),
            nbytes,
            kv_cache_config.num_blocks,
            [(t.size, tuple(t.layers)) for t in kv_cache_config.kv_cache_tensors],
            [
                (tuple(g.layer_names), repr(g.kv_cache_spec))
                for g in kv_cache_config.kv_cache_groups
            ],
        )
    )


def layout_mismatch(held: dict | None, new: dict | None) -> list[str]:
    """Names of the layout fields that differ (empty if only the detailed
    per-layer layout does)."""
    held, new = held or {}, new or {}
    return [k for k in sorted(held.keys() | new.keys()) if held.get(k) != new.get(k)]


# --------------------------------------------------------------------------
# Daemon side
# --------------------------------------------------------------------------


@dataclass
class _Pool:
    key: str
    fds: list[int]
    size: int
    nbytes: int
    num_blocks: int
    layout: dict = field(default_factory=dict)
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    holder: object | None = None


@dataclass
class _Directory:
    key: str
    generations: tuple[str, ...]
    # block_id -> (block hash, num tokens covered)
    entries: dict[int, tuple[bytes, int | None]] = field(default_factory=dict)
    holder: object | None = None


class KVRestoreStore:
    """The daemon's KV state: at most one pool (this GPU's KV memory, held as
    fds) and one prefix directory (kept by the daemon of worker 0)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pool: _Pool | None = None
        self.directory: _Directory | None = None

    def probe(self) -> dict | None:
        with self._lock:
            pool = self.pool
            if pool is None:
                return None
            return {
                "num_blocks": pool.num_blocks,
                "nbytes": pool.nbytes,
                "size": pool.size,
                "layout": pool.layout,
                "busy": pool.holder is not None,
            }

    def attach(self, holder: object, key: str, layout: dict) -> tuple[str, Any]:
        """``("restore", pool)``, ``("fresh", mismatched fields or None)`` or
        ``("busy", None)``."""
        with self._lock:
            pool = self.pool
            if pool is not None and pool.holder is not None:
                return "busy", None
            if pool is not None and pool.key == key:
                pool.holder = holder
                return "restore", pool
            mismatch = None
            if pool is not None:
                mismatch = layout_mismatch(pool.layout, layout)
                logger.warning(
                    "KV restore: the engine's KV cache layout differs from the "
                    "retained pool's (%s); dropping the retained pool (%.2f GiB, "
                    "%d blocks) and its prefix cache",
                    ", ".join(mismatch) or "per-layer layout",
                    pool.size / (1 << 30),
                    pool.num_blocks,
                )
            self._drop_locked()
            return "fresh", mismatch

    def deposit(
        self,
        holder: object,
        key: str,
        fds: list[int],
        size: int,
        nbytes: int,
        num_blocks: int,
        layout: dict,
    ) -> _Pool:
        with self._lock:
            self._drop_locked()
            self.pool = _Pool(key, fds, size, nbytes, num_blocks, layout, holder=holder)
            return self.pool

    def release(self, holder: object) -> None:
        with self._lock:
            if self.pool is not None and self.pool.holder is holder:
                self.pool.holder = None
            if self.directory is not None and self.directory.holder is holder:
                self.directory.holder = None

    def open_directory(
        self, holder: object, key: str, generations: tuple[str, ...]
    ) -> list[tuple[int, bytes, int | None]] | None:
        """Entries to import, or None if another scheduler holds it."""
        with self._lock:
            d = self.directory
            if d is not None and d.holder is not None:
                return None
            if d is None or d.key != key or d.generations != generations:
                d = self.directory = _Directory(key, generations)
            d.holder = holder
            return [(bid, h, n) for bid, (h, n) in d.entries.items()]

    def apply(self, holder: object, msg: dict) -> None:
        with self._lock:
            d = self.directory
            if d is None or d.holder is not holder:
                return
            op = msg.get("op")
            if op == "commit":
                for block_id, block_hash, num_tokens in msg["blocks"]:
                    d.entries[block_id] = (block_hash, num_tokens)
            elif op == "evict":
                for block_id in msg["block_ids"]:
                    d.entries.pop(block_id, None)
            elif op == "clear":
                d.entries.clear()
            # Unknown ops are ignored, so newer engines can add messages.

    def drop(self, force: bool = False) -> int | None:
        """Drop the retained pool and directory. Unless ``force``, refuse
        while an engine holds them. Returns the bytes released, or None if
        refused. A forced drop only closes the daemon's fds: an engine using
        the pool keeps its own mapping until it exits."""
        with self._lock:
            pool, d = self.pool, self.directory
            if not force and (
                (pool is not None and pool.holder is not None)
                or (d is not None and d.holder is not None)
            ):
                return None
            size = 0 if pool is None else pool.size
            self._drop_locked()
        if size:
            logger.info(
                "KV restore: dropped the retained KV pool (%.2f GiB) on request",
                size / (1 << 30),
            )
        return size

    def _drop_locked(self) -> None:
        if self.pool is not None:
            for fd in self.pool.fds:
                os.close(fd)
            self.pool = None
        self.directory = None


def handle_kv_request(store: KVRestoreStore, conn: socket.socket, first: dict) -> None:
    """Answer a one-shot KV request: ``kv_probe`` or ``kv_drop``."""
    if first["cmd"] == "kv_probe":
        send_msg(conn, {"status": "ok", "pool": store.probe()})
        return
    freed = store.drop(force=bool(first.get("force")))
    if freed is None:
        send_msg(conn, {"status": "busy"})
    else:
        send_msg(conn, {"status": "ok", "freed_bytes": freed})


def serve_kv_session(store: KVRestoreStore, conn: socket.socket, first: dict) -> None:
    """Run one persistent KV session (``kv_attach`` or ``kv_directory``) on
    its own thread; the open connection is the lease."""
    from vllm.device_allocator.cuda_vmm import recv_fd, send_fd

    holder = object()
    try:
        if first["cmd"] == "kv_attach":
            layout = first.get("layout") or {}
            status, result = store.attach(holder, first["key"], layout)
            if status == "restore":
                send_msg(
                    conn,
                    {
                        "status": "restore",
                        "generation": result.generation,
                        "size": result.size,
                        "num_fds": len(result.fds),
                    },
                )
                for fd in result.fds:
                    send_fd(conn, fd)
            elif status == "busy":
                send_msg(conn, {"status": "busy"})
                return
            else:
                send_msg(conn, {"status": "fresh", "layout_mismatch": result})
                deposit = recv_msg(conn)
                if deposit.get("cmd") != "kv_deposit":
                    return
                fds = [recv_fd(conn) for _ in range(deposit["num_fds"])]
                pool = store.deposit(
                    holder,
                    first["key"],
                    fds,
                    deposit["size"],
                    first["nbytes"],
                    first["num_blocks"],
                    layout,
                )
                send_msg(conn, {"status": "ok", "generation": pool.generation})
            # Hold the lease until the engine goes away.
            while conn.recv(4096):
                pass
        elif first["cmd"] == "kv_directory":
            entries = store.open_directory(
                holder, first["key"], tuple(first["generations"])
            )
            if entries is None:
                send_msg(conn, {"status": "busy"})
                return
            send_msg(conn, {"status": "ok", "entries": entries})
            while True:
                store.apply(holder, recv_msg(conn))
    except (ConnectionError, EOFError, OSError):
        pass
    finally:
        store.release(holder)
        conn.close()


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------


def _connect(socket_path: str, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(socket_path)
    except OSError:
        sock.close()
        raise
    return sock


class KVRestoreClient:
    """A worker's KV session with its GPU's daemon."""

    def __init__(self, socket_path: str, timeout: float = 60.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.session: socket.socket | None = None
        self.allocation: ShareableAllocation | None = None
        self.restored = False
        self.generation: str | None = None

    def probe(self) -> dict | None:
        try:
            with _connect(self.socket_path, self.timeout) as sock:
                send_msg(sock, {"cmd": "kv_probe"})
                return recv_msg(sock).get("pool")
        except OSError as e:
            logger.warning("KV restore: cannot reach the daemon (%s)", e)
            return None

    def attach(
        self,
        key: str,
        nbytes: int,
        num_blocks: int,
        device: torch.device,
        layout: dict | None = None,
    ) -> torch.Tensor | None:
        """Return the KV backing buffer (restored or fresh), or None to let the
        engine allocate it normally (daemon busy or unreachable)."""
        from vllm.device_allocator.cuda_vmm import (
            ShareableAllocation,
            recv_fd,
            send_fd,
        )

        try:
            sock = _connect(self.socket_path, self.timeout)
        except OSError as e:
            logger.warning("KV restore: cannot reach the daemon (%s)", e)
            return None
        device_index = device.index if device.index is not None else 0
        send_msg(
            sock,
            {
                "cmd": "kv_attach",
                "key": key,
                "nbytes": nbytes,
                "num_blocks": num_blocks,
                "layout": {**(layout or {}), "nbytes": nbytes},
            },
        )
        reply = recv_msg(sock)
        status = reply.get("status")
        if status == "restore":
            fds = [recv_fd(sock) for _ in range(reply["num_fds"])]
            try:
                allocation = ShareableAllocation.import_fd(
                    fds[0], reply["size"], device_index
                )
            finally:
                for fd in fds:
                    os.close(fd)
            self.restored = True
            self.generation = reply["generation"]
        elif status == "fresh":
            mismatch = reply.get("layout_mismatch")
            if mismatch is not None:
                logger.warning(
                    "KV restore: the daemon's retained KV pool has a different "
                    "layout (%s) and was dropped; starting with an empty KV cache",
                    ", ".join(mismatch) or "per-layer layout",
                )
            allocation = ShareableAllocation.create(nbytes, device_index)
            allocation.tensor(0, allocation.size).zero_()
            torch.accelerator.synchronize(device_index)
            fd = allocation.export_fd()
            try:
                send_msg(
                    sock, {"cmd": "kv_deposit", "size": allocation.size, "num_fds": 1}
                )
                send_fd(sock, fd)
            finally:
                os.close(fd)
            self.generation = recv_msg(sock)["generation"]
        else:
            sock.close()
            if status == "busy":
                logger.warning(
                    "KV restore: another engine holds this GPU's KV pool; "
                    "starting without restore"
                )
            else:
                logger.warning(
                    "KV restore: the daemon refused the KV session (%s); "
                    "starting without restore",
                    reply.get("message", status),
                )
            return None
        sock.settimeout(None)
        self.session = sock
        self.allocation = allocation
        logger.info(
            "KV restore: %s KV cache (%.2f GiB) in a daemon-held allocation",
            "re-attached the previous" if self.restored else "allocated a new",
            nbytes / (1 << 30),
        )
        return allocation.tensor(0, nbytes).view(torch.int8)

    def status(self) -> dict:
        return {
            "socket_path": self.socket_path,
            "restored": self.restored,
            "generation": self.generation,
        }


def drop_kv_pools(
    socket_dir: str | None, force: bool = False
) -> tuple[bool, list[str]]:
    """Ask every daemon in ``socket_dir`` to drop its retained KV pool.
    Returns whether all of them did, and one line per daemon."""
    directory = get_socket_dir(socket_dir)
    pattern = SOCKET_NAME_TEMPLATE.format(gpu_uuid="*", role="")
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        return False, [f"No weight cache daemon sockets in {directory}"]
    ok, lines = True, []
    for path in paths:
        try:
            with _connect(path, 10.0) as sock:
                send_msg(sock, {"cmd": "kv_drop", "force": force})
                reply = recv_msg(sock)
        except OSError as e:
            ok = False
            lines.append(f"{path}: unreachable ({e})")
            continue
        if reply.get("status") == "ok":
            lines.append(f"{path}: freed {reply['freed_bytes'] / (1 << 30):.2f} GiB")
        else:
            ok = False
            lines.append(
                f"{path}: {reply.get('status')}; an engine is using its KV pool "
                "(retry after it exits, or pass --force)"
            )
    return ok, lines


# --------------------------------------------------------------------------
# Scheduler side
# --------------------------------------------------------------------------


def directory_key(block_pool: BlockPool) -> str:
    from vllm.v1.core import kv_cache_utils

    return _digest(
        (
            bytes(kv_cache_utils.NONE_HASH),
            block_pool.num_gpu_blocks,
            block_pool.hash_block_size,
        )
    )


class KVRestoreBlockPool(BlockPool):
    """Publishes the prefix cache to the weight cache daemon.

    Blocks cached while step N is scheduled are committed when step N's
    output is processed, i.e. after the GPU has written them. Evictions are
    sent synchronously, before the block can be rewritten.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._session: socket.socket | None = None
        self._unsealed: list[tuple[int, Any, int | None]] = []
        self._in_flight: dict[int, list[tuple[int, Any, int | None]]] = {}

    def start_kv_restore(
        self, socket_path: str, generations: list[str | None], restored: bool
    ) -> int:
        """Open the directory session; import it if every worker re-attached
        its previous KV pool. Returns the number of imported blocks."""
        if not self.enable_caching or None in generations:
            return 0
        sock = _connect(socket_path, 60.0)
        send_msg(
            sock,
            {
                "cmd": "kv_directory",
                "key": directory_key(self),
                "generations": list(generations),
            },
        )
        reply = recv_msg(sock)
        if reply.get("status") != "ok":
            sock.close()
            logger.warning(
                "KV restore: the prefix directory is held by another engine; "
                "the prefix cache will not be kept"
            )
            return 0
        sock.settimeout(None)
        self._session = sock
        entries = reply["entries"] if restored else []
        if not entries:
            return 0
        entries.sort()
        imported = self.import_cached_blocks(
            [h for _, h, _ in entries],
            [b for b, _, _ in entries],
            [n for _, _, n in entries],
        )
        stale = sorted({b for b, _, _ in entries} - set(imported))
        if stale:
            self._send({"op": "evict", "block_ids": stale})
        logger.info("KV restore: imported %d cached blocks", len(imported))
        return len(imported)

    def _send(self, msg: dict) -> None:
        if self._session is None:
            return
        try:
            send_msg(self._session, msg)
        except OSError as e:
            logger.warning("KV restore: lost the directory session (%s)", e)
            self._session = None

    def on_blocks_cached(self, blocks, kv_cache_group_id: int) -> None:
        if self._session is None:
            return
        for block in blocks:
            if block.block_hash is not None:
                self._unsealed.append(
                    (block.block_id, block.block_hash, block.block_hash_num_tokens)
                )

    def on_step_scheduled(self, scheduler_output: "SchedulerOutput") -> None:
        if self._session is None:
            return
        self._in_flight[id(scheduler_output)] = self._unsealed
        self._unsealed = []

    def on_step_output_processed(self, scheduler_output: "SchedulerOutput") -> None:
        batch = self._in_flight.pop(id(scheduler_output), None)
        if not batch:
            return
        # Only blocks still cached under the same hash: a block evicted (and
        # possibly re-cached) since then is not this step's content.
        committed = [
            (block_id, bytes(block_hash), num_tokens)
            for block_id, block_hash, num_tokens in batch
            if self.blocks[block_id].block_hash == block_hash
        ]
        if committed:
            self._send({"op": "commit", "blocks": committed})

    def on_cached_block_evicted(self, block, block_hashes) -> None:
        if self._session is None:
            return
        block_id = block.block_id
        for pending in (self._unsealed, *self._in_flight.values()):
            pending[:] = [e for e in pending if e[0] != block_id]
        self._send({"op": "evict", "block_ids": [block_id]})

    def reset_prefix_cache(self) -> bool:
        reset = super().reset_prefix_cache()
        if reset:
            self._unsealed.clear()
            self._in_flight.clear()
            self._send({"op": "clear"})
        return reset
