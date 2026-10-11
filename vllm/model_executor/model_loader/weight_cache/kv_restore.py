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
  (no zero-fill) and imports the directory into its prefix cache.

Only one engine at a time writes a pool (the *writer*); another writer runs
without restore. With ``"kv_restore_role": "standby"`` an engine instead maps
the writer's pool next to it (no second copy, no writes) and stays idle until
an orchestrator calls the ``kv_restore_promote`` engine utility. Promotion
revokes the writer at once (it may be frozen rather than dead): the standby
adopts the committed directory, and blocks the revoked writer may still write
(allocated but uncommitted, or touched after revocation) are quarantined
until its session closes plus a grace period.
"""

import hashlib
import os
import select
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.protocol import (
    recv_msg,
    send_msg,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.device_allocator.cuda_vmm import ShareableAllocation
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

KV_RESTORE_BLOCK_POOL = (
    "vllm.model_executor.model_loader.weight_cache.kv_restore.KVRestoreBlockPool"
)


KV_RESTORE_ROLES = ("writer", "standby")
# How long a standby waits for the writer's pool at startup.
DEFAULT_STANDBY_WAIT_S = 1800.0


def kv_restore_enabled(vllm_config: "VllmConfig") -> bool:
    load_config = vllm_config.load_config
    extra = load_config.model_loader_extra_config or {}
    return load_config.load_format == "ipc_cache" and bool(extra.get("kv_restore"))


def kv_restore_role(vllm_config: "VllmConfig") -> str:
    extra = vllm_config.load_config.model_loader_extra_config or {}
    role = extra.get("kv_restore_role", "writer")
    if role not in KV_RESTORE_ROLES:
        raise ValueError(
            f"kv_restore_role must be one of {KV_RESTORE_ROLES}, got {role!r}"
        )
    return role


def kv_restore_wait_s(vllm_config: "VllmConfig") -> float:
    extra = vllm_config.load_config.model_loader_extra_config or {}
    return float(extra.get("kv_restore_wait_s", DEFAULT_STANDBY_WAIT_S))


def _digest(parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def kv_layout_key(
    kv_cache_config: "KVCacheConfig", nbytes: int, vllm_config: "VllmConfig"
) -> str:
    """Everything that must match for a worker to reuse KV bytes as-is."""
    cache_config = vllm_config.cache_config
    return _digest(
        (
            vllm_config.model_config.model,
            vllm_config.model_config.dtype,
            cache_config.cache_dtype,
            cache_config.get_resolved_kv_cache_layout(),
            nbytes,
            kv_cache_config.num_blocks,
            [(t.size, tuple(t.layers)) for t in kv_cache_config.kv_cache_tensors],
            [
                (tuple(g.layer_names), repr(g.kv_cache_spec))
                for g in kv_cache_config.kv_cache_groups
            ],
        )
    )


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
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    holder: object | None = None
    standbys: set = field(default_factory=set)


@dataclass
class _Directory:
    key: str
    generations: tuple[str, ...]
    # block_id -> (block hash, num tokens covered)
    entries: dict[int, tuple[bytes, int | None]] = field(default_factory=dict)
    # Blocks the writer has allocated and not freed: they may still be written.
    live: set[int] = field(default_factory=set)
    holder: object | None = None
    # Sends a message to the holder's engine (on its directory session).
    push: Any = None
    # A writer revoked by promotion whose session is still open: it may still
    # be running, so what it touches is quarantined, not trusted.
    fenced: object | None = None
    # Blocks of a revoked writer the holder must keep out of reuse.
    quarantine: set[int] = field(default_factory=set)


class KVRestoreStore:
    """The daemon's KV state: at most one pool (this GPU's KV memory, held as
    fds) and one prefix directory (kept by the daemon of worker 0)."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self.pool: _Pool | None = None
        self.directory: _Directory | None = None

    # -- requests ---------------------------------------------------------

    def probe(self, wait_s: float = 0.0) -> dict | None:
        """The held pool, waiting up to ``wait_s`` for one to be deposited."""
        with self._lock:
            self._lock.wait_for(lambda: self.pool is not None, timeout=wait_s)
            pool = self.pool
            if pool is None:
                return None
            return {
                "num_blocks": pool.num_blocks,
                "nbytes": pool.nbytes,
                "size": pool.size,
                "busy": pool.holder is not None,
            }

    def attach(
        self, holder: object, key: str, role: str = "writer", wait_s: float = 0.0
    ) -> tuple[str, _Pool | None]:
        """For a writer: ``("restore", pool)``, ``("fresh", None)`` or
        ``("busy", None)``. For a standby: ``("standby", pool)`` once a pool
        with this key is held (waiting up to ``wait_s`` for its deposit),
        ``("mismatch", None)`` or ``("timeout", None)``."""
        with self._lock:
            if role == "standby":
                self._lock.wait_for(lambda: self.pool is not None, timeout=wait_s)
                pool = self.pool
                if pool is None:
                    return "timeout", None
                if pool.key != key:
                    return "mismatch", None
                pool.standbys.add(holder)
                return "standby", pool
            pool = self.pool
            if pool is not None and pool.holder is not None:
                return "busy", None
            if pool is not None and pool.key == key:
                pool.holder = holder
                return "restore", pool
            self._drop_pool_locked()
            return "fresh", None

    def promote(self, holder: object) -> _Pool | None:
        """Make a standby the pool's writer now. A previous writer still
        connected is revoked, not waited for: the orchestrator has decided it
        is gone (it may be frozen, e.g. stuck in a collective with a dead
        rank), and the directory quarantines whatever it may still touch."""
        with self._lock:
            pool = self.pool
            if pool is None or holder not in pool.standbys:
                return None
            pool.standbys.discard(holder)
            pool.holder = holder
            return pool

    def deposit(
        self,
        holder: object,
        key: str,
        fds: list[int],
        size: int,
        nbytes: int,
        num_blocks: int,
    ) -> _Pool:
        with self._lock:
            self._drop_pool_locked()
            self.pool = _Pool(key, fds, size, nbytes, num_blocks, holder=holder)
            self._lock.notify_all()
            return self.pool

    def release(self, holder: object) -> None:
        with self._lock:
            if self.pool is not None:
                if self.pool.holder is holder:
                    self.pool.holder = None
                self.pool.standbys.discard(holder)
            d = self.directory
            if d is not None and d.holder is holder:
                d.holder = None
                d.push = None
            if d is not None and d.fenced is holder:
                # The revoked writer is gone: its quarantine can end after
                # the holder's grace period.
                d.fenced = None
                self._push_locked({"op": "fenced_closed"})
            self._lock.notify_all()

    def open_directory(
        self,
        holder: object,
        key: str,
        generations: tuple[str, ...],
        revoke: bool = False,
        push: Any = None,
    ) -> dict | None:
        """Hand the directory to a new holder, or None if another scheduler
        holds it and ``revoke`` is False.

        With ``revoke`` (a promoted standby) a holder still connected is
        revoked and fenced. Returns ``entries`` (committed blocks to adopt),
        ``quarantine`` (blocks the previous writer had allocated but not
        committed, plus any still quarantined: it may still write them) and
        ``fenced_open`` (whether a revoked writer is still connected; its
        quarantine then lasts until it disconnects)."""
        with self._lock:
            d = self.directory
            if d is not None and d.holder is not None and not revoke:
                return None
            if d is None or d.key != key or d.generations != generations:
                if d is not None and d.holder is not None:
                    return None  # another pool's writer: nothing to take over
                d = self.directory = _Directory(key, generations)
            if d.holder is not None:
                d.fenced = d.holder
            d.quarantine |= d.live - d.entries.keys()
            d.live = set()
            d.holder = holder
            d.push = push
            return {
                "entries": [(bid, h, n) for bid, (h, n) in d.entries.items()],
                "quarantine": sorted(d.quarantine),
                "fenced_open": d.fenced is not None,
            }

    def apply(self, holder: object, msg: dict) -> None:
        with self._lock:
            d = self.directory
            if d is None:
                return
            op = msg.get("op")
            if holder is d.fenced:
                self._apply_fenced_locked(d, op, msg)
                return
            if holder is not d.holder:
                return
            if op == "commit":
                for block_id, block_hash, num_tokens in msg["blocks"]:
                    d.entries[block_id] = (block_hash, num_tokens)
            elif op == "evict":
                for block_id in msg["block_ids"]:
                    d.entries.pop(block_id, None)
            elif op == "live":
                freed = set(msg["free"])
                d.live -= freed
                d.quarantine -= freed  # released by the holder
                d.live.update(msg["alloc"])
            elif op == "clear":
                d.entries.clear()

    def _apply_fenced_locked(self, d: _Directory, op: str, msg: dict) -> None:
        """A revoked writer's late messages: an evict means it may rewrite
        the block and an allocation that it may write it, so both quarantine
        the block (and withdraw it from the directory); commits are not
        trusted."""
        if op == "evict":
            blocks = set(msg["block_ids"])
        elif op == "live":
            blocks = set(msg["alloc"])
        else:
            return
        for block_id in blocks:
            d.entries.pop(block_id, None)
        new = blocks - d.quarantine
        if new:
            d.quarantine |= new
            self._push_locked({"op": "quarantine", "block_ids": sorted(new)})

    def _push_locked(self, msg: dict) -> None:
        d = self.directory
        if d is not None and d.push is not None:
            try:
                d.push(msg)
            except OSError:
                d.push = None

    def drop_all(self) -> None:
        with self._lock:
            self._drop_pool_locked()

    def _drop_pool_locked(self) -> None:
        if self.pool is not None:
            for fd in self.pool.fds:
                os.close(fd)
            self.pool = None
        self.directory = None


def serve_kv_session(store: KVRestoreStore, conn: socket.socket, first: dict) -> None:
    """Run one KV request on its own thread: ``kv_probe`` (may wait for a
    pool), or a persistent ``kv_attach`` / ``kv_directory`` session whose open
    connection is the lease."""
    from vllm.device_allocator.cuda_vmm import recv_fd, send_fd

    holder = object()
    try:
        if first["cmd"] == "kv_probe":
            pool = store.probe(float(first.get("wait_s", 0.0)))
            send_msg(conn, {"status": "ok", "pool": pool})
        elif first["cmd"] == "kv_attach" and first.get("role") == "standby":
            _serve_standby(store, conn, first, holder)
        elif first["cmd"] == "kv_attach":
            status, pool = store.attach(holder, first["key"])
            if status == "restore":
                assert pool is not None
                send_msg(
                    conn,
                    {
                        "status": "restore",
                        "generation": pool.generation,
                        "size": pool.size,
                        "num_fds": len(pool.fds),
                    },
                )
                for fd in pool.fds:
                    send_fd(conn, fd)
            elif status == "busy":
                send_msg(conn, {"status": "busy"})
                return
            else:
                send_msg(conn, {"status": "fresh"})
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
                )
                send_msg(conn, {"status": "ok", "generation": pool.generation})
            # Hold the lease until the engine goes away.
            while conn.recv(4096):
                pass
        elif first["cmd"] == "kv_directory":
            send_lock = threading.Lock()

            def push(msg: dict) -> None:
                with send_lock:
                    send_msg(conn, msg)

            # Holding send_lock: no push can overtake the reply.
            with send_lock:
                opened = store.open_directory(
                    holder,
                    first["key"],
                    tuple(first["generations"]),
                    revoke=bool(first.get("revoke")),
                    push=push,
                )
                if opened is None:
                    send_msg(conn, {"status": "busy"})
                    return
                send_msg(conn, {"status": "ok", **opened})
            while True:
                store.apply(holder, recv_msg(conn))
    except (ConnectionError, EOFError, OSError):
        pass
    finally:
        store.release(holder)
        conn.close()


def _serve_standby(
    store: KVRestoreStore, conn: socket.socket, first: dict, holder: object
) -> None:
    from vllm.device_allocator.cuda_vmm import send_fd

    status, pool = store.attach(
        holder, first["key"], "standby", float(first.get("wait_s", 0.0))
    )
    if pool is None:
        send_msg(conn, {"status": status})
        return
    send_msg(
        conn,
        {
            "status": "standby",
            "generation": pool.generation,
            "size": pool.size,
            "num_fds": len(pool.fds),
        },
    )
    for fd in pool.fds:
        send_fd(conn, fd)
    while True:
        msg = recv_msg(conn)
        if msg.get("cmd") != "kv_promote":
            continue
        if store.promote(holder) is None:
            send_msg(conn, {"status": "gone"})
            continue
        send_msg(conn, {"status": "ok"})
        # Now the writer: hold the lease until the engine goes away.
        while conn.recv(4096):
            pass
        return


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------


class KVRestoreClient:
    """A worker's KV session with its GPU's daemon."""

    def __init__(
        self,
        socket_path: str,
        timeout: float = 60.0,
        role: str = "writer",
        standby_wait_s: float = DEFAULT_STANDBY_WAIT_S,
    ):
        self.socket_path = socket_path
        self.timeout = timeout
        self.role = role
        self.standby_wait_s = standby_wait_s
        self.session: socket.socket | None = None
        self.allocation: ShareableAllocation | None = None
        self.restored = False
        self.generation: str | None = None

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        return sock

    def probe(self) -> dict | None:
        """The pool this GPU's daemon holds. A standby waits for the writer
        to deposit one."""
        wait_s = self.standby_wait_s if self.role == "standby" else 0.0
        try:
            with self._connect() as sock:
                sock.settimeout(self.timeout + wait_s)
                send_msg(sock, {"cmd": "kv_probe", "wait_s": wait_s})
                return recv_msg(sock).get("pool")
        except OSError as e:
            logger.warning("KV restore: cannot reach the daemon (%s)", e)
            return None

    def attach(
        self, key: str, nbytes: int, num_blocks: int, device: torch.device
    ) -> torch.Tensor | None:
        """Return the KV backing buffer (restored or fresh), or None to let the
        engine allocate it normally (daemon busy or unreachable)."""
        from vllm.device_allocator.cuda_vmm import (
            ShareableAllocation,
            recv_fd,
            send_fd,
        )

        device_index = device.index if device.index is not None else 0
        if self.role == "standby":
            return self._attach_standby(key, nbytes, num_blocks, device_index)
        try:
            sock = self._connect()
        except OSError as e:
            logger.warning("KV restore: cannot reach the daemon (%s)", e)
            return None
        send_msg(
            sock,
            {
                "cmd": "kv_attach",
                "key": key,
                "nbytes": nbytes,
                "num_blocks": num_blocks,
            },
        )
        reply = recv_msg(sock)
        if reply["status"] == "busy":
            sock.close()
            logger.warning(
                "KV restore: another engine holds this GPU's KV pool; "
                "starting without restore"
            )
            return None
        if reply["status"] == "restore":
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
        else:
            allocation = ShareableAllocation.create(nbytes, device_index)
            allocation.tensor(0, allocation.size).zero_()
            torch.cuda.synchronize(device_index)
            fd = allocation.export_fd()
            try:
                send_msg(
                    sock, {"cmd": "kv_deposit", "size": allocation.size, "num_fds": 1}
                )
                send_fd(sock, fd)
            finally:
                os.close(fd)
            self.generation = recv_msg(sock)["generation"]
        sock.settimeout(None)
        self.session = sock
        self.allocation = allocation
        logger.info(
            "KV restore: %s KV cache (%.2f GiB) in a daemon-held allocation",
            "re-attached the previous" if self.restored else "allocated a new",
            nbytes / (1 << 30),
        )
        return allocation.tensor(0, nbytes).view(torch.int8)

    def _attach_standby(
        self, key: str, nbytes: int, num_blocks: int, device_index: int
    ) -> torch.Tensor:
        """Map the writer's pool (waiting for its deposit) without writing it."""
        from vllm.device_allocator.cuda_vmm import ShareableAllocation, recv_fd

        sock = self._connect()
        sock.settimeout(self.timeout + self.standby_wait_s)
        send_msg(
            sock,
            {
                "cmd": "kv_attach",
                "role": "standby",
                "key": key,
                "nbytes": nbytes,
                "num_blocks": num_blocks,
                "wait_s": self.standby_wait_s,
            },
        )
        reply = recv_msg(sock)
        if reply["status"] != "standby":
            sock.close()
            raise RuntimeError(
                f"KV restore standby: no matching KV pool from the writer "
                f"({reply['status']}); the standby needs the writer's KV layout"
            )
        fds = [recv_fd(sock) for _ in range(reply["num_fds"])]
        try:
            allocation = ShareableAllocation.import_fd(
                fds[0], reply["size"], device_index
            )
        finally:
            for fd in fds:
                os.close(fd)
        sock.settimeout(None)
        self.session = sock
        self.allocation = allocation
        self.restored = True
        self.generation = reply["generation"]
        logger.info(
            "KV restore: standby mapped the writer's KV cache (%.2f GiB)",
            nbytes / (1 << 30),
        )
        return allocation.tensor(0, nbytes).view(torch.int8)

    def promote(self, timeout_s: float) -> None:
        """Become the pool's writer now, revoking a previous writer that is
        still connected (``timeout_s`` bounds the daemon round trip)."""
        if self.role != "standby":
            return
        assert self.session is not None
        self.session.settimeout(timeout_s)
        send_msg(self.session, {"cmd": "kv_promote"})
        reply = recv_msg(self.session)
        self.session.settimeout(None)
        if reply["status"] != "ok":
            raise RuntimeError(
                f"KV restore: the daemon no longer holds this standby's KV pool "
                f"({reply['status']})"
            )
        self.role = "writer"

    def status(self) -> dict:
        return {
            "socket_path": self.socket_path,
            "restored": self.restored,
            "generation": self.generation,
            "role": self.role,
        }


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
        # block_id -> allocated (True) or freed (False) since the last report.
        self._live_changes: dict[int, bool] = {}
        # Blocks a revoked or dead writer may still write: held out of reuse.
        self._quarantine: dict[int, KVCacheBlock] = {}
        # Quarantined blocks still referenced by running requests here.
        self._pending_quarantine: set[int] = set()
        self._quarantine_grace_s = 2.0
        self._quarantine_deadline: float | None = None
        self._next_poll = 0.0

    def start_kv_restore(
        self,
        socket_path: str,
        generations: list[str | None],
        restored: bool,
        timeout_s: float = 60.0,
        quarantine_grace_s: float = 2.0,
        revoke: bool = False,
    ) -> int:
        """Open the directory session; with ``revoke`` (a promoted standby)
        take it over from a writer that may still be connected. If every
        worker re-attached its previous KV pool, import the committed
        directory and quarantine the blocks the previous writer may still
        write. Returns the number of imported blocks."""
        if not self.enable_caching or None in generations:
            return 0
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        sock.connect(socket_path)
        send_msg(
            sock,
            {
                "cmd": "kv_directory",
                "key": directory_key(self),
                "generations": list(generations),
                "revoke": revoke,
            },
        )
        reply = recv_msg(sock)
        if reply["status"] != "ok":
            sock.close()
            logger.warning("KV restore: directory held by another engine")
            return 0
        sock.settimeout(None)
        self._session = sock
        if not restored:
            return 0
        self._quarantine_grace_s = quarantine_grace_s
        self._quarantine_ids(reply["quarantine"])
        if self._quarantine and not reply["fenced_open"]:
            self._quarantine_deadline = time.monotonic() + quarantine_grace_s
        entries = reply["entries"]
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
        logger.info(
            "KV restore: imported %d cached blocks, quarantined %d of the "
            "previous writer",
            len(imported),
            len(self._quarantine) + len(self._pending_quarantine),
        )
        return len(imported)

    # -- quarantine -------------------------------------------------------

    def _quarantine_ids(self, block_ids: list[int]) -> None:
        """Hold blocks a revoked or dead writer may still write out of reuse
        and out of the prefix cache."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.is_null or block_id in self._quarantine:
                continue
            if block.ref_cnt != 0:
                # A running request already uses it (adopted, then evicted by
                # the old writer): unavoidable race; keep it out of reuse.
                self._pending_quarantine.add(block_id)
                logger.warning(
                    "KV restore: block %d is in use but the previous writer "
                    "may rewrite it; quarantining it once freed",
                    block_id,
                )
                continue
            self._hold(block)

    def _hold(self, block: KVCacheBlock) -> None:
        self._maybe_evict_cached_block(block)
        self.free_block_queue.remove(block)
        block.ref_cnt = 1
        self._quarantine[block.block_id] = block
        self._live_changes.pop(block.block_id, None)

    def release_quarantine(self) -> int:
        """Return quarantined blocks to the free pool. Returns their number."""
        blocks = list(self._quarantine.values())
        self._quarantine.clear()
        self._pending_quarantine.clear()
        self._quarantine_deadline = None
        if not blocks:
            return 0
        for block in blocks:
            block.ref_cnt = 0
        self.free_block_queue.append_n(blocks)
        self._send({"op": "live", "alloc": [], "free": [b.block_id for b in blocks]})
        logger.info("KV restore: released %d quarantined blocks", len(blocks))
        return len(blocks)

    def _poll_daemon(self) -> None:
        """Apply what the daemon pushed: blocks a revoked writer touched
        after promotion, and its disconnect (which starts the grace)."""
        now = time.monotonic()
        if self._session is None or now < self._next_poll:
            return
        self._next_poll = now + 0.005
        try:
            while select.select([self._session], [], [], 0)[0]:
                msg = recv_msg(self._session)
                if msg.get("op") == "quarantine":
                    self._quarantine_ids(msg["block_ids"])
                elif msg.get("op") == "fenced_closed":
                    self._quarantine_deadline = now + self._quarantine_grace_s
        except (OSError, EOFError) as e:
            logger.warning("KV restore: lost the directory session (%s)", e)
            self._session = None

    def _maybe_release_quarantine(self) -> None:
        self._poll_daemon()
        if (
            self._quarantine_deadline is not None
            and time.monotonic() >= self._quarantine_deadline
        ):
            self.release_quarantine()

    def get_num_free_blocks(self) -> int:
        self._maybe_release_quarantine()
        return super().get_num_free_blocks()

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        self._maybe_release_quarantine()
        return super().get_new_blocks(num_blocks)

    # -- live blocks ------------------------------------------------------

    def on_blocks_allocated(self, blocks: list[KVCacheBlock]) -> None:
        if self._session is None:
            return
        for block in blocks:
            if not block.is_null:
                self._live_changes[block.block_id] = True

    def on_blocks_released(self, blocks: list[KVCacheBlock]) -> None:
        if self._session is None:
            return
        for block in blocks:
            if block.is_null:
                continue
            if block.block_id in self._pending_quarantine:
                self._pending_quarantine.discard(block.block_id)
                self._hold(block)
            else:
                self._live_changes[block.block_id] = False

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
        if self._live_changes:
            # Before the step runs: a dead writer's in-flight blocks must be
            # known to whoever takes the pool over.
            changes, self._live_changes = self._live_changes, {}
            self._send(
                {
                    "op": "live",
                    "alloc": [b for b, live in changes.items() if live],
                    "free": [b for b, live in changes.items() if not live],
                }
            )
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
