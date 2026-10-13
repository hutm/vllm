# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV restore through the weight cache daemon: daemon state, the session
protocol and the scheduler's directory publishing (CPU only; file
descriptors of pipes stand in for the KV memory)."""

import os
import socket
import threading

import pytest

from vllm.device_allocator.cuda_vmm import recv_fd, send_fd
from vllm.model_executor.model_loader.weight_cache.kv_restore import (
    KVRestoreBlockPool,
    KVRestoreStore,
    drop_kv_pools,
    handle_kv_request,
    serve_kv_session,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
    SOCKET_NAME_TEMPLATE,
    recv_msg,
    send_msg,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash


@pytest.fixture(autouse=True)
def _none_hash():
    init_none_hash(sha256)


def _session(store: KVRestoreStore, first: dict) -> socket.socket:
    """Start a daemon-side session thread; return the engine's end."""
    engine_end, daemon_end = socket.socketpair()
    send_msg(engine_end, first)
    daemon_first = recv_msg(daemon_end)
    threading.Thread(
        target=serve_kv_session, args=(store, daemon_end, daemon_first), daemon=True
    ).start()
    return engine_end


def _attach(store, key="k", layout=None):
    return _session(
        store,
        {
            "cmd": "kv_attach",
            "key": key,
            "nbytes": 4,
            "num_blocks": 2,
            "layout": layout or {"model": "m", "block_size": 16},
        },
    )


def _deposit(conn) -> tuple[int, str]:
    """Deposit a pipe as the engine's KV memory; return its write end and the
    pool generation."""
    read_end, write_end = os.pipe()
    send_msg(conn, {"cmd": "kv_deposit", "size": 4096, "num_fds": 1})
    send_fd(conn, read_end)
    os.close(read_end)
    return write_end, recv_msg(conn)["generation"]


def _serve(store, path: str) -> None:
    """Serve ``store`` on a Unix socket like the daemon does."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()

    def serve():
        while True:
            conn, _ = server.accept()
            first = recv_msg(conn)
            if first["cmd"] in ("kv_probe", "kv_drop"):
                with conn:
                    handle_kv_request(store, conn, first)
            else:
                threading.Thread(
                    target=serve_kv_session, args=(store, conn, first), daemon=True
                ).start()

    threading.Thread(target=serve, daemon=True).start()


def _request(store, msg: dict) -> dict:
    engine_end, daemon_end = socket.socketpair()
    with engine_end, daemon_end:
        handle_kv_request(store, daemon_end, msg)
        return recv_msg(engine_end)


def _wait_released(store):
    for _ in range(200):
        if store.pool is None or store.pool.holder is None:
            return
        threading.Event().wait(0.01)
    raise AssertionError("lease not released")


def test_fresh_deposit_then_restore_hands_back_the_same_memory():
    store = KVRestoreStore()

    a = _attach(store)
    assert recv_msg(a) == {"status": "fresh", "layout_mismatch": None}
    write_end, generation = _deposit(a)

    # A second engine is refused while the first holds the lease.
    busy = _attach(store)
    assert recv_msg(busy)["status"] == "busy"

    a.close()  # engine A exits (or dies): the lease ends on EOF
    _wait_released(store)

    b = _attach(store)
    reply = recv_msg(b)
    assert reply == {
        "status": "restore",
        "generation": generation,
        "size": 4096,
        "num_fds": 1,
    }
    fd = recv_fd(b)
    os.write(write_end, b"kv")
    assert os.read(fd, 2) == b"kv"  # the very object engine A deposited
    os.close(fd)
    b.close()
    os.close(write_end)


def test_layout_change_drops_the_held_pool_and_names_the_fields():
    store = KVRestoreStore()
    a = _attach(store, key="old", layout={"model": "m", "block_size": 16})
    recv_msg(a)
    write_end, _ = _deposit(a)
    a.close()
    _wait_released(store)

    b = _attach(store, key="new", layout={"model": "m", "block_size": 32})
    assert recv_msg(b) == {"status": "fresh", "layout_mismatch": ["block_size"]}
    # The daemon closed its copy of the old pool's fd.
    assert store.pool is None
    os.close(write_end)
    b.close()


def test_drop_frees_a_retained_pool_but_not_one_in_use_unless_forced(tmp_path):
    store = KVRestoreStore()
    a = _attach(store)
    recv_msg(a)
    write_end, _ = _deposit(a)
    probe = _request(store, {"cmd": "kv_probe"})["pool"]
    assert probe["busy"] and probe["num_blocks"] == 2

    # `vllm preload --kv-drop` against this daemon's socket.
    path = str(tmp_path / SOCKET_NAME_TEMPLATE.format(gpu_uuid="0", role=""))
    _serve(store, path)
    ok, lines = drop_kv_pools(str(tmp_path))
    assert not ok and "busy" in lines[0]
    assert store.pool is not None

    a.close()  # the engine exits cleanly: the pool is kept...
    _wait_released(store)
    assert _request(store, {"cmd": "kv_probe"})["pool"]["busy"] is False
    ok, lines = drop_kv_pools(str(tmp_path))  # ...until it is dropped
    assert ok and lines == [f"{path}: freed 0.00 GiB"]
    assert store.pool is None
    os.close(write_end)

    b = _attach(store)
    recv_msg(b)
    write_end, _ = _deposit(b)
    assert _request(store, {"cmd": "kv_drop", "force": True}) == {
        "status": "ok",
        "freed_bytes": 4096,
    }
    assert store.pool is None
    os.close(write_end)
    b.close()


def test_directory_survives_only_with_matching_generations():
    store = KVRestoreStore()
    holder = object()
    assert store.open_directory(holder, "dk", ("g0",)) == []
    store.apply(holder, {"op": "commit", "blocks": [(1, b"h1", 16), (2, b"h2", 32)]})
    store.apply(holder, {"op": "evict", "block_ids": [2]})
    store.release(holder)

    assert store.open_directory(object(), "dk", ("g0",)) == [(1, b"h1", 16)]
    store.release(store.directory.holder)
    # A worker re-allocated its KV (new generation): nothing is reusable.
    assert store.open_directory(object(), "dk", ("g1",)) == []


class _Out:
    """Stands in for a SchedulerOutput (only its identity is used)."""


def _pool_with_session():
    pool = KVRestoreBlockPool(8, enable_caching=True, hash_block_size=4)
    engine_end, daemon_end = socket.socketpair()
    pool._session = engine_end
    daemon_end.settimeout(5)
    return pool, daemon_end


def _cache(pool, block_id, block_hash):
    block = pool.blocks[block_id]
    pool._insert_block_hash(block_hash, block, num_tokens=4 * block_id)
    pool.on_blocks_cached([block], 0)


def test_blocks_commit_after_their_step_output_and_evictions_go_first():
    pool, daemon = _pool_with_session()
    step1, step2 = _Out(), _Out()

    _cache(pool, 1, b"h1")
    _cache(pool, 2, b"h2")
    pool.on_step_scheduled(step1)
    # Async scheduling: step 2 is scheduled before step 1's output arrives.
    _cache(pool, 3, b"h3")
    pool.on_step_scheduled(step2)

    # Block 2 is evicted for reuse: retracted at once, never committed.
    pool._maybe_evict_cached_block(pool.blocks[2])
    assert recv_msg(daemon) == {"op": "evict", "block_ids": [2]}

    pool.on_step_output_processed(step1)
    assert recv_msg(daemon) == {"op": "commit", "blocks": [(1, b"h1", 4)]}
    pool.on_step_output_processed(step2)
    assert recv_msg(daemon) == {"op": "commit", "blocks": [(3, b"h3", 12)]}


def test_reset_prefix_cache_clears_the_directory():
    pool, daemon = _pool_with_session()
    _cache(pool, 1, b"h1")
    pool.on_step_scheduled(_Out())
    assert pool.reset_prefix_cache()
    assert recv_msg(daemon) == {"op": "clear"}


def test_restart_imports_the_directory(tmp_path):
    store = KVRestoreStore()
    path = str(tmp_path / "d.sock")
    _serve(store, path)

    # Engine A publishes two committed blocks, then dies.
    a = KVRestoreBlockPool(8, enable_caching=True, hash_block_size=4)
    assert a.start_kv_restore(path, ["g0"], restored=False) == 0
    _cache(a, 1, b"h1")
    _cache(a, 5, b"h5")
    out = _Out()
    a.on_step_scheduled(out)
    a.on_step_output_processed(out)
    a._session.close()
    for _ in range(200):
        if store.directory.holder is None:
            break
        threading.Event().wait(0.01)

    # Engine B re-attached the same pools: it gets the prefix cache back.
    b = KVRestoreBlockPool(8, enable_caching=True, hash_block_size=4)
    assert b.start_kv_restore(path, ["g0"], restored=True) == 2
    assert b.cached_block_hash_to_block.get_one_block(b"h1").block_id == 1
    assert b.cached_block_hash_to_block.get_one_block(b"h5").block_id == 5
    assert b.blocks[5].block_hash_num_tokens == 20
    b._session.close()
