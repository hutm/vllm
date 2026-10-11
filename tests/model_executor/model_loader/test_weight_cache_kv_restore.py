# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV restore through the weight cache daemon: daemon state, the session
protocol and the scheduler's directory publishing (CPU only)."""

import os
import socket
import threading

import pytest

from vllm.device_allocator.cuda_vmm import recv_fd, send_fd
from vllm.model_executor.model_loader.weight_cache.kv_restore import (
    KVRestoreBlockPool,
    KVRestoreStore,
    serve_kv_session,
)
from vllm.model_executor.model_loader.weight_cache.protocol import (
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


def _attach(store, key="k"):
    return _session(
        store, {"cmd": "kv_attach", "key": key, "nbytes": 4, "num_blocks": 2}
    )


def _wait_released(store):
    for _ in range(200):
        if store.pool is None or store.pool.holder is None:
            return
        threading.Event().wait(0.01)
    raise AssertionError("lease not released")


def test_fresh_deposit_then_restore_hands_back_the_same_memory():
    store = KVRestoreStore()
    read_end, write_end = os.pipe()

    a = _attach(store)
    assert recv_msg(a)["status"] == "fresh"
    send_msg(a, {"cmd": "kv_deposit", "size": 4096, "num_fds": 1})
    send_fd(a, read_end)
    os.close(read_end)
    generation = recv_msg(a)["generation"]

    # A second engine is refused while the first holds the lease.
    busy = _attach(store)
    assert recv_msg(busy)["status"] == "busy"

    a.close()  # engine A dies: the daemon keeps the memory
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


def test_layout_change_drops_the_held_pool():
    store = KVRestoreStore()
    read_end, write_end = os.pipe()
    a = _attach(store, key="old")
    recv_msg(a)
    send_msg(a, {"cmd": "kv_deposit", "size": 4096, "num_fds": 1})
    send_fd(a, read_end)
    os.close(read_end)
    recv_msg(a)
    a.close()
    _wait_released(store)

    b = _attach(store, key="new")
    assert recv_msg(b)["status"] == "fresh"
    # The daemon closed its copy of the old pool's fd.
    os.close(write_end)
    assert store.pool is None
    b.close()


def test_directory_survives_only_with_matching_generations():
    store = KVRestoreStore()
    holder = object()
    assert store.open_directory(holder, "dk", ("g0",))["entries"] == []
    store.apply(holder, {"op": "commit", "blocks": [(1, b"h1", 16), (2, b"h2", 32)]})
    store.apply(holder, {"op": "evict", "block_ids": [2]})
    store.release(holder)

    assert store.open_directory(object(), "dk", ("g0",)) == {
        "entries": [(1, b"h1", 16)],
        "quarantine": [],
        "fenced_open": False,
    }
    # A second scheduler is refused while the first holds the directory.
    assert store.open_directory(object(), "dk", ("g0",)) is None
    store.release(store.directory.holder)
    # A worker re-allocated its KV (new generation): nothing is reusable.
    assert store.open_directory(object(), "dk", ("g1",))["entries"] == []


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


def _directory_server(store: KVRestoreStore, tmp_path) -> str:
    """A daemon socket serving each KV session on its own thread."""
    path = str(tmp_path / "d.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen()

    def serve():
        while True:
            conn, _ = server.accept()
            threading.Thread(
                target=serve_kv_session,
                args=(store, conn, recv_msg(conn)),
                daemon=True,
            ).start()

    threading.Thread(target=serve, daemon=True).start()
    return path


def test_restart_imports_the_directory(tmp_path):
    store = KVRestoreStore()
    path = _directory_server(store, tmp_path)

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


# -- standby and promotion ---------------------------------------------------


def _standby(store, key="k", wait_s=5.0):
    return _session(
        store,
        {
            "cmd": "kv_attach",
            "role": "standby",
            "key": key,
            "nbytes": 4,
            "num_blocks": 2,
            "wait_s": wait_s,
        },
    )


def _deposit(store, key="k"):
    read_end, write_end = os.pipe()
    a = _attach(store, key)
    assert recv_msg(a)["status"] == "fresh"
    send_msg(a, {"cmd": "kv_deposit", "size": 4096, "num_fds": 1})
    send_fd(a, read_end)
    os.close(read_end)
    generation = recv_msg(a)["generation"]
    return a, write_end, generation


def test_standby_waits_for_the_writers_pool_and_maps_it_without_the_lease():
    store = KVRestoreStore()
    # The standby usually boots next to the writer and attaches first.
    standby = _standby(store)
    other = _standby(store)
    a, write_end, generation = _deposit(store)
    for s in (standby, other):
        reply = recv_msg(s)
        assert reply == {
            "status": "standby",
            "generation": generation,
            "size": 4096,
            "num_fds": 1,
        }
        fd = recv_fd(s)
        os.write(write_end, b"x")
        assert os.read(fd, 1) == b"x"  # the writer's memory
        os.close(fd)
    # Standbys never take the lease: a second writer is still refused.
    assert recv_msg(_attach(store))["status"] == "busy"
    assert store.pool.holder is not None and len(store.pool.standbys) == 2


def test_standby_with_another_layout_or_no_writer_is_refused():
    store = KVRestoreStore()
    assert recv_msg(_standby(store, wait_s=0.05))["status"] == "timeout"
    _deposit(store, key="k")
    assert recv_msg(_standby(store, key="other"))["status"] == "mismatch"


def test_promote_revokes_a_writer_that_is_still_connected():
    """The old writer may be frozen (e.g. stuck in a collective with a dead
    rank): promotion takes the lease at once, without waiting for it."""
    store = KVRestoreStore()
    a, _, _ = _deposit(store)
    s = _standby(store)
    assert recv_msg(s)["status"] == "standby"
    recv_fd(s)

    send_msg(s, {"cmd": "kv_promote"})
    assert recv_msg(s) == {"status": "ok"}  # writer A is still connected
    holder = store.pool.holder
    assert holder is not None and not store.pool.standbys
    a.close()  # the revoked writer's later exit does not drop the new lease
    threading.Event().wait(0.1)
    assert store.pool.holder is holder


def _wait(cond):
    for _ in range(300):
        if cond():
            return
        threading.Event().wait(0.01)
    raise AssertionError("condition not met")


def test_promotion_fences_the_old_writer_and_quarantines_what_it_touches(tmp_path):
    store = KVRestoreStore()
    path = _directory_server(store, tmp_path)

    # Writer A: blocks 1 and 5 committed; 2 and 3 allocated, not committed.
    a = KVRestoreBlockPool(8, enable_caching=True, hash_block_size=4)
    assert a.start_kv_restore(path, ["g0"], restored=False) == 0
    _cache(a, 1, b"h1")
    _cache(a, 5, b"h5")
    out = _Out()
    a.on_step_scheduled(out)
    a.on_step_output_processed(out)
    a._send({"op": "live", "alloc": [2, 3], "free": []})
    _wait(lambda: store.directory.live == {2, 3})

    # Standby B is promoted while A is still connected (frozen).
    b = KVRestoreBlockPool(8, enable_caching=True, hash_block_size=4)
    assert b.start_kv_restore(path, ["g0"], restored=True, revoke=True) == 2
    assert set(b._quarantine) == {2, 3}
    assert b._quarantine_deadline is None  # A's session is still open
    assert b.cached_block_hash_to_block.get_one_block(b"h5").block_id == 5

    # A's late messages: its commit is ignored; its evict of adopted block 5
    # and its allocation of block 4 quarantine them in B.
    a._send({"op": "commit", "blocks": [(6, b"h6", 28)]})
    a._send({"op": "evict", "block_ids": [5]})
    a._send({"op": "live", "alloc": [4], "free": []})
    _wait(lambda: store.directory.quarantine >= {2, 3, 4, 5})
    _wait(
        lambda: (b._maybe_release_quarantine(), set(b._quarantine))[1] == {2, 3, 4, 5}
    )
    assert 6 not in store.directory.entries
    assert 5 not in store.directory.entries
    assert b.cached_block_hash_to_block.get_one_block(b"h5") is None
    free = b.get_num_free_blocks()

    # A exits: the quarantine ends after the grace period.
    a._session.close()
    b._quarantine_grace_s = 0.05
    _wait(lambda: b._quarantine_deadline is not None or b._poll_daemon())
    threading.Event().wait(0.1)
    assert b.get_num_free_blocks() == free + 4
    assert not b._quarantine
    _wait(lambda: not store.directory.quarantine)
    b._session.close()


def test_a_block_in_use_is_quarantined_once_freed(tmp_path):
    pool, daemon_end = _pool_with_session()
    blocks = pool.get_new_blocks(1)
    pool._quarantine_ids([blocks[0].block_id])
    assert blocks[0].block_id in pool._pending_quarantine
    pool.free_blocks(blocks)
    assert blocks[0].block_id in pool._quarantine
    assert pool.get_num_free_blocks() == 6  # 8 blocks - null - quarantined


def test_promotion_quarantines_the_old_writers_uncommitted_blocks():
    store = KVRestoreStore()
    old = object()
    store.open_directory(old, "dk", ("g0",))
    store.apply(old, {"op": "commit", "blocks": [(1, b"h1", 16)]})
    # Blocks 1-4 allocated; 4 freed again; 2 and 3 never committed.
    store.apply(old, {"op": "live", "alloc": [1, 2, 3, 4], "free": []})
    store.apply(old, {"op": "live", "alloc": [], "free": [4]})
    store.release(old)  # the old writer's scheduler died

    opened = store.open_directory(object(), "dk", ("g0",))
    assert opened == {
        "entries": [(1, b"h1", 16)],
        "quarantine": [2, 3],
        "fenced_open": False,
    }


def test_pool_quarantines_blocks_until_the_grace_or_an_explicit_release():
    pool, daemon_end = _pool_with_session()
    free = pool.get_num_free_blocks()
    pool._quarantine_ids([2, 3])
    assert pool.get_num_free_blocks() == free - 2
    got = pool.get_new_blocks(free - 2)
    assert {b.block_id for b in got}.isdisjoint({2, 3})
    pool.free_blocks(got)

    # The orchestrator proved the old writer's GPU work stopped.
    assert pool.release_quarantine() == 2
    assert recv_msg(daemon_end) == {"op": "live", "alloc": [], "free": [2, 3]}
    assert pool.get_num_free_blocks() == free

    # Without an explicit release, the grace period ends the quarantine.
    pool._quarantine_ids([4])
    pool._quarantine_deadline = 0.0
    assert pool.get_num_free_blocks() == free
    assert recv_msg(daemon_end) == {"op": "live", "alloc": [], "free": [4]}


def test_live_blocks_are_reported_before_the_step_runs():
    pool, daemon_end = _pool_with_session()
    blocks = pool.get_new_blocks(2)
    pool.free_blocks(blocks[1:])
    pool.on_step_scheduled(_Out())
    assert recv_msg(daemon_end) == {
        "op": "live",
        "alloc": [blocks[0].block_id],
        "free": [blocks[1].block_id],
    }
