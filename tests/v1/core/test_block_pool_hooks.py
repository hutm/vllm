# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool lifecycle hooks and the ``--block-pool-cls`` selection."""

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.config.cache import CacheConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash

BLOCK_SIZE = 16


class RecordingBlockPool(BlockPool):
    def __init__(self, *args, **kwargs):
        self.events: list[tuple[str, tuple[int, ...]]] = []
        super().__init__(*args, **kwargs)

    def on_blocks_allocated(self, blocks):
        self.events.append(("allocated", tuple(b.block_id for b in blocks)))

    def on_blocks_released(self, blocks):
        self.events.append(("released", tuple(b.block_id for b in blocks)))

    def on_cached_block_evicted(self, block, block_hashes):
        assert block_hashes
        self.events.append(("evicted", (block.block_id,)))

    def on_blocks_cached(self, blocks, kv_cache_group_id):
        assert kv_cache_group_id == 0
        self.events.append(("cached", tuple(b.block_id for b in blocks)))


class VetoBlockPool(BlockPool):
    def on_blocks_allocated(self, blocks):
        raise RuntimeError("external capacity exhausted")


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


def _manager(num_blocks: int, block_pool_cls=RecordingBlockPool):
    return make_kv_cache_manager(
        make_kv_cache_config(BLOCK_SIZE, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        block_pool_cls=block_pool_cls,
    )


def _run(manager, request_id: str, token_ids: list[int]):
    req = make_request(request_id, token_ids, BLOCK_SIZE, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req, len(token_ids) - num_computed, num_computed, computed
    )
    assert blocks is not None
    return req


def test_selected_block_pool_class_is_used():
    manager = _manager(6)
    assert type(manager.block_pool) is RecordingBlockPool


def test_hooks_follow_the_block_lifecycle():
    manager = _manager(6)
    pool = manager.block_pool
    # Two full blocks plus a partial one.
    req0 = _run(manager, "0", [1] * BLOCK_SIZE + [2] * BLOCK_SIZE + [3] * 4)
    assert pool.events == [("allocated", (1, 2, 3)), ("cached", (1, 2))]

    pool.events.clear()
    manager.free(req0)
    # Freed in eviction order: the uncached tail first, then the cached prefix.
    assert pool.events == [("released", (3, 2, 1))]

    pool.events.clear()
    # A different prompt needs every free block, so it reuses and evicts the
    # cached ones.
    _run(manager, "1", [9] * (5 * BLOCK_SIZE))
    kinds = [kind for kind, _ in pool.events]
    assert kinds[0] == "allocated"
    assert sorted(b for kind, ids in pool.events if kind == "evicted" for b in ids) == [
        1,
        2,
    ]
    assert kinds.index("evicted") > kinds.index("allocated")


def test_allocation_veto_returns_blocks_untouched():
    manager = _manager(6, block_pool_cls=VetoBlockPool)
    pool = manager.block_pool
    free_before = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]

    with pytest.raises(RuntimeError, match="external capacity exhausted"):
        pool.get_new_blocks(2)

    free_after = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]
    assert free_after == free_before
    assert all(b.ref_cnt == 0 for b in pool.blocks[1:])


def test_cache_config_resolves_block_pool_class():
    assert CacheConfig().get_block_pool_cls() is BlockPool
    config = CacheConfig(block_pool_cls=f"{__name__}.{RecordingBlockPool.__qualname__}")
    assert config.get_block_pool_cls() is RecordingBlockPool


def test_cache_config_rejects_non_block_pool_class():
    config = CacheConfig(block_pool_cls="collections.OrderedDict")
    with pytest.raises(ValueError, match="not a BlockPool subclass"):
        config.get_block_pool_cls()


def test_completed_pause_notifies_the_block_pool():
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    calls = []
    core = object.__new__(EngineCore)
    core.model_executor = SimpleNamespace(collective_rpc=lambda method: None)
    core.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(
                on_scheduler_paused=lambda: calls.append("paused")
            )
        )
    )
    core._finish_pause(clear_cache=False)
    assert calls == ["paused"]
