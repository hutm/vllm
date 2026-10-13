# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool hooks, ``--block-pool-cls`` and ``import_cached_blocks``."""

import pytest

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.cache import CacheConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.outputs import ModelRunnerOutput

BLOCK_SIZE = 16


class RecordingBlockPool(BlockPool):
    def __init__(self, *args, **kwargs):
        self.events: list[tuple[str, tuple[int, ...]]] = []
        super().__init__(*args, **kwargs)

    def on_cached_block_evicted(self, block, block_hashes):
        assert block_hashes
        self.events.append(("evicted", (block.block_id,)))

    def on_blocks_cached(self, blocks, kv_cache_group_id):
        assert kv_cache_group_id == 0
        self.events.append(("cached", tuple(b.block_id for b in blocks)))


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


def _manager(num_blocks: int = 8, **kwargs):
    kwargs.setdefault("block_pool_cls", RecordingBlockPool)
    return make_kv_cache_manager(
        make_kv_cache_config(BLOCK_SIZE, num_blocks),
        max_model_len=8192,
        hash_block_size=BLOCK_SIZE,
        **{"enable_caching": True, **kwargs},
    )


def _run(manager, request_id: str, token_ids: list[int]):
    req = make_request(request_id, token_ids, BLOCK_SIZE, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req, len(token_ids) - num_computed, num_computed, computed
    )
    assert blocks is not None
    return req


def _prompt():
    return [1] * BLOCK_SIZE + [2] * BLOCK_SIZE + [3] * 5


def _hashes(token_ids):
    req = make_request("probe", token_ids, BLOCK_SIZE, sha256)
    return [make_block_hash_with_group_id(h, 0) for h in req.block_hashes]


def test_cache_config_selects_the_block_pool_class():
    assert CacheConfig().get_block_pool_cls() is BlockPool
    config = CacheConfig(block_pool_cls=f"{__name__}.{RecordingBlockPool.__qualname__}")
    assert config.get_block_pool_cls() is RecordingBlockPool
    assert type(_manager().block_pool) is RecordingBlockPool
    with pytest.raises(ValueError, match="not a BlockPool subclass"):
        CacheConfig(block_pool_cls="collections.OrderedDict").get_block_pool_cls()


def test_cached_and_evicted_hooks_follow_the_prefix_cache():
    manager = _manager(6)
    pool = manager.block_pool
    # Two full blocks plus a partial one: only the full ones are cached.
    req0 = _run(manager, "0", _prompt())
    assert pool.events == [("cached", (1, 2))]
    manager.free(req0)

    pool.events.clear()
    # A different prompt needs every free block and reuses the cached ones.
    _run(manager, "1", [9] * (5 * BLOCK_SIZE))
    evicted = sorted(ids[0] for kind, ids in pool.events if kind == "evicted")
    assert evicted == [1, 2]


def test_scheduler_brackets_each_step_for_the_block_pool():
    """Blocks cached while a step is scheduled are only written once that
    step's output is processed; the pool sees both ends of each step, in
    order, with the same SchedulerOutput."""
    scheduler = create_scheduler(enable_prefix_caching=True)
    pool = scheduler.kv_cache_manager.block_pool
    events = []
    pool.on_step_scheduled = lambda out: events.append(("scheduled", out))
    pool.on_step_output_processed = lambda out: events.append(("processed", out))

    (request,) = create_requests(num_requests=1, num_tokens=40)
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert events == [("scheduled", output)]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    assert events == [("scheduled", output), ("processed", output)]


def test_imported_blocks_serve_prefix_hits():
    manager = _manager()
    pool = manager.block_pool
    hashes = _hashes(_prompt())
    # KV for the two full blocks already sits in blocks 5 and 3.
    imported = pool.import_cached_blocks(hashes, [5, 3], [BLOCK_SIZE, 2 * BLOCK_SIZE])
    assert imported == [5, 3]
    assert all(pool.blocks[i].ref_cnt == 0 for i in (5, 3))
    # Most-recently-used end of the free queue: evicted after colder blocks.
    tail = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()][-2:]
    assert tail == [5, 3]

    req = make_request("0", _prompt(), BLOCK_SIZE, sha256)
    computed, num_computed, _ = manager.get_computed_blocks(req)
    assert computed.get_block_ids() == ([5, 3],)
    assert num_computed == 2 * BLOCK_SIZE


def test_import_skips_null_in_use_and_duplicate_blocks():
    manager = _manager()
    pool = manager.block_pool
    hashes = _hashes(_prompt())
    in_use = pool.get_new_blocks(1)[0]

    imported = pool.import_cached_blocks(
        [hashes[0], hashes[0], hashes[1], hashes[1]],
        [0, 4, in_use.block_id, 6],
    )

    # The null block and the in-use block are skipped; the second import of
    # hashes[0] would duplicate block 4's entry.
    assert imported == [4, 6]
    assert pool.get_cached_block(get_block_hash(hashes[0]), [0])[0].block_id == 4
    assert in_use.block_hash is None


def test_import_evicts_stale_entries_of_the_target_block():
    pool = _manager().block_pool
    old, new = _hashes([7] * BLOCK_SIZE)[0], _hashes(_prompt())[0]
    assert pool.import_cached_blocks([old], [2]) == [2]
    pool.events.clear()

    assert pool.import_cached_blocks([new], [2]) == [2]

    assert pool.events == [("evicted", (2,))]
    assert pool.get_cached_block(get_block_hash(old), [0]) is None
    assert pool.get_cached_block(get_block_hash(new), [0])[0].block_id == 2


def test_import_validates_arguments_and_needs_prefix_caching():
    pool = _manager().block_pool
    hashes = _hashes(_prompt())
    with pytest.raises(ValueError, match="same length"):
        pool.import_cached_blocks(hashes, [1])
    with pytest.raises(ValueError, match="num_tokens"):
        pool.import_cached_blocks(hashes, [1, 2], [BLOCK_SIZE])
    with pytest.raises(ValueError, match="Invalid block_id"):
        pool.import_cached_blocks(hashes[:1], [99])

    no_cache = _manager(enable_caching=False, block_pool_cls=None).block_pool
    assert no_cache.import_cached_blocks(hashes, [1, 2]) == []
