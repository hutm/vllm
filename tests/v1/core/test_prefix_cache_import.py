# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``BlockPool.import_cached_blocks``: adopting KV that is already valid."""

import pytest

from tests.v1.core.test_block_pool_hooks import RecordingBlockPool
from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    init_none_hash,
    make_block_hash_with_group_id,
)

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _hash_fn():
    init_none_hash(sha256)


def _manager(num_blocks: int = 8, **kwargs):
    kwargs.setdefault("block_pool_cls", RecordingBlockPool)
    return make_kv_cache_manager(
        make_kv_cache_config(BLOCK_SIZE, num_blocks),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
        **kwargs,
    )


def _prompt():
    return [1] * BLOCK_SIZE + [2] * BLOCK_SIZE + [3] * 5


def _hashes(token_ids):
    req = make_request("probe", token_ids, BLOCK_SIZE, sha256)
    return [make_block_hash_with_group_id(h, 0) for h in req.block_hashes]


def test_imported_blocks_serve_prefix_hits():
    manager = _manager()
    pool = manager.block_pool
    hashes = _hashes(_prompt())
    # KV for the two full blocks already sits in blocks 5 and 3.
    assert pool.import_cached_blocks(hashes, [5, 3], [BLOCK_SIZE, 2 * BLOCK_SIZE]) == [
        5,
        3,
    ]
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

    # null block and in-use block skipped; the second import of hashes[0]
    # would duplicate block 4's entry.
    assert imported == [4, 6]
    assert pool.get_cached_block(get_block_hash(hashes[0]), [0])[0].block_id == 4
    assert in_use.block_hash is None


def test_import_evicts_stale_entries_of_the_target_block():
    manager = _manager()
    pool = manager.block_pool
    old, new = _hashes([7] * BLOCK_SIZE)[0], _hashes(_prompt())[0]
    assert pool.import_cached_blocks([old], [2]) == [2]
    pool.events.clear()

    assert pool.import_cached_blocks([new], [2]) == [2]

    assert pool.events == [("evicted", (2,))]
    assert pool.get_cached_block(get_block_hash(old), [0]) is None
    assert pool.get_cached_block(get_block_hash(new), [0])[0].block_id == 2


def test_import_validates_arguments_and_needs_prefix_caching():
    manager = _manager()
    pool = manager.block_pool
    hashes = _hashes(_prompt())
    with pytest.raises(ValueError, match="same length"):
        pool.import_cached_blocks(hashes, [1])
    with pytest.raises(ValueError, match="num_tokens"):
        pool.import_cached_blocks(hashes, [1, 2], [BLOCK_SIZE])
    with pytest.raises(ValueError, match="Invalid block_id"):
        pool.import_cached_blocks(hashes[:1], [99])

    no_cache = make_kv_cache_manager(
        make_kv_cache_config(BLOCK_SIZE, 8),
        max_model_len=8192,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
    )
    assert no_cache.block_pool.import_cached_blocks(hashes, [1, 2]) == []
