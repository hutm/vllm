# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_persistence import (
    KVCacheBlockCandidate,
    KVCacheBlockClaim,
    KVCacheBlockProvider,
    KVCacheBlockRecord,
    KVCacheBlockUnavailable,
)
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


@dataclass
class _SharedStore:
    generations: dict[int, int] = field(default_factory=dict)
    states: dict[int, str] = field(default_factory=dict)
    records: dict[bytes, KVCacheBlockRecord] = field(default_factory=dict)


class _GeometryConnector(KVConnectorBase_V1):
    @classmethod
    def get_kv_cache_num_blocks_override(cls, vllm_config):
        return 123


class _FakeProvider(KVCacheBlockProvider):
    def __init__(self, store: _SharedStore, owner: str):
        self.store = store
        self.owner = owner
        self.rollbacks: list[int] = []
        self.discards: list[tuple[int, bool]] = []
        self.activations = 0
        self.flushes = 0
        self.acquire_calls = 0

    def bind(self, num_blocks: int, reserved_block_ids):
        reserved = set(reserved_block_ids)
        for block_id in range(num_blocks):
            self.store.generations.setdefault(block_id, 0)
            self.store.states.setdefault(
                block_id, "reserved" if block_id in reserved else "free"
            )

    def get_num_available(self, native_free_blocks: int) -> int:
        available = sum(
            state in {"free", "ready"} for state in self.store.states.values()
        )
        return min(native_free_blocks, available)

    def _claim(self, block_id: int) -> KVCacheBlockClaim:
        generation = self.store.generations[block_id] + 1
        self.store.generations[block_id] = generation
        self.store.states[block_id] = "active"
        return KVCacheBlockClaim(block_id, (self.owner, generation))

    def acquire(self, candidates, count):
        self.acquire_calls += 1
        selected: list[KVCacheBlockCandidate] = []
        for candidate in candidates:
            if self.store.states[candidate.block_id] in {"free", "ready"}:
                selected.append(candidate)
                if len(selected) == count:
                    break
        if len(selected) != count:
            return None
        selected_ids = {candidate.block_id for candidate in selected}
        self.store.records = {
            key: record
            for key, record in self.store.records.items()
            if record.block_id not in selected_ids
        }
        return [self._claim(candidate.block_id) for candidate in selected]

    def activate(self, records):
        self.activations += 1
        claims = []
        for record in records:
            state = self.store.states[record.block_id]
            if state not in {"active", "ready"}:
                return None
            claims.append(self._claim(record.block_id))
        return claims

    def commit(self, records):
        for record in records:
            self.store.states[record.block_id] = "ready"
            for block_hash in record.block_hashes:
                self.store.records[bytes(block_hash)] = record

    def release(self, claims):
        for claim in claims:
            self.store.states[claim.block_id] = "free"
            self.store.records = {
                key: record
                for key, record in self.store.records.items()
                if record.block_id != claim.block_id
            }

    def recover(self, block_hashes):
        records = [self.store.records.get(bytes(key)) for key in block_hashes]
        if any(record is None for record in records):
            return None
        recovered = []
        for key, old in zip(block_hashes, records):
            assert old is not None
            if self.store.states[old.block_id] != "ready":
                return None
            claim = self._claim(old.block_id)
            recovered.append(
                KVCacheBlockRecord(
                    block_id=old.block_id,
                    block_hash=key,
                    num_tokens=old.num_tokens,
                    claim=claim,
                )
            )
        return recovered

    def discard(self, records, *, release_claims):
        self.discards.extend((record.block_id, release_claims) for record in records)
        for record in records:
            for block_hash in record.block_hashes:
                self.store.records.pop(bytes(block_hash), None)
            if release_claims:
                self.store.states[record.block_id] = "free"

    def rollback(self, claims):
        self.rollbacks.extend(claim.block_id for claim in claims)
        super().rollback(claims)

    def flush(self):
        self.flushes += 1


def _pool(provider: KVCacheBlockProvider) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
        persistence_provider=provider,
    )


def _cache_and_free(pool: BlockPool, block_hash: bytes):
    block = pool.get_new_blocks(1)[0]
    key = make_block_hash_with_group_id(block_hash, 0)
    pool._insert_block_hash(key, block, num_tokens=16)
    pool.free_blocks([block])
    return block, key


def test_provider_candidates_are_consumed_lazily():
    store = _SharedStore()

    class CountingProvider(_FakeProvider):
        candidates_seen = 0

        def acquire(self, candidates, count):
            def tracked_candidates():
                for candidate in candidates:
                    self.candidates_seen += 1
                    yield candidate

            return super().acquire(tracked_candidates(), count)

    provider = CountingProvider(store, "primary")
    pool = BlockPool(
        num_gpu_blocks=4096,
        enable_caching=True,
        hash_block_size=16,
        persistence_provider=provider,
    )

    assert pool.reserve_persistence_claims(3)
    assert provider.candidates_seen == 3
    pool.cancel_persistence_reservation()


def test_provider_commits_and_reactivates_native_cache_block():
    store = _SharedStore()
    pool = _pool(_FakeProvider(store, "primary"))
    block, key = _cache_and_free(pool, b"a" * 32)

    assert store.states[block.block_id] == "ready"
    assert bytes(key) in store.records

    hit = pool.get_cached_block(b"a" * 32, [0])
    assert hit == [block]
    assert store.states[block.block_id] == "active"


def test_shadow_recovers_exact_hash_into_native_index():
    store = _SharedStore()
    primary = _pool(_FakeProvider(store, "primary"))
    original, _ = _cache_and_free(primary, b"b" * 32)

    shadow = _pool(_FakeProvider(store, "shadow"))
    hit = shadow.get_cached_block(b"b" * 32, [0])

    assert hit is not None
    assert hit[0].block_id == original.block_id
    assert (
        shadow.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(b"b" * 32, 0)
        )
        is hit[0]
    )
    assert store.states[original.block_id] == "active"


def test_recovery_rejects_locally_busy_slot_and_rolls_back():
    store = _SharedStore()
    primary = _pool(_FakeProvider(store, "primary"))
    original, _ = _cache_and_free(primary, b"c" * 32)

    provider = _FakeProvider(store, "shadow")
    shadow = _pool(provider)
    key = make_block_hash_with_group_id(b"c" * 32, 0)
    busy = shadow.blocks[original.block_id]
    shadow.free_block_queue.remove(busy)
    busy.ref_cnt = 1

    assert shadow.get_cached_block(b"c" * 32, [0]) is None
    assert provider.discards == [(original.block_id, True)]
    assert bytes(key) not in store.records
    assert store.states[original.block_id] == "free"
    assert busy.block_hash is None


def test_recovery_does_not_reactivate_an_already_adopted_claim():
    store = _SharedStore()
    primary = _pool(_FakeProvider(store, "primary"))
    original, _ = _cache_and_free(primary, b"e" * 32)

    provider = _FakeProvider(store, "shadow")
    shadow = _pool(provider)
    generation_before = store.generations[original.block_id]

    assert shadow.get_cached_block(b"e" * 32, [0]) is not None
    assert provider.activations == 0
    assert store.generations[original.block_id] == generation_before + 1


def test_copy_on_write_invalidates_source_before_it_can_be_overwritten():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)
    source, key = _cache_and_free(pool, b"f" * 32)
    assert pool.get_cached_block(b"f" * 32, [0]) == [source]
    pool.touch([source])
    destination = pool.get_new_blocks(1)[0]

    pool.move_block_hashes(source, destination)

    assert provider.discards[-1] == (source.block_id, False)
    assert bytes(key) not in store.records
    assert source.block_hash is None
    assert destination.block_hash == key


def test_provider_persists_and_invalidates_all_native_hash_aliases():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)
    source = pool.get_new_blocks(1)[0]
    primary = make_block_hash_with_group_id(b"g" * 32, 0)
    alias = make_block_hash_with_group_id(b"h" * 32, 0)
    pool._insert_block_hash(primary, source, num_tokens=16)
    pool._insert_block_hash(alias, source, num_tokens=16)
    pool.free_blocks([source])

    assert bytes(primary) in store.records
    assert bytes(alias) in store.records

    assert pool.get_cached_block(b"g" * 32, [0]) == [source]
    pool.touch([source])
    destination = pool.get_new_blocks(1)[0]
    pool.move_block_hashes(source, destination)

    assert bytes(primary) not in store.records
    assert bytes(alias) not in store.records


def test_provider_contention_is_scheduler_backpressure():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)
    for block_id in range(1, 8):
        store.states[block_id] = "active"

    with pytest.raises(KVCacheBlockUnavailable):
        pool.get_new_blocks(1)
    assert pool.free_block_queue.num_free_blocks == 7


def test_reservation_backpressures_before_native_pool_mutation():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)
    for block_id in range(1, 8):
        store.states[block_id] = "active"

    assert not pool.reserve_persistence_claims(1)
    assert pool.free_block_queue.num_free_blocks == 7
    assert pool._persistence_reservation == []


def test_new_blocks_consume_one_preacquired_reservation():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)

    assert pool.reserve_persistence_claims(2)
    blocks = pool.get_new_blocks(2)
    pool.cancel_persistence_reservation()

    assert len(blocks) == 2
    assert provider.acquire_calls == 1
    assert pool._persistence_reservation == []


def test_unused_reservation_is_rolled_back():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)
    cached, key = _cache_and_free(pool, b"i" * 32)
    for block_id in range(1, 8):
        if block_id != cached.block_id:
            store.states[block_id] = "active"

    assert pool.reserve_persistence_claims(1)
    pool.cancel_persistence_reservation()

    assert provider.rollbacks == [cached.block_id]
    assert cached.block_hash is None
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert bytes(key) not in store.records
    assert pool.free_block_queue.num_free_blocks == 7


def test_invalid_provider_claim_is_rolled_back_without_mutating_native_pool():
    store = _SharedStore()

    class InvalidProvider(_FakeProvider):
        def acquire(self, candidates, count):
            return [KVCacheBlockClaim(999, (self.owner, 1))]

    provider = InvalidProvider(store, "primary")
    pool = _pool(provider)

    with pytest.raises(ValueError, match="invalid block claims"):
        pool.get_new_blocks(1)

    assert provider.rollbacks == [999]
    assert pool.free_block_queue.num_free_blocks == 7


def test_reset_discards_all_external_content_records():
    store = _SharedStore()
    pool = _pool(_FakeProvider(store, "primary"))
    block, _ = _cache_and_free(pool, b"d" * 32)

    assert pool.reset_prefix_cache()
    assert store.records == {}
    assert store.states[block.block_id] == "free"
    assert pool._persistence_claims == {}


def test_connector_factory_exposes_persistent_geometry_before_sizing(monkeypatch):
    config = SimpleNamespace(kv_transfer_config=object())
    monkeypatch.setattr(
        KVConnectorFactory,
        "get_connector_class",
        lambda _transfer_config: _GeometryConnector,
    )

    assert KVConnectorFactory.get_kv_cache_num_blocks_override(config) == 123


def test_connector_factory_preserves_native_geometry_without_connector():
    config = SimpleNamespace(kv_transfer_config=None)

    assert KVConnectorFactory.get_kv_cache_num_blocks_override(config) is None


def test_block_pool_flushes_provider_at_handoff_barrier():
    store = _SharedStore()
    provider = _FakeProvider(store, "primary")
    pool = _pool(provider)

    pool.flush_persistence()

    assert provider.flushes == 1
