# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension points for externally owned persistent KV cache blocks."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId


class KVCacheBlockUnavailable(RuntimeError):
    """The external provider could not safely make blocks available."""


@dataclass(frozen=True)
class KVCacheBlockClaim:
    """An opaque external ownership claim for one physical KV block."""

    block_id: int
    token: Any = None


@dataclass(frozen=True)
class KVCacheBlockCandidate:
    """A locally free block that an external provider may acquire."""

    block_id: int
    block_hash: BlockHashWithGroupId | None
    num_tokens: int | None
    claim: KVCacheBlockClaim | None


@dataclass(frozen=True)
class KVCacheBlockRecord:
    """A content-addressed block and its current external claim."""

    block_id: int
    block_hash: BlockHashWithGroupId
    num_tokens: int | None
    claim: KVCacheBlockClaim
    aliases: tuple[BlockHashWithGroupId, ...] = ()

    @property
    def block_hashes(self) -> tuple[BlockHashWithGroupId, ...]:
        return (self.block_hash, *self.aliases)


class KVCacheBlockProvider(ABC):
    """Coordinates vLLM block IDs with an external memory owner.

    Implementations must not mutate vLLM objects. Methods run on the scheduler
    thread and must be bounded. Remote discovery should be maintained in an
    asynchronous local view before ``recover`` is called.
    """

    @abstractmethod
    def bind(self, num_blocks: int, reserved_block_ids: Sequence[int]) -> None:
        """Bind this provider to one newly constructed vLLM block pool."""

    def get_num_available(self, native_free_blocks: int) -> int:
        """Return the number of blocks immediately available for allocation."""
        return native_free_blocks

    @abstractmethod
    def acquire(
        self,
        candidates: Iterable[KVCacheBlockCandidate],
        count: int,
    ) -> Sequence[KVCacheBlockClaim] | None:
        """Atomically acquire ``count`` candidate block IDs.

        Candidates are lazy and ordered by vLLM eviction priority. Providers
        should stop consuming them once they can satisfy the request.

        Selecting a cached candidate must also invalidate its old external
        content mapping. Return ``None`` to apply scheduler backpressure.
        """

    def activate(
        self, records: Sequence[KVCacheBlockRecord]
    ) -> Sequence[KVCacheBlockClaim] | None:
        """Pin cached records before vLLM exposes them as cache hits."""
        return [record.claim for record in records]

    @abstractmethod
    def commit(self, records: Sequence[KVCacheBlockRecord]) -> None:
        """Publish completed immutable blocks while retaining their claims."""

    @abstractmethod
    def release(self, claims: Sequence[KVCacheBlockClaim]) -> None:
        """Release claims for blocks that have no reusable content."""

    def recover(
        self, block_hashes: Sequence[BlockHashWithGroupId]
    ) -> Sequence[KVCacheBlockRecord] | None:
        """Claim exact persisted blocks in the requested order, if available."""
        return None

    @abstractmethod
    def discard(
        self,
        records: Sequence[KVCacheBlockRecord],
        *,
        release_claims: bool,
    ) -> None:
        """Remove external content mappings and optionally release claims."""

    def rollback(self, claims: Sequence[KVCacheBlockClaim]) -> None:
        """Undo claims after local validation or mutation fails."""
        self.release(claims)

    def flush(self) -> None:
        """Flush accepted metadata mutations before a clean handoff."""
        return None
