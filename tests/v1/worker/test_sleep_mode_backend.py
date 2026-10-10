# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the sleep-mode backend abstraction (RFC #34303).

These cover the registry/factory contract and capability flags. They do not
touch CUDA - the ``cumem`` suspend/resume path is exercised end-to-end on GPU
in ``tests/basic_correctness/memory/``.
"""

import pytest

from vllm.device_allocator.sleep_mode_backend import (
    CuMemBackend,
    SleepModeBackend,
    SleepModeBackendFactory,
)


def test_cumem_is_the_default_registered_backend():
    backend_cls = SleepModeBackendFactory.get_backend_class("cumem")
    assert backend_cls is CuMemBackend
    assert issubclass(backend_cls, SleepModeBackend)


def test_cumem_capability_flags():
    # cumem leaves communicators untouched but does not preserve compiled
    # artifacts, graphs, or durable state - these flags are what the executor and
    # /health introspect to decide reinit / persistence behavior.
    assert CuMemBackend.is_supported() is True
    assert CuMemBackend.preserves_communicators() is True
    assert CuMemBackend.preserves_compiled_artifacts() is False
    assert CuMemBackend.preserves_graphs_with_communicators() is False
    assert CuMemBackend.supports_durable_storage() is False


def test_new_backend_starts_in_running_state():
    # Constructing a backend must not touch the GPU; only suspend/resume do.
    assert CuMemBackend().state() == "RUNNING"


@pytest.mark.parametrize("enable_nccl_comm_suspend", [True, False])
def test_worker_drives_communicator_suspension(monkeypatch, enable_nccl_comm_suspend):
    """Comm walkers run around sleep/wake only when explicitly enabled."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    calls: list[tuple[str, object]] = []

    class Backend:
        def suspend(self, level: int = 1) -> None:
            calls.append(("backend.suspend", level))

        def resume(self, tags: list[str] | None = None) -> None:
            calls.append(("backend.resume", tuple(tags) if tags else None))

    worker = object.__new__(Worker)
    worker._sleep_mode_backend = Backend()
    worker._sleep_saved_parameters = {}
    worker._sleep_saved_buffers = {}
    worker._sleep_saved_draft_buffers = {}
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_nccl_comm_suspend=enable_nccl_comm_suspend)
    )

    monkeypatch.setattr("torch.accelerator.synchronize", lambda: None)
    monkeypatch.setattr("torch.accelerator.get_memory_info", lambda: (0, 0))
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.suspend_device_comms",
        lambda: calls.append(("comms.suspend", None)),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.resume_device_comms",
        lambda: calls.append(("comms.resume", None)),
    )

    worker.sleep(level=1)
    worker.wake_up(tags=["weights"])

    expected = [
        ("backend.suspend", 1),
        ("comms.suspend", None),
        ("backend.resume", ("weights",)),
        ("comms.resume", None),
    ]
    if not enable_nccl_comm_suspend:
        expected = [c for c in expected if not c[0].startswith("comms.")]
    assert calls == expected


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unsupported sleep-mode backend"):
        SleepModeBackendFactory.get_backend_class("does-not-exist")


def test_duplicate_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        SleepModeBackendFactory.register_backend(
            "cumem",
            "vllm.device_allocator.sleep_mode_backend",
            "CuMemBackend",
        )


def test_third_party_backend_registration_and_resolution():
    """A plugin registers a backend by name; the factory resolves it lazily."""
    name = "_pytest_dummy_backend"
    try:
        SleepModeBackendFactory.register_backend(
            name,
            "tests.v1.worker.test_sleep_mode_backend",
            "DummyBackend",
        )
        resolved = SleepModeBackendFactory.get_backend_class(name)
        assert resolved is DummyBackend
        assert resolved.supports_durable_storage() is True
    finally:
        SleepModeBackendFactory._registry.pop(name, None)


def test_suspend_resume_state_transitions():
    """Lifecycle state advances RUNNING -> SUSPENDED -> RUNNING without GPU."""
    backend = DummyBackend()
    assert backend.state() == "RUNNING"
    backend.suspend(level=1)
    assert backend.state() == "SUSPENDED"
    backend.resume()
    assert backend.state() == "RUNNING"


class DummyBackend(SleepModeBackend):
    """A no-GPU backend used to exercise lifecycle + registration in CPU tests."""

    def suspend(self, level: int = 1) -> None:
        self._state = "SUSPENDED"

    def resume(self, tags: list[str] | None = None) -> None:
        self._state = "RUNNING"

    @classmethod
    def supports_durable_storage(cls) -> bool:
        return True


def test_backends_do_not_preserve_kv_cache_by_default():
    # cumem unmaps KV memory at level 1+, so the prefix cache must be reset.
    assert SleepModeBackend.preserves_kv_cache() is False
    assert CuMemBackend.preserves_kv_cache() is False


class _KVPreservingBackend(CuMemBackend):
    @classmethod
    def preserves_kv_cache(cls, level: int = 1) -> bool:
        return True


@pytest.fixture
def kv_preserving_backend():
    name = "test-kv-preserving"
    SleepModeBackendFactory._registry.pop(name, None)
    SleepModeBackendFactory.register_backend(
        name, __name__, _KVPreservingBackend.__name__
    )
    yield name
    SleepModeBackendFactory._registry.pop(name, None)


def _sleep_decision(backend: str, level: int, clear_cache: bool | None) -> bool:
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    core = object.__new__(EngineCore)
    core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(sleep_mode_backend=backend)
    )
    return core._sleep_clears_prefix_cache(level, clear_cache)


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_resets_prefix_cache_when_backend_discards_kv(level):
    assert _sleep_decision("cumem", level, None) is True
    assert _sleep_decision("cumem", level, True) is True
    with pytest.raises(ValueError, match="preserves KV cache contents"):
        _sleep_decision("cumem", level, False)


@pytest.mark.parametrize("level", [1, 2])
def test_sleep_keeps_prefix_cache_when_backend_preserves_kv(
    kv_preserving_backend, level
):
    assert _sleep_decision(kv_preserving_backend, level, None) is False
    assert _sleep_decision(kv_preserving_backend, level, False) is False
    # Callers can still ask for a reset, e.g. before loading new weights.
    assert _sleep_decision(kv_preserving_backend, level, True) is True


def test_level_zero_sleep_only_clears_when_asked():
    assert _sleep_decision("cumem", 0, None) is False
    assert _sleep_decision("cumem", 0, False) is False
    assert _sleep_decision("cumem", 0, True) is True


def _single_buffer_kv_cache_config(size: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=size, layers=["layer.0"])],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer.0"],
                kv_cache_spec=SimpleNamespace(has_layer_views=False),
            )
        ],
    )


def test_kv_cache_allocation_zero_fills_by_default(monkeypatch):
    import torch

    from vllm.v1.worker import utils

    monkeypatch.setattr(
        utils.torch, "empty", lambda *a, **k: pytest.fail("must zero-fill")
    )
    caches = utils.allocate_kv_cache(
        _single_buffer_kv_cache_config(64), torch.device("cpu"), layout=None
    )
    assert torch.count_nonzero(caches["layer.0"]) == 0


def test_kv_cache_allocation_keeps_preserved_contents(monkeypatch):
    import torch

    from vllm.v1.worker import utils

    monkeypatch.setattr(
        utils.torch, "zeros", lambda *a, **k: pytest.fail("must not zero-fill")
    )
    with utils.preserve_kv_cache_contents():
        assert utils.kv_cache_contents_preserved()
        caches = utils.allocate_kv_cache(
            _single_buffer_kv_cache_config(64), torch.device("cpu"), layout=None
        )
    assert not utils.kv_cache_contents_preserved()
    assert caches["layer.0"].numel() == 64


def _worker_with_backend(backend, *, enable_sleep_mode=True):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker._sleep_mode_backend = backend
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=enable_sleep_mode, enable_cumem_allocator=False
        )
    )
    return worker


def test_worker_allocates_inside_backend_allocation_context():
    from contextlib import nullcontext

    seen = []
    placed = nullcontext()

    class Backend:
        def allocation_context(self, tag, *, kv_cache_config=None):
            seen.append((tag, kv_cache_config))
            return placed

    worker = _worker_with_backend(Backend())
    kv_config = object()
    assert (
        worker._maybe_get_memory_pool_context("kv_cache", kv_cache_config=kv_config)
        is placed
    )
    assert worker._maybe_get_memory_pool_context("weights") is placed
    assert seen == [("kv_cache", kv_config), ("weights", None)]


def test_worker_falls_back_when_backend_declines_a_tag(monkeypatch):
    from contextlib import nullcontext

    from vllm.v1.worker import gpu_worker

    class Backend:
        def allocation_context(self, tag, *, kv_cache_config=None):
            return None

    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)
    worker = _worker_with_backend(Backend())
    # cumem allocator disabled -> the default path is a plain allocation.
    assert isinstance(worker._maybe_get_memory_pool_context("kv_cache"), nullcontext)


def test_worker_ignores_backend_without_sleep_mode(monkeypatch):
    from contextlib import nullcontext

    from vllm.v1.worker import gpu_worker

    class Backend:
        def allocation_context(self, tag, *, kv_cache_config=None):
            pytest.fail("backend must not be consulted without sleep mode")

    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: True)
    worker = _worker_with_backend(Backend(), enable_sleep_mode=False)
    assert isinstance(worker._maybe_get_memory_pool_context("kv_cache"), nullcontext)


class _SelfAllocatingBackend(CuMemBackend):
    @classmethod
    def requires_cumem_allocator(cls) -> bool:
        return False


def test_cumem_allocator_requirement_follows_the_backend():
    name = "test-self-allocating"
    SleepModeBackendFactory._registry.pop(name, None)
    SleepModeBackendFactory.register_backend(
        name, __name__, _SelfAllocatingBackend.__name__
    )
    try:
        assert SleepModeBackendFactory.requires_cumem_allocator("cumem") is True
        assert SleepModeBackendFactory.requires_cumem_allocator(name) is False
        # A plugin backend not registered yet keeps today's behavior.
        assert SleepModeBackendFactory.requires_cumem_allocator("unknown") is True
    finally:
        SleepModeBackendFactory._registry.pop(name, None)


def test_default_backend_allocation_context_is_unset():
    assert CuMemBackend().allocation_context("kv_cache") is None


def test_offload_kv_backend_offloads_kv_cache_only_at_level_one():
    from vllm.device_allocator.sleep_mode_backend import CuMemOffloadKVBackend

    assert (
        SleepModeBackendFactory.get_backend_class("cumem_offload_kv")
        is CuMemOffloadKVBackend
    )
    backend = CuMemOffloadKVBackend()
    assert "kv_cache" in backend.offload_tags(1)
    assert "weights" in backend.offload_tags(1)
    assert "kv_cache" not in backend.offload_tags(2)
    assert backend.offload_tags(2) == CuMemBackend().offload_tags(2)
    assert CuMemOffloadKVBackend.preserves_kv_cache(1) is True
    assert CuMemOffloadKVBackend.preserves_kv_cache(2) is False
    assert "kv_cache" not in CuMemBackend().offload_tags(1)


def test_offload_kv_backend_keeps_prefix_cache_only_at_level_one():
    assert _sleep_decision("cumem_offload_kv", 1, None) is False
    assert _sleep_decision("cumem_offload_kv", 2, None) is True
    with pytest.raises(ValueError, match="preserves KV cache contents"):
        _sleep_decision("cumem_offload_kv", 2, False)
