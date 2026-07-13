# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (  # noqa: E501
    ExampleConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker import kv_connector_model_runner_mixin as mixin_module
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# Importing utils registers TestExampleConnector with the factory
from .utils import create_vllm_config


def _make_empty_scheduler_output():
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=ExampleConnectorMetadata(),
    )


def test_kv_connector_mixin_clears_metadata():
    vllm_config = create_vllm_config(
        kv_connector="TestExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"name": "unit"},
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    # Initialize the global connector instance.
    # kv_transfer init now syncs engine_id across TP, so unit tests need
    # a minimal mocked TP group.
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value

    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)

    try:
        # Minimal scheduler output with empty metadata; mixin should still
        # bind/clear metadata even if no loads happen
        scheduler_output = _make_empty_scheduler_output()

        # Invoke the no-forward path which uses the mixin context manager
        KVConnectorModelRunnerMixin.kv_connector_no_forward(
            scheduler_output, vllm_config
        )

        # Verify clear_connector_metadata was called on the connector
        connector = get_kv_transfer_group()
        assert connector._connector_metadata is None
        # Test connector wrapper records method calls
        assert connector.call_record.get("bind_connector_metadata", 0) == 1
        assert connector.call_record.get("clear_connector_metadata", 0) == 1
    finally:
        # Ensure we clean up the global connector between tests
        ensure_kv_transfer_shutdown()


def test_kv_cache_backing_uses_native_allocation_without_connector(monkeypatch):
    monkeypatch.setattr(mixin_module, "has_kv_transfer_group", lambda: False)

    tensor = KVConnectorModelRunnerMixin.allocate_kv_cache_backing(
        [KVCacheTensor(size=8, shared_by=["layer.0"])],
        8,
        torch.device("cpu"),
    )

    assert tensor.shape == (8,)
    assert tensor.dtype == torch.int8
    assert torch.count_nonzero(tensor) == 0


def test_kv_cache_backing_delegates_once_at_semantic_boundary(monkeypatch):
    logical = [KVCacheTensor(size=8, shared_by=["layer.0"])]
    calls = []

    def allocator(tensors, size, device):
        calls.append((tensors, size, device))
        return torch.empty(size, dtype=torch.int8, device=device)

    connector = MagicMock()
    connector.get_kv_cache_tensor_allocator.return_value = allocator
    monkeypatch.setattr(mixin_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mixin_module, "get_kv_transfer_group", lambda: connector)

    KVConnectorModelRunnerMixin.allocate_kv_cache_backing(
        logical, 8, torch.device("cpu")
    )

    assert calls == [(logical, 8, torch.device("cpu"))]


def test_kv_cache_backing_rejects_invalid_external_tensor(monkeypatch):
    connector = MagicMock()
    connector.get_kv_cache_tensor_allocator.return_value = (
        lambda _tensors, _size, _device: torch.empty(7, dtype=torch.int8)
    )
    monkeypatch.setattr(mixin_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mixin_module, "get_kv_transfer_group", lambda: connector)

    with pytest.raises(ValueError, match="invalid KV cache tensor allocator"):
        KVConnectorModelRunnerMixin.allocate_kv_cache_backing(
            [KVCacheTensor(size=8, shared_by=["layer.0"])],
            8,
            torch.device("cpu"),
        )
