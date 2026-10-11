# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the IPC weight cache loader.

A cold start (no weight cache daemon, so the loader falls back to disk) and
warm restarts (weights mapped from the daemon via CUDA IPC) must both serve
identical outputs.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest
import regex as re
import torch

from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


class WeightCacheDaemon:
    """Context manager running the real weight cache daemon as a subprocess."""

    def __init__(
        self,
        model: str,
        tp_size: int,
        extra_args: list[str] | None = None,
    ):
        # Short base path: Unix socket paths are limited to ~107 characters.
        self.socket_dir = tempfile.mkdtemp(prefix="vllm_ipc_")
        self.health_port = get_open_port()
        self._cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "preload",
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--weight-cache-socket-dir",
            self.socket_dir,
            "--enforce-eager",
            "--weight-cache-health-host",
            "127.0.0.1",
            "--weight-cache-health-port",
            str(self.health_port),
            *(extra_args or []),
        ]
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "WeightCacheDaemon":
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop()

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        self._proc.stderr.read()

    def check_health(self) -> None:
        url = f"http://127.0.0.1:{self.health_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                assert response.status == 200
        except urllib.error.HTTPError as error:
            raise AssertionError(
                f"Weight cache daemon is not ready: HTTP {error.code}"
            ) from error

    def wait_until_ready(self, timeout_s: float = 900) -> None:
        """Poll until READY (which covers the autotune tuners) or the preload
        process dies."""
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.check_health()
                return
            except (AssertionError, urllib.error.URLError):
                if time.monotonic() > deadline:
                    raise
                if self._proc is not None and self._proc.poll() is not None:
                    raise AssertionError("vllm preload exited before ready") from None
                time.sleep(2)

    def _stop(self) -> None:
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


@dataclass
class ModelCase:
    model: str
    prompts: list[str]
    images: list | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    daemon_args: list[str] = field(default_factory=list)
    # Additionally assert the daemons tune in place before serving and write
    # the on-disk FlashInfer autotune cache, and the restart engine reuses it
    # (needs SM90+/FlashInfer).
    check_flashinfer_cache: bool = False


def generate(
    vllm_runner,
    case: ModelCase,
    socket_dir: str | None,
    fallback: bool,
):
    extra_config = (
        {} if socket_dir is None else {"socket_dir": socket_dir, "fallback": fallback}
    )
    with vllm_runner(
        case.model,
        load_format="auto" if socket_dir is None else "ipc_cache",
        model_loader_extra_config=extra_config,
        # Greedy outputs are compared across engine restarts; cap the batch
        # size so scheduling cannot change the results.
        max_num_seqs=1,
        **case.llm_kwargs,
    ) as llm:
        sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        return llm.generate(case.prompts, sampling_params, images=case.images)


# Both hybrid-recurrent models need chunked prefill for their mamba-style
# cache mode.
QWEN_CASE = ModelCase(
    model="Qwen/Qwen3.5-0.8B",
    prompts=[
        "Hello, my name is",
        "The capital of France is",
    ],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_chunked_prefill=True,
    ),
)

# Kimi K3 adds the mxfp4-pack MoE path (post-load kernel setup runs in
# pre-processed mode) and a vision tower, exercised with an image request.
K3_CASE = ModelCase(
    model="tiny-random/kimi-k3",
    prompts=[
        "The capital of France is",
        "def quicksort(arr):",
        "<|kimi_image_placeholder|>What is shown in the image?",
    ],
    images=[None, None, ImageAsset("stop_sign").pil_image],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        # The image placeholder expands to >1k tokens for the stop_sign asset.
        max_model_len=4096,
    ),
    daemon_args=["--trust-remote-code"],
    # Kimi K3 has tunable FlashInfer ops (a dense model like Qwen2.5 tunes
    # nothing and upstream never writes the cache file for it).
    check_flashinfer_cache=True,
)

# Qwen3.5-0.8B ships one MTP layer in the target checkpoint, so method="mtp"
# loads the draft from the same model. The daemon must cache it in its draft
# group for the warm runs (fallback=False) to succeed.
QWEN_MTP_CASE = ModelCase(
    model="Qwen/Qwen3.5-0.8B",
    prompts=[
        "Hello, my name is",
        "The capital of France is",
    ],
    llm_kwargs=dict(
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_chunked_prefill=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    ),
    daemon_args=[
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 1}',
    ],
)


@pytest.mark.parametrize(
    "case",
    [QWEN_CASE, K3_CASE, QWEN_MTP_CASE],
    ids=["qwen3.5", "kimi-k3", "qwen3.5-mtp"],
)
def test_ipc_cache_cold_start_and_warm_restart(
    vllm_runner, case: ModelCase, tmp_path, monkeypatch, capfd
):
    """Cold start falls back to disk; warm restarts load weights via CUDA IPC.

    All runs must produce outputs identical to a default-loader baseline. The
    warm runs disable the disk fallback, so they only pass if the weights
    really came from the daemon — for the MTP case, both the target's and the
    draft's daemon groups.

    Cases with ``check_flashinfer_cache`` additionally cover in-daemon
    autotune (default-on with `--enable-flashinfer-autotune`): each daemon
    tunes before binding its socket and must write the on-disk FlashInfer
    autotune cache, and the restart engine must load tactics from it instead
    of re-profiling. The daemon's cache file is keyed with the
    OPENAI_API_SERVER batch defaults while these engines run under LLM_CLASS,
    so they never hit the daemon's file by design; reuse is proven against
    the file the earlier in-process engines wrote.
    """
    if not current_platform.is_cuda_alike():
        pytest.skip("Weight cache IPC sharing requires CUDA or ROCm")
    if case is K3_CASE and not current_platform.is_device_capability_family(100):
        pytest.skip("Kimi K3 IPC weight cache requires an SM100 MXFP4 backend")
    if case.check_flashinfer_cache:
        from vllm.utils.flashinfer import has_flashinfer

        if not (has_flashinfer() and current_platform.has_device_capability(90)):
            pytest.skip("FlashInfer autotune requires FlashInfer and SM90+")
        # Isolate the on-disk autotune cache. The engines are in-process
        # (envs read lazily); the daemon subprocesses spawn later and
        # inherit it.
        monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))

    # Baseline: plain disk loading with the default loader.
    baseline_outputs = generate(
        vllm_runner,
        case,
        None,
        fallback=True,
    )
    assert all(text for _, texts in baseline_outputs for text in texts)

    # Cold start: no daemon is serving, so the loader falls back to disk.
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_empty_") as empty_socket_dir:
        cold_outputs = generate(vllm_runner, case, empty_socket_dir, fallback=True)

    if case.check_flashinfer_cache:
        # The baseline engine already wrote its own (LLM_CLASS-keyed) table.
        engine_cache_files = set(tmp_path.rglob("autotune_configs*.json"))

    with WeightCacheDaemon(
        case.model,
        tp_size=1,
        extra_args=case.daemon_args,
    ) as d:
        warm_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        d.wait_until_ready()
        if case.check_flashinfer_cache:
            # Each daemon tunes before its ready report; exactly one new file
            # (their OPENAI_API_SERVER-keyed table) must appear.
            tuner_files = set(tmp_path.rglob("autotune_configs*.json"))
            tuner_files -= engine_cache_files
            assert len(tuner_files) == 1, (
                f"expected the tuner's tuned table, got {tuner_files}"
            )
        # Warm restart: a second engine lifetime against the same daemon.
        if case.check_flashinfer_cache:
            capfd.readouterr()  # drain: the assertion reads the increment
        restart_outputs = generate(vllm_runner, case, d.socket_dir, fallback=False)
        if case.check_flashinfer_cache:
            # The restart's EngineCore subprocess inherits pytest's fds; the
            # flashinfer autotuner logs "[Autotuner]: Loaded N configs from
            # <path>" (INFO, not once-deduplicated) on a disk cache hit.
            out = capfd.readouterr()
            m = re.search(
                r"\[Autotuner\]: Loaded (\d+) configs from", out.out + out.err
            )
            assert m and int(m.group(1)) > 0, (
                "restart engine did not load the on-disk autotune cache"
            )

    assert cold_outputs == baseline_outputs
    assert warm_outputs == baseline_outputs
    assert restart_outputs == baseline_outputs


def test_vmm_fd_engine_keeps_serving_after_daemon_dies(vllm_runner):
    """With --weight-cache-transport vmm_fd the engine holds its own reference
    to the cached weights, so it keeps producing the same outputs after the
    daemon is killed."""
    if not current_platform.is_cuda():
        pytest.skip("vmm_fd needs CUDA VMM")
    case = ModelCase(
        model="Qwen/Qwen3-0.6B",
        prompts=["Hello, my name is", "The capital of France is"],
        llm_kwargs=dict(gpu_memory_utilization=0.3, enforce_eager=True),
    )
    baseline_outputs = generate(vllm_runner, case, None, fallback=True)
    sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    with (
        WeightCacheDaemon(
            case.model,
            tp_size=1,
            extra_args=["--weight-cache-transport", "vmm_fd"],
        ) as d,
        vllm_runner(
            case.model,
            load_format="ipc_cache",
            model_loader_extra_config={"socket_dir": d.socket_dir, "fallback": False},
            max_num_seqs=1,
            **case.llm_kwargs,
        ) as llm,
    ):
        warm_outputs = llm.generate(case.prompts, sampling_params)
        assert d._proc is not None
        d._proc.kill()
        d._proc.wait(timeout=60)
        after_kill_outputs = llm.generate(case.prompts, sampling_params)

    assert warm_outputs == baseline_outputs
    assert after_kill_outputs == baseline_outputs


_KV_RESTORE_ENGINE = """
import json, sys, time
from vllm import LLM, SamplingParams
cfg = json.loads(sys.argv[1])
llm = LLM(
    model=cfg["model"],
    load_format="ipc_cache",
    model_loader_extra_config={
        "socket_dir": cfg["socket_dir"],
        "fallback": False,
        "kv_restore": cfg["kv_restore"],
    },
    gpu_memory_utilization=0.3,
    enforce_eager=True,
    max_num_seqs=1,
    enable_prefix_caching=True,
)
sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
results = []
for prompt in cfg["prompts"]:
    start = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)[0]
    results.append(
        {
            "text": out.outputs[0].text,
            "cached_tokens": out.num_cached_tokens,
            "prompt_tokens": len(out.prompt_token_ids),
            "latency_s": time.perf_counter() - start,
        }
    )
print("KV_RESTORE_RESULT " + json.dumps(results), flush=True)
if cfg["hold"]:
    time.sleep(3600)
"""


def _run_kv_restore_engine(cfg: dict, hold: bool = False):
    """Run one engine in its own process group; return (results, process)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _KV_RESTORE_ENGINE, json.dumps({**cfg, "hold": hold})],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    log = []
    for line in proc.stdout:
        log.append(line)
        if line.startswith("KV_RESTORE_RESULT "):
            return json.loads(line.split(" ", 1)[1]), proc, log
    proc.wait()
    raise AssertionError("engine exited without results:\n" + "".join(log[-60:]))


def test_kv_restore_survives_engine_crash():
    """An engine SIGKILLed mid-life leaves its KV cache with the daemon; the
    next engine re-attaches it and serves the same prompts from the restored
    prefix cache, with identical greedy outputs."""
    if not current_platform.is_cuda():
        pytest.skip("KV restore needs CUDA VMM")
    model = "Qwen/Qwen3-0.6B"
    prefix = " ".join(
        f"Fact {i}: the weight cache daemon keeps state across restarts."
        for i in range(120)
    )
    prompts = [
        prefix + " Question: what does the daemon keep?",
        prefix + " Question: what survives a restart?",
    ]
    with WeightCacheDaemon(model, tp_size=1) as d:
        cfg = {"model": model, "socket_dir": d.socket_dir, "prompts": prompts}
        cold, _, _ = _run_kv_restore_engine({**cfg, "kv_restore": False})

        warm_a, engine_a, _ = _run_kv_restore_engine(
            {**cfg, "kv_restore": True}, hold=True
        )
        os.killpg(engine_a.pid, signal.SIGKILL)
        engine_a.wait(timeout=60)
        time.sleep(2)  # let the daemon see the dropped lease

        restored, engine_b, log_b = _run_kv_restore_engine({**cfg, "kv_restore": True})
        engine_b.wait(timeout=120)

    print("cold", cold, "\nengine A", warm_a, "\nengine B", restored)
    assert any("imported" in line and "cached blocks" in line for line in log_b)
    # Engine A started cold: its first prompt had no cache hits.
    assert warm_a[0]["cached_tokens"] == 0
    # Engine B's very first request hits the prefix cache engine A built.
    assert restored[0]["cached_tokens"] > 0.9 * restored[0]["prompt_tokens"] - 16
    assert [r["text"] for r in restored] == [r["text"] for r in cold]
    assert [r["text"] for r in warm_a] == [r["text"] for r in cold]


_KV_STANDBY_ENGINE = """
import json, sys, time
from vllm import LLM, SamplingParams
cfg = json.loads(sys.argv[1])
llm = LLM(
    model=cfg["model"],
    load_format="ipc_cache",
    model_loader_extra_config={
        "socket_dir": cfg["socket_dir"],
        "fallback": False,
        "kv_restore": True,
        "kv_restore_role": "standby",
    },
    gpu_memory_utilization=0.3,
    enforce_eager=True,
    max_num_seqs=1,
    enable_prefix_caching=True,
)
print("KV_STANDBY_READY", flush=True)
assert sys.stdin.readline().strip() == "promote"
start = time.perf_counter()
adopted = llm.llm_engine.engine_core.call_utility("kv_restore_promote", 60.0, 2.0)
promote_s = time.perf_counter() - start
sp = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
results = []
for prompt in cfg["prompts"]:
    out = llm.generate([prompt], sp, use_tqdm=False)[0]
    results.append(
        {
            "text": out.outputs[0].text,
            "cached_tokens": out.num_cached_tokens,
            "prompt_tokens": len(out.prompt_token_ids),
        }
    )
print(
    "KV_RESTORE_RESULT "
    + json.dumps({"adopted": adopted, "promote_s": promote_s, "results": results}),
    flush=True,
)
"""


def _gpu_used_bytes() -> int:
    free, total = torch.cuda.mem_get_info(0)
    return total - free


def test_kv_restore_standby_takes_over_from_a_frozen_writer():
    """A standby engine maps the writer's KV pool and the daemon's weights
    (no second copy). The writer is frozen (SIGSTOP: alive, still connected,
    like ranks stuck in a collective with a dead one); the standby is promoted
    without waiting for it and serves the writer's prompts from the inherited
    prefix cache, with identical greedy outputs."""
    if not current_platform.is_cuda():
        pytest.skip("KV restore needs CUDA VMM")
    model = "Qwen/Qwen3-0.6B"
    prefix = " ".join(
        f"Fact {i}: the standby engine takes over the writer's KV cache."
        for i in range(120)
    )
    prompts = [
        prefix + " Question: who takes over?",
        prefix + " Question: what is inherited?",
    ]
    with WeightCacheDaemon(model, tp_size=1) as d:
        cfg = {"model": model, "socket_dir": d.socket_dir, "prompts": prompts}
        cold, _, _ = _run_kv_restore_engine({**cfg, "kv_restore": False})

        warm_a, engine_a, _ = _run_kv_restore_engine(
            {**cfg, "kv_restore": True}, hold=True
        )
        used_before_b = _gpu_used_bytes()
        engine_b = subprocess.Popen(
            [sys.executable, "-c", _KV_STANDBY_ENGINE, json.dumps(cfg)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        assert engine_b.stdout is not None and engine_b.stdin is not None
        log_b = []
        for line in engine_b.stdout:
            log_b.append(line)
            if line.startswith("KV_STANDBY_READY"):
                break
        else:
            raise AssertionError("standby exited:\n" + "".join(log_b[-60:]))
        standby_added = _gpu_used_bytes() - used_before_b

        os.killpg(engine_a.pid, signal.SIGSTOP)
        engine_b.stdin.write("promote\n")
        engine_b.stdin.flush()
        result = None
        for line in engine_b.stdout:
            log_b.append(line)
            if line.startswith("KV_RESTORE_RESULT "):
                result = json.loads(line.split(" ", 1)[1])
                break
        engine_b.stdin.close()
        engine_b.wait(timeout=120)
        os.killpg(engine_a.pid, signal.SIGKILL)
        engine_a.wait(timeout=60)
        assert result is not None, "".join(log_b[-60:])

    pool_bytes = next(
        float(m.group(1))
        for line in log_b
        if (
            m := re.search(r"standby mapped the writer's KV cache \(([\d.]+) GiB", line)
        )
    ) * (1 << 30)
    print(
        f"standby added {standby_added / (1 << 30):.2f} GiB next to a "
        f"{pool_bytes / (1 << 30):.2f} GiB KV pool; promote took "
        f"{result['promote_s']:.3f}s, adopted {result['adopted']} blocks;",
        result["results"],
    )
    # The standby did not allocate a second KV cache.
    assert standby_added < pool_bytes / 2
    assert result["adopted"] > 0
    first = result["results"][0]
    assert first["cached_tokens"] > 0.9 * first["prompt_tokens"] - 16
    assert [r["text"] for r in result["results"]] == [r["text"] for r in cold]
    assert [r["text"] for r in warm_a] == [r["text"] for r in cold]


def _parallel(**kw):
    from types import SimpleNamespace

    base = dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        nnodes=1,
        node_rank=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_daemon_places_tp_and_dp_ranks_on_local_gpus():
    """Each node serves a contiguous global-rank block (DP-major), so without
    DP local GPU i is TP rank r*local+i, and single-node DP launchers offset
    by --data-parallel-start-rank."""
    from vllm.model_executor.model_loader.weight_cache.daemon import plan_local_ranks

    tp_second_node = _parallel(tensor_parallel_size=8, nnodes=2, node_rank=1)
    assert plan_local_ranks(tp_second_node) == [(i, 0, 0, 4 + i) for i in range(4)]

    dp_third_node = _parallel(
        data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=8
    )
    assert plan_local_ranks(dp_third_node) == [(i, 8 + i, 0, 0) for i in range(4)]

    dp_tp = _parallel(
        tensor_parallel_size=2,
        data_parallel_size=4,
        data_parallel_size_local=2,
        data_parallel_rank=2,
    )
    assert plan_local_ranks(dp_tp) == [
        (0, 2, 0, 0),
        (1, 2, 0, 1),
        (2, 3, 0, 0),
        (3, 3, 0, 1),
    ]

    # DP + nnodes: node-local TP replicas
    dp_tp_node0 = _parallel(tensor_parallel_size=8, data_parallel_size=2, nnodes=2)
    assert plan_local_ranks(dp_tp_node0) == [(i, 0, 0, i) for i in range(8)]
    dp_tp_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=2, node_rank=1
    )
    assert plan_local_ranks(dp_tp_node1) == [(i, 1, 0, i) for i in range(8)]

    # DP + nnodes: TP group spans nodes (TP8 x DP2 on 4 nodes)
    tp_span_node1 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=1
    )
    assert plan_local_ranks(tp_span_node1) == [(i, 0, 0, 4 + i) for i in range(4)]
    tp_span_node2 = _parallel(
        tensor_parallel_size=8, data_parallel_size=2, nnodes=4, node_rank=2
    )
    assert plan_local_ranks(tp_span_node2) == [(i, 1, 0, i) for i in range(4)]

    # DP + nnodes: several DP replicas per node (TP4 x DP4 on 2 nodes)
    dp_multi_node0 = _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    assert plan_local_ranks(dp_multi_node0) == [(i, i // 4, 0, i % 4) for i in range(8)]

    # PP adds one daemon placement per stage while preserving TP rank order.
    pp_tp = _parallel(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert plan_local_ranks(pp_tp) == [
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (2, 0, 1, 0),
        (3, 0, 1, 1),
    ]


def test_daemon_rejects_unmappable_parallelism():
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        _reject_unsupported_parallelism,
    )

    _reject_unsupported_parallelism(
        _parallel(
            data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=12
        )
    )
    # DP combines with --nnodes when the world size divides evenly.
    _reject_unsupported_parallelism(
        _parallel(tensor_parallel_size=4, data_parallel_size=4, nnodes=2)
    )
    _reject_unsupported_parallelism(_parallel(pipeline_parallel_size=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(_parallel(tensor_parallel_size=3, nnodes=2))
    with pytest.raises(ValueError, match="evenly divide"):
        _reject_unsupported_parallelism(
            _parallel(tensor_parallel_size=2, data_parallel_size=3, nnodes=4)
        )
    with pytest.raises(ValueError, match="exceeds"):
        _reject_unsupported_parallelism(
            _parallel(
                data_parallel_size=16, data_parallel_size_local=4, data_parallel_rank=13
            )
        )


def test_weight_cache_key_distinguishes_dp_ranks():
    from dataclasses import replace

    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey

    key = WeightCacheKey(
        checkpoint="ckpt",
        model_arch="Arch",
        tp_size=1,
        tp_rank=0,
        dtype="bf16",
        quantization=None,
        quant_config_hash="h",
        revision=None,
        vllm_version="v",
        dp_size=16,
        dp_rank=3,
        pp_size=2,
        pp_rank=1,
    )
    assert key.mismatched_fields(replace(key, dp_rank=4)) == ["dp_rank"]
    assert key.mismatched_fields(replace(key, dp_size=8, dp_rank=3)) == ["dp_size"]
    assert key.mismatched_fields(replace(key, pp_rank=0)) == ["pp_rank"]


@pytest.mark.parametrize("hf_quant_config", [None, {"quant_algo": "NVFP4"}])
def test_weight_cache_key_distinguishes_nvfp4_activation_override(
    tmp_path, hf_quant_config
):
    """Cache keys isolate activation modes and reject ambiguous legacy hashes."""
    import json
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey
    from vllm.utils.hashing import safe_hash

    model_config = SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        quantization="modelopt_fp4",
        quantization_config=None,
        revision=None,
        hf_config=SimpleNamespace(
            architectures=["NemotronHForCausalLM"],
            quantization_config=hf_quant_config,
        ),
    )
    static_key = WeightCacheKey.from_model_config(model_config, tp_size=1, tp_rank=0)
    model_config.quantization_config = QuantizationConfigArgs(
        moe={"activation": "nvfp4_per_token"}
    )
    per_token_key = WeightCacheKey.from_model_config(model_config, tp_size=1, tp_rank=0)
    assert static_key.mismatched_fields(per_token_key) == ["quant_config_hash"]

    legacy_hash = (
        ""
        if hf_quant_config is None
        else safe_hash(
            json.dumps(hf_quant_config, sort_keys=True).encode(),
            usedforsecurity=False,
        ).hexdigest()
    )
    assert static_key.mismatched_fields(
        replace(static_key, quant_config_hash=legacy_hash)
    ) == ["quant_config_hash"]


def test_ipc_loader_copy_mode_reports_no_external_weight_memory():
    """Copy mode clones the weights into the engine, so nothing is external
    (vllm_config is never touched)."""
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import (
        IpcModelLoader,
    )

    copy_mode = LoadConfig(
        load_format="ipc_cache", model_loader_extra_config={"mode": "copy"}
    )
    loader = IpcModelLoader(copy_mode)
    assert loader.get_external_weight_memory(None) == 0
