# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end: `vllm serve` crashes and a new engine resumes from the weight
cache daemon, without any orchestrator.

A plain user runs `vllm preload` (the weight cache daemon, vmm_fd transport)
next to `vllm serve --load-format ipc_cache` with KV restore. The server is
SIGKILLed with a streaming request in flight; then:

- restart: a new `vllm serve` attaches to the same daemon. Its weights come
  from the daemon (no disk load), its very first request hits the prefix
  cache the crashed server built, and its greedy outputs match the pre-crash
  server and a plain disk-loaded server. The crashed server's in-flight
  (allocated, uncommitted) blocks are quarantined, not adopted.
- warm standby: a standby engine maps the writer's weights and KV pool next
  to it, warms up without writing the pool, and is promoted with the
  `kv_restore_promote` engine utility once the writer is killed.
"""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import requests

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 32
READY_TIMEOUT_S = 900.0
# Same KV size for every engine, so the restarted ones match the daemon's pool,
# and the same batch limits (`vllm serve` and AsyncLLM default to different
# ones), so greedy outputs compare across engines.
SERVER_ARGS = [
    "--max-num-batched-tokens",
    "8192",
    "--max-model-len",
    "4096",
    "--max-num-seqs",
    "8",
    "--kv-cache-memory-bytes",
    str(2 << 30),
    "--enable-prefix-caching",
]

PREFIX = " ".join(
    f"Record {i}: the weight cache daemon keeps the model weights and the "
    "KV cache of crashed engines."
    for i in range(60)
)
PROMPTS = [
    PREFIX + " Question: what does the daemon keep?",
    PREFIX + " Question: what survives a crash?",
    PREFIX + " Question: who restarts the engine?",
]
IN_FLIGHT_PROMPT = PREFIX + " Write a very long essay about engine restarts."

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="KV restore needs CUDA VMM"
)


class _Process:
    """A subprocess in its own process group, logging to a file."""

    def __init__(self, name: str, cmd: list[str], log_dir: Path, **kwargs):
        self.name = name
        self.log_path = log_dir / f"{name}.log"
        self._log = open(self.log_path, "w")  # noqa: SIM115
        self.start = time.monotonic()
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            **kwargs,
        )

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait_for(self, ready: Callable[[], bool], timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready():
                return time.monotonic() - self.start
            if not self.alive():
                raise AssertionError(
                    f"{self.name} exited ({self.proc.returncode}) before {what}:\n"
                    + self.log()[-6000:]
                )
            time.sleep(0.2)
        raise AssertionError(f"{self.name}: no {what} in {timeout}s")

    def kill(self) -> None:
        """SIGKILL the whole process tree (the process group)."""
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=60)
        self._log.close()

    def terminate(self, timeout: float = 60) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=timeout)
        self.kill()


@pytest.fixture
def processes():
    """Processes to tear down, in reverse order, however the test ends."""
    procs: list[_Process] = []
    yield procs
    for p in reversed(procs):
        p.terminate()


def _http_ok(url: str) -> bool:
    try:
        return requests.get(url, timeout=2).status_code == 200
    except requests.RequestException:
        return False


def start_daemon(log_dir: Path, socket_dir: str, procs: list) -> _Process:
    health_port = get_open_port()
    daemon = _Process(
        "daemon",
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "preload",
            "--model",
            MODEL,
            "--enforce-eager",
            "--weight-cache-socket-dir",
            socket_dir,
            "--weight-cache-transport",
            "vmm_fd",
            "--weight-cache-health-host",
            "127.0.0.1",
            "--weight-cache-health-port",
            str(health_port),
        ],
        log_dir,
    )
    procs.append(daemon)
    daemon.wait_for(
        lambda: _http_ok(f"http://127.0.0.1:{health_port}/health"),
        READY_TIMEOUT_S,
        "daemon health",
    )
    return daemon


class Server:
    """`vllm serve` as a user runs it, with its time to ready."""

    def __init__(self, name: str, log_dir: Path, procs: list, loader: dict | None):
        self.port = get_open_port()
        self.url = f"http://127.0.0.1:{self.port}"
        extra = []
        if loader is not None:
            extra = [
                "--load-format",
                "ipc_cache",
                "--model-loader-extra-config",
                json.dumps(loader),
            ]
        self.proc = _Process(
            name,
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--enable-prompt-tokens-details",
                *SERVER_ARGS,
                *extra,
            ],
            log_dir,
        )
        procs.append(self.proc)
        self.ready_s = self.proc.wait_for(
            lambda: _http_ok(self.url + "/health"), READY_TIMEOUT_S, "/health"
        )

    def complete(self, prompt: str) -> dict:
        r = requests.post(
            self.url + "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
            timeout=120,
        )
        r.raise_for_status()
        body = r.json()
        details = body["usage"].get("prompt_tokens_details") or {}
        return {
            "text": body["choices"][0]["text"],
            "cached_tokens": details.get("cached_tokens") or 0,
            "prompt_tokens": body["usage"]["prompt_tokens"],
        }

    def run(self, prompts: list[str]) -> list[dict]:
        return [self.complete(p) for p in prompts]

    def crash_mid_stream(self, min_chunks: int = 32) -> int:
        """Stream a long request and SIGKILL the server tree while it is
        decoding. Returns the number of chunks received before the kill."""
        received = 0
        errors: list[Exception] = []
        started = threading.Event()

        def stream():
            nonlocal received
            try:
                with requests.post(
                    self.url + "/v1/completions",
                    json={
                        "model": MODEL,
                        "prompt": IN_FLIGHT_PROMPT,
                        "max_tokens": 2000,
                        "temperature": 0,
                        "ignore_eos": True,
                        "stream": True,
                    },
                    stream=True,
                    timeout=120,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line.startswith(b"data: {"):
                            received += 1
                            if received >= min_chunks:
                                started.set()
            except (requests.RequestException, ConnectionError) as e:
                errors.append(e)
            finally:
                started.set()

        thread = threading.Thread(target=stream, daemon=True)
        thread.start()
        assert started.wait(120)
        assert received >= min_chunks, f"stream ended before the crash: {errors}"
        self.proc.kill()
        thread.join(30)
        return received

    def model_load(self) -> tuple[float, float]:
        """(GiB, seconds) the engine spent loading the model."""
        m = re.search(
            r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
            self.proc.log(),
        )
        assert m, "no model loading time in the log"
        return float(m.group(1)), float(m.group(2))


def _restore_counts(log: str) -> tuple[int, int]:
    m = re.search(
        r"KV restore: imported (\d+) cached blocks, quarantined (\d+) of the "
        r"previous writer",
        log,
    )
    assert m, "KV restore did not import the crashed engine's prefix cache"
    return int(m.group(1)), int(m.group(2))


def _texts(results: list[dict]) -> list[str]:
    return [r["text"] for r in results]


@pytest.fixture(scope="module")
def disk_reference(tmp_path_factory):
    """Outputs and time to ready of a plain disk-loading `vllm serve`.

    Started twice: the first run fills the compile caches, so the measured
    second run is comparable with the restarts below.
    """
    log_dir = tmp_path_factory.mktemp("disk")
    procs: list[_Process] = []
    try:
        Server("disk-warmup", log_dir, procs, loader=None).proc.terminate()
        server = Server("disk", log_dir, procs, loader=None)
        results = server.run(PROMPTS)
        load_gib, load_s = server.model_load()
        server.proc.terminate()
    finally:
        for p in procs:
            p.terminate()
    return {
        "ready_s": server.ready_s,
        "load_gib": load_gib,
        "load_s": load_s,
        "results": results,
    }


def _loader(socket_dir: str, **kw) -> dict:
    return {"socket_dir": socket_dir, "fallback": False, "kv_restore": True, **kw}


def test_serve_restart_reuses_daemon_weights_and_kv_cache(
    tmp_path, processes, disk_reference
):
    """SIGKILL `vllm serve` mid-stream; a new `vllm serve` on the same daemon
    loads no weights from disk and serves from the crashed one's prefix
    cache with identical greedy outputs."""
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_") as socket_dir:
        start_daemon(tmp_path, socket_dir, processes)

        writer = Server("writer", tmp_path, processes, _loader(socket_dir))
        before = writer.run(PROMPTS)
        chunks = writer.crash_mid_stream()

        restarted = Server("restarted", tmp_path, processes, _loader(socket_dir))
        after = restarted.run(PROMPTS)
        log = restarted.proc.log()
        imported, quarantined = _restore_counts(log)
        restart_gib, restart_load_s = restarted.model_load()

    ref = disk_reference
    print(
        f"\n[restart] disk serve: ready {ref['ready_s']:.1f}s (model load "
        f"{ref['load_gib']} GiB in {ref['load_s']:.2f}s); daemon restart: ready "
        f"{restarted.ready_s:.1f}s (model load {restart_gib} GiB in "
        f"{restart_load_s:.2f}s); first writer ready "
        f"{writer.ready_s:.1f}s; killed after {chunks} streamed chunks; "
        f"imported {imported} blocks, quarantined {quarantined}; cached tokens "
        f"before {[r['cached_tokens'] for r in before]}, after "
        f"{[r['cached_tokens'] for r in after]} of {after[0]['prompt_tokens']}"
    )
    # (a) The weights are mapped from the daemon: nothing is read from disk
    # and the engine allocates no weight memory of its own.
    assert "tensors from the weight cache daemon" in log
    assert "Loading weights took" not in log
    assert restart_gib == 0 < ref["load_gib"]
    # (b) The first request after the restart hits the restored prefix cache.
    assert before[0]["cached_tokens"] == 0
    assert after[0]["cached_tokens"] > 0.9 * after[0]["prompt_tokens"] - 16
    # (c) Greedy outputs match before the crash and a disk-loaded server.
    assert _texts(before) == _texts(ref["results"])
    assert _texts(after) == _texts(ref["results"])
    # (d) The crashed writer's in-flight blocks were quarantined, not adopted.
    assert imported > 0
    assert quarantined > 0


_STANDBY_DRIVER = """
import asyncio, json, sys, time

import torch

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

cfg = json.loads(sys.argv[1])


def kv_pool_checksum(worker):
    alloc = worker._kv_restore.allocation
    words = alloc.tensor(0, alloc.size).view(torch.int64)
    return [int(words.sum()), int(torch.count_nonzero(words))]


def emit(obj):
    print("DRIVER " + json.dumps(obj), flush=True)


async def main():
    start = time.perf_counter()
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**cfg))
    emit({"ready_s": time.perf_counter() - start})
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        cmd = json.loads(line)
        if cmd["op"] == "checksum":
            emit(await llm.collective_rpc(kv_pool_checksum))
        elif cmd["op"] == "warmup":
            emit(await llm.collective_rpc("kv_restore_warmup", args=(cmd["sizes"],)))
        elif cmd["op"] == "try_generate":
            sp = SamplingParams(max_tokens=1)
            try:
                async for out in llm.generate(cmd["prompt"], sp, request_id="early"):
                    final = out
                emit({"finish_reason": final.outputs[0].finish_reason})
            except Exception as e:
                emit({"finish_reason": "error", "exception": repr(e)})
        elif cmd["op"] == "promote":
            t0 = time.perf_counter()
            adopted = await llm.engine_core.call_utility_async(
                "kv_restore_promote", 60.0, 2.0
            )
            emit({"adopted": adopted, "promote_s": time.perf_counter() - t0})
        elif cmd["op"] == "generate":
            results = []
            for i, prompt in enumerate(cmd["prompts"]):
                sp = SamplingParams(
                    temperature=0, max_tokens=cmd["max_tokens"], ignore_eos=True
                )
                async for out in llm.generate(prompt, sp, request_id=f"r{i}"):
                    final = out
                results.append(
                    {
                        "text": final.outputs[0].text,
                        "cached_tokens": final.num_cached_tokens,
                        "prompt_tokens": len(final.prompt_token_ids),
                    }
                )
            emit(results)
    llm.shutdown()


asyncio.run(main())
"""


class StandbyDriver:
    """A standby engine (AsyncLLM) driven over stdin: `vllm serve` has no
    endpoint for the `kv_restore_promote` engine utility."""

    def __init__(self, log_dir: Path, procs: list, socket_dir: str):
        engine_args = {
            "model": MODEL,
            "load_format": "ipc_cache",
            "model_loader_extra_config": _loader(socket_dir, kv_restore_role="standby"),
            "max_model_len": 4096,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "kv_cache_memory_bytes": 2 << 30,
            "enable_prefix_caching": True,
        }
        self.proc = _Process(
            "standby",
            [sys.executable, "-c", _STANDBY_DRIVER, json.dumps(engine_args)],
            log_dir,
            stdin=subprocess.PIPE,
            # The KV checksum runs a test-defined function in the worker.
            env={**os.environ, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
        )
        procs.append(self.proc)
        self._seen = 0
        self.ready_s = self._reply(READY_TIMEOUT_S)["ready_s"]

    def _reply(self, timeout: float):
        found: list = []

        def ready() -> bool:
            replies = re.findall(r"^DRIVER (.*)$", self.proc.log(), re.M)
            if len(replies) > self._seen:
                found.append(json.loads(replies[self._seen]))
                self._seen += 1
                return True
            return False

        self.proc.wait_for(ready, timeout, "driver reply")
        return found[0]

    def call(self, timeout: float = 300.0, **cmd):
        stdin = self.proc.proc.stdin
        assert stdin is not None
        stdin.write(json.dumps(cmd) + "\n")
        stdin.flush()
        return self._reply(timeout)


def test_serve_standby_promotion_after_writer_crash(
    tmp_path, processes, disk_reference
):
    """A warm standby next to `vllm serve` takes over its weights and KV cache
    when the server is killed: warmup writes no KV, promotion is fast, and the
    promoted engine serves the dead writer's prefix cache with identical
    greedy outputs."""
    with tempfile.TemporaryDirectory(prefix="vllm_ipc_") as socket_dir:
        start_daemon(tmp_path, socket_dir, processes)
        writer = Server("writer", tmp_path, processes, _loader(socket_dir))
        before = writer.run(PROMPTS)

        standby = StandbyDriver(tmp_path, processes, socket_dir)
        # The writer is idle: the shared pool must not change while the
        # standby warms up.
        checksum = standby.call(op="checksum")
        assert standby.call(op="warmup", sizes=[1, 8, 64, 512]) == [4]
        assert standby.call(op="checksum") == checksum
        # Before promotion the standby must not serve: its requests would
        # write blocks of the writer's live KV cache.
        early = standby.call(op="try_generate", prompt=PROMPTS[0])
        assert early["finish_reason"] == "error", early
        assert standby.call(op="checksum") == checksum
        # The writer keeps serving (and committing) after the standby attached.
        assert _texts(writer.run(PROMPTS[:1])) == _texts(before[:1])

        chunks = writer.crash_mid_stream()
        promoted = standby.call(op="promote")
        after = standby.call(op="generate", prompts=PROMPTS, max_tokens=MAX_TOKENS)
        log = standby.proc.log()
        imported, quarantined = _restore_counts(log)

    print(
        f"\n[standby] standby ready {standby.ready_s:.1f}s next to the writer "
        f"(writer ready {writer.ready_s:.1f}s); killed after {chunks} streamed "
        f"chunks; promote {promoted['promote_s'] * 1000:.1f} ms, adopted "
        f"{promoted['adopted']} blocks, quarantined {quarantined}; cached tokens "
        f"after {[r['cached_tokens'] for r in after]} of "
        f"{after[0]['prompt_tokens']}; KV pool checksum {checksum}"
    )
    # (a) The standby mapped the daemon's weights and the writer's KV pool.
    assert "tensors from the weight cache daemon" in log
    assert "Loading weights took" not in log
    assert "standby mapped the writer's KV cache" in log
    assert promoted["promote_s"] < 5.0
    # (b) The first request after promotion hits the inherited prefix cache.
    assert promoted["adopted"] == imported > 0
    assert after[0]["cached_tokens"] > 0.9 * after[0]["prompt_tokens"] - 16
    # (c) Greedy outputs match the writer before the crash and a disk load.
    assert _texts(before) == _texts(disk_reference["results"])
    assert _texts(after) == _texts(disk_reference["results"])
    # (d) The dead writer's in-flight blocks were quarantined, not adopted.
    assert quarantined > 0
