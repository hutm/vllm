# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end: `vllm serve` crashes and a restarted server resumes from the
weight cache daemon's weights and KV cache, without any orchestrator.

A plain user runs `vllm preload` (the weight cache daemon, vmm_fd transport)
next to `vllm serve --load-format ipc_cache` with KV restore. The server is
SIGKILLed with a streaming request in flight; a new `vllm serve` attaches to
the same daemon. Its weights come from the daemon (no disk load), its very
first request hits the prefix cache the crashed server built, and its greedy
outputs match the pre-crash server and a plain disk-loaded server.
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
# Tokens the crashed server streams before it is killed; afterwards the test
# resends the prompt plus that much of its output.
STREAMED_CHUNKS = 32

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

    def complete(self, prompt: str, max_tokens: int = MAX_TOKENS) -> dict:
        r = requests.post(
            self.url + "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
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

    def crash_mid_stream(self, min_chunks: int = STREAMED_CHUNKS) -> list[str]:
        """Stream a long request and SIGKILL the server tree while it is
        decoding. Returns the text chunks received before the kill."""
        chunks: list[str] = []
        errors: list[Exception] = []
        started = threading.Event()

        def stream():
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
                            body = json.loads(line[len(b"data: ") :])
                            chunks.append(body["choices"][0]["text"])
                            if len(chunks) >= min_chunks:
                                started.set()
            except (requests.RequestException, ConnectionError) as e:
                errors.append(e)
            finally:
                started.set()

        thread = threading.Thread(target=stream, daemon=True)
        thread.start()
        assert started.wait(120)
        assert len(chunks) >= min_chunks, f"stream ended before the crash: {errors}"
        self.proc.kill()
        thread.join(30)
        return chunks

    def model_load(self) -> tuple[float, float]:
        """(GiB, seconds) the engine spent loading the model."""
        m = re.search(
            r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds",
            self.proc.log(),
        )
        assert m, "no model loading time in the log"
        return float(m.group(1)), float(m.group(2))


def _imported_blocks(log: str) -> int:
    m = re.search(r"KV restore: imported (\d+) cached blocks", log)
    assert m, "KV restore did not import the crashed engine's prefix cache"
    return int(m.group(1))


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
        in_flight = server.complete(IN_FLIGHT_PROMPT, max_tokens=STREAMED_CHUNKS)
        resend = server.complete(IN_FLIGHT_PROMPT + in_flight["text"])
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
        "in_flight": in_flight,
        "resend": resend,
    }


def _loader(socket_dir: str, **kw) -> dict:
    return {"socket_dir": socket_dir, "fallback": False, "kv_restore": True, **kw}


def _drop_retained_kv(socket_dir: str) -> str:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "preload",
        "--kv-drop",
        "--weight-cache-socket-dir",
        socket_dir,
    ]
    for _ in range(30):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        # The daemon may not have seen the exited engine's sockets close yet.
        if "busy" not in result.stdout:
            break
        time.sleep(1)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


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
        streamed = "".join(chunks[:STREAMED_CHUNKS])

        restarted = Server("restarted", tmp_path, processes, _loader(socket_dir))
        after = restarted.run(PROMPTS)
        # The killed request's prompt and committed decode blocks may be
        # reused; the block it was writing when killed must not be.
        resend = restarted.complete(IN_FLIGHT_PROMPT + streamed)
        log = restarted.proc.log()
        imported = _imported_blocks(log)
        restart_gib, restart_load_s = restarted.model_load()

        # A clean exit keeps the pool for the next engine; dropping frees it.
        restarted.proc.terminate()
        dropped = _drop_retained_kv(socket_dir)
        dropped_again = _drop_retained_kv(socket_dir)

    ref = disk_reference
    print(
        f"\n[restart] disk serve: ready {ref['ready_s']:.1f}s (model load "
        f"{ref['load_gib']} GiB in {ref['load_s']:.2f}s); daemon restart: ready "
        f"{restarted.ready_s:.1f}s (model load {restart_gib} GiB in "
        f"{restart_load_s:.2f}s); first writer ready "
        f"{writer.ready_s:.1f}s; killed after {len(chunks)} streamed chunks; "
        f"imported {imported} blocks; cached tokens "
        f"before {[r['cached_tokens'] for r in before]}, after "
        f"{[r['cached_tokens'] for r in after]} of {after[0]['prompt_tokens']}; "
        f"resent killed request: {resend['cached_tokens']} of "
        f"{resend['prompt_tokens']} cached; "
        f"drop after a clean exit: {dropped.strip()}"
    )
    # (a) The weights are mapped from the daemon: nothing is read from disk
    # and the engine allocates no weight memory of its own.
    assert "tensors from the weight cache daemon" in log
    assert "Loading weights took" not in log
    assert restart_gib == 0 < ref["load_gib"]
    # (b) The first request after the restart hits the restored prefix cache.
    assert imported > 0
    assert before[0]["cached_tokens"] == 0
    assert after[0]["cached_tokens"] > 0.9 * after[0]["prompt_tokens"] - 16
    # (c) Greedy outputs match before the crash and a disk-loaded server.
    assert _texts(before) == _texts(ref["results"])
    assert _texts(after) == _texts(ref["results"])
    # (d) Resending the killed request: its prompt blocks are reused, and
    # every reused block holds the right KV (the output matches a disk load).
    assert streamed == ref["in_flight"]["text"]
    prompt_blocks = ref["in_flight"]["prompt_tokens"] // 16 * 16
    assert prompt_blocks <= resend["cached_tokens"] < resend["prompt_tokens"]
    assert resend["text"] == ref["resend"]["text"]
    # (e) The pool outlived the clean exit until it was dropped.
    assert re.search(r"freed (?!0\.00)[\d.]+ GiB", dropped), dropped
    assert "freed 0.00 GiB" in dropped_again, dropped_again
