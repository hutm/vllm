# Preload

vLLM's preload feature keeps model artifacts resident in GPU memory across
engine restarts, so a restarting engine reuses them instead of rebuilding them
from scratch. The design is extensible to different kinds of artifacts; it
currently supports **model weights** via the weight cache daemon.

With weight preloading, a daemon process per GPU holds its rank's
post-quantized, TP-sharded weights and serves CUDA IPC handles to vLLM engines
over a Unix domain socket, so a restarting engine maps the weights via
zero-copy IPC instead of reloading them from disk.

Key benefits:

- **Fast engine restarts**: weight loading reduces to mapping existing GPU
  memory, cutting restart time from minutes (disk load + quantization
  processing) to seconds.
- **Zero-copy sharing**: in the default `zero_copy` mode the engine directly
  shares the daemon's GPU memory, so a restart adds no extra GPU memory
  footprint.
- **Safe fallback**: if the daemon is unavailable or its cached weights don't
  match the engine's configuration, the engine falls back to loading from
  disk.

!!! note
    Weight preloading is only supported on CUDA and ROCm platforms. Tensor,
    expert and data parallelism are supported, including across nodes;
    pipeline parallelism is rejected when launching the daemon.

## Quick start

Launch one weight cache daemon per GPU with a single command:

```bash
vllm preload --model meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4
```

The daemon process loads and post-processes the weights once, then waits for
engines. Each rank binds its Unix socket only after its shard is fully cached,
so engines connecting before that simply fall back to disk loading.

Then start (or restart) engines with the `ipc_cache` load format:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --load-format ipc_cache
```

The daemon itself must load from disk; passing `--load-format ipc_cache` to
`vllm preload` is an error.

## Preloading the FlashInfer autotune cache

FlashInfer has several implementations of each operation and chooses between
them by benchmarking. vLLM runs that pass during kernel warmup and persists the
result per rank in the on-disk FlashInfer autotune cache (under
`VLLM_CACHE_ROOT`, or `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`), so a later engine
with the same configuration loads the tuned configs instead of profiling again.
Normally the first serving engine pays for the pass, and on a large MoE model it
dominates what is left of the startup time once the weights come from the
daemon.

Unless FlashInfer autotune is disabled (`--no-enable-flashinfer-autotune`),
`vllm preload` pays it at preload time: each daemon runs the autotune pass in
place — profiling dummy runs against the shard it already holds — after
loading and before binding its socket, then writes the cache:

```bash
vllm preload --model /path/to/model --tensor-parallel-size 4
```

## How it works

1. `vllm preload` spawns one daemon process per GPU. Each daemon loads its
   shard from disk using the configured loader, runs quantization
   post-processing, and exports every parameter/buffer as a CUDA IPC handle.
2. An engine started with `--load-format ipc_cache` builds its model on the
   meta device (no storage) and requests the tensors from the daemon on its
   GPU.
3. Before serving anything, the engine and daemon compare a fingerprint of the
   cached weights: checkpoint content (hashed from safetensors metadata, so
   identical weights in different directories still match), model
   architecture, TP/DP size/rank, dtype, quantization method and config, model
   revision, and vLLM version. On any mismatch the engine falls back to disk
   loading (unless `fallback` is disabled).
4. The engine also verifies the daemon's GPU UUID matches its own device, so a
   stale socket cannot serve weights for the wrong GPU.

Tied weights (e.g. `lm_head.weight` sharing storage with
`embed_tokens.weight`) are exported once and re-established as aliases in the
engine, preserving parameter identity.

## Multi-node and data parallelism

CUDA IPC handles are node-local, so each node serves only its local GPUs'
shards. For multi-node tensor parallelism, run one `vllm preload` launcher per
node with a shared rendezvous: reuse the `--nnodes` / `--node-rank` /
`--master-addr` flags you pass the engine, plus a `--weight-cache-master-port`
distinct from the engine's `--master-port`:

```bash
# node 0 (8 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 16 \
    --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 \
    --weight-cache-master-port 29600
# node 1 (8 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 16 \
    --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 \
    --weight-cache-master-port 29600
```

For data parallelism (e.g. a TP1 x DP16 x EP decode fleet), run one launcher
per node with the engine's DP placement flags. Local GPU `i` serves DP rank
`start_rank + i // tp_size` and TP rank `i % tp_size`, and all
`dp_size * tp_size` daemons form one world group on `--data-parallel-address`
/ `--weight-cache-master-port` so the expert shards are laid out exactly as in
the engine:

```bash
# node r (4 local GPUs)
vllm preload --model /path/to/model --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 --data-parallel-size-local 4 \
    --data-parallel-start-rank 4r --data-parallel-address 10.0.0.1 \
    --weight-cache-master-port 29600
```

Data parallelism also combines with multi-node tensor parallelism: pass both
flag sets. Each node then serves a contiguous block of the
`dp_size * tp_size` global ranks.

## Speculative decoding

With MTP, EAGLE or EAGLE3 speculative decoding, `vllm preload` additionally
starts a draft daemon group that caches the draft model. It uses its own cache
key, Unix sockets (`*_draft.sock`) and rendezvous port
(`--weight-cache-draft-master-port`, default `--weight-cache-master-port + 1`),
so each daemon process serves exactly one model role. Other draft types are
not cached and keep loading from disk in the engine.

## Cache modes

The loader supports two modes, selected via `--model-loader-extra-config`:

- `zero_copy` (default): the engine maps the daemon's CUDA allocations
  directly. The daemon must stay alive for the engine's lifetime, and each
  GPU carries only one copy of the weights.
- `copy`: the engine clones every tensor into its own GPU memory and then
  asks the daemon to release its cache. Use this when the daemon should free
  GPU memory after handing off, at the cost of a full copy per restart.

!!! warning
    In `zero_copy` mode the weights live in the daemon's CUDA IPC allocations,
    so [sleep mode](sleep_mode.md) (CuMemAllocator weight offloading) must not
    be used with this loader.

## Loader configuration

The `ipc_cache` loader accepts extra keys via `--model-loader-extra-config`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"mode": "copy", "fallback": false}'
```

| Key | Default | Description |
| --- | ------- | ----------- |
| `socket_path` | per-GPU path derived from the GPU UUID | Explicit daemon socket path. |
| `socket_dir` | per-user private dir under the temp dir | Directory containing the daemon sockets. |
| `mode` | `zero_copy` | `zero_copy` or `copy` (see above). |
| `fallback` | `true` | Fall back to disk loading when the daemon is unavailable or the fingerprints mismatch. |
| `connect_timeout_s` | `5.0` | Socket connect timeout in seconds. |
| `state_timeout_s` | `300.0` | Timeout for the weight-transfer request in seconds. |
| `kv_restore` | `false` | Keep the KV cache and prefix cache with the daemon across engine restarts (see below). |

The daemon's `--weight-cache-socket-dir` and the loader's `socket_dir` /
`socket_path` must agree when the default per-user directory is not used.
Socket paths are derived from the GPU UUID, so they are stable regardless of
`CUDA_VISIBLE_DEVICES` index remapping.

## KV cache restore

With `"kv_restore": true` the daemon also keeps the engine's KV cache, and the
prefix cache that indexes it, so a restarted (or crashed) engine serves its
first requests from the prefix cache instead of recomputing them:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --load-format ipc_cache \
    --model-loader-extra-config '{"kv_restore": true}'
```

- Each worker allocates its KV cache as one shareable CUDA VMM allocation and
  hands its file descriptor to its GPU's daemon. The daemon only holds the
  descriptor, which keeps the memory alive after the engine exits.
- The scheduler streams its prefix cache to the daemon: a block is recorded
  only after the step that wrote it has finished, and removed before its
  memory can be reused, so a crash never leaves a half-written block behind.
- A restarted engine with the same KV cache layout maps the same memory and
  imports the recorded blocks into its prefix cache. Its KV cache is sized to
  the retained pool automatically.

One engine at a time owns a pool: while it runs, a second engine on the same
GPU starts without restore. If the layout differs (model, dtype, KV cache
dtype or layout, block size, per-layer shapes), the daemon logs the differing
fields, drops the retained pool and the engine starts with an empty KV cache.

The daemon cannot tell a clean engine exit from a crash, so it keeps the pool
in both cases. The memory is freed when `vllm preload` exits, when an engine
with a different layout attaches, or on request:

```bash
vllm preload --kv-drop [--weight-cache-socket-dir DIR] [--force]
```

`--kv-drop` talks to the running daemons and exits. It refuses to drop a pool
that an engine is using unless `--force` is given; a forced drop only releases
the daemon's reference, so that engine keeps its memory until it exits and
nothing is restored afterwards.

Requirements and limits:

- CUDA only (VMM). Not compatible with sleep mode.
- Prefix caching hashes must be stable across restarts: the default `sha256`
  hash uses a fixed seed; with `xxhash`, set `PYTHONHASHSEED`.
- With tensor parallelism, the prefix cache is imported only if every worker
  re-attached its own pool.
- A retained pool keeps its GPU memory allocated after the engine exits, until
  it is reused or dropped.

## Limitations

- **Platform**: CUDA and ROCm only; other platforms raise
  `UnsupportedPlatformForIPCError` even when `fallback` is enabled, since it
  is a permanent misconfiguration rather than a transient daemon outage.
- **Parallelism**: tensor, expert, pipeline and data parallelism are
  supported, including across nodes.
- **Autotune preloading**: tuning runs across the daemon's world group,
  including across nodes and DP ranks. With model-based speculative decoding
  both daemon groups skip it (the target cannot run the drafter's dummy
  passes, and a table tuned on the draft model would key on the draft's
  config, which no engine looks up), so those deployments' first engine
  tunes and fills the cache instead.
- **Quantization**: every quantization method in the model must declare
  support for pre-processed weights (the daemon transfers weights *after*
  quantization post-processing). Unsupported methods raise
  `UnsupportedQuantForIPCError`.
- **Consistency**: the daemon and engine must run the same vLLM version and
  agree on model, dtype, quantization, and TP/DP layout, otherwise the
  fingerprint mismatch triggers the disk fallback.
- **Localhost only**: the daemon serves over a Unix domain socket; both the
  daemon and the engines must run on the same node as the same user.

## Security

The socket protocol uses pickle and is intended only for trusted local
processes owned by the same user:

- Daemon sockets live in a per-user private directory (mode `0700`) and the
  socket files are restricted to the owner (`0600`).
- Both sides reject symlinked or non-owned socket paths; the auto-derived
  directory is additionally rejected if it is group/world accessible.
- On Linux the daemon verifies the connecting peer's UID via `SO_PEERCRED`.

See the [security documentation](../usage/security.md) for vLLM's general
threat model.

## Daemon lifecycle

- Each GPU's socket path is guarded by an exclusive lock file, so a second
  daemon for the same GPU fails fast instead of clobbering the live socket.
- `vllm preload` reports readiness once every rank is serving; if any rank
  dies during startup, the remaining ranks are terminated and the command
  exits with that rank's exit code.
- `SIGINT`/`SIGTERM` to the `vllm preload` process terminates all daemon
  ranks, which unlinks their sockets and releases the GPU memory.

See [vllm preload](../cli/preload.md) for the full CLI reference.
