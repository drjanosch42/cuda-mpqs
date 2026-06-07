# CLUSTER.md -- Cluster Sieve Usage Guide

Multi-node distributed sieve for CUDA-MPQS. Multiple GPU nodes cooperatively sieve smooth relations over a LAN; the coordinator runs the remaining pipeline stages (matrix, linear algebra, square root) locally.

For architecture details see [docs/modules/cluster.md](docs/modules/cluster.md).

---

## Quick Start

**Minimal 2-node example** (localhost coordinator + remote worker):

```bash
# Terminal 1 -- Coordinator (RTX 5070 Ti)
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode coordinator --listen_port 9300 --expected_workers 1 \
    --fb_bound 7000000 --sieve_bound 262144 --lp1_bound 2000000000000 \
    --sieve_batch_size 32 --cuda_graph_unroll 4 --lp_interval 1 \
    --lp1_hash_bits 21 --lp1_max_witnesses 8388608 --verbose

# Terminal 2 -- Worker (retries automatically until coordinator is up)
ssh ${USER}@conceptd "cd ~/dev/cuda-mpqs && ./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode worker --coordinator_host <coordinator-ip> --coordinator_port 9300 \
    --sieve_batch_size 2 --verbose"
```

**Note:** Workers retry connecting automatically (every 5s, up to 300s default). Launch order doesn't matter.

---

## CLI Parameters

### Cluster Mode Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cluster_mode` | `coordinator` or `worker` | (solo) | Enable cluster mode. Omit for solo operation. |

### Coordinator Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--listen_port` | `uint16_t` | 9100 | TCP port to listen on |
| `--expected_workers` | `uint32_t` | 0 | Number of remote workers to wait for (0 = coordinator-only with Thread B local sieve) |

### Worker Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--coordinator_host` | `string` | -- | Coordinator IP or hostname |
| `--coordinator_port` | `uint16_t` | 9100 | Coordinator TCP port |

### Shared Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cluster_init_timeout` | `uint32_t` | 300 | Initialization window in seconds: workers retry connecting within this window; coordinator waits this long for workers. |
| `--cluster_node_weights` | `string` | (auto) | Per-node throughput weights as comma-separated floats (e.g. `"2.5,1.0"`). Overrides SM×clock auto-detection for initial range split. |
| `--cluster_headroom` | `double` | 10.0 | Per-node headroom percentage. Extends each node's initial contiguous range by this fraction to reduce straggler wait at range boundaries. |
| `--dedup_safety_factor` | `double` | 1.05 | Dedup oversample margin. Coordinator collects `target × factor` relations before stopping to account for dedup losses. Auto-set to 1.35 for <80d inputs. |
| `--probe_timeout` | `double` | 120.0 | Hard timeout (seconds) for `--estimate_only` probes. In cluster mode, the coordinator runs the full cluster sieve for this duration, then broadcasts STOP and prints the runtime estimate. Increase for slow GPUs (e.g., 600 for Jetson at RSA-120 scale). |

The transport is fixed to TCP: there is **no** `--transport` CLI flag. The `MPQSConfig::transport` field defaults to `"tcp"` and TCP is the only backend.

All standard flags (`--fb_bound`, `--sieve_bound`, `--lp1_bound`, `--sieve_batch_size`, `--cuda_graph_unroll`, etc.) work identically to solo mode and can be set independently per node. Workers receive N, factor base, and polynomial parameters from the coordinator via `WORK_ASSIGN`.

---

## Runtime Estimation (`--estimate_only`)

Cluster mode supports `--estimate_only` for parameter sweeps. The coordinator runs the full cluster topology (Thread A + Thread B + all workers) for `--probe_timeout` seconds, then broadcasts STOP and prints a runtime estimate.

```bash
# 2-node cluster estimate (30s probe)
# Terminal 1 -- Coordinator
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode coordinator --listen_port 9300 --expected_workers 1 \
    --fb_bound 7000000 --sieve_bound 262144 --lp1_bound 2000000000000 \
    --sieve_batch_size 32 --lp_interval 1 --lp1_hash_bits 21 \
    --estimate_only --probe_timeout 30 --verbose

# Terminal 2 -- Worker
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode worker --coordinator_host <COORD_IP> --coordinator_port 9300 \
    --sieve_batch_size 2 --verbose
```

Output includes `[Estimate]` lines: node count, F/M/L, FB size, projected sieve/matrix/linalg/total time, throughput, and probe duration.

**Tip:** For Jetson at RSA-120+ scale, a single sieve batch can take >100s. Use `--probe_timeout 600` to allow enough batches for a meaningful estimate.

---

## Cluster Telemetry

The coordinator logs aggregate telemetry every ~5 seconds during **all** cluster sieve operations (full pipeline, sieve_only, and estimate_only):

```
[Cluster] Progress: 15735 / 119252 (13.2%) | 1034.6 rel/s
[Cluster] ETA: 95.6s | Total est.: 00:01:51
[Cluster] LP: 22690 witnesses (1467.3/s) | 596 combines (38.5/s) | yield 2.6%
```

At sieve completion, an aggregate summary is printed:
```
[Thread A] Aggregate: 119846 rels in 104.9s (1142.1 rel/s)
[Thread A] LP yield: 12.3% (17746 combines / 143742 inserts)
```

This telemetry is unconditional — it runs during every cluster sieve, not just `--estimate_only` probes.

---

## Setup Requirements

1. **Build on all nodes.** Each node must have a built binary at `~/dev/cuda-mpqs/build/tests/cuda-mpqs`.
   - RTX / x86: `cmake -B build -DGPU_TARGET=native && cmake --build build -j16`
   - Jetson: `cmake -B build -DGPU_TARGET=Orin && cmake --build build -j4`

2. **Sync code via git.** Push from the development node, pull on all other nodes:
   ```bash
   git push origin main
   ssh ${USER}@conceptd "cd ~/dev/cuda-mpqs && git pull && cmake --build build -j16"
   ssh ${USER}@<jetson-host-1> "cd ~/dev/cuda-mpqs && git pull && cmake --build build -j4"
   ```

3. **SSH access.** All nodes must be reachable via `ssh ${USER}@<hostname>` without interactive password prompt (key-based auth).

4. **Network.** Nodes must be on the same LAN. The coordinator's LAN IP must be routable from all workers. Network is not a bottleneck: RSA-100 transfers ~59 MB total relation data across the entire run.

---

## Launch Order and Timing

1. **All nodes can be launched simultaneously.** Workers retry connecting every 5s within the `--cluster_init_timeout` window (default 300s). No `sleep` between launches is needed.
2. **Coordinator accept window.** The coordinator waits up to `--cluster_init_timeout` seconds for all `expected_workers` to connect. If the window expires, it proceeds with those that connected.
3. **Order is irrelevant.** A worker that starts before the coordinator simply retries until the coordinator's TCP listener is ready. See [Headless Launch Pattern](#headless-launch-pattern) for the validated simultaneous-launch pattern.

---

## Timeouts

| Constant | Value | Description |
|----------|-------|-------------|
| `kHeartbeatIntervalMs` | 5,000 ms | Workers send heartbeats every 5s (background thread, independent of sieve loop). |
| `kFlushTimeoutMs` | 120,000 ms | Maximum time between heartbeats before coordinator considers a worker dead. Set high for Jetson CUDA graph capture (~30s/replay). |
| Init timeout (`--cluster_init_timeout`) | 300,000 ms (300s) | How long coordinator waits for workers / workers retry connecting. Configurable via CLI. |
| Barrier timeout | 30,000 ms | How long coordinator waits for `FLUSH_ACK` after sending `STOP`. |

**Common timeout causes:**
- CUDA graph capture + JIT compilation on cold cache (Jetson: 20+ minutes with cold JIT)
- Large `batch_size` on slow GPUs (`batch_size=32` on 2080 Super MaxQ causes timeout)
- Network issues or firewall blocking the coordinator port

---

## Platform-Specific Parameters

### RTX 5070 Ti (Coordinator)

```
--sieve_batch_size 32 --cuda_graph_unroll 4
--sieve_bound 262144 --fb_bound 7000000
```

### RTX 2080 Super MaxQ (Worker)

```
--sieve_batch_size 2
```

**Critical:** `batch_size=32` causes heartbeat timeout on this GPU. Use `batch_size=2`. `cuda_graph_unroll=2` is safe.

### Jetson Orin Nano Super (Coordinator or Worker)

```
--sieve_batch_size 8 --sieve_bound 131072
```

**Hard constraint:** `M <= 131072` for RSA-100+ on Jetson. `M=262144` causes sieve kernel failures on SM 8.7.

For Jetson workers with cold JIT cache, use `--cuda_graph_unroll 0` to skip graph capture (avoids 20+ minute compilation). The coordinator can still use graph unrolling if it has a warm cache.

### LP Parameters (RSA-100+)

```
--lp1_bound 2000000000000 --lp_interval 1 --lp1_hash_bits 21 --lp1_max_witnesses 8388608
```

**Critical:** `--lp1_hash_bits 21` is required for LP bounds >= 1T. The auto-derived value (hash_bits=18) causes tag collisions and zero LP contribution.

### RSA-110 Parameters

```
--fb_bound 9000000 --sieve_bound 131072 --lp1_bound 20000000000000
--lp1_hash_bits 21 --lp1_max_witnesses 4194304
```

---

## Launch Scripts

All scripts are in `tools/cluster/`. They handle coordinator-first ordering, checkpoint cleanup, and per-node parameter tuning. The directory covers three configurations: RTX 2-node, Jetson 2-node, and 4-node heterogeneous.

**RTX 2-node** (localhost RTX 5070 Ti coordinator + conceptd RTX 2080 Super MaxQ worker):

| Script | Input | Description |
|--------|-------|-------------|
| `rtx_cluster_default_launch.sh` | Default (~80d) | Quick validation. RTX coordinator (bs=32), 2080 worker (bs=2). |
| `rtx_cluster_launch.sh` | RSA-100 | 2-node RTX cluster with full LP and CUDA graph. |
| `rtx_cluster_rsa110_launch.sh` | RSA-110 | 2-node RTX RSA-110 with F=9M, L=20T, hash_bits=21. |

**Jetson 2-node** (<jetson-host-1> coordinator + <jetson-host-2> worker):

| Script | Input | Description |
|--------|-------|-------------|
| `jetson_cluster_launch.sh` | RSA-100 | 2-node Jetson cluster. Worker uses `cuda_graph_unroll=0`. |
| `jetson_90d_cluster_launch.sh` | 90-digit | 2-node Jetson 90d run with M6 matrix preprocessing. |
| `jetson_rsa100_launch.sh` | RSA-100 | 2-node Jetson RSA-100 baseline. |
| `jetson_rsa100_m6v2_launch.sh` | RSA-100 | 2-node Jetson RSA-100 with M6v2 matrix preprocessing. |
| `jetson_rsa110_m6v3_launch.sh` | RSA-110 | 2-node Jetson RSA-110 with M6v3 matrix preprocessing. |
| `jetson_rsa110_overnight_launch.sh` | RSA-110 | 2-node Jetson RSA-110 overnight run. Uses `nohup` for SSH resilience. |
| `jetson_rsa110_overnight_launch_v2.sh` | RSA-110 | Variant: both nodes use CUDA graph capture. |
| `jetson_rsa110_overnight_launch_v3.sh` | RSA-110 | Variant: 200% oversieving + disk I/O. |
| `jetson_rsa110_overnight_launch_v4.sh` | RSA-110 | Variant: 100% oversieving + disk I/O + character columns. |
| `jetson_rsa110_overnight_v5_launch.sh` | RSA-110 | Variant: F=9M, M=131072, L=20T. |
| `rsa110_jetson_2node_2026_04_27.sh` | RSA-110 | Dated reproduction script (M12-S5b). |
| `rsa110_jetson_2node_2026_04_28.sh` | RSA-110 | Dated reproduction script (post-M12). |

**4-node heterogeneous** (localhost + conceptd + <jetson-host-1> + <jetson-host-2>):

| Script | Input | Description |
|--------|-------|-------------|
| `rtx_4node_rsa100_launch.sh` | RSA-100 | 4-node heterogeneous cluster (3 workers). |
| `rtx_4node_rsa110_launch.sh` | RSA-110 | 4-node RSA-110 with L=20T, hash_bits=21. |

**Usage:**
```bash
bash tools/cluster/rtx_cluster_launch.sh [extra_coordinator_flags]
```

**Logs** are written to `/tmp/` on the machine running the script:
- `/tmp/rtx_cluster_coord.log` -- coordinator output
- `/tmp/rtx_cluster_worker.log` -- worker output

---

## Headless Launch Pattern

For overnight or remote runs where the terminal cannot stay open, processes must be fully detached from the SSH session.

### SSH Detach Pattern

```bash
ssh user@host "cd /path && nohup binary args > /tmp/log 2>&1 < /dev/null &" < /dev/null &
wait  # after all parallel SSH launches
```

Each component serves a distinct purpose:

| Component | Location | Purpose |
|-----------|----------|---------|
| `nohup` | remote | Survives SSH hangup; ignores SIGHUP |
| `< /dev/null` (remote) | remote | Closes remote stdin; prevents read blocking |
| `&` (remote) | remote | Backgrounds the process on the remote shell |
| `< /dev/null` (local) | local SSH client | Severs pipe FD inheritance; prevents local SSH from blocking on stdin |
| `&` (local) | local shell | Backgrounds the local SSH command |
| `wait` | local shell | Waits for all parallel SSH launches to complete the handshake before the script exits |

**Why `disown` does not work:** `disown` is a bash job-control builtin. SSH sessions run in non-interactive mode (no job control), so `disown` has no effect. Use the `nohup ... &` pattern with the local `< /dev/null &` redirect instead.

### Pipe Settings for Log Tailing

When stdout is not a TTY (as under nohup), the C library switches to block-buffered output. Log lines accumulate in a 4–8 KB buffer and only flush periodically, making `tail -f` appear to stall.

Fix: prefix with `stdbuf -oL -eL` to force line-buffered stdout and stderr:

```bash
nohup stdbuf -oL -eL ./binary args > /tmp/log 2>&1 < /dev/null &
```

| Mode | Pattern | Use case |
|------|---------|---------|
| RTX foreground scripts | `stdbuf -oL -eL` + foreground wait | Coordinator output visible in terminal; real-time `tail -f` tailing works |
| Jetson overnight scripts | plain `nohup` (no stdbuf) | Fully headless; log tailing is post-hoc via `scp` |

### Initialization Window

Both coordinator and workers share a single initialization window controlled by `--cluster_init_timeout` (default 300s):

- **Coordinator:** waits up to N seconds for all `expected_workers` to connect, then proceeds with those that did
- **Workers:** retry connecting every 5s within the window; no manual timing needed

All nodes can therefore be launched simultaneously:

```bash
ssh coordinator "cd /path && nohup binary --cluster_mode coordinator ... &" < /dev/null &
ssh worker1     "cd /path && nohup binary --cluster_mode worker ...     &" < /dev/null &
ssh worker2     "cd /path && nohup binary --cluster_mode worker ...     &" < /dev/null &
wait
```

No `sleep` between launches is required. Workers that start before the coordinator simply retry until the coordinator's TCP listener is ready.

### Pre-flight Checklist

All launch scripts run a standard pre-flight sequence before launching:

1. **Process conflict check** — `pgrep -x cuda-mpqs`: aborts if a prior run is still alive
2. **GPU availability** — `nvidia-smi --query-gpu=name,memory.free`: confirms GPU is accessible
3. **Code sync** — `git pull --ff-only`: ensures all nodes run the same binary
4. **Build** — `cmake --build build -jN`: compiles any changed files

### cluster_watch Integration

Scripts auto-start `cluster_watch` in `--text-continuous` mode after launching all nodes:

```bash
cd /home/${USER}/dev/cluster-watch && python3 -m cluster_watch \
    --text-continuous --interval 5 \
    --nodes <node1>,<node2>,... \
    --telemetry /tmp/cluster-watch-telemetry \
    >> /tmp/cluster-watch-<script>-<timestamp>.log 2>&1 &
```

- Log: `/tmp/cluster-watch-<script>-<timestamp>.log`
- Telemetry JSONL: `/tmp/cluster-watch-telemetry/`
- `--nodes` is set to exactly the nodes launched by the script
- `cluster_watch` runs in the background; the script exits immediately after printing monitor commands

---

## Monitoring

The `cluster_watch` tool (at `~/dev/cluster-watch/`) provides real-time monitoring:
- Relation progress across all nodes
- Per-worker heartbeat status and throughput
- LP matching rates on the coordinator
- ETA based on global relation accumulation rate

Alternatively, tail the log files directly:
```bash
tail -f /tmp/rtx_cluster_coord.log
```

---

## Troubleshooting

### Heartbeat timeout during CUDA graph capture

**Symptom:** Worker disconnected after 120s silence.
**Cause:** CUDA graph capture + JIT compilation blocks the sieve thread. The background heartbeat thread should prevent this, but if graph compilation exceeds `kFlushTimeoutMs` (120s), the worker is declared dead.
**Fix:** Use `--cuda_graph_unroll 0` on workers with cold JIT cache. Or run a solo warmup to populate the JIT cache first.

### JIT cold-cache on workers

**Symptom:** Worker takes 20+ minutes before first sieve batch.
**Cause:** NVIDIA JIT compiler compiling PTX for the worker's GPU architecture for the first time.
**Fix:** Run the worker binary once in solo mode (e.g. with a quick default factorization) to warm the cache, or use `--cuda_graph_unroll 0`.

### Low GPU utilization on workers

**Symptom:** Worker GPU utilization is 30-40% instead of 90%+.
**Cause (M3 and earlier):** `NetworkDataTap::onBatchComplete()` serialized synchronously on the sieve thread with small `batch_size`.
**Fix (M5):** `AsyncNetworkDataTap` uses an SPSC ring buffer + dedicated I/O thread. `onBatchComplete()` is now <50us. If utilization is still low, increase `--sieve_batch_size` to amortize per-batch overhead.

### LP slab overflow (zero LP contribution)

**Symptom:** LP witness count stays at 0 or near-0 despite `lp1_bound > 0`.
**Cause:** Auto-derived `hash_bits` is too small for large LP bounds, causing hash tag collisions.
**Fix:** Add `--lp1_hash_bits 21` explicitly.

### Connection refused on worker start

**Symptom:** Worker logs `Connection attempt N failed` at DEBUG_1, eventually connects.
**Cause:** Coordinator not started yet, or hasn't reached the TCP listen phase.
**Behavior:** Workers automatically retry every 5s within the init timeout window (default 300s). No manual timing required.
**Override:** `--cluster_init_timeout <SEC>` to change the retry window (applies to both coordinator accept and worker retries).

### Sieve kernel failure on Jetson with M=262144

**Symptom:** CUDA error in sieve kernel launch.
**Cause:** SM 8.7 with 2 MB L2 cannot handle `M=262144` at RSA-100+ factor base sizes.
**Fix:** Use `--sieve_bound 131072` on all Jetson nodes.

### "Address already in use" on coordinator

**Symptom:** Coordinator fails to bind to port.
**Cause:** Previous run's socket in TIME_WAIT state, or another process using the port.
**Fix:** Wait 30s, use a different `--listen_port`, or check with `ss -tlnp | grep <port>`.

---

## Hardware Inventory

| Node | GPU | SM | SMs | VRAM | LAN IP | Build Target |
|------|-----|----|-----|------|--------|-------------|
| localhost | RTX 5070 Ti | 12.0 | 70 | 16 GB GDDR7 | <coordinator-ip> | `native` |
| conceptd | RTX 2080 Super MaxQ | 7.5 | ~48 | 8 GB GDDR6 | -- | `Turing` |
| <jetson-host-1> | Jetson Orin Nano Super | 8.7 | 8 | 8 GB unified | <worker-ip> | `Orin` |
| <jetson-host-2> | Jetson Orin Nano Super | 8.7 | 8 | 8 GB unified | -- | `Orin` |

Access: `ssh ${USER}@<hostname>`, repo at `~/dev/cuda-mpqs` on all nodes.

Validated per-node optimal parameters and performance data are maintained
alongside the cluster launch scripts under `tools/cluster/`.

---

## Validated Results

### 2-Node RTX RSA-100 (localhost + conceptd)

**106.90s** core time (M5, vs 112.64s at M3). RTX coordinator with `batch_size=32`, 2080 Super MaxQ worker with `batch_size=2`.

### 2-Node RTX RSA-110 (localhost + conceptd)

**1009.65s** core time (M5). RTX coordinator + 2080 Super MaxQ worker. F=9M, L=20T, hash_bits=21.

### 2-Node Jetson RSA-100 (jetson-01 + jetson-02)

**1692.97s (28m 13s)** core time (M5, 22% faster than M3 baseline of 2182s).

### Solo Regression

All solo-mode benchmarks GREEN (zero regression from cluster code paths).

---

## Architecture Overview

```
Coordinator Node                    Worker Node(s)
+----------------------------------+  +------------------+
| Thread A (CPU)  | Thread B (GPU) |  | Orchestrator     |
|                 |                |  | SieveStage()     |
| Network I/O     | SieveStage()   |  |                  |
| CPU LP table    | [same code     |  | AsyncNetworkDataTap |
| Accumulator     |  as solo mode] |  |   SPSC ring      |
| Dedup           |                |  |   I/O thread:    |
| Telemetry       |                |  |   TCP send,      |
|                 |                |  |   heartbeat,     |
|                 |                |  |   STOP poll      |
| Target check    | DirectChannel -+->|                  |
|                 |                |  +------------------+
| <- TCP recv from workers         |
| -> STOP broadcast when done      |
+----------------------------------+
         |
         v
  Matrix -> LinAlg -> Sqrt (coordinator only)
```

The single sieve loop in `MPQSOrchestrator::SieveStage()` serves all three modes (solo, coordinator Thread B, worker). The `DataTap*` callback is `nullptr` in solo mode (zero overhead), `DirectChannel` for the coordinator's local GPU sieve, and `NetworkDataTap` for remote workers.

For full architectural details and the design overview, see
[docs/modules/cluster.md](docs/modules/cluster.md).

---

## Wire Protocol Summary

Frame format: `[magic:2B "MQ"][msg_type:1B][seq_no:4B][payload_len:4B][payload][CRC32:4B]`

Key message types:

| Message | Code | Direction | Purpose |
|---------|------|-----------|---------|
| HELLO / HELLO_ACK | 0x01/0x02 | W <-> C | Registration and GPU info exchange |
| WORK_ASSIGN | 0x10 | C -> W | N, factor base, sieve params, poly range, AFactorsSnapshot |
| CHUNK_ASSIGN / CHUNK_COMPLETE | 0x13/0x14 | Both | Dynamic work distribution (chunks of a-values) |
| INCREMENTAL_BATCH | 0x22 | W -> C | Combined full + partial relations (DataTap) |
| HEARTBEAT | 0x30 | W -> C | Alive signal + batch count |
| STOP / FLUSH_ACK | 0xF0/0xF1 | Both | Graceful shutdown sequence |

All multi-byte fields are little-endian. Protocol version: 1. Full protocol specification in `src/cluster/cluster_common.h`.
