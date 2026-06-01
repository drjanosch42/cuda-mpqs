# Cluster Module (`src/cluster/`)

## Overview

Distributed sieve stage for the CUDA-MPQS pipeline. Enables multiple GPU nodes to cooperatively sieve smooth relations over a LAN, aggregating results on a single coordinator node that drives the remaining pipeline stages (matrix, linear algebra, square root).

**Design philosophy:** The orchestrator's `SieveStage()` is the sole sieve implementation for solo, coordinator, and worker modes. Cluster mode hooks into the sieve loop via a **DataTap** callback interface injected at two points: `onBatchComplete()` after each postprocessor batch, and `shouldStop()` in the loop condition. In solo mode the DataTap pointer is `nullptr` (zero overhead). This eliminates the duplicate sieve loops that plagued the v1 cluster implementation.

**Architecture:** The coordinator runs two threads. Thread B executes the standard `SieveStage()` with a `DirectChannel` DataTap that pushes extracted relations to Thread A via a bounded circular buffer. Thread A runs a CPU-only network loop: receives incremental batches from remote workers via `CommBackend`, performs CPU-side LP matching, deduplicates, and tracks global progress. When the relation target is reached, Thread A signals STOP to all workers and Thread B.

Static library `mpqs_cluster`. Separable CUDA compilation ON. Namespace: `mpqs::cluster`.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `data_tap.h` | 29 | Abstract `DataTap` callback interface: `onBatchComplete()`, `shouldStop()` |
| `direct_channel.h` | 120 | `DirectChannel`: mutex-guarded SPSC circular buffer, implements `DataTap` for coordinator Thread B -> Thread A |
| `spsc_ring.h` | ~80 | `SPSCRing<T, N>`: lock-free single-producer single-consumer ring buffer template |
| `async_network_data_tap.h` | ~120 | `AsyncNetworkDataTap`: SPSC ring DataTap for workers; `onBatchComplete()` <50us (memcpy into ring slot). Explicit `shutdown()`. |
| `async_network_data_tap.cpp` | ~200 | I/O thread: serialization, batch coalescing (`mergeRelationBatches()`), TCP send, heartbeat, STOP polling |
| `network_data_tap.h` | 10 | **Deprecated stub.** Superseded by `async_network_data_tap.h`; retained for git history only, contains no active code. |
| `accumulator.h` | 244 | `AccumulatorQueue` (MPSC thread-safe queue), `RelationAccumulator` (single-thread dedup + counting), `FinalBatchHandoff` (blocking condition-variable handoff) |
| `cpu_lp.h` | 73 | `CPULargePrimeTable`: CPU hash table for single large prime matching in cluster mode |
| `cpu_lp.cu` | 140 | `CPULargePrimeTable` implementation: insert-and-match, Montgomery-based partial combination, sorted factor merge |
| `cluster_common.h` | 177 | Wire protocol: `FrameHeader`, `MsgType` enum (16+ types), payload structs, protocol constants |
| `comm_backend.h` | 113 | Abstract `CommBackend` interface: lifecycle, point-to-point, collective, info. Factory `createCommBackend()` |
| `tcp_transport.h` | 63 | `TcpSocket` RAII wrapper: listen/accept/connect, length-prefixed framing + CRC32 |
| `tcp_transport.cpp` | 289 | POSIX TCP implementation: CRC32 table-driven, buffered recv, `sendExact` with EAGAIN retry |
| `tcp_backend.h` | 66 | `TCPBackend`: epoll-based coordinator, single-socket worker |
| `tcp_backend.cpp` | 408 | TCP backend: HELLO/HELLO_ACK handshake, epoll multiplexing, barrier, peer management |
| `serialization.h` | 85 | Binary serialization for `HostRelationBatch`, `WORK_ASSIGN`, and `INCREMENTAL_BATCH` |
| `serialization.cpp` | 300 | Serialization implementation: bounds-checked `SafeReader`, CSR-aware batch encoding, backward-compatible M3 snapshot extension |
| `work_pool.h` | 112 | `WorkPool`: thread-safe polynomial work-unit pool with tracked checkout, reclaim, and cursor restore |
| `work_pool.cpp` | 130 | WorkPool implementation: LIFO reclaim queue, linear cursor fallback, per-worker in-flight tracking |
| `chunk_scheduler.h` | 117 | `ChunkScheduler`: EMA throughput tracker, adaptive chunk sizing, contiguous range computation |
| `chunk_scheduler.cpp` | 165 | Scheduler implementation: SM-proportional initial split, quantum/hypercube alignment, confidence ramp |
| **Total** | **~3100** | |

## Architecture

### Execution Topology

```
Solo Mode:      Orchestrator::SieveStage()  [data_tap_ = nullptr]
                  |
                  v
                GPU sieve -> postprocess -> accumulate -> dedup -> matrix -> ...

Coordinator:    Thread B                        Thread A
                Orchestrator::SieveStage()      networkLoop()
                  [data_tap_ = DirectChannel]     |
                  |                               +-- recv(CommBackend) from workers
                  +-- onBatchComplete() --->       +-- CPULargePrimeTable::insertAndMatch()
                  |   [DirectChannel push]         +-- RelationAccumulator::addRelations()
                  +-- shouldStop() <---            +-- signalStop() when target reached
                      [DirectChannel poll]         |
                                                   +-- broadcast(STOP) to workers

Worker:         Orchestrator::SieveStage()
                  [data_tap_ = AsyncNetworkDataTap]
                  |
                  +-- onBatchComplete() ---> memcpy into SPSCRing slot (<50us)
                  |                          I/O thread: coalesce, serialize, TCP send
                  +-- shouldStop() <--- I/O thread polls STOP/ERROR from CommBackend
```

### DataTap Injection

The `DataTap*` pointer is set in `MPQSConfig::data_tap` before `SieveStage()` entry. The sieve loop checks `data_tap_ != nullptr` to gate extraction:

1. **Batch path:** After `processBatchBufferedCandidates()`, calls `data_tap_->onBatchComplete()` with the incremental snapshot.
2. **Legacy path:** After `consolidateToPersistent()`, calls `data_tap_->onBatchComplete()`.
3. **Loop condition:** Both paths add `&& !data_tap_->shouldStop()` alongside existing truncation checks.

Async data extraction uses double-buffered host staging on a dedicated `extract_stream`, adding 0.38% overhead at RSA-100 scale.

### Coordinator Threading Model

| Thread | Role | CUDA Context | Key Components |
|--------|------|-------------|----------------|
| Thread B | GPU sieve via standard `SieveStage()` | Yes | `DirectChannel` (DataTap), siever, postprocessor |
| Thread A | CPU network loop, LP matching, dedup | No | `CommBackend`, `AccumulatorQueue`, `RelationAccumulator`, `CPULargePrimeTable`, `WorkPool`, `ChunkScheduler` |

Thread B pushes to the `AccumulatorQueue` via `DirectChannel` -> Thread A's drain loop. Remote workers push to the same queue via `CommBackend` -> deserialization -> `pushRemoteRelations()`. Thread A is the sole consumer.

**Worker CUDA graph warmup:** Workers execute one warmup `SieveStage()` call (1 batch, `cuda_graph_unroll=0`, null DataTap) immediately after receiving `WORK_ASSIGN`, before the first real chunk. This populates the JIT PTX cache so the first real graph capture is fast. JIT cache is persisted across runs via `CUDA_CACHE_PATH`.

## Key Data Structures

### DataTap

Abstract callback interface (`data_tap.h`). Injected into `SieveStage()`.

| Method | Description |
|--------|-------------|
| `onBatchComplete(full, partials, batch_index)` | Called after each postprocessor batch. Must be non-blocking (< 50us). |
| `shouldStop()` | Polled in sieve loop condition. Returns `true` when coordinator signals STOP. |

### DirectChannel

Bounded SPSC circular buffer (`direct_channel.h`). Header-only. Implements `DataTap` (producer side, Thread B) with a consumer API for Thread A.

| Method / Field | Description |
|----------------|-------------|
| `DirectChannel(capacity=64)` | Constructor. Allocates `capacity` payload slots. |
| `onBatchComplete()` | Enqueue payload (vector copy). Backpressure: drops oldest if full. |
| `shouldStop()` | Atomic load of `stop_` flag. |
| `tryPop(out)` | Non-blocking consumer pop (Thread A). |
| `signalStop()` | Atomic store of `stop_` flag (Thread A -> Thread B). |
| `Payload` | `{HostRelationBatch full, HostRelationBatch partials, uint64_t batch_idx}` |

Guard: `std::mutex`. Low contention (Thread B produces at ~30ms, Thread A polls at ~10ms).

### AsyncNetworkDataTap

Remote worker DataTap (`async_network_data_tap.h/.cpp`). Replaces the old synchronous `NetworkDataTap`. Uses an `SPSCRing<TapSlot, 32>` + dedicated I/O thread to decouple the sieve loop from TCP I/O.

| Method / Field | Description |
|----------------|-------------|
| `AsyncNetworkDataTap(backend)` | Starts I/O thread. I/O thread sends an immediate heartbeat on startup. |
| `onBatchComplete()` | Memcpy extracted data into next SPSC ring slot. Returns in <50us; never blocks on TCP. |
| `shouldStop()` | Atomic flag set by I/O thread when it receives STOP or ERROR. Also `true` on range exhaustion. |
| `setRange(count, chunk_id, batch_a_vals)` | Set current chunk parameters before each `SieveStage()`. |
| `shutdown()` | Signal I/O thread to flush and exit. Must be called before `CommBackend` destruction. |
| `receivedStop()` | Distinguishes explicit STOP from range exhaustion. |
| `batchesSent()` | Atomic counter for telemetry. |

**I/O thread responsibilities:** Drains the SPSC ring, calls `mergeRelationBatches()` to coalesce small batches before TCP send, serializes and sends `INCREMENTAL_BATCH` messages, sends `HEARTBEAT` every `kHeartbeatIntervalMs` (5s), and polls for incoming STOP/ERROR messages. This thread is independent of the sieve loop — heartbeats continue during CUDA graph capture + JIT compilation regardless of how long the sieve thread is blocked.

### AccumulatorQueue

Thread-safe MPSC queue (`accumulator.h`). Four producer methods, one consumer `drain()`.

| Method | Thread | Description |
|--------|--------|-------------|
| `pushRelations(batch)` | Thread B / Network | Enqueue local sieve full relations |
| `pushPartials(batch)` | Thread B / Network | Enqueue local sieve 1-partials |
| `pushRemoteRelations(batch, worker_id)` | Thread A | Enqueue remote worker full relations |
| `pushRemotePartials(batch, worker_id)` | Thread A | Enqueue remote worker 1-partials |
| `drain()` | Thread A | Non-blocking drain returns `DrainResult` with all queued items |
| `signalSieveDone()` / `isSieveDone()` | Thread B / Thread A | Atomic sieve completion flag |

### RelationAccumulator

Single-thread dedup + counting (`accumulator.h`). Owned exclusively by Thread A.

| Method | Description |
|--------|-------------|
| `RelationAccumulator(target, dedup_margin=1.05)` | Target with 5% margin for dedup losses (auto 1.35 for <80d). Configurable via `--dedup_safety_factor`. |
| `addRelations(batch, source_id)` | Ingest with hash-based dedup. `source_id`: 0=local, 1-254=workers, 255=LP. |
| `addLPRelations(batch)` | Alias for `addRelations(batch, 255)`. |
| `targetReached()` | True when `accumulated_.num_relations >= effective_target_`. |
| `extractFinal()` | Move-extract the accumulated batch. Accumulator is empty after this. |
| `relationsFrom(source_id)` | Per-source breakdown for logging. |

**Dedup hash:** `(len << 48) | (exp_xor << 32) | body_xor`, matching `deduplicateHostBatch` in `mpqs_soa.cu`.

### FinalBatchHandoff

Blocking producer-consumer handoff (`accumulator.h`). Thread A calls `deliver()`, Thread B blocks on `await()`.

### NodeTelemetry

Per-node telemetry struct (`accumulator.h` or `cluster_common.h`). Populated by Thread A after sieve completion and logged as a formatted table.

| Field | Description |
|-------|-------------|
| `full_relations` | Full (smooth) relations received from this node |
| `partial_relations` | 1-partial relations received |
| `lp_combined` | LP-combined relations credited to this node's partials |
| `first_batch_ts` / `last_batch_ts` | Timestamps of first and last `INCREMENTAL_BATCH` received |
| `throughput_rels_per_sec` | Computed full-relation throughput over the run |

### CommBackend

Abstract communication backend (`comm_backend.h`). Decouples coordinator/worker logic from transport.

| Method | Description |
|--------|-------------|
| `setLocalInfo(hello)` | Set worker handshake info before `initialize()`. |
| `initialize()` | Coordinator: listen + accept + HELLO/HELLO_ACK. Worker: connect + handshake. |
| `finalize()` | Close all connections. Idempotent. |
| `send(target_id, type, data, len)` | Point-to-point send. |
| `recv(out)` / `recvBlocking(out, timeout_ms)` | Non-blocking / blocking receive. |
| `broadcast(type, data, len)` | Coordinator: send to all workers. |
| `barrier()` | Coordinator: STOP + wait for FLUSH_ACKs. Worker: send FLUSH_ACK. |
| `peerCount()`, `selfId()`, `peerInfo(id)` | Connection info. |
| `isPeerConnected(id)`, `disconnectPeer(id)` | Peer health management. |

Factory: `createCommBackend("tcp", is_coordinator, host, port, expected_workers)`.

### TCPBackend

TCP implementation of `CommBackend` (`tcp_backend.h`, `tcp_backend.cpp`).

- **Coordinator mode:** `TcpSocket::listen()` on port, `epoll_create1()` for multiplexed I/O. Accept loop with 180s timeout (Jetson workers take ~80s to start). HELLO/HELLO_ACK handshake assigns worker IDs 1-254. Worker sockets set to non-blocking after handshake. Buffered recv via `recvFromEpoll()` scans all peer buffers before falling back to `epoll_wait()`.
- **Worker mode:** Single blocking `TcpSocket::connect()`. Non-blocking recv uses 1ms timeout.

### TcpSocket

RAII TCP socket wrapper (`tcp_transport.h`, `tcp_transport.cpp`). Length-prefixed framing with CRC32 integrity.

| Method | Description |
|--------|-------------|
| `listen(port)` | Bind + listen with `SO_REUSEADDR`. |
| `accept()` | Blocking accept, returns new `TcpSocket` with `TCP_NODELAY`. |
| `connect(host, port)` | Client connect with DNS resolution fallback. |
| `sendMsg(type, payload, len)` | Frame: `FrameHeader + payload + CRC32`. |
| `recvMsg(type, payload)` | Buffered recv: accumulates partial frames in `recv_buf_`. |

### WorkPool

Thread-safe polynomial work-unit pool (`work_pool.h`, `work_pool.cpp`). Tracked-checkout model with fault tolerance.

| Method | Description |
|--------|-------------|
| `WorkPool(a_start, total_a, unit_size=64)` | Linear cursor from `a_start` to `a_start + total_a`. |
| `checkoutWork(count, worker_id)` | Returns `CheckedOutWork` with unique `chunk_id`. Serves reclaimed work before linear cursor. |
| `completeChunk(chunk_id)` | Remove from in-flight tracking. |
| `reclaimWork(worker_id)` | Reclaim all in-flight chunks for a dead worker (returned to LIFO queue). |
| `reclaimPartial(worker_id, chunk_id)` | Reclaim a specific in-flight chunk from a straggler (CHUNK_RECALL path). |
| `remaining()`, `exhausted()`, `inFlight()` | Pool status queries. |
| `setCursor(cursor)` | Restore from checkpoint (startup only). |

### ChunkScheduler

Adaptive chunk sizing with EMA throughput tracking (`chunk_scheduler.h`, `chunk_scheduler.cpp`).

| Method | Description |
|--------|-------------|
| `ChunkScheduler(pool, num_workers, total_a, H, Q)` | H = hypercube size, Q = batch_size * max(graph_unroll, 1). |
| `computeContiguousRanges(sm_counts, clock_rates, target_rels, H, Q)` | Initial SM×clock-proportional split. W_est = max(4*N, ceil(target/(5H))*2). Weights overridable via `--cluster_node_weights`. |
| `recordCompletion(worker_id, rels, a_vals, elapsed_s)` | Update EMA throughput model (alpha=0.3). |
| `nextChunkSize(worker_id)` | Adaptive size proportional to worker throughput. Hypercube-aligned, quantum-rounded. |
| `minChunk()` | max(H, 4*Q, 16). |

**Initial balance modes:** `SM_COUNT` (default, proportional to SM count × clock frequency), `THROUGHPUT_PROBE` (5s probe), `MANUAL` (capacity_estimate from HelloPayload). Override weights via `--cluster_node_weights` (comma-separated floats). Per-node headroom extends each node's initial range by `--cluster_headroom` % (default 10%) to reduce straggler wait.

### CPULargePrimeTable

CPU-side single large prime hash table (`cpu_lp.h`, `cpu_lp.cu`). Replaces GPU `LargePrimeVariant` on the coordinator to avoid blocking the GPU sieve.

| Method | Description |
|--------|-------------|
| `CPULargePrimeTable(lp1_bound, fdata)` | Initialize Montgomery context from N. Reserve 1M buckets. |
| `insertAndMatch(partials, accumulator)` | Insert new partials, combine matches into full relations via Montgomery multiply. |
| `witnesses()`, `totalInserts()`, `totalMatches()`, `totalCombines()` | Telemetry. |

**Combination:** Two partials with matching LP value p are combined: `sqrt_Q = a*b mod N` (Montgomery), `sign = XOR`, `val_2_exp = sum`, factors merged via sorted merge with exponent summation.

## Communication Protocol

### Wire Format

Every message is framed as:

```
[FrameHeader: 11B][payload: variable][CRC32: 4B]
```

`FrameHeader` layout (packed, 11 bytes):

| Field | Type | Description |
|-------|------|-------------|
| `magic` | `uint16_t` | `0x4D51` ("MQ") |
| `msg_type` | `uint8_t` | `MsgType` enum value |
| `seq_no` | `uint32_t` | Monotonic per-connection sequence number |
| `payload_len` | `uint32_t` | Byte length of payload (excludes header and CRC) |

CRC32 covers header + payload (polynomial 0xEDB88320, table-driven). All platforms are little-endian (static assertion enforced).

### Message Types

| Type | Code | Direction | Payload | Description |
|------|------|-----------|---------|-------------|
| `HELLO` | 0x01 | W -> C | `HelloPayload` (88B) | Worker registration: GPU name, SM version, SMs, VRAM, capacity, resume fields |
| `HELLO_ACK` | 0x02 | C -> W | `HelloAckPayload` (2B) | Accept + assign worker_id (1-254) |
| `WORK_ASSIGN` | 0x10 | C -> W | Variable | N + factor base + sieve params + poly range + AFactorsSnapshot |
| `WORK_MORE` | 0x11 | C -> W | `WorkMorePayload` (16B) | Additional poly range |
| `WORK_REQUEST` | 0x12 | W -> C | -- | Request more work |
| `CHUNK_ASSIGN` | 0x13 | C -> W | `ChunkAssignPayload` (24B) | Chunk of a-values with flags (initial/final/overflow) |
| `CHUNK_COMPLETE` | 0x14 | W -> C | `ChunkCompletePayload` (32B) | Chunk done + elapsed + relations + a-values consumed |
| `CHUNK_RECALL` | 0x17 | C -> W | `ChunkRecallPayload` (4B) | Reclaim a specific in-flight chunk from a straggler; anti-thrashing guards (30s min, 60s cooldown) |
| `RELATION_BATCH` | 0x20 | W -> C | Variable | Serialized `HostRelationBatch` |
| `PARTIAL_BATCH` | 0x21 | W -> C | Variable | 1-partial relations for LP matching |
| `INCREMENTAL_BATCH` | 0x22 | W -> C | Variable | Combined full + partial batch (DataTap) |
| `HEARTBEAT` | 0x30 | W -> C | `HeartbeatPayload` | Alive signal + batch count + GPU temp |
| `STATUS` | 0x31 | C -> W | `StatusPayload` | Global progress + ETA |
| `STOP` | 0xF0 | C -> W | `StopPayload` (1B) | Terminate sieving (reason: target/error/abort) |
| `FLUSH_ACK` | 0xF1 | W -> C | -- | Worker has sent all remaining data |
| `ERROR` | 0xFF | Either | -- | Fatal error |

### Serialization Formats

**HostRelationBatch** (`serializeRelationBatch`):
```
[num_relations:u32][num_factors:u32]
[sqrt_Q: N*64B][signs: N*1B][val_2_exps: N*4B][large_primes: N*16B]
[factor_offsets: (N+1)*8B][factor_indices: NNZ*4B][factor_counts: NNZ*1B]
```

**INCREMENTAL_BATCH** (`serializeIncrementalBatch`):
```
[full_data_len:u32][full_batch_data][partial_data_len:u32][partial_batch_data]
```

**WORK_ASSIGN** (`serializeWorkAssign`):
```
[N:64B][fb_size:u32][M:u32][F:u32][sieve_batch_size:u32]
[shc_dim:u8][pad:3B][threshold_override:u64][lp1_bound:u64]
[poly_range_start:u64][poly_range_count:u64][target_relations:u64]
[factorBase: fb_size*4B][rootN: fb_size*4B]
[snapshot_dim:u32][snapshot_a_factors: dim*4B]  -- M3 extension (optional)
[lowerHalfStart:u32][upperHalfStart:u32]
```

M3 snapshot extension is backward-compatible: M2-era messages without snapshot fields are accepted.

All deserialization uses a bounds-checked `SafeReader` that tracks position and validates remaining buffer length before every read.

## Work Distribution

### Contiguous Range Assignment

At setup, the `ChunkScheduler` computes contiguous a-value ranges proportional to each node's SM count × clock frequency:

```
W_est = max(4 * num_nodes, ceil(target_rels / (5 * H)) * 2)
R_i   = ceil(W_est * W_i / W_total) * H    -- W_i = SM_count_i × clock_rate_i (or CLI override)
```

Where H = 2^shc_dim (hypercube size). Each range is extended by `cluster_headroom` % to absorb imbalance. Weights can be overridden via `--cluster_node_weights`. Each range is also quantum-aligned (Q = batch_size * max(graph_unroll, 1)).

### Dynamic Rebalancing

After initial ranges are exhausted, the overflow pool (WorkPool) distributes additional chunks dynamically:

1. Worker sends `CHUNK_COMPLETE` with elapsed time and throughput stats.
2. Coordinator calls `ChunkScheduler::recordCompletion()` to update EMA model.
3. Coordinator calls `nextChunkSize()` for adaptive sizing (proportional to worker's throughput relative to mean).
4. Coordinator sends `CHUNK_ASSIGN` from `WorkPool::checkoutWork()`.

Chunks from reclaimed dead-worker work (via `reclaimWork()`) are served before advancing the linear cursor.

**Alignment:** All chunk sizes are rounded to multiples of H (hypercube alignment) and Q (quantum alignment). Minimum chunk size: max(H, 4Q, 16).

## LP Processing

In cluster mode, large prime matching runs on the coordinator's CPU (Thread A), not on the GPU:

1. Workers send 1-partial relations as part of `INCREMENTAL_BATCH` messages.
2. Thread A drains partials from `AccumulatorQueue` and calls `CPULargePrimeTable::insertAndMatch()`.
3. Matches are combined into full relations via Montgomery multiplication and added to the `RelationAccumulator` with `source_id=255`.

This avoids GPU LP processing on workers (which would require GPU synchronization and complicate the DataTap extraction path). GPU LP remains active in solo mode, unchanged.

## Data Extraction

Async extraction from the GPU sieve loop to the DataTap uses double-buffered host staging on a dedicated `extract_stream`. The extraction runs between CUDA graph replays (when `cuda_graph_unroll > 0`), ensuring the graph capture is not invalidated.

**Overhead:** 0.38% at RSA-100 scale (measured). The `onBatchComplete()` callback must complete in < 50us to avoid stalling the sieve pipeline. `DirectChannel` achieves this via mutex-guarded vector copy (~50us for ~500 KB at RSA-100). `AsyncNetworkDataTap` achieves this via memcpy into an SPSC ring slot (<50us); the dedicated I/O thread handles all TCP I/O, serialization, and heartbeats asynchronously.

## CUDA Graph Compatibility

CUDA graph capture (`--cuda_graph_unroll N`) is fully compatible with cluster mode. The extraction callback runs *between* graph replays, not during capture. Sequence:

1. CUDA graph is captured (N sieve + postprocess iterations).
2. Graph is replayed.
3. After replay completes, extraction runs on `extract_stream`.
4. `onBatchComplete()` is called with the extracted data.
5. Next replay begins.

The I/O thread in `AsyncNetworkDataTap` ensures heartbeats continue during graph capture + compilation, which can block the sieve thread for >120s on Jetson.

## Protocol Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `kProtocolMagic` | `0x4D51` | Frame magic ("MQ") |
| `kProtocolVersion` | 1 | Wire protocol version |
| `kDefaultPort` | 9100 | Default TCP port |
| `kHeartbeatIntervalMs` | 5000 | Background heartbeat period |
| `kFlushTimeoutMs` | 120000 | Flush/heartbeat timeout (120s, for Jetson graph capture) |
| `kBatchSendThreshold` | 1000 | Relations per send trigger |
| `kBatchSendCeilingMs` | 10000 | Maximum time between sends |
| `kCRC32Size` | 4 | CRC32 trailer size |

## Known Issues

- **Thread B range enforcement:** Coordinator Thread B may slightly overshoot its assigned contiguous range due to batch quantization.
- **Heartbeat timeout during graph capture:** `AsyncNetworkDataTap`'s I/O thread sends heartbeats independently of the sieve loop, so CUDA graph compilation does not cause timeouts. Workers must still complete their first graph replay within `kFlushTimeoutMs` (120s) for the SPSC ring not to overflow; Jetson workers with `cuda_graph_unroll=8` are near this limit.
- **LP below 85 digits:** LP causes 100% sqrt failure below ~85 digits due to a-factor/sieve-prime structural dependence. This is a mathematical limitation, not a cluster-specific bug. LP is disabled below 85 digits.
