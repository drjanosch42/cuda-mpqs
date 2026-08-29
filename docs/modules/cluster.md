# Cluster Module (`src/cluster/`)

## Overview

Distributed sieve stage for the CUDA-MPQS pipeline. Enables multiple GPU nodes to cooperatively sieve smooth relations over a LAN, aggregating results on a single coordinator node that drives the remaining pipeline stages (matrix, linear algebra, square root).

**Design philosophy:** The orchestrator's `SieveStage()` is the sole sieve implementation for solo, coordinator, and worker modes. Cluster mode hooks into the sieve loop via a **DataTap** callback interface injected at two points: `onBatchComplete()` after each postprocessor batch, and `shouldStop()` in the loop condition. In solo mode the DataTap pointer is `nullptr` (zero overhead). This eliminates the duplicate sieve loops that plagued the v1 cluster implementation.

**Architecture:** The coordinator runs two threads. Thread B executes the standard `SieveStage()` with a `DirectChannel` DataTap that pushes extracted relations to Thread A via a bounded circular buffer. Thread A runs a CPU-only network loop: receives incremental batches from remote workers via `CommBackend`, performs CPU-side LP matching, deduplicates, and tracks global progress. When the relation target is reached, Thread A signals STOP to all workers and Thread B.

Static library `mpqs_cluster`. Separable CUDA compilation ON. Namespace: `mpqs::cluster`.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `data_tap.h` | 43 | Abstract `DataTap` callback interface: `onBatchComplete()`, `shouldStop()` |
| `direct_channel.h` | 182 | `DirectChannel`: mutex-guarded SPSC circular buffer, implements `DataTap` for coordinator Thread B -> Thread A |
| `spsc_ring.h` | 107 | `SPSCRing<T, N>`: lock-free single-producer single-consumer ring buffer template |
| `async_network_data_tap.h` | 141 | `AsyncNetworkDataTap`: SPSC ring DataTap for workers; `onBatchComplete()` <50us (memcpy into ring slot). Explicit `shutdown()`, `tryTakeChunkAssign()`. |
| `async_network_data_tap.cpp` | 271 | I/O thread: serialization, batch coalescing (`mergeRelationBatches()`), TCP send, heartbeat, STOP/RECALL/CHUNK_ASSIGN polling (sole socket reader) |
| `network_data_tap.h` | 9 | **Deprecated stub.** Superseded by `async_network_data_tap.h`; retained for git history only, contains no active code. |
| `accumulator.h` | 242 | `AccumulatorQueue` (MPSC thread-safe queue), `RelationAccumulator` (single-thread dedup + counting, non-consuming `peek()` for checkpoints), `FinalBatchHandoff` (blocking condition-variable handoff) |
| `cpu_lp.h` | 90 | `CPULargePrimeTable`: CPU hash table for single large prime matching in cluster mode |
| `cpu_lp.cu` | 170 | `CPULargePrimeTable` implementation: insert-and-match, Montgomery-based partial combination, sorted factor merge, same-`sqrt_Q` identity guard, char-bit XOR-combine |
| `cluster_common.h` | 187 | Wire protocol: `FrameHeader`, `MsgType` enum (17 types incl. `CHUNK_REQUEST`), payload structs, protocol constants |
| `comm_backend.h` | 136 | Abstract `CommBackend` interface: lifecycle, point-to-point, collective, info. Factory `createCommBackend()` (takes `init_timeout_ms`) |
| `tcp_transport.h` | 77 | `TcpSocket` RAII wrapper: listen/accept/connect, length-prefixed framing + CRC32 |
| `tcp_transport.cpp` | 327 | POSIX TCP implementation: CRC32 table-driven, buffered recv, `sendExact` with EAGAIN retry, 64 MiB `kMaxPayloadBytes` frame cap |
| `tcp_backend.h` | 74 | `TCPBackend`: epoll-based coordinator, single-socket worker |
| `tcp_backend.cpp` | 452 | TCP backend: HELLO/HELLO_ACK handshake, epoll multiplexing, barrier, peer management |
| `serialization.h` | 123 | Binary serialization for `HostRelationBatch`, `WORK_ASSIGN` (v2: FB hash), and `INCREMENTAL_BATCH`; `computeFactorBaseHash()` |
| `serialization.cpp` | 418 | Serialization implementation: bounds-checked `SafeReader`, CSR-aware batch encoding, FNV-1a factor-base hash, backward-compatible M3 snapshot extension |
| `work_pool.h` | 159 | `WorkPool`: thread-safe polynomial work-unit pool with tracked checkout, reclaim, single-chunk return, completed-prefix cursor, and cursor restore |
| `work_pool.cpp` | 171 | WorkPool implementation: LIFO reclaim/return queue, linear cursor fallback, per-worker in-flight tracking |
| `chunk_scheduler.h` | 148 | `ChunkScheduler`: EMA throughput tracker, adaptive chunk sizing, contiguous range computation, default-inert debug window cap |
| `chunk_scheduler.cpp` | 280 | Scheduler implementation: SM-proportional initial split, quantum/hypercube alignment, confidence ramp |
| **Total** | **~3800** | |

The 64-bit relation dedup hash shared with the solo path lives outside this module in
`src/common/relation_hash.h` (see [RelationAccumulator](#relationaccumulator)).

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
| `onBatchComplete(full, partials, batch_index, a_values_advanced=0)` | Called after each postprocessor batch. Must be non-blocking (< 50us). `a_values_advanced`: a-values consumed since the previous call — `0` means "use the per-batch default from `setRange`"; the CUDA-graph replay loop passes the true count (`sieve_batch_size × cuda_graph_unroll`) since one replay advances `graph_N` batches but fires this callback once. Without it, the per-worker a-range guard under-counted by the unroll factor and nodes overran into each other's a-ranges, producing byte-identical duplicate partials (the cluster duplicate-partial bug, fixed `9881c00`). |
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
| `tryTakeChunkAssign(out)` | Take a `CHUNK_ASSIGN` frame captured by the I/O thread (mid-sieve overflow re-dispatch), if pending. Returns `true` and fills `out` (and clears the slot) when a frame is waiting. The I/O thread is the **sole** socket reader, so the main thread's chunk-wait loop must route overflow assignments through here rather than a second, racing `recv()`. |
| `batchesSent()` | Atomic counter for telemetry. |

**I/O thread responsibilities:** Drains the SPSC ring, calls `mergeRelationBatches()` to coalesce small batches before TCP send, serializes and sends `INCREMENTAL_BATCH` messages, sends `HEARTBEAT` every `kHeartbeatIntervalMs` (5s), and is the **sole reader** of the worker socket — polling for incoming STOP/ERROR (set atomically), `CHUNK_RECALL` (atomic), and mid-sieve `CHUNK_ASSIGN` (overflow re-dispatch) frames. A `CHUNK_ASSIGN` is captured into a mutex-guarded slot (`pending_assign_`) and handed to the main thread's chunk-wait loop via `tryTakeChunkAssign()`. Having a single reader eliminates a former two-readers race that silently swallowed mid-sieve `CHUNK_ASSIGN` frames (stranding workers waiting for overflow work) and avoids `recv_buf_` framing corruption from interleaved reads. This thread is independent of the sieve loop — heartbeats continue during CUDA graph capture + JIT compilation regardless of how long the sieve thread is blocked.

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
| `peek()` | Non-consuming const view of the accumulated batch (coordinator checkpoint). |
| `relationsFrom(source_id)` | Per-source breakdown for logging. |

**Dedup hash:** `(len << 48) | (exp_xor << 32) | body_xor`, where `body_xor` folds
`factor_indices·MAGIC`, sign, and `val_2_exp`. The formula is defined ONCE in
`src/common/relation_hash.h` (`mpqs::computeRelationHash`, m-sharedTU) and delegated to by the
accumulator, the solo checkpoint host-dedup, and the GPU `compute_relation_hashes_soa`
(`src/postprocessing/postprocessing.cu`) — byte-for-byte agreement across all three.
`char_bits` is deliberately NOT folded in (it is a deterministic function of `(ax+b)`, so
including it cannot change dedup identity and would diverge from the GPU hash).

### FinalBatchHandoff

Blocking producer-consumer handoff (`accumulator.h`). Thread A calls `deliver()`, Thread B blocks on `await()`.

### NodeTelemetry

Per-node telemetry struct (local to `networkLoop()` in `src/orchestrator/orchestrator.cpp`). Populated by Thread A after sieve completion and logged as a formatted table.

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

Factory: `createCommBackend("tcp", is_coordinator, host, port, expected_workers, init_timeout_ms = kDefaultInitTimeoutMs)`.

### TCPBackend

TCP implementation of `CommBackend` (`tcp_backend.h`, `tcp_backend.cpp`).

- **Coordinator mode:** `TcpSocket::listen()` on port, `epoll_create1()` for multiplexed I/O. Accept loop bounded by `init_timeout_ms` (default `kDefaultInitTimeoutMs` = 300s, CLI `--cluster_init_timeout`; Jetson workers take ~80s of cold JIT to start). HELLO/HELLO_ACK handshake assigns worker IDs 1-254. Worker sockets set to non-blocking after handshake. Buffered recv via `recvFromEpoll()` scans all peer buffers before falling back to `epoll_wait()`.
- **Worker mode:** `TcpSocket::connect()` with retry every `kConnectRetryIntervalMs` (5s) within the same `init_timeout_ms` window. Non-blocking recv uses 1ms timeout.

### TcpSocket

RAII TCP socket wrapper (`tcp_transport.h`, `tcp_transport.cpp`). Length-prefixed framing with CRC32 integrity.

| Method | Description |
|--------|-------------|
| `listen(port)` | Bind + listen with `SO_REUSEADDR`. |
| `accept()` | Blocking accept, returns new `TcpSocket` with `TCP_NODELAY`. |
| `connect(host, port)` | Client connect with DNS resolution fallback. |
| `sendMsg(type, payload, len)` | Frame: `FrameHeader + payload + CRC32`. |
| `recvMsg(type, payload)` | Buffered recv: accumulates partial frames in `recv_buf_`. Rejects any frame whose `payload_len` exceeds `kMaxPayloadBytes` = 67,108,864 (64 MiB, `tcp_transport.cpp:222`) — the frame cap that a full-FB `WORK_ASSIGN` (~105 MB at F=500M) used to trip before the FB-hash fix (`b220a61`). |

### WorkPool

Thread-safe polynomial work-unit pool (`work_pool.h`, `work_pool.cpp`). Tracked-checkout model with fault tolerance.

| Method | Description |
|--------|-------------|
| `WorkPool(a_start, total_a, unit_size=64)` | Linear cursor from `a_start` to `a_start + total_a`. |
| `checkoutWork(count, worker_id)` | Returns `optional<CheckedOutWork>` with unique `chunk_id`. Serves reclaimed work before linear cursor. |
| `completeChunk(chunk_id)` | Remove from in-flight tracking. |
| `returnChunk(chunk_id)` | Return a single checked-out chunk to the assignable (`returned_`) pool without consuming any of it — used when a `CHUNK_ASSIGN` send fails so the chunk can be re-dispatched immediately. Returns the a-value count returned (0 if not in flight). |
| `reclaimWork(worker_id)` | Reclaim all in-flight chunks for a dead worker (returned to LIFO queue). Returns total a-values reclaimed. |
| `reclaimPartial(chunk_id, consumed_count)` | Reclaim the unconsumed tail of a specific in-flight chunk from a straggler (CHUNK_RECALL path); equivalent to `completeChunk` when fully consumed. |
| `remaining()`, `exhausted()`, `inFlight()`, `inFlightFor(worker_id)`, `remainingOrInFlight()` | Pool status queries. |
| `nextCursor()` | Raw linear cursor `next_` (diagnostics only — drops in-flight/returned chunks). |
| `completedPrefixCursor()` | `min(next_, min start over in_flight_ ∪ returned_)` — the checkpoint-safe cursor (see Coordinator Checkpointing below). |
| `setCursor(cursor)` | Restore from checkpoint (startup only; asserts clean state). |

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

**Debug repro hook (default-inert):** `applyDebugWindowCap()` caps every computed chunk (initial contiguous ranges *and* overflow chunks) to `MPQS_DEBUG_MAX_CHUNK_WINDOWS` hypercube windows (HC alignment preserved), forcing many `CHUNK_COMPLETE` → `CHUNK_ASSIGN` turnover cycles to exercise the overflow-assignment path. Parsed once per process; identity when the env var is unset. Used together with the orchestrator's `MPQS_DEBUG_DROP_CHUNK_ASSIGN=<worker>:<occurrence>` lost-send injector as the chunk-delivery race regression harness.

### CPULargePrimeTable

CPU-side single large prime hash table (`cpu_lp.h`, `cpu_lp.cu`). Replaces GPU `LargePrimeVariant` on the coordinator to avoid blocking the GPU sieve.

| Method | Description |
|--------|-------------|
| `CPULargePrimeTable(lp1_bound, fdata)` | Initialize Montgomery context from N. |
| `insertAndMatch(partials, accumulator)` | Insert new partials, combine matches into full relations via Montgomery multiply. |
| `witnesses()` | Telemetry. ⚠ **Live table OCCUPANCY (`table_.size()`, `cpu_lp.h:51`), NOT cumulative partial arrivals** — a successful match `erase()`s the entry (`cpu_lp.cu:44`), so this counter *falls* on every combine. Arrivals are `P = W + 2·matches ≈ W + 2·combines` and are **never printed**. Misreading `W` as `P` produced the retracted coordinator-throughput ceiling. |
| `bucketCount()` | `table_.bucket_count()`. The ctor `reserve()`s 1<<20 buckets (`cpu_lp.cu:21`), so a step in this value between two stats samples dates a stop-the-world rehash stall exactly. |
| `totalInserts()`, `totalMatches()`, `totalCombines()`, `totalDupDropped()` | Cumulative counters. |

**Combination:** Two partials with matching LP value p are combined: `sqrt_Q = a*b mod N` (Montgomery), sign via encoding-agnostic XOR of the "negative iff `!= 1`" booleans (M11c pattern — output encoded `{1, 0xFF}`), `val_2_exp = sum`, `char_bits = XOR` (Stage 5 branch-character combine), factors merged via sorted merge with exponent summation.

**Identity guard:** a match whose two partials share the same `sqrt_Q` (byte-identical cross-node duplicate) is dropped and counted in `totalDupDropped()` — combining them would yield a perfect square (`X == Y`) and a trivial sqrt (defense-in-depth for the duplicate-partial pathology fixed in `9881c00`).

## Communication Protocol

### Wire Format

Every message is framed as:

```
[FrameHeader: 11B][payload: variable][CRC32: 4B]
```

`FrameHeader` layout (packed, 11 bytes):

| Field | Type | Description |
|-------|------|-------------|
| `magic` | `uint16_t` | `0x4D52` ("MR") — bumped from `0x4D51` ("MQ") with the v2 WORK_ASSIGN layout so a stale binary fails fast at the frame-magic check instead of mis-parsing |
| `msg_type` | `uint8_t` | `MsgType` enum value |
| `seq_no` | `uint32_t` | Monotonic per-connection sequence number |
| `payload_len` | `uint32_t` | Byte length of payload (excludes header and CRC); rejected above `kMaxPayloadBytes` (64 MiB) |

CRC32 covers header + payload (polynomial 0xEDB88320, table-driven). All platforms are little-endian (static assertion enforced). All frames share the magic, enforcing all-or-nothing build lockstep across the cluster.

### Message Types

| Type | Code | Direction | Payload | Description |
|------|------|-----------|---------|-------------|
| `HELLO` | 0x01 | W -> C | `HelloPayload` (88B) | Worker registration: GPU name, SM version, SMs, VRAM, capacity, resume fields |
| `HELLO_ACK` | 0x02 | C -> W | `HelloAckPayload` (2B) | Accept + assign worker_id (1-254) |
| `WORK_ASSIGN` | 0x10 | C -> W | Variable (~220 B, F-independent) | N + FB hash + sieve params + poly range + AFactorsSnapshot; worker regenerates + verifies the FB |
| `WORK_MORE` | 0x11 | C -> W | `WorkMorePayload` (16B) | Additional poly range |
| `WORK_REQUEST` | 0x12 | W -> C | -- | Request more work |
| `CHUNK_ASSIGN` | 0x13 | C -> W | `ChunkAssignPayload` (24B) | Chunk of a-values with flags (initial/final/overflow) |
| `CHUNK_COMPLETE` | 0x14 | W -> C | `ChunkCompletePayload` (32B) | Chunk done + elapsed + relations + a-values consumed |
| `CHUNK_REQUEST` | 0x15 | W -> C | -- | Idle worker re-requests work (carries **no** stats, distinct from `CHUNK_COMPLETE` so a retry never re-applies relation/partial counts). Recovers a dropped/failed `CHUNK_ASSIGN`; rate-limited to once / 15s on a socket-alive timeout. |
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
[char_bits: N*4B]                                -- Stage 4 branch char vector (always present;
                                                 --   a defined 0 under --char_mode norm; never
                                                 --   enters the dedup hash)
[factor_offsets: (N+1)*8B][factor_indices: NNZ*4B][factor_counts: NNZ*1B]
```

**INCREMENTAL_BATCH** (`serializeIncrementalBatch`):
```
[full_data_len:u32][full_batch_data][partial_data_len:u32][partial_batch_data]
```

**WORK_ASSIGN** (`serializeWorkAssign`, v2 layout — protocol magic 0x4D52):
```
[N:64B][fb_size:u32][M:u32][F:u32][sieve_batch_size:u32]
[shc_dim:u8][pad:3B][threshold_override:u64][lp1_bound:u64]
[poly_range_start:u64][poly_range_count:u64][target_relations:u64]
[fb_hash:u64]                                    -- computeFactorBaseHash(fdata); REPLACES the
                                                 --   former [factorBase][rootN] blob, making the
                                                 --   payload F-independent (~220 B vs >64 MiB at
                                                 --   RSA-155 F=500M, which exceeded the frame cap)
[snapshot_dim:u32][snapshot_a_factors: dim*4B]  -- M3 extension (optional)
[lowerHalfStart:u32][upperHalfStart:u32]
```

**Factor-base hash + regen-and-verify:** `computeFactorBaseHash(fdata)` (`serialization.h`) is a
canonical 64-bit FNV-1a over the raw bytes of `fb_size` (u32), then `factorBase[]` (u32 primes,
ascending), then `rootN[]` (u32 normalized roots, index-aligned) — the SINGLE hash used by both
sides, never duplicated. `deserializeWorkAssign` does NOT populate `fdata.factorBase/rootN`; the
worker regenerates the FB locally from the coordinator's authoritative `(N, F)` via the
deterministic `generateFactorBase()`, recomputes the hash, and verifies it (plus `fb_size`)
before sieving — on any mismatch it logs `LOG_ERROR_CRITICAL` "factor-base hash mismatch" and
exits rather than sieving a divergent FB (`orchestrator.cpp:399-400`). Raw-byte hashing assumes a
homogeneous little-endian cluster (static-assert enforced). Fix commit `b220a61`; validated at
cluster scale by the F=500M A100 valprobe (job 33466828).

**`sieve_batch_size` symmetry (WORK_ASSIGN field 5).** Coordinator and worker **must** run a
symmetric `--sieve_batch_size` — an asymmetric pair makes the coordinator the bottleneck (it carries
the LP matching) and drops cluster throughput below single-node solo. The wire enforces this *by
default but not unconditionally*: the worker adopts the coordinator's value only when it did not pin
its own on the command line — `if (recv_batch_size > 0 && !config_.isPinned("sieve_batch_size"))
config_.sieve_batch_size = recv_batch_size;` (`src/orchestrator/orchestrator.cpp:433-435`), where
`isPinned` is set by the CLI parser (`tests/cuda-mpqs.cpp:767-769`). So a worker launched with an
explicit `--sieve_batch_size` **keeps its own value and silently diverges** from the coordinator;
that is the only way asymmetry arises, and there is no check for it.

M3 snapshot extension is backward-compatible: v2-magic messages without snapshot fields are accepted.

All deserialization uses a bounds-checked `SafeReader` that tracks position and validates remaining buffer length before every read.

## Work Distribution

### Contiguous Range Assignment

At setup, the `ChunkScheduler` computes contiguous a-value ranges proportional to each node's SM count × clock frequency:

```
W_est = max(4 * num_nodes, ceil(target_rels / (5 * H)) * 2)
R_i   = ceil(W_est * W_i / W_total) * H    -- W_i = SM_count_i × clock_rate_i (or CLI override)
```

Where H = 2^shc_dim (hypercube size). Each range is extended by `cluster_headroom` % to absorb imbalance. Weights can be overridden via `--cluster_node_weights`. Each range is also quantum-aligned (Q = batch_size * max(graph_unroll, 1)).

### Overflow Pool Sizing

The overflow `WorkPool` is pure on-demand index space, drawn only after a node exhausts its initial contiguous range with the relation target still unmet. The run is bounded by the relation cap, not the pool, so over-provisioning is free — but under-provisioning is fatal (the pool drains, workers idle out their timeout, and the matrix stage aborts on insufficient relations).

The coordinator sizes the overflow pool (in `Orchestrator::Run()`, coordinator branch) from the **relation target**, not from a 1× doubling of the contiguous total:

```
windows_for_target = ceil(target_rels / (0.04 * H))   -- 0.04 = conservative per-a-index yield floor
total_windows      = max(contiguous_windows, windows_for_target) * cluster_pool_oversize
total_windows      = min(total_windows, (fb_size - 150) / 2)   -- a-factor walk capacity clamp
overflow_windows   = total_windows - contiguous_windows
```

The `0.04` yield floor sits several times below the measured per-a-index relation yield on large composites, giving margin without risk. `--cluster_pool_oversize <float>` (coordinator only, default 1.0) further enlarges the pool. The clamp to `(fb_size - 150) / 2` windows is the a-factor sliding-window walk's capacity (each window slide yields one new a-coefficient; the walk begins near FB index ~150 and advances its upper half with stride 2 until it runs off the factor base), ensuring no overflow chunk maps to an invalid a-coefficient. The coordinator logs an auditable `a-value pool: contiguous=… overflow=… total=… windows (oversize=…x, target=… rels, walk_ceiling=… windows)` line at startup.

> The earlier `overflow_size = overflow_start` (1× the contiguous total) combined with the scheduler's optimistic `w_est` (`5 * H` divisor, ~5 rel/a-index) sized the pool ~tens of times too small on large composites; it has been replaced by the target-derived sizing above. Initial contiguous ranges (`computeContiguousRanges` / `w_est`) are unchanged, so per-node startup work is unaffected.

### Dynamic Rebalancing

After initial ranges are exhausted, the overflow pool (WorkPool) distributes additional chunks dynamically:

1. Worker sends `CHUNK_COMPLETE` with elapsed time and throughput stats.
2. Coordinator calls `ChunkScheduler::recordCompletion()` to update EMA model.
3. Coordinator calls `nextChunkSize()` for adaptive sizing (proportional to worker's throughput relative to mean).
4. Coordinator sends `CHUNK_ASSIGN` from `WorkPool::checkoutWork()`.

Chunks from reclaimed dead-worker work (via `reclaimWork()`) are served before advancing the linear cursor.

**Alignment:** All chunk sizes are rounded to multiples of H (hypercube alignment) and Q (quantum alignment). Minimum chunk size: max(H, 4Q, 16).

#### Chunk-Assignment Delivery Recovery

All overflow `CHUNK_ASSIGN` sends route through a single send-checked routine (`assignChunkTo()` in the coordinator's Thread A). It checks out a chunk, sends it, and sets the worker's `current_chunk_id` **only on a confirmed (`true`) send**; on a failed send it returns the chunk to the assignable pool (`WorkPool::returnChunk()`), bumps `failed_assign_sends`, and leaves the worker with no current chunk. It is idempotent — a no-op when the worker already holds a chunk — so the three dispatch sources below can never double-assign:

- **`CHUNK_COMPLETE` path:** normal post-completion assignment (records stats first).
- **Worker re-request (`CHUNK_REQUEST`):** an idle worker re-requests work at most once / 15s on a socket-alive timeout. A `CHUNK_REQUEST` is explicit proof the worker is idle, so the handler **forces** re-assignment — it returns any chunk the coordinator still believes is in flight to the pool and clears `current_chunk_id` before calling `assignChunkTo()`, so a previously-lost assignment cannot wedge the worker. Carries no stats, so a retry never double-counts.
- **Proactive idle sweep:** each 5s timeout tick the coordinator hands a chunk to every alive worker that holds no chunk (`current_chunk_id == UINT32_MAX`) while assignable work remains and the target is unmet, draining reclaimed/returned chunks even without a re-request.

**Worker `current_chunk_id` sentinels:** a worker starts with `kInitialChunk` (`UINT32_MAX - 1`, "busy with its initial WORK_ASSIGN range — not a pool chunk"), cleared to `UINT32_MAX` ("idle, eligible for overflow") only on its first `CHUNK_COMPLETE`. The idle sweep, heartbeat-timeout reclaim, and `CHUNK_REQUEST` handler all treat the sentinel as "not a pool chunk" (no spurious `returnChunk`/reclaim), preventing the sweep from handing a second chunk to a worker still on its initial range.

`failed_assign_sends` and `redispatched_chunks` (chunks handed out via re-request or the idle sweep) are surfaced in the cluster sieve telemetry summary. The worker's 600s chunk-wait hard cap is now an absolute backstop that fires only on persistent timeout (heartbeats flowing, coordinator silent); a closed socket exits immediately.

## LP Processing

In cluster mode, large prime matching runs on the coordinator's CPU (Thread A), not on the GPU:

1. Workers send 1-partial relations as part of `INCREMENTAL_BATCH` messages.
2. Thread A drains partials from `AccumulatorQueue` and calls `CPULargePrimeTable::insertAndMatch()`.
3. Matches are combined into full relations via Montgomery multiplication and added to the `RelationAccumulator` with `source_id=255`.

This avoids GPU LP processing on workers (which would require GPU synchronization and complicate the DataTap extraction path). GPU LP remains active in solo mode, unchanged.

**Workers sieve NO-LP, structurally.** `largeprime_` is not merely unused in cluster mode, it is
never constructed: `SieveStage()` guards the matcher's construction with
`if (!cluster_mode) { initLargePrimes(); }` (`src/orchestrator/orchestrator.cpp:4360-4362`). This is
what makes invariant **I1** (`include/graph_capture_scope.h:16-20`) structural rather than a policy —
`cap.capture_lp = (achieved == FULL) && has_lp && !is_cluster` is false in cluster mode on two
independent grounds. Cross-node LP matching requires *all* partials at one place, which is the
coordinator.

### Coordinator LP Telemetry

Two `LOG_STATS` lines at the ~5 s stats cadence (`src/orchestrator/orchestrator.cpp:7520-7556`):

- `[Cluster] LP: <W> witnesses (<rate>/s) | <C> combines (<rate>/s) | yield <pct>%` — note `W` is
  table **occupancy**, not arrivals (see [CPULargePrimeTable](#cpulargeprimetable)).
- `[Cluster] LPocc: elapsed_us=… recv_us=… deser_us=… bufpart_us=… addrel_us=… insmatch_us=…
  ctl_us=… sleep_us=… insmatch_max_us=… insmatch_calls=… buckets=… witnesses=…` — 12
  space-separated `key=value` pairs partitioning Thread A's wall time.

**Parser contract** (stated in the source comment, `orchestrator.cpp:7526-7544`): field order is
stable and may be extended **only by appending**; every value is a raw plain `uint64` decimal;
all counters are cumulative and monotonic (differentiate consecutive samples for an interval rate).
Thread-A busy fraction = `(elapsed_us − sleep_us) / elapsed_us`; per-stage share =
`<stage>_us / elapsed_us`. **Saturation signature: `sleep_us` stops growing while `insmatch_us`
tracks `elapsed_us`.**

⚠ **`fmtSize()` is NEVER used on either line — hard rule.** Its K/M truncation above 1,000 silently
froze a whole campaign's witness extraction (fixed `b7ff2ce` / `2504647`). Verify a binary carries
the occupancy line with `strings <bin> | grep "\[Cluster\] LPocc:"`.

## Data Extraction

Async extraction from the GPU sieve loop to the DataTap uses double-buffered host staging on a dedicated `extract_stream`. The extraction runs between CUDA graph replays (when `cuda_graph_unroll > 0`), ensuring the graph capture is not invalidated. As of 1.0.6 that between-replay ordering is established by a **completion event recorded on the launch stream outside the capture** — a graph is a single stream work item, so that event signals the whole graph including its post-processing branch — which `extract_stream` waits on before the per-replay `cudaStreamSynchronize(extract_stream)` that gives extraction its exact counters. The **partial-counter reset is gated on the launch stream, immediately before the graph is launched**: waiting on the post-processing stream would order nothing, because at replay time that stream carries no work of its own, and the next replay's in-graph trial division could then write partials while `extract_stream` was still resetting the counters (a silent partial loss with no error and no counter).

**Overhead:** 0.38% at RSA-100 scale (measured). The `onBatchComplete()` callback must complete in < 50us to avoid stalling the sieve pipeline. `DirectChannel` achieves this via mutex-guarded vector copy (~50us for ~500 KB at RSA-100). `AsyncNetworkDataTap` achieves this via memcpy into an SPSC ring slot (<50us); the dedicated I/O thread handles all TCP I/O, serialization, and heartbeats asynchronously.

## CUDA Graph Compatibility

CUDA graph capture (`--cuda_graph_unroll N`) is fully compatible with cluster mode. The extraction callback runs *between* graph replays, not during capture. Step 1's "N sieve + postprocess iterations" was aspirational before 1.0.6 and is now literally true. Sequence:

1. CUDA graph is captured: N iterations of the batch body (sieve + batch trial division), double-buffered inside the graph.
2. The partial-reset gate (if an extraction is in flight) is issued on the launch stream, then the graph is launched and a completion event is recorded on the launch stream, outside the capture.
3. `extract_stream` waits on that event and is synchronized, so the counters are exact.
4. Extraction runs on `extract_stream`; `onBatchComplete()` is called with the extracted data.
5. Next replay begins.

**Cluster runs capture at scope `postproc`.** The large-prime pipeline is **never** captured in cluster mode — workers sieve without LP and the coordinator matches on the CPU — and requesting `--cuda_graph_capture full` in a cluster run is downgraded to `postproc` with a `LOG_WARNING` (`capture scope=postproc (LP excluded: cluster)`). Cluster also keeps its per-replay synchronization; only single-node runs drop it.

The I/O thread in `AsyncNetworkDataTap` ensures heartbeats continue during graph capture + compilation, which can block the sieve thread for >120s on Jetson.

## Sieve Geometry and the Cluster Path (v1.0.6)

`src/cluster/` itself has **no source changes** since v1.0.5 (`git diff 7c154c9..HEAD -- src/cluster` is
empty). Everything below is a property of the shared `SieveStage()` sieve that cluster ranks execute,
recorded here so a cluster operator knows what applies.

**Solo-relevant capabilities that no cluster launcher exercises.** Both v1.0.6's SM-aligned
(non-power-of-two) `{np, metaGridDim, sasGridDim}` narrow-batch geometry and v1.0.6's
`--sieve_block_size` / `--sieve_big_prime_start` / `--sieve_bucket_overflow_stats` knobs are reachable
from a cluster rank in principle — they are `MPQSConfig` fields consumed by the shared sieve, not
solo-gated — but every run that has ever pinned a non-power-of-two `--params` tuple or set the
v1.0.6 knobs was a **single-GPU solo probe or bench**, none of them passing `--cluster_mode`.
Nothing is claimed or measured for these geometries at cluster scale, and the production
RSA-150/155 cluster sieves run the **wide** (u8sat) path, where the v1.0.6 overrides are rejected
outright (`validateConfigs()`, narrow-batch only) and pow2 geometry stays mandatory.

**The narrow-batch coverage invariant applies to cluster ranks.** `numIntervals × sievingBlockSize ≥ 2M`
is checked in `DeviceSievingController::validateConfigs()` for **any** narrow batch run — coordinator,
worker or solo — and is **not** gated on the v1.0.6 overrides. A config that violates it now aborts
**loudly** pre-sieve (`LOG_ERROR_CRITICAL`, naming the required `numIntervals` = `--params` field 2)
instead of silently sieving only `[-M, 0)` for a ~50 % yield loss. A cluster rank carrying a pinned
`--params` tuple must therefore satisfy it; `loadStandardConfig` geometries and every shipped tuple
already do. Because coordinator and worker must use symmetric `--sieve_batch_size`, they resolve the
same batch/legacy predicate and so face the same check.

**The v1.0.6 GPU `sqrt_Q` identity guard fix was solo-only — cluster was never exposed.** That fix
added the device mirror of the guard to the solo GPU-LP `global_combine_kernel`. Cluster ranks never ran
the defective path: workers sieve NO-LP, and the coordinator's CPU matcher has carried the canonical
guard since 1.0.3a — still present at `src/cluster/cpu_lp.cu:69-71`, dropping a match whose two partials
share the same `sqrt_Q` and counting it in `totalDupDropped()` (see [CPULargePrimeTable](#cpulargeprimetable)).

**Still-pending validation (v1.0.6).** A **live 2-node cluster smoke** remains the only test that
closes the silent-partial-loss risk the launch-stream partial-reset gate addresses (see
[Data Extraction](#data-extraction)); no such run is recorded, so treat it as outstanding. The
remaining v1.0.6 measurements that have not been made — a three-arm H100
`capture=full` / `capture=sieve` / `cgu=0` speed and energy comparison, and a wide-path regression at
RSA-155 scale — are not cluster-blocking.

## Protocol Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `kProtocolMagic` | `0x4D52` | Frame magic ("MR"); bumped from `0x4D51` with the v2 WORK_ASSIGN layout |
| `kProtocolVersion` | 2 | Wire protocol version (defined for future handshake use, NOT transmitted in `FrameHeader` — cross-version rejection is via the magic) |
| `kDefaultPort` | 9100 | Default TCP port |
| `kDefaultInitTimeoutMs` | 300000 | Init window: coordinator accept + worker connect-retry (300s; Jetson cold JIT ~80s) |
| `kConnectRetryIntervalMs` | 5000 | Worker connect retry interval during init window |
| `kHeartbeatIntervalMs` | 5000 | Background heartbeat period |
| `kFlushTimeoutMs` | 120000 | Flush/heartbeat timeout (120s, for Jetson graph capture) |
| `kBatchSendThreshold` | 1000 | Relations per send trigger |
| `kBatchSendCeilingMs` | 10000 | Maximum time between sends |
| `kCRC32Size` | 4 | CRC32 trailer size |
| `kMaxPayloadBytes` | 67108864 | 64 MiB per-frame payload cap (`tcp_transport.cpp:222`, local to `recvMsg`) |

## Coordinator Checkpointing and Resume

Mid-sieve checkpointing is **coordinator-only** (workers are stateless and never checkpoint).
Workers always reconnect and request work on a resume — no worker-side changes needed.

### Checkpoint Flags (coordinator only)

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint_interval <T>` | 0 (OFF) | Wall-seconds between checkpoints. |
| `--checkpoint_dir <path>` | `work_dir/checkpoint` | Checkpoint directory; must be on a run-stable path (not jobid-based). |
| `--resume` | false | On startup, load `sieve.ckpt` from `checkpoint_dir` and resume. |

`--checkpoint_batches` is a solo-sieve notion and has no effect on the coordinator path
(the coordinator Thread A has no sieve-batch counter). Setting only `--checkpoint_batches` on a
coordinator is a silent no-op.

### Checkpoint Write (Thread A)

The coordinator checkpoint is emitted from **Thread A (`networkLoop`)** at the
**bottom-of-loop boundary** (immediately before the 1 ms idle sleep), on the wall-clock interval.
Thread A is the **sole mutator** of `cluster_accumulator_`, `cluster_raw_partials_`, and
`cluster_cpu_lp_`, so a snapshot taken there is host-consistent with nothing mid-flight.

The checkpoint uses `RelationAccumulator::peek()` (non-consuming; `extractFinal()` is
destructive) to read `accumulated_` without emptying the live pool.

The cluster block appended between the progress trailer and the fixed EOF footer contains:
- `completedPrefixCursor` — the completed contiguous prefix of the overflow `WorkPool`,
  computed as `min(next_, min start over in_flight_ ∪ returned_)`. NOT `nextCursor()` (which
  silently drops in-flight/returned chunks — see B2 rationale below).
- per-node initial-range contiguous high-water array — captures `CHUNK_COMPLETE`/reclaim events
  on the initial a-range specifically. Does NOT use `node_telemetry[].a_values_consumed` (which
  sums initial + overflow and would overshoot). Conservative (under-report ⇒ re-sieve a little
  on resume, never skip).

Atomicity and crash safety are identical to the solo path (unlink stale `.tmp` → write → fsync
→ rename-prev → rename → dir-fsync). Workers are byte-unchanged; default-off ⇒ a cluster run
with no checkpoint flags is byte-unchanged.

### Cluster Resume

On `--resume` at the coordinator, **before Thread A starts and before any checkout**:

1. Load `sieve.ckpt` via `readCheckpoint`; validate trailer `N == config N` (stale checkpoint
   rejected — never silently consumed). Fall back to `sieve.ckpt.prev` on a torn live file.
2. **Topology guard (`clusterResumeTopologyOk`):** reject if checkpoint `node_count` differs from
   the current run, or if `completedPrefixCursor` falls outside the current overflow pool bounds.
   A rejected resume starts fresh (safe; never skips a-values).
3. **Re-inject smooths:** `cluster_accumulator_->addRelations(loaded_smooths, 0)` — rebuilds
   `accumulated_` **and** the `seen_` dedup set, so the live pooled count includes the loaded
   smooths and `targetReached()` works unchanged.
4. **Re-feed partials:** `cluster_cpu_lp_->insertAndMatch(loaded_partials, *cluster_accumulator_)`
   — rebuilds `table_`, re-emits deduped combines (same-`sqrt_Q` identity guard prevents
   self-combine). Also buffers partials into `cluster_raw_partials_`.
5. **Restore overflow cursor (B2):** `cluster_work_pool_->setCursor(completedPrefixCursor)`.
   Using `nextCursor()` (= `next_`) instead would silently drop `in_flight_ ∪ returned_` chunks
   (lost on kill), leaving those a-values never sieved → under-collection and faster overflow
   drain → re-introduces the `4d20d7b` pool exhaustion. The completed prefix conservatively
   re-sieves only the out-of-order overflow tail above it (dedup-safe).
6. **Re-issue trimmed initial ranges (M1):** each node's `WORK_ASSIGN` is trimmed to
   `[orig_start + hw_node, orig_count − hw_node)` from the per-node high-water array.
   A node whose initial range is exactly complete re-sieves the last hypercube H (count==0
   boundary guard; dedup-safe). The coordinator's own (node-0) initial range is handled via
   `poly_range_start`/`poly_range_count` so the existing `resetAndAdvanceTo` jump gate applies.

Thread A then proceeds; the cluster sieves only the remaining a-space (overflow tail above the
prefix + trimmed initial tails) and tops up to the target.

### Run-Stable Checkpoint Path

A `CKPT_DIR` keyed on `SLURM_JOB_ID` breaks cross-resubmit resume (fresh jobid → fresh empty
dir → checkpoint never found). The production sbatch uses a **jobid-independent** path:

```bash
RUN_TAG="${RUN_TAG:-rsa140_F40M_M131K_L40T}"
CKPT_DIR="$RUN_BASE/cuda-mpqs/${RUN_TAG}_ckpt"
if [ -s "$CKPT_DIR/sieve.ckpt" ]; then
  COORD_RESUME="--resume"
else
  COORD_RESUME=""
fi
COORD_CKPT="--checkpoint_dir $CKPT_DIR --checkpoint_interval 1800 $COORD_RESUME"
# Pass $COORD_CKPT to the coordinator rank only; workers run without checkpoint flags.
```

`$RUN_TAG` changes only to start a genuinely independent run (a new N or new param set). The
loader validates the trailer `N == config N` so a stale checkpoint from a different run is
rejected rather than silently consumed. The final `relations.v2` (Phase-2 matrix handoff) is
written only at sieve completion, unchanged; `sieve.ckpt` is an internal resume artifact only.

## Known Issues

- **Thread B range enforcement:** Coordinator Thread B may slightly overshoot its assigned contiguous range due to batch quantization.
- **Heartbeat timeout during graph capture:** `AsyncNetworkDataTap`'s I/O thread sends heartbeats independently of the sieve loop, so CUDA graph compilation does not cause timeouts. Workers must still complete their first graph replay within `kFlushTimeoutMs` (120s) for the SPSC ring not to overflow; Jetson workers with `cuda_graph_unroll=8` are near this limit.
- **LP below 85 digits:** LP causes 100% sqrt failure below ~85 digits due to a-factor/sieve-prime structural dependence. This is a mathematical limitation, not a cluster-specific bug. LP is disabled below 85 digits.
- **Coordinator single-threaded CPU LP-matching ceiling — ⚠ SUPERSEDED (2026-07-17), retained for the structural facts:** the *throughput* claim below (~3,200–3,500 witnesses/s, and the 108-GPU rejection resting on it) was **refuted by direct measurement on the real production stream**: the matcher sustains ~753,652 arrivals/s, far above any offered load reachable here, so the real ceiling is coordinator **host RAM** (measured ~473–477 B/witness ⇒ ≈74 GB at 64-GPU production), not matching throughput. The single-threadedness, the `table_` ownership, and the `--lp1_max_witnesses` note below all remain true as stated (line refs have since drifted: the not-thread-safe contract is `cpu_lp.h:38-39`, `table_` is `cpu_lp.h:82`). Original text:
  **Coordinator single-threaded CPU LP-matching ceiling:** `CPULargePrimeTable::insertAndMatch` (`cpu_lp.h`/`cpu_lp.cu`) is explicitly not thread-safe (`cpu_lp.h:39`) and owns a single `std::unordered_map<uint64_t, PartialRelation> table_` (`cpu_lp.h:78`) — all cross-node LP matching runs on one coordinator CPU thread regardless of GPU count. Empirical throughput ceiling **~3,200–3,500 witnesses/s**: 32 GPU/L=200T sustains 1,731.8 witnesses/s cleanly, but 64 GPU (production scale) already shows ~8.4% sub-linear scaling off the linear projection. A **108-GPU (27-node) scale-up was evaluated and REJECTED** on this basis (2026-07-15) — it would offer ~6,060 witnesses/s, but the coordinator saturates well before that, wasting ~44 GPUs. Host RAM is the other constraint: ~110–120M accumulated witnesses ≈ ~35–50 GB coordinator RAM; **`--lp1_max_witnesses` does not bound `table_`** (it sizes the solo-mode GPU `LargePrimeVariant` capacity instead) — only lowering the LP bound `L` shrinks the coordinator's table. Future lever: hash-shard the witness table, or pool-allocate `PartialRelation` instead of its 2 per-entry `std::vector`s.
