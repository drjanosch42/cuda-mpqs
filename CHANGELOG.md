# Changelog

All notable changes to cuda-mpqs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-06-22

cuda-mpqs 1.0.3 consolidates the multi-node cluster correctness and performance
work that landed since 1.0.2. It is validated end-to-end by an RSA-130
factorization on 2 nodes × 4 A100-SXM4-40GB (8 GPUs) on the PC2 cluster: at
L=13 T it ran clean with ~51.8 % large-prime fraction and ~44.9 % per-solution
square-root success (well clear of the high-LP 2-cycle cliff), with the
cross-node overflow-chunk overlap eliminated (byte-identical duplicate-partial
drops reduced to a negligible residual) and the coordinator GPU kept at full
duty for the whole run.

### Fixed
- Cluster overflow-chunk **overlap** (initial-range `CHUNK_COMPLETE` splitting a
  live overflow chunk). The coordinator's `CHUNK_COMPLETE` handler called
  `reclaimPartial(cc.chunk_id, …)` unconditionally. A worker's *first*
  `CHUNK_COMPLETE` reports its initial `WORK_ASSIGN` range, which it labels
  `chunk_id=0` (worker-side default); pool overflow ids also start at 0
  (`WorkPool::next_chunk_id_`). An initial-range completion arriving while pool
  chunk 0 was still in flight therefore aliased it: the reclaim split the live
  chunk and pushed a bogus remainder onto `WorkPool::returned_`, which the next
  overflow checkout served as an *overlapping* chunk. The symptom on a multi-node
  run is two GPUs sieving the same a-range: one node delivers only a fraction of
  the relations it sieves and the byte-identical duplicate partials are dropped by
  the `sqrt_Q` identity guard (correctness held, but roughly one GPU's worth of
  sieving was wasted).
  Fix: gate the reclaim on the coordinator's authoritative tracker — reclaim only
  the pool chunk the coordinator believes the worker holds (`current_chunk_id ==
  cc.chunk_id`, not `kInitialChunk`/idle), mirroring the heartbeat-reclaim and
  `CHUNK_REQUEST` guards. A cheap dispatch-time overlap invariant in
  `assignChunkTo` now warns (never aborts) on any future overlap regression.
  (`src/cluster/direct_channel.h`, `src/orchestrator/orchestrator.cpp`)
- Cluster overflow `CHUNK_ASSIGN` delivery race: workers no longer strand
  permanently when an overflow chunk assignment is lost. The coordinator
  discarded the boolean result of the overflow `CHUNK_ASSIGN` send, then
  unconditionally logged "Assigned" and marked the chunk in flight; a
  failed/lost send left the worker idle to its 600 s hard cap while the
  coordinator only re-handed the reclaimed chunk on a `CHUNK_COMPLETE` the
  stranded worker never sent. On a large multi-node run this cascaded into
  several-percent relation under-delivery. Five-part fix (P1-P5): **(P1)** all
  overflow `CHUNK_ASSIGN` sends route through one send-checked routine
  (`assignChunkTo`); on a `false` send the just-checked-out chunk is returned
  to the assignable pool (new `WorkPool::returnChunk`,
  `src/cluster/work_pool.{h,cpp}`), a counter is bumped, and the worker is left
  with no current chunk — `current_chunk_id` is set only on a confirmed send.
  **(P2)** new stats-free message type `CHUNK_REQUEST` (0x15,
  `src/cluster/cluster_common.h`): after a grace period an idle worker
  re-requests work at most once / 15 s on a socket-alive timeout, and the
  coordinator services it via `assignChunkTo()` *without* recording
  relation/partial stats, so a retry never double-counts. **(P3)** a
  coordinator proactive sweep, each 5 s timeout tick, hands a chunk to every
  alive worker that holds no chunk while assignable work remains and the target
  is unmet, draining reclaimed/returned chunks even without a re-request.
  **(P4)** the 600 s worker hard cap is now an absolute backstop that only
  fires on persistent timeout (heartbeats flowing, coordinator silent); a
  closed socket still exits immediately. **(P5)** `failed_assign_sends` /
  `redispatched_chunks` telemetry counters surfaced in the cluster sieve
  summary. `assignChunkTo()` is idempotent (no-op when the worker already holds
  a chunk; sets `current_chunk_id` only on a confirmed send) so P2 and P3 can
  never double-assign. Solo and single-node behaviour are unaffected (no
  overflow pool, no remote workers). (`src/cluster/work_pool.{h,cpp}`,
  `src/cluster/cluster_common.h`, `src/orchestrator/orchestrator.cpp`)
- Worker mid-sieve `CHUNK_ASSIGN` recv race (true root cause of the residual
  deadlock after the P1-P5 set). The worker socket had **two concurrent
  readers**: the `AsyncNetworkDataTap` I/O thread and the main thread's
  chunk-wait loop both called `recv`/`recvBlocking` on the shared coordinator
  socket with no synchronization. The I/O thread's tight `recv()` loop won the
  race and silently discarded any mid-sieve `CHUNK_ASSIGN` frame (it handled
  only STOP/ERROR/RECALL), so an overflow chunk never reached the worker. Fix:
  the I/O thread is now the **sole** socket reader — it captures `CHUNK_ASSIGN`
  into a mutex-guarded slot (`pending_assign_`) that the main thread takes via
  the new `tryTakeChunkAssign()`, and the chunk-wait loop no longer touches the
  socket (it polls the tap for STOP and `CHUNK_ASSIGN`). This also removes the
  `recv_buf_` framing-corruption hazard from interleaved reads.
  (`src/cluster/async_network_data_tap.{h,cpp}`,
  `src/orchestrator/orchestrator.cpp`)
- Re-request wedge: a `CHUNK_REQUEST` was being blocked by `assignChunkTo()`'s
  own idempotency guard. Once P3 re-dispatched a chunk (confirmed send →
  `current_chunk_id` set) that the worker never received (the recv race above),
  every subsequent re-request hit the `current_chunk_id != UINT32_MAX` guard and
  was dropped, wedging the worker permanently. A `CHUNK_REQUEST` is explicit
  proof the worker is idle, so the handler now **forces** re-assignment: it
  returns any chunk the coordinator believes is in flight back to the pool and
  clears `current_chunk_id` before calling `assignChunkTo()`.
  (`src/orchestrator/orchestrator.cpp`)
- Startup over-assignment by the P3 sweep: the initial `WORK_ASSIGN` range is
  not a pool chunk and left `current_chunk_id == UINT32_MAX`, so the sweep handed
  a second (overflow) chunk to a worker still sieving its initial range. Workers
  now start with a `kInitialChunk` sentinel (`UINT32_MAX - 1`, "busy with initial
  range"), cleared to `UINT32_MAX` only on the first `CHUNK_COMPLETE`; the sweep,
  the heartbeat-timeout reclaim, and the `CHUNK_REQUEST` handler all treat the
  sentinel as "not a pool chunk" (no spurious `returnChunk`/reclaim).
  (`src/orchestrator/orchestrator.cpp`)
- Cluster a-value (polynomial) overflow-pool sizing: the coordinator's
  on-demand overflow pool of polynomial a-value windows was sized far too small
  and ran dry long before the relation target. The size derived from the
  scheduler's `w_est` (whose `5 * H` divisor assumes ~5 relations per a-index —
  wildly optimistic for large composites, where measured yield is an order of
  magnitude lower) combined with `overflow_size = overflow_start` (overflow =
  1× the initial contiguous total). On a large composite the pool drained at
  ~15 % of the relation target, workers idled out the 600 s timeout, and the run
  died with the matrix stage aborting on insufficient relations. Fix
  (coordinator branch, `src/orchestrator/orchestrator.cpp`): the **overflow**
  pool is now sized from the relation target at a conservative per-a-index yield
  floor (0.04 rel/a-index, several times below measured), times the new
  `--cluster_pool_oversize` multiplier, clamped to the a-factor walk's capacity
  `(fb_size - 150) / 2` windows so it can never map to invalid a-coefficients.
  Overflow windows are pure on-demand index space drawn only after a node
  exhausts its initial range with the target still unmet, and the run still
  stops at the relation cap — so over-provisioning is free. Initial contiguous
  ranges (`computeContiguousRanges` / `w_est`), solo mode, and the default-inert
  debug repro harness are untouched. The coordinator now logs an auditable
  `a-value pool: contiguous=… overflow=… total=… windows (oversize=…x,
  target=… rels, walk_ceiling=… windows)` line at startup.
  (`include/orchestrator.h`, `src/orchestrator/orchestrator.cpp`,
  `tests/cuda-mpqs.cpp`)
- Cluster duplicate-partial corruption (trivial-sqrt failure on multi-node runs).
  Multi-node runs produced large fractions of byte-identical duplicate partials,
  so almost every cross-node LP combine multiplied a relation by its own duplicate
  → a perfect square → trivial sqrt (`X ≡ ±Y`). Three compounding defects fixed:
  (1) the CUDA-graph replay loop advances `cuda_graph_unroll × sieve_batch_size`
  a-values per replay but invoked the DataTap callback once per replay, so the
  per-worker a-range guard under-counted by the `cuda_graph_unroll` factor and
  nodes overran into each other's a-ranges. `onBatchComplete` now carries the true
  a-values-consumed count (`src/cluster/data_tap.h`,
  `src/cluster/async_network_data_tap.{h,cpp}`, graph call site in
  `src/orchestrator/orchestrator.cpp`). (2) The coordinator's own (node-0) local
  sieve runs through `DirectChannel`, which had no a-range bound, and node-0's
  `poly_range_start == 0` skips the worker-only `resetAndAdvanceTo` path — so it
  sieved unbounded from a_index 0 into worker ranges. `DirectChannel` gained a
  `setRange`/range-checked `shouldStop`, and node-0 is now bounded to `ranges[0]`
  (`src/cluster/direct_channel.h`, `src/orchestrator/orchestrator.cpp`). (3) The
  CPU LP matcher (`combinePartials`) keyed only on the large prime and would
  multiply a partial by an identical copy of itself; it now drops the match when
  the two partials share the same `sqrt_Q` (with a `dup_dropped` telemetry counter)
  (`src/cluster/cpu_lp.{h,cu}`, `src/orchestrator/orchestrator.cpp`). The GPU LP
  matcher (`src/largeprimes/largeprime.cu`) runs only in solo/single-node mode
  (no duplicates) so it gets a documented TODO rather than an in-kernel guard.
  Solo, single-node, and legacy-sieve behaviour are unchanged
  (`range_a_limit_ == 0` / `cuda_graph_unroll ≤ 1` preserve the original paths).
- `--sieve_only` now honours `--autotune_stage1`. Autotune was mode-gated to
  `FULL_PIPELINE` (and `AUTOTUNE_ONLY`) only, so cluster coordinators — and solo
  sieve-only runs — launched with `--sieve_only --autotune_stage1` skipped kernel
  autotuning entirely. With `config_.useParams` left false the sieve fell back to
  the memory-bound `loadStandardConfig()` geometry (num_polys=8192, ~8.6 GB global
  bucket) instead of the efficient `loadPartialCustomConfig()` (num_polys≈512) that
  `FULL_PIPELINE` workers receive. `SIEVE_ONLY` now tunes its sieve geometry like
  `FULL_PIPELINE`. (`src/orchestrator/orchestrator.cpp`)
- Cluster workers no longer drop out permanently while waiting for an overflow
  chunk. After the duplicate-partial fix workers finish their assigned a-ranges
  much faster and request overflow chunks far more often; the worker's wait for
  the next `CHUNK_ASSIGN`/`STOP` used a single 30 s blocking receive that, on
  timeout, exited the worker for good. When the coordinator's Thread A is busy
  CPU-LP-matching a large witness table it can take longer than 30 s to assign a
  chunk, so workers progressively went `DEAD` over a long run. The wait is now a
  retry loop that polls in `kHeartbeatIntervalMs` (5 s) slices and keeps the
  worker alive until a real message arrives, only giving up after a 600 s hard
  deadline (coordinator truly gone). Liveness during the wait is maintained by
  the existing `AsyncNetworkDataTap` I/O thread, which is still running and emits
  a `HEARTBEAT` every 5 s — well under the coordinator's 120 s
  (`kFlushTimeoutMs`) dead-threshold — so no second concurrent writer is added to
  the worker socket. The loop also honours a `STOP` intercepted by the I/O thread
  off the shared socket. Solo and coordinator paths are untouched.
  (`src/orchestrator/orchestrator.cpp`)
- Coordinator chunk-assignment latency no longer scales with the CPU-LP backlog,
  so workers stay fed and GPUs stay busy. Thread A's recv-drain processed messages
  strictly in TCP receive order, and since each worker ships many `INCREMENTAL_BATCH`
  messages (each triggering a synchronous `insertAndMatch` over the witness table)
  ahead of its `CHUNK_COMPLETE` on the same in-order socket, a `CHUNK_COMPLETE` was
  queued behind the entire preceding partials/LP-match backlog — delaying the cheap
  O(1) `CHUNK_ASSIGN` response and idling the worker (which, post the chunk-wait fix,
  no longer dies but sits waiting). The drain now services control messages first —
  computing and sending `CHUNK_ASSIGN` immediately on `CHUNK_COMPLETE` — and defers
  the heavy `INCREMENTAL_BATCH` LP-match work to a post-drain pass that replays the
  deferred batches in strict FIFO receive order. The `insertAndMatch` call sequence,
  accumulator/LP-table state, and dedup (`dup_dropped`/Fix-3) data path are therefore
  byte-identical to before; only the *scheduling* relative to chunk assignment changes
  (the chunk pool / scheduler touch disjoint state from the partial→LP→accumulator
  path). Solo and single-node behaviour are unaffected (no remote workers → no
  `INCREMENTAL_BATCH` deferral, no `CHUNK_COMPLETE`). (`src/orchestrator/orchestrator.cpp`)
- Worker chunk-wait loop no longer spins and floods its log when the coordinator
  closes the connection (disk-blowup regression: an unbounded spin could write tens
  of GB to each worker's log and exhaust available disk). The chunk-wait retry loop logged
  one INFO line per `recvBlocking` false return on the assumption that false always
  meant a full 5 s poll timeout. But the worker-side `recvBlocking` →
  `TcpSocket::recvMsg` → `::recv()` returns immediately when the socket is closed
  (recv()==0 on a FIN'd socket) or errored, collapsing EOF/error and timeout to the
  same bare `false`; when the coordinator dropped the connection (target reached,
  shutdown, chunk pool exhausted, any drop) the loop became a tight spin emitting the
  "still awaiting CHUNK_ASSIGN/STOP" INFO line continuously for the full 600 s hard
  cap, ballooning each per-worker `.out`. Two fixes: (1) `recvMsg`/`recvBlocking` now
  report *why* they returned via an optional `RecvStatus {GOT_MSG,TIMEOUT,CLOSED}`
  out-param (recv()==0 / real errno ⇒ CLOSED; EAGAIN/EWOULDBLOCK after `SO_RCVTIMEO`
  ⇒ TIMEOUT; framing/CRC desync ⇒ TIMEOUT, socket still alive), and the chunk-wait
  loop exits promptly with a single warning on CLOSED instead of spinning. (2)
  Defense-in-depth: the loop now floors each iteration to ~`kHeartbeatIntervalMs` (5 s)
  by sleeping the remainder on a fast false return, and rate-limits the "still
  awaiting" INFO log to ≤ once per 30 s — so no return path can flood the log. The
  busy-coordinator survival behaviour is preserved (a genuinely busy coordinator still
  keeps the worker alive to the 600 s cap). Solo and coordinator (epoll) paths are
  unchanged. (`src/cluster/comm_backend.h`, `src/cluster/tcp_transport.{h,cpp}`,
  `src/cluster/tcp_backend.{h,cpp}`, `src/orchestrator/orchestrator.cpp`)

### Changed
- Cluster coordinator now **self-assigns overflow chunks** so its local GPU keeps
  sieving instead of idling after its initial a-range. The duplicate-partial fix
  hard-bounds the coordinator's node-0 sieve to its initial range to prevent
  cross-node a-range overrun, but nothing wired node 0 into the overflow rotation
  that keeps remote workers busy, so the coordinator GPU could sit idle for most
  of a long run (≈ 7.1 of 8 effective GPUs on an 8-GPU cluster). After its initial
  bounded sieve, the coordinator's GPU thread now loops as one more consumer of the
  same mutex-guarded `cluster_work_pool_` that feeds workers — drawing overflow
  chunks from the *local* pool (disjoint from every worker's by construction),
  jumping the siever to each chunk's absolute a-index, sieving, then reclaiming —
  until the target is reached / stop is signalled / the pool drains. It never
  traverses the network `CHUNK_COMPLETE`/`worker_trackers` path, so it cannot
  reintroduce the chunk_id aliasing fixed above. Engaged only with remote workers
  present; solo and worker paths are byte-for-byte unchanged. Expected ≈ +12.7 %
  cluster sieve throughput on an 8-GPU run (grows with run length). A new
  `DirectChannel::stopped()` reports the explicit stop signal distinct from
  per-chunk a-range exhaustion. (`src/cluster/direct_channel.h`,
  `src/orchestrator/orchestrator.cpp`)

### Added
- `--cluster_pool_oversize <float>` (coordinator only, default 1.0): a-value
  overflow-pool over-provisioning multiplier. `>1` enlarges the on-demand
  overflow pool of polynomial windows so it cannot run dry before the relation
  target is met. Because overflow windows are drawn only on demand and the run
  stops at the relation cap, over-sizing is essentially free; it does not affect
  the initial contiguous ranges or single-node/solo behaviour.
- Default-inert cluster chunk-delivery repro harness (regression test): two
  debug hooks that deterministically reproduce the overflow `CHUNK_ASSIGN`
  delivery race — `MPQS_DEBUG_DROP_CHUNK_ASSIGN=<worker>:<occurrence>` injects a
  targeted lost send, and `MPQS_DEBUG_MAX_CHUNK_WINDOWS=<W>` caps every computed
  chunk to `W` hypercube windows to force many `CHUNK_COMPLETE` → `CHUNK_ASSIGN`
  turnover cycles. Both are parsed once and are identity/no-op when unset
  (byte-identical default behaviour). (`src/cluster/chunk_scheduler.{h,cpp}`,
  `src/orchestrator/orchestrator.cpp`)
- Overflow-chunk validation mode for multi-node hardware regression testing:
  pinning `MPQS_DEBUG_MAX_CHUNK_WINDOWS=1` (one hypercube window per chunk) forces
  a fresh `CHUNK_COMPLETE` → `CHUNK_ASSIGN` turnover on every assigned chunk,
  maximising overlap-regression coverage; the pass criterion is `dup_dropped == 0`
  with contiguously tiled chunks.

## [1.0.2] - 2026-06-09
### Fixed
- Matrix preprocessing (CPU `--matrix_mode preprocess`) no longer silently returns
  only trivial congruences (X ≡ ±Y, no factor). Three independent defects fixed:
  - **Higher-weight merge** column elimination is now all-or-nothing: the pivot row
    was previously deleted even when a fill-in skip left the column un-eliminated,
    making the reduction non-kernel-preserving. (`src/matrix/merge_filter.cpp`)
  - **Truncation** is now size-gated — skipped when the reduced matrix is already
    Block-Wiedemann-tractable, avoiding the lightest-row selection that confined the
    kernel to the trivial subspace. (`--truncation_min_rows`, default 5,000,000)
  - **High-LP partial handling**: raw single-large-prime relations are no longer
    materialized as 2-cycle matrix rows at high large-prime fraction (which captured
    the genus character and forced every dependency trivial); gated by the
    combined-smooth LP fraction. (`--preprocess_lp_materialize_max`, default 0.45)

### Added
- `--truncation_min_rows` and `--preprocess_lp_materialize_max` — the facet-2 and
  facet-3 preprocessing fix controls (above).
- `--merge_max_weight` and `--force_preprocess` — DIAGNOSTIC flags for the
  preprocessing investigation; default-inert (no effect on normal runs).
- Diagnostic tooling under `tools/preprocessing_analysis/` (relation / genus
  verifiers used to characterise and validate the fixes).

### Notes
- Preprocessing at high LP fraction is now **correct** but remains **≤ legacy** in
  yield; prefer legacy / keep the large-prime bound below the cliff at high LP. The
  default `--matrix_mode` AUTO → legacy is unchanged. Validated: RSA-100 factored
  (235 s); RSA-100 / 94-digit / RSA-110 preprocess all factor.

## [1.0.1] - 2026-06-05
### Added
- `--char_mode {norm,branch,none}` — selectable quadratic character-column symbol:
  `branch` = correct branch-fixed field-element character (fixed Tonelli root),
  `norm` = legacy genus-blind NORM symbol, `none` = zero character columns.
- 64-bit number-theory primitives (Tonelli–Shanks, Jacobi, deterministic
  Miller–Rabin primality) for large-prime-bound auxiliary primes.
- `--sqrt_diagnostic`: per-solution reporting of the nontrivial-GCD rate per
  Block-Wiedemann solution (debug level; capture with `--log_file`).
- `relation_validator` tool — deterministic relation + large-prime validation
  (composite / out-of-range large-prime detection).

### Changed
- Default `--char_mode` is now **`none`** (character columns off by default).
  Validation found character columns confer no demonstrable factoring benefit at
  reachable scales — inert where the quadratic sieve is unobstructed (RSA-100/110),
  insufficient where it is (the 94-digit 2-cycle cliff).
- `--matrix_mode` AUTO now resolves to **legacy** for normal runs; preprocessing
  engages only via explicit `--matrix_mode preprocess` or `--matrix_only`. The
  automatic LP-fraction → preprocess switch was removed because preprocessing
  degrades the obstructed high-LP regime.
- Sieve telemetry: the per-LP-call "Combined N full relations in this batch."
  line is demoted from `--verbose` (STATS) to `--debug` (DEBUG_1). `--verbose`
  now shows only the consolidated ETA / throughput / buffer-fill telemetry during
  sieving, with the detached technical large-prime combination counts at `--debug`.

### Deprecated
- `--lp_preprocess_threshold` (and alias `--lp_matrix_threshold`) — now inert;
  AUTO no longer auto-selects preprocess from LP fraction.

### Fixed
- GPU sieve no longer drops relations whose candidate falls exactly on a
  sieving-block boundary: the backward (factor-recording) pass now scans the
  same position range as the forward sieve, so boundary candidates get their
  full prime factorization recorded instead of being silently discarded.

## [1.0.0] - 2026-06-01
### Added
- Initial public release.
- Full SIQS/MPQS pipeline: parameter tuning, optional autotuning, GPU sieving,
  sparse GF(2) matrix construction, Block Wiedemann linear algebra, and square
  root refinement.
- Single-node and distributed cluster execution across heterogeneous multi-node
  setups (explicitly tested with more than two nodes).
- Autotuning: 4-stage joint (F, L) optimizer with persistent history.
- GPU preprocessing: packed sparse GF(2) matrix with compact-merge cycles.
- Large-prime variant: single large prime via GPU slab hash table.
- Supports NVIDIA Turing, Ampere, Hopper, and Blackwell GPU architectures,
  including the A100 (Ampere) and H100 (Hopper) data-center accelerators.
- Supports the Jetson Orin Nano Super 8 GB embedded platform.
