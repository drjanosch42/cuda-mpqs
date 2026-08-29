# Postprocessing Module (`src/postprocessing/`)

Completes sieve candidates into full SIQS relations via GPU trial division,
classifies them (full vs. 1-partial), accumulates into SoA buffers, deduplicates,
and adaptively predicts sieve convergence.

Two execution paths exist, matching the sieve module's mode selection (`sieve_batch_size`):
- **Legacy path** (`sieve_batch_size == 0`): host-driven accumulate/process/consolidate loop.
- **Batch path** (`sieve_batch_size > 0`): zero-sync double-buffered factorization writing directly to persistent storage.

Large prime (LP) classification is supported in both paths — when `lp1_bound > 0`, candidates with cofactor < `lp1_bound` are classified as 1-partials and written to `d_partial_batch`.

Namespace: `mpqs::postprocessing`. CMake target: `mpqs_postproc`.

## Files

| File                | Lines | Purpose |
|---------------------|------:|---------|
| `postprocessing.h`  |  ~420 | `DevicePostProcessingController` class, `PostProcConfig`, `PredictionResult`, `BufferFillSnapshot`/`BufferFillHistory`/`LPFillHistory`, public API |
| `postprocessing.cu` | ~1665 | CUDA kernels, buffer management, deduplication, adaptive prediction, buffer-fill telemetry, controller implementation |
| `CMakeLists.txt`    |  ~200 | Build config (standalone + integrated), links `mpqs_common`, `mpqs_sieve` |

## PredictionResult

GPU-written adaptive convergence state. Allocated as `cudaHostAllocMapped` so the
host can poll without any CPU/GPU synchronization.

| Field | Type | Description |
|-------|------|-------------|
| `should_terminate` | `volatile uint32_t` | 1 = stop launching sieve batches |
| `effective_R` | `uint32_t` | Current persistent relation count R (LP matches already included) |
| `yield_rate` | `float` | λ = R / total_steps (relations per step) |
| `lp_match_rate` | `float` | μ = LP_full / (LP_full + witnesses) |

## BufferFillSnapshot / Histories

`BufferFillSnapshot` is a GPU-written, mapped-pinned fill snapshot of the three
major buffers (accumulation, partial, persistent) with a `volatile uint64_t
generation` ticket for lock-free reads (same pattern as `SLPPinnedStats`);
published by `publish_buffer_fill_kernel` after each batch factorization and read
via `getBufferFillSnapshot()`. The host-side `BufferFillHistory` (max/avg
accumulators) and `LPFillHistory` (witness fill + cumulative overflow counters)
are updated by the orchestrator each time the generation advances, feeding the
periodic buffer-telemetry log lines and near-full warnings.

## PostProcConfig

| Field | Type | Description |
|-------|------|-------------|
| `accumulate_buffer_size` | `uint32_t` | Dense candidate buffer capacity (header recommends 65536/131072; CLI **`--accum_buf_size`**, whose `0 = auto` derivation is `max(4096, sieve_batch_size × 2048)` — `orchestrator.cpp:3327-3330` — further reduced to `pow2(max(4096, 4·target_relations))` when `target_relations < 16384`, `:3342-3353`. The "524K default" in the source comment at `:3340` describes a superseded formula). **Note (1.0.6):** with post-processing captured inside the CUDA graph (`--cuda_graph_capture postproc\|full`) trial division runs **per batch** instead of once per `graph_N` batches, so peak occupancy falls by roughly `graph_N`× (measured RSA-100, `graph_N = 4`: peak 8,025 → 2,093 of 16,384) and the near-full/contention warning no longer appears at the previous settings. ⚠ **Do NOT score the consequence as a regression:** candidates that used to be dropped on a full accumulation buffer (`expandAndAccumulateKernel`'s reserve + `clampCounterKernel`) now survive, so a run may legitimately yield **more** relations at identical settings. Solo overshoot with the per-replay host sync removed measured **+0.33 %**, well inside `relation_cap`. The sizing guidance above predates this and may be revisited. |
| `accumulate_batch_purge_threshold` | `uint32_t` | Triggers factorization (orchestrator sets 0.8 × buffer size) |
| `persistent_device_buffer_size` | `uint32_t` | Long-term device relation storage; 0 = disabled. CLI **`--persistent_buf_size`**; `0 = auto` ⇒ `target_relations × 2 + accumulate_buffer_size` (`orchestrator.cpp:3359-3363`) |
| `lp1_bound` | `uint64_t` | Single large prime bound; 0 = disabled |
| `partial_buffer_size` | `uint32_t` | 1-partial buffer capacity; CLI **`--partial_buf_size`**, defaults to `accumulate_buffer_size` if 0 and is clamped up to it if smaller (`orchestrator.cpp:3369-3378`) |
| `shc_dim` | `uint32_t` | Hypercube dimension (number of prime factors in `a`) |
| `device_id` | `int` | CUDA device index |
| `char_branch` | `bool` | When true (`--char_mode branch`), capture the r-bit branch character vector at relation birth. Default false: char-bit computation is fully gated off (zero hot-path cost). |
| `char_r` | `uint32_t` | Number of branch aux primes (`r ≤ 32`); valid only when `char_branch` |
| `d_char_q` | `const uint64_t*` | Device array of r branch aux primes `q_s` (64-bit, `> lp1_bound`) |
| `d_char_t` | `const uint64_t*` | Device array of r fixed Tonelli roots `t_s` (`t_s² == N mod q_s`) |

## Kernels

### `expandAndAccumulateKernel`

Compacts sparse `candidateRelation` into `DenseCandidate`. Per-thread:
1. Guards against overflow: skips if `num_factors > 32` (checked before reserving a slot).
2. Atomic append to reserve slot in accumulation buffer.
3. Insertion-sorts sieve hint factors in-register (O(N) when nearly sorted).
4. Two-pointer merge of sorted sieve factors with `a`-factors into `factor_indices[48]`.
5. Counter is clamped by `clampCounterKernel`.

### `processCandidate` (device function, shared trial-division core)

Both factorization kernels call this per-candidate pipeline (`postprocessing.cu:294-424`):
1. Compute `sqrt_Q = |ax+b|` and the sign of the signed `(ax+b)` via the 5-arg `calculate_sqrt_of_QX()`.
2. Compute `Q = |(ax+b)² − N|` and its sign via `mpqs::abs_square_minus_N()` (`:323`) — the square is formed in a **1024-bit accumulator** because it reaches ~2N ~ 2^513 for N ≥ 2^511 (RSA-155); bit-for-bit identical to the legacy truncating path for N < 2^511.
3. Factor out powers of 2 (`val_2_exp = countr_zero(Q)`); right-shift Q.
4. Trial divide by each sieve hint factor index from the factor base (`:334-351`); record `(fb_idx, exponent)` pairs (up to 64 entries). **Only the candidate's hint list is divided — not the full FB** (a dropped small prime therefore stays hidden in the LP remainder; see the acceptance-gate caveat below).
5. Classify (`:353-375`): full (`Q == 1`) or 1-partial (`Q` fits 128 bits and `remainder < cfg.lp1_bound`, `:371`). **Known issue:** this gate has no primality test and no `remainder > fb_max` check, so composite or in-FB-range remainders are accepted as "large primes" (relation-hygiene only — composite LPs never reach the matrix; see the sqrt-failure sieve-bug reproduction notes).
6. Warp-aggregated dual append via `append_to_soa()` to `full_view` and `partial_view`.

### `batchedFactorizationKernel`

Legacy-path wrapper: reads the candidate count from a host parameter, calls `processCandidate` per thread.

### `batchedBatchFactorizationKernelSoA`

Identical factorization logic but reads the candidate count from a device pointer
(`*d_count`) instead of a host-provided parameter. Enables the batch sieve loop to
launch factorization with zero CPU/GPU synchronization. Writes full relations
directly to `d_persistent_batch`; writes partials to `d_partial_batch` when LP is
enabled.

### `append_to_soa()` (device function)

Warp-level aggregated SoA append:
1. `__ballot_sync` — identify threads with valid data.
2. Kogge-Stone intra-warp inclusive prefix sum over per-thread factor counts.
3. Leader lane (lowest active lane) performs atomic CAS on packed dual counter `(R<<32 | F)`.
4. Cap check: if `R >= view.target_cap`, the entire warp is discarded (sentinel `UINT64_MAX`).
5. `__shfl_sync` broadcasts base offsets to all lanes.
6. Each thread scatter-writes its relation scalars (`sqrt_Q`, `sign`, `val_2_exp`,
   `large_prime_remainder`, `char_bits`) and factor arrays (indices + counts) into CSR layout.

### Branch Character-Bit Capture (`--char_mode branch`)

When `PostProcConfig::char_branch` is set, `processCandidate` computes the r-bit branch character
vector for every **emitted** relation (full smooth or valid 1-partial — never for discards,
`postprocessing.cu:383-404`). For each of the `char_r` aux primes `q_s` it recovers the signed
`(ax+b) mod q_s` (via `sqrt_Q.mod_uint64(q_s)` and the sign of `ax+b`) and evaluates
`mpqs::matrix::branchCharBit(axb_mod_q, t_s, q_s)`, packing bit `s` into `view.char_bits[i]`.
Because the vector depends only on `(ax+b)`, the same value feeds both the smooth and partial
append. Under the default `--char_mode none`/`norm` the entire block is skipped and `char_bits` is
written as a defined 0 (zero added hot-path arithmetic). The captured vector is deliberately
**excluded** from the dedup hash (below) and is propagated by XOR through LP combination in the
[largeprimes module](largeprimes.md) and through the matrix-reduction merges (see
[matrix.md](matrix.md)). Persisted per relation in `HostRelationBatch::char_bits` and the
`relation_io` v2 format (see [common.md](common.md)).

### Deduplication kernels

| Kernel | Purpose |
|--------|---------|
| `compute_relation_hashes_soa` (`postprocessing.cu:509`) | 64-bit hash: `[63:48]` num_factors, `[47:32]` XOR of exponents, `[31:0]` XOR(factor_idx × 0x9e3779b9) ^ sign·2^val_2_exp. **The sign term genuinely contributes** (`:543-548`): sign is extracted via the encoding-agnostic "negative iff `signs[idx] != 1`" rule (the old `!= 0` check fired always). `char_bits` is **excluded** (it is a deterministic function of `(ax+b)`, so two relations with the same factorization carry identical char bits and must still dedup). The identical host-side formula lives in `src/common/relation_hash.h::computeRelationHash` (single source of truth for the cluster accumulator and checkpoint dedup). |
| `compute_new_lengths_kernel` (`postprocessing.cu:602`) | Computes per-survivor factor lengths for exclusive scan. **Emits `uint32_t`** — see the open defect below |
| `gather_soa_relations_kernel` | Compacts SoA by copying survivors (scalars incl. `char_bits` + CSR segments) into a fresh batch |

#### ⚠ OPEN DEFECT — uint32 length underflow in post-dedup compaction (found 2026-08-22, **still open in v1.0.6**)

`compute_new_lengths_kernel` (`postprocessing.cu:602-613`) writes
`out_lengths[idx] = (uint32_t)(old_offsets[old_idx+1] - old_offsets[old_idx])` into a
`thrust::device_vector<uint32_t>` (`:1412`). If the CSR offsets are ever non-monotonic for the last
surviving row, that 64-bit subtraction underflows and is then **truncated to 32 bits**. In
`deduplicatePersistentBatch()` (`:1273-1491`) the tail is read back as a `uint32_t`
(`uint32_t last_len;`, `:1435`), summed as `uint64_t total_new_factors = last_offset + last_len`
(`:1444`), and handed **straight to** `d_clean_batch->resize(new_size, total_new_factors)` (`:1454`)
with no capacity or sanity check. The subsequent `gather_soa_relations_kernel` (`:1467`) then reads
and writes out of bounds; the illegal access surfaces at the next synchronising `CUDA_CHECK`
(`cudaStreamSynchronize`, `:1487`), followed by a `thrust` `cudaErrorIllegalAddress` on free.

**Trigger precondition:** a run of `[RelationBatch] Append DISCARDING data. Buffer Full. Added 0/N
relations.` — i.e. the LP relation buffer saturates and appends are dropped, leaving
`compute_new_lengths_kernel` a wrapped negative length for the last surviving row. That discard path
**is already logged** at `LOG_WARNING` (`src/common/mpqs_soa.cu:375`, the only `DISCARDING` site in
the tree; unchanged since before v1.0.5), so the precondition is **observable, not silent** — the
warning is the signal to watch for. *(Corrected 2026-08-25: the earlier characterisation of this
path as "silent" is **superseded**.)*

**Observed once, deterministically**, on an A100 under `ncu` serialization:
`Total new factors: 4294967272` (= 2³² − 24) → `resize(…, 4294967272)` →
`CUDA Error: an illegal memory access was encountered`, app `rc=6`, no factors. It is **a latent
defect in the normal LP path, not a profiling artefact** — ncu only exposed it, because its 46
replay passes slowed the sieve until the buffer saturated.

**Status: STILL OPEN in v1.0.6.** The arithmetic above is unchanged. v1.0.6 makes the precondition
materially harder to reach — the pinned index-staging fix cut raw over-collection from 9.22 % to
0.34 %, so the LP buffer no longer saturates under ncu replay, and the re-run under the
byte-identical launcher saw zero `DISCARDING` lines and every ncu pass `rc=0` — but it **does not
fix the defect**. Absence in one run is not proof it cannot recur; the `Append DISCARDING` path is a
normal LP code path and is production-reachable.

**What remains unfixed is purely the uint32 arithmetic** in the compaction (`:1435` / `:1444` /
`:1454`). **Fix direction (not applied):** compute the lengths in `uint64_t` (or assert
`offsets[i+1] >= offsets[i]` in the kernel), and treat a `total_new_factors` above the batch's
capacity as a **hard error** rather than a `resize()` request.

### Yield prediction kernel

| Kernel | Purpose |
|--------|---------|
| `yield_prediction_kernel` | Single-thread kernel: reads `global_count` and optional `SLPPinnedStats`, computes λ and μ, sets `should_terminate = 1` when `effective_R ≥ target + target/20` (5% margin). Writes to mapped pinned `PredictionResult`. **`total_steps` is a `const uint64_t*` device pointer** (1.0.6), not a by-value host scalar: kernel arguments are baked at CUDA-graph capture time, so a host scalar would freeze and every replay would compute a wrong λ. Still `uint64_t` — RSA-140-scale a-value counts exceed 2^32. |
| `advance_steps_kernel` | Single-thread kernel enqueued **after** the yield-prediction read in every batch: `*steps += delta` with `delta = sieve_batch_size`. Read-then-advance keeps `total_steps` equal to the a-values launched *before* the batch, matching the pre-v1.0.6 host semantics. The host seeds the counter once per sieve leg via `seedPredictionSteps()`, outside any capture; graph and non-graph paths enqueue the same nodes. |

### Utility kernels

| Kernel | Purpose |
|--------|---------|
| `clampCounterKernel` | Caps atomic counter at buffer capacity |
| `commit_dual_counter_kernel` | Unpacks `(R,F)` from dual counter into separate `global_count` / `global_factor_idx` (assignment, not addition — used after each batch) |
| `sync_dual_counter_from_batch_kernel` | Inverse of commit: packs current `global_count`/`global_factor_idx` back into the dual counter. Used by `resyncPersistentDualCounter()` after external appends (e.g., LP). |
| `publish_buffer_fill_kernel` | Single-thread, sub-microsecond: publishes accum/partial/persistent fill levels + generation ticket to the mapped `BufferFillSnapshot` |
| `find_first_bad_offset` | Debug (`#ifdef DEBUG_SOA`): detects CSR offset corruption |

## DevicePostProcessingController API

### Public member variables

| Member | Type | Description |
|--------|------|-------------|
| `h_pinned_accumulation_counter` | `volatile uint32_t*` | Zero-copy batch-sieve polling: CPU reads accumulation count without sync |
| `h_pinned_persistent_count` | `volatile uint32_t*` | Zero-copy polling of persistent relation count |
| `h_pinned_partial_count` | `volatile uint32_t*` | Zero-copy polling of the LP partial batch count |
| `h_prediction_result` | `PredictionResult*` | Mapped pinned memory; host polls `should_terminate` without sync |

### Lifecycle

| Method | Description |
|--------|-------------|
| `initiate(factoringData, devicePointers, PostProcConfig)` | Allocates all device buffers, streams, events, pinned memory, mapped prediction result and buffer-fill snapshot |
| `clearBuffers()` | Frees working buffers, double-buffer events, pinned counters, prediction result |
| `clearPersistentBuffer()` | Frees persistent batch and pinned persistent counter |

### Legacy path

| Method | Description |
|--------|-------------|
| `accumulate(raw_input, raw_size, current_a, dev_a_factors_ptr, num_a_factors, start_index, stream)` | Compacts candidates into accumulation buffer via `expandAndAccumulateKernel`; returns `true` when purge threshold reached |
| `processBufferedCandidates()` | Swap buffers, Thrust sort by `true_x`, launch `batchedFactorizationKernel`, commit dual counters, publish buffer-fill telemetry |
| `consolidateToPersistent()` | Appends `d_full_batch` into `d_persistent_batch` (device-to-device); resets `d_full_batch` counters |
| `flush()` | Reads remaining accumulation count, then calls `processBufferedCandidates` + `consolidateToPersistent` |

### Batch sieve path

| Method | Description |
|--------|-------------|
| `processBatchBufferedCandidates()` | Waits on `safe_to_read_event`, launches `batchedBatchFactorizationKernelSoA` writing directly to persistent batch, commits dual counters, polls telemetry to pinned memory, optionally runs `yield_prediction_kernel` (+ `advance_steps_kernel`), resets device counter, records `safe_to_write_event`, toggles buffer. **This method's node sequence is what the CUDA graph captures at scope `postproc`/`full`** (1.0.6); the buffer toggle at its end is *host* state and therefore executes **at capture time only**, which is precisely what bakes the alternating buffer assignment into the graph. Its opening `cudaStreamWaitEvent` on `safe_to_read_event` is also why the deferred-tail root must record that event in-capture first — a wait on an externally recorded event is `cudaErrorStreamCaptureIsolation`. See [the deferred tail](#the-deferred-tail-solo-scope-full-only) below. |
| `toggleActiveBuffer()` | `active_accum_idx ^= 1` |
| `getActiveAccumulationBuffer()` | Returns `&buffers[active_accum_idx]` (a `DoubleBuffer*`) |
| `setFlushedState(bool)` | Overrides internal flush guard for device-to-device population |

#### The deferred tail (solo, scope `full` only)

Consecutive `cudaGraphLaunch`es into one stream are fully serialized, so the **last** captured batch's
post-processing (+LP) would be an exposed tail at the end of every replay. It is therefore deferred into
the **next** replay, where it is a **root** of the proc branch and overlaps that replay's `sieve(0)`.

The gate is exactly (`orchestrator.cpp:4828`):

```cpp
const bool deferred_tail = (graph_N % 2 == 0) && capture_lp;
```

- **Even `graph_N`** — the root's target buffer equals the last batch's target only when `graph_N` is
  even; for odd `graph_N` (only `graph_N == 1` survives the CLI's rounding) the root would post-process
  a buffer the graph never fills.
- **`capture_lp` (solo, scope `full`)** — ⚠ **the restriction is load-bearing, not conservatism.** With
  LP left **between** replays (scope `postproc`) the deferred body **corrupts the persistent batch**:
  the between-replay path appends through the plain counters plus a host-side
  `resyncPersistentDualCounter()`, while the deferred trial-division node appends through the **dual**
  counter one batch later, and the two interleave. Measured at RSA-100 scope `postproc`, **4/4 runs**:
  a corrupted persistent factor count (**4,301,343,056 factors for 200,382 relations**) followed by an
  illegal memory access in the subsequent `resize`, or a failed BW pre-flight SpMM `AT` verification.
  Re-ordering the post-loop drain after the LP drain does **not** fix it — the interleaving is per
  replay, not just at the end. At scope `full` every LP dispatch is a graph node ordered immediately
  after its own trial division, so the hazard cannot arise (`orchestrator.cpp:4817-4827`).

When the gate is closed the orchestrator logs `deferred-tail disabled (graph_N=…)` or
`deferred-tail disabled (LP runs between replays)` at `LOG_DEBUG_1` (`:4829-4835`), and the
non-deferred body is captured instead.

### Deduplication

| Method | Description |
|--------|-------------|
| `deduplicatePersistentBatch()` | Hash, Thrust `sort_by_key`, `unique_by_key`, rebuild compacted SoA batch; updates atomic counters on the new batch. The set it produces is the **post-dedup ground truth** the Sieve Stage Summary's `Sieved full` / `LP combined` lines are counted from (1.0.6): `LP combined` is recounted over the final relations as `large_primes[i] > 1` and `Sieved full` is the remainder, *not* `SLPPinnedStats::total_full_relations`, which is a pre-dedup cumulative and would mix scales (`orchestrator.cpp:6182-6202`). ⚠ Carries the **still-open uint32 length underflow** described under [Deduplication kernels](#deduplication-kernels) |

### Large prime support

| Method | Description |
|--------|-------------|
| `resetPartialBatch()` / `resetPartialBatch(stream)` | Resets `d_partial_batch` counters and `d_partial_dual_counter` for the next LP cycle (stream-parameterized variant for cluster extraction on a non-default stream) |
| `resyncPersistentDualCounter()` | Repacks `d_persistent_dual_counter` from actual batch counts via `sync_dual_counter_from_batch_kernel`; also async-copies to `h_pinned_persistent_count` |

### Adaptive convergence

| Method | Description |
|--------|-------------|
| `setPredictionParams(target, lp_stats_device_ptr)` | Sets `prediction_target_` and the device pointer to `SLPPinnedStats` (nullable) |
| `seedPredictionSteps(initial_steps, per_batch_increment)` | Seeds the **device** sieve-step counter `d_prediction_steps_` (`s_0 = initial_steps`, resume-aware: the restored global a-index) and fixes the per-batch increment, so `s_k = s_0 + k·delta` after `k` post-processed batches. Issued once per sieve leg, **outside** any CUDA-graph capture. Replaces the removed host setter `updatePredictionSteps()`. |

### Accessors and telemetry

| Method | Description |
|--------|-------------|
| `getPartialBatch()` / `getPersistentBatch()` | Raw `RelationBatch*` pointers for LP handshake or final download |
| `getPartialCount()` / `getPersistentCount()` | Synchronous counter reads (device-to-host copy) |
| `getPartialCapRels()` / `getPartialCapFactors()` | Capacity of partial batch in relations / factors |
| `getAccumulatedCount(stream)` | Synchronous read of current accumulation counter |
| `requestStats(stream)` / `updateStats()` | Async counter polling via pinned memory |
| `getPartialStats()` / `getPersistentStats()` | Read `(rel_count, factor_count)` from pinned memory |
| `getBufferFillSnapshot()` | Lock-free accessor for the mapped `BufferFillSnapshot` |
| `getCudaStream()` | Returns `proc_stream` for external synchronization |
| `getResetEvent()` | Returns `reset_event` for multi-stream coordination |
| `isProcessing()` | Returns `processing_active` flag |
| `getAccumulationBufferPtr()` / `getCounterPtr()` | Raw device pointers for siever handshake |

## Data Flow

**Legacy path** (`sieve_batch_size == 0`):
```
candidateRelation[] (sieve output)
  --> expandAndAccumulateKernel --> DenseCandidate[] (accumulation buffer)
  --> batchedFactorizationKernel --> d_full_batch + d_partial_batch (SoA)
  --> consolidateToPersistent   --> d_persistent_batch (SoA)
  --> deduplicatePersistentBatch --> compacted d_persistent_batch
  --> moveToHost                --> HostRelationBatch
```

**Batch path** (`sieve_batch_size > 0`):
```
candidateRelation[] (sieve output)
  --> compactCandidatesBatchKernel --> DenseCandidate[] (double buffer)
  --> batchedBatchFactorizationKernelSoA --> d_persistent_batch + d_partial_batch (SoA)
  --> deduplicatePersistentBatch --> compacted d_persistent_batch
  --> moveToHost                --> HostRelationBatch
```

In the batch path, `batchedBatchFactorizationKernelSoA` writes directly to
`d_persistent_batch` (and `d_partial_batch` when LP is active), bypassing the
intermediate `d_full_batch` and the consolidation step. After each batch,
`yield_prediction_kernel` updates `h_prediction_result` so the orchestrator can
terminate sieving without a GPU sync.

## Double-Buffer Coordination (Batch Mode)

Two `DoubleBuffer` instances (`buffers[0]`, `buffers[1]`) alias the same underlying
device memory as `d_accumulation_buffer` / `d_processing_buffer`. The sieve fills
`buffers[i]` and records `safe_to_read_event`. The postprocessor waits on that event,
factorizes, resets the device counter with `cudaMemsetAsync`, records
`safe_to_write_event`, and toggles to `buffers[1-i]`. Pinned counters
(`h_pinned_accumulation_counter`, `h_pinned_persistent_count`,
`h_pinned_partial_count`) enable CPU-side polling without GPU stalls. `reset_event`
on `proc_stream` prevents the legacy-path sieve from writing to the just-swapped
accumulation buffer before the reset completes.

## Packed Dual Counter

The factorization kernels use a single 64-bit atomic to track both relation count
and factor count simultaneously: `pack(R,F) = (uint64_t(R) << 32) | uint64_t(F)`.
A CAS loop in the warp leader reserves space for the entire warp in one atomic
operation. After kernel completion, `commit_dual_counter_kernel` assigns the
cumulative values into the `RelationBatch` counters (not addition — the dual counter
accumulates across all batch invocations in the batch-sieve path). After any external
append (e.g., LP matches), `resyncPersistentDualCounter()` must be called to realign
the dual counter.

Limit: total factor count must stay below 2^32 per batch (for full/partial batches).

## Dependencies

| Dependency | Source |
|------------|--------|
| `sieving_data_structs.h` | `candidateRelation`, `DenseCandidate`, `DoubleBuffer`, `devicePointers`, `factoringData` |
| `mpqs_soa.h` | `RelationBatch`, `RelationBatchView` |
| `mpqs_structures.h` | `Relation` |
| `largeprime.h` | `SLPPinnedStats` (forward-declared in the header; full definition for the prediction kernel) |
| `uint512.cuh` | 512-bit integer arithmetic, `abs_square_minus_N` |
| `math_utils.cuh` | `calculate_sqrt_of_QX()` |
| Thrust | `sort`, `sort_by_key`, `unique_by_key`, `exclusive_scan`, `gather`, `sequence`, `copy` |
| CMake links | `mpqs_common`, `mpqs_sieve`, `cudampqs_build_flags` |
