# Large Primes Module (`src/largeprimes/`)

Single Large Prime (SLP) variant via a 3-stage GPU slab hash table pipeline.
Combines pairs of 1-partial relations sharing the same large prime cofactor into full relations.

LP is **orthogonal to pipeline selection** — it operates identically in both the batch (double-buffered) and legacy (single-cube) sieve paths. The orchestrator initializes a single shared `LargePrimeVariant` instance before the pipeline branch point. The legacy path uses the synchronous `processAndCommit()`; the batch path uses the **async pipeline** `processAndCommitAsync()` + event-driven deferred append (see below).

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `largeprime.h` | ~345 | `LargePrimeVariant` class, `LargePrimeConfig`, `SLPStatus` enum, `SLPPinnedStats`, `CombineProvenanceEntry` |
| `largeprime.cu` | ~1750 | 10 CUDA kernels, hash table management, sync + async pipeline orchestration |
| `CMakeLists.txt` | 16 | Build target `mpqs_largeprimes` (static library) |

## `LargePrimeConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_witness_capacity` | `uint64_t` | 16,777,216 (16M) | Max unique large primes stored; slab rows capped at min(this, 2^hash_bits) |
| `max_combined_output` | `uint32_t` | 0 | Capacity of the combined full-relations output buffer |
| `hash_bits` | `uint32_t` | 20 | Directory size = 2^B buckets (default 1,048,576) |
| `lp1_bound` | `uint64_t` | 0 | Large prime bound; 0 disables the LP variant |
| `purge_after_match` | `bool` | false | Delete a matched witness from the slab after its first match — prevents one witness generating multiple combined relations (degenerate kernel vectors) at small factor-base sizes |
| `capture_provenance` | `bool` | false | Capture per-combined-relation constituent provenance (`CombineProvenanceEntry`: probe/witness roots, signs, val_2_exps, LP) into a separate device buffer before the purge erases the linkage. Strictly additive diagnostic — gated by CLI `--dump_combine_provenance`; when false, no buffer is allocated and no extra kernel launched |
| `device_id` | `int` | 0 | CUDA device index |

## Hash Table Architecture

Two-level append-only slab hash table.

**Directory** — `2^B` entries, 8 bytes each:

| Bits | Field | Width |
|------|-------|-------|
| 63 | Lock (spin-lock) | 1 |
| 48–62 | Count | 15 |
| 0–47 | RowIdx (into payload slab) | 48 |

Hash function: `h(p) = (p >> 1) & ((1 << B) - 1)`.

**Payload slabs** — Fixed-width rows of `ROW_WIDTH_ELEMS` = 16 elements × 8 bytes = 128 bytes (one L2 cache line). Each entry:

| Bits | Field | Width |
|------|-------|-------|
| 32–63 | Tag | 32 |
| 0–31 | WitnessIdx (into global witness SoA) | 32 |

Tag function: `T(p) = (p >> 1) >> B`. Requires `p < 2^(32 + B + 1)` for tag to fit 32 bits.

**Locking**: Per-bucket spin-lock on bit 63, acquired via `atomicCAS`, released via `atomicExch` + `__threadfence()`. Full rows (count ≥ 16) reject new entries (graceful degradation).

## `SLPStatus` State Machine

| Value | Name | Meaning |
|-------|------|---------|
| 0 | `CONSUMED` | Default zero-initialized state: dropped, redundant, or already merged |
| 2 | `NEEDS_ALLOCATION` | Empty bucket encountered; allocate new slab row |
| 3 | `NEEDS_PROBE` | Bucket non-empty; vectorized probe required |
| 4 | `MATCH_FOUND` | Tag matched existing witness |
| 5 | `NEEDS_APPEND` | No match; append to existing slab row |

## 3-Stage Pipeline (`processAndCommit` / `processAndCommitAsync`)

The intra-batch sort/dedup pre-pass was removed; identical primes within one batch
are now resolved naturally during Stage 3 commit (the second arrival probes the slab,
matches the first as a witness, and combines). This eliminates two CUB radix sorts and
one dedup kernel per LP batch.

**Stage 1 — Directory Fetch.** Read hash bucket for each input 1-partial. Classify as `NEEDS_ALLOCATION` (count=0) or `NEEDS_PROBE` (count>0). Pack 64-bit routing key: `[Status:4 | RowIdx:28 | Tag:32]`.

**Stage 2 — Vectorized Hash Probe.** For `NEEDS_PROBE` elements: 128-bit `__ldg()` reads (`ulonglong2`) scan each slab row (`ROW_WIDTH_ELEMS/2` = 8 iterations of 16 bytes = 128 bytes). Tag match yields `MATCH_FOUND` + recorded `WitnessIdx`; miss yields `NEEDS_APPEND`. Under `purge_after_match`, the matched slab entry is erased here.

**Stage 3 — Global Commit.**
- **3A** (`global_combine_kernel`): `MATCH_FOUND` elements merge with stored witness to produce full relation (`sqrt_Q_combined = sqrt_Q_A * sqrt_Q_B mod N`, CSR factor merge with exponent summation, max 64 factors).
- **3A′** (`capture_provenance_kernel`, diagnostic): when `capture_provenance`, records each combine's constituents before witness linkage is lost — no-op otherwise.
- **3B** (`global_append_kernel`): `NEEDS_ALLOCATION` / `NEEDS_APPEND` elements acquire bucket spin-lock, allocate row if needed, write payload entry, store witness SoA data.
- **3C**: Unpack dual counters back to SoA batches (`sync_dual_counter_kernel`); output is appended to persistent storage — synchronously by `processAndCommit`, or deferred via `device_append_kernel` in the async path.

### Async Pipeline (batch sieve path)

`processAndCommitAsync()` enqueues the full kernel chain on the dedicated `lp_stream` and returns immediately (no CPU blocking): `snapshot_count_kernel` captures the input count device-side (records `count_snapshot_event_` so the orchestrator can reset the partial batch), the 3-stage chain runs, and `lp_done_event_` is recorded at the end. The orchestrator polls `isComplete()` (`cudaEventQuery`) and, once done, drains the batch with `launchDeviceAppend()` — a GPU-side `device_append_kernel` copy of the LP output batch into the persistent batch on the postprocessing stream, with no host-side counter reads — then `resetOutputBatch()`. LP therefore overlaps the next sieve batch entirely.

## CUDA Kernels

| Kernel | Stage | Block | Grid |
|--------|-------|-------|------|
| `snapshot_count_kernel` | pre (async) | 1 | 1 |
| `directory_fetch_kernel` | 1 | 256 | ceil(N/256) |
| `probe_hash_table_kernel` | 2 | 256 | ceil(N/256) |
| `sync_dual_counter_kernel` | 3 | 1 | 1 |
| `global_combine_kernel` | 3A | 256 | ceil(N/256) |
| `capture_provenance_kernel` | 3A′ (diagnostic) | 256 | ceil(N/256) |
| `diagnose_combine_inputs_kernel` | 3A (debug) | 256 | ceil(N/256) |
| `global_append_kernel` | 3B | 256 | ceil(N/256) |
| `device_append_kernel` | deferred append | — | — |
| `update_telemetry_kernel` | post | 1024 | 1 |

## `LargePrimeVariant` API

| Method | Description |
|--------|-------------|
| `initiate(cfg, N)` | Allocate hash table, witness/output SoA, pinned telemetry, async events |
| `processAndCommit(input_partials, persistent_storage, input_count_hint = 0)` | Synchronous: run 3-stage pipeline; append results to persistent storage. `input_count_hint` = pinned counter from the postprocessor (0 = fall back to `getCount`) |
| `processAndCommitAsync(input_partials, persistent_storage)` | Enqueue the full LP kernel chain on `lp_stream` and return immediately; records `lp_done_event_` |
| `isComplete()` | Non-blocking `cudaEventQuery(lp_done_event_)` |
| `getDoneEvent()` / `getCountSnapshotEvent()` / `getPartialsReadyEvent()` | CUDA events for orchestrator-side async signaling |
| `getOutputBatch()` / `resetOutputBatch([stream])` | Access / reset the LP combined-output batch (deferred-append flow) |
| `launchDeviceAppend(persistent, stream)` | GPU-side append of the LP output batch into the persistent batch (`device_append_kernel`); resets the output counters on the same stream |
| `moveWitnessesToHost(dest, stream)` | Download all stored witness partials to host SoA (after the sieve loop, before matrix construction / serialization) |
| `moveProvenanceToHost(dest, stream)` | Download captured `CombineProvenanceEntry` records (no-op unless `capture_provenance`) |
| `clearBuffers()` | Free all device and host memory |
| `requestStats()` | No-op (kept for API compatibility) |
| `updateStats()` | Launch telemetry kernel (1 block × 1024 threads) |
| `getTelemetry()` | Return pointer to pinned `SLPPinnedStats` |
| `getCombinedCountPinned()` | Most recent LP combined relation count from pinned memory (no sync) |
| `getWitnessStats()` | Return `(witness_count, witness_factor_count)` from the witness SoA |
| `getWitnessCapacityRels()` / `getWitnessCapacityFactors()` | Max relations / factors in witness SoA |

## Telemetry (`SLPPinnedStats`)

Lock-free pinned mapped struct. GPU writes fields 0–9, executes `__threadfence_system()`, then increments `total_iterations` as a generation ticket. Host polls `total_iterations` (declared `volatile`) for consistency.

| Field | Type | Description |
|-------|------|-------------|
| `new_partials_buffer_fill` | `uint64_t` | 1-partials in last consumed batch |
| `total_witnesses` | `uint64_t` | Total unique large primes stored |
| `total_full_relations` | `uint64_t` | Cumulative full relations produced |
| `last_batch_full_relations` | `uint64_t` | Full relations from last batch |
| `last_batch_new_witnesses` | `uint64_t` | New witnesses from last batch |
| `empty_hash_buckets` | `uint64_t` | Buckets with count = 0 |
| `full_hash_buckets` | `uint64_t` | Buckets with count ≥ 16 |
| `slab_overflow_count` | `uint64_t` | Cumulative silently dropped slab appends (row full) |
| `witness_overflow_count` | `uint64_t` | Cumulative silently dropped witness SoA reservations |
| `output_overflow_count` | `uint64_t` | Cumulative silently dropped combined output reservations |
| `total_iterations` | `volatile uint64_t` | Generation ticket (must be last) |

## Relation Algebra

Two 1-partials A, B sharing large prime L combine as:

- `sqrt_Q = (sqrt_Q_A * sqrt_Q_B) mod N`
- `sign = sign_A * sign_B`
- `val_2_exp = val_2_exp_A + val_2_exp_B`
- Factors: sorted merge of CSR arrays with exponent summation (capped at 64 factors)
- `char_bits = char_bits_A ^ char_bits_B` — under `--char_mode branch`, the combined relation's branch character vector is the XOR of its constituents' vectors (`branchCharBit` is an F2 homomorphism over the field-element product; bits are never re-derived from `sqrt_Q_combined`). Mode-agnostic: under `norm`/`none` both inputs carry 0, so the result stays 0. The cluster CPU LP path (`cpu_lp.cu`) does the same.

Note: the sieve's LP acceptance gate performs no primality test, so a small fraction of "large primes" are composite or below fb_max (see [postprocessing.md](postprocessing.md) and the sqrt-failure sieve-bug reproduction notes) — a relation-hygiene issue only; such witnesses rarely match and composite-LP combines never reach the matrix.

## Memory Footprint (B=20, 16M witnesses)

| Component | Formula | Size |
|-----------|---------|------|
| Directory | 2^20 × 8 | 8 MB |
| Payload slabs | min(16M, 2^20) × 16 × 8 | 128 MB |
| Witness SoA | 16M rels × ~128 B avg | ~2,048 MB |
| Output buffer | max_combined_output × ~128 B | ~4 MB |
| **Total** | | **~2.2 GB** |

Operationally: on a 40 GB A100 at RSA-140+ scale, `--lp1_max_witnesses 64M` (~24 GB) silently starves the rest of the pipeline — use ≤ 32M.

## Orchestrator Integration

### Shared Initialization (both pipelines)

LP initialization is performed **once** before the batch/legacy branch point:

1. `SieveStage()` sets the sieve threshold override: `siever_->setThresholdOverride(config_.lp1_bound)` (`orchestrator.cpp:4096`) — this raises the sieve threshold from the default ⌊log₂(F)⌋ to ⌊log₂(lp1_bound)⌋, allowing candidates with one large cofactor ≤ `lp1_bound` to pass the scan.
2. `initLargePrimes()` (`orchestrator.cpp:3359`) resolves `LargePrimeConfig` and creates the `LargePrimeVariant` on the postprocessor's CUDA stream:
   - `max_combined_output`: `--lp1_max_combined_output`, default 2^15 (32K).
   - `max_witness_capacity`: `--lp1_max_witnesses` (snapped up to a power of two), else auto-derived — base `4·fb_size`, scaled by an LP-to-FB-ratio multiplier `min(4, max(1, log₂(L/F/10)))`, snapped to a power of two clamped to [2^18, 2^24].
   - `hash_bits`: `--lp1_hash_bits`, else `log₂(witness_capacity) − 4`.
   - `purge_after_match = (fb_size < 20000)`; `capture_provenance = --dump_combine_provenance`.

```cpp
// orchestrator.cpp — shared LP init (serves both batch and legacy)
config_.lp_config = lp_conf;
largeprime_ = std::make_unique<mpqs::lp::LargePrimeVariant>(postprocessor_->getCudaStream());
largeprime_->initiate(config_.lp_config, config_.N);
```

### Batch Sieve Path (double-buffered, async)

In batch mode (`sieve_batch_size > 0`), LP partials are processed **periodically** every `adaptive_lp_batch_interval_` batches (initially 10, recalibrated at 5%/20% progress, clamped to [1,100]) via the **async pipeline**: drain the previous in-flight LP (wait `lp_done_event_` → `launchDeviceAppend` → `resyncPersistentDualCounter`), then launch `processAndCommitAsync()` on the freshly-swapped partial batch and continue sieving immediately. A final drain + flush processes remaining partials after the batch loop exits. The CUDA-graph path mirrors this with its own `lp_graph_pending` state.

### Legacy Sieve Path (single-cube)

In legacy mode (`sieve_batch_size == 0`), the synchronous `processAndCommit()` is called on every postprocessor buffer-full event. LP telemetry is polled asynchronously for progress reporting and witness fill projection.

### TruncatedSieveRun (autotune probes)

The truncated sieve run (`orchestrator.cpp:5774`) **fully initializes LP** when `config_.lp1_bound > 0`: threshold override (`:5816`), postprocessor partial buffers, hash table, and witness SoA. LP `processAndCommit` is called during buffer-full events, and LP telemetry is fed to the `LPFillProjector` for witness fill estimation.

## Autotune Integration

### LP Bound Selection

LP bound (`lp1_bound`, denoted L) is a **first-class search variable** of the autotune joint (F,L) optimizer (`SieveParameterOptimizer`, `sieve_optimizer.cpp` — L-shaped path A/B/C: local refinement, pattern probes, gradient descent; coordinate descent retained for legacy callers). L values are quantized to 1M (`quantize_L`) with floor `L_min(F) = max(1M, 10·F)`.

**Theory fallback (cold start, no history)** — `ParameterProjector::theoryFallback` (`autotune_projection.cpp:170`):
```
L = 10 * F   for bit_length ≥ 200   (clamped to F² as a safety invariant)
L = 0        otherwise
```
For RSA-100 (330 bits, F=3M), this seeds L=30M. This non-zero seed ensures the optimizer explores LP-on configurations even on first run.

**History-based projection:** When autotune history exists, L is projected via:
- **Interpolation** — log-space interpolation if both bracketing entries have LP active; ratio scaling if only one does.
- **Extrapolation** — OLS model on LP-active entries, or median `lp1/F` ratio fallback.
- **Exact match** — L restored directly from history.

### LP Yield Discount in Runtime Estimation

Short truncated probes see linear witness accumulation, but LP combined relations
scale quadratically (birthday paradox: matches ≈ W²/2B). The runtime estimator
(`runtime_estimator.cpp`, LP birthday model ~`:153-178`) projects the witness
arrival rate to full-run scale and applies:

```
B        = π(L) − π(F)                    (distinct LP values, x/ln x approximation)
lp_frac  = clamp( (W²/2B) / target, 0, 0.35 )
sieve_total_sec *= (1.0 - lp_frac)
```

The 0.35 clamp reflects that LP rarely contributes more than ~35% of relations. (This birthday-paradox model supersedes the earlier flat `0.5·fill_pct` discount.)

### LP Search Bounds

Derived from the projected F (`cost_models.cpp:49-50`):
```
lp_lo = 0           (LP-disabled always a candidate)
lp_hi = min(50·F, 500M)
```

## Dependencies

| Link target | Provides |
|-------------|----------|
| `mpqs_postproc` | `RelationBatch`, `RelationBatchView` |
| `mpqs_common` | `uint512`, `HPCLogger` |
| `cudampqs_build_flags` | Compiler flags, `CUDA::cudart`, OpenMP |

Namespace: `mpqs::lp`.
