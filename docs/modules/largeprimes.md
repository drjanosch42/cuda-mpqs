# Large Primes Module (`src/largeprimes/`)

Single Large Prime (SLP) variant via a 3-stage GPU slab hash table pipeline.
Combines pairs of 1-partial relations sharing the same large prime cofactor into full relations.

LP is **orthogonal to pipeline selection** — it operates identically in both the batch (double-buffered) and legacy (single-cube) sieve paths. The orchestrator initializes a single shared `LargePrimeVariant` instance before the pipeline branch point; both paths call `processAndCommit()` with the same interface.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `largeprime.h` | ~217 | `LargePrimeVariant` class, `LargePrimeConfig`, `SLPStatus` enum, `SLPPinnedStats` |
| `largeprime.cu` | ~1345 | 7 CUDA kernels, hash table management, pipeline orchestration |
| `CMakeLists.txt` | 13 | Build target `mpqs_largeprimes` (static library) |

## `LargePrimeConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_witness_capacity` | `uint64_t` | 16,777,216 (16M) | Max unique large primes stored; slab rows capped at min(this, 2^hash_bits) |
| `max_combined_output` | `uint32_t` | 0 | Capacity of the combined full-relations output buffer |
| `hash_bits` | `uint32_t` | 20 | Directory size = 2^B buckets (default 1,048,576) |
| `enable_second_sort` | `bool` | false | Secondary CUB radix sort before probing for perfect 128B coalescing |
| `lp1_bound` | `uint64_t` | 0 | Large prime bound; 0 disables the LP variant |
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

**Payload slabs** — Fixed-width rows of 16 elements × 8 bytes = 128 bytes (one L2 cache line). Each entry:

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

## 3-Stage Pipeline (`processAndCommit`)

The intra-batch sort/dedup pre-pass was removed; identical primes within one batch
are now resolved naturally during Stage 3 commit (the second arrival probes the slab,
matches the first as a witness, and combines). This eliminates two CUB radix sorts and
one dedup kernel per LP batch.

**Stage 1 — Directory Fetch.** Read hash bucket for each input 1-partial. Classify as `NEEDS_ALLOCATION` (count=0) or `NEEDS_PROBE` (count>0). Pack 64-bit routing key: `[Status:4 | RowIdx:28 | Tag:32]`.

**Stage 2 — Vectorized Hash Probe.** For `NEEDS_PROBE` elements: 128-bit `__ldg()` reads (`ulonglong2`) scan each slab row (8 iterations of 16 bytes = 128 bytes). Tag match yields `MATCH_FOUND` + recorded `WitnessIdx`; miss yields `NEEDS_APPEND`.

**Stage 3 — Global Commit.**
- **3A** (`global_combine_kernel`): `MATCH_FOUND` elements merge with stored witness to produce full relation (`sqrt_Q_combined = sqrt_Q_A * sqrt_Q_B mod N`, CSR factor merge with exponent summation, max 64 factors).
- **3B** (`global_append_kernel`): `NEEDS_ALLOCATION` / `NEEDS_APPEND` elements acquire bucket spin-lock, allocate row if needed, write payload entry, store witness SoA data.
- **3C**: Unpack dual counters back to SoA batches; record completion event; deferred append of output to persistent storage.

## CUDA Kernels

| Kernel | Stage | Block | Grid |
|--------|-------|-------|------|
| `directory_fetch_kernel` | 1 | 256 | ceil(N/256) |
| `probe_hash_table_kernel` | 2 | 256 | ceil(N/256) |
| `sync_dual_counter_kernel` | 3 | 1 | 1 |
| `global_combine_kernel` | 3A | 256 | ceil(N/256) |
| `global_append_kernel` | 3B | 256 | ceil(N/256) |
| `update_telemetry_kernel` | post | 1024 | 1 |

## `LargePrimeVariant` API

| Method | Description |
|--------|-------------|
| `initiate(cfg, N)` | Allocate hash table, witness/output SoA, pinned telemetry |
| `processAndCommit(input_partials, persistent_storage)` | Run 3-stage pipeline; append results to persistent storage |
| `clearBuffers()` | Free all device and host memory |
| `requestStats()` | No-op (kept for API compatibility) |
| `updateStats()` | Launch telemetry kernel (1 block × 1024 threads) |
| `getTelemetry()` | Return pointer to pinned `SLPPinnedStats` |
| `getWitnessStats()` | Return `(witness_count, witness_factor_count)` from the witness SoA |
| `getWitnessCapacityRels()` | Max relations in witness SoA |
| `getWitnessCapacityFactors()` | Max factors in witness SoA |

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

## Memory Footprint (B=20, 16M witnesses)

| Component | Formula | Size |
|-----------|---------|------|
| Directory | 2^20 × 8 | 8 MB |
| Payload slabs | min(16M, 2^20) × 16 × 8 | 128 MB |
| Witness SoA | 16M rels × ~128 B avg | ~2,048 MB |
| Output buffer | max_combined_output × ~128 B | ~4 MB |
| **Total** | | **~2.2 GB** |

## Orchestrator Integration

### Shared Initialization (both pipelines)

LP initialization is performed **once** in `SieveStage()` before the batch/legacy branch point (`orchestrator.cpp:785–834`). The orchestrator:

1. Sets the sieve threshold override: `siever_->setThresholdOverride(config_.lp1_bound)` — this lowers the sieve threshold from the default (derived from F) to `min(lp1_bound, fb_bound)`, allowing candidates with one large cofactor ≤ `lp1_bound` to pass through.
2. Configures `LargePrimeConfig` — witness capacity (default 1M, rounded to power of 2), hash bits (auto-derived as `log2(witness_cap) - 4`), combined output buffer (default 32K).
3. Creates the `LargePrimeVariant` on the postprocessor's CUDA stream.

```cpp
// orchestrator.cpp — shared LP init (serves both batch and legacy)
if (config_.lp1_bound > 0) {
    siever_->setThresholdOverride(config_.lp1_bound);
    // ... configure lp_conf ...
    largeprime_ = std::make_unique<lp::LargePrimeVariant>(postprocessor_->getCudaStream());
    largeprime_->initiate(config_.lp_config, config_.N);
}
```

### Batch Sieve Path (double-buffered)

In batch mode (`sieve_batch_size > 0`), LP partials are processed **periodically** every `adaptive_lp_batch_interval_` batches (initially 10, calibrated at 5%/20% progress). Both streams are synchronized before each LP call. A final flush processes remaining partials after the batch loop exits.

### Legacy Sieve Path (single-cube)

In legacy mode (`sieve_batch_size == 0`), `processAndCommit()` is called on every postprocessor buffer-full event. LP telemetry is polled asynchronously for progress reporting and witness fill projection.

### Common LP Call Pattern

```cpp
largeprime_->processAndCommit(
    postprocessor_->getPartialBatch(),
    postprocessor_->getPersistentBatch()
);
```

### TruncatedSieveRun (autotune probes)

The truncated sieve run (`orchestrator.cpp:1596–1865`) **fully initializes LP** when `config_.lp1_bound > 0`: threshold override, postprocessor partial buffers, hash table, and witness SoA. LP `processAndCommit` is called during buffer-full events, and LP telemetry is fed to the `LPFillProjector` for witness fill estimation.

## Autotune Integration

### LP Bound Selection

LP bound (`lp1_bound`, denoted L) is a **first-class search variable** in the autotune Stage 3 coordinate descent optimizer (`sieve_optimizer.cpp`).

**Theory fallback (cold start, no history):**
```
L = 10 * F   for bit_length ≥ 200
L = 0        otherwise
```
For RSA-100 (330 bits, F=3M), this seeds L=30M. This non-zero seed ensures the optimizer explores LP-on configurations even on first run.

**History-based projection:** When autotune history exists, L is projected via:
- **Interpolation** — log-space interpolation if both bracketing entries have LP active; ratio scaling if only one does.
- **Extrapolation** — OLS model on LP-active entries, or median `lp1/F` ratio fallback.
- **Exact match** — L restored directly from history.

**Stage 3 optimizer:**
1. Warm-starts at projected `(F, M, L)`.
2. Per round: after optimizing F (golden section, quantize=1000) and M (exhaustive powers of 2), evaluates L=0 baseline.
3. **Unconditional golden section** on L ∈ [1M, min(50·F, 500M)] with quantize=1M — runs regardless of warm-start L value.
4. Takes whichever of L=0 or L_opt yields lower estimated runtime.

### LP Yield Discount in Runtime Estimation

Short truncated probes (12% of target) understate LP benefit because match rates accelerate as witnesses accumulate. The runtime estimator (`runtime_estimator.cpp:123–131`) applies a conservative correction:

```
sieve_total_sec *= (1.0 - 0.5 * fill_pct)
```

where `fill_pct` is the projected witness fill fraction at full-run completion (from `LPFillProjector`). The 0.5 discount factor acknowledges that LP's acceleration effect is only partially captured by short probes.

### LP Search Bounds

Derived from the projected F (`cost_models.cpp:27–36`):
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
| CUB (header-only) | `DeviceRadixSort` |

Namespace: `mpqs::lp`.
