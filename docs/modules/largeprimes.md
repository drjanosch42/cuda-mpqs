# Large Primes Module (`src/largeprimes/`)

Single Large Prime (SLP) variant via a 3-stage GPU slab hash table pipeline.
Combines pairs of 1-partial relations sharing the same large prime cofactor into full relations.

LP is **orthogonal to pipeline selection** — it operates identically in both the batch (double-buffered) and legacy (single-cube) sieve paths. The orchestrator initializes a single shared `LargePrimeVariant` instance before the pipeline branch point. The legacy path uses the synchronous `processAndCommit()`; the batch path uses the **async pipeline** `processAndCommitAsync()` + event-driven deferred append (see below).

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `largeprime.h` | ~350 | `LargePrimeVariant` class, `LargePrimeConfig`, `SLPStatus` enum, `SLPPinnedStats`, `CombineProvenanceEntry` |
| `largeprime.cu` | ~1840 | 11 CUDA kernels, hash table management, sync + async pipeline orchestration |
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
The spin's *read* of the directory word must itself be **coherent** — since 1.0.6 it is
`atomicAdd((unsigned long long*)&directory[hash], 0ULL)` (`largeprime.cu:695`), this module's established
coherent-read idiom (cf. `atomic_reserve_dual`), not a plain load; see
[v1.0.6 correctness fixes](#v106-correctness-fixes-solo-gpu-lp-path).

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
  Two guards run **before** the merge (`largeprime.cu:424-449`): (i) an LP-equality check that rejects tag collisions,
  and (ii) since 1.0.6 the **`sqrt_Q` identity guard** that rejects a partial matched against itself — see
  [v1.0.6 correctness fixes](#v106-correctness-fixes-solo-gpu-lp-path) below.
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
| `record_appended_kernel` | append accounting (legacy sync path) | 1 | 1 |

## `LargePrimeVariant` API

| Method | Description |
|--------|-------------|
| `initiate(cfg, N)` | Allocate hash table, witness/output SoA, pinned telemetry, async events |
| `processAndCommit(input_partials, persistent_storage, input_count_hint = 0)` | Synchronous: run 3-stage pipeline; append results to persistent storage. `input_count_hint` = pinned counter from the postprocessor (0 = fall back to `getCount`) |
| `processAndCommitAsync(input_partials, persistent_storage)` | Enqueue the full LP kernel chain on `lp_stream` and return immediately; records `lp_done_event_` |
| `isComplete()` | Non-blocking `cudaEventQuery(lp_done_event_)` |
| `getDoneEvent()` / `getCountSnapshotEvent()` / `getPartialsReadyEvent()` | CUDA events for orchestrator-side async signaling |
| `getStream()` | The LP stream (== the post-processor stream). Used to re-record the LP events from a non-capturing stream after a CUDA-graph capture; an event whose last record was inside a capture is unusable from the host until re-recorded |
| `getOutputBatch()` / `resetOutputBatch([stream])` | Access / reset the LP combined-output batch (deferred-append flow) |
| `launchDeviceAppend(persistent, stream)` | GPU-side append of the LP output batch into the persistent batch (`device_append_kernel`); resets the output counters on the same stream |
| `moveWitnessesToHost(dest, stream)` | Download all stored witness partials to host SoA (after the sieve loop, before matrix construction / serialization) |
| `moveProvenanceToHost(dest, stream)` | Download captured `CombineProvenanceEntry` records (no-op unless `capture_provenance`) |
| `clearBuffers()` | Free all device and host memory |
| `requestStats()` | No-op (kept for API compatibility) |
| `updateStats()` | Launch telemetry kernel (1 block × 1024 threads). **All telemetry inputs are device pointers** (1.0.6) — the per-dispatch combined count, the cumulative full-relation counter and the previous witness snapshot were host scalars read from mapped pinned memory and passed **by value**, which a CUDA-graph capture would freeze at capture time. The cumulative and per-dispatch LP relation counters are now maintained by `device_append_kernel` (and `record_appended_kernel` on the legacy synchronous path), i.e. counted exactly once at the point where LP relations are committed to the persistent batch. **Visible consequence:** the pre-v1.0.6 host-side accumulation was *lossy* under the overlapped pipeline — it sampled the counter sequence instead of summing it, undercounting by ~26 % at RSA-100 (`--cuda_graph_unroll 0`: cumulative 103,662 vs 139,961), low enough to fall *below* the post-dedup ground truth (105,163), which is impossible for a pre-dedup total. Relation output was never affected; this was a reporting error. |
| `getTelemetry()` | Return pointer to pinned `SLPPinnedStats` |
| `getCombinedCountPinned()` | Most recent LP combined relation count from pinned memory (no sync) |
| `getWitnessStats()` | Return `(witness_count, witness_factor_count)` from the witness SoA |
| `getWitnessCapacityRels()` / `getWitnessCapacityFactors()` | Max relations / factors in witness SoA |

## CUDA graph capture (1.0.6)

In **single-node** runs at capture scope `full`, the asynchronous LP pipeline is captured into the sieve
graph and dispatched **once per captured batch** (`--cuda_graph_lp_stride K` makes that every `K`-th
batch; `K ≥ graph_N` restores the pre-v1.0.6 one-dispatch-per-replay cadence). The captured block is
`cudaEventRecord(partials_ready)` → `processAndCommitAsync` → `launchDeviceAppend` →
`resyncPersistentDualCounter` → `resetPartialBatch`, all on the post-processing stream —
`LargePrimeVariant` is constructed with `postprocessor_->getCudaStream()`, so `lp_stream == proc_stream`
and the historical `cudaStreamWaitEvent(proc, lp_done)` hand-offs were self-waits; in-stream order
supplies every gate the between-replay deferral needed events for.

In **cluster** runs LP is never captured (workers sieve without LP; the coordinator matches on the CPU),
and requesting scope `full` there is downgraded to `postproc` with a warning. `MPQS_LP_DIAG=1` forces the
scope down to `sieve`, because the diagnostic block inside `processAndCommitAsync` issues `cudaMalloc`,
`cudaStreamSynchronize` and synchronous `cudaMemcpy` — all illegal inside a capture. The **synchronous**
legacy `processAndCommit()` is not capturable at all and continues to run only outside the sieve loop
(the end-of-sieve flush).

## v1.0.6 correctness fixes (solo GPU-LP path)

Two defects were found and fixed in v1.0.6. **Both are pre-existing — latent in every binary up to
and including v1.0.5 — and both live only on the solo GPU-LP path.**
Cluster runs were never exposed to either: workers sieve with LP disabled, and the coordinator
matches on the CPU (`src/cluster/cpu_lp.cu`), which has carried the equivalent host guard since
1.0.3a (`9881c00`).

### 1. GPU `sqrt_Q` identity guard in `global_combine_kernel` (`0b71be8`)

`largeprime.cu:449`:

```cpp
if (input_view.sqrt_Q[my_idx] == global_witness_view.sqrt_Q[target_global_idx]) return;
```

This is the device mirror of the canonical host guard `CPULargePrimeTable::combinePartials`
(`src/cluster/cpu_lp.cu:69`). It replaces a `TODO` whose premise — "this kernel runs only in
solo/single-node mode, which emits no duplicate partials" — is **falsified**. The reasoning
recorded in the source is:

- Matched witnesses are **not** purged at production factor-base sizes: the orchestrator sets
  `purge_after_match = (f_data_.size < 20000)` (`orchestrator.cpp:3448`), which is false for every
  RSA-100+ run. So a witness stays in the slab after it has been matched.
- Any **re-presentation** of the same partial — a duplicate emission from the sieve, or a cluster
  worker's end-of-assignment repeat — therefore self-matches against the witness copy it inserted
  itself.
- Input and witness then share the large prime *P* **and** the Q-root, so the merge yields
  `Q_res = sqrt_Q(P)² mod N` with exponent vector `2·e ≡ 0 (mod 2)` — a perfect-square row. It is a
  valid congruence, but it can only ever produce the trivial dependency `X ≡ ±Y`.
- Such a row is **unique**, so it survives the end-of-sieve dedup into the matrix as pure trivial
  ballast, diluting the per-solution nontrivial-GCD rate.
- `sqrt_Q` is a sufficient and cheapest discriminator: two *distinct* relations sharing an LP arise
  from different `(a,b)` and therefore have different `sqrt_Q`.

Measured effect: roughly **7–8 rows removed per sieve batch** at RSA-100 in every mode, and the
per-solution nontrivial sqrt rate rises **48.2 % → 52.2 %** at RSA-100 solo (218 solutions
/ 105 nontrivial pre-fix vs 226 / 118 post-fix); on real RSA-155 cluster data the host guard drops
**2,909** such rows (0.041 %).

### 2. Directory spin livelock in `global_append_kernel` (`703dc74`)

The bucket-directory spin (`largeprime.cu:673-700`) read `directory[hash]` with a **plain load**,
which compiles to a weak, L1-cacheable `LDG.E.64`. L1 is **not** coherent across SMs, while the lock
release below is an `atomicExch` serviced at L2 — so it never invalidates another SM's cached line.
A thread that once observed the locked value could re-read its own stale L1 line forever: an
**unbounded livelock** that wedges `lp_stream` and, through the event DAG, the entire pipeline.

Fix — read coherently at L2:

```cpp
uint64_t current_dir = atomicAdd((unsigned long long*)&directory[hash], 0ULL);
```

Latent since the kernel was written. It first surfaced under the v1.0.6 SM-aligned launch
geometries (**~23 % of runs** at `--params 560,8,8,8,70,1024,560,1024`, `--cuda_graph_unroll 0`,
RTX 5070 Ti), whose occupancy shape leaves LP blocks alone on an SM with no L1 eviction pressure.
Symptom: GPU pinned at 100 % utilisation drawing an **idle-level ~71 W** instead of the ~263 W a
healthy sieve draws on that card, host output stopped. Confirmed by instrumentation: the plain load
returned `lock=SET/count=4` for 2²⁰ consecutive iterations while the L2 view of the same address was
`lock=CLEAR/count=5`.

### ⚠ The solo GPU-LP path is ±1 nondeterministic

Independently of the two fixes above: run-to-run on **one** binary, at one operating point, the solo
GPU-LP path moves by **±1** on `LP combined`, `Total (deduped)` and `Cumulative LP full` — a single
LP-combined relation appearing or not at the run's tail. Measured on both RTX 5070 Ti (`LP
combined` 123,958–123,959 over n=3 at `--cuda_graph_unroll 0`) and A100 (75,455/75,456 relations
with 22,408/22,409 LP fulls, **intra-arm**, = 0.0013 %), and on the **pre-change baseline binary** as
well as after — so it is **not a v1.0.6 regression**, and the `cgu=4` arm jitters the same way. The
counters that *are* exact are `Sieved full`, `Duplicates`, `Batches processed` and the whole
`[Config]` block.

**Consequence for anyone using this module's counters as evidence: work identity between arms must
be judged at ±1, never by exact equality** — an exact-equality acceptance criterion cries wolf. The
jitter itself is **unexplained**.

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

Two 1-partials A, B sharing large prime L combine as (**only when `sqrt_Q_A != sqrt_Q_B`** — an
A == B self-combine is a perfect square and is rejected by the 1.0.6 identity guard, see
[v1.0.6 correctness fixes](#v106-correctness-fixes-solo-gpu-lp-path)):

- `sqrt_Q = (sqrt_Q_A * sqrt_Q_B) mod N`
- `sign = sign_A * sign_B`
- `val_2_exp = val_2_exp_A + val_2_exp_B`
- Factors: sorted merge of CSR arrays with exponent summation (capped at 64 factors)
- `char_bits = char_bits_A ^ char_bits_B` — under `--char_mode branch`, the combined relation's branch character vector is the XOR of its constituents' vectors (`branchCharBit` is an F2 homomorphism over the field-element product; bits are never re-derived from `sqrt_Q_combined`). Mode-agnostic: under `norm`/`none` both inputs carry 0, so the result stays 0. The cluster CPU LP path (`cpu_lp.cu`) does the same.

Note: the LP acceptance gate is only `remainder < cfg.lp1_bound` (`src/postprocessing/postprocessing.cu:371`) — no primality test and no `remainder > fb_max` check — so a small fraction of "large primes" are composite or below fb_max (see [postprocessing.md](postprocessing.md) and the sqrt-failure sieve-bug reproduction notes) — a relation-hygiene issue only; such witnesses rarely match and composite-LP combines never reach the matrix.

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

1. `SieveStage()` sets the sieve threshold override: `siever_->setThresholdOverride(config_.lp1_bound)` (`orchestrator.cpp:4158`) — this raises the sieve threshold from the default ⌊log₂(F)⌋ to ⌊log₂(lp1_bound)⌋, allowing candidates with one large cofactor ≤ `lp1_bound` to pass the scan.
2. `initLargePrimes()` (`orchestrator.cpp:3395`; called from `SieveStage` at `:4361`, guarded `if (!cluster_mode)` — invariant I1) resolves `LargePrimeConfig` and creates the `LargePrimeVariant` on the postprocessor's CUDA stream:
   - `max_combined_output`: CLI **`--lp1_combined_buf`** (config field `lp1_max_combined_output`), default 2^15 (32K) (`orchestrator.cpp:3405-3407`).
   - `max_witness_capacity`: `--lp1_max_witnesses` (snapped up to a power of two), else auto-derived — base `4·fb_size`, scaled by an LP-to-FB-ratio multiplier `min(4, max(1, log₂(L/F/10)))`, snapped to a power of two clamped to [2^18, 2^24].
   - `hash_bits`: `--lp1_hash_bits`, else `log₂(witness_capacity) − 4`.
   - `purge_after_match = (fb_size < 20000)`; `capture_provenance = --dump_combine_provenance`.

```cpp
// orchestrator.cpp — shared LP init (serves both batch and legacy)
config_.lp_config = lp_conf;
largeprime_ = std::make_unique<mpqs::lp::LargePrimeVariant>(postprocessor_->getCudaStream());
largeprime_->initiate(config_.lp_config, config_.N);
```

### CLI flags (as accepted by `tests/cuda-mpqs.cpp`)

| Flag | Config field | Default | Effect |
|------|--------------|---------|--------|
| `--lp1_bound <L>` | `lp1_bound` | 0 | LP bound; **0 disables the whole module** (`:498`) |
| `--lp1_max_witnesses <N>` | `lp1_max_witness_capacity` | 0 = auto | Witness-SoA capacity, snapped to a power of two (`:506`) |
| `--lp1_hash_bits <B>` | `lp1_hash_bits` | 0 = auto (`log₂(cap) − 4`) | Directory size `2^B` (`:920`) |
| `--lp1_combined_buf <N>` | `lp1_max_combined_output` | 0 = auto (2^15) | Combined-output buffer capacity (`:916`) |
| `--lp_interval <N>` | `lp_interval` | **1** | Batches between LP dispatches; `0` = adaptive (`:514`) |
| `--lp_matrix_threshold <f>` | `lp_matrix_threshold` | 0.01 | **DEPRECATED** alias for `lp_preprocess_threshold`; itself inert (`:532`) |
| `--dump_combine_provenance` | `dump_combine_provenance` | false | Enables `capture_provenance` (`:428`) |

⚠ There is **no** `--lp1_max_combined_output` flag — the config field of that name is set by
`--lp1_combined_buf`. `--cuda_graph_lp_stride` / `--cuda_graph_capture` are documented under
[CUDA graph capture](#cuda-graph-capture-106).

### Batch Sieve Path (double-buffered, async)

In batch mode (`sieve_batch_size > 0`), LP partials are processed **periodically** every `effective_lp_interval` batches via the **async pipeline**. The cadence is `(config_.lp_interval > 0) ? config_.lp_interval : adaptive_lp_batch_interval_` (`orchestrator.cpp:5497-5501`), and **`lp_interval` defaults to 1** (`include/orchestrator.h:120`) — so the shipped default is **one LP dispatch per batch**, not the adaptive schedule. The adaptive path (`adaptive_lp_batch_interval_`, initially 10, recalibrated at 5 %/20 % progress, clamped to [1,100] — `orchestrator.cpp:5679`, `:5696`) is opt-in via `--lp_interval 0`. Per cycle: drain the previous in-flight LP (wait `lp_done_event_` → `launchDeviceAppend` → `resyncPersistentDualCounter`), then launch `processAndCommitAsync()` on the freshly-swapped partial batch and continue sieving immediately. A final drain + flush processes remaining partials after the batch loop exits. The CUDA-graph path mirrors this with its own `lp_graph_pending` state; note that when LP is **inside** the graph (scope `full`) the in-graph cadence is set by `--cuda_graph_lp_stride`, and **`--lp_interval` then governs only the non-graph tail** (`orchestrator.cpp:5067-5075`).

### Legacy Sieve Path (single-cube)

In legacy mode (`sieve_batch_size == 0`), the synchronous `processAndCommit()` is called on every postprocessor buffer-full event. LP telemetry is polled asynchronously for progress reporting and witness fill projection.

### TruncatedSieveRun (autotune probes)

The truncated sieve run (`orchestrator.cpp:6390`) **fully initializes LP** when `config_.lp1_bound > 0`: threshold override (`:6437`), postprocessor partial buffers, hash table, and witness SoA. LP `processAndCommit` is called during buffer-full events, and LP telemetry is fed to the `LPFillProjector` for witness fill estimation.

## Autotune Integration

### LP Bound Selection

LP bound (`lp1_bound`, denoted L) is a **first-class search variable** of the autotune joint (F,L) optimizer (`SieveParameterOptimizer`, `sieve_optimizer.cpp` — L-shaped path A/B/C: local refinement, pattern probes, gradient descent; coordinate descent retained for legacy callers). L values are quantized to 1M (`quantize_L`) with floor `L_min(F) = max(1M, 10·F)`.

**Theory fallback (cold start, no history)** — `ParameterProjector::theoryFallback` (`autotune_projection.cpp:171`, L seeded at `:185-188`):
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
(`runtime_estimator.cpp`, LP birthday model `:128-177`) projects the witness
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
