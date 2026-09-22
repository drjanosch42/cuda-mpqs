# Sieve Module (`src/sieve/`)

In-tree flattened folder (demoted from a former Git submodule in Stage A — only `src/linalg` remains a submodule). GPU-accelerated SIQS polynomial sieving with two execution modes: legacy host-driven and batch GPU-only. Mode selection is controlled by `sieve_batch_size`: 0 (default) selects legacy, any positive value selects batch.

Large prime (LP) support is orthogonal to pipeline selection — both legacy and batch modes support LP when the orchestrator sets a threshold override via `setThresholdOverride()`.

The batch sieve-and-scan kernel additionally exists in three **accumulator widths** (legacy uint8, wide uint16, wide saturating-uint8) — the RSA-155 dual-path fork; see *Dual-Path Sieve Accumulator* below. The legacy uint8 kernels were byte-for-byte untouched by the fork (enforced by a source byte-identity check in the development tree). ⚠ **v1.0.7 deliberately modifies the two narrow GATHER kernels** (small-prime mask, backward-sieve restructure), so that byte-identity check fails against any pre-v1.0.7 baseline by design and must be re-anchored at v1.0.7; the wide kernels remain untouched.

Namespaces: `mpqs::sieve` (all sieving structures and kernels), `mpqs::postprocessing` (DoubleBuffer).

**Current as of v1.0.7 (2026-09-18).** Unlike v1.0.6, v1.0.7 **does change `kernel.cu`** — but
only the two **narrow** GATHER kernels, `sieveAndScanKernel` (legacy) and `sieveAndScanBatchKernel`
(batch); the SCATTER kernels and both wide kernels (`...Wide`, `...WideU8Sat`) are untouched, so the
RSA-150/155 production path (wide/u8sat) is unaffected by construction. v1.0.7 adds, **always on**
for both narrow GATHER kernels, the small-prime log mask and a restructured backward sieve; and, as
**opt-in** features, the 11-field `--params11` pin, the tuning-complex parameter search
(`--param_test`, the old grid search renamed `--param_test_legacy`) and `--sieve_offsets_global`. It
also tightens `validateConfigs()` (band ordering, exact coverage, launch-true GATHER shared memory).
See *v1.0.7 — Small-Prime Mask, Backward Sieve, `--params11` and the Parameter Search* below; the
v1.0.6 paragraph that follows is kept as the record of that release.

**v1.0.6 (2026-08-25).** Across the whole of v1.0.6 `src/sieve/kernel.cu` has an
**empty diff** — every change landed in `device_sieving_controller.{h,cpp}`, `sieve_memory_model.h`
and one `kernel.cuh` declaration. The extension of CUDA-graph capture to post-processing and solo LP
touched **no file in this module at all**; the graph is assembled by the orchestrator out of the
same `prepareSievingBatchFromStaged()` / `setJobArrays()` / `runSievingBatch()` calls documented
below (see [orchestrator.md](orchestrator.md)). What this module gained is SM-aligned launch
geometry, two shared-memory geometry overrides, narrow bucket telemetry, the narrow-batch coverage
invariant, and a fix for a pinned-staging overwrite race. All new geometry knobs are **default-off,
narrow-batch only and `--params`-gated**: with none of them passed, the derived geometry is
byte-identical to v1.0.5 on every path. The two **defect fixes** in this module (the pinned
index-staging race and the loaders' zeroing of all four `batch_size` fields) are *not* opt-in and do
change observed relation counts — see the notes below before scoring an A/B.

## Files

| File | Purpose |
|------|---------|
| `kernel.cu` / `kernel.cuh` | All CUDA kernels (legacy + batch variants + the wide/u8sat accumulator forks), device math helpers, polynomial/root helpers, host launch wrappers. v1.0.6 left `kernel.cu` untouched (its only change was a `kernel.cuh` *declaration* of `globalMetaSieveBatchKernel` for the occupancy preflight). **v1.0.7 changes the two narrow GATHER kernels only** — mask initialisation, the restructured backward sieve, the `recordBackwardFactor()` helper (`kernel.cu:133`) and the optional global-memory offsets slice — plus `loadSievingData` / `loadSievingDataParamTest` (allocate `dev_sieveOffsets`) and a host-side GATHER launch-error check in `runSievingBatch` (`kernel.cu:2879`) |
| `dynamicMask.cuh` | **v1.0.7.** Small-prime log mask: `generateMask()` (host; builds the periodic mask and the CRT basis, uploads both) and `findMaskOffsets()` (device; the per-sieving-block phase of the mask for each of the two root classes). See the v1.0.7 section |
| `sieving_data_structs.h` | All data structures: primes, candidates, contexts, configs, `gpuInfo`, `DoubleBuffer`; `MAX_SHC_DIM` |
| `sieve_memory_model.h` | Single source-of-truth device-memory model: `sieveBucketBudget()` (`kSieveBudgetNum/Den` = 4/5 = 0.80·VRAM), `estimateSieveFootprint()`, `reduceNumPolysToBudget()`, `clampWideNumPolys()`/`kWideNumPolysCap` = 512 — mirrors the ten `cudaMalloc` calls in `kernel.cu` (loadSievingData), computed in 64-bit (fixes the 32-bit product wrap that OOMed M=262K/RSA-140). **v1.0.6** also puts the header-only narrow-batch geometry predicates here — `narrowBatchCoverageOk()`, `admissibleSieveBlockSize()`, `admissibleBigPrimeStart()` — so the CLI, `validateConfigs()` and the `sieve_geometry_overrides` unit test share exactly one statement of each rule |
| `device_sieving_controller.h` / `.cpp` | Main API class: initialization, execution, batch orchestration, config loaders, accumulator-width dispatch, autotune probe harness, state management, snapshot / cluster hooks |
| `prime_algorithms.cu` / `.h` | Factor base generation, Tonelli-Shanks, Hensel lifting, hypercube walk, batch index preparation; 64-bit number-theory primitives (`Tonelli_Shanks_u64`, `jacobi_u64`, `is_prime_u64`) for branch-fixed character columns |
| `graycode.cuh` | Gray code enumeration: `gray()`, `advanceGray()`, `grayBitToFlip()` (all `__host__ __device__`) |
| `common.h` | `factoringData` struct (`mpqs::sieve` sieving state) and `AFactorsSnapshot` |
| `json_helper.h` | Minimal `JSONString` / `JSON_IO` builder used only by the optional debug-snapshot path |
| `debug_dump.cu` / `.cuh` / `.h` | GPU debug snapshot tooling |
| `README.md` | Module overview and integrated-build notes (no standalone build path) |
| `CMakeLists.txt` | Integrated build configuration (declares the `mpqs_sieve` static library) |

## Key Data Structures

### primeDataSIQS -- per-prime GPU data (20 bytes)

| Field | Type | Description |
|-------|------|-------------|
| `p` | `uint32_t` | The prime |
| `r` | `uint32_t` | Root r with r² = N (mod p) |
| `mod_inverse_a` | `uint32_t` | a⁻¹ mod p |
| `inv_aN` | `uint32_t` | a⁻¹ · r mod p |
| `inactive` | `uint32_t` | Set to 1 by `markInactivePrimesKernel` iff the prime divides a (skipped for this a); zero-initialized (0 = participates in sieving). Kernels consume it multiplicatively: `log2p·(1−inactive)` and `p·(1−2·inactive)` |

**B-values are decoupled from this struct.** The per-prime Gray-code update values B_k · a⁻¹ mod p live in a separate device array `devicePointers::dev_primeBValues`, **column-major** `[k*fb_size + primeIndex]`, sized `fb_size * shc_dim * sizeof(uint32_t)` (the *actual* run shc_dim, not `MAX_SHC_DIM`). The hot meta-sieve kernels stream `primeDataSIQS` once per factor-base prime per pass; embedding `B_values[MAX_SHC_DIM]` inflated the stream stride (84→148 B at `MAX_SHC_DIM`=32) and made the sieve bandwidth-bound on padding — the v1.0.4d struct-bloat regression. Decoupling keeps the struct at 20 B for all sizes and reads only the live shc_dim B-values, coalesced.

`MAX_SHC_DIM = 32` (`sieving_data_structs.h:33`) bounds only per-thread B-value caches and the separate array headroom (RSA-140 needs shc_dim=19; 32 covers RSA-155 and beyond).

### candidateRelation -- raw sieve output (sparse)

| Field | Type | Description |
|-------|------|-------------|
| `b` | `uint512` | Polynomial coefficient b (stored explicitly for stateless postprocessing) |
| `poly_id` | `uint32_t` | Gray code index identifying the specific b coefficient |
| `sieve_offset` | `int32_t` | x offset in sieve interval where candidate was found |
| `global_idx` | `uint32_t` | Unique index for debugging/tracking |
| `num_factors` | `uint32_t` | Count of small prime factors found |
| `factors[32]` | `uint32_t[32]` | Prime indices from trial division |

### DenseCandidate -- compacted candidate for postprocessing (336 bytes, 16-byte aligned)

| Field | Type | Description |
|-------|------|-------------|
| `a`, `b` | `uint512` | Polynomial coefficients |
| `true_x` | `int32_t` | Absolute x = startIndex + offset |
| `num_factors` | `uint32_t` | Total hint count (a-factors + sieve factors) |
| `factor_indices[48]` | `uint32_t[48]` | Merged, sorted factor indices |
| `_padding[2]` | `uint32_t[2]` | Padding for 16-byte alignment (total 336 bytes) |

Implements `operator<` (descending by `num_factors`) for Thrust sorting to improve warp convergence.

### relation -- fully reconstructed host-side relation

| Field | Type | Description |
|-------|------|-------------|
| `a_factors[16]` | `uint32_t[16]` | Indices of primes composing a |
| `axb` | `uint512` | Value ax + b |
| `factors[64]` | `uint32_t[64]` | Full factor list |
| `num_factors` | `uint32_t` | Factor count |

### DeviceConstants -- immutable GPU-resident constants

| Field | Type | Description |
|-------|------|-------------|
| `dev_N` | `uint512*` | N (512-bit) |
| `dev_factorBase` | `uint32_t*` | All factor base primes |
| `dev_rootN` | `uint32_t*` | Square roots of N mod each prime |
| `num_primes` | `uint32_t` | Factor base size |

### DeviceHypercubeContext -- per-batch GPU state

| Field | Type | Description |
|-------|------|-------------|
| `current_a` | `uint512` | Fixed coefficient a for this batch |
| `dev_a_factors` | `uint32_t*` | Prime indices composing a |
| `dev_b_components` | `uint512*` | Precomputed B-components for Gray code b-construction |
| `dev_primeData` | `primeDataSIQS*` | Per-prime roots/inverses for current a |
| `dev_globalBucketEntries` | `uint64_t*` | Large-prime bucket entry storage |
| `dev_globalBucketCounts` | `uint32_t*` | Bucket fill counters |
| `dev_indexToCandidate` | `uint32_t*` | Sieve-to-candidate index map |
| `dev_candidateRelations` | `candidateRelation*` | Raw candidate output |

### devicePointers -- legacy pointer aggregate (used by legacy kernels)

Flat struct bundling all device pointers including `dev_a_factors`, `dev_B_values`, `dev_factorBase`, `dev_primeData`, `dev_primeBValues` (decoupled column-major B-values, see above), `dev_rootN`, bucket arrays, candidate arrays (`dev_indexToCandidate` is `int32_t*` here), and batch job arrays (`dev_job_a_array`, `dev_job_B_flat`, `dev_job_factor_indices`, `dev_blockRelationCounts`), plus postprocessing integration fields (`dev_pp_accumulation_buffer`, `dev_pp_counter`, `pp_max_capacity`). Marked for future replacement by `DeviceConstants` / `DeviceHypercubeContext`.

### DoubleBuffer -- sieve-to-postprocessor handshake (in `mpqs::postprocessing`)

| Field | Type | Description |
|-------|------|-------------|
| `d_data` | `DenseCandidate*` | GPU candidate payload buffer |
| `d_counter` | `uint32_t*` | GPU atomic fill counter |
| `capacity` | `uint32_t` | Maximum candidates before overflow |
| `safe_to_write_event` | `cudaEvent_t` | Sieve stream waits before overwriting (postprocessor done) |
| `safe_to_read_event` | `cudaEvent_t` | Postprocessor stream waits before reading (sieve done filling) |

Cross-stream synchronization without CPU stalls: events preserve DAG execution order between the sieve stream and the postprocessing stream.

### factoringData -- host-side factorization state (`common.h`)

| Field | Type | Description |
|-------|------|-------------|
| `N`, `a_target`, `a` | `uint512` | Number to factor, target a magnitude, current a |
| `B_values` | `vector<uint512>` | B-component values for Gray code traversal |
| `factorBase`, `rootN` | `vector<uint32_t>` | Primes and their square roots mod N |
| `a_factors` | `vector<uint32_t>` | Indices of primes composing current a |
| `current_a_index`, `max_a_index` | `uint32_t` | Hypercube walk position |
| `lowerHalfStart`, `upperHalfStart` | `uint32_t` | Prime selection range boundaries |
| `M`, `F`, `size` | `uint32_t` | Sieve interval radius, FB bound, FB actual size |

### Configuration Structs

Five config structs control kernel launches (the first four carry `batch_size`):

- **`initConfig`**: `num_threadsPerBlock`, `num_threadBlocks`, `batch_size`
- **`generalSievingConfig`**: `sievingBlockSize`, `log2_sievingBlockSize`, `num_polysPerSieveCall`, `num_subCubes`, `num_sievingBlocksPerSieveCall`, `num_sievingBlockBatches`, `batch_size`, `globalBucketSize`, `bigPrimeStartIndex`, `midPrimeStartIndex`, `maxRelationsPerBlock`
- **`globalMetaSieveConfig`**: `num_threadsPerBlock`, `num_threadBlocks`, `batch_size`, `maxActiveBucketsTotal`, `num_activeBucketsPerThreadBlock`, `polyBlockSize`, `log2_polyBlockSize`, `num_polyBlocksPerThreadBlock`, `log2_num_polyBlocksPerThreadBlock`, `num_activeBlocksPerCycle`, `num_metaSieveCycles`, `sharedMemReq`
- **`sieveAndScanConfig`**: `num_threadsPerBlock`, `num_threadBlocks`, `sharedMemReq`, `batch_size`
- **`processRelationsConfig`**: `num_threadsPerBlock`, `num_threadBlocks`
- **`fixedSievingParams`**: constant per factorization — `fb_size`, `shc_dim`, `M`, `approxPolyRoot`, `threshold`
- **`dynamicSievingParams`**: changes per step — `a`, `log2_a`, `startIndex`, `subCube`, `newCube`
- **`polyData`**: `approxPolyRoot`, `log2_a`, `threshold` (passed to candidate scan)

`loadStandardConfig()` sets defaults (see *Standard Config Geometry* below); tuned 8-parameter tuples go through `loadPartialCustomConfig()`; individual configs can be overridden via `setConfig()`. The autotune Stage-1 optimizer seeds its search from `HEURISTIC_DEFAULTS = {512, 8, 4, 8, 256, 1024, 256, 1024}` (`src/autotune/kernel_param_optimizer.h:37` — the `Params8` tuple `{subCubeSize, numIntervals, polyBlockSize, blocksPerCycle, metaGridDim, metaBlockDim, sasGridDim, sasBlockDim}` consumed via `loadPartialCustomConfig`).

## Polynomial Memory Hierarchy

The sieve decomposes work into nested partitions:

```
Polynomial space: hypercube -> subCubes -> polyBlocks -> polys
Sieve interval:   full interval -> SievingCycles -> sievingBlocks
fullPolyId  = [subCubeId | polyBlockId | polyId]
globalBucketId = [polyBlockId | polyId | cycle | sievingBlock]
```

One kernel launch covers one subCube and a given number of SievingCycles. This constrains the global metasieve bucket memory and shared-memory cache usage.

## SIQS Polynomial Selection

Q(x) = (ax + b)² - N where:
- **a** = product of `shc_dim` factor base primes, enumerated via **hypercube walk** (`init_a_factors`, `advance_a_factors`, `generateHypercubePath`)
- **b** = sum of B-component subsets via **Gray code** — successive polynomials differ by ±2·B_k where k = `advanceGray(index)`

### Gray Code (`graycode.cuh`)

| Function | Description |
|----------|-------------|
| `gray(i)` | G(i) = i ^ (i >> 1) |
| `advanceGray(i)` | Index of the bit that flips from G(i) to G(i+1) (= ctz(G(i+1)^G(i))) |
| `grayBitToFlip(i1,i2)` | Bit position that differs between G(i1) and G(i2) |

All three are `__host__ __device__ __forceinline__`.

### Device Polynomial/Root Helpers (`kernel.cuh`)

The root helpers take the decoupled B-value array explicitly (post struct-bloat fix): `(id, shc_dim, primeData, bvalues, primeIndex, fb_size, ...)`.

| Function | Purpose |
|----------|---------|
| `rootsFromPolyId(id, shc_dim, primeData, bvalues, primeIndex, fb_size, r1, r2)` | Reconstruct sieve roots for a specific poly ID from hypercube |
| `bFromPolyId(id, shc_dim, B_values, result)` | Reconstruct coefficient b from poly ID |
| `advanceRoots(id1, id2, primeData, bvalues, primeIndex, fb_size, r1, r2)` | Update roots when transitioning between Gray code states |
| `advance_b(id1, id2, B_values, b)` | Update b when transitioning between Gray code states |

### Device Math Helpers (`kernel.cuh` / `kernel.cu`)

| Function | Description |
|----------|-------------|
| `modAdd(a, b, m)` | (a + b) mod m, safe for a,b < m |
| `modSub(a, b, m)` | (a − b) mod m, safe for unsigned a,b < m |
| `modSub_shifted(a, b, m)` | Returns result in [1, m] for computing positive sieve offsets |
| `modSum(a, b, m)` | (a + b) mod m where b is signed (\|b\| < m) — used for Gray code root updates |
| `log2(a)` | Floor log₂ via `clz32` |
| `align_up_to_hit(x, bound, p)` | Smallest y ≥ bound with y ≡ x (mod p) |
| `atomicByteAdd(array, idx, x)` | Byte-granularity atomic add via 32-bit word atomics (also as `ATOMIC_BYTE_ADD` / `ATOMIC_BYTE_ADD_RETURN` macros, `kernel.cu:42-46`) |
| `ATOMIC_HALF_ADD` / `ATOMIC_HALF_ADD_RETURN` | uint16-granularity atomic add macros (2×uint16 per 32-bit word, `kernel.cu:50-54`) — wide-accumulator counterpart of the byte macros; the byte macros are untouched |
| `atomicByteAddSat(array, idx, x)` | Saturating byte add: read-clamp-CAS on the enclosing 32-bit word, clamps at 255 without carrying into the adjacent byte lane (`kernel.cu:80-93`) — used only by the u8sat wide kernel's contended accumulation paths |
| `excludeNonRelations(...)` | Block-strided scan for threshold-exceeding candidates (`kernel.cu:850`); each candidate thread reserves its output slot via a per-thread `atomicAdd` on a shared counter (no warp-level intrinsics — see below). Backward trial division extracts factor lists. Overflow-safe (see below). Wide fork `excludeNonRelationsWide` (`kernel.cu:921`) is a near-verbatim copy whose only diff is the `uint16_t*` blockEntries width (enforced by a source width-diff check in the development tree) |

## 3-Kernel Legacy Pipeline (per polynomial step)

Selected when `sieve_batch_size == 0` (default). The host drives one polynomial at a time through three kernels:

1. **`initPrimeDataKernel`** — Compute a⁻¹ mod p for each factor base prime and write the per-prime B-update values B_k · a⁻¹ mod p into the column-major `dev_primeBValues` array (coalesced across the grid-stride threads at fixed k).
2. **`globalMetaSieveKernel`** — Pre-compute bucket offsets for large primes (index ≥ `bigPrimeStartIndex`). Stores `(offset | log2_p << 24)` entries (with the prime's factor-base index in the high 32 bits of the 64-bit bucket entry) via `atomicAdd` write heads into global bucket arrays, indexed by `globalBucketId`. **Nested-loop control flow** (see below).
3. **`sieveAndScanKernel`** — Forward sieve: adds log_p at arithmetic-progression offsets for small primes (shared memory byte array), applies large-prime buckets from global memory, then scans for threshold-exceeding positions and backward trial-divides to extract `candidateRelation` records. Annotated with **`__launch_bounds__(1024)`** to prevent the CUDA compiler from over-allocating registers, which would reduce occupancy or cause launch failures at 1024 threads per block.

Each kernel has a host wrapper function (same name without `Kernel` suffix) that configures grid/block dimensions and launches on a specified CUDA stream.

Additionally, **`markInactivePrimesKernel`** sets `primeDataSIQS.inactive = 1` for primes that divide a.

### sieveAndScanKernel Detail

The kernel operates in two phases per sieve block:

**Forward sieve phase** (v1.0.7 layout; identical in the batch kernel):
- **Mask primes** (index < `smallPrimesUsed`): the shared byte array (`blockEntries`) is no longer zeroed — it is initialised, one 32-bit word per thread iteration, from the precomputed periodic small-prime log mask (sum of the two root-class phases returned by `findMaskOffsets()`), after which these primes' offsets are advanced past the block end. See the v1.0.7 section.
- Small primes (`smallPrimesUsed` ≤ index < `midPrimeStartIndex`): walked in **32-lane groups** — lane `threadIdx.x % 32` strides a prime's progression by `32·p`, and the block works on `blockDim/32` primes at once — accumulating with `ATOMIC_BYTE_ADD` and **no per-prime barriers** (≤ v1.0.6: one prime at a time across the whole block, plain adds, a `__syncthreads()` per prime).
- Mid-range primes (`midPrimeStartIndex` ≤ index < `bigPrimeStartIndex`): each thread handles its own prime via `ATOMIC_BYTE_ADD` (no inter-thread sync needed — disjoint access).
- Large primes (index ≥ `bigPrimeStartIndex`): applied from pre-computed global buckets filled by `globalMetaSieveKernel`.

This kernel family is referred to as the **GATHER** kernel in tuning reports (it gathers/dumps bucket entries and scans); the meta-sieve kernels are the **SCATTER** side (they scatter bucket writes).

**Candidate extraction phase (`excludeNonRelations`):**
- Block-strided over `blockEntries`: each thread evaluates one position, flagging `isCandidate` where the accumulated log-sum exceeds `approxPolyVal - threshold` (`approxPolyVal ≈ log₂|Q(x)|`), and overwrites `blockEntries[index]` with the boolean for the backward scan.
- **Per-thread slot reservation**: each candidate thread reserves its output slot with `atomicAdd(&candidateWriteHead, 1)` on a shared counter. The sieve deliberately uses **no warp-level functions** (`__ballot_sync` / `__shfl_sync`) for this compaction: at MPQS smoothness rates, candidate positions within a sieve block are too sparse for warp-level compaction to beat the simple per-thread atomic — atomic contention is already negligible when relations are rare — so the serial reservation is retained. (An earlier warp-ballot rewrite of `excludeNonRelations` was reverted in review and did not land.)
- Writes `candidateRelation` records (b, poly_id, sieve_offset, num_factors=0) for qualifying positions.
- **Overflow-safe clamping**: a candidate whose reserved slot `>= maxPerBlock` is dropped and its `blockEntries[index]` is reset to 0, preventing the backward scan from reading uninitialized `indexToCandidate`. The returned count is clamped so `candidatesFound` never grows past the per-block buffer limit `maxPerBlock`.

**Backward trial division phase** (v1.0.7 restructure; recording is `recordBackwardFactor()`, which uses `ATOMIC_BYTE_ADD_RETURN` on `blockEntries` as the per-candidate slot allocator):
- Mask primes (index < `smallPrimesUsed`): **candidate-driven** — the block strides over `newCandidates × smallPrimesUsed` (candidate, prime) pairs and records `p` iff `(offset_{1,2}[i] − x) mod p == 0`. Walking these primes' residue classes costs `2·SB/p` probes to service a handful of candidates; the direct test costs `2·nCandidates`.
- `smallPrimesUsed` ≤ index < `midPrimeStartIndex`: walked backward in the same 32-lane groups as the forward pass, **without** the two per-prime barriers ≤ v1.0.6 carried (the pass only reads offsets/primes and slot allocation is atomic).
- `midPrimeStartIndex` ≤ index < `bigPrimeStartIndex`: unchanged (one prime per thread).
- Large-prime bucket entries are similarly scanned for factor extraction (prime index stored in upper 32 bits of the 64-bit bucket entry).

### globalMetaSieveKernel Nested-Loop Control Flow

The meta-sieve (SCATTER) bucketing logic is structured as explicit nested loops (replacing an earlier state machine): **cycles → polyBlocks → primes → polys → offsets**. See `kernel.cu:399-530` and the batch variant `globalMetaSieveBatchKernel` at `kernel.cu:988-1127` — both share the identical loop body. Note the cycles loop wraps the whole factor-base primes loop, so capping cycles multiplies FB re-reads (which is why `--sieve_meta_cycle_cap` is a locality ablation knob, not a speedup lever). ⭐ **This nesting is load-bearing for tuning, not just an implementation detail: `num_metaSieveCycles = numIntervals / blocksPerCycle`, so any `numIntervals` change that is not matched by `blocksPerCycle` silently re-traverses the entire factor base (measured ×1.892 instructions at 2 cycles). See *`num_metaSieveCycles` — why the halving price existed, and how `blocksPerCycle` removes it*.**

1. **Cycles loop** — `num_metaSieveCycles` iterations; each covers `num_activeBlocksPerCycle` sieving blocks starting at `currentStart`.
2. **PolyBlocks loop** — `num_polyBlocksPerThreadBlock` iterations; zeroes the shared per-active-bucket write heads, computes `polyBlockId = blockIdx.x + num_threadBlocks * curPolyBlock`.
3. **Primes loop** — thread-strided over `currentPrimeIndex` from `bigPrimeStartIndex` to `fb_size` (stride `blockDim.x`). Each thread seeds `polyIndex = (threadIdx.x/32) % polyBlockSize`, so the warp lane group selects its starting Gray-code polynomial. `polyId = gray(polyIndex)`, `fullPolyId = fullPolyIdPrefix | polyId`, and the per-prime roots are reconstructed via `rootsFromPolyId`. `maxOffsetCount` is computed from the *first* prime in the block to keep the inner loop length warp-uniform (constant trip count across lanes).
4. **Polys loop** — `polyBlockSize` iterations. Each iteration emits this polynomial's bucket entries, then advances to the next Gray-code state: `polyIndex = modAdd(polyIndex, 1, polyBlockSize)`, recompute `polyId`/`fullPolyId`, and `advanceRoots(prevFullPolyId, fullPolyId, ...)` performs the O(1) root update.
5. **Offsets loop** — for each of the two roots (`offset1`, `offset2`), strides by `+p` across `maxOffsetCount` hits, mapping each hit to a `sievingBlockHit` and atomically reserving a slot in the corresponding active bucket (drops the entry if the bucket is full).

The `polyIndex` cursor is **warp-uniform** by construction (seeded from `threadIdx.x/32`), so all lanes in a warp share the same polynomial and Gray-code transition, avoiding divergent root reconstruction.

**64-bit bucket indexing (overflow fix):** the bucket-write index is formed as `long long globalIndex = ((long long)globalBucketIdPrefix + sievingBlockHit) * globalBucketSize + index` (`kernel.cu:490` legacy / `:1087` batch), and the fill-level writeback uses a `uint64_t i_globalBucketId` (`:521` / `:1118`). The original 32-bit multiply wrapped once total bucket entries exceeded 2³² (reachable on >40 GB GPUs at large M), silently corrupting bucket writes — root cause of the RSA-155 H100 M-sweep zero-yield.

**Bucket-overflow flag:** the per-bucket fill count is encoded as `min(amountWritten, globalBucketSize) | (overflowed ? 0x80000000 : 0)` (`:523` / `:1120`). The GATHER kernels mask bit 31 off when dumping, so overflow (silently dropped hits past `globalBucketSize`) is otherwise invisible; `DeviceSievingController::getBucketOverflowStats()` surfaces it host-side (wide paths only).

### Extended Shared Memory (`sharedMemPerBlockOptin`)

Per-block shared-memory budget is taken from `prop.sharedMemPerBlockOptin - 1024` (`getDeviceInfo`, `kernel.cu:712`) rather than the classic 48 KB static limit, so the sieve can opt into the architecture's full dynamic shared-memory window (`gpuInfo.maxSharedMemPerBlock`). Kernels requesting more than 48 KB obtain it via the dynamic shared-memory launch argument (`sharedMemReq` in `globalMetaSieveConfig` / `sieveAndScanConfig`) after a `cudaFuncSetAttribute` opt-in; the host caps `sharedMemReq` against this opt-in ceiling.

## Dual-Path Sieve Accumulator (uint8 / uint16 / saturating-uint8)

At RSA-150/155 scale a full smooth accumulates `log₂|Q/a| − δ ≈ 258–274` in the forward-sieve accumulator; the legacy `uint8` `blockEntries` caps at 255 and **wraps**, silently rejecting ~98–99.5 % of full smooths (RSA-140's ~250 stays under the cap — unaffected). The fix is a dual-path fork; **the fork left the legacy kernels byte-for-byte untouched** (enforced by a source byte-identity check in the development tree; v1.0.7 later changed the narrow GATHER kernels on purpose — see the v1.0.7 section).

**Kernels** (all batch-mode; there is no non-batch wide kernel):

| Kernel | Accumulator | Notes |
|--------|-------------|-------|
| `sieveAndScanBatchKernel` (`kernel.cu:1224`) | `uint8`, wrapping | Legacy narrow path; carries all validated ≤RSA-140 records. v1.0.7: small-prime mask + restructured backward sieve |
| `sieveAndScanBatchKernelWide` (`kernel.cu:1584`) | `uint16` via `ATOMIC_HALF_ADD*` | Near-verbatim width-only fork (+ `excludeNonRelationsWide`); doubles the `blockEntries` shared-memory footprint, halving SB |
| `sieveAndScanBatchKernelWideU8Sat` (`kernel.cu:1898`) | `uint8`, **saturating** at 255 | Copy of the narrow kernel whose five forward-accumulation sites saturate (via `atomicByteAddSat` on contended paths, clamped store on the single-owner mid-prime path); restores SB to narrow's full width |

**Host dispatch** (`DeviceSievingController::initiate()`):
- `use_wide_accumulator_` — computed from an exact 3-point evaluation of `approxPolyVal` at x ∈ {−M, 0, M−1} (mirroring the device arithmetic): `use_wide = (APV_max + WIDE_MARGIN) >= 256` with `WIDE_MARGIN = 4` (`device_sieving_controller.cpp:119-132`). Auto: uint16-family for RSA-150/155, uint8 for ≤RSA-140. CLI override `--sieve_accumulator {auto|u8|u16}` (`setAccumulatorMode`).
- `wide_u8sat_selected_` — within the wide regime, the saturating-uint8 kernel is selected iff the hardened exactness gate `max_target = APV_max − threshold ≤ 252` holds (254 minus a 2-count margin for non-pow2-M sampling slop and per-poly `a.msb()` drift; `device_sieving_controller.cpp:168-190`). Under the gate, saturating-uint8 candidate selection is bit-for-bit identical to uint16 (`stored = min(255, sum)` ⇒ `(stored > target) ⇔ (sum > target)` for every target ≤ 254). CLI override `--wide_accum {auto|u8sat|u16}` (`setWideAccumMode`); a forced u8sat still honours the gate (falls back to uint16, never to the wrap bug).
- `runSievingBatch(..., use_wide, wide_u8sat)` (`kernel.cu:2687-2726`) selects the kernel via a launch ternary; when `use_wide == false` the launch is character-for-character the pre-fork legacy launch.
- `accumElemBytes()` (1 for narrow/u8sat, 2 for uint16) is the single source of truth for every SB / `sharedMemReq` derivation.

## Standard Config Geometry (`loadStandardConfig`)

`loadStandardConfig()` (`device_sieving_controller.cpp:1873`) derives the no-autotune/no-history geometry. Narrow-path values are byte-for-byte the historical ones; every wide deviation is gated on `use_wide_accumulator_`:

- **`sievingBlockSize` (SB):** narrow `pow2leq(3/4·maxShared)`; wide `pow2leq((3/4·maxShared − 3·1024·4)/accumElemBytes())` — the uint16 accumulator budget quarters SB vs narrow (2 bytes/entry + the reserved bigPrime floor), a structural throughput cost the autotune cannot touch; u8sat restores the 1-byte width.
- **`globalBucketSize` = SB/2** (legacy exact) unless the ablation knob `--bucket_size_factor <F>` is set, in which case `computeGlobalBucketSize()` returns `round(F·SB)` in both config loaders (F=0.5 reproduces legacy; F=1.0 doubles the bucket). The resized bucket is charged against the VRAM budget before allocation, so an over-large factor degrades `num_polysPerSieveCall` or is validator-rejected — never an OOM.
- **`bigPrimeStartIndex`:** wide `SB/32` (mirrors the tuned custom-path split); narrow keeps exactly 1024. `midPrimeStartIndex` = 32 (`:1940`) — ⚠ **still 32 in v1.0.7**, whereas `loadPartialCustomConfig` (the `--params` / autotune / history path, `:2241`) and `loadPartialCustomConfigDynamic` (`--params11`, `:2452`) use 96, so a bare or `loadStandardConfig`-seeded run gets the mask with a narrower small-prime walk band than a pinned run. (The `bigPrimeStart=SB/32` + `num_polys≤512` pair is the wide-path default-geometry fix that recovered 18× on H100.)
- **`num_polysPerSieveCall`:** seed `min(32768, 2^(shc_dim−1))`, wide-clamped to ≤ 512, then reduced by `reduceNumPolysToBudget()` until the bucket buffer (`num_polys · num_sievingBlocks · globalBucketSize · 8` B, computed in 64-bit — see `sieve_memory_model.h`) fits `0.80·totalGlobalMem`; `num_subCubes` absorbs the change so total polys sieved is unaffected. An optional autotune OOM-guard knob (`setMaxTotalSieveBytes`, default off) further halves the seed until the *entire* footprint (`estimateSieveFootprint().total()`) fits.
- **Meta-sieve (SCATTER) grid:** wide `2·pow2geq(multiProcessorCount)` (SM-aware — the hardcoded 64-block grid left an A100 at 12.5 % occupancy, 1.4–1.9× SCATTER win); narrow keeps exactly 64.
- **Wide-path `num_polys ≥ num_threadBlocks` validation floor — CLAMPED (shipped, `c1e49ac`, not an OOM).** The meta-sieve SCATTER grid `num_threadBlocks = 2·pow2geq(SMs)` is **512 on H100 (132 SMs), 256 on A100 (108 SMs)** and is fixed independent of `M`. Historically the config validator (`device_sieving_controller.cpp`, the `LEQ_CHECK`/`EQUAL_CHECK` block now at `:2105-2107`) required `num_polysPerSieveCall ≥ num_threadBlocks` (and `num_polyBlocksPerThreadBlock = num_polys/num_threadBlocks` to be a nonzero power of two), so when a large `M` drove `reduceNumPolysToBudget()` to degrade `num_polys` **below** that grid width the pipeline **aborted with a config-validation error before any sieve memory was allocated** — it was *not* an OOM and *not* a VRAM ceiling (VRAM could be almost entirely free at the abort). This is exactly why the H100 M-ladder's M=8M/bf2.0 leg (job 1986430) aborted: np degraded 512→256 < the H100 grid's 512. **The floor is now clamped away** (`device_sieving_controller.cpp:1461`, wide-gated on `use_wide_accumulator_`): `if (use_wide_accumulator_) gms_conf.num_threadBlocks = std::min(gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall);`, applied after the two np-budget reductions and before the `polyBlockSize` loop — every SM-derived block that still owns a poly is kept and the pow2/product/LEQ checks pass (the min of two powers of two is a power of two). The narrow (uint8, ≤RSA-140) path keeps its exact grid (64), byte-for-byte unchanged. **np-retention closed form** (still the relevant sizing guide — a degraded np now means fewer active SCATTER blocks and thus a *slower* run rather than an abort): with `bucket_bytes = 16·num_polys·bf·M` (SB cancels), `num_polys=512` is retained iff `16·512·bf·M ≤ 0.8·VRAM_total`, i.e. **`bf·M ≤ 9.76×10⁶` on H100** and **`bf·M ≤ 4.14×10⁶` on A100**. **Outcome since the clamp shipped:** M=16M (H100, np=256) was measured against M=8M (np=512) at F=300M and F=400M and lost on both rungs (−13.3%/−2.6% fulls/s respectively) — the np-halving penalty outweighs the M-doubling gain; the M-axis is closed at M=8M on H100 (2026-07-12 RSA-155 H100 M=16M analysis). Do not pursue M=32M.
- **GATHER launch:** `ss_conf.num_threadsPerBlock = 256`, `num_threadBlocks = min(256, num_polysPerSieveCall)`, `sharedMemReq = SB·accumElemBytes() + 3·bigPrimeStartIndex·4` (v1.0.7: the second term is dropped under `--sieve_offsets_global`, narrow only).
- **Ablation knobs** (both loaders, all default-off/byte-identical): `--sieve_meta_cycle_cap <N>` caps `num_activeBlocksPerCycle` at `min(pow2_floor(N), derived)` and raises `num_metaSieveCycles` correspondingly (SCATTER write-locality experiment; net-negative as a speedup — see loop-nesting note above); `--sieve_gather_block_dim <N>` overrides the GATHER blockDim (result-invariant: the shared accumulator is sized per sieving-block, not per-thread, and every work loop strides by `blockDim.x`).

## SM-Aligned Launch Geometry (v1.0.6 — narrow BATCH only, `--params`-gated)

Historically all eight `Params8` entries had to be powers of two. That was a **host convention, not a
mathematical requirement**: the only kernel-side pow2 dependencies on `{num_polysPerSieveCall,
metaGridDim, sasGridDim, num_polyBlocksPerThreadBlock}` live in the *legacy* GATHER mask
(`polyId & (np−1)`) and the legacy subCube partition, and the **batch** kernels read neither.
v1.0.6 therefore relaxes pow2 on those four quantities **on the narrow batch path only**, so the
SCATTER and GATHER grids can be set to the device's SM count and run exact waves.

**Scope and rejection.** The relaxation is gated in `validateConfigs()`
(`device_sieving_controller.cpp:1960`) on
`relaxed_geometry = (init_conf.batch_size > 0 && gs_conf.batch_size > 0) && !use_wide_accumulator_`:

- **legacy** (`--sieve_batch_size 0`) keeps mandatory pow2 — `kernel.cu` is never touched and a
  source byte-identity check keeps its baseline;
- **wide** (uint16 / u8sat) keeps mandatory pow2 — its bucket VRAM is proportional to `np` and the
  RSA-150/155 production regime already rides the np = 512 retention boundary.

The wide/legacy exclusion is applied by the **callers**, not by the validator helper:
`validateConfigs()` computes `relaxed_geometry` itself (`:1960`), and the autotune-side preflight is
told which rule set applies by the orchestrator, which passes
`(config_.sieve_batch_size > 0) && !siever_->isWideAccumulator()` into `preflightKernelLaunch()`
(`src/orchestrator/orchestrator.cpp:4320-4325`). `buildSieveConstants()`
(`src/autotune/kernel_launch_validator.cpp:419`) merely **accepts** `allow_nonpow2_geometry` as a
parameter and forwards it into `SieveConstants`; it applies no wide/narrow policy of its own, so the
gate lives in exactly two places and both are named above.

For any pow2 tuple the relaxed predicates are strict supersets of the old ones and the new invariants
are tautologies, so **no pre-v1.0.6 configuration changes behaviour**.

**Invariants that the pow2 world used to supply for free**, now stated explicitly (all in
`validateConfigs()`, mirrored in `src/autotune/kernel_launch_validator.cpp`):

| Tag | Invariant | Why it is load-bearing |
|-----|-----------|------------------------|
| G1 | `sasGridDim` must **divide** `num_polysPerSieveCall` | a non-divisor GATHER grid overlaps `polyIdPrefix`es *and* leaves a polynomial tail ungathered — silently wrong relations, no error |
| G2 | the GATHER chunk `np / sasGridDim` must be a **power of two** | batch GATHER composes `polyIdPrefix \| gray(poly)`; the OR equals addition only for a pow2 chunk |
| G3 | `sasGridDim ≤ np` | otherwise the trip count is zero ⇒ a silent zero-relation sieve |
| V5 | `num_polyBlocksPerThreadBlock · polyBlockSize · metaGridDim == np` (unchanged `EQUAL_CHECK`) | the batch SCATTER partition has a **fixed trip count and no `if (id < n)` guard**: under-coverage silently drops polynomials, over-coverage writes past the bucket array |

`polyBlockSize` stays pow2 **forever** (`advanceRoots`' cyclic Gray wrap is single-bit only for a
pow2 block, and the SCATTER seed composes `polyBlockId << log2_polyBlockSize`).

**Exactness-checked loader.** `loadPartialCustomConfig` no longer derives
`num_polyBlocksPerThreadBlock` by flooring twice (`(np/metaB)/polyBlockSize`). It now requires
`metaGridDim · polyBlockSize` to divide `np` exactly and, on failure, sets `custom_config_invalid_`
and emits `[Sieve] Invalid pinned geometry: …` naming the divisibility that failed;
`validateConfigs()` consumes and clears that flag so the single escalation path
(`LOG_ERROR_CRITICAL` + throw at the orchestrator) is preserved. **Nothing is ever silently
floored.** (`log2_num_polyBlocksPerThreadBlock` is dead state — written into
`globalMetaSieveConfig` but read by no kernel — so a non-pow2 value merely makes it stale, not
wrong.)

**Narrow-batch occupancy preflight.** When the tuple is *genuinely* SM-aligned (i.e. at least one of
`np`, `metaGridDim`, `sasGridDim` is not a power of two), `validateConfigs()` additionally runs a
launch-feasibility gate mirroring the wide one: GATHER shared memory
(`ss_conf.sharedMemReq + shc_dim·sizeof(uint512)`, matching `runSievingBatch`'s `sieve_smem`) against
`maxSharedMemPerBlock`, both grids against `cudaDevAttrMaxGridDimX`, and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` on `globalMetaSieveBatchKernel` and
`sieveAndScanBatchKernel` (≥ 1 block/SM required). Contract is **graceful** — `LOG_WARNING` +
`validFlag = false`, never a hard abort mid-probe. On success it logs
`[SM-aligned geometry] np=… SCATTER grid=… (occ … blk/SM => … waves/SM), GATHER grid=… …` at
`LOG_INFO`, so a run's utilisation claim is self-documenting rather than a paper calculation. The
block does not run for an all-pow2 tuple, so pre-v1.0.6 configurations execute byte-identical code.

> ⚠ **Observability limitation — the occupancy line is gated on *non-pow2*, not on *interesting*.**
> The gate is `relaxed_geometry && !(isPowerOfTwo(np) && isPowerOfTwo(metaGridDim) &&
> isPowerOfTwo(sasGridDim))` (`device_sieving_controller.cpp:2170-2173`). A tuple that is
> deliberately tuned for co-residency but happens to be **all powers of two** therefore gets **no
> `[SM-aligned geometry]` line at all**, and its achieved blocks/SM is never observed — it can only
> be *derived*. This is not hypothetical: the H100 halved arm `s3h`
> (`--params 1024,16,4,16,256,1024,256,512`) is **all powers of two**, so although it was designed
> around 2 blk/SM co-residency it printed no such line and that co-residency was **never observed** —
> only inferred, from an offline register measurement (56 regs/thread, 0 spills, `sm_90`). If you
> need the occupancy line for a pow2 tuple, there is currently no flag for it.

**Both loaders now zero all four `*_conf.batch_size` fields** (mirroring `setSievingBatchSize(0)`).
Before v1.0.6 only `init_conf` was zeroed, leaving `gs_conf`/`gms_conf`/`ss_conf.batch_size`
indeterminate in a legacy run — harmless while nothing read them, but `relaxed_geometry` does, and a
stale non-zero value would let a legacy run accept a tuple its GATHER mask cannot execute.

**Access is `--params`-only by design.** The autotuner was deliberately **not** taught these
geometries: `CANDIDATE_VALUES_*` / `HEURISTIC_DEFAULTS` in
`src/autotune/kernel_param_optimizer.h` remain powers of two and nothing proposes, projects or
auto-applies an SM-aligned tuple. Automatic discovery is a named but **unscheduled** follow-up.
SM counts are **never hardcoded** — an SM-aligned tuple is derived per device from
`multiProcessorCount` and must be recomputed for each GPU, never copied.

**Measured (A100, 108 SMs, RSA-100, narrow batch, job 34117883 — 3 arms × 3 reps, one binary, one
node).** `--params 864,8,8,8,108,1024,864,1024` sieved **106.453 s** vs **118.223 s** for the best
pow2 tuple and **128.863 s** for the previous pin: **−9.96 % / −17.39 %**, 49× the 0.240 s noise
floor, relations/polynomial identical to 4 s.f. across all three arms. `ncu` confirms the mechanism:
SCATTER `Waves Per SM` 0.590 → 2.370 → **1.000** and the idle-SM discount collapsing
0.5924 → 0.9826; board energy −6.84 %, above-idle energy −6.17 %. **Nothing is claimed for
RSA-150/155** — the wide path is excluded by construction.

### ⚠ `np ≤ 2^(shc_dim−1)` decides the SIGN of the SM-alignment lever — check it before deriving a rung

`validateConfigs()` caps the polynomial count against the hypercube dimension
(`device_sieving_controller.cpp:2107`):

```
LEQ_CHECK(gs_conf.num_polysPerSieveCall, (1u << (fs_params.shc_dim - 1)), validFlag);
```

An SM-aligned tuple must satisfy `np = SMs · polyBlockSize · num_polyBlocksPerThreadBlock` **and**
this cap, and the interaction is what makes the lever device-dependent:

| Device | SMs | `shc_dim` | `np` cap | reachable SM-aligned rung | effect on `np` |
|---|---:|---:|---:|---|---|
| A100 | 108 | 12 | 2,048 | `np = 864` (`108·8·1`) | **raises** np 512 → 864 |
| H100 | 132 | 11 | 1,024 | `np = 528` (`132·4·1`) — the next rung `132·8·1 = 1,056` exceeds the cap | **cuts** np 1,024 → 528 |

(`shc_dim` is **not** a device constant: it is auto-selected per run and forceable with
`--sieve_hc_dim`. The two values above are the ones the RSA-100 configurations actually resolved to
on each card — read `shc_dim` out of the run's own `[Config]` block before applying this table.)

So on H100 exact waves are bought with 48 % fewer polynomials per sieve call, doubling the batch
count. **This is measured, not projected: the 132-SM rung `528,8,4,8,132,1024,528,1024` LOSES —
+2.520 % Total / +3.634 % sieve** vs the pow2 tuple (job 2636199, 2026-08-26; +2.28 % / +3.42 %
drift-corrected). A100's −17.4 % result does **not** transfer. The lever's sign is set by `shc_dim`,
not by SM count — always check `np ≤ 2^(shc_dim−1)` before proposing an SM-aligned rung on a new
device. (The mechanism is source-derived and end-to-end measured; the kernel-level attribution
remains a hypothesis: profiling is unavailable on the H100 hosts measured, which set
`RmProfilingAdminOnly: 1`.)

**Rollback trap.** An `autotune_history.json` written by a v1.0.6-or-newer binary can carry a non-pow2
tuple that an older binary rejects **loudly** (not silently, and not corruptly). When downgrading,
delete the history file or pass `--autotune_no_history`.

## Narrow-Batch Shared-Memory Geometry Overrides (v1.0.6)

Three default-off knobs, all **narrow batch only**. The two geometry overrides additionally require a
pinned `--params` and are **forbidden with any `--autotune*` flag**; all three rejections happen
loudly at the CLI (`tests/cuda-mpqs.cpp`, the "cross-flag guards" block) and scope is re-asserted in
`validateConfigs()` (`:1979-1996`). Nothing is ever silently floored, clamped or downgraded.

| Flag | Config field | Default | Admissible | Effect |
|------|--------------|---------|------------|--------|
| `--sieve_block_size <N>` | `sieve_block_size` (u32) | `0` = off | `0`, or a **power of two ≥ 256**; additionally `≤ M` and within the shared-memory sum (both enforced downstream) | Replaces the `gs_conf.sievingBlockSize` derivation `min(M, pow2leq(¾·maxSharedMemPerBlock))` **inside `loadPartialCustomConfig` only** (`:1560-1561`). Applied at the derivation site, so `log2_sievingBlockSize`, `computeGlobalBucketSize(SB)` and `ss_conf.sharedMemReq` all follow automatically |
| `--sieve_big_prime_start <N>` | `sieve_big_prime_start` (u32) | `0` = off | `0`, or **strictly > 32** at the CLI; ⚠ **since v1.0.7 effectively `≥ midPrimeStartIndex` = 96** on this (`loadPartialCustomConfig`) path — `validateConfigs()` now rejects `midPrimeStartIndex > bigPrimeStartIndex` loudly, so 33…95 parse and then abort pre-sieve; **power of two NOT required**; additionally `≤ fb_size` | Replaces `gs_conf.bigPrimeStartIndex = sievingBlockSize/32` (`:1609-1610`). Cancels the unconditional SB↔bPSI coupling, so an SB override does not *also* silently migrate ~1024 factor-base primes from GATHER's in-block path onto SCATTER's bucketed path |
| `--sieve_bucket_overflow_stats` | `sieve_bucket_overflow_stats` (bool) | `false` = off | boolean, no argument | Diagnostic only. Relaxes the previously wide-only host reader `getBucketOverflowStats()` so the **narrow** path also emits `[BucketOverflow]` |

**Admissible-set rationale** (`sieve_memory_model.h`, shared with the unit test):
`sievingBlockSize` must be a power of two because the GATHER offset mask is `SB − 1` and
`validateConfigs` `POW2_CHECK`s it; `≥ 256` because an accumulator shorter than a warp tile is
degenerate. `bigPrimeStartIndex` must be **strictly** above 32 because the mid-prime loops run
`[midPrimeStart, bigPrimeStartIndex)` — at `N ≤ 32` that range **inverts and the whole mid-prime band
is dropped silently**. It need not be a power of two: every consumer is a grid-stride loop over
`[0,bPSI)`, `[32,bPSI)` or `[bPSI,fb_size)` and the GATHER shared-memory layout is plain pointer
arithmetic. The bounds that depend on run-time quantities unknown at parse time (`N ≤ M`,
`N ≤ fb_size`, and the sum `SB·accumElemBytes + 3·bPSI·4 + shc_dim·64 ≤ maxSharedMemPerBlock`) are
deliberately **not** checked at the CLI; they are enforced by `validateConfigs`' `LEQ_CHECK` on
`ss_conf.sharedMemReq` and by the narrow-batch occupancy preflight.

⚠ **v1.0.7: the "> 32" rationale above is stale.** `loadPartialCustomConfig` now sets
`midPrimeStartIndex = 96`, and the CLI predicate `admissibleBigPrimeStart()` (`sieve_memory_model.h`)
and the loader comment still say "> 32". The *silent* inversion is nevertheless closed: v1.0.7's
band-ordering invariant in `validateConfigs()` (`:2792`) rejects `midPrimeStartIndex >
bigPrimeStartIndex` (and `smallPrimesUsed > midPrimeStartIndex`) with `LOG_ERROR_CRITICAL`, so an
out-of-band value fails loudly before sieving — just later than at parse time.

### Coverage invariant — a silent half-sieve that was latent in every binary up to and including v1.0.5

`runSievingBatch()` — the **production batch entry** (`kernel.cu:2554`) — reads
`ds_params.startIndex` for every launch but **never advances it**; it is set once in `updateState()`
(`= −M`), and the per-batch re-offsetting that `sieveFullCubeSnapshot`, the legacy `sieveFullCube`
and `benchmarkSievingConfig` each perform does not happen there. So `gs_conf.num_sievingBlockBatches`
is a **fiction on that path** — yet the loader's `[C1]` guard (`:1650-1685`) multiplies coverage *by*
it on the narrow branch and therefore **passes** a configuration that sieves only `[−M, 0)`: a ~50 %
yield loss with no error and no warning. Worse, it can present as a **higher** relation count (a
smaller region suffers less `maxRelationsPerBlock = 64` truncation) — a signature that reads like a
win.

v1.0.6 adds the real invariant, stated on the **per-launch** coverage
(`narrowBatchCoverageOk()`, consumed in `validateConfigs()` `:2007-2026`):

```
num_sievingBlocksPerSieveCall × sievingBlockSize  ≥  2·M      (uint64 arithmetic)
```

`≥`, not `==`: over-coverage is legitimate and already blessed by the loader (the autotune M-sweep
drives `sievingBlockSize = min(M, …) == M` with the tuple's interval count fixed). Only
*under*-coverage is the silent half-sieve. The check is uint64 because `intervals × SB` overflows
uint32 at `M ≥ 2³¹`, which would wrap into a **false pass**. On violation the run aborts pre-sieve
with `LOG_ERROR_CRITICAL` naming the smallest `numIntervals` that restores coverage.

The check is **not** gated on the new overrides — it is a pre-existing defect, reachable on any
device whose shared-memory budget yields `SB < 2M / numIntervals`. It is a no-op for every
`loadStandardConfig` geometry (which derives `num_sievingBlocksPerSieveCall = 2M/SB` itself) and for
every shipped `--params` tuple on A100 / H100 / RTX 5070 Ti. The `[C1]` block is left textually
unchanged. **Status: confirmed in source, never reproduced in a live run** — no probe arm has yet
attempted an under-covering configuration.

⭐ **A documented configuration has now been caught carrying it.** The TITAN RTX (`sm_75`) autotune
alternative `512,8,8,8,32,1024,256,1024` is an under-covering tuple:
Turing's 64 KB opt-in budget gives `SB = pow2leq(¾·65,536) = 32,768`, so at `M = 262,144` the
coverage is `numIntervals × SB = 8 × 32,768 = 262,144 < 2M = 524,288` — **a silent half-sieve, and
the first concrete instance of this defect in a documented configuration anywhere in the project**.
It aborts loudly on v1.0.6 and later. It has **not** been executed (the TITAN benchmark's `autopin` arm is
off by default), so this remains a source-and-arithmetic finding, not a live reproduction; the
complementary half was confirmed live, in that the documented `rec` tuple lands exactly on `2M` on
3/3 reps.

⚠ **v1.0.7 tightens this to EQUALITY on both narrow paths.** `validateConfigs()` (`:2800`) now
requires `numIntervals × SB == 2M` exactly, in uint64, whenever the path is narrow — **batch and
legacy alike** (the batch-only gate is gone), and names the required `numIntervals = 2M/SB`.
Over-coverage is now rejected too: it sieves past `[−M, M)`, which is cheap per position and nearly
barren, so a timing probe (the parameter search in particular) would otherwise be walked into it.
Under-coverage on the *legacy* path (which does advance `startIndex` and would cover `2M` over
several launches) is rejected because the per-call work would be a config-dependent fraction of the
interval. ⚠ Consequences: (a) the "`≥`, not `==`" paragraph above describes v1.0.6 only; (b) the
autotune M-sweep's over-covering probes and legacy-path probes with `intervals × SB < 2M` — both
legal in v1.0.6 — are now rejected (they surface as failed/skipped probes, not as errors); (c)
`narrowBatchCoverageOk()` (`sieve_memory_model.h`, `≥`) no longer governs production — it survives
only in the `sieve_geometry_overrides` unit test and in comments. The wide path is unaffected (it
derives `numIntervals = 2M/SB` itself).

⚠ **Deliberate behaviour change (v1.0.6).** Configurations on smaller-shared-memory devices (Turing-class,
Jetson) that are half-sieving *today, silently* now fail **loudly**. The escape is `numIntervals` =
`--params` field 2. **Whenever `SB` changes, `numIntervals` must change with it so that
`numIntervals × SB ≥ 2M`.**

⚠ **Related `M` trap: `--sieve_bound` snaps to the NEAREST power of two** (`tests/cuda-mpqs.cpp:126-155`,
`:716-732`; ties round down — e.g. 3,145,728 → 2,097,152, which once invalidated two probe legs).
A snap emits `[CLI] Warning: --sieve_bound M=… is not a power of two; snapped to M=…`
on `stderr` (not through the logger — `HPCLogger` is not initialized until after argument parsing),
so it is no longer silent. **Always verify the realized `M` in the run's own `[Config]` block**, and
remember that a snapped `M` changes the `2M` the coverage invariant is measured against.

### The wide path has no `SB` knob — and one `--params` trap that the narrow path does not have

The v1.0.6 overrides are rejected on wide (`:1979-1996`), but the deeper reason they were never
offered there is that **wide derives the same quantities in closed form and self-heals two of them**:

| Quantity | narrow batch | wide |
|---|---|---|
| `sievingBlockSize` | `--params`-pinned tuple + optional `--sieve_block_size` override (`:1560-1561`) | **closed form, no knob** — `pow2leq(((¾·maxShared − 3·1024·4) / accumElemBytes()))` (`:1286`), with the override explicitly gated off by `!use_wide_accumulator_` |
| `numIntervals` (`--params` field 2) | honoured verbatim; the per-launch coverage invariant above is what keeps it honest | **silently overwritten** — `num_sievingBlocksPerSieveCall = 2M / SB` (`:1628`), so field 2 of a wide `--params` tuple is ignored, by design (tag `[C1 - CRITICAL]`: reusing a narrow tuple's interval count would cover only `[−M,0)`) |
| `num_activeBlocksPerCycle` | `--params` field 4 (`blocksPerCycle`) | `loadStandardConfig` **self-heals** it to `1 << countr_zero(num_sievingBlocksPerSieveCall)` (`:1464`) ⇒ `num_metaSieveCycles ≡ 1` |

⚠ **The trap: the self-heal is a `loadStandardConfig` property, not a wide property.**
`loadPartialCustomConfig` assigns `gms_conf.num_activeBlocksPerCycle = blocksPerCycle` verbatim
(`:1634`) on **both** paths. So under a pinned `--params` on wide, `numIntervals` is replaced by
`2M/SB` while `blocksPerCycle` is not — and since
`num_metaSieveCycles = num_sievingBlocksPerSieveCall / num_activeBlocksPerCycle`, a wide `--params`
tuple silently runs multi-cycle (re-traversing the whole factor base, see the `num_metaSieveCycles`
section below) unless it carries **`blocksPerCycle = 2M/SB`** exactly. Nothing warns about this; only
the `EQUAL_CHECK` on `num_metaSieveCycles · num_activeBlocksPerCycle == num_sievingBlocksPerSieveCall`
(`:2102`) fires, and only when the division is inexact.

⚠ **Terminology, exactly.** `use_wide_accumulator_` is the **path** flag, **not** the accumulator
width. Production RSA-150/155 run the **wide path at u8sat (1-byte) width**; uint16 is only the
fallback taken when the exactness gate `max_target ≤ 252` fails (`:168-190`), and the width dispatch
sits *inside* the wide branch. The v1.0.6 knobs are gated on the **path**, so they are unreachable
at RSA-150/155 regardless of width — forcing `--sieve_accumulator u8` to reach them is reachable but
**catastrophically wrong** (RSA-150 measures `APV_max = 269 > 255`, i.e. the original RSA-155
accumulator-overflow blocker, deliberately re-entered).

### `--bucket_size_factor 1.0` is a correctness requirement on any halved-`bPSI` arm

At **production** narrow geometry the knob is **free** — measured 0.00 % wall delta, relations
identical to ±1, identical absolute `max_fill` (v1.0.6 on RTX 5070 Ti + A100 job 34136780). But
halving `bigPrimeStartIndex` moves ~1024 factor-base primes onto the bucketed SCATTER path, and the
legacy `SB/2` bucket then goes from **81.6 % to 97.2 % fill** — 904 slots of headroom. On any halved-
`bPSI` configuration `--bucket_size_factor 1.0` is therefore **not** an ablation knob but a
correctness requirement.

> The earlier wording that `--bucket_size_factor` is "inert on the narrow path at RSA-140 and below"
> is **corrected (2026-08-25)**: the *knob* is live on narrow (`computeGlobalBucketSize` is
> width-agnostic); what was inert was its measured *effect* at RSA-140.

### First narrow bucket-fill data anywhere in the project

`--sieve_bucket_overflow_stats` produced the first narrow occupancy numbers ever measured here.
Production geometry (`SB` 65,536 / `bPSI` 2,048 / legacy bucket) peaks at **81.6 % fill on an RTX
5070 Ti and 81.8 % on an A100, at 0.00 % overflow on 24/24 reps** — confirming a figure that had only
ever been inferred. Fill is **`SB`-invariant** and depends on `bPSI` and the bucket factor only
(`fill = (2/bsf)·Σ_{i ≥ bPSI} 1/p_i`), so the 5070 Ti anchors transfer to A100 to ≤ 0.2 pp; bucket
VRAM is likewise `SB`-invariant at fixed factor. **Measured cost of the telemetry at production
geometry: ≤ 0.09 % of the sieve wall** — i.e. free. It stays default-off because it is diagnostic:
the read costs a `cudaMemcpyAsync` + `cudaStreamSynchronize` **on the siever stream** at the ~5 s
stats cadence, and narrow production is otherwise a zero-sync double-buffered pipeline. (Gating it on
the debug log level was considered and rejected — narrow production runs `--verbose --debug` for
other telemetry, so that would have silently turned the sync on in production.)

### ⚠ The optimisation these knobs were built for was REFUTED **at `blocksPerCycle = 8`** — and RECOVERED at `blocksPerCycle = 16`

> **SCOPE, up front.** The refutation below is real and is **specific to `blocksPerCycle = 8`
> (`num_metaSieveCycles = 2`)**. At `blocksPerCycle = 16` (1 cycle) the same halved footprint
> measures **−9.92 %**, not +6.27 %. Read this subsection for *why the price existed*, then the
> subsection **`num_metaSieveCycles` — why the halving price existed and how it was removed** below
> for *how it was removed*. Do not quote either half without the `blocksPerCycle` it belongs to.

The knobs exist to test one hypothesis: that halving the GATHER shared-memory footprint would put
**two co-resident blocks on every SM at production `M`** for the first time. **The mechanism is
confirmed; the optimisation is not — at `bpc = 8`.** On a 108-SM A100 at RSA-100, `M = 262,144`,
narrow uint8 batch, `cgu = 0`, 400 batches, pinned `--params`, 8 arms × n = 3 (job **34136780**):

- co-residency **alone** is worth **−7.40 s = −15.38 %** of the sieve wall (74× the noise floor,
  99.4 % GATHER closure; `ncu`: kernel duration −28.99 %, `BARRIER` −32.0 %, `issue_active/cyc`
  +26.5 %, occupancy limits both = 2, `waves_per_multiprocessor` = 4.0000);
- **reaching it costs +9.80 s = +25.59 %**, so the shippable configuration is **+6.27 % SLOWER** and
  **+28.28 % board / +35.21 % above-idle energy** worse than the production tuple;
- `bigPrimeStartIndex` is a wash (+0.49 %) — the win is co-residency and the price is `numIntervals`
  8 → 16, not the transition move.

**That `bpc = 8` halved geometry was NOT adopted**; `--params 864,8,8,8,108,1024,864,1024` remains
the A100 RSA-100 pin of record. The plan's model — that the price is doubled barrier density inside
GATHER — is **refuted**: GATHER duration is flat (+0.015 %) and its barrier stall *falls* 16.9 %,
explaining 0.04 % of the +9.80 s. The cost sits **outside** GATHER, in the non-GATHER per-batch
budget (SCATTER / bucket / post-processing), which rises **+24.4 ms/batch = +75.7 %**. That budget
had **never been profiled anywhere in this project** (every `ncu` pass to date was
`-k sieveAndScanBatchKernel`); the recommended two-arm `ncu` pass on `globalMetaSieveBatchKernel`
was subsequently **run on four arms** (job **34138776**) and the cost turned out to be **64.3 %
amortisation** — see the next subsection.

Note the lever's **sign depends on `M`**: at `M = 32,768` co-residency measured −6.40 % whole-sieve
(job 34135885) *without* any halving being needed to obtain it. Do not port any of this to H100 or to
RSA-150/155 speculatively — both the prize and the price are A100-specific.

### ⭐ `num_metaSieveCycles` — why the halving price existed, and how `blocksPerCycle` removes it

**Read this before ever re-deriving the halving idea, so that `blocksPerCycle = 8` is not measured
and the route re-closed a third time.**

The `+25.59 %` price above is **not** a property of `numIntervals`. It is a property of the **ratio**:

```
num_metaSieveCycles = num_sievingBlocksPerSieveCall / num_activeBlocksPerCycle
                    = numIntervals / blocksPerCycle
```
(`device_sieving_controller.cpp:1727`, inside `loadPartialCustomConfig` at `:1495`;
`num_activeBlocksPerCycle = blocksPerCycle` at `:1634`. The `loadStandardConfig` copy is at `:1477`.)

`globalMetaSieveBatchKernel` (`kernel.cu:988`) nests its **full grid-stride pass over
`[bigPrimeStartIndex, fb_size)`** (`:1038`) **inside** the cycles loop (`:1016`) — the loop nesting
already noted under *globalMetaSieveKernel Nested-Loop Control Flow*. So:

| `numIntervals` | `blocksPerCycle` | `num_metaSieveCycles` | factor-base traversals per launch |
|---|---|---|---|
| 8 | 8 | **1** | 1 (production) |
| 16 | 8 | **2** | **2 — the entire FB walked twice** |
| 16 | **16** | **1** | 1 |

At `nI 16 / bpc 8` the whole big-prime range (237,673 of 238,697 primes at `F = 7,000,000`,
`bPSI = 1,024`) is walked twice, and the `primeData` load + `rootsFromPolyId` + modular setup at
`:1044-1052` is paid twice. **Measured: `sm__sass_thread_inst_executed.sum` ×1.892** per SCATTER
launch. The compensating term `maxOffsetCount = (blocksPerCycle × SB)/p + 1` (`:1048`) halves per
cycle, so the *useful* per-hit work is conserved — **bank conflicts rise only ×1.144**. The per-prime
**fixed** cost doubles; the per-hit cost does not.

Setting **`blocksPerCycle = 16`** makes `num_metaSieveCycles = 16/16 = 1` and restores
`bpc × SB = 16 × 32,768 = 524,288`, i.e. production's per-cycle span. **Instructions return to
×1.047** and 6.3 of the 9.8 lost seconds come back — enough for co-residency's −17.9 ms/batch of
GATHER to beat the residual +8.2 ms/batch of SCATTER. Cost of the change: `sharedMemReq` for SCATTER
goes `num_activeBucketsPerThreadBlock × 4 = bpc × polyBlockSize × 4` = 256 → **512 B** — SCATTER sits
on the minimum 8,192 B carveout rung either way and never competes for L1.

Measured on A100 (job **34138776**, 108 SM, 4 arms × n = 3, 400 batches, `--sieve_only`,
`M = 262,144`, `cgu = 0`, `bs = 8`, pinned `--params`, work-identical at ±1, 0.00 % bucket overflow):

| arm | `--params` | overrides | cycles | blk/SM | wall |
|---|---|---|---|---|---|
| `s0` | `864,8,8,8,108,1024,864,1024` | `bsf 1.0` | 1 | 1 | **38.300 s** (the pin) |
| `s1` | `864,16,8,8,108,1024,864,1024` | `SB 32768`, `bPSI 1024`, `bsf 1.0` | **2** | 1 | 48.100 s (+25.59 %) |
| `s2` | `864,16,8,16,108,1024,864,1024` | same | **1** | 1 | 41.800 s (+9.14 %) |
| `s3` | `864,16,8,16,108,1024,864,512` | same | **1** | **2** | **34.500 s (−9.92 %)** |

`s1 → s2` is a **pure SCATTER effect** (−15.982 ms/batch of SCATTER against a −15.750 ms window =
101.5 % closure); `s2 → s3` is a **pure GATHER effect** (−18.491 vs −18.250 ms = 101.3 %). The
`s0 → s3` budget closes to **0.000 ms residual**: GATHER −17.856 + SCATTER +8.204 + other +0.151 =
−9.500 ms/batch.

⚠ **`s3` is a measured sieve-only candidate, NOT a pin.** It has **never been run end to end** (no
halved geometry has ever produced a matrix, a BW solution or a factor), and its energy is **+3.49 %
board / +7.63 % above idle** from **+13.1 % mean power** — it buys GPU-hours, not kWh. Adoption
would be an operator ruling, not a measurement, and it remains unadopted: the A100 RSA-100 pin of
record is unchanged.

⚠ **`--sieve_meta_cycle_cap` moves this same dial in the WRONG direction** — it *raises*
`num_metaSieveCycles`, and it is net-negative at every cap for exactly the reason above. It is a
locality-ablation knob, not a lever.

### ⚠ The solo GPU-LP path is ±1 nondeterministic

Measured on **both** architectures and on the *pre-change* baseline binary as well as after
(2026-08-25): localhost RTX 5070 Ti RSA-100 at `--cuda_graph_unroll 0`, n = 3, moves by **±1** on
`LP combined`, `Total (deduped)` and `Cumulative LP full`; A100 jobs 34135902 / 34136780 show the
same ±1 on relations and LP fulls *intra-arm* (= 0.0013 %). `Sieved full`, `Duplicates`,
`Batches processed` and the whole `[Config]` block **are** exact. **Consequence: work identity
between arms must be judged at ±1, never by exact equality** — an exact-equality acceptance criterion
cries wolf, and earlier "bit-identical" claims about this path (the `cgu=0` invariant, and
"localhost is bit-reproducible") are **overstated and superseded (2026-08-25)**. The jitter
itself is unexplained. The **legacy** path is not a bit-identity arm at all — it stops on a target at
a batch boundary; its real evidence is source byte-identity, checked directly against the
previous release's `kernel.cu`.

## v1.0.7 — Small-Prime Mask, Backward Sieve, `--params11` and the Parameter Search

Scope: the two **narrow** GATHER kernels and host configuration. The wide kernels, the SCATTER
kernels and the whole RSA-150/155 production path (wide/u8sat) are untouched.

### Small-prime log mask (always on, narrow GATHER, legacy and batch)

The smallest factor-base primes cost the most in the forward sieve (`SB/p` hits each) yet their
contribution to every sieving block is periodic. v1.0.7 precomputes it once
(`src/sieve/dynamicMask.cuh`, called from `loadData()`):

- **`generateMask()`** (host) takes the longest factor-base prefix `p_0 … p_{k−1}` with
  `4·∏p_i < 65,536`; `period P = ∏p_i` and `smallPrimesUsed = k`. It builds a `4P`-byte mask (one
  period viewed as `P` 32-bit words) holding `⌊log₂ p⌋` at every multiple of each `p_i` — the
  kernel's own log values, so relation sets are unchanged — and the CRT basis
  `e_i = (P/p_i)·((P/p_i)⁻¹ mod p_i) mod P`. Both go to **global** memory
  (`dev_smallPrimeMask`, `dev_CRT_baseElements`); no shared memory is used. At RSA-100 / F = 5.5M
  the mask primes are {3, 5, 13, 19}, `P = 3,705`, 14,820 B.
- **`findMaskOffsets()`** (device, once per sieving block) computes, for each root class,
  `shift = Σ e_i·(root_i mod p_i) mod P` — the unique residue mod `P` that is ≡ `root_i` mod every
  `p_i` — and the word offset of `sieveBlockStart − shift` (adjusted by whole periods to a multiple
  of 4; `P` is odd, so at most three additions).
- **Initialisation.** `blockEntries` is filled word-wise with
  `mask[(o₁+i) mod P] + mask[(o₂+i) mod P]` instead of being zeroed (per byte at most
  `2·Σ⌊log₂ p_i⌋`, far below 255, so the word add never carries between bytes). The mask primes'
  offsets are then advanced past the block end, and the forward walk starts at `smallPrimesUsed`.
- **Backward sieve.** Mask primes are recovered by testing the candidates directly (see
  *sieveAndScanKernel Detail*); the rest of the small band is walked in 32-lane groups without
  per-prime barriers.
- **Band ordering.** `smallPrimesUsed ≤ midPrimeStartIndex ≤ bigPrimeStartIndex` is now a
  `validateConfigs()` invariant (`:2792`); either inversion aborts loudly. `midPrimeStartIndex` is
  96 in both custom loaders but still 32 in `loadStandardConfig`.

**Measured (first version of the mask, `cf2264b`, all `cgu 0`, pinned `--params`, product-verified):**
RTX 5070 Ti RSA-100 at the `rec` tuple −7.76 % sieve / −6.02 % Total; H100 at the pin −14.1 % sieve,
board energy −6.79 %; A100 at the pin (coverage-exact `M = 262,144`) −21.3 % sieve, board energy
−11.23 %. ⚠ These predate the backward-sieve restructure and the application of the mask to the
legacy kernel (`10e7e1d`, `4624166`); the released kernel has not been re-measured on the cluster
GPUs. On H100 the halved `s3h` geometry reverses sign under the mask (+4.13 % wall); do not pursue it.

### `--params11` and `loadPartialCustomConfigDynamic()`

`--params11 <np,numIntervals,polyBlockSize,blocksPerCycle,metaGridDim,metaBlockDim,sasGridDim,sasBlockDim,SB,bPSI,midPS>`
— the first eight fields are exactly `--params`, the last three are `sievingBlockSize`,
`bigPrimeStartIndex` and `midPrimeStartIndex` (`enum SieveParam`, `device_sieving_controller.h`).
All 11 values must be given; a **0 means "unset"** and keeps the loader's derivation:

| Field | Derivation when 0 |
|---|---|
| `SB` | narrow `min(M, pow2leq(¾·maxSharedMemPerBlock))`; wide as in `loadPartialCustomConfig` |
| `numIntervals` | `2M / SB` (the coverage invariant) |
| `blocksPerCycle` | `= numIntervals` (one meta-sieve cycle) |
| `bPSI` | `SB / 32` |
| `midPS` | 96 |
| `np`, `pbs`, `metaGridDim`, `metaBlockDim`, `sasGridDim`, `sasBlockDim` | ⚠ **no derivation** — the loader keeps whatever the config structs already hold; pass these explicitly |

`globalBucketSize` is sized to the **predicted peak** bucket occupancy unless
`--bucket_size_factor` pins it: `μ = 2·SB·Σ_{i ≥ bPSI} 1/p_i` (`expectedBucketEntries()`), peak
`μ + z·√μ` with `z = √(2 ln(np·numIntervals))`, rounded up to a multiple of 32
(`autoGlobalBucketSize()`, `:2394`). The bucket therefore grows with a lower `bPSI` and shrinks with
a higher one; whether it fits is decided by the VRAM check in `validateConfigs()`. That check also
**warns** (`Bucket UNDERSIZED`, every loader) when `globalBucketSize` is below the predicted peak.

`--params11` is consumed by `SieveStage()` and `TruncatedSieveRun()` in the same `else if` as
`--params` and wins when both are set. ⚠ It does **not** get the `--params`-only safeguards: the
`preflightKernelLaunch()` check, the LP `sasGridDim` floor and the legacy `sasBlockDim ≤ 1024` cap are
all keyed on `useParams`; and the small-N adaptive branch (`max_polys < 64`) runs ahead of it. The
v1.0.6 `--sieve_block_size` / `--sieve_big_prime_start` overrides are not consumed by this loader
(and the CLI rejects them without `--params`); use fields 9–10 instead.

⚠ **`--params11` is NOT narrow-gated, and on wide it is an UNGUARDED second door.** No `useParams11`
site tests `use_wide_accumulator_` (`orchestrator.cpp:4283`/`:4312`, `:6471`), and this loader's wide
branches (`:2436`, `:2447`, `:2461`, `:2524`) resolve a usable wide configuration — so a wide tuple is
neither rejected nor ignored. But `set(P_SIEVING_BLOCK_SIZE, …)` (`:2444`) and `set(P_BIG_PRIME_START, …)`
(`:2450`) are **ungated**, so fields 9–10 reach `SB`/`bPSI` on the wide path, bypassing the
`validateConfigs:2774-2782` scope rejection that `--sieve_block_size` / `--sieve_big_prime_start` are
given there — a rejection whose stated reason is that wide's `SB` comes from a different budget solve
and its bucket VRAM rides the RSA-150/155 retention boundary. The `== 2M` coverage check is likewise
`!use_wide_accumulator_`-gated, so a wide tuple pinning `numIntervals` (field 2, `set()` at `:2472`)
inconsistently with `SB` is checked by **nothing**. **No wide `--params11` run exists on record** —
treat it as unvalidated and do not use it at RSA-150/155 until it is guarded or measured.

### `--sieve_offsets_global` (default off, narrow only)

Moves the GATHER `offsets1/offsets2/primes` arrays (`3·bPSI·4` B) from shared memory into a
per-block slice of `dev_sieveOffsets` (stride `gatherOffsetsStride(bPSI)`, 128-B aligned).
GATHER shared memory drops to `SB` (+ the batch `B_values`), which decouples `bPSI` from the shared
budget and can buy a second resident block. The slice is charged against the 0.80·VRAM budget.
Forced off on the wide path. **Unmeasured as a production setting.**

### `--param_test` — the tuning-complex parameter search (narrow only)

`runParamTest()` (`device_sieving_controller.cpp:864`) replaces the grid search (kept as
`--param_test_legacy` → `runParamTestLegacy()`). It is seeded from `--params11` if given, otherwise
from `loadStandardConfig()` with the SCATTER/GATHER grids raised to at least `pow2geq(SMs)`. Nine
axes are searched — `SB, np, sasGrid, pbs, bPSI, metaGrid, metaBlock, sasBlock, midPS`;
`numIntervals` and `blocksPerCycle` are not axes (derived from `SB` as above). Ladders: powers of two
plus SM-derived rungs for the grids and `np` (read from the device, never hardcoded); `bPSI` and
`midPS` on the lattice `smallPrimesUsed + 32k`, `bPSI` bounded by the shared-memory budget (or by
`fb_size/2` under `--sieve_offsets_global`).

- **Faces.** Axes that share a constraint are swept together over the full product of their
  ladders: `occupancy` {SB, bPSI, sasBlock}, `scatter-partition` {np, metaGrid, pbs},
  `gather-partition` {np, sasGrid}, `scatter-waves` {metaGrid, metaBlock}, `gather-waves`
  {sasGrid, sasBlock}, `big-prime-split` {bPSI}, `mid-prime-split` {midPS}.
  `--param_test_radius R` (default 3, must be ≥ 1) is the **maximum face dimension** swept, not a
  distance.
- **Scoring.** Each probe is `sieveMini()` over ≤ 2 subcubes through the **legacy** kernel (a
  deliberate design decision: the legacy kernel is cheap to probe and has so far proven representative
  of the batch kernels' optimal launch parameters, which the production run then uses), after one
  untimed warm-up step, normalised to µs per poly-position. Candidates are scored relative to the
  incumbent re-probed every 25 candidates; a face winner must beat the incumbent by ≥ 0.5 % in an
  alternating two-rep duel to be adopted. Cycles repeat (≤ 64) until no face improves, then one
  single-axis descent over full ladders runs; the log ends with the chain gain, the machine drift
  (seed re-probed) and the winner as a `--params11` line.
- **Pruned before launch:** grids below the SM count, more than one meta-sieve cycle, more than 2
  sieve calls per hypercube, and tuples whose **batch** GATHER footprint (with `B_values`) would not
  fit shared memory. Inadmissible tuples are rejected by `validateConfigsSilent()`; a tuple that
  fails at launch is rejected via `cudaGetLastError()`.
- **Output and exit.** The winner is printed, not applied: the process ends with `exit(0)` inside the
  sieve stage (an unusable seed ends with `exit(1)` after a loud `validateConfigs()`). On the wide
  path it logs an error and returns. Because the probe runs the legacy kernel while the output is
  meant for a batch run, the recommended tuple should be confirmed with a timed batch run.

### Validation changes (all paths unless noted)

- **Coverage** is an equality `numIntervals × SB == 2M` on both narrow paths (see the coverage
  section above).
- **GATHER shared memory** is checked as launched: `sharedMemReq + shc_dim·64 B` on the batch path
  (the `B_values` term was missing, so a pow2 tuple could pass validation and then fail to launch).
- **Launch failures are visible:** `runSievingBatch` checks `cudaGetLastError()` after the GATHER
  launch and logs `[Sieve] GATHER launch FAILED` at `LOG_ERROR_CRITICAL` (host-side, no sync). It
  does not abort.

## Batch Sieving (GPU-Only Mode)

Selected when `sieve_batch_size > 0`. Eliminates CPU-GPU synchronization in the inner loop by pre-uploading K polynomial configurations and running all steps on-device:

1. **`prepareSievingBatch()`** (host) — calls `prepareNextBatchIndices()` to advance `a_factors` for K steps on the host, then uploads flattened factor indices (via a pinned staging buffer for truly async H2D) to `dev_job_factor_indices` and launches `generatePolynomialsKernel` to compute all a and B values on-device.

   ⚠ **The pinned staging buffer is DOUBLE-BUFFERED and event-gated since v1.0.6** (`kPinnedIndexSlots = 2`, `device_sieving_controller.cpp:901-922` (allocate/release) and `:924-971` (`prepareSievingBatch`)). A single reused slot was a **correctness hazard**, not a performance detail: the host `memcpy` into the slot is immediate while the `cudaMemcpyAsync` that consumes it is stream-deferred, so with the host Δ batches ahead in the launch queue every in-flight H2D read whichever index set the host wrote *last*. In steady state that is a pure relabeling; at any host stop — sieve-loop exit, checkpoint quiesce — the Δ still-queued copies all read the **same final set** and the final polynomial batch is sieved Δ times, re-emitting its candidates and (with LP on) its partials verbatim. Measured at RSA-100 solo `--cuda_graph_unroll 0`: **~9 % of all collected relations were pure waste** (duplicates 25,673 → 889 after the fix; raw over-collection 9.22 % → 0.34 %). The fix claims a slot, host-waits on the completion event of the H2D that last consumed it (`cudaEventSynchronize` — a host wait on one already-recorded event; it does **not** serialize the device or drain the stream), then records a fresh event on `stream` after the copy is enqueued. **Side effect, do NOT score as a regression:** a `--cuda_graph_unroll 0` run under a v1.0.6-or-newer binary legitimately reports *lower raw* relation counts than any v1.0.5-or-earlier binary. Root-caused and fixed in v1.0.6.

2. **`runSievingBatch(num_steps, start_batch_index)`** (host) — launches the GPU inner loop, which for each step executes:
   - `resetBatchCountersKernel` — zero bucket counters and per-block relation counts
   - `initPrimeDataBatchKernel` — compute per-prime inverses for this step's a
   - `markInactivePrimesBatchKernel` — mark primes dividing a
   - `globalMetaSieveBatchKernel` — large-prime bucket pre-computation
   - `sieveAndScanBatchKernel` / `...Wide` / `...WideU8Sat` — sieve, scan, trial divide (width selected by the host dispatch, see above)
   - `compactCandidatesBatchKernel` — write `DenseCandidate` records to the active `DoubleBuffer`

The batch kernels accept raw scalar arguments (step index, subCube, sieveIntervalStart) instead of `dynamicSievingParams` to reduce struct-passing overhead.

### generatePolynomialsKernel Detail

One block per batch step (64 threads). Computes a = ∏ p_k via a **variable-depth pairwise tree reduction** over `rdim = max(32, pow2geq(shc_dim))` shared-memory leaves (inactive leaves padded with the multiplicative identity; `__syncthreads()` outside the divergent guard so all threads reach every barrier), then each thread k computes B_k = (a/p_k)·γ_k with γ_k centered (Knuth-Schroeppel). The variable-depth reduction replaced a hardcoded 16-element tree that silently dropped a-primes whenever `shc_dim > 16` (wrong a → zero relations at RSA-140 M=262K; the shc_dim>16 silent-sieve fix).

### sieveAndScanBatchKernel Detail

Annotated with **`__launch_bounds__(1024)`** (same as the legacy variant) to cap register allocation and ensure reliable launches at high thread counts. Structurally identical to `sieveAndScanKernel` with two key differences:

- **B_values in shared memory**: loads the current step's B-components from the pre-uploaded `batch_B_flat` array into shared memory (`s_B_values`) for fast Gray code b-construction. The shared memory layout is: `[s_B_values | offsets1 | offsets2 | primes | blockEntries]` (v1.0.7 under `--sieve_offsets_global`: `[s_B_values | blockEntries]`, the three int arrays living in the block's `dev_sieveOffsets` slice).
- **Per-block relation counts**: writes `candidatesFound` to `dev_blockRelationCounts[blockIdx.x]` at kernel exit, enabling `compactCandidatesBatchKernel` to skip empty blocks.

### compactCandidatesBatchKernel Detail

Grid: one block per sieve block. Each block reads `dev_blockRelationCounts[blockIdx.x]`; blocks with zero candidates exit immediately. For each valid `candidateRelation`:

1. Atomic-reserve a slot in the `DoubleBuffer` output via `atomicAdd(counter, 1)`.
2. In-register insertion sort of `candidateRelation.factors[]` (O(N) for nearly-sorted input).
3. Two-pointer merge of sorted sieve factors with `a`-factors (from shared memory) into `DenseCandidate.factor_indices[48]`.
4. Write `a`, `b`, `true_x`, and merged factor list to the output buffer.

Overflow guard: if `pos >= max_capacity`, the thread returns without writing.

## DeviceSievingController API

### Constructors

| Constructor | Description |
|-------------|-------------|
| `DeviceSievingController(device, stream)` | Batch/async mode on given CUDA stream |
| `DeviceSievingController(device)` | Legacy mode on stream 0 |

### Lifecycle

| Method | Description |
|--------|-------------|
| `initiate(factoringData&)` | Copy factoring parameters, compute `fixedSievingParams`, resolve the accumulator-width dispatch (`use_wide_accumulator_`, `wide_u8sat_selected_`) — requires `f_data.a` finalized (`recalc_a` before `initiate`) |
| `loadData()` | Upload factor base, roots, and a-factors to GPU; **v1.0.7:** also builds and uploads the small-prime mask (`generateMask()`, `device_sieving_controller.cpp:346`) and sets `fs_params.period` / `smallPrimesUsed` — on every path, including wide, whose kernels do not read it |
| `loadStandardConfig()` | Derive the default kernel launch configs (see *Standard Config Geometry*) |
| `loadPartialCustomConfig(totalPolys, totalIntervals, polyBlockSize, blocksPerCycle, metaB, metaT, sasB, sasT)` | Load a tuned 8-parameter (`Params8`) tuple — the autotune / pinned-params / history path. v1.0.7: `midPrimeStartIndex = 96` |
| `loadPartialCustomConfigDynamic(const ParamSet&)` | **v1.0.7**, `--params11` and the parameter search: sparse 11-field loader (`SieveParam` order); a set entry overrides, an unset (0) one keeps the derivation — see the v1.0.7 section |
| `allocateBatchBuffers()` | Allocate batch job arrays (`dev_job_a_array`, `dev_job_B_flat`, `dev_job_factor_indices`, `dev_blockRelationCounts`) |
| `updateState()` | Sync `dynamicSievingParams` after an a-coefficient change (call after `loadData()`) |
| `advance_a(step)` | Advance to the next a coefficient via hypercube walk |

### Execution

| Method | Description |
|--------|-------------|
| `sieveStep()` | Legacy: run one polynomial through the 3-kernel pipeline. **v1.0.7:** `initPrimeData()` (inverse/B-value derivation, a function of `a` alone) runs only when `ds_params.newCube` is set; `updateState()` now sets `newCube = true` whenever `a` changes, so the derivation is paid once per hypercube instead of once per call |
| `sieveFullCube()` | Legacy: iterate all Gray code polynomials for the current a |
| `prepareSievingBatch()` | Batch: advance host state, upload next K polynomial configurations, launch `generatePolynomialsKernel` |
| `prepareSievingBatchFromStaged(idx, a_out, B_out)` | Graph-capturable prep: launch only `generatePolynomialsKernel` from already-staged device indices (no H2D copy, no host state advance) |
| `setJobArrays(a, B, factor_idx)` | Redirect the job-array pointers used by `runSievingBatch` to per-batch staged arrays (for CUDA-graph replay) |
| `runSievingBatch(n, start)` | Batch: execute n polynomial steps on GPU from batch offset `start`. Checks `*external_stop_` (if set) before launching kernels and returns early — host-side only. Passes `use_wide_accumulator_` / `wide_u8sat_selected_` into the kernel dispatch |
| `runParamTest(factoringData&, const ParamSet* seed = nullptr, uint32_t radius = 3)` | **v1.0.7**, `--param_test`: tuning-complex parameter search (narrow only), seeded from `--params11` or from `loadStandardConfig()`; logs the winner as a ready-to-paste `--params11` line and **calls `exit(0)`** (seed rejected ⇒ `exit(1)`). See the v1.0.7 section |
| `runParamTestLegacy(factoringData&)` | `--param_test_legacy` (≤ v1.0.6 `runParamTest`): exhaustive `Params8` sweep; returns `ParamTestResult {best_params, best_timing_us, configs_tested, json_path}` |
| `evaluateConfig(params, num_subcubes, reload_needed)` | Evaluate one `Params8` tuple (loadPartialCustomConfig + validateConfigs + timed run); −1.0f if infeasible |
| `validateResults(factoringData&)` | Pull candidates from GPU, validate via CPU trial division |
| `validateConfigsSilent()` | **v1.0.7**: `validateConfigs()` with every diagnostic suppressed (same verdict) — used per probe by the parameter search |

### Autotune Probe Harness (wide path)

| Method | Description |
|--------|-------------|
| `sieveMini(num_subcubes)` | Lightweight sieve benchmark (µs). On the wide path — where no non-batch kernel exists — dispatches to `sieveMiniBatch()`; the narrow path is the unchanged non-batch probe |
| `sieveMiniBatch(repeats)` | Wide-only: drives the REAL batch pipeline over a warm-up-then-measure wall-clock window and returns a candidate-**survivors/sec rate** (higher = better); `repeats` is ignored — `setProbeWindow()` governs duration |
| `sieveMiniStandardWide()` | Wide-only floor gate: measure `loadStandardConfig`-wide under the same window harness; a tuned winner is applied only if it beats this floor |
| `wideGatherOccupancyBlockDim(candidates, count)` | Wide-only: occupancy-optimal GATHER blockDim via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at the wide GATHER smem footprint (0 = keep existing seed) |
| `setProbeWindow(window_sec, warmup_sec)` | Set the survivors/sec measurement window (defaults 10 s / 2.5 s) |
| `setAutotuneProbePolys(n)` | CLI `--autotune_probe_polys`: force the probe's staged polynomial count (0 = auto-scale by N) |

### Configuration

| Method | Description |
|--------|-------------|
| `setConfig(cfg)` | Overloaded for `initConfig`, `generalSievingConfig`, `globalMetaSieveConfig`, `sieveAndScanConfig`, `processRelationsConfig` |
| `setSievingBatchSize(n)` | Propagates batch_size into all four config structs (call AFTER the config loaders — they reset it to 0) |
| `setThresholdOverride(bound)` | Overrides `fs_params.threshold` to ⌊log₂(bound)⌋; used by orchestrator when LP is active to set threshold = ⌊log₂(lp1_bound)⌋ |
| `setAccumulatorMode(mode)` | `--sieve_accumulator`: 0 auto (predicate), 1 force u8, 2 force u16; consumed in `initiate()` |
| `setWideAccumMode(mode)` | `--wide_accum`: 0 auto (gate), 1 force u8sat (gate-honoured), 2 force u16; consumed in `initiate()` |
| `setMetaCycleCap(cap)` | `--sieve_meta_cycle_cap`: SCATTER active-blocks-per-cycle cap (0 = off); call before either loader |
| `setGatherBlockDim(n)` | `--sieve_gather_block_dim`: GATHER blockDim A/B override, power of two in [32,1024] (0 = off) |
| `setBucketSizeFactor(f)` | `--bucket_size_factor`: `globalBucketSize = f·SB` (0.0 = off, legacy SB/2 exactly) |
| `setSievingBlockSizeOverride(n)` | **v1.0.6**, `--sieve_block_size`: overrides `gs_conf.sievingBlockSize` inside `loadPartialCustomConfig` only. `n = 0` (default) = off, byte-identical on every path; `n > 0` must be a power of two in `[256, M]`. Call before `initiate()`. Ignored on the wide path at the point of use **and** rejected by `validateConfigs()` on wide/legacy |
| `setBigPrimeStartOverride(n)` | **v1.0.6**, `--sieve_big_prime_start`: overrides `gs_conf.bigPrimeStartIndex` (otherwise `SB/32`) inside `loadPartialCustomConfig` only. `n = 0` (default) = off; `n > 0` must satisfy `32 < n ≤ fb_size`. **Not** required to be a power of two. Call before `initiate()` |
| `setOffsetsInGlobal(on)` | **v1.0.7**, `--sieve_offsets_global`: moves the GATHER `offsets1/offsets2/primes` arrays from shared memory into a per-block slice of `dev_sieveOffsets` (global). `false` (default) = shared layout. Narrow only — forced off on wide. Call before either loader |
| `setNarrowOverflowStats(on)` | **v1.0.6**, `--sieve_bucket_overflow_stats`: enables `getBucketOverflowStats()` on the narrow path. `false` (default) = off, byte-identical. Wide behaviour unaffected either way |
| `setMaxTotalSieveBytes(bytes)` | Autotune OOM-guard cap on the total sieve footprint (0 = off, production default); clamp diagnostic via `getLastSeedClamp()` |
| `isWideAccumulator()` / `isWideU8Sat()` | Resolved accumulator-width dispatch state (both false on the narrow path) |

### Integration

| Method | Description |
|--------|-------------|
| `setPostProcessingLinks(DoubleBuffer*)` | Connect sieve output to postprocessor DoubleBuffer |
| `pushCounterToHostAsync(volatile uint32_t*)` | Async copy of candidate count to pinned host memory |
| `clearSievingBuffers()` | Free GPU sieve buffers (buckets, candidates, batch arrays, pinned staging) before matrix stage; factor base and roots are retained |
| `getDevicePointers()` | Expose `devicePointers` for postprocessor access |
| `getCudaStream()` | Return the CUDA stream for stream-ordered operations |
| `getFactoringData()` | Return host-side `factoringData` copy |
| `getFactoringDataRef()` | Mutable reference to the controller's owned `factoringData` (CUDA-graph path must advance the same `f_data` the siever owns) |
| `getBucketOverflowStats(out)` | Telemetry snapshot of `dev_globalBucketCounts`: total/overflowed buckets (bit 31), max fill (masked with `0x00FFFFFF`), per-bucket capacity, overflow fraction. Wide always reports. **Narrow reports only under `--sieve_bucket_overflow_stats`** (v1.0.6, `device_sieving_controller.cpp:203`); default-off because the read costs a `cudaMemcpyAsync` + `cudaStreamSynchronize` **on the siever stream** at the ~5 s stats cadence, which the narrow zero-sync batch pipeline otherwise never pays. The device-side bit-31 flag was always width-agnostic — only this host reader was gated. Measured cost when enabled: ≤ 0.09 % of the sieve wall |

### Cluster / Snapshot Hooks

Added for distributed sieve coordination (`setExternalStop`/`resetAndAdvanceTo` are the two permitted submodule-era changes per the cluster spec). All are host-side only — no device-state modification.

| Method | Description |
|--------|-------------|
| `setExternalStop(std::atomic<bool>*)` | Register an external stop flag. When non-null, `runSievingBatch()` checks `*external_stop_` before each launch and returns early if set. (`device_sieving_controller.h:209`) |
| `saveSnapshot()` | Save an `AFactorsSnapshot` of the current a-factor state. Call immediately after `initiate()` + `init_a_factors()`, before any sieving. |
| `resetAndAdvanceTo(uint64_t global_a_index)` | Reconstruct the exact polynomial state at a global a-index on the Hamiltonian path: restore snapshot, simulate window slides + Gray-code steps, `recalc_a()`, `updateState()`. ~5 ms, independent of jump distance. (`device_sieving_controller.h:312`) |
| `getSnapshot()` | Return the saved `AFactorsSnapshot` (for serialization to workers). |

The private member `external_stop_` (`device_sieving_controller.h:410`, default `nullptr`) holds the registered flag; `snapshot_` holds the saved a-factor state.

## primeAlgorithms Functions

| Function | Description |
|----------|-------------|
| `generateFactorBase(fData)` | Sieve primes up to F, compute rootN via `findRoot` |
| `determineParams(fData)` | Set M, a_target, shc_dim based on N size (a_target = √(2N)/M via the overflow-safe `mpqs::isqrt_2N`) |
| `generateHypercubePath(dir, sign, dim)` | Precompute traversal order for the hypercube of a-coefficients |
| `init_a_factors(fData)` | Initialize a_factors for the first a |
| `advance_a_factors(fData, steps)` | Advance hypercube walk by `steps` |
| `recalc_a(fData)` | Recompute a from current a_factors |
| `prepareNextBatchIndices(fData, batch_size)` | Advance host state for `batch_size` steps; return flattened `uint32_t` vector of size `batch_size * shc_dim` |
| `Tonelli_Shanks(n, p)` | Modular square root mod prime p (32-bit) |
| `findRoot(d, N, p)` | Root of N mod p (Tonelli-Shanks / lifting front-end) |
| `liftRoot(r, N_red, p)` | Hensel lifting for roots mod prime powers |
| `modInv(a, p)` / `modInvSquare(a, p)` | Modular inverses mod prime p |

### 64-bit Number-Theory Primitives

Overflow-safe 64-bit primitives added for the branch-fixed character columns (the branch aux primes
`q_s` are chosen `> lp1_bound ~1e11`, exceeding `uint32_t`). All modular steps route through the
`__int128`-safe `mpqs::math::{mul_mod, pow_mod}` helpers. Consumed by the matrix module's
`selectAuxPrimes`/`branchCharBit` and the postprocessing branch char-bit capture.

| Function | Description |
|----------|-------------|
| `Tonelli_Shanks_u64(n, p)` | Modular square root mod prime p (64-bit), used to fix the per-aux-prime Tonelli root `t_s` |
| `jacobi_u64(a, n)` | 64-bit Jacobi symbol |
| `is_prime_u64(n)` | Deterministic Miller–Rabin primality test over the full `uint64_t` range |

## Dependencies

Imports from `src/common/`: `uint512.cuh`, `math_utils.cuh`, `montgomery.cuh`, `mpqs_soa.h`, `hpc_logger.h`.

Carries a `README.md` (module overview + integrated-build notes). The module is **integrated-build only** — its `CMakeLists.txt` is consumed by the parent cuda-mpqs project (which provides `mpqs_common`, the `cudampqs_build_flags` interface library, and the global include dirs); there is no standalone build path.
