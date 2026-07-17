# Sieve Module (`src/sieve/`)

In-tree flattened folder (demoted from a former Git submodule in Stage A — only `src/linalg` remains a submodule). GPU-accelerated SIQS polynomial sieving with two execution modes: legacy host-driven and batch GPU-only. Mode selection is controlled by `sieve_batch_size`: 0 (default) selects legacy, any positive value selects batch.

Large prime (LP) support is orthogonal to pipeline selection — both legacy and batch modes support LP when the orchestrator sets a threshold override via `setThresholdOverride()`.

The batch sieve-and-scan kernel additionally exists in three **accumulator widths** (legacy uint8, wide uint16, wide saturating-uint8) — the RSA-155 dual-path fork; see *Dual-Path Sieve Accumulator* below. The legacy uint8 kernels are byte-for-byte untouched by the fork (enforced by `tools/sieve/assert_legacy_untouched.sh`).

Namespaces: `mpqs::sieve` (all sieving structures and kernels), `mpqs::postprocessing` (DoubleBuffer).

## Files

| File | Purpose |
|------|---------|
| `kernel.cu` / `kernel.cuh` | All CUDA kernels (legacy + batch variants + the wide/u8sat accumulator forks), device math helpers, polynomial/root helpers, host launch wrappers |
| `sieving_data_structs.h` | All data structures: primes, candidates, contexts, configs, `gpuInfo`, `DoubleBuffer`; `MAX_SHC_DIM` |
| `sieve_memory_model.h` | Single source-of-truth device-memory model: `sieveBucketBudget()` (`kSieveBudgetNum/Den` = 4/5 = 0.80·VRAM), `estimateSieveFootprint()`, `reduceNumPolysToBudget()` — mirrors the ten `cudaMalloc` calls in `kernel.cu` (loadSievingData), computed in 64-bit (fixes the 32-bit product wrap that OOMed M=262K/RSA-140) |
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
| `modSum(a, b, m)` | (a + b) mod m where b is signed (|b| < m) — used for Gray code root updates |
| `log2(a)` | Floor log₂ via `clz32` |
| `align_up_to_hit(x, bound, p)` | Smallest y ≥ bound with y ≡ x (mod p) |
| `atomicByteAdd(array, idx, x)` | Byte-granularity atomic add via 32-bit word atomics (also as `ATOMIC_BYTE_ADD` / `ATOMIC_BYTE_ADD_RETURN` macros, `kernel.cu:42-46`) |
| `ATOMIC_HALF_ADD` / `ATOMIC_HALF_ADD_RETURN` | uint16-granularity atomic add macros (2×uint16 per 32-bit word, `kernel.cu:50-54`) — wide-accumulator counterpart of the byte macros; the byte macros are untouched |
| `atomicByteAddSat(array, idx, x)` | Saturating byte add: read-clamp-CAS on the enclosing 32-bit word, clamps at 255 without carrying into the adjacent byte lane (`kernel.cu:80-93`) — used only by the u8sat wide kernel's contended accumulation paths |
| `excludeNonRelations(...)` | Block-strided scan for threshold-exceeding candidates (`kernel.cu:850`); each candidate thread reserves its output slot via a per-thread `atomicAdd` on a shared counter (no warp-level intrinsics — see below). Backward trial division extracts factor lists. Overflow-safe (see below). Wide fork `excludeNonRelationsWide` (`kernel.cu:921`) is a near-verbatim copy whose only diff is the `uint16_t*` blockEntries width (enforced by `tools/sieve/assert_fork_widthdiff.sh`) |

## 3-Kernel Legacy Pipeline (per polynomial step)

Selected when `sieve_batch_size == 0` (default). The host drives one polynomial at a time through three kernels:

1. **`initPrimeDataKernel`** — Compute a⁻¹ mod p for each factor base prime and write the per-prime B-update values B_k · a⁻¹ mod p into the column-major `dev_primeBValues` array (coalesced across the grid-stride threads at fixed k).
2. **`globalMetaSieveKernel`** — Pre-compute bucket offsets for large primes (index ≥ `bigPrimeStartIndex`). Stores `(offset | log2_p << 24)` entries (with the prime's factor-base index in the high 32 bits of the 64-bit bucket entry) via `atomicAdd` write heads into global bucket arrays, indexed by `globalBucketId`. **Nested-loop control flow** (see below).
3. **`sieveAndScanKernel`** — Forward sieve: adds log_p at arithmetic-progression offsets for small primes (shared memory byte array), applies large-prime buckets from global memory, then scans for threshold-exceeding positions and backward trial-divides to extract `candidateRelation` records. Annotated with **`__launch_bounds__(1024)`** to prevent the CUDA compiler from over-allocating registers, which would reduce occupancy or cause launch failures at 1024 threads per block.

Each kernel has a host wrapper function (same name without `Kernel` suffix) that configures grid/block dimensions and launches on a specified CUDA stream.

Additionally, **`markInactivePrimesKernel`** sets `primeDataSIQS.inactive = 1` for primes that divide a.

### sieveAndScanKernel Detail

The kernel operates in two phases per sieve block:

**Forward sieve phase:**
- Initializes a shared-memory byte array (`blockEntries`) to zero.
- Small primes (index < `midPrimeStartIndex`): cooperative sieve via direct shared-memory adds, one prime at a time with full-block synchronization.
- Mid-range primes (`midPrimeStartIndex` ≤ index < `bigPrimeStartIndex`): each thread handles its own prime via `ATOMIC_BYTE_ADD` (no inter-thread sync needed — disjoint access).
- Large primes (index ≥ `bigPrimeStartIndex`): applied from pre-computed global buckets filled by `globalMetaSieveKernel`.

This kernel family is referred to as the **GATHER** kernel in tuning reports (it gathers/dumps bucket entries and scans); the meta-sieve kernels are the **SCATTER** side (they scatter bucket writes).

**Candidate extraction phase (`excludeNonRelations`):**
- Block-strided over `blockEntries`: each thread evaluates one position, flagging `isCandidate` where the accumulated log-sum exceeds `approxPolyVal - threshold` (`approxPolyVal ≈ log₂|Q(x)|`), and overwrites `blockEntries[index]` with the boolean for the backward scan.
- **Per-thread slot reservation**: each candidate thread reserves its output slot with `atomicAdd(&candidateWriteHead, 1)` on a shared counter. The sieve deliberately uses **no warp-level functions** (`__ballot_sync` / `__shfl_sync`) for this compaction: at MPQS smoothness rates, candidate positions within a sieve block are too sparse for warp-level compaction to beat the simple per-thread atomic — atomic contention is already negligible when relations are rare — so the serial reservation is retained. (An earlier warp-ballot rewrite of `excludeNonRelations` was reverted in review and did not land.)
- Writes `candidateRelation` records (b, poly_id, sieve_offset, num_factors=0) for qualifying positions.
- **Overflow-safe clamping**: a candidate whose reserved slot `>= maxPerBlock` is dropped and its `blockEntries[index]` is reset to 0, preventing the backward scan from reading uninitialized `indexToCandidate`. The returned count is clamped so `candidatesFound` never grows past the per-block buffer limit `maxPerBlock`.

**Backward trial division phase:**
- For each active prime, walks backward through the sieve offsets. At positions flagged as candidates, records the prime's factor base index into `candidateRelation.factors[]` using `ATOMIC_BYTE_ADD_RETURN` for index allocation.
- Large-prime bucket entries are similarly scanned for factor extraction (prime index stored in upper 32 bits of the 64-bit bucket entry).

### globalMetaSieveKernel Nested-Loop Control Flow

The meta-sieve (SCATTER) bucketing logic is structured as explicit nested loops (replacing an earlier state machine): **cycles → polyBlocks → primes → polys → offsets**. See `kernel.cu:399-530` and the batch variant `globalMetaSieveBatchKernel` at `kernel.cu:988-1127` — both share the identical loop body. Note the cycles loop wraps the whole factor-base primes loop, so capping cycles multiplies FB re-reads (which is why `--sieve_meta_cycle_cap` is a locality ablation knob, not a speedup lever).

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

At RSA-150/155 scale a full smooth accumulates `log₂|Q/a| − δ ≈ 258–274` in the forward-sieve accumulator; the legacy `uint8` `blockEntries` caps at 255 and **wraps**, silently rejecting ~98–99.5 % of full smooths (RSA-140's ~250 stays under the cap — unaffected). The fix is a dual-path fork; **the legacy kernels are byte-for-byte untouched** (`tools/sieve/assert_legacy_untouched.sh`).

**Kernels** (all batch-mode; there is no non-batch wide kernel):

| Kernel | Accumulator | Notes |
|--------|-------------|-------|
| `sieveAndScanBatchKernel` (`kernel.cu:1129`) | `uint8`, wrapping | Legacy narrow path; carries all validated ≤RSA-140 records |
| `sieveAndScanBatchKernelWide` (`kernel.cu:1440`) | `uint16` via `ATOMIC_HALF_ADD*` | Near-verbatim width-only fork (+ `excludeNonRelationsWide`); doubles the `blockEntries` shared-memory footprint, halving SB |
| `sieveAndScanBatchKernelWideU8Sat` (`kernel.cu:1754`) | `uint8`, **saturating** at 255 | Option A: copy of the narrow kernel whose five forward-accumulation sites saturate (via `atomicByteAddSat` on contended paths, clamped store on the single-owner mid-prime path); restores SB to narrow's full width |

**Host dispatch** (`DeviceSievingController::initiate()`):
- `use_wide_accumulator_` — computed from an exact 3-point evaluation of `approxPolyVal` at x ∈ {−M, 0, M−1} (mirroring the device arithmetic): `use_wide = (APV_max + WIDE_MARGIN) >= 256` with `WIDE_MARGIN = 4` (`device_sieving_controller.cpp:119-132`). Auto: uint16-family for RSA-150/155, uint8 for ≤RSA-140. CLI override `--sieve_accumulator {auto|u8|u16}` (`setAccumulatorMode`).
- `wide_u8sat_selected_` — within the wide regime, the saturating-uint8 kernel is selected iff the hardened exactness gate `max_target = APV_max − threshold ≤ 252` holds (254 minus a 2-count margin for non-pow2-M sampling slop and per-poly `a.msb()` drift; `device_sieving_controller.cpp:168-190`). Under the gate, saturating-uint8 candidate selection is bit-for-bit identical to uint16 (`stored = min(255, sum)` ⇒ `(stored > target) ⇔ (sum > target)` for every target ≤ 254). CLI override `--wide_accum {auto|u8sat|u16}` (`setWideAccumMode`); a forced u8sat still honours the gate (falls back to uint16, never to the wrap bug).
- `runSievingBatch(..., use_wide, wide_u8sat)` (`kernel.cu:2687-2726`) selects the kernel via a launch ternary; when `use_wide == false` the launch is character-for-character the pre-fork legacy launch.
- `accumElemBytes()` (1 for narrow/u8sat, 2 for uint16) is the single source of truth for every SB / `sharedMemReq` derivation.

## Standard Config Geometry (`loadStandardConfig`)

`loadStandardConfig()` (`device_sieving_controller.cpp:1169`) derives the no-autotune/no-history geometry. Narrow-path values are byte-for-byte the historical ones; every wide deviation is gated on `use_wide_accumulator_`:

- **`sievingBlockSize` (SB):** narrow `pow2leq(3/4·maxShared)`; wide `pow2leq((3/4·maxShared − 3·1024·4)/accumElemBytes())` — the uint16 accumulator budget quarters SB vs narrow (2 bytes/entry + the reserved bigPrime floor), a structural throughput cost the autotune cannot touch; u8sat restores the 1-byte width.
- **`globalBucketSize` = SB/2** (legacy exact) unless the ablation knob `--bucket_size_factor <F>` is set, in which case `computeGlobalBucketSize()` returns `round(F·SB)` in both config loaders (F=0.5 reproduces legacy; F=1.0 doubles the bucket). The resized bucket is charged against the VRAM budget before allocation, so an over-large factor degrades `num_polysPerSieveCall` or is validator-rejected — never an OOM.
- **`bigPrimeStartIndex`:** wide `SB/32` (mirrors the tuned custom-path split); narrow keeps exactly 1024. `midPrimeStartIndex` = 32. (The `bigPrimeStart=SB/32` + `num_polys≤512` pair is the wide-path default-geometry fix that recovered 18× on H100.)
- **`num_polysPerSieveCall`:** seed `min(32768, 2^(shc_dim−1))`, wide-clamped to ≤ 512, then reduced by `reduceNumPolysToBudget()` until the bucket buffer (`num_polys · num_sievingBlocks · globalBucketSize · 8` B, computed in 64-bit — see `sieve_memory_model.h`) fits `0.80·totalGlobalMem`; `num_subCubes` absorbs the change so total polys sieved is unaffected. An optional autotune OOM-guard knob (`setMaxTotalSieveBytes`, default off) further halves the seed until the *entire* footprint (`estimateSieveFootprint().total()`) fits.
- **Meta-sieve (SCATTER) grid:** wide `2·pow2geq(multiProcessorCount)` (SM-aware — the hardcoded 64-block grid left an A100 at 12.5 % occupancy, 1.4–1.9× SCATTER win); narrow keeps exactly 64.
- **Wide-path `num_polys ≥ num_threadBlocks` validation floor — CLAMPED (shipped, `c1e49ac`, not an OOM).** The meta-sieve SCATTER grid `num_threadBlocks = 2·pow2geq(SMs)` is **512 on H100 (132 SMs), 256 on A100 (108 SMs)** and is fixed independent of `M`. Historically the config validator (`device_sieving_controller.cpp:1736,1745,1757`) required `num_polysPerSieveCall ≥ num_threadBlocks` (and `num_polyBlocksPerThreadBlock = num_polys/num_threadBlocks` to be a nonzero power of two), so when a large `M` drove `reduceNumPolysToBudget()` to degrade `num_polys` **below** that grid width the pipeline **aborted with a config-validation error before any sieve memory was allocated** — it was *not* an OOM and *not* a VRAM ceiling (VRAM could be almost entirely free at the abort). This is exactly why the H100 M-ladder's M=8M/bf2.0 leg (job 1986430) aborted: np degraded 512→256 < the H100 grid's 512. **The floor is now clamped away** (`device_sieving_controller.cpp:1376-1377`, wide-gated on `use_wide_accumulator_`): `if (use_wide_accumulator_) gms_conf.num_threadBlocks = std::min(gms_conf.num_threadBlocks, gs_conf.num_polysPerSieveCall);`, applied after the two np-budget reductions and before the `polyBlockSize` loop — every SM-derived block that still owns a poly is kept and the pow2/product/LEQ checks pass (the min of two powers of two is a power of two). The narrow (uint8, ≤RSA-140) path keeps its exact grid (64), byte-for-byte unchanged. **np-retention closed form** (still the relevant sizing guide — a degraded np now means fewer active SCATTER blocks and thus a *slower* run rather than an abort): with `bucket_bytes = 16·num_polys·bf·M` (SB cancels), `num_polys=512` is retained iff `16·512·bf·M ≤ 0.8·VRAM_total`, i.e. **`bf·M ≤ 9.76×10⁶` on H100** and **`bf·M ≤ 4.14×10⁶` on A100**. **Outcome since the clamp shipped:** M=16M (H100, np=256) was measured against M=8M (np=512) at F=300M and F=400M and lost on both rungs (−13.3%/−2.6% fulls/s respectively) — the np-halving penalty outweighs the M-doubling gain; the M-axis is closed at M=8M on H100 (2026-07-12 RSA-155 H100 M=16M analysis). Do not pursue M=32M.
- **GATHER launch:** `ss_conf.num_threadsPerBlock = 256`, `num_threadBlocks = min(256, num_polysPerSieveCall)`, `sharedMemReq = SB·accumElemBytes() + 3·bigPrimeStartIndex·4`.
- **Ablation knobs** (both loaders, all default-off/byte-identical): `--sieve_meta_cycle_cap <N>` caps `num_activeBlocksPerCycle` at `min(pow2_floor(N), derived)` and raises `num_metaSieveCycles` correspondingly (SCATTER write-locality experiment; net-negative as a speedup — see loop-nesting note above); `--sieve_gather_block_dim <N>` overrides the GATHER blockDim (result-invariant: the shared accumulator is sized per sieving-block, not per-thread, and every work loop strides by `blockDim.x`).

## Batch Sieving (GPU-Only Mode)

Selected when `sieve_batch_size > 0`. Eliminates CPU-GPU synchronization in the inner loop by pre-uploading K polynomial configurations and running all steps on-device:

1. **`prepareSievingBatch()`** (host) — calls `prepareNextBatchIndices()` to advance `a_factors` for K steps on the host, then uploads flattened factor indices (via a pinned staging buffer for truly async H2D) to `dev_job_factor_indices` and launches `generatePolynomialsKernel` to compute all a and B values on-device.

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

- **B_values in shared memory**: loads the current step's B-components from the pre-uploaded `batch_B_flat` array into shared memory (`s_B_values`) for fast Gray code b-construction. The shared memory layout is: `[s_B_values | offsets1 | offsets2 | primes | blockEntries]`.
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
| `loadData()` | Upload factor base, roots, and a-factors to GPU |
| `loadStandardConfig()` | Derive the default kernel launch configs (see *Standard Config Geometry*) |
| `loadPartialCustomConfig(totalPolys, totalIntervals, polyBlockSize, blocksPerCycle, metaB, metaT, sasB, sasT)` | Load a tuned 8-parameter (`Params8`) tuple — the autotune / pinned-params / history path |
| `allocateBatchBuffers()` | Allocate batch job arrays (`dev_job_a_array`, `dev_job_B_flat`, `dev_job_factor_indices`, `dev_blockRelationCounts`) |
| `updateState()` | Sync `dynamicSievingParams` after an a-coefficient change (call after `loadData()`) |
| `advance_a(step)` | Advance to the next a coefficient via hypercube walk |

### Execution

| Method | Description |
|--------|-------------|
| `sieveStep()` | Legacy: run one polynomial through the 3-kernel pipeline |
| `sieveFullCube()` | Legacy: iterate all Gray code polynomials for the current a |
| `prepareSievingBatch()` | Batch: advance host state, upload next K polynomial configurations, launch `generatePolynomialsKernel` |
| `prepareSievingBatchFromStaged(idx, a_out, B_out)` | Graph-capturable prep: launch only `generatePolynomialsKernel` from already-staged device indices (no H2D copy, no host state advance) |
| `setJobArrays(a, B, factor_idx)` | Redirect the job-array pointers used by `runSievingBatch` to per-batch staged arrays (for CUDA-graph replay) |
| `runSievingBatch(n, start)` | Batch: execute n polynomial steps on GPU from batch offset `start`. Checks `*external_stop_` (if set) before launching kernels and returns early — host-side only. Passes `use_wide_accumulator_` / `wide_u8sat_selected_` into the kernel dispatch |
| `runParamTest(factoringData&)` | Exhaustive `Params8` sweep; returns `ParamTestResult {best_params, best_timing_us, configs_tested, json_path}` |
| `evaluateConfig(params, num_subcubes, reload_needed)` | Evaluate one `Params8` tuple (loadPartialCustomConfig + validateConfigs + timed run); −1.0f if infeasible |
| `validateResults(factoringData&)` | Pull candidates from GPU, validate via CPU trial division |

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
| `getBucketOverflowStats(out)` | Wide-only telemetry snapshot of `dev_globalBucketCounts`: total/overflowed buckets (bit 31), max fill, per-bucket capacity. Returns false on the narrow path (no extra DtoH sync on the validated production hot path) |

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
