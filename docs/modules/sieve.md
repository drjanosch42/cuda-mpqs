# Sieve Module (`src/sieve/`)

In-tree flattened folder (demoted from a former Git submodule in Stage A — only `src/linalg` remains a submodule). GPU-accelerated SIQS polynomial sieving with two execution modes: legacy host-driven and batch GPU-only. Mode selection is controlled by `sieve_batch_size`: 0 (default) selects legacy, any positive value selects batch.

Large prime (LP) support is orthogonal to pipeline selection — both legacy and batch modes support LP when the orchestrator sets a threshold override via `setThresholdOverride()`.

Namespaces: `mpqs::sieve` (all sieving structures and kernels), `mpqs::postprocessing` (DoubleBuffer).

## Files

| File | Purpose |
|------|---------|
| `kernel.cu` / `kernel.cuh` | All CUDA kernels (legacy + batch variants), device math helpers, polynomial/root helpers |
| `sieving_data_structs.h` | All data structures: primes, candidates, contexts, configs, `gpuInfo`, `DoubleBuffer` |
| `device_sieving_controller.h` / `.cpp` | Main API class: initialization, execution, batch orchestration, state management, snapshot / cluster hooks |
| `prime_algorithms.cu` / `.h` | Factor base generation, Tonelli-Shanks, Hensel lifting, hypercube walk, batch index preparation |
| `graycode.cuh` | Gray code enumeration: `gray()`, `advanceGray()`, `grayBitToFlip()` (all `__host__ __device__`) |
| `common.h` | `factoringData` struct (`mpqs::sieve` sieving state) and `AFactorsSnapshot` |
| `json_helper.h` | Minimal `JSONString` / `JSON_IO` builder used only by the optional debug-snapshot path |
| `debug_dump.cu` / `.cuh` / `.h` | GPU debug snapshot tooling |
| `README.md` | Module overview and integrated-build notes (no standalone build path) |
| `CMakeLists.txt` | Integrated build configuration (declares the `mpqs_sieve` static library) |

## Key Data Structures

### primeDataSIQS -- per-prime GPU data

| Field | Type | Description |
|-------|------|-------------|
| `p` | `uint32_t` | The prime |
| `r` | `uint32_t` | Root r with r² = N (mod p) |
| `mod_inverse_a` | `uint32_t` | a⁻¹ mod p |
| `B_values[16]` | `uint32_t[16]` | B_k · a⁻¹ mod p for Gray code updates (k = 0..shc_dim-1) |
| `inv_aN` | `uint32_t` | a⁻¹ · r mod p |
| `inactive` | `uint32_t` | 0 = prime divides a (skip); nonzero = participates in sieving |

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

Flat struct bundling all device pointers including `dev_a_factors`, `dev_B_values`, `dev_factorBase`, `dev_primeData`, `dev_rootN`, bucket arrays, candidate arrays, and batch job arrays (`dev_job_a_array`, `dev_job_B_flat`, `dev_job_factor_indices`), plus postprocessing integration fields (`dev_pp_accumulation_buffer`, `dev_pp_counter`, `pp_max_capacity`). Marked for future replacement by `DeviceConstants` / `DeviceHypercubeContext`.

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

Three config structs control kernel launches (all carry `batch_size`):

- **`initConfig`**: `num_threadsPerBlock`, `num_threadBlocks`, `batch_size`
- **`generalSievingConfig`**: `sievingBlockSize`, `log2_sievingBlockSize`, `num_polysPerSieveCall`, `num_subCubes`, `num_sievingBlocksPerSieveCall`, `batch_size`, `globalBucketSize`, `bigPrimeStartIndex`, `midPrimeStartIndex`, `maxRelationsPerBlock`
- **`globalMetaSieveConfig`**: `num_threadBlocks` (default 64), `maxActiveBucketsTotal`, `polyBlockSize`, `log2_polyBlockSize`, `num_activeBlocksPerCycle`, `num_metaSieveCycles`, `sharedMemReq`, `batch_size`
- **`sieveAndScanConfig`**: `num_threadsPerBlock`, `num_threadBlocks`, `sharedMemReq`, `batch_size`
- **`processRelationsConfig`**: `num_threadsPerBlock`, `num_threadBlocks`
- **`fixedSievingParams`**: constant per factorization — `fb_size`, `shc_dim`, `M`, `approxPolyRoot`, `threshold`
- **`dynamicSievingParams`**: changes per step — `a`, `log2_a`, `startIndex`, `subCube`, `newCube`
- **`polyData`**: `approxPolyRoot`, `log2_a`, `threshold` (passed to candidate scan)

`loadStandardConfig()` sets reasonable defaults; individual configs can be overridden via `setConfig()`.

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

| Function | Purpose |
|----------|---------|
| `rootsFromPolyId(id, shc_dim, primeData, r1, r2)` | Reconstruct sieve roots for a specific poly ID from hypercube |
| `bFromPolyId(id, shc_dim, B_values, result)` | Reconstruct coefficient b from poly ID |
| `advanceRoots(id1, id2, primeData, r1, r2)` | Update roots when transitioning between Gray code states |
| `advance_b(id1, id2, B_values, b)` | Update b when transitioning between Gray code states |

### Device Math Helpers (`kernel.cuh`)

| Function | Description |
|----------|-------------|
| `modAdd(a, b, m)` | (a + b) mod m, safe for a,b < m |
| `modSub(a, b, m)` | (a − b) mod m, safe for unsigned a,b < m |
| `modSub_shifted(a, b, m)` | Returns result in [1, m] for computing positive sieve offsets |
| `modSum(a, b, m)` | (a + b) mod m where b is signed (|b| < m) — used for Gray code root updates |
| `log2(a)` | Floor log₂ via `clz32` |
| `align_up_to_hit(x, bound, p)` | Smallest y ≥ bound with y ≡ x (mod p) |
| `atomicByteAdd(array, idx, x)` | Byte-granularity atomic add via 32-bit word atomics |
| `excludeNonRelations(...)` | Block-strided scan for threshold-exceeding candidates; each candidate thread reserves its output slot via a per-thread `atomicAdd` on a shared counter (no warp-level intrinsics — see below). Backward trial division extracts factor lists. Overflow-safe (see below) |

## 3-Kernel Legacy Pipeline (per polynomial step)

Selected when `sieve_batch_size == 0` (default). The host drives one polynomial at a time through three kernels:

1. **`initPrimeDataKernel`** — Compute a⁻¹ mod p and B_values[k] = B_k · a⁻¹ mod p for each factor base prime. Sets `inactive` flag for primes dividing a.
2. **`globalMetaSieveKernel`** — Pre-compute bucket offsets for large primes (index ≥ `bigPrimeStartIndex`). Stores `(offset | log2_p << 24)` tuples (with the prime's factor-base index in the high 32 bits of the 64-bit bucket entry) via `atomicAdd` write heads into global bucket arrays, indexed by `globalBucketId`. **Nested-loop control flow** (see below): the former state machine was replaced by three explicit nested loops — primes → polys → offsets — driven by a `polyIndex` cursor.
3. **`sieveAndScanKernel`** — Forward sieve: adds log_p at arithmetic-progression offsets for small primes (shared memory byte array), applies large-prime buckets from global memory, then scans for threshold-exceeding positions and backward trial-divides to extract `candidateRelation` records. Annotated with **`__launch_bounds__(1024)`** to prevent the CUDA compiler from over-allocating registers, which would reduce occupancy or cause launch failures at 1024 threads per block.

Each kernel has a host wrapper function (same name without `Kernel` suffix) that configures grid/block dimensions and launches on a specified CUDA stream.

Additionally, **`markInactivePrimesKernel`** marks primes that divide a as inactive in `primeDataSIQS.inactive`.

### sieveAndScanKernel Detail

The kernel operates in two phases per sieve block:

**Forward sieve phase:**
- Initializes a shared-memory byte array (`blockEntries`) to zero.
- Small primes (index < `midPrimeStartIndex`): cooperative sieve via direct shared-memory adds, one prime at a time with full-block synchronization.
- Mid-range primes (`midPrimeStartIndex` ≤ index < `bigPrimeStartIndex`): each thread handles its own prime via `ATOMIC_BYTE_ADD` (no inter-thread sync needed — disjoint access).
- Large primes (index ≥ `bigPrimeStartIndex`): applied from pre-computed global buckets filled by `globalMetaSieveKernel`.

**Candidate extraction phase (`excludeNonRelations`):**
- Block-strided over `blockEntries`: each thread evaluates one position, flagging `isCandidate` where the accumulated log-sum exceeds `approxPolyVal - threshold` (`approxPolyVal ≈ log₂|Q(x)|`), and overwrites `blockEntries[index]` with the boolean for the backward scan.
- **Per-thread slot reservation**: each candidate thread reserves its output slot with `atomicAdd(&candidateWriteHead, 1)` on a shared counter. The sieve deliberately uses **no warp-level functions** (`__ballot_sync` / `__shfl_sync`) for this compaction: at MPQS smoothness rates, candidate positions within a sieve block are too sparse for warp-level compaction to beat the simple per-thread atomic — atomic contention is already negligible when relations are rare — so the serial reservation is retained.
- Writes `candidateRelation` records (b, poly_id, sieve_offset, num_factors=0) for qualifying positions.
- **Overflow-safe clamping**: a candidate whose reserved slot `>= maxPerBlock` is dropped and its `blockEntries[index]` is reset to 0, preventing the backward scan from reading uninitialized `indexToCandidate`. The returned count is clamped so `candidatesFound` never grows past the per-block buffer limit `maxPerBlock`.

**Backward trial division phase:**
- For each active prime, walks backward through the sieve offsets. At positions flagged as candidates, records the prime's factor base index into `candidateRelation.factors[]` using `ATOMIC_BYTE_ADD_RETURN` for index allocation.
- Large-prime bucket entries are similarly scanned for factor extraction (prime index stored in upper 32 bits of the 64-bit bucket entry).

### globalMetaSieveKernel Nested-Loop Control Flow

The meta-sieve bucketing logic is structured as three explicit nested loops (replacing an earlier state machine; see `kernel.cu:398-461`, and the batch variant `globalMetaSieveBatchKernel` at `kernel.cu:960-1023`). Both variants share the identical loop body:

1. **Primes loop** — block-strided over `currentPrimeIndex` from `bigPrimeStartIndex` to `fb_size`. Each thread seeds `polyIndex = (threadIdx.x/32) % polyBlockSize`, so the warp lane group selects its starting Gray-code polynomial. `polyId = gray(polyIndex)`, `fullPolyId = fullPolyIdPrefix | polyId`, and the per-prime roots are reconstructed via `rootsFromPolyId`. `maxOffsetCount` is computed from the *first* prime in the block to keep the inner loop length warp-uniform (constant trip count across lanes).
2. **Polys loop** — `polyBlockSize` iterations. Each iteration emits this polynomial's bucket entries, then advances to the next Gray-code state: `polyIndex = modAdd(polyIndex, 1, polyBlockSize)`, recompute `polyId`/`fullPolyId`, and `advanceRoots(prevFullPolyId, fullPolyId, ...)` performs the O(1) root update.
3. **Offsets loop** — for each of the two roots (`offset1`, `offset2`), strides by `+p` across `maxOffsetCount` hits, mapping each hit to a `sievingBlockHit` and atomically reserving a slot in the corresponding active bucket (drops the entry if the bucket is full).

The `polyIndex` cursor is **warp-uniform** by construction (seeded from `threadIdx.x/32`), so all lanes in a warp share the same polynomial and Gray-code transition, avoiding divergent root reconstruction.

### Extended Shared Memory (`sharedMemPerBlockOptin`)

Per-block shared-memory budget is taken from `prop.sharedMemPerBlockOptin - 1024` (`kernel.cu:635`) rather than the classic 48 KB static limit, so the sieve can opt into the architecture's full dynamic shared-memory window (`gpuInfo.maxSharedMemPerBlock`). Kernels requesting more than 48 KB obtain it via the dynamic shared-memory launch argument (`sharedMemReq` in `globalMetaSieveConfig` / `sieveAndScanConfig`); the host caps `sharedMemReq` against this opt-in ceiling.

## Batch Sieving (GPU-Only Mode)

Selected when `sieve_batch_size > 0`. Eliminates CPU-GPU synchronization in the inner loop by pre-uploading K polynomial configurations and running all steps on-device:

1. **`prepareSievingBatch()`** (host) — calls `prepareNextBatchIndices()` to advance `a_factors` for K steps on the host, then uploads flattened factor indices to `dev_job_factor_indices` and launches `generatePolynomialsKernel` to compute all a and B values on-device.

2. **`runSievingBatch(num_steps, start_batch_index)`** (host) — launches the GPU inner loop, which for each step executes:
   - `resetBatchCountersKernel` — zero bucket counters and per-block relation counts
   - `initPrimeDataBatchKernel` — compute per-prime inverses for this step's a
   - `markInactivePrimesBatchKernel` — mark primes dividing a
   - `globalMetaSieveBatchKernel` — large-prime bucket pre-computation
   - `sieveAndScanBatchKernel` — sieve, scan, trial divide (see below)
   - `compactCandidatesBatchKernel` — write `DenseCandidate` records to the active `DoubleBuffer`

The batch kernels accept raw scalar arguments (step index, subCube, sieveIntervalStart) instead of `dynamicSievingParams` to reduce struct-passing overhead.

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
| `initiate(factoringData&)` | Copy factoring parameters, compute `fixedSievingParams`, call `updateState()` |
| `loadData()` | Upload factor base, roots, and a-factors to GPU |
| `loadStandardConfig()` | Set reasonable default kernel launch configs |
| `allocateBatchBuffers()` | Allocate batch job arrays (`dev_job_a_array`, `dev_job_B_flat`, `dev_job_factor_indices`, `dev_blockRelationCounts`) |
| `updateState()` | Sync `dynamicSievingParams` after an a-coefficient change |
| `advance_a(step)` | Advance to the next a coefficient via hypercube walk |

### Execution

| Method | Description |
|--------|-------------|
| `sieveStep()` | Legacy: run one polynomial through the 3-kernel pipeline |
| `sieveFullCube()` | Legacy: iterate all Gray code polynomials for the current a |
| `prepareSievingBatch()` | Batch: advance host state, upload next K polynomial configurations, launch `generatePolynomialsKernel` |
| `prepareSievingBatchFromStaged(idx, a_out, B_out)` | Graph-capturable prep: launch only `generatePolynomialsKernel` from already-staged device indices (no H2D copy, no host state advance) |
| `setJobArrays(a, B, factor_idx)` | Redirect the job-array pointers used by `runSievingBatch` to per-batch staged arrays (for CUDA-graph replay) |
| `runSievingBatch(n, start)` | Batch: execute n polynomial steps on GPU from batch offset `start`. Checks `*external_stop_` (if set) before launching kernels and returns early — host-side only |
| `runParamTest(factoringData&)` | Parameter sweep for tuning kernel launch configurations |
| `validateResults(factoringData&)` | Pull candidates from GPU, validate via CPU trial division |

### Configuration

| Method | Description |
|--------|-------------|
| `setConfig(cfg)` | Overloaded for `initConfig`, `generalSievingConfig`, `globalMetaSieveConfig`, `sieveAndScanConfig`, `processRelationsConfig` |
| `setSievingBatchSize(n)` | Propagates batch_size into all four config structs |
| `setThresholdOverride(bound)` | Overrides `fs_params.threshold` to ⌊log₂(bound)⌋; used by orchestrator when LP is active to set threshold = ⌊log₂(lp1_bound)⌋ |

### Integration

| Method | Description |
|--------|-------------|
| `setPostProcessingLinks(DoubleBuffer*)` | Connect sieve output to postprocessor DoubleBuffer |
| `pushCounterToHostAsync(volatile uint32_t*)` | Async copy of candidate count to pinned host memory |
| `clearSievingBuffers()` | Free GPU sieve buffers (buckets, candidates, batch arrays) before matrix stage; factor base and roots are retained |
| `getDevicePointers()` | Expose `devicePointers` for postprocessor access |
| `getCudaStream()` | Return the CUDA stream for stream-ordered operations |
| `getFactoringData()` | Return host-side `factoringData` copy |
| `getFactoringDataRef()` | Mutable reference to the controller's owned `factoringData` (CUDA-graph path must advance the same `f_data` the siever owns) |

### Cluster / Snapshot Hooks

Added for distributed sieve coordination (`setExternalStop`/`resetAndAdvanceTo` are the two permitted submodule-era changes per the cluster spec). All are host-side only — no device-state modification.

| Method | Description |
|--------|-------------|
| `setExternalStop(std::atomic<bool>*)` | Register an external stop flag. When non-null, `runSievingBatch()` checks `*external_stop_` before each launch and returns early if set. (`device_sieving_controller.h:133`) |
| `saveSnapshot()` | Save an `AFactorsSnapshot` of the current a-factor state. Call immediately after `initiate()` + `init_a_factors()`, before any sieving. |
| `resetAndAdvanceTo(uint64_t global_a_index)` | Reconstruct the exact polynomial state at a global a-index on the Hamiltonian path: restore snapshot, simulate window slides + Gray-code steps, `recalc_a()`, `updateState()`. ~5 ms, independent of jump distance. (`device_sieving_controller.h:147`) |
| `getSnapshot()` | Return the saved `AFactorsSnapshot` (for serialization to workers). |

The private member `external_stop_` (`device_sieving_controller.h:244`, default `nullptr`) holds the registered flag; `snapshot_` holds the saved a-factor state.

## primeAlgorithms Functions

| Function | Description |
|----------|-------------|
| `generateFactorBase(fData)` | Sieve primes up to F, compute rootN via `findRoot` |
| `determineParams(fData)` | Set M, a_target, shc_dim based on N size |
| `generateHypercubePath(dir, sign, dim)` | Precompute traversal order for the hypercube of a-coefficients |
| `init_a_factors(fData)` | Initialize a_factors for the first a |
| `advance_a_factors(fData, steps)` | Advance hypercube walk by `steps` |
| `recalc_a(fData)` | Recompute a from current a_factors |
| `prepareNextBatchIndices(fData, batch_size)` | Advance host state for `batch_size` steps; return flattened `uint32_t` vector of size `batch_size * shc_dim` |
| `Tonelli_Shanks(n, p)` | Modular square root mod prime p |
| `liftRoot(r, N_red, p)` | Hensel lifting for roots mod prime powers |
| `modInv(a, p)` | Modular inverse mod prime p |

## Dependencies

Imports from `src/common/`: `uint512.cuh`, `math_utils.cuh`, `montgomery.cuh`, `mpqs_soa.h`, `hpc_logger.h`.

Carries a `README.md` (module overview + integrated-build notes). The module is **integrated-build only** — its `CMakeLists.txt` is consumed by the parent cuda-mpqs project (which provides `mpqs_common`, the `cudampqs_build_flags` interface library, and the global include dirs); there is no standalone build path.
