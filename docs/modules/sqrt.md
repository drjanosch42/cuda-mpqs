# Square Root Module (`src/sqrt/`)

## Overview

Final pipeline stage: derives factors from kernel vectors via the congruence of squares X² ≡ Y² (mod N). Namespace: `mpqs::sqrt` (test utilities in `mpqs::test`).

Static library `mpqs_sqrt`. Separable CUDA compilation ON. Links `mpqs_common`, `cudampqs_build_flags`.

## Files

| File | Purpose |
|------|---------|
| `sqrt_step.h` | `SquareRootRefinement` class, `DeviceBuffers` struct (`mpqs::sqrt`) |
| `sqrt_step.cu` | CUDA kernels (M2–M4, M9–M10), CPU oracles, `Perform()` driver. Explicit template instantiation for `FBType = uint32_t`. |
| `fake_relation_generator.h` | `FakeData` struct + `FakeRelationGenerator` class interface (`mpqs::test`) |
| `fake_relation_generator.cpp` | Synthetic problem generation (Blum integer N = p·q, Miller-Rabin primality, CRT square roots) |
| `CMakeLists.txt` | Build config: standalone mode with local common dependency, or project mode linking global `mpqs_common` |

## Algorithm

1. **Unpack** packed `uint64_t` solution bits into a per-relation binary mask.
2. **ComputeX** (CPU oracle): X = Π sqrt\_Q[i] (mod N) for selected relations, in Montgomery arithmetic. `sqrt_Q[i] = |a_i·x_i + b_i|`.
3. **ComputeY** (CPU oracle): Y = (-1)^s · 2^(e₂/2) · Π p_j^(e_j/2) · LP (mod N), where 2·e_j = sum of factor exponents across selected relations; s = total sign count / 2. All exponents guaranteed even by the GF(2) null space.
4. **GPU validation** (inline in `Perform()`): constructs a single-solution `BWKernelSolutionView`, runs `ComputeXBatchedGPU` (M2) / `ComputeYBatchedGPU` (M3) and compares against CPU oracles; also verifies X²_gpu ≡ Y²_gpu (mod N). This is a debug/verification step ensuring GPU and CPU paths agree.
5. **SanityCheck**: verify X² ≡ Y² (mod N) using `modpow`. On failure, calls `RelationBatch::validate_host_batch()` for diagnostics.
6. **Factor extraction**: `BatchedGCDKernel` computes gcd(|X−Y|, N) and gcd(X+Y, N) per solution. `RefineFactorsKernel` (M10) then extracts pairwise-coprime factors via iterative GCD refinement.

## CUDA Kernels

### Tuning Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `SQRT_CHUNK_SIZE` | 256 | Relations per chunk for parallel reduction |
| `SQRT_TRANSFORM_BLOCK` | 256 | Threads per block for transform kernels |
| `SQRT_REDUCE_BLOCK` | 256 | Threads per block for final reduction (must be power of 2) |
| `SQRT_EXP_THREADS` | 256 | Threads per block for BatchedExponentiateY |
| `SQRT_CSR_BLOCK` | 256 | Threads per block for CSR scatter |
| `SQRT_WINDOW_BITS` | 4 | Window size for windowed exponentiation |
| `SQRT_WINDOW_THRESHOLD` | 16 | Use windowed exp for exponents > this |
| `REFINE_BLOCK_SIZE` | 128 | Threads per block for RefineFactorsKernel |
| `MAX_POOL` | 64 | Max distinct factors in refinement pool |
| `MAX_REFINE_ROUNDS` | 8 | Max pairwise GCD refinement iterations |

### ComputeX Pipeline (3-Phase, M2/M9)

#### Phase 1: `TransformSqrtQ`
Embarrassingly parallel pre-transform: one thread per sqrt_Q value converts standard form → Montgomery form.

Grid: `<<<ceil(K / 256), 256>>>`. Shared memory: none.

#### Phase 2: `ComputeX_ChunkReduce`
Block-level partial products via tree reduction. 2D grid: `(n_chunks, n_solutions)`. Each block handles `SQRT_CHUNK_SIZE` (256) consecutive relations for one solution. Threads load pre-transformed Montgomery values, mask with solution bit (Montgomery identity for unselected), and tree-reduce in shared memory.

Grid: `<<<dim3(n_chunks, n_solutions), 256>>>`. Shared memory: `256 × sizeof(uint512)` = 16 KB.

#### Phase 3: `FinalReduce`
Reduces n_chunks partial products into one final value per solution. One block per solution. Threads partition chunks via grid-stride loop, then tree-reduce. Reusable: `do_reduce=true` outputs standard form; `do_reduce=false` stays in Montgomery form (used for LP products).

Grid: `<<<n_solutions, 256>>>`. Shared memory: `256 × sizeof(uint512)`.

### ComputeY Pipeline (3-Phase, M3/M9)

#### Phase 1a: LP Product (Parallel)
`TransformLargePrimes` → `ComputeX_ChunkReduce` (over d_lp_mont) → `FinalReduce` (do_reduce=false). Converts `__int128` large primes to Montgomery uint512 (identity for lp ≤ 1), then reduces per-solution LP products via the same chunk/reduce pipeline as ComputeX.

**`TransformLargePrimes`**: One thread per relation. Grid: `<<<ceil(K / 256), 256>>>`.

#### Phase 1b: CSR Scatter + Sign/Exp2 (`AccumExponents_Parallel`, M9)
2D grid: `<<<dim3(n_chunks, n_solutions), 256>>>`. One thread per relation: checks solution bit, scatters CSR exponents via `atomicAdd` to `d_exp_matrix`, accumulates sign count and val_2_exp via `atomicAdd`. No shared memory. All target arrays must be zero-initialized before launch.

#### Phase 1b (Legacy): `BatchedAccumulateExponents`
Behind `#ifdef SQRT_LEGACY_KERNELS`. Single-block: one thread per solution iterates all relations sequentially. Superseded by AccumExponents_Parallel.

#### Phase 2: `HalveExponents`
2D grid: x-dimension over FB primes (blocks of 256), y-dimension over solutions. Each thread halves its exponent in `d_exp_matrix`; thread 0 per solution checks and halves `d_exp2_accum` and checks `d_sign_counts` parity. Odd exponents set `d_valid[sol] = 0`.

#### Phase 3: `BatchedExponentiateY`
One block per solution (`blockIdx.x = sol_idx`), 256 threads (power-of-2 required). Threads partition FB primes in a grid-stride loop. Three exponentiation paths:
- **exp == 1**: direct multiply (fast path, ~30-50% of non-zero primes)
- **exp ≤ 16**: standard binary exponentiation via `mont.pow()`
- **exp > 16**: 4-bit windowed exponentiation with precomputed table[15]

Products are tree-reduced in shared memory. Thread 0 applies sign (via `negate_mod_inplace`), factor-2 (via `mont.pow(2, exp2_half)`), and LP Montgomery product, then reduces to standard form.

Grid: `<<<n_solutions, 256>>>`. Shared memory: `256 × sizeof(uint512)`.

**`TransformFactorBase`** (M9): Pre-transforms factor base primes from uint32 → Montgomery form. Run once per allocation (guarded by `fb_transformed_` flag). Eliminates redundant `mont.transform()` in BatchedExponentiateY.

### Factor Extraction

#### `BatchedGCDKernel` (M4)
One thread per solution. Computes `diff = |X[j]−Y[j]|` (explicit comparison to avoid unsigned wrap), `f1 = gcd(diff, N)`. If trivial, computes `sum = X[j]+Y[j]`, `f2 = gcd(sum, N)`. Sets `d_factor_status[j]` to 0 (trivial), 1 (via |X−Y|), or 2 (via X+Y). Uses `mpqs::math::gcd()` (`__host__ __device__`).

**Per-solution nontrivial-GCD rate (`--sqrt_diagnostic`).** After `BatchedGCD` downloads
`d_factor_status`, the host counts how many of the `n` Block-Wiedemann solutions yielded a
nontrivial factor and logs the rate `k/n` (with the distinct factor pairs found) at `LOG_DEBUG_1`
(`sqrt_step.cu:1474`). This is the diagnostic for the high-LP collapse: an unobstructed run sits
near the ~50% theoretical cap, whereas an obstructed (2-cycle-dominated) run collapses to 0%.
Capture it with `--sqrt_diagnostic --log_file <path>` (it is suppressed at the default `--verbose`/
info level).

#### `RefineFactorsKernel` (M10)
Extracts finest factorization from BatchedGCDKernel output via coprime refinement.

Three-phase single-block kernel (`<<<1, 128>>>`):
- **Phase 0**: Compact non-trivial factors, insertion sort, deduplicate → shared `pool[m]`.
- **Phase 1**: Iterative pairwise GCD refinement. Each thread takes one pair from the triangular matrix (`decode_triangular` helper). Splits recorded in scratch memory; thread 0 rebuilds pool each round. Converges when pool is pairwise coprime (O(k²) rounds).
- **Phase 2**: Write refined factors to global memory.

For semiprimes (N = pq): m = 1, zero refinement rounds → ~2μs.

Grid: `<<<1, 128>>>`. Shared memory: `MAX_POOL × 4 × sizeof(uint512) + 12` bytes ≈ 16.4 KB.

### Legacy Kernels (behind `#ifdef SQRT_LEGACY_KERNELS`)

#### `BatchedComputeX` (M2 original)
Single-block: one thread per solution (≤ 64). Thread 0 loads each sqrt_Q[i] and transforms to Montgomery in shared memory; all threads broadcast-multiply. Two `__syncthreads()` per iteration. Superseded by the 3-phase pipeline.

## `DeviceBuffers` Struct

Pre-allocated device buffer pool, defined in `sqrt_step.h`. Allocated lazily on first GPU call with 1.5× safety margin, reused across invocations, freed in destructor. All `cudaMalloc` calls are error-checked.

| Field | Size | Description |
|-------|------|-------------|
| `d_sqrt_Q` | [max_K] | sqrt_Q values in standard form |
| `d_sqrt_Q_mont` | [max_K] | Montgomery-transformed sqrt_Q |
| `d_signs` | [max_K] | Relation signs |
| `d_val_2_exps` | [max_K] | Power-of-2 exponents |
| `d_large_primes` | [max_K] | Large prime cofactors (__int128) |
| `d_factor_offsets` | [max_K+1] | CSR row pointers |
| `d_factor_indices` | [max_nnz] | CSR column indices |
| `d_factor_counts` | [max_nnz] | CSR values (exponent counts) |
| `d_lp_mont` | [max_K] | Montgomery-transformed large primes |
| `d_factor_base` | [max_fb] | Factor base primes (uint32) |
| `d_factor_base_mont` | [max_fb] | Montgomery-transformed factor base |
| `d_X` | [max_n] | X output values |
| `d_Y` | [max_n] | Y output values |
| `d_exp_matrix` | [max_n × max_fb] | Exponent accumulation matrix |
| `d_sign_counts` | [max_n] | Sign accumulators |
| `d_exp2_accum` | [max_n] | Power-of-2 accumulators |
| `d_Y_lp_mont` | [max_n] | LP Montgomery accumulators |
| `d_valid` | [max_n] | Solution validity flags |
| `d_partials_x` | [max_chunks × max_n] | Chunk partial products for ComputeX |
| `d_partials_lp` | [max_chunks × max_n] | Chunk partial products for LP reduction |
| `d_factors` | [max_n] | GCD factor candidates |
| `d_factor_status` | [max_n] | GCD status (0/1/2) |
| `d_refined_factors` | [64] | Refined pairwise-coprime factors (M10) |
| `d_refined_count` | scalar | Number of refined factors |

## `SquareRootRefinement` API

**Constructor:** `SquareRootRefinement(const mpqs::uint512& N)` — stores modulus as `N_`, creates CUDA stream (`stream_`).
**Destructor:** destroys `stream_`, calls `freeDeviceBuffers()`.

### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `Perform` | `pair<uint512,uint512>(const vector<uint64_t>& solution_bits, const HostRelationBatch& batch, const vector<FBType>& fb)` | CPU fallback path: unpacks solution bits, runs CPU X/Y oracles, GPU validation (M2/M3), sanity check (X²≡Y² mod N), CPU-side GCD. Returns `{factor, cofactor}` or `{0,0}`/`{1,N}`. Templated on `FBType` (instantiated for `uint32_t`). |
| `ComputeXBatchedGPU` | `void(const BWKernelSolutionView& solutions, const HostRelationBatch& batch)` | 3-phase GPU pipeline: TransformSqrtQ → ComputeX_ChunkReduce → FinalReduce. Results on device, retrieve via `getDeviceX()`. |
| `ComputeYBatchedGPU` | `void(const BWKernelSolutionView& solutions, const HostRelationBatch& batch, const vector<FBType>& fb)` | GPU pipeline: LP transform/reduce, AccumExponents_Parallel, HalveExponents, BatchedExponentiateY. Results on device, retrieve via `getDeviceY()`. Templated on `FBType`. |
| `BatchedGCD` | `pair<uint512,uint512>(const uint512* d_X, const uint512* d_Y, int n)` | Launches BatchedGCDKernel + RefineFactorsKernel (M10). Host-side statistics, coprimality verification, product-divides-N check. Returns `{factor, cofactor}`. |
| `getDeviceX` | `const uint512*() const` | Device pointer to X results (valid after ComputeXBatchedGPU). |
| `getDeviceY` | `const uint512*() const` | Device pointer to Y results (valid after ComputeYBatchedGPU). |
| `allocateDeviceBuffers` | `void(uint32_t max_K, uint32_t max_nnz, uint32_t max_fb, uint32_t max_n)` | Pre-allocate device buffer pool. Called automatically by GPU methods; may be called explicitly with complete sizing to avoid mid-pipeline reallocation. |

### Private Methods

| Method | Description |
|--------|-------------|
| `ComputeX(mask, batch, mont)` | CPU oracle: sequential Montgomery product over selected `sqrt_Q` values. |
| `ComputeY(mask, batch, fb, mont)` | CPU oracle: accumulate exponents then compute Montgomery product with sign/exp2/LP handling. |
| `SanityCheck(X, Y)` | Returns `modpow(X,2,N) == modpow(Y,2,N)`. |
| `freeDeviceBuffers()` | Frees all device memory, resets `bufs_` and `fb_transformed_`. |

### Private Members

| Member | Type | Description |
|--------|------|-------------|
| `N_` | `uint512` | Modulus being factored |
| `bufs_` | `DeviceBuffers` | Pre-allocated device buffer pool |
| `stream_` | `cudaStream_t` | Async CUDA stream for H2D transfers and kernel launches |
| `fb_transformed_` | `bool` | Guard: true after `TransformFactorBase` ran for current allocation |

## Stream Management

The class owns a dedicated CUDA stream (`stream_`) created in the constructor and destroyed in the destructor. All `cudaMemcpyAsync` transfers and kernel launches use this stream. Explicit `cudaStreamSynchronize(stream_)` calls separate pipeline phases that have data dependencies (e.g., after Phase 1b before Phase 2 HalveExponents).

## NVCC Workaround

The `__noinline__` attribute may be required on certain device functions to work around NVCC code generation issues with deeply nested template instantiations in Montgomery arithmetic.

## `BWKernelSolutionView` (from `src/linalg/include/bw_solution_view.h`)

Lightweight device-side view passed to kernels:

| Field | Type | Description |
|-------|------|-------------|
| `d_data` | `uint64_t*` | Device pointer to packed solution bit-matrix (row-major: solution × words_per_vec) |
| `num_solutions` | `uint32_t` | Number of solution vectors (≤ 64) |
| `words_per_vec` | `uint32_t` | `ceil(num_relations / 64)` |
| `num_relations` | `uint32_t` | Number of relations |

## `FakeRelationGenerator` (testing)

Generates a synthetic factorization problem `N = p·q` (Blum integers, p ≡ q ≡ 3 mod 4) with mathematically consistent relations for unit testing the sqrt step in isolation.

### `FakeData` struct

| Member | Type | Description |
|--------|------|-------------|
| `N` | `uint512` | Composite N = p·q |
| `p, q` | `uint512` | Known prime factors |
| `factor_base` | `vector<uint32_t>` | Odd primes used in synthetic relations |
| `relations` | `vector<Relation>` | AoS relations (every relation is a perfect square; all exponents even) |
| `solution_bits` | `vector<uint64_t>` | Packed bitmask selecting all generated relations as the solution subset |

### `FakeRelationGenerator` class

| Member | Description |
|--------|-------------|
| `FakeRelationGenerator(int bit_size, int fb_size)` | Constructs factor base cache (odd primes by trial division); seeds RNG. |
| `FakeData generate(int num_relations)` | Generates p, q (Miller-Rabin, 5 rounds), N = p·q; constructs each relation by choosing random even exponents, computing RHS, then solving `(ax+b)² ≡ RHS (mod N)` via `sqrt_mod_3mod4` + CRT. |

## Integration with Linear Algebra Output

The sqrt stage consumes the output of the Block Wiedemann linear algebra solver via two interfaces:

1. **`Perform()` path (single solution):** accepts a packed `std::vector<uint64_t>` bitmask (`solution_bits`) where bit _i_ indicates that relation _i_ is selected by the kernel vector. Internally unpacks to a per-relation `uint8_t` mask via `unpack_bits_local()`.

2. **Batched GPU path (`ComputeXBatchedGPU` / `ComputeYBatchedGPU` / `BatchedGCD`):** accepts a `BWKernelSolutionView` — a device-resident packed bit-matrix where row _j_ is solution _j_ and column _i_ is relation _i_. This enables processing all kernel vectors (≤ 64) simultaneously in a single kernel launch.

The orchestrator calls the batched GPU path for all solutions simultaneously, falling back to the `Perform()` CPU loop if no nontrivial factors are found.

## Dependencies

Imports: `uint512.cuh`, `montgomery.cuh`, `math_utils.cuh`, `mpqs_soa.h`, `bw_solution_view.h`, `hpc_logger.h`, `mpqs_structures.h`.
Links: `mpqs_common`, `cudampqs_build_flags`.

## Build

`libmpqs_sqrt.a` (static, RDC enabled). Supports standalone build (`cmake -B build` inside `src/sqrt/`) producing `sqrt_benchmark` test binary. In project mode, links global `mpqs_common`; in standalone mode, builds a local `mpqs_common_local` from `../common/`.
