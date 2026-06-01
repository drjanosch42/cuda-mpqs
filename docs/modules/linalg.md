# Linear Algebra Module (`src/linalg/`)

Git submodule `block-wiedemann`. CUDA-accelerated Block Wiedemann algorithm for finding kernel vectors of large sparse binary matrices over GF(2). All arithmetic: addition = XOR, multiplication = AND.

## Directory Structure

```
src/linalg/
├── include/
│   ├── bw_solver.h                        # Public API: BlockWiedemannSolver, BWSolverConfig, stage configs
│   ├── bw_solution_view.h                 # BWKernelSolutionView (device-side solution bit-matrix)
│   ├── validation_io.h                    # Validation I/O helpers
│   ├── hpc_logger.h                       # Logger interface (shared with parent)
│   └── lingen/types.h                     # PackedBitMatView, PackedBitPolyView, PolyMatrixView
├── src/lingen/
│   ├── bw_solver.cu                       # Top-level solver orchestration
│   ├── io/                                # Checkpoint I/O (BWIOSystem, hash.h)
│   ├── operations/
│   │   ├── bw_operator.cu/.h              # Abstract BwOperator + BwOperatorA/AT/ColAtA/RowAAt
│   │   ├── matmul_gf2.cu/.h               # Dense GF(2) matrix multiplication
│   │   ├── poly_arithmetic.cu/.h          # Polynomial multiplication (Karatsuba)
│   │   ├── poly_mat_vec.cu/.h             # Polynomial-matrix-vector products
│   │   ├── poly_arith_engine.cu/.h        # Polynomial arithmetic engine
│   │   └── karatsuba.cuh                  # Karatsuba helper kernels
│   ├── stage1/
│   │   ├── krylov_generator.cu/.h         # Krylov sequence generation
│   │   └── krylov_kernels.cu/.h           # GPU kernels for batch projection
│   ├── stage2/
│   │   ├── basecase_solver.cu/.h          # GPU/CPU hybrid Block Berlekamp-Massey
│   │   ├── basecase_solver_reference.cu/.h # CPU reference implementation
│   │   ├── basecase_ops.cu/.h             # Basecase helper operations
│   │   └── device_poly.h                  # Device-side polynomial types
│   └── stage3/
│       └── solution_reconstructor.cu/.h   # Kernel vector reconstruction
├── cuda_spmm/                             # SpMM sub-library (see below)
│   ├── include/
│   │   ├── bw_spmm_interface.h            # SpMM public API (BlockWiedemannSpMM, SpMMAutoTuneConfig)
│   │   ├── common.h                       # HostMatrix, HostMatrixView, SpMMConfig
│   │   ├── gpu_autotuner.h                # GPUAutoTuner class
│   │   ├── device_csr.h                   # DeviceCSR, DeviceCSRSlice, upload/transpose/permute utilities
│   │   ├── device_format_convert.h        # GPU-side CSR→format converters (TiledCOO, Delta16, M4RM, PFor, Golomb)
│   │   ├── optimizer_common.h             # KernelID, KernelConfig, ExecutionPlan, SegmentRecipe, LaunchConfig
│   │   ├── spmm_optimized.h               # OptimizedSpMM (multi-stream kernel dispatcher)
│   │   ├── format_arena.h                 # FormatArena: GPU memory lifecycle (temporary→persistent promotion)
│   │   ├── autotuner.h                    # Legacy CPU autotuner
│   │   └── kernels.h / vec_type.h         # Kernel launchers, VecType<BITS> template
│   └── src/                               # Kernels, gpu_autotuner.cu, device_format_convert.cu
├── tests/                                 # test_s_disk_io.cu
├── benchmarks/                            # Standalone benchmark executables
├── python/                                # Verification scripts (verify_bw_pipeline.py)
├── logs/                                  # Run logs and golden reference outputs
└── CMakeLists.txt                         # Dual-mode: standalone & submodule
```

## Three-Stage Algorithm

### Stage 1: Krylov Sequence Generation

Computes the sequence S_k = X^T · B^(k+1) · Z for k = 0..L-1, where B is A or A^T and X, Z are random dense binary blocks of dimension N × m and N × n respectively. Double-buffered GPU pipeline: V ← B·V via SpMM, then S_k ← X^T · V via templated batch projection kernel. The operator B is selected based on `stage1_prefer_faster_op`: when true, Stage 1 picks whichever of A or A^T was faster at autotune, exploiting the identity S_k = (Z^T · B_alt^k · X)^T. When `stage1_keep_S_on_device` is true (default), the full S sequence is kept in device memory for zero-copy handoff to Stage 2, eliminating a GPU→CPU→GPU round-trip. Supports checkpoint save/load for X, Z, S.

### Stage 2: Block Berlekamp-Massey (Lingen)

Finds generator polynomial Pi(x) in F_2[x]^((m+n) × (m+n)) such that S(x) · Pi(x) ≡ 0 mod x^L. Coppersmith/Thomé basecase algorithm on (m+n) × (m+n) dense matrices over GF(2). Default execution is CPU reference (`gpu_mode = false`); hybrid GPU mode (`gpu_mode = true`) offloads discrepancy computation and elimination to the GPU while pivot decisions remain CPU-resident. Receives S directly from Stage 1 device buffer (zero-copy) or loads from disk (`stage2_load_S_from_disk`). Verification options: GPU and CPU annihilation checks, step-by-step oracle verification, post-run legacy comparison. The `k_eliminationN` fallback kernel (512 threads vs. 1024 for the primary `k_elimination`) is selected automatically via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on GPUs with insufficient register pressure.

### Stage 3: Solution Reconstruction

For each candidate column u(x) from Pi, computes w = sum_k B^(deg-k) · Z · u_k via Horner evaluation, then strips valuation (w ← B·w) until B·w = 0. Batch mode processes multiple candidates in parallel. Circular history buffer (default depth 64) enables backtracking to recover solutions that annihilate during stripping. If `stage3_perform_unpermutation` is true (default), applies inverse row permutation P^T to results before output. Outputs linearly independent kernel vectors — both host-side (`get_solutions()`) and device-side packed bit-matrix (`get_device_solutions()` → `BWKernelSolutionView`).

## BlockWiedemannSolver API

| Method | Description |
|--------|-------------|
| `BlockWiedemannSolver(config, matrix)` | Constructor; takes `BWSolverConfig` and `HostMatrix` |
| `Solve()` | Runs AutoTune, Stage 1, Stage 2, Stage 3 |
| `AutoTune()` | Benchmark SpMM kernels, select optimal config |
| `RunStage1()` | Krylov sequence generation |
| `RunStage2()` | Block Berlekamp-Massey solve |
| `RunStage3()` | Kernel vector reconstruction |
| `get_solutions()` | Returns `const vector<vector<uint64_t>>&` — bit-packed kernel vectors (host) |
| `get_device_solutions()` | Returns `BWKernelSolutionView` — device-side packed solution bit-matrix |
| `pad_to_square(matrix)` | Free function; pads rectangular matrix to square (required by Block Wiedemann) |

## BWSolverConfig

Single source of truth. Field naming convention: no prefix = invariant, `autotune_` / `stage1_` / `stage2_` / `stage3_` = stage-specific.

| Group | Key Parameters | Defaults |
|-------|---------------|----------|
| Topology | `device_id`, `nrows`, `m_block`, `n_block`, `solve_transposed`, `seed` | 0, 0, 64, 64, false, 12345 |
| Global switches | `checkpoint_prefix`, `enable_all_hashing`, `enable_all_validation`, `enable_all_oracle_verification` | "", all false |
| AutoTune (general) | `autotune_tune_spmm`, `autotune_skip_if_present`, `autotune_verify_spmm`, `autotune_db_path` | true, true, true, "bw_tuning.csv" |
| AutoTune (GPU pipeline) | `autotune_gpu_only`, `autotune_m4rm_rows`, `autotune_skip_m4rm_benchmark`, `autotune_block_growth` (FIXED/EXPONENTIAL) | true, 8, false, EXPONENTIAL |
| AutoTune (block sizes) | `autotune_initial_block_size`, `autotune_max_block_size` | 8, 65536 |
| AutoTune (per-format) | `autotune_enable_tiled_coo`, `autotune_enable_delta16`, `autotune_enable_pfor_be`, `autotune_enable_golomb` | true, true, true, true (but Golomb disabled in GPUAutoTuner::Config) |
| Stage 1 | `stage1_skip`, `stage1_seq_len`, `stage1_gpu_batch_size`, `stage1_prefer_faster_op`, `stage1_keep_S_on_device`, `stage1_save_S_to_disk` | false, 0 (auto), 64, true, true, false |
| Stage 1 I/O | checkpoints (load/save), hash validation, file suffixes | load=true, save=false, hashing=false |
| Stage 2 | `stage2_skip`, `stage2_seq_len`, `stage2_delta`, `stage2_gpu_mode`, `stage2_load_S_from_disk` | false, 0, 0 (auto: L/2), false, false |
| Stage 2 verification | annihilation checks (GPU/legacy), oracle verification, post-run legacy check | all false |
| Stage 3 | `stage3_skip`, `stage3_batch_mode`, `stage3_max_solutions`, `stage3_perform_unpermutation`, `stage3_history_depth`, `stage3_check_interval`, `stage3_stripping_limit` | false, true, -1 (all), true, 64, 16, 0 (heuristic) |

Parent project overrides: `m_block = n_block = 256` (submodule default: 64).

## SpMM Sub-Library (`cuda_spmm/`)

Ten kernel implementations for different row-density profiles:

| Kernel | Format / Profile |
|--------|-----------------|
| M4RM | Method of Four Russians (very dense, 8-row batches) |
| Dense_Bitslice | Bitsliced dense columns |
| Sparse_WarpCSR | Warp-level CSR (medium-dense) |
| Sparse_TiledCOO | Tiled coordinate format (main workhorse) |
| Sparse_TiledCOO_Unrolled | Unrolled variant of above |
| Sparse_PForDelta | PFor-compressed column indices |
| Sparse_PForDelta_BitExact | Bit-exact PFor variant |
| Sparse_Delta16 | Delta-16 encoding |
| Sparse_Golomb | Golomb-Rice compression (disabled by default — CPU fallback data mismatch) |
| Sparse_Ellpack | ELLPACK format |

**GPU-only autotuning pipeline** (`autotune_gpu_only = true`, default): `HostMatrix` is uploaded once to `DeviceCSR` via `upload_host_matrix_to_device_csr()`, then rows are reordered by descending density via `device_csr_permute_by_density()`. The transpose is built on-device via `device_csr_transpose()`. Format conversion and benchmarking happen entirely on-device via `GPUAutoTuner::tune()`. Memory is managed by `FormatArena` — candidate formats are allocated as temporaries, winners are promoted to persistent, and losers are freed after the DP solver picks optimal segments. Winning format's device data is adopted directly into `OptimizedSpMM::compile()`, eliminating CPU re-preprocessing. Falls back to legacy `tune_global()` on exception. Block partitioning uses exponential growth (8→16→32→…→65536 rows per block, ~19 blocks) rather than the legacy fine-grained fixed strategy.

**GPU format conversion** (`device_format_convert.h`): On-device CSR-to-format converters for the GPU autotuning pipeline. Each takes a `DeviceCSRSlice` and writes into a `DeviceMatrix` or `M4RMContext`:

| Function | Target Format |
|----------|---------------|
| `gpu_convert_csr_to_tiledcoo` | TiledCOO (coords + block pointers) |
| `gpu_convert_csr_to_delta16` | Delta-16 stream + offsets |
| `gpu_convert_csr_to_m4rm` | M4RM pattern stream |
| `gpu_convert_csr_to_pfor_be` | PFor-BitExact (GPU with optional CPU fallback) |
| `gpu_convert_csr_to_golomb` | Golomb-Rice (GPU with optional CPU fallback) |

**GPUAutoTuner::Config** controls per-format enables at the autotuner level. Notable differences from `BWSolverConfig`: adds `enable_tiledcoo_unrolled`, `enable_warp_csr`, `allow_cpu_fallback` (for PFor/Golomb), and `n_spmm_calls` (expected invocations for conversion cost amortization; 0 = auto-estimate).

**Permutation system**: Two permutation maps (`P_row_`, `P_col_`) sort dense rows first and group frequently-referenced columns. Input/output vectors must be transformed between logical and permuted domains — both CPU (`preprocess_domain_*`) and GPU (`permute_vec_*_gpu`) methods are provided.

**Execution strategy** (`LaunchConfig`): Three binary flags controlling multi-stream kernel dispatch — `spawn_dense_group` (parallelize dense+M4RM), `spawn_sparse_group` (parallelize sparse kernels), `spawn_dense_sparse_merge` (run dense and sparse phases concurrently). All 8 combinations are benchmarked by `tune_execution_strategy`.

Vector width templated via `VecType<BITS>` (32, 64, 128, 256, 512 bits). SpMM tuning cache stored as CSV; may become stale if matrix properties change.

## Key Data Types

| Type | Location | Description |
|------|----------|-------------|
| `HostMatrix` | `cuda_spmm/include/common.h` | Sparse binary matrix: `n_rows`, `n_cols`, `vector<vector<idx_t>> rows` |
| `HostMatrixView` | `cuda_spmm/include/common.h` | Non-owning view into a row range of `HostMatrix` |
| `DeviceCSR` | `cuda_spmm/include/device_csr.h` | Canonical on-device CSR representation (row_ptr + col_ind + density_perm arrays) |
| `DeviceCSRSlice` | `cuda_spmm/include/device_csr.h` | Non-owning view into a contiguous row range of `DeviceCSR`; invalidated on parent free/realloc |
| `ExecutionPlan` | `cuda_spmm/include/optimizer_common.h` | Ordered sequence of `SegmentRecipe`s with tuned kernel configs + density permutation |
| `SegmentRecipe` | `cuda_spmm/include/optimizer_common.h` | Row range + best `KernelConfig` + live device data from GPU autotuning |
| `LaunchConfig` | `cuda_spmm/include/optimizer_common.h` | Execution strategy flags: 3 binary options for multi-stream kernel dispatch |
| `PackedBitMatView` | `include/lingen/types.h` | Row-major dense binary matrix view, 64-bit word aligned, stride = ceil(cols/64) |
| `PackedBitPolyView` | `include/lingen/types.h` | Polynomial of dense binary matrix coefficients, contiguous storage |
| `PolyMatrixView<N>` | `include/lingen/types.h` | Fixed-size N×N binary matrix polynomial (templated, for Stage 2 basecase) |
| `BWKernelSolutionView` | `include/bw_solution_view.h` | Kernel-passable device pointer + metadata for solution bit-matrix; layout: `d_data[j * words_per_vec + i/64]` |
| `SpMMConfig` | `cuda_spmm/include/common.h` | Kernel enables, vector width, tiling parameters |
| `SpMMAutoTuneConfig` | `cuda_spmm/include/bw_spmm_interface.h` | Autotuning config passed to `setup_and_benchmark()` |
| `SpMMPearformanceReport` | `cuda_spmm/include/bw_spmm_interface.h` | Timing and throughput results from autotune (A and A^T) |

## Key Classes

| Class | Role |
|-------|------|
| `BlockWiedemannSolver` | Public API; orchestrates AutoTune + 3 stages |
| `BwOperator` | Abstract linear operator B: V → V (pure virtual `mul()`) |
| `BwOperatorA` / `BwOperatorAT` | Concrete B = A and B = A^T operators |
| `BwOperatorColAtA` / `BwOperatorRowAAt` | Operators B = P^T A^T A P and B = A A^T for rectangular matrices |
| `KrylovSequenceGenerator` | Stage 1: double-buffered Krylov pipeline |
| `BasecaseSolver` | Stage 2: CPU/GPU hybrid Block BM (default: CPU reference) |
| `BasecaseSolverReference` | Stage 2: independent CPU reference for post-run verification |
| `SolutionReconstructor` | Stage 3: batch kernel vector reconstruction with backtracking |
| `BlockWiedemannSpMM` | SpMM lifecycle: permute, preprocess, tune, execute for A and A^T |
| `GPUAutoTuner` | GPU-only autotuner: exponential block partitioning, on-device format conversion + benchmarking |
| `FormatArena` | GPU memory manager for format conversion: temporary allocations freed after benchmarking, winners promoted to persistent |
| `OptimizedSpMM` | Executes an `ExecutionPlan` via multi-stream kernel dispatch |

## Build

Dual-mode `CMakeLists.txt`:

| Mode | Libraries Built | Extras |
|------|----------------|--------|
| Standalone | `lingen_ops`, `spmm_core`, `hpc_logger` | Benchmarks (`bw_lingen_smoke`, `bw_lingen_bench`, `bench_matmulgf2`, `bench_karatsuba`, `bench_lingen_apply_pi`, `test_s_disk_io`) |
| Submodule | `lingen_ops`, `spmm_core` | Inherits `cudampqs_build_flags` from parent; no benchmarks |

Both use CUDA separable compilation. Submodule mode excludes `poly_arithmetic.cu` and `poly_arith_engine.cu` (unused in parent pipeline).

Build option: `-DENABLE_LINGEN_DEVICE_SYNC=ON` — defines `BASECASE_SOLVER_CUDA_STREAM_SYNC`, enabling periodic `cudaStreamSynchronize` (every 10 steps) in Stage 2 basecase loop. Required on Turing where too many sequential kernel launches cause failures.

## Integration

```cpp
lingen::BWSolverConfig cfg;
cfg.m_block = cfg.n_block = 256;
cfg.solve_transposed = true;
auto A_sq = lingen::pad_to_square(host_matrix);  // from matrix stage
lingen::BlockWiedemannSolver solver(cfg, A_sq);
solver.Solve();
auto& kernel_vectors = solver.get_solutions();    // host bit-packed vectors → sqrt stage
auto dev_view = solver.get_device_solutions();    // BWKernelSolutionView → direct GPU access
```

Receives `HostMatrix` (converted from CSR by matrix stage). Outputs kernel vectors as both `vector<vector<uint64_t>>` (host) and `BWKernelSolutionView` (device packed bit-matrix) consumed by the sqrt stage.

## Known Limitations

- Turing GPUs may need `ENABLE_LINGEN_DEVICE_SYNC=ON` for Stage 2 stability.
- Non-square matrices are padded automatically by `pad_to_square`.
- SpMM tuning cache (CSV) may be stale if matrix dimensions or density change significantly.
- Golomb-Rice kernel is disabled by default in `GPUAutoTuner::Config` (`enable_golomb = false`) due to CPU fallback producing incorrect format data.
- 
