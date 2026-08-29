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
├── cmake/                                 # version.cmake (LINALGBW_VERSION, single source of truth) + bw_version.h.in
├── src/lingen/
│   ├── bw_solver.cu                       # Top-level solver orchestration
│   ├── bw_version.cpp                     # Version string accessors (configured from cmake/version.cmake)
│   ├── io/                                # bw_io.h (BWIOSystem checkpoint/solution I/O), hash.h
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
├── python/                                # verify_bw_pipeline.py, block_wiedemann_lingen_v5.py (CPU reference)
├── logs/                                  # Run logs and golden reference outputs
└── CMakeLists.txt                         # Dual-mode: standalone & submodule
```

## Three-Stage Algorithm

### Stage 1: Krylov Sequence Generation

Computes the sequence S_k = X^T · B^(k+1) · Z for k = 0..L-1, where B is A or A^T and X, Z are random dense binary blocks of dimension N × m and N × n respectively. Double-buffered GPU pipeline: V ← B·V via SpMM, then S_k ← X^T · V via templated batch projection kernel. The operator B is selected based on `stage1_prefer_faster_op`: when true, Stage 1 picks whichever of A or A^T was faster at autotune, exploiting the identity S_k = (Z^T · B_alt^k · X)^T. When `stage1_keep_S_on_device` is true (default), the full S sequence is kept in device memory for zero-copy handoff to Stage 2, eliminating a GPU→CPU→GPU round-trip. Supports checkpoint save/load for X, Z, S.

### Stage 2: Block Berlekamp-Massey (Lingen)

Finds generator polynomial Pi(x) in F_2[x]^((m+n) × (m+n)) such that S(x) · Pi(x) ≡ 0 mod x^L. Coppersmith/Thomé basecase algorithm on (m+n) × (m+n) dense matrices over GF(2). The submodule default is the CPU reference (`stage2_gpu_mode = false`); hybrid GPU mode (`stage2_gpu_mode = true`, which the parent orchestrator always sets) offloads discrepancy computation and elimination to the GPU while pivot decisions remain CPU-resident. Receives S directly from Stage 1 device buffer (zero-copy) or loads from disk (`stage2_load_S_from_disk`). Verification options: GPU and CPU annihilation checks, step-by-step oracle verification, post-run legacy comparison. The `k_eliminationN` fallback kernel (512 threads vs. 1024 for the primary `k_elimination`) is selected automatically via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on GPUs with insufficient register pressure.

### Stage 3: Solution Reconstruction

For each candidate column u(x) from Pi, computes w = sum_k B^(deg-k) · Z · u_k via Horner evaluation, then strips valuation (w ← B·w) until B·w = 0. Batch mode processes multiple candidates in parallel. Circular history buffer (default depth 64) enables backtracking to recover solutions that annihilate during stripping. If `stage3_perform_unpermutation` is true (default), applies inverse row permutation P^T to results before output. Outputs linearly independent kernel vectors — both host-side (`get_solutions()`) and device-side packed bit-matrix (`get_device_solutions()` → `BWKernelSolutionView`).

⚠ **Stage 3 looks silent for hours at production scale, and that is a logging artefact.** The Horner
loop's per-degree progress line (`[SolutionReconstructor] Horner Loop Degree = k (…%)`,
`src/lingen/stage3/solution_reconstructor.cu:441`) is emitted at **`LOG_DEBUG_2`**, which `--debug`
(= `LOG_DEBUG_1`) does **not** reach; only the per-batch header (`:421`, `LOG_DEBUG_1`) and the
per-solution hits (`:674`) show up at `--debug`. Run the LA stage at **`--log_level 2`** to watch
reconstruction advance. The multi-hour "silent init" observed on RSA-155 was this loop, not lingen
Stage-2 initialisation.

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
| AutoTune (general) | `autotune_tune_spmm`, `autotune_tune_poly`, `autotune_skip_if_present`, `autotune_verify_spmm`, `autotune_db_path` | true, false, true, true, "bw_tuning.csv" |
| AutoTune (GPU pipeline) | `autotune_gpu_only`, `autotune_m4rm_rows`, `autotune_skip_m4rm_benchmark`, `autotune_block_growth` (FIXED/EXPONENTIAL) | true, 8, false, EXPONENTIAL |
| AutoTune (block sizes) | `autotune_initial_block_size`, `autotune_max_block_size` | 8, 65536 |
| AutoTune (per-format) | `autotune_enable_tiled_coo`, `autotune_enable_delta16`, `autotune_enable_pfor_be`, `autotune_enable_golomb` | true, true, true, true (but Golomb disabled in GPUAutoTuner::Config) |
| Stage 1 | `stage1_skip`, `stage1_seq_len`, `stage1_gpu_batch_size`, `stage1_prefer_faster_op`, `stage1_keep_S_on_device`, `stage1_save_S_to_disk` | false, 0 (auto), 64, true, true, false |
| Stage 1 I/O | checkpoints (load/save), hash validation, file suffixes | load=true, save=false, hashing=false |
| Stage 2 | `stage2_skip`, `stage2_seq_len`, `stage2_delta`, `stage2_gpu_mode`, `stage2_load_S_from_disk` | false, 0, 0 (auto: L/2), false, false |
| Stage 2 verification | annihilation checks (GPU/legacy), oracle verification, post-run legacy check | all false |
| Stage 3 | `stage3_skip`, `stage3_batch_mode`, `stage3_max_solutions`, `stage3_perform_unpermutation`, `stage3_history_depth`, `stage3_check_interval`, `stage3_stripping_limit` | false, true, -1 (all), true, 64, 16, 0 (heuristic) |

**Parent project overrides** (`MPQSOrchestrator::LinearAlgebraStage()`, `src/orchestrator/orchestrator.cpp:6883-6941`): `m_block = n_block = 256` from `--bw_m`/`--bw_n` (submodule default: 64), adaptively downscaled to 64 (matrix dim < 4000) or 128 (dim < 16000) when neither flag is pinned (`block_size_pinned` records CLI pinning); `stage2_gpu_mode = true`; `solve_transposed = true`; `stage1_gpu_batch_size = 8`; `autotune_tune_spmm = false` when matrix dim < 100000 (autotune overhead dominates); `checkpoint_prefix = <work_dir>/bw`; `stage3_save_solutions = true` under `--dump_kernel_vectors`.

**Note on the Stage 1 I/O row above:** the submodule's own default is `stage1_load_checkpoints = true` (`stage2_load_checkpoints`/`stage3_load_checkpoints` default `false`), but as of the `--bw_checkpoint_dir`/`--bw_resume` wiring (`orchestrator.cpp:6894-6907`) the parent now forces **all three** (`stage1/2/3_load_checkpoints`) to `false` unless `--bw_resume` is explicitly passed — this closes a latent hazard where a default work-dir checkpoint prefix plus the submodule's own `load=true` default could silently consume a stale/partial checkpoint. So in the shipped pipeline, loading never happens implicitly; see below.

### BW Stage-Boundary Checkpointing (`--bw_checkpoint_dir`, `--bw_resume`, `--bw_max_solutions`)

Three orchestrator-level flags (parsed `tests/cuda-mpqs.cpp:926-945`, wired `src/orchestrator/orchestrator.cpp:6894-6907`; config fields `include/orchestrator.h:294-296`), independent of the submodule's own always-on Stage-1 S/Z/X checkpoint mechanism described above:

| Flag | Config field | Default | Description |
|---|---|---|---|
| `--bw_max_solutions <N>` | `bw_max_solutions` (int32) | `-1` (ALL) | Stop BW Stage-3 reconstruction after `N` solutions; wires the submodule's pre-existing `stage3_max_solutions`. A solution-**count** stop, not a batch-count cap — reconstruction batches (`ceil((m+n)/n)`) run regardless of `N`; this only limits how many candidate columns are converted to solutions. |
| `--bw_checkpoint_dir <path>` | `bw_checkpoint_dir` (string) | `""` (off) | Save Krylov S-sequence / lingen Π-polynomial / final solutions at stage boundaries under `<path>/bw*`. Sets `checkpoint_prefix` and enables the save flags for all three stages. |
| `--bw_resume` | `bw_resume` (bool) | `false` | Load previously-completed-stage artifacts from `--bw_checkpoint_dir` and skip them. Without this flag, `--bw_checkpoint_dir` alone only **saves** — it never loads (see the note above). |

**Scope — stage boundaries only:** resumes from the last **completed** stage, not mid-stage — a crash partway through a multi-hour Stage-3 reconstruction still restarts Stage 3 from scratch; it only saves re-running Stage 1 and/or Stage 2 if those had already finished. Integrity is checked via an FNV-1a `_ckpt_tag.bin` hash on load (mismatch → recompute, never load stale data).

**Status: NOT yet validated end-to-end** (as of 2026-07-15; still open at 2026-08-25) — needs a kill/resume smoke test (checkpoint, kill mid-run, resume, confirm stage-skip + bit-identical factors) before production reliance; that validation is a pending follow-up. Specifically, **`--bw_resume`'s load path has never been exercised**: every production LA run to date (including both RSA-150 LA jobs, 2592783 and 2618471) was a fresh run with `resume=none`, so only the *save* side has ever executed in anger.

⚠ **`--bw_max_solutions` is a solution-COUNT stop, not a batch cap — and that gap is real, not theoretical.** Because the stop counts solutions rather than bounding batches, a config that under-delivers in batch 1 will still *enter* a later batch and can burn hours there for zero yield. Measured at RSA-150: at `BW_BLOCK=128` the solver found only **3** solutions in batch 1 and then spent **3 h 25 m** on a completely zero-yield batch 2. At `BW_BLOCK=256` the same matrix produced **56** solutions in batch 1, the cap (`--bw_max_solutions 24`) bound for the first time in production, and batch 2 was skipped entirely. See the block-width section below.

### Block Width: `BW_BLOCK=256` is the production default, bracketed from both sides (2026-08-25)

`--bw_m` / `--bw_n` both default to **256** (`include/orchestrator.h:292-293`; the submodule's own
`BWSolverConfig` default is 64, overridden by the orchestrator). At RSA-150 scale this is now
measured from **both** directions, and 256 is the only value that works.

**Below — `BW_BLOCK=128` is a 2.687× total LA regression.** Measured on the **same matrix, same
binary, same relation file** (RSA-150, job 2618471 at n=256 vs 2592783 at n=128; only uncontrolled
variable = a different H100 node, far too small to explain the effect):

| Stage | bw128 | bw256 | Ratio 128/256 |
|---|---|---|---|
| Stage 1 Krylov | 9,240.63 s | 5,013.48 s | **1.843×** |
| Stage 2 lingen | 43,016.44 s | 15,259.05 s | **2.819×** |
| Stage 3 reconstruction | 20,164.96 s | 5,684.18 s | **3.548×** |
| **BW total** | 74,107.01 s | 27,580.40 s | **2.687×** |

The launcher's old rationale — *"reconstruction cost ∝ n, so a smaller block halves it"* — is
**REFUTED**. The sequence length is `L = 2N/n`, so the step count scales ∝ 1/n and exactly cancels
the per-step Horner halving; reconstruction is therefore ~invariant in n by that argument, while
Stages 1–2 blow up outright. Every degree-like quantity (`L`, `deg Π`, Stage-3 `MaxDeg`) halves to
within 0.5 % when n doubles. Reconstruction did not merely fail to improve — it got **3.548× worse**.

**Above — `BW_BLOCK=512` OOMs in Stage 3.** Job 2607723 died 27.7 s into Stage 3 on a single
**63.71 GiB** `cudaMalloc` at `src/lingen/stage3/solution_reconstructor.cu:264` — verified in source
as `cudaMalloc(&d_history_, total_bytes * history_depth_)`, i.e. the depth-64 backtracking history
buffer. n=512 does not fit 94 GB under any accounting. lingen additionally hits a **~16×**
generic-matmul cliff at `dim = m+n = 1024` (the specialized basecase kernels stop at dim 512), which
also **pre-emptively refutes** the `m=512 / n=256` mitigation — dim 768 is on the same generic path.

**A second, independent reason for 256: solution supply.** bw128 yielded only **3** solutions in
batch 1 (2 nontrivial); bw256 on the identical matrix yielded **56** — 18.7× the supply — capped to
24 by `--bw_max_solutions`, of which **11/24 = 45.8 %** were nontrivial (identical to RSA-155's
rate). At the measured per-solution rate, bw128's 3-solution batch was a ~1-in-6 chance of finding
no factor at all after 20.6 GPU-h; at n=256 that risk falls to ≈4×10⁻⁷.

### ⚠ Dead Asymmetric Krylov Dispatch Branches — STILL OPEN (found 2026-08-23)

`dispatch_batch_proj()` in `src/lingen/stage1/krylov_generator.cu` selects the templated batch
projection kernel by `(m_block_, n_block_)`. Three branches **repeat the `n_block_ == 256` condition
where 512 was intended**, so they can never be reached — the preceding `== 256` branch always wins:

| Line | Source condition | Kernel it launches | Status |
|---|---|---|---|
| **122** | `m_block_ == 64 && n_block_ == 256` | `LAUNCH_KERNEL(64, 512)` | **unreachable** |
| **126** | `m_block_ == 128 && n_block_ == 256` | `LAUNCH_KERNEL(128, 512)` | **unreachable** |
| **130** | `m_block_ == 256 && n_block_ == 256` | `LAUNCH_KERNEL(256, 512)` | **unreachable** |

Consequently `(m,n) = (64,512)`, `(128,512)` and `(256,512)` fall through to the `else` at `:135`
and throw — `[KrylovGen] Uninstantiated Kernel M=… N=…` (`:136`) plus a `std::runtime_error`
(`:137`) — **even though `krylov_kernels.cu` explicitly instantiates all three**
(`INSTANTIATE(64,512)` `:154`, `INSTANTIATE(128,512)` `:158`, `INSTANTIATE(256,512)` `:162`). The
kernels exist and are compiled in; only the dispatcher cannot reach them.

`m=512 / n=256` (`:133`) and `m=n=512` (`:134`) are **unaffected** — their conditions are correct.
The fix is one line each (`256` → `512` in the second half of the three conditions).

**Status: unfixed and untouched.** This is a **submodule** defect and `src/linalg` is out of scope
for edits under the parent repo's Code Rule 1. Verified identical at both the gitlink recorded by
the parent at HEAD (`7dab002`) and the currently checked-out submodule working tree (`dc48659`) —
those two commits differ only under `tools/release/`. Practical impact today is nil: production
runs symmetric `m = n = 256`.

### Delta-16 SpMM Autotuner uint32 Overflow Guard

The GPU SpMM autotuner's Delta-16 escape encoding accumulates its expanded per-row stream length in a running sum; at very large matrices (padded `n_cols > 65535`, e.g. RSA-155 scale: ~1.7e7 rows × ~1.6e7 cols, ~7.1e8 NNZ) the un-guarded uint32 sum could overflow, wrap, under-allocate the encode buffer, and write out of bounds — corrupting the CUDA context. Fixed (submodule `block-wiedemann` v1.0.1, `07c11ad`) by summing per-row sizes in uint64 before the uint32 scan and `throw`ing if the total exceeds `UINT32_MAX`; the whole-range Phase-5b try/catch then drops Delta-16 for that call and falls back to Warp-CSR, leaving the CUDA context intact. No-op at `n_cols ≤ 65535` and for per-block slices (`n_rows ≤ 65536`); GF(2) results are unaffected.

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
| `SpMMPerformanceReport` | `cuda_spmm/include/bw_spmm_interface.h` | Timing and throughput results from autotune (A and A^T) |

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
| Standalone | `lingen_ops`, `spmm_core`, `hpc_logger` | Benchmarks (`bench_matmulgf2`, `bench_karatsuba`, `bench_lingen_apply_pi`, `bw_lingen_smoke`, `bw_lingen_bench`, `test_s_disk_io`) |
| Submodule | `lingen_ops`, `spmm_core` | Inherits `cudampqs_build_flags` from parent; no benchmarks |

Both use CUDA separable compilation. Submodule mode excludes `poly_arithmetic.cu` and `poly_arith_engine.cu` (unused in parent pipeline). Both modes compile `bw_version.cpp`.

Build options:
- `-DENABLE_LINGEN_DEVICE_SYNC=ON` — defines `BASECASE_SOLVER_CUDA_STREAM_SYNC`, enabling periodic `cudaStreamSynchronize` (every 10 steps) in Stage 2 basecase loop. Required on Turing where too many sequential kernel launches cause failures.
- `-DENABLE_CUDA_GRAPHS=ON` (default OFF) — defines `BW_ENABLE_CUDA_GRAPHS`, enabling CUDA graph capture paths in the solver; when OFF all graph code is compiled out.

**Version management:** `cmake/version.cmake` is the single source of truth for `LINALGBW_VERSION_{MAJOR,MINOR,PATCH}` (independent namespace from the parent); configured into a generated `bw_version.h` exposed via the `bw_version_header` interface library.

## Integration

```cpp
lingen::BWSolverConfig cfg;
cfg.m_block = cfg.n_block = 256;   // --bw_m / --bw_n (adaptively downscaled for small matrices)
cfg.solve_transposed = true;       // left kernel of A
cfg.stage2_gpu_mode = true;
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
- **Asymmetric block pairs `(64,512)`, `(128,512)`, `(256,512)` throw `Uninstantiated Kernel`** despite being instantiated — dead dispatch branches at `krylov_generator.cu:122/126/130`. Unfixed (submodule). See the section above.
- **Block width is effectively pinned to `m = n = 256` at RSA-150/155 scale:** 128 is a 2.687× LA regression and 512 OOMs in Stage 3. See the block-width section above.
- **`--bw_resume` has never been exercised** — only the checkpoint *save* path has run in production.
