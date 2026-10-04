# Changelog

All notable changes to the `block-wiedemann` library are documented in this
file.

The format is based on
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning 2.0.0](https://semver.org/).

## [Unreleased]

## [1.0.2] - 2026-09-28

### Added
- Stage 2 initialization basis on the GPU: a dedicated single-warp kernel
  (`k_find_init_basis`, `src/lingen/stage2/init_basis.cu`) computes t0 and the
  m basis pairs directly on the device-resident Krylov sequence S, keeping a
  fully reduced (Gauss-Jordan) basis in shared memory; a second kernel
  (`k_build_f_init`) writes the initial generator F and gamma straight into
  the solver's device buffers. Only a 16-byte result record (t0, rank,
  status) returns to the host. The result is bit-identical to the CPU
  routine (t0, rank, pairs in order, F bytes, gamma, and the rank-failure
  path with its log lines); the kernel documentation carries the proof.
  Supports m <= 512 (CPU fallback with a warning above).
  New `BWSolverConfig` fields `stage2_init_on_gpu` (default true) and
  `stage2_init_cross_check` (default false; implied by the Stage 2 oracle),
  which runs both routines and aborts with `Basecase InitCheck mismatch` on
  any divergence. `bw_lingen_bench` flags `--s2_init_cpu` and
  `--s2_init_verify`. New log line
  `[Basecase] Init basis: path=GPU|CPU, t0=, rank=, time= ms`; the existing
  `Initialization: t0=` line is unchanged.
- `BWSolverConfig::stage1_force_host_S` (debug override: always keep the
  Stage 1 host copy of S).
- Test `tests/test_init_basis_gpu.cu`: device vs CPU initialization, bit for
  bit, over structured cases (rank-deficient leading coefficients, t0 > 2,
  zero columns/coefficients, dependent columns, rank failure) and 300 fuzz
  instances.
- Stage-boundary checkpointing for the Block Wiedemann solver
  (**experimental — not yet validated end-to-end**): save and load of the
  Stage 1 Krylov S-sequence, the Stage 2 lingen Pi polynomial, and the
  Stage 3 solution artifacts at stage boundaries, controlled by new
  per-stage save/load fields in `BWSolverConfig`. A checkpoint set is
  integrity-tagged with an FNV-1a hash (`_ckpt_tag.bin`); on any mismatch
  the stage is recomputed rather than loading stale data. Resume is
  stage-granular: a run restarts from the last *completed* stage, not
  mid-stage. GF(2) results are unchanged when checkpointing is off.

### Changed
- S is no longer downloaded to the host by default. Stage 2 downloads S only
  when a host consumer needs it (CPU-only mode, CPU initialization or its
  cross-check, the per-step oracle, the legacy annihilation check), and then
  as one bulk copy instead of one synchronous copy per coefficient. Stage 1
  makes its host copy of S only for a consumer (S not kept on the device, S
  saved to disk, hashing, Stage 1 checkpoints) and logs
  `[BWStage 1] S host copy: ON (...) / OFF (no consumer)`; without it the
  generator synchronizes its compute stream before returning. The post-run
  legacy oracle copies S device-to-device when no host copy exists.
  Outputs are bit-identical.

### Fixed
- The bulk variant of the Stage 2 S download (previously commented out)
  copied into empty vectors; it now assigns each coefficient.
- Stage 2 host-upload fallback with no S available now fails with a clear
  error instead of a zero-size allocation.

### Follow-up (not in this release)
- The Stage 1 S disk write (`save_S_to_disk`, checkpoint `_S.bin`) and its
  FNV hash still run synchronously after generation; an asynchronous writer
  joined before any Stage 2 use of the host copy is a possible follow-up.

## [1.0.1] - 2026-07-13

### Fixed
- SpMM autotuner: guard against a uint32 overflow of the Delta-16
  escape-expanded stream length in `gpu_convert_csr_to_delta16`
  (`cuda_spmm/src/device_format_convert.cu`). When the autotuner pads a
  matrix to square and `n_cols > 65535`, the Delta-16 escape path accumulates
  the per-row expanded stream length in uint32 (`total_deltas` and the
  `d_delta_offsets` inclusive scan). For very large matrices (e.g. ~1.7e7 rows
  with a ~1.6e7 max column index, ~7.1e8 NNZ — the RSA-155 512-bit quadratic
  sieve matrix) this exceeds 2^32, wraps, under-allocates the stream buffer,
  and the encode kernel writes out of bounds — an illegal memory access that
  poisons the CUDA context (surfacing at `gpu_autotuner.cu:666` and cascading
  through the legacy fallback at `device_csr.cu:80` / `autotuner.cpp:267`). The
  per-row sizes are now summed in uint64 before the uint32 scan; if the total
  exceeds `UINT32_MAX`, Delta-16 is rejected for that slice via
  `std::runtime_error`, which the whole-range autotune try/catch already
  handles by dropping Delta-16 and falling back to a valid kernel (e.g.
  Warp-CSR) with the context intact. No effect for `n_cols <= 65535` (fast
  path) or per-block slices (`n_rows <= 65536`); the GF(2) SpMM result is
  unchanged. Validated on RTX 5070 Ti and H100: the autotuner clears the
  previously-crashing site and selects Warp-CSR.

## [1.0.0] - 2026-05-19

Initial public release of the CUDA-accelerated Block Wiedemann solver over
GF(2).

### Added
- Three-stage Block Wiedemann solver: Krylov sequence generation (Stage 1),
  block Berlekamp-Massey linear generator (Stage 2), and parallel solution
  reconstruction (Stage 3).
- Public C++ API via `BlockWiedemannSolver` and `BWSolverConfig`
  (`include/bw_solver.h`); device-side accessor
  `BWKernelSolutionView` for in-GPU consumption of kernel vectors
  (`include/bw_solution_view.h`).
- `cuda_spmm/` SpMM sub-library: ten kernel implementations covering dense
  (M4RM, dense bitslice), warp-CSR, tiled-COO (plus unrolled), PFor-Delta
  (BitExact), Delta16, Golomb, and ELLPACK formats; templated over
  `VecType<BITS>` for 64/128/256/512-bit vector widths.
- GPU-only autotuner (`autotune_gpu_only`, default ON): on-device CSR upload,
  format conversion, and benchmarking; legacy CPU-side autotuning retained as
  a fallback.
- Adaptive A vs A^T operator selection in Stage 1 with optional S-on-device
  handoff to eliminate host-device copies between iterations.
- GPU transpose kernel and on-device accumulation in the Krylov pipeline.
- CUDA graph capture/replay for the Stage 1 inner loop and Stage 3
  projection chain (build-time `-DENABLE_CUDA_GRAPHS=ON`, runtime
  `BWSolverConfig::graph_enable`).
- Stream-aware SpMM operator overloads that elide the legacy
  `cudaDeviceSynchronize` barriers between iterations.
- Stage 2 GPU basecase solver with double-buffered preallocation and an
  `eliminationN` (512-thread) fallback kernel selected at runtime via
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for register-constrained
  GPUs.
- Stage 2 CPU reference solver (`BasecaseSolverReference`) for verification
  against the GPU implementation.
- Robust binary-header disk I/O for the Stage 1 S sequence and kernel-vector
  solutions.
- Jetson Orin Nano (SM 8.7) port: build option `-DGPU_TARGET=87`,
  `-DENABLE_LINGEN_DEVICE_SYNC=ON` recommended for SM 8.7 to prevent kernel
  launch queue overflow; runtime `is_jetson` detection respects user-pinned
  block sizes.
- Standalone benchmark binaries: `bw_lingen_smoke`, `bw_lingen_bench`,
  `bench_matmulgf2`, `bench_karatsuba`, `bench_lingen_apply_pi`,
  `test_s_disk_io`, plus the SpMM-level
  `spmm_full_benchmark`, `spmm_single_benchmark`, `spmm_verify_interface`,
  `bench_gpu_autotune`.
- Validation tests in `cuda_spmm/tests/`:
  `test_format_correctness`, `test_autotuner_ab`, `test_spmm_endtoend`,
  `test_memory_leak`, `test_golden_regression`.
- Python reference implementation (`python/block_wiedemann_lingen_v5.py`)
  and verifier (`python/verify_bw_pipeline.py`) for the smoke-test golden
  regression flow.
- LGPL-3.0-only license with NVIDIA CUDA Toolkit linking exception
  (`LICENSE`); SPDX headers on every source file.
- Public release pipeline under `tools/release/`: allowlist-driven
  extraction, anti-leakage grep sweep, and clean-build verification gate.
