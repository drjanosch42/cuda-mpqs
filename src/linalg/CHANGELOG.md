# Changelog

All notable changes to the `block-wiedemann` library are documented in this
file.

The format is based on
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning 2.0.0](https://semver.org/).

## [Unreleased]

### Added
- Stage-boundary checkpointing for the Block Wiedemann solver
  (**experimental — not yet validated end-to-end**): save and load of the
  Stage 1 Krylov S-sequence, the Stage 2 lingen Pi polynomial, and the
  Stage 3 solution artifacts at stage boundaries, controlled by new
  per-stage save/load fields in `BWSolverConfig`. A checkpoint set is
  integrity-tagged with an FNV-1a hash (`_ckpt_tag.bin`); on any mismatch
  the stage is recomputed rather than loading stale data. Resume is
  stage-granular: a run restarts from the last *completed* stage, not
  mid-stage. GF(2) results are unchanged when checkpointing is off.

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
