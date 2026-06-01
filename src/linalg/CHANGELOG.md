# Changelog

All notable changes to the `block-wiedemann` library are documented in this
file.

The format is based on
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning 2.0.0](https://semver.org/).

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
