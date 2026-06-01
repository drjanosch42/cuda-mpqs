# GPU Quadratic Sieve Kernel

`src/sieve/` implements the GPU-resident polynomial sieving stage of cuda-mpqs.
Given a set of polynomials `Q(x) = (a x + b)^2 - N` enumerated by hypercube /
Gray-code traversal of factor-base primes, the kernels produce candidate
relations whose `Q(x)` is `F`-smooth (or near-smooth, for the LP variant).

## Description

The sieving subsystem is a 3-kernel CUDA pipeline:

1. **`globalMetaSieveKernel`** — distributes per-prime root updates into
   global "meta" buckets, segmented by polynomial and sieving block.
2. **`sieveKernel`** — accumulates byte-sized log-prime increments into
   shared-memory tiles for each sieving block.
3. **`sieveAndScanKernel`** — combines the meta-bucket dump with a final
   threshold scan, emitting `candidateRelation` records into a per-block
   buffer ready for the postprocessing stage.

Polynomial coefficients (`a`, `B`, factor-index lists) are generated on
device via `generatePolynomialsKernel`, fed by the host's flattened
`prepareNextBatchIndices()` plan. The traversal uses standard reflected
Gray codes (`graycode.cuh`) so consecutive polynomials differ in exactly
one prime, allowing O(1) root updates between steps.

## Module Structure

| File | Role |
|------|------|
| `kernel.cu` / `kernel.cuh` | Sieve pipeline kernels and host launchers (init, meta-sieve, sieve-and-scan). |
| `prime_algorithms.cu` / `prime_algorithms.h` | Tonelli–Shanks, Jacobi symbol, Hensel lifting, factor-base generation, hypercube path / `init_a_factors` / `recalc_a` / `prepareNextBatchIndices`. |
| `device_sieving_controller.cpp` / `device_sieving_controller.h` | Host-side orchestration: device buffer lifecycle, batch staging, configuration, snapshot/`resetAndAdvanceTo`, postprocessor handshake. |
| `graycode.cuh` | `__host__ __device__` Gray-code helpers (`gray`, `advanceGray`, `grayBitToFlip`). |
| `sieving_data_structs.h` | Shared host/device struct definitions: `primeDataSIQS`, `candidateRelation`, `DenseCandidate`, `devicePointers`, all sieve config structs, `gpuInfo`. |
| `common.h` | Sieve-stage `factoringData` (N, factor base, polynomial state) and `AFactorsSnapshot`. |
| `json_helper.h` | Minimal `JSONString` / `JSON_IO` builder used only by the optional debug-snapshot path. |
| `debug_dump.cu` / `debug_dump.cuh` / `debug_dump.h` | Optional candidate-buffer dump kernel and JSON pretty-printer for offline inspection. |
| `CMakeLists.txt` | Integrated build configuration (declares the `mpqs_sieve` static library). |

## Build

The sieve module is **integrated-build only** — its `CMakeLists.txt`
is consumed by the parent cuda-mpqs project, which provides the
`mpqs_common` target, the `cudampqs_build_flags` interface library,
and the global include directories for `src/common/` and
`src/common/logger/`. There is no standalone build path.

From the repository root:

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
```

This produces the static library `mpqs_sieve`, linked into the final
`cuda-mpqs` executable.

CUDA separable compilation (RDC) is enabled: kernels link across
translation-unit boundaries, so any new CUDA source files added here
must keep `CUDA_SEPARABLE_COMPILATION ON`.

## Origin

The kernel core derives from earlier sieve code by Christoph Heinrichs, with
substantial restructuring for cuda-mpqs: namespace migration into
`mpqs::sieve`, strict 32/64-bit integer typing, fixed-`uint512`
arithmetic, batch / CUDA-graph execution paths, `AFactorsSnapshot`
persistence (for cluster worker resync), pinned-host async H2D copies,
and the postprocessing handshake protocol.

## License

Licensed under **GNU Lesser General Public License v3.0**, with an
NVIDIA CUDA Toolkit linking exception. See the `LICENSE.md` file at the
repository root for the full license text and exception clause.

Each source file in this directory carries an `SPDX-License-Identifier:
LGPL-3.0-only` header.
