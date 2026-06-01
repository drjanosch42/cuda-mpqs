# cuda-mpqs

A high-performance GPU implementation of the Self-Initializing Multiple Polynomial
Quadratic Sieve (SIQS/MPQS) for integer factorization. cuda-mpqs factors composite
integers up to 512 bits (RSA-100 through RSA-155 range) end-to-end on a single GPU
or across a small cluster of GPUs, combining a fully on-device sieve with a packed
sparse GF(2) preprocessor and a Block Wiedemann linear algebra solver.

## Features

- Complete six-stage SIQS/MPQS pipeline: parameter tuning, optional autotuning,
  GPU sieving, sparse matrix construction, Block Wiedemann linear algebra, and
  GPU-parallel square root refinement.
- Self-initializing polynomial enumeration with hypercube walk and Gray-code
  traversal of `b`-values.
- Single large prime variant via a GPU slab hash table for partial-relation
  pairing.
- 4-stage parameter autotuner with persistent history, including a joint `(F, L)`
  convex optimizer and zero-probe history-based auto-apply.
- Packed GPU matrix preprocessing (singleton removal, batch merges, compaction
  cycles, GF(2)-aware planning, character columns).
- Block Wiedemann GF(2) solver with auto-tuned SpMM kernels.
- Single-node and multi-node cluster execution with TCP transport, async network
  data tap, and dynamic work distribution.
- CUDA graph batch unrolling for low-launch-overhead steady-state sieving.
- Validated on consumer NVIDIA GPUs from Turing (CC 7.5) through Blackwell
  (CC 12.0), including Jetson Orin (CC 8.7).

## Hardware Requirements

- NVIDIA GPU with compute capability **7.5+** (Turing or newer).
  Validated on Turing (RTX 20-series), Ampere, Ada, Blackwell (RTX 50-series),
  and Jetson AGX/Orin (CC 8.7).
- 8 GB GPU memory recommended for inputs up to RSA-110; more for larger inputs.
  On 8 GB cards, RSA-100 and above may require `--lp1_max_witnesses 1M` (or smaller)
  to avoid out-of-memory — the default large-prime witness capacity is auto-sized
  for throughput, not capped to available VRAM.
- CUDA Toolkit 12.x or newer.
- Linux host. Other platforms have not been tested.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/drjanosch42/cuda-mpqs
cd cuda-mpqs
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
./build/tests/cuda-mpqs --RSA100 --verbose
```

The default invocation (no arguments) factors a built-in ~80-digit composite.
The binary returns exit code `0` on successful factorization, `1` otherwise.

## Build

cuda-mpqs uses CMake. The minimum supported toolchain is:

- CUDA Toolkit 12.x (with `nvcc`)
- CMake 3.22+ (3.24+ recommended for native GPU detection)
- C++20 host compiler (GCC 11+ or Clang 14+)
- OpenMP

A typical build for the host GPU:

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
```

For Jetson Orin or other targets, set `GPU_TARGET` accordingly (see
[`docs/architecture.md`](docs/architecture.md) for the full table).

For complete build instructions including cross-compilation and submodule
handling, see [`BUILD.md`](BUILD.md).

## Usage

The main binary is `./build/tests/cuda-mpqs`. Common modes:

```bash
./build/tests/cuda-mpqs --RSA100 --verbose                      # full pipeline
./build/tests/cuda-mpqs --N <decimal_number>                    # custom input
./build/tests/cuda-mpqs --RSA100 --autotune --verbose           # autotune + run
./build/tests/cuda-mpqs --RSA100 --autotune_only --verbose      # autotune, print results, exit
./build/tests/cuda-mpqs --RSA100 --estimate_only --verbose      # runtime probe (broken at RSA-120+, see below)
./build/tests/cuda-mpqs --RSA100 --sieve_only --verbose         # save relations to disk
./build/tests/cuda-mpqs --RSA100 --linalg_only --verbose        # replay: load relations → matrix → BW → sqrt
./build/tests/cuda-mpqs --RSA100 --matrix_only --verbose        # replay: load v2 relations → matrix → BW → sqrt
```

For the full command-line reference, parameter tuning notes, execution modes,
and worked examples, see [`USER_GUIDE.md`](USER_GUIDE.md).

## Cluster Mode

cuda-mpqs supports distributed sieving across multiple GPU nodes connected
over a LAN. One node acts as coordinator and runs the matrix, linear algebra,
and square root stages locally; one or more workers contribute sieving
throughput. Solo-mode performance is unaffected — cluster code paths have zero
overhead when `--cluster_mode` is not set. See [`CLUSTER.md`](CLUSTER.md) for
setup, parameters, and launch examples.

## Performance

End-to-end wall-clock time on a single RTX 5070 Ti (Blackwell, CC 12.0), full
pipeline (sieve + linear algebra + square root):

| Input         | Digits | Time (s) | Hardware     |
|---------------|--------|----------|--------------|
| 70d composite | 70     | 1.60     | RTX 5070 Ti  |
| 80d composite | 80     | 7.72     | RTX 5070 Ti  |
| 90d composite | 90     | 41.36    | RTX 5070 Ti  |
| RSA-100       | 100    | 119.49   | RTX 5070 Ti  |
| RSA-110       | 110    | 1040.03  | RTX 5070 Ti  |

Numbers reflect the full pipeline with autotuned parameters. The single large
prime variant (L > 0) is only used at ~90 digits and above; the 70d and 80d
rows run with L = 0. Performance on other GPUs scales roughly with
sieve-relevant SM count and memory bandwidth.

## Known Limitations

- **`--sqrt_only` mode is broken.** Use `--linalg_only` (load relations → matrix
  → BW → sqrt) or the default full pipeline instead.
- **Single large prime below ~85 digits causes 100 % sqrt failure.** This is a
  mathematical limitation of the LP variant at small sizes, not a bug. Run
  without LP (`--lp1_bound 0`) for inputs below ~85 digits.
- **`--estimate_only` is broken at RSA-120+.** The mode forces legacy sieve,
  which is inaccurate at that scale. Use a short `--sieve_only` run to estimate
  throughput instead.

## Citation

If you use cuda-mpqs in academic work, please cite it. A
[`CITATION.cff`](CITATION.cff) file is provided for citation managers.

BibTeX:

```bibtex
@software{cuda_mpqs_2026,
  author  = {Heinrichs, Christoph and Januszewski, Fabian},
  title   = {cuda-mpqs: A GPU-accelerated Self-Initializing Multiple Polynomial
             Quadratic Sieve},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/drjanosch42/cuda-mpqs},
  license = {LGPL-3.0-only}
}
```

## License

cuda-mpqs is licensed under the **GNU Lesser General Public License version 3
only (LGPL-3.0-only)**, with an additional permission granted under Section 7
of the GNU GPL version 3 to allow linking with the proprietary NVIDIA CUDA
Toolkit (including `libcudart` and related libraries).

The exception clause and the full text of the LGPL/GPL are reproduced in
[`LICENSE.md`](LICENSE.md) and [`LICENSE.GPL`](LICENSE.GPL). See
[`COPYRIGHT.md`](COPYRIGHT.md) for copyright and attribution details.

Copyright © 2025-2026 Christoph Heinrichs and Fabian Januszewski.

## Acknowledgements

- The GPU sieving kernels in `src/sieve/` are derived from earlier sieve code by
  Christoph Heinrichs. The original
  sources have been substantially restructured, extended with batch and large
  prime modes, and integrated into the cuda-mpqs pipeline.
- The Block Wiedemann linear algebra implementation in `src/linalg/` was
  written by Fabian Januszewski and is also released as a standalone library
  at [block-wiedemann](https://github.com/drjanosch42/block-wiedemann) under
  the same license.

See [`AUTHORS.md`](AUTHORS.md) for the full contributor list.
