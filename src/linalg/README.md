# Block Wiedemann Solver for GF(2)

A CUDA-accelerated solver for finding kernel vectors of large sparse binary
matrices over the finite field GF(2). Used as the linear-algebra stage of the
`cuda-mpqs` quadratic-sieve implementation.

## Author

Fabian Januszewski
<fabian.januszewski@math.uni-paderborn.de>
<drjanosch@gmail.com>

## Repository

https://github.com/drjanosch42/block-wiedemann

## License

LGPL-3.0-only with the NVIDIA CUDA Toolkit linking exception. See
[`LICENSE`](LICENSE) for the full text. In short: you may link this library
with the proprietary NVIDIA CUDA Toolkit and convey the resulting binaries,
provided you comply with the LGPLv3 for everything outside the CUDA Toolkit.

## What it does

Given a sparse matrix `A` over GF(2) (typically tall-and-skinny, several
million rows and tens of nonzeros per row) the solver computes vectors
`v` such that `A · v = 0 (mod 2)`. It implements the three-stage
Block Wiedemann algorithm:

1. **Stage 1 — Krylov sequence generation.** Iterates the SpMM operator
   `L` times to build the scalar sequence `S_k = X^T · A^(k+1) · Z`,
   where `X`, `Z` are random GF(2) block vectors. This stage dominates
   runtime at production scale and uses the optimized SpMM kernels in
   the `cuda_spmm/` sub-library. Kernel are by default chosen by a
   fast autotuning procedure.

2. **Stage 2 — Lingen (linear generator).** Solves a Block
   Berlekamp–Massey problem on the scalar sequence to find a generator
   polynomial `Pi(x)` such that `S(x) · Pi(x) ≡ 0 (mod x^L)`. A GPU
   basecase solver operates on dense `(m+n)×(m+n)` matrices over GF(2);
   a CPU reference implementation is also provided.

3. **Stage 3 — Solution reconstruction.** Evaluates `Pi` at `A` to
   produce kernel vectors. Uses batch-mode parallel reconstruction with
   a circular history buffer for backtracking.

*All* stages are (by default) executed on GPU. Correctness of matrix
multiplication with chosen kernels is by default validated against CPU
reference as pre-flight check.

Stage 2 is currently Block-Berlekamp-Massey only. Thomé's subquadratic
divide-and-conquer approach will be fully implemented at a later stage.

Validation cascade: GPU / CUDA is validated against the reference CPU
implementation. The reference CPU implementation is validated against
the (extremely slow) python reference implementation `python/`.

## Requirements

- CUDA Toolkit (tested with 11.x and 12.x)
- C++17 compiler (host) and CUDA 17 standard (device)
- CMake 3.22 or newer
- OpenMP (host-side parallelism)

## Directory layout

```
include/                Public headers (BlockWiedemannSolver, BWSolverConfig, ...)
src/lingen/             Solver core
  bw_solver.cu          Top-level orchestration
  stage1/               Krylov sequence generator
  stage2/               Basecase Block-BM solver (GPU + CPU reference)
  stage3/               Solution reconstructor
  operations/           Polynomial arithmetic, GF(2) matrix multiply
  io/                   Checkpointing, hashing helpers
cuda_spmm/              SpMM sub-library (sparse matrix × block-vector kernels)
  src/, include/        10 kernel implementations + autotuner
benchmarks/             Standalone benchmark executables
utils/                  Shared utilities (hpc_logger)
python/			A reference Python implementation for validation purposes
docs/                   Internal design and audit notes
```

## Build

### Standalone

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
```

`GPU_TARGET` accepts `native` (auto-detect), an architecture name
(`Turing`/`Ampere`/`Ada`/`Hopper`/`Blackwell`), or an explicit SM number
(e.g. `87` for Jetson Orin, `120` for Blackwell). Standalone builds also
produce benchmark executables in `build/`:

```bash
./build/bw_lingen_smoke           # Quick smoke test
./build/bw_lingen_bench           # Full Block Wiedemann benchmark
./build/bench_matmulgf2           # GF(2) matrix multiplication
./build/bench_karatsuba           # Karatsuba polynomial multiplication
./build/bench_lingen_apply_pi     # Pi-application benchmark
```

### As part of `cuda-mpqs`

When built as part of the parent project, only the static library
target `lingen_ops` is produced (no benchmark binaries). The parent
`CMakeLists.txt` adds this directory via `add_subdirectory(src/linalg)`.

### Useful options

- `-DENABLE_LINGEN_DEVICE_SYNC=ON` — periodic stream synchronization in
  the Stage 2 basecase loop. Recommended for Turing and SM 8.7 hardware
  where long kernel-launch chains otherwise overflow the queue.
- `-DENABLE_CUDA_GRAPHS=ON` — compiles CUDA graph capture/replay paths
  in Stages 1 and 3. Activated at runtime via `BWSolverConfig::graph_enable`.
  Requires CUDA 10.1+. No measurable speedup at production scale; off
  by default.

## Public API

```cpp
#include "bw_solver.h"

lingen::BWSolverConfig cfg;
cfg.m_block = 256;         // block dimensions 64, 256, 512 are supported,
cfg.n_block = 256;         // block dim 256 is a sweet spot on most GPUs
cfg.nrows   = A.n_rows;
// ... configure stage1/stage2/stage3 sub-configs as needed ...

lingen::BlockWiedemannSolver solver(cfg, A);
solver.Solve();            // AutoTune → Stage 1 → Stage 2 → Stage 3
const auto& kernel_vectors = solver.get_solutions();
```

`BWSolverConfig` is the single source of truth for parameters. Field
prefixes denote the stage they affect: `autotune_`, `stage1_`,
`stage2_`, `stage3_`. Individual stages may be skipped via
`stageN_skip` flags for partial / restart workflows.

## Algorithm references

- Coppersmith, "Solving homogeneous linear equations over GF(2) via
  block Wiedemann algorithm", Math. Comp. 62 (1994).
- Thomé, "Subquadratic computation of vector generating polynomials and
  improvement of the block Wiedemann algorithm", J. Symbolic Comput.
  33 (2002).
