# cuda-mpqs Architecture

This document gives an overview of how cuda-mpqs is organized: the six-stage
factorization pipeline, the available execution modes, the GPU build targets,
the source-tree layout, and validated performance figures. For day-to-day
command-line usage see [`USER_GUIDE.md`](../USER_GUIDE.md); for build
instructions see [`BUILD.md`](../BUILD.md); for distributed sieving see
[`CLUSTER.md`](../CLUSTER.md).

---

## Pipeline Overview

cuda-mpqs implements the Self-Initializing Multiple Polynomial Quadratic Sieve
(SIQS/MPQS). For an input composite `N`, the pipeline searches for pairs
`(X, Y)` with `X^2 ≡ Y^2 (mod N)` such that `gcd(X ± Y, N)` is a non-trivial
factor. This is realized as six sequential stages driven by a single
orchestrator:

```
                        +---------------------------+
   N (decimal)  --->    |  1. Parameter Tuning      |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  2. Autotuning (optional) |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  3. GPU Sieving           |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  4. Matrix Construction   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  5. Block Wiedemann       |
                        |     Linear Algebra        |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |  6. Square Root           |
                        |     Refinement            |
                        +-------------+-------------+
                                      |
                                      v
                              factor1, factor2
```

| # | Stage                      | Purpose                                                                                                |
|---|----------------------------|--------------------------------------------------------------------------------------------------------|
| 1 | Parameter Tuning           | Heuristic selection of factor base bound `F`, sieve interval `M`, and large prime bound `L` from `N`.   |
| 2 | Autotuning (optional)      | Multi-stage optimizer that refines `(F, M, L)` and GPU kernel parameters via truncated probes.          |
| 3 | GPU Sieving                | Polynomial enumeration `Q(x) = (ax + b)^2 - N` and on-device collection of smooth and partial relations.|
| 4 | Matrix Construction        | Builds a sparse GF(2) matrix from collected relations, with optional preprocessing (singletons, merges).|
| 5 | Block Wiedemann            | Solves for kernel vectors of the GF(2) matrix using a blocked Wiedemann algorithm.                      |
| 6 | Square Root Refinement     | Reconstructs `(X, Y)` per kernel vector and extracts non-trivial factors via `gcd`.                     |

### Stage details

**Parameter Tuning.** Picks initial values of `F`, `M`, and `L` (and target
relation count) based on the bit-length of `N`, generates the factor base,
and initializes polynomial coefficients.

**Autotuning.** When enabled, runs in up to four sub-stages: history-based
projection (Stage 0), kernel-parameter micro-benchmarking (Stage 1),
truncated-probe runtime estimation (Stage 2), and coordinate descent over
`(F, M, L)` (Stage 3). Results are persisted to a JSON history file. Even
without `--autotune`, prior history can be applied automatically.

**GPU Sieving.** A three-kernel pipeline (polynomial generation, meta-sieve,
sieve-and-scan) enumerates polynomials over a hypercube of factor base
primes with Gray-code traversal of `b`-values. Three execution paths are
supported: legacy (single-step), batch (no LP), and batch (LP). The batch
path uses double-buffered zero-sync overlap of sieving and post-processing
and supports CUDA graph capture for low-launch-overhead replay.

**Matrix Construction.** Two backends:
- *Legacy.* Projected matrix with `FB + 2` columns; suitable for low LP fractions.
- *Preprocessed.* Expanded matrix with `FB + 2 + LP` columns followed by
  singleton removal, batch merges, optional truncation, and 32 product
  character columns. The GPU preprocessor uses a packed CSR representation
  in which each entry packs a 24-bit column index and an 8-bit exponent into
  a single `uint32_t`, eliminating the merge tree by tracking
  Montgomery `sqrt_Q` products through merges.

**Block Wiedemann Linear Algebra.** Three sub-stages — Krylov sequence
generation, Berlekamp–Massey (lingen), and reconstruction — produce kernel
vectors of the GF(2) matrix. The GPU SpMM kernel ships with multiple
implementations and an auto-tuner.

**Square Root Refinement.** Three-phase `X` computation and parallel `Y`
computation process all kernel vectors simultaneously on the GPU. A CPU
fallback path is available for debugging and benchmarking.

---

## Execution Modes

The orchestrator supports several execution modes, selected by command-line
flag. Only one mode is active at a time.

| Flag                | Mode             | Stages run                                | Disk I/O                                   |
|---------------------|------------------|-------------------------------------------|--------------------------------------------|
| `--full` (default)  | `FULL_PIPELINE`  | Tuning, [Autotune], Sieve, Matrix, LA, √  | Optional save with `--disk_io`             |
| `--sieve_only`      | `SIEVE_ONLY`     | Tuning, Sieve                             | Always saves relations to the work dir     |
| `--linalg_only`     | `LINALG_ONLY`    | Tuning, Matrix, LA, √                     | Loads previously saved relations           |
| `--matrix_only`     | `MATRIX_ONLY`    | Matrix replay over stored relations       | Loads previously saved relations           |
| `--autotune_only`   | `AUTOTUNE_ONLY`  | Tuning, Autotune                          | Reads/writes autotune history              |
| `--param_test`      | `PARAM_TEST`     | Tuning, sieve grid search                 | None                                       |
| `--estimate_only`   | (special)        | Truncated sieve probe + estimation        | None                                       |

`--sieve_only` followed by `--linalg_only` (or `--matrix_only`) provides a
clean split between the long-running sieve phase and the matrix/linear
algebra phases, which is useful for re-running the back end with different
parameters without re-sieving.

---

## GPU Build Targets

`GPU_TARGET` controls which CUDA architectures the build emits code for.

| Value                 | Architecture                      |
|-----------------------|-----------------------------------|
| `native`              | Auto-detect (default)             |
| `Turing` / `75`       | SM 7.5 (RTX 20-series, GTX 16-series) |
| `Ampere` / `80`       | SM 8.0 (A100, RTX 30-series)      |
| `Ada` / `89`          | SM 8.9 (RTX 40-series)            |
| `Hopper` / `90`       | SM 9.0 (H100)                     |
| `Blackwell` / `120`   | SM 12.0 (RTX 50-series, B200)     |
| `Orin` / `87`         | SM 8.7 (Jetson AGX/Orin/Orin Nano)|
| `all`                 | All supported architectures       |
| Any numeric SM        | Specific compute capability       |

Native auto-detection requires CMake 3.24+. On older CMake (e.g. Jetson),
the build falls back to parsing `nvidia-smi` output.

---

## Module Overview

The source tree is organized as a small set of focused modules, each with
its own `CMakeLists.txt`. One git submodule (`src/linalg/`, the Block
Wiedemann solver) is included.

| Path                  | Module          | Description                                                                                       |
|-----------------------|-----------------|---------------------------------------------------------------------------------------------------|
| `src/common/`         | Common          | 512-bit integers, Montgomery arithmetic, SoA containers, structured logger.                       |
| `src/sieve/`          | Sieve           | GPU polynomial sieving: 3-kernel pipeline, batch and legacy modes, optional CUDA graph capture.   |
| `src/postprocessing/` | Postprocessing  | Batch trial division, SoA accumulation, deduplication, smoothness checks, yield prediction.       |
| `src/largeprimes/`    | Large Primes    | Single large prime variant via GPU slab hash table for partial pairing.                           |
| `src/matrix/`         | Matrix          | Sparse GF(2) matrix construction, CPU and GPU preprocessing, merges, truncation, character cols.  |
| `src/linalg/`         | Linear Algebra  | Block Wiedemann solver, GF(2) SpMM (submodule).                                                   |
| `src/sqrt/`           | Square Root     | GPU-parallel reconstruction of `(X, Y)` and factor extraction.                                    |
| `src/autotune/`       | Autotune        | 4-stage parameter optimizer, joint `(F, L)` convex optimizer, persistent history, auto-apply.     |
| `src/orchestrator/`   | Orchestrator    | Pipeline driver, configuration, execution modes, disk I/O.                                        |
| `src/cluster/`        | Cluster         | Distributed sieve: data tap abstraction, TCP transport, work distribution, CPU LP path.           |
| `include/`            | Public headers  | `orchestrator.h` (configuration struct, top-level orchestrator class).                            |
| `tests/`              | Tests           | `cuda-mpqs` driver binary and test composites.                                                    |
| `tools/`              | Scripts         | Composite generation, log parsing, cluster launch helpers.                                        |

All CUDA libraries enable separable compilation (RDC). The final executable
is linked with `CUDA_RESOLVE_DEVICE_SYMBOLS ON` so device symbols are
resolved across static libraries.

---

## Performance Notes

The following figures are wall-clock times for the full pipeline (sieve +
matrix + linear algebra + square root) on a single RTX 5070 Ti (Blackwell,
SM 12.0), using autotuned parameters and the single large prime variant
where applicable.

| Input         | Digits | `F`   | `M`    | `L`   | Time (s) |
|---------------|--------|-------|--------|-------|----------|
| 70d composite | 70     | 300K  | 262K   | 0     | 1.60     |
| 80d composite | 80     | 700K  | 65K    | 0     | 7.72     |
| 90d composite | 90     | 3M    | 262K   | 300M  | 41.36    |
| RSA-100       | 100    | 7M    | 262K   | 1T    | 119.49   |
| RSA-110       | 110    | 9M    | 262K   | 1T    | 1040.03  |

Below ~85 digits the large prime variant is disabled; above that threshold,
enabling LP dramatically increases relation yield.

Performance on other GPUs scales roughly with sieve-relevant SM count and
memory bandwidth. Jetson Orin (8 SMs, 1 MB L2) prefers larger `F` and
smaller `M` than discrete GPUs at the same input size. Detailed
parameter-tuning guidance for various inputs and devices is in
[`USER_GUIDE.md`](../USER_GUIDE.md).
