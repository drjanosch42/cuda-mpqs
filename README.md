# cuda-mpqs

A high-performance GPU implementation of the Self-Initializing Multiple Polynomial
Quadratic Sieve (SIQS/MPQS) for integer factorization. cuda-mpqs factors composite
integers up to 512 bits (RSA-100 through RSA-155 range) end-to-end, scaling from a
single GPU to large multi-node clusters — production runs use 64 H100 GPUs across
16 nodes, with measured coordinator headroom supporting 100+ GPUs deployments. It
combines a fully on-device sieve with a packed sparse GF(2) preprocessor and a
Block Wiedemann linear algebra solver.

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
- Periodic mid-sieve checkpointing (`--checkpoint_interval`, `--checkpoint_batches`,
  `--checkpoint_dir`, `--resume`): writes atomic `sieve.ckpt` snapshots; a killed or
  wall-clock-expired run resumes from the last checkpoint instead of re-sieving from
  zero. Default-off; coordinator-only in cluster mode.
- 512-bit (RSA-155) capability via a wide `uint16`/`u8sat` sieve accumulator
  path, tunable large-prime bucket capacity (`--bucket_size_factor`), and cluster
  scaling to 64+ GPUs across many nodes.
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

### Factorization records

Every result below is product-verified — the pipeline recomputes `p × q` and checks
it against `N` — and was produced by the standard pipeline, not by a special-cased
run. GPU-hours are summed over all participating GPUs; wall-clock is given
separately.

| Modulus | Bits | Factored | Sieve hardware | Sieve | Linear algebra | Total | Energy |
|---------|-----:|----------|----------------|-------|----------------|------:|--------|
| RSA-155 | 512 | 2026-07-14 | 64× H100, 16 nodes | 689.74 GPU-h (10.78 h wall) | 10.90 GPU-h, 1× H100 | **700.64 GPU-h** | 242.26 kWh |
| RSA-150 | 496 | 2026-08-23 | 64× H100, 16 nodes | 295.18 GPU-h (4.61 h wall) | 7.68 GPU-h, 1× H100 | **302.86 GPU-h** | 107.506 kWh |
| RSA-140 | 463 | 2026-06-29 | 16× H100, 4 nodes | ≈ 103.5 GPU-h (6 h 28 m wall) | 0.36 GPU-h, 1× H100 | **103.9 GPU-h** | not measured |
| RSA-130 | 430 | 2026-06-22 | 8× A100, 2 nodes | 25.67 GPU-h (3.21 h wall) | 0.08 GPU-h, 1× A100 | **25.75 GPU-h** | not measured |
| RSA-120 | 397 | 2026-06-06 | 1× RTX 5070 Ti | 10,498 s | 109 s | **2.947 h** | not measured |

- **RSA-155** — two 78-digit primes. 17,272,258 relations, 41.4 % of them contributed
  by the single large-prime variant; Block Wiedemann on a
  16,700,000 × 15,666,202 GF(2) matrix with 683,983,246 nonzeros. Sieve
  configuration `--fb_bound 600000000 --sieve_bound 8388608 --lp1_bound 80000000000000
  --bucket_size_factor 1.0`. The 41 % large-prime fraction is structural to
  `fb_bound = 600M`, not a tuning shortfall.
- **RSA-150** — two 75-digit (248-bit) primes. 17,269,643 relations at 44.90 %
  large-prime fraction, sieved in 4.61 h wall-clock on 64 H100 GPUs; Block Wiedemann
  on a 16,700,000 × 15,663,546 GF(2) matrix with 690,586,752 nonzeros, 7.675 h on a
  single H100 (1.796 kWh) at Block Wiedemann block width 256. Sieve configuration
  `--fb_bound 600000000 --sieve_bound 8388608 --lp1_bound 200000000000000
  --bucket_size_factor 1.0`. The linear algebra was run twice: the quoted 7.68 GPU-h is
  the block-width-256 run of record, while an earlier run of the same matrix at block
  width 128 took 20.60 GPU-h, so the machine time actually expended across the campaign
  was 323.45 GPU-h / 111.213 kWh. Block width 256 is the default; 128 is not
  recommended.
- **RSA-140** — two 70-digit primes. 3,027,706 relations at 56.8 % large-prime
  fraction; matrix 3,027,706 × 2,059,597 with 123.5 M nonzeros, solved in 21 m 31 s on
  one H100. Sieve configuration `--fb_bound 70000000 --sieve_bound 524288 --lp1_bound
  60000000000000`. That run used `--dedup_safety_factor 1.4` and over-collected roughly
  25 % of its relations; at the 1.05 default the same configuration projects to
  ≈ 78 GPU-h, but that has not been measured.
- **RSA-130** — two 65-digit primes. 1,023,475 relations at 51.8 % large-prime
  fraction on 8 A100 GPUs across 2 nodes (88.6 relations/s cluster-wide), end to end in
  ≈ 3 h 17 m. Sieve configuration `--fb_bound 30000000 --sieve_bound 131072 --lp1_bound
  13000000000000 --sieve_batch_size 16 --cuda_graph_unroll 4 --matrix_mode legacy`.
- **RSA-120** — single-GPU record: 10,609 s (2 h 56 m 49 s) end to end on one
  RTX 5070 Ti, 48.7 % large-prime fraction. Configuration `--fb_bound 24000000
  --sieve_bound 131072 --lp1_bound 1000000000000 --lp1_max_witnesses 16M
  --sieve_batch_size 16 --cuda_graph_unroll 4 --lp_interval 1 --matrix_mode legacy
  --autotune_stage1`.

**Record RSA-100 runtime:** the full pipeline factors RSA-100 (100 decimal digits)
end-to-end in **50.12 s** on a single NVIDIA H100 SXM under v1.0.6 — mean of three
product-verified runs, fastest run 49.99 s — with `--fb_bound 5500000 --sieve_bound
524288 --lp1_bound 1000000000000 --sieve_batch_size 8 --cuda_graph_unroll 0
--params 1024,8,4,8,256,1024,256,1024 --matrix_mode legacy --char_mode none`.
The per-GPU breakdown is in the table below.

### RSA-100 across GPU generations

End-to-end RSA-100 (100-digit) full-pipeline factorization time across NVIDIA GPUs,
ordered by CUDA-core count. Each device runs its own tuned configuration, so a row is
a measurement of that device at that configuration, not a device-versus-device ratio:

| GPU | Architecture (CC) | CUDA cores | RSA-100 (full pipeline) | Reps | Version |
|-----|-------------------|-----------:|-------------------------|------|---------|
| Jetson Orin Nano Super 8 GB (25 W) | Ampere (8.7) | 1,024 | 2,405.49 s (40 m 05 s) | n = 1 | 1.0.6 |
| TITAN RTX 24 GB | Turing (7.5) | 4,608 | 199.08 s (median) | n = 3 | 1.0.6 |
| A100 SXM4 40 GB | Ampere (8.0) | 6,912 | 114.79 s (mean) | n = 3 | 1.0.6 |
| RTX 5070 Ti 16 GB | Blackwell (12.0) | 8,960 | 76.03 s (mean) | n = 5 | 1.0.6 |
| H100 SXM 94 GB | Hopper (9.0) | 16,896 | 50.12 s (mean) | n = 3 | 1.0.6 |

How to read the table:

- **Operating points differ per row.** The 5070 Ti and TITAN rows run
  `--fb_bound 7000000 --sieve_bound 262144 --sieve_batch_size 32`, the A100 row
  `--fb_bound 7000000 --sieve_bound 262144 --sieve_batch_size 8`, the H100 row
  `--fb_bound 5500000 --sieve_bound 524288 --sieve_batch_size 8`, and the Jetson row
  `--fb_bound 7000000 --sieve_bound 131072 --sieve_batch_size 8`. Walls are comparable
  within a device, not across devices.
- **The A100 and H100 rows are measured, not optimized.** Both inherit
  `--sieve_batch_size 8` from the cluster probe configuration they were run under; the
  single-GPU configurations use 32, and no batch-size sweep has been run on either device at
  this operating point. Treat them as honest measurements of the configuration shown,
  not as each card's best achievable time.
- **`--params` tuples are derived from a device's SM count** and must not be copied
  between GPUs. All rows except Jetson run `--cuda_graph_unroll 0`, `--matrix_mode
  legacy`, `--char_mode none` with a pinned `--params` tuple; the Jetson row uses its
  documented `--autotune_stage1 --cuda_graph_unroll 4` configuration and is therefore a
  single autotuned run.
- **TITAN RTX shows a monotone thermal ramp across its three repetitions**
  (196.99 → 199.08 → 199.65 s, 77 → 83 °C), so the median is quoted rather than the
  mean of 198.57 s.
- **The Jetson figure is 0.4 % slower than the 1.0.5 measurement** of the same
  configuration (2,395.69 s). Both are single runs on a thermally constrained 25 W
  board, and a separate independent run of the same configuration measured
  2,395.31 s; the difference is run-to-run variation, not a regression.
- A halved shared-memory sieve geometry (`--sieve_block_size` / `--sieve_big_prime_start`,
  new in 1.0.6) measures faster still on the H100 — 48.86 s, reproduced to +0.24 % in a
  second job — but costs 7.6 % more GPU board energy and is not a recommended
  configuration, so the record above is the one the documented settings reproduce.

The table below lists end-to-end wall-clock time on a single RTX 5070 Ti
(Blackwell, CC 12.0) across input sizes, using the autotuned default parameters,
full pipeline (sieve + linear algebra + square root):

| Input         | Digits | Time (s) | Hardware     |
|---------------|--------|----------|--------------|
| 70d composite | 70     | 1.60     | RTX 5070 Ti  |
| 80d composite | 80     | 7.72     | RTX 5070 Ti  |
| 90d composite | 90     | 41.36    | RTX 5070 Ti  |
| RSA-100       | 100    | 84.72    | RTX 5070 Ti  |
| RSA-110       | 110    | 1040.03  | RTX 5070 Ti  |

Numbers reflect the full pipeline with autotuned parameters, measured under releases
1.0.1–1.0.5b; they have not been re-measured under 1.0.6. The RSA-100 row differs from
the 76.03 s in the table above because it is an `--autotune_stage1` run rather than a
run with a pinned `--params` tuple: on this card the gap is the parameter pin, not the
release. The single large prime variant (L > 0) is only used at ~90 digits and above;
the 70d and 80d rows run with L = 0. Performance on other GPUs scales roughly with
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
  version = {1.0.6},
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
