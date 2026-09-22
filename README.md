# cuda-mpqs

A high-performance GPU implementation of the Self-Initializing Multiple Polynomial
Quadratic Sieve (SIQS/MPQS) for integer factorization. cuda-mpqs factors composite
integers up to 512 bits (RSA-100 through RSA-155 range) end-to-end, scaling from a
single GPU to large multi-node clusters — production runs use 64 H100 GPUs across
16 nodes, and scaling past that is bounded by coordinator host memory (about
475 bytes per large-prime witness, so ≈ 74 GB at 64 GPUs and ≈ 93 GB projected at
108) rather than by the coordinator's matching throughput, which is measured at
roughly 125× the offered load of a 108-GPU fleet. It combines a fully on-device
sieve with a packed sparse GF(2) preprocessor and a Block Wiedemann linear
algebra solver.

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

For every result below the pipeline recomputes `p × q` and checks it against `N`,
and every one was produced by the standard pipeline, not by a special-cased run.
GPU-hours are summed over all participating GPUs; wall-clock is given
separately. The **Version** column names the release each record was produced
under; none of these campaigns has been re-run on a later release, so the figures
are historical and a current release would be expected to beat them.

| Modulus | Bits | Factored | Version | Sieve hardware | Sieve | Linear algebra | Total | Energy |
|---------|-----:|----------|---------|----------------|-------|----------------|------:|--------|
| RSA-155 | 512 | 2026-07-14 | 1.0.5e | 64× H100, 16 nodes | 689.74 GPU-h (10.78 h wall) | 10.90 GPU-h, 1× H100 | **700.64 GPU-h** | 242.26 kWh |
| RSA-150 | 496 | 2026-08-23 | 1.0.5 | 64× H100, 16 nodes | 295.18 GPU-h (4.61 h wall) | 7.68 GPU-h, 1× H100 | **302.86 GPU-h** | 107.506 kWh |
| RSA-140 | 463 | 2026-06-29 | 1.0.4d | 16× H100, 4 nodes | ≈ 103.5 GPU-h (6 h 28 m wall) | 0.36 GPU-h, 1× H100 | **≈ 103.9 GPU-h** | not measured |
| RSA-130 | 430 | 2026-06-22 | 1.0.3b | 8× A100, 2 nodes | 25.67 GPU-h (3.21 h wall) | 0.08 GPU-h, 1× A100 | **25.75 GPU-h** | not measured |
| RSA-120 | 397 | 2026-06-06 | 1.0.1 | 1× RTX 5070 Ti | 10,498 s | 109 s | **2.947 h** | not measured |

- **RSA-155** — two 78-digit primes. 17,272,258 relations, 41.4 % of them contributed
  by the single large-prime variant; Block Wiedemann on a
  16,700,000 × 15,666,202 GF(2) matrix with 683,983,246 nonzeros. Sieve
  configuration `--fb_bound 600000000 --sieve_bound 8388608 --lp1_bound 80000000000000
  --bucket_size_factor 1.0`. The linear algebra ran with `--matrix_max_rows
  16700000`, which caps the relation batch so that both matrix dimensions stay
  under the 2^24 − 1 = 16,777,215 limit above which the tiled sparse-matrix
  kernels are inadmissible; the sieve had delivered 17,272,258 rows. The 41 %
  large-prime fraction is structural to `fb_bound = 600M`, not a tuning shortfall.
- **RSA-150** — two 75-digit (248-bit) primes. 17,269,643 relations at 44.90 %
  large-prime fraction, sieved in 4.61 h wall-clock on 64 H100 GPUs; Block Wiedemann
  on a 16,700,000 × 15,663,546 GF(2) matrix with 690,586,752 nonzeros, 7.675 h on a
  single H100 (1.796 kWh) at Block Wiedemann block width 256. Sieve configuration
  `--fb_bound 600000000 --sieve_bound 8388608 --lp1_bound 200000000000000
  --bucket_size_factor 1.0`. The linear algebra ran with `--matrix_max_rows
  16700000`: the sieve delivered 17,269,643 rows, above the 2^24 − 1 = 16,777,215
  limit above which the tiled sparse-matrix kernels are inadmissible, so the cap
  was load-bearing and dropped 569,643 rows. The linear algebra was run twice: the
  quoted 7.68 GPU-h is the block-width-256 run of record, while an earlier run of
  the same matrix at block width 128 took 20.60 GPU-h, so the machine time actually
  expended across the campaign was 323.45 GPU-h / 111.213 kWh. Block width 256 is
  the default and is bracketed on both sides: 128 is a 2.68× regression on the same
  matrix, and 512 does not run at this scale — solution reconstruction requests a
  single 63.71 GiB allocation and exhausts device memory.
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
end-to-end in **38.30 s** on a single NVIDIA H100 SXM under v1.0.7 — median of three
runs, spread 37.78–38.58 s — with `--fb_bound 5500000 --sieve_bound 524288
--lp1_bound 1000000000000 --sieve_batch_size 8 --cuda_graph_unroll 0
--params11 2048,8,8,8,256,1024,1024,1024,131072,8100,260 --sieve_hc_dim 12
--lp1_max_witnesses 8388608 --bw_max_solutions 64 --matrix_mode legacy
--char_mode none`. The per-GPU breakdown is in the table below.

### RSA-100 across GPU generations

End-to-end RSA-100 (100-digit) full-pipeline factorization time across NVIDIA GPUs,
ordered by CUDA-core count. Each device runs its own tuned configuration, so a row is
a measurement of that device at that configuration, not a device-versus-device ratio:

| GPU | Architecture (CC) | CUDA cores | RSA-100 (full pipeline) | Reps | Version |
|-----|-------------------|-----------:|-------------------------|------|---------|
| Jetson Orin Nano Super 8 GB (25 W) | Ampere (8.7) | 1,024 | 1,602.64 s (26 m 43 s, mean)† | n = 2 | 1.0.7 |
| TITAN RTX 24 GB | Turing (7.5) | 4,608 | 137.76 s (median, 137.48–138.03) | n = 3 | 1.0.7 |
| A100 SXM4 40 GB | Ampere (8.0) | 6,912 | 83.80 s (median, 83.66–87.33) | n = 3 | 1.0.7 |
| RTX 5070 Ti 16 GB | Blackwell (12.0) | 8,960 | 66.38 s (median, 66.35–68.21) | n = 3 | 1.0.7 |
| H100 SXM 94 GB | Hopper (9.0) | 16,896 | 38.30 s (median, 37.78–38.58) | n = 3 | 1.0.7 |

† The Jetson row was **not** re-measured under the benchmark protocol the other four
rows share (see the next bullet), so it is not directly comparable even as a
version-to-version figure.

How to read the table:

- **The four non-Jetson rows share one benchmark protocol**: full pipeline,
  `--lp1_max_witnesses 8388608 --bw_max_solutions 64 --matrix_mode legacy
  --char_mode none`, every run product-verified. The Jetson row predates that
  protocol by one day and runs neither cap, so its wall is not on the same footing;
  it is quoted because it is the only 1.0.7 measurement of that board.
- **Operating points differ per row.** All five rows run `--lp1_bound 1000000000000`.
  The TITAN row runs `--fb_bound 7000000 --sieve_bound 262144 --sieve_batch_size 32`,
  the A100 and 5070 Ti rows the same bounds at `--sieve_batch_size 8`, the H100 row
  `--fb_bound 5500000 --sieve_bound 524288 --sieve_batch_size 8`, and the Jetson row
  `--fb_bound 7000000 --sieve_bound 131072 --sieve_batch_size 8`. Walls are comparable
  within a device, not across devices.
- **Every row runs a pinned `--params11` tuple found by `--param_test` on that
  device**, together with `--sieve_hc_dim 12`:

  | GPU | `--params11` tuple |
  |-----|--------------------|
  | Jetson Orin Nano Super | `2048,4,32,4,8,1024,1024,512,65536,1284,164` |
  | TITAN RTX | `1024,16,8,16,128,1024,1024,1024,32768,2340,164` |
  | A100 SXM4 | `2048,8,4,8,512,1024,1024,512,65536,1284,164` |
  | RTX 5070 Ti | `2048,8,8,8,256,1024,1024,1024,65536,2820,96` |
  | H100 SXM | `2048,8,8,8,256,1024,1024,1024,131072,8100,260` |

  A tuple encodes the device's shared-memory size, multiprocessor count and
  factor-base partition, so **tuples must not be copied between GPUs**. All rows
  except Jetson run `--cuda_graph_unroll 0`; the Jetson row runs
  `--cuda_graph_unroll 4 --lp_interval 1 --lp1_hash_bits 21`.
- **The TITAN row additionally passes `--bucket_size_factor 1.0`**, which its tuple
  requires: without it the large-prime bucket on that device runs at 98 % occupancy
  and silently discards hits. The other four rows let 1.0.7 size the bucket from the
  predicted peak.
- **These are measurements of the configurations shown, not each card's best
  achievable time.** No batch-size or CUDA-graph sweep has been run at these
  operating points on any of the five devices.

### Scaling with input size

The table below lists end-to-end wall-clock time on a single RTX 5070 Ti
(Blackwell, CC 12.0) across input sizes, full pipeline (sieve + linear algebra +
square root):

| Input         | Digits | Time (s) | Hardware     |
|---------------|--------|----------|--------------|
| 70d composite | 70     | 2.88     | RTX 5070 Ti  |
| 80d composite | 80     | 9.88     | RTX 5070 Ti  |
| 90d composite | 90     | 19.63    | RTX 5070 Ti  |
| RSA-100       | 100    | 75.01    | RTX 5070 Ti  |
| RSA-110       | 110    | 638.67   | RTX 5070 Ti  |

Measured under **v1.0.7**. Parameters come from `--autotune_stage1` — the autotuner
chooses them at run time — rather than from a pinned tuple, and `--autotune_no_history`
prevents any earlier run's result being carried in. Each row is the median of three
timed runs preceded by a discarded warm-up; every run was product-verified.

The RSA-100 row reads 75.01 s here against **66.38 s** in the cross-GPU table above.
The difference is the parameter pin, not the release: that table runs this card's
pinned `--params11` tuple, while this one autotunes.

**The 70d and 80d rows are slower than the 1.0.1-era figures they replace** — 2.88 s
against 1.60 s (+80 %) and 9.88 s against 7.72 s (+28 %). The cost is entirely
post-sieve. At 70d the sieve accounts for 0.65 s of the 2.88 s total and the linear
algebra for 2.13 s; at 80d the sieve is 6.72 s of 9.88 s. The three larger rows, where
the sieve is two thirds or more of the wall, all improved against the same baseline —
the 90d row by roughly a factor of two and RSA-110 by close to 40 %.

The single large prime variant (L > 0) is only used at ~90 digits and above; the 70d
and 80d rows run with L = 0. The RSA-110 row additionally passes
`--cuda_graph_lp_stride 4`: at L = 1T on this card a per-batch large-prime cadence
pushes the relation graph past the point where the square-root step can recover a
factor. Performance on other GPUs scales roughly with sieve-relevant SM count and
memory bandwidth.

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

The archived releases carry the concept DOI
[10.5281/zenodo.21619473](https://doi.org/10.5281/zenodo.21619473), which always
resolves to the latest cuda-mpqs release. Cite it unless you need to pin a
specific version, in which case use that version's own DOI from the Zenodo
record.

BibTeX:

```bibtex
@software{cuda_mpqs_2026,
  author  = {Heinrichs, Christoph and Januszewski, Fabian},
  title   = {cuda-mpqs: A GPU-accelerated Self-Initializing Multiple Polynomial
             Quadratic Sieve},
  year    = {2026},
  version = {1.0.7},
  doi     = {10.5281/zenodo.21619473},
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
