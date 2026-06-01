# Usage Manual

## Overview

This a Block Wiedemann linear algebra implementation (used as a submodule in `cuda-mpqs`) -- a CUDA-accelerated solver for finding kernel vectors of sparse binary matrices over GF(2). When built in standalone mode, it produces several benchmark and test binaries.

## Build instructions

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j16
```

## Build Options

These CMake flags affect runtime behavior of the binaries:

| Flag                             | Default    | Description |
|----------------------------------|------------|-------------|
| `-DGPU_TARGET=<target>`          | (required) | GPU architecture: `native`, `Turing`/`75`, `Ampere`/`80`, `Ada`/`89`, `Hopper`/`90`, `Blackwell`/`120`, `all`, or a specific SM number |
| `-DENABLE_CUDA_GRAPHS=ON`        |        OFF | Compile CUDA graph capture/replay paths in Stages 1 and 3. At runtime, also requires `--graph_enable` in `bw_lingen_bench` |
| `-DENABLE_LINGEN_DEVICE_SYNC=ON` |        OFF | Periodic stream synchronization in Stage 2 basecase loop. Required on Turing; recommended on Jetson Orin (SM 8.7) |

## Binary Reference

All binaries are built to `build/` when configured in standalone mode.

---

### bw_lingen_bench

Full Block Wiedemann pipeline benchmark. Generates a random sparse matrix and runs the complete solver (AutoTune, Stage 1, Stage 2, Stage 3), then verifies the resulting kernel vectors.

**Source:** `benchmarks/bw_lingen_benchmark.cu`

#### Generator Parameters

| Flag             | Type  | Default | Description |
|------------------|-------|---------|-------------|
| `--rows <N>`     | int   |   10100 | Number of matrix rows (relations) |
| `--cols <N>`     | int   |   10000 | Number of matrix columns (factor base) |
| `--alpha <F>`    | float |    25.0 | Density parameter (approx. average NNZ per row) |
| `--gen_seed <N>` | int   |      43 | Seed for matrix generation |

#### Global Solver Settings

| Flag                    | Type   | Default  | Description |
|-------------------------|--------|----------|-------------|
| `--seed <N>`            | int    |    12345 | Solver random seed |
| `--device <ID>`         | int    |        0 | CUDA device ID |
| `--m_block <N>`         | int    |       64 | Block size m (left projection / S rows) |
| `--n_block <N>`         | int    |       64 | Block size n (starting block / S cols) |
| `--out <PATH>`          | string |       "" | Checkpoint directory/prefix |
| `--solutions <N>`       | int    |        5 | Target number of solutions |
| `--sequence_length <N>` | int    | 0 (auto) | Global sequence length override (L). Conflicts with `--s1_sequence_length` |

#### Global Switches

| Flag                  | Description |
|-----------------------|-------------|
| `--enable_hashing`    | Enable result hashing in all stages |
| `--enable_validation` | Validate results against expected hashes (implies `--enable_hashing`) |
| `--enable_oracle`     | Enable strict bit-exact oracle verification (implies hashing + GPU mode) |
| `--graph_enable`      | Enable CUDA graph capture/replay (requires build with `-DENABLE_CUDA_GRAPHS=ON`) |

#### AutoTune Flags

| Flag                     | Description |
|--------------------------|-------------|
| `--autotune_skip_spmm`   | Disable SpMM kernel tuning |
| `--autotune_force_check` | Force pre-flight SpMM correctness check |

#### Stage 1 (Krylov Generation)

| Flag                       | Type | Default  | Description |
|----------------------------|------|----------|-------------|
| `--s1_skip`                | bool |    false | Skip Stage 1 (load from checkpoint) |
| `--s1_sequence_length <N>` | int  | 0 (auto) | Stage 1 sequence length |
| `--s1_gpu_batch_size <N>`  | int  |       64 | Pipeline batch size |
| `--s1_enable_hashing`      | bool |    false | Enable hashing for Stage 1 |
| `--s1_no_save`             | bool |    false | Disable saving checkpoints (X, Y, S) |

#### Stage 2 (Linear Generator)

| Flag                       | Type | Default  | Description |
|----------------------------|------|----------|-------------|
| `--s2_skip`                | bool |    false | Skip Stage 2 (load from checkpoint) |
| `--s2_sequence_length <N>` | int  | 0 (auto) | Input sequence length to consume |
| `--s2_delta <N>`           | int  | 0 (auto) | Explicit degree bound delta |
| `--s2_cpu_mode`            | bool |    false | Force CPU-only basecase (disable GPU default) |
| `--s2_verify_gpu`          | bool |    false | Enable GPU annihilation check |
| `--s2_verify_legacy`       | bool |    false | Enable slow CPU legacy annihilation check |
| `--s2_post_run_legacy`     | bool |    false | Run full legacy solver after main run for comparison |
| `--s2_enable_hashing`      | bool |    false | Enable hashing for Stage 2 |

#### Stage 3 (Reconstruction)

| Flag                       | Type | Default    | Description |
|----------------------------|------|------------|-------------|
| `--s3_skip`                | bool |      false | Skip Stage 3 |
| `--s3_serial_mode`         | bool |      false | Force legacy serial reconstructor (disable batch) |
| `--s3_history_depth <N>`   | int  |         64 | Backtracking history depth |
| `--s3_check_interval <N>`  | int  |         16 | Zero-column check interval |
| `--s3_stripping_limit <N>` | int  |   0 (auto) | Valuation stripping step limit |
| `--s3_enable_hashing`      | bool |      false | Enable hashing for Stage 3 |
| `--s3_no_save`             | bool |      false | Disable saving solutions |

#### Misc

| Flag        | Description |
|-------------|-------------|
| `--verbose` | Enable verbose logging (LOG_DEBUG_2) |
| `--help`    | Show built-in help message |

#### Examples

```bash
# Default run: 10100x10000 matrix, alpha=25, full pipeline
./build/bw_lingen_bench

# Larger matrix with custom density
./build/bw_lingen_bench --rows 50000 --cols 49900 --alpha 30.0

# Skip autotuning, enable oracle verification, save checkpoints
./build/bw_lingen_bench --autotune_skip_spmm --enable_oracle --out /tmp/bw_run1/

# GPU 1, custom block size, explicit sequence length
./build/bw_lingen_bench --device 1 --m_block 128 --n_block 128 --sequence_length 500

# Resume from Stage 2 checkpoint, CPU-only basecase with legacy comparison
./build/bw_lingen_bench --s1_skip --s2_cpu_mode --s2_post_run_legacy --out /tmp/bw_run1/
```

---

### bw_lingen_smoke

Quick smoke test and golden reference generator. Creates a controlled-rank dense matrix and runs the full pipeline with oracle verification enabled. Useful for regression testing and generating reference data for the Python verification script.

**Source:** `benchmarks/bw_lingen_smoke.cu`

| Flag           | Type   | Default | Description |
|----------------|--------|---------|-------------|
| (positional 1) | int    |       0 | CUDA device ID |
| (positional 2) | string |  "test" | Checkpoint prefix |
| `--golden`     | bool   |   false | Golden mode: deterministic 512x512 test (N=512, m=n=64, seed=12345, prefix="golden") |
| `--m <N>`      | int    |      64 | Block size m |
| `--n <N>`      | int    |      64 | Block size n |

In default mode, generates a 4096x4096 matrix with rank 4091 (nullity 5). In golden mode, generates a 512x512 matrix with rank 507.

SpMM autotuning is disabled (to preserve natural matrix ordering for Python verification). Oracle verification and hashing are always enabled.

#### Examples

```bash
# Default smoke test (4096x4096, device 0, prefix "test")
./build/bw_lingen_smoke

# Golden reference run (512x512 deterministic)
./build/bw_lingen_smoke --golden

# Custom device and prefix
./build/bw_lingen_smoke 1 my_run

# Custom block size
./build/bw_lingen_smoke --m 128 --n 128
```

After running, verify with:
```bash
python python/verify_bw_pipeline.py <prefix>
```

---

### bench_matmulgf2

Benchmarks dense binary matrix multiplication (C = A * B over GF(2)) at fixed matrix sizes. Measures both hot-cache and cold-cache throughput across a sweep of matrix dimensions (64, 128, 256, 512). Reports throughput in Gops and matrices/second.

**Source:** `benchmarks/bench_matmulgf2.cu`

| Flag           | Type | Default | Description |
|----------------|------|---------|-------------|
| (positional 1) | int  |       0 | CUDA device ID |

The benchmark runs a fixed sweep with 50 iterations per size:

| Matrix Size | Batch Size |
|-------------|------------|
| 64x64       |    400,000 |
| 128x128     |    100,000 |
| 256x256     |     25,000 |
| 512x512     |      6,000 |

No other parameters are configurable.

#### Examples

```bash
# Run on default device (GPU 0)
./build/bench_matmulgf2

# Run on GPU 1
./build/bench_matmulgf2 1
```

---

### bench_karatsuba

Karatsuba polynomial multiplication benchmark and autotuner. Has two modes: a default mode that runs correctness tests, and an `--autotune` mode that sweeps over (N, threshold, leaf kernel) combinations to find the optimal Stage 2 polynomial multiplication configuration.

**Source:** `benchmarks/bench_karatsuba.cu`

#### Default Mode (Correctness)

Runs correctness validation of naive and Karatsuba multiplication kernels at N=64, 128, 256, 512 with CPU reference comparison. No configurable parameters beyond the device.

```bash
./build/bench_karatsuba
```

#### Autotune Mode

Activated with `--autotune`. Sweeps over polynomial degrees, Karatsuba thresholds, and leaf kernel variants to find the best configuration for a given BW degree target.

| Flag                | Type   | Default | Description |
|---------------------|--------|---------|-------------|
| `--autotune`        | bool   |   false | Enable autotune mode |
| `--device <id>`     | int    |       0 | CUDA device ID |
| `--bw-degree <d>`   | int    |    4096 | Target BW degree (used to weight the scoring) |
| `--deg-min <d>`     | int    |    1024 | Minimum polynomial degree to benchmark |
| `--deg-max <d>`     | int    |   32768 | Maximum polynomial degree to benchmark |
| `--thr-min <t>`     | int    |      32 | Minimum Karatsuba threshold |
| `--thr-max <t>`     | int    |     256 | Maximum Karatsuba threshold |
| `--thr-step <t>`    | int    |      32 | Karatsuba threshold step size |
| `--warmup <n>`      | int    |       3 | Warmup iterations before timing |
| `--iters <n>`       | int    |      10 | Timed iterations per candidate |
| `--cold <0|1>`      | int    |       1 | Measure cold-cache performance |
| `--verbose <0|1|2>` | int    |       1 | Verbosity level (0=quiet, 1=summary, 2=per-candidate detail) |
| `--csv <path>`      | string |  (none) | Append per-candidate benchmark points to CSV file |
| `--help`            | bool   |   false | Show help |

#### Examples

```bash
# Correctness tests only
./build/bench_karatsuba

# Autotune for BW degree 8192, save results to CSV
./build/bench_karatsuba --autotune --bw-degree 8192 --csv results.csv

# Autotune with custom threshold range, no cold-cache measurement
./build/bench_karatsuba --autotune --thr-min 16 --thr-max 512 --thr-step 16 --cold 0

# Autotune on GPU 1 with more iterations for stability
./build/bench_karatsuba --autotune --device 1 --iters 30 --warmup 10
```

---

### bench_lingen_apply_pi

Validation and benchmark for polynomial-matrix-vector multiplication (`PolyMatVec::apply_right` and `apply_left`). Compares GPU kernel results against a CPU reference for correctness, then benchmarks GPU throughput.

**Source:** `benchmarks/bench_lingen_apply_pi.cu`

| Flag           | Type | Default | Description |
|----------------|------|---------|-------------|
| (positional 1) | int  |       0 | CUDA device ID |

Runs a fixed set of test configurations:

| M   | G   | Pi Length | Sequence Length (L) | Iterations |
|-----|-----|-----------|---------------------|------------|
|  64 |  64 |        32 |                4096 |         20 |
| 128 | 128 |        32 |                4096 |         20 |
| 128 |  64 |        32 |                4096 |         20 |

No other parameters are configurable.

#### Examples

```bash
# Run on default device (GPU 0)
./build/bench_lingen_apply_pi

# Run on GPU 1
./build/bench_lingen_apply_pi 1
```

## Typical Workflows

### Quick validation (smoke test)

Verify the full pipeline produces correct kernel vectors:

```bash
./build/bw_lingen_smoke --golden
python python/verify_bw_pipeline.py golden
```

### Full benchmark with default parameters

Run a complete pipeline benchmark with a 10100x10000 matrix:

```bash
./build/bw_lingen_bench --verbose
```

### Benchmark at production scale

Test with a larger matrix closer to production workloads:

```bash
./build/bw_lingen_bench --rows 250000 --cols 249900 --alpha 25.0 --out /tmp/bw_250k/
```

### Stage 2 only (resume from checkpoint)

Skip matrix generation and Stage 1, run only the linear generator:

```bash
./build/bw_lingen_bench --s1_skip --s3_skip --out /tmp/bw_run1/
```

### Verify Stage 2 GPU vs CPU

Run Stage 2 on GPU, then cross-check against the legacy CPU solver:

```bash
./build/bw_lingen_bench --s2_post_run_legacy --s2_verify_gpu --enable_hashing
```

### Kernel-level microbenchmarks

Profile individual operations independent of the full pipeline:

```bash
# GF(2) matrix multiplication throughput
./build/bench_matmulgf2

# Karatsuba polynomial multiplication tuning
./build/bench_karatsuba --autotune --csv karatsuba_results.csv

# Pi application throughput
./build/bench_lingen_apply_pi
```
