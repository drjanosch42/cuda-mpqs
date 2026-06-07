# CUDA-MPQS User Guide

A GPU-accelerated implementation of the Self-Initializing Multiple Polynomial Quadratic Sieve (SIQS/MPQS) for integer factorization. Targets composites from ~60 digits up to 155 digits (RSA-155, 512 bits).

---

## Quick Start

```bash
# Build
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)

# Factor the default ~80-digit composite
./build/tests/cuda-mpqs

# Factor RSA-100 with verbose output
./build/tests/cuda-mpqs --RSA100 --verbose

# Factor a custom number
./build/tests/cuda-mpqs --N 1234567891011121314151617181920212223242526272829
```

---

## Build Instructions

### Prerequisites

- **CUDA Toolkit** (with `nvcc`)
- **CMake** 3.24+
- **C++20** capable host compiler
- **OpenMP** support
- A supported NVIDIA GPU (Turing or newer recommended)

### Build

```bash
cmake -B build -DGPU_TARGET=native
cmake --build build -j$(nproc)
```

The `GPU_TARGET` CMake variable controls which GPU architectures to compile for:

| Value | Meaning |
|-------|---------|
| `native` | Auto-detect the installed GPU (default) |
| `Turing` or `75` | SM 7.5 (RTX 20xx, GTX 16xx) |
| `Ampere` or `80` | SM 8.0 (A100, RTX 30xx) |
| `Ada` or `89` | SM 8.9 (RTX 40xx) |
| `Hopper` or `90` | SM 9.0 (H100) |
| `Blackwell` or `120` | SM 12.0 (B200) |
| `all` | All supported architectures |
| Any SM number | Specific compute capability |

---

## Basic Usage

The main binary is `./build/tests/cuda-mpqs`. With no arguments, it factors a default ~80-digit composite using the full pipeline (tuning, sieving, matrix construction, linear algebra, square root extraction).

The binary returns exit code `0` on successful factorization, `1` on failure.

---

## Command-Line Reference

### Input Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--N <DEC>` | string | ~80-digit composite | Integer to factor, as a decimal string. Parsed into a 512-bit internal representation. |
| `--RSA100` | boolean | — | Sets N to RSA-100 (330 bits, 100 digits). |
| `--RSA110` | boolean | — | Sets N to RSA-110 (364 bits, 110 digits). |
| `--RSA120` | boolean | — | Sets N to RSA-120 (397 bits, 120 digits). |
| `--RSA129` | boolean | — | Sets N to RSA-129 (426 bits, 129 digits). |
| `--RSA130` | boolean | — | Sets N to RSA-130 (430 bits, 130 digits). |
| `--RSA140` | boolean | — | Sets N to RSA-140 (463 bits, 140 digits). |
| `--RSA150` | boolean | — | Sets N to RSA-150 (496 bits, 150 digits). |
| `--RSA155` | boolean | — | Sets N to RSA-155 (512 bits, 155 digits). |

**Notes:**
- `--N` takes precedence: if `--N` is given first, subsequent `--RSAxx` flags print a warning and are ignored.
- If `--RSAxx` is given first, a subsequent `--N` overwrites it silently.
- If no input is specified, the default ~80-digit composite is used:
  `6024065079889642469495026789749787328504528247460180000248150504804066095061017`

### Device and Working Directory

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--device <ID>` | uint32 | `0` | CUDA GPU device index. Pinned (tracked as user-set). |
| `--dir <PATH>` | string | `./mpqs_work` | Working directory for relation files, autotune history, and other I/O. Created automatically when `--disk_io` is enabled. |
| `--disk_io` | boolean | `false` | Enable writing relations to disk after sieving. Relations are saved to `<work_dir>/relations.soa` in a custom binary SoA format. |

### Execution Modes

The pipeline supports seven execution modes. Only one can be active at a time; the last mode flag on the command line wins.

| Flag | Mode | Stages Run | Description |
|------|------|-----------|-------------|
| `--full` | `FULL_PIPELINE` | Tuning, [Autotune], Sieve, Matrix, LinAlg, Sqrt | Default. Runs the complete factorization pipeline. |
| `--sieve_only` | `SIEVE_ONLY` | Tuning, Sieve | Sieve for relations and write them to disk, then exit. Always saves relations to `<work_dir>/relations.soa` regardless of `--disk_io`. |
| `--linalg_only` | `LINALG_ONLY` | Tuning, Matrix, LinAlg, Sqrt | Load previously saved relations from `<work_dir>/relations.soa`, construct matrix, solve, and extract factors. Requires a prior `--sieve_only` run. |
| `--matrix_only` | `MATRIX_ONLY` | Load v2 relations → Matrix → BW → Sqrt | Load v2-format relations and run matrix preprocessing, Block Wiedemann, and square root without sieving. Useful for replaying stored relation sets through different matrix configurations. Always uses the CPU matrix backend. |
| `--sqrt_only` | `SQRT_ONLY` | Sqrt | **BROKEN — use `--linalg_only` instead.** There is no on-disk kernel-vector loader, so this mode cannot reconstruct the linear-algebra state it needs. |
| `--param_test` | `PARAM_TEST` | Tuning, Sieve (grid search) | Run an exhaustive parameter grid search over sieve kernel configurations. Reports the best timing found. Does not produce factorizations. |
| `--autotune_only` | `AUTOTUNE_ONLY` | Tuning, Autotune | Run parameter optimization only, print results, and exit. Writes optimized parameters to the autotune history file but does not proceed to sieving. |
| `--estimate_only` | (special) | Truncated sieve probe | Run a short sieve probe to estimate total runtime, then exit. Prints projected sieve, matrix, linear algebra, and total times. In solo mode, handled before orchestrator creation. In cluster mode, runs the full cluster sieve with a time limit (`--probe_timeout`) and prints a cluster-aware estimate. |

**Disk I/O Interaction:**
- `--sieve_only` always writes `relations.soa` to the work directory.
- `--disk_io` with `--full` writes relations after sieving but continues to matrix/linalg/sqrt.
- `--linalg_only` reads `relations.soa` from the work directory.
- Both `--sieve_only` and `--linalg_only` run the tuning stage to regenerate the factor base (needed for matrix construction).

### Tuning Parameters

These parameters control the core SIQS algorithm. When set to 0 (the default), they are auto-calculated based on the bit-length of N.

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--fb_bound <N>` | uint32 | Auto | Factor base bound F. All primes p <= F with Legendre symbol (N/p) = 1 form the factor base. Larger F means more primes to sieve with (slower per polynomial but more likely to find smooth values). | Yes |
| `--sieve_bound <N>` | uint32 | Auto (262144) | Sieve interval half-width M. Each polynomial Q(x) is evaluated for x in [-M, M]. Larger M explores more candidates per polynomial but increases memory and reduces smoothness probability. | Yes |
| `--lp1_bound <N>` | uint64 | `0` (disabled) | Large prime bound. When > 0, enables the single large prime variant: partially smooth values with one cofactor prime <= `lp1_bound` are kept and matched pairwise via a GPU hash table. Dramatically increases relation yield for larger inputs. Typical values: 100M-500M. | Yes |
| `--target_rels <N>` | uint32 | Auto (FB_size + 128) | Target number of relations to collect before proceeding to matrix construction. When unset, defaults to factor-base size + 128 to give the Block Wiedemann solver sufficient overdetermination. | Yes |
| `--dedup_safety_factor <F>` | double | `1.05` (auto `1.35` for <80d) | Dedup oversample margin: the sieve collects `target * factor` relations to absorb duplicate loss. Values outside `[1.0, 2.0]` trigger a warning but are still applied. | Yes |
| `--lp_interval <N>` | uint32 | `0` (auto) | LP processing frequency. When 0, the pipeline uses adaptive scheduling (processes LP matches when the partial buffer fills sufficiently). When N > 0, forces LP processing every N sieve batches. Useful for tuning the trade-off between LP matching latency and overhead. | Yes |
| `--params <p1,...,p8>` | 8x uint32 | Auto | Manually specify the 8-parameter sieve kernel configuration tuple. Parentheses around the list are optional and stripped. See [Parameter Tuple Format](#parameter-tuple-format) below. | Yes |

#### Auto-Calculated Factor Base Bound (F)

When `--fb_bound` is not specified (or set to 0), F is chosen by bit-length of N:

| Bits of N | F |
|-----------|---|
| < 180 | 60,000 |
| 180-219 | 350,000 |
| 220-234 | 500,000 |
| 235-249 | 900,000 |
| 250-279 | 1,500,000 |
| 280-339 (RSA-100) | 3,000,000 |
| 340-369 (RSA-110) | 4,000,000 |
| 370-399 (RSA-120) | 5,000,000 |
| 400-439 (RSA-130) | 7,500,000 |
| >= 440 (RSA-140+) | 10,000,000 |

The sieve interval M defaults to 262,144 (2^18) for all input sizes when not overridden.

#### Parameter Tuple Format

The `--params` flag accepts 8 comma-separated uint32 values controlling the GPU sieve kernel launch configuration:

| Index | Name | Description |
|-------|------|-------------|
| 0 | `subCubeSize` | Number of polynomials per sieve call |
| 1 | `numIntervals` | Number of sieving blocks per sieve call |
| 2 | `polyBlockSize` | Polynomial generator block size |
| 3 | `blocksPerCycle` | Active blocks per cycle in metaSieve |
| 4 | `metaGridDim` | MetaSieve kernel grid dimension (blocks) |
| 5 | `metaBlockDim` | MetaSieve kernel block dimension (threads) |
| 6 | `sasGridDim` | SieveAndScan kernel grid dimension (blocks) |
| 7 | `sasBlockDim` | SieveAndScan kernel block dimension (threads, max 1024 for legacy) |

Example: `--params 32,16,128,4,64,256,32,512`

**Safety adjustments:** When LP is active (`--lp1_bound > 0`) and custom params are set, `sasGridDim` (index 6) is automatically raised to a minimum floor derived from `subCubeSize * numIntervals / 64` (rounded to next power of 2) to prevent candidate buffer overflow. The legacy kernel `sasBlockDim` (index 7) is capped at 1024 to match `__launch_bounds__`.

### Buffer Sizing

All buffer size flags accept an optional `K` or `M` suffix for base-1024 scaling (e.g., `512K` = 524288, `4M` = 4194304).

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--accum_buf_size <SIZE>` | uint64 | `max(4096, sieve_batch_size * 2048)` | Accumulation buffer capacity. Sieved candidates land here before being merged into the persistent store. 80% fill triggers a purge cycle. Auto-sized to roughly 2x the worst observed per-batch candidate peak. | Yes |
| `--partial_buf_size <SIZE>` | uint64 | `= accum_buf_size` | Partial (LP staging) buffer. Holds 1-partial relations awaiting LP matching. Only used when `--lp1_bound > 0`. Defaults to 1x the accumulation buffer size; user values below the accumulation buffer size are floor-clamped up to it (with a warning). | Yes |
| `--persistent_buf_size <SIZE>` | uint64 | `target * 2 + accum` | Persistent relation store capacity. Must be >= target_relations or the pipeline will fail. Holds all confirmed full relations on the GPU. | Yes |
| `--lp1_combined_buf <SIZE>` | uint64 | `32K` (32768) | LP match output buffer. Holds the combined full relations produced by LP hash table matching in each LP processing round. | Yes |
| `--lp1_max_witnesses <SIZE>` | uint64 | `1M` (1048576) | LP witness hash table capacity (number of unique large prime slots). Rounded up to the next power of 2. Minimum: 16. The hash table uses open addressing with slab-based collision resolution. | Yes |
| `--lp1_hash_bits <N>` | uint32 | Auto | Number of bits for the LP hash table directory. Default: `log2(witness_capacity) - 4`. Recommended range: 10-28. Controls the number of hash buckets (2^hash_bits). | Yes |

**Small-N down-scaling:** When `--accum_buf_size` is left at its default and `target_relations < 16384`, the accumulation buffer is shrunk to `max(4096, target_relations * 4)` rounded up to the next power of 2. This prevents a single sieve batch from overshooting a small target by hundreds of thousands of relations.

**Pin tracking:** All buffer size flags are tracked as "pinned" (user-set). When the autotune system recommends different buffer sizes, pinned values are preserved. Only unpinned (auto-calculated) values are overwritten by autotune recommendations.

### Estimation Options

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--probe_timeout <SEC>` | double | `120.0` | Hard timeout in seconds for sieve probes (`--estimate_only`). In solo mode, limits the `TruncatedSieveRun` duration. In cluster mode, limits the cluster sieve before STOP is broadcast to all workers. Increase for slow GPUs (e.g., Jetson at RSA-120 scale where a single batch takes >100s). | No |

### Sieve Stage Options

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--sieve_batch_size <N>` | uint32 | `0` (legacy) | Number of `a`-values per sieve batch. When 0, uses the legacy single-step sieve loop. When > 0, enables the double-buffered zero-sync batch pipeline that overlaps sieve and postprocessing kernels. | Yes |
| `--cuda_graph_unroll <N>` | uint32 | `0` (disabled) | Capture N sieve batches as a CUDA graph and replay it, reducing per-batch launch overhead. Must be even (odd values are rounded up with a warning); capped at 16. Recommended: 2 or 4 for production runs. | Yes |
| `--sieve_max_relations <N>` | uint64 | `0` (disabled) | Stop the sieve after collecting N relations. Accepts a K/M/B/T suffix (base-1000). 0 disables the cap. | Yes |
| `--sieve_max_batches <N>` | uint64 | `0` (disabled) | Stop the sieve after N batch iterations. 0 disables the cap. | Yes |
| `--sieve_truncate_continue` | boolean | `false` | When a sieve-truncation cap (`--sieve_max_relations`/`--sieve_max_batches`) fires, continue the pipeline (matrix, BW, sqrt) with the relations collected so far instead of stopping. | No |
| `--sieve_gms_blocks <N>` | uint32 | Auto (64) | Number of CUDA blocks for the globalMetaSieve kernel. | Yes |
| `--sieve_hc_dim <N>` | uint32 | Auto | Hypercube dimension for polynomial construction (number of prime factors in the `a` coefficient). Currently hidden from `--help` output but functional. | Yes |

**Batch vs. Legacy sieving:**
- **Legacy** (`--sieve_batch_size 0`, default): Single-step loop. The CPU drives each sieve step sequentially. Simpler, well-tested. LP processing happens every 10 steps.
- **Batch** (`--sieve_batch_size N`): Double-buffered GPU pipeline. Sieving and postprocessing overlap via CUDA events. Higher throughput on modern GPUs. LP processing is periodic (adaptive interval).

### Matrix Options

These flags control the matrix construction and preprocessing stage, which converts smooth relations into a sparse GF(2) matrix for the Block Wiedemann solver.

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--matrix_mode <MODE>` | string | auto | Matrix construction mode: `legacy` (projected FB+2 columns) or `preprocess` (expanded FB+2+LP columns with merge-based preprocessing). AUTO resolves to `legacy` for normal runs; `preprocess` engages only via explicit `--matrix_mode preprocess` or `--matrix_only`. The former LP-fraction auto-switch was removed because preprocessing degrades the obstructed high-LP regime. | No |
| `--matrix_backend <BACKEND>` | string | `cpu` | Preprocessing backend: `cpu`, `gpu`, or `auto`. In solo mode with `gpu` or `auto` (if GPU available and >10K rows), activates the V2 packed GPU preprocessing pipeline with batch GPU merges. Cluster, `MATRIX_ONLY`, and `LINALG_ONLY` modes always use CPU. | No |
| `--char_mode <MODE>` | string | `none` | Quadratic character-column symbol: `none` (no character columns — default), `branch` (correct branch-fixed field-element character with a fixed Tonelli root), or `norm` (legacy genus-blind NORM symbol). Never auto-enabled — only this flag sets it. Character columns confer no measured factoring benefit at reachable scales; `branch` is the mathematically correct symbol, kept as a tool. | No |
| `--truncation_factor <F>` | double | `1.05` | Post-GF(2) row truncation enable flag: `> 0` enables truncation, `0` disables it. The actual kept-row target is excess-based (see `--matrix_truncation_excess`), not a direct multiple of this value. | No |
| `--matrix_truncation_excess <N>` | uint32 | `200` | Number of excess rows to keep above `(n_cols + n_extra_cols)` when truncation is enabled. Controls overdetermination of the truncated matrix. | No |
| `--compact_cycles <N>` | uint32 | `5` | Maximum compact-merge cycles for the GPU preprocessing backend. `0` runs a single merge pass with no compaction. | No |
| `--matrix_gf2_floor_factor <F>` | double | `0.5` | M12-S2 GF(2) column-diversity floor: stop compact-merge cycles when GF(2)-alive columns fall below `factor * initial_gf2_cols`. Range `[0.0, 1.0]`. | No |
| `--matrix_gf2_min_floor <N>` | uint32 | `8192` | M12-S2 absolute minimum GF(2) column floor; compact-merge stops if alive GF(2) columns drop below this value. | No |
| `--partial_subsample <F>` | double | `1.0` | Fraction of partial / LP-combined relations to retain (for `--matrix_only` experiments). Range `[0.0, 1.0]`. | No |
| `--smooth_subsample <F>` | double | `1.0` | Fraction of pure smooth relations to retain (LP-combined relations are always kept) for `--matrix_only` experiments. Range `[0.0, 1.0]`. | No |
| `--lp_preprocess_threshold <F>` | double | `0.55` | **Deprecated / inert.** Formerly the LP-fraction threshold for auto-selecting `preprocess` mode; AUTO no longer auto-selects preprocess from LP fraction (use `--matrix_mode preprocess` to opt in). Retained as a no-op for backwards compatibility. (`--lp_matrix_threshold` is a deprecated alias.) | No |

**GPU preprocessing pipeline (V2, `--matrix_backend gpu`):**

When GPU backend is active in solo mode, the matrix stage uses the packed GPU preprocessing pipeline (M9v2):

1. **Packed CSR construction** -- builds the expanded matrix directly on device from GPU-resident relation data. Each entry stores a 24-bit column index and 8-bit exponent in a single `uint32_t`, retaining full exponent information while using the same 4 bytes/entry as binary CSR.
2. **GPU singleton removal** -- iteratively removes rows containing unique columns (singletons) entirely on device.
3. **GPU truncation** (optional) -- selects the sparsest rows to control matrix density.
4. **GPU batch merges** -- CPU plans merge candidates (weight-2 and Markowitz higher-weight), GPU executes non-conflicting merges in parallel. Each merge performs a packed factor-list merge (exponent addition) and Montgomery `sqrt_Q` multiplication, maintaining per-row 1-partial metadata throughout.
5. **GF(2) extraction** -- filters odd-exponent entries to produce the binary CSR for the BW solver.
6. **Product character columns** -- Jacobi-only evaluation on pre-merged `sqrt_Q` values (no Montgomery recomputation needed).

The V2 pipeline eliminates the merge tree entirely. The sqrt stage consumes merged 1-partials directly, reading pre-computed `sqrt_Q` products and summed exponents without tree expansion.

The CPU backend (`--matrix_backend cpu`) uses the V1 pipeline: binary CSR with GF(2) XOR merges, merge tree for sqrt reconstruction, and separate product character column computation. This is the only path available in cluster mode and replay modes (`--matrix_only`, `--linalg_only`).

### Linear Algebra Options

| Flag | Type | Default | Description | Pinned |
|------|------|---------|-------------|--------|
| `--bw_m <N>` | uint32 | `256` | Block Wiedemann block width m. Controls the blocking factor for the Krylov sequence generation. | Yes |
| `--bw_n <N>` | uint32 | `256` | Block Wiedemann block width n. Controls the blocking factor for reconstruction. | Yes |

### Square Root Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sqrt_legacy` | boolean | `false` | Use the CPU-based sequential square root path instead of the GPU batched path. The CPU path processes solution vectors one at a time; the GPU path processes all solution vectors simultaneously using batched Montgomery arithmetic. Primarily for debugging and benchmarking. |
| `--sqrt_diagnostic` | boolean | `false` | Log extra square-root diagnostics: HalveExponents validity checks and solution-vector diversity. Useful for diagnosing trivial-factor or non-squarefree relation issues. |

### Autotune Options

The autotune system performs a multi-stage parameter optimization before sieving. It uses truncated sieve probes and runtime estimation to find optimal (F, M, L, kernel params) without running a full factorization.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--autotune` | boolean | `false` | Enable autotune before sieving in `FULL_PIPELINE` mode. After optimization, the pipeline continues with the tuned parameters. |
| `--autotune_only` | boolean | — | Run autotune only, print results, and exit. Implies `--autotune`. Sets mode to `AUTOTUNE_ONLY`. |
| `--autotune_stage0` | boolean | (all enabled) | Enable Stage 0 (projection from history). When any `--autotune_stageN` flag is given, only the explicitly specified stages run. |
| `--autotune_stage1` | boolean | (all enabled) | Enable Stage 1 (kernel parameter micro-benchmarking). |
| `--autotune_stage2` | boolean | (all enabled) | Enable Stage 2 (runtime estimation via truncated sieve probes). |
| `--autotune_stage3` | boolean | (all enabled) | Enable Stage 3 (coordinate descent over F/M/L). |
| `--autotune_max_iter <N>` | uint32 | `2` | Maximum autotune iterations (Stage 2-3 repeat loop). |
| `--autotune_timeout <sec>` | double | `300.0` | Wall-clock timeout for the entire autotune process (seconds). |
| `--autotune_history <P>` | string | `<work_dir>/autotune_history.json` | Path to the GPU-specific autotune history file. Used for projection (Stage 0) and saving results. |
| `--autotune_benign_history <P>` | string | `<work_dir>/benign_history.json` | Path to the benign (cross-GPU) history file. Contains parameter data that transfers across different GPUs. |
| `--autotune_no_history` | boolean | `false` | Disable both loading and saving of parameter history. Also disables the auto-apply feature (zero-probe history-based parameter selection). |
| `--autotune_candidates <P>` | string | — | Path to a candidates file for bootstrap mode. One decimal number per line; blank lines and `#` comments are skipped. |
| `--autotune_bootstrap` | boolean | `false` | Bootstrap mode: read candidates from `--autotune_candidates`, factor each one using the full pipeline (sorted smallest to largest by bit-length), and accumulate history. Requires `--autotune_candidates`. |

**Autotune Stages:**

| Stage | Name | What It Does |
|-------|------|-------------|
| 0 | Projection | Loads history file, finds nearest-neighbor entries by bit-length, and projects initial (F, M, L) parameters. Zero GPU cost. |
| 1 | Kernel Params | Micro-benchmarks the 8-parameter kernel tuple via the sieve grid search (`runParamTest`). Skipped if F/M haven't changed since last run. |
| 2 | Runtime Estimation | Runs a truncated sieve probe (default 12% of target) to measure throughput and estimate total runtime. |
| 3 | Sieve Params | Coordinate descent optimization over F, M, and LP bound using runtime estimates from Stage 2. |

**`--autotune` vs `--autotune_only`:**
- `--autotune` enables autotune as a pre-sieve optimization step within `FULL_PIPELINE`. After autotune completes, the pipeline continues with the optimized parameters through sieving, matrix, linalg, and sqrt.
- `--autotune_only` sets the execution mode to `AUTOTUNE_ONLY`. After autotune completes, results are printed and the program exits without sieving.

**Auto-Apply (implicit history-based parameter selection):**
Even without `--autotune`, when history files exist, the pipeline automatically loads the best known parameters for the current N and GPU ("auto-apply"). This is a zero-probe optimization that runs during tuning. Disable with `--autotune_no_history`.

### Output Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--verbose` | boolean | `false` | Enable verbose logging. Sets the console log severity to `LOG_STATS` (0): shows statistics summaries, buffer telemetry, and throughput. Equivalent to `--log_level 0`. |
| `--debug` | boolean | `false` | Enable debug logging. Sets the console log severity to `LOG_DEBUG_1` (1): per-batch telemetry, downloads, and internal state. More output than `--verbose`. Equivalent to `--log_level 1`. |
| `--log_level <N>` | int | `-1` (`LOG_INFO`) | Set the console verbosity threshold directly. Accepts any value in `[-4 .. 3]` (`LOG_RESULT` to `LOG_DEBUG_3`); out-of-range values are rejected. |
| `--mute` | boolean | `false` | Suppress all console output except the final factorization result (`LOG_RESULT`, -4). Also strips all log-line prefixes. |
| `--log_file <PATH>` | string | — | Add a file sink capturing the full log (down to `LOG_DEBUG_3`) with full prefixes. |
| `--error_log <PATH>` | string | — | Add a file sink capturing only warnings and critical errors (`LOG_WARNING` through `LOG_ERROR_CRITICAL`). |
| `--log_csv` | boolean | `false` | Emit the `--log_file` sink output in CSV format. |
| `--log_date` | boolean | `false` | Include the date in console log timestamps. |
| `--log_no_time` | boolean | `false` | Suppress console timestamps entirely. |
| `--log_no_stage` | boolean | `false` | Suppress the `[Stage N: Name]` prefix on console lines. |
| `--log_show_rank` | boolean | `false` | Show the `[Rank N]` prefix on console lines (multi-process / cluster). |
| `--log_wrap <N>` | int | `120` | Console line-wrap width. `0` disables wrapping. |
| `--version` | boolean | — | Print the version string (`cuda-mpqs X.Y.Z (lingen X.Y.Z, git <sha>, built <date>)`) and exit. |
| `--help` | boolean | — | Print usage information and exit. |

**Logging Levels (from most to least urgent):**

| Level | Value | Description |
|-------|-------|-------------|
| `LOG_RESULT` | -4 | Factorization result only (shown under `--mute`) |
| `LOG_ERROR_CRITICAL` (alias `LOG_ERROR`) | -3 | Fatal errors, unrecoverable CUDA failures |
| `LOG_WARNING` (alias `LOG_ERROR_MAJOR`) | -2 | Warnings, non-fatal errors, degraded operation |
| `LOG_INFO` | -1 | Stage transitions, key milestones (default console threshold) |
| `LOG_STATS` | 0 | Statistics summaries, buffer telemetry, throughput (`--verbose`) |
| `LOG_DEBUG_1` | 1 | Per-batch telemetry, downloads, internal state (`--debug`) |
| `LOG_DEBUG_2` | 2 | Per-kernel details, allocation internals |
| `LOG_DEBUG_3` | 3 | Developer trace (no active call sites; retained for forward compatibility) |

A console message is shown when its urgency is at least as urgent as the threshold (i.e. its numeric level is `<=` the configured `--log_level`). `LOG_ERROR_MAJOR` is now a backward-compatible alias for `LOG_WARNING` (-2), not a distinct level.

### Debug Options (Compile-Time)

These flags are only available when the binary is compiled with `SIEVING_DEBUG_FLAG` defined. They are used for debugging the sieve kernel internals.

| Flag | Type | Description |
|------|------|-------------|
| `--metaSnapshot <k>` | uint32 | Snapshot metaSieve bucket state at step k. |
| `--metaP <idx>` | uint32 | Filter metaSnapshot to prime index idx. |
| `--metaO <o>` | uint32 | Filter metaSnapshot to offset o. |
| `--sasSnapshot <k>` | uint32 | Snapshot candidate relations at step k. |

---

## Examples

### Factor a specific number

```bash
./build/tests/cuda-mpqs --N 15347774734723794379428745798237912838912730918239
```

### Factor RSA-100 with verbose output

```bash
./build/tests/cuda-mpqs --RSA100 --verbose
```

### Factor RSA-100 with tuned parameters

```bash
./build/tests/cuda-mpqs --RSA100 \
    --fb_bound 9000000 \
    --sieve_bound 524288 \
    --lp1_bound 100000000 \
    --verbose
```

### Run autotune optimization

```bash
# Autotune and factor in one run
./build/tests/cuda-mpqs --RSA100 --autotune --verbose

# Autotune only (print results, exit)
./build/tests/cuda-mpqs --RSA100 --autotune_only --verbose

# Autotune with specific stages only
./build/tests/cuda-mpqs --RSA100 --autotune_stage0 --autotune_stage2 --verbose
```

### Get a runtime estimate without factoring

```bash
# Solo estimate
./build/tests/cuda-mpqs --RSA100 --estimate_only --verbose

# Cluster estimate (2-node, 30s probe)
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode coordinator --listen_port 9300 --expected_workers 1 \
    --fb_bound 7000000 --sieve_bound 262144 --lp1_bound 2000000000000 \
    --estimate_only --probe_timeout 30 --verbose

# Longer probe for slow GPUs (Jetson)
./build/tests/cuda-mpqs --RSA120 --estimate_only --probe_timeout 600 --verbose
```

### Run with large prime support

```bash
# Enable LP with bound 100M and custom witness table
./build/tests/cuda-mpqs --RSA100 \
    --lp1_bound 100000000 \
    --lp1_max_witnesses 2M \
    --verbose
```

### Export and reimport relations (split pipeline)

```bash
# Step 1: Sieve and save relations
./build/tests/cuda-mpqs --RSA100 --sieve_only --lp1_bound 100000000 --verbose

# Step 2: Load relations and solve
./build/tests/cuda-mpqs --RSA100 --linalg_only --verbose
```

Note: both steps must use the same `--dir` (defaults to `./mpqs_work`) and the same N/fb_bound so the factor base is regenerated identically.

### Use batch sieving mode

```bash
./build/tests/cuda-mpqs --RSA100 \
    --sieve_batch_size 128 \
    --lp1_bound 100000000 \
    --verbose
```

### Bootstrap autotune history from a set of composites

```bash
# Create a candidates file (one number per line)
cat > candidates.txt << 'EOF'
# 60-digit composites
123456789012345678901234567890123456789012345678901234567891
# 70-digit composites
...
EOF

./build/tests/cuda-mpqs \
    --autotune_bootstrap \
    --autotune_candidates candidates.txt \
    --verbose
```

### Use a specific GPU

```bash
./build/tests/cuda-mpqs --RSA100 --device 1 --verbose
```

### Factor RSA-110 with GPU preprocessing

```bash
./build/tests/cuda-mpqs --RSA110 \
    --fb_bound 9000000 \
    --lp1_bound 1000000000000 \
    --sieve_batch_size 32 \
    --lp_interval 1 \
    --matrix_backend gpu \
    --verbose
```

### Compare CPU vs GPU square root performance

```bash
# GPU batched (default)
./build/tests/cuda-mpqs --RSA100 --verbose

# CPU sequential (for comparison/debugging)
./build/tests/cuda-mpqs --RSA100 --sqrt_legacy --verbose
```

---

## Cluster Mode (Distributed Sieve)

Multiple GPU nodes can cooperatively sieve smooth relations over a LAN. The coordinator collects relations from all nodes and runs the remaining pipeline stages (matrix, linear algebra, square root) locally. Solo-mode performance is unaffected -- cluster code paths have zero overhead when `--cluster_mode` is not set.

### 2-Node Example

```bash
# Terminal 1 -- Coordinator (start first)
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode coordinator --listen_port 9300 --expected_workers 1 \
    --fb_bound 7000000 --sieve_bound 262144 --lp1_bound 2000000000000 \
    --sieve_batch_size 32 --cuda_graph_unroll 4 --lp_interval 1 \
    --lp1_hash_bits 21 --verbose

# Terminal 2 -- Worker (start 2-10s after coordinator)
./build/tests/cuda-mpqs --RSA100 \
    --cluster_mode worker --coordinator_host <COORDINATOR_IP> --coordinator_port 9300 \
    --sieve_batch_size 2 --verbose
```

### Cluster Flags

| Flag | Description |
|------|-------------|
| `--cluster_mode` | `coordinator` or `worker` (omit for solo) |
| `--listen_port` | Coordinator TCP port (default 9100) |
| `--expected_workers` | Number of workers to wait for |
| `--coordinator_host` | Worker: coordinator IP or hostname |
| `--coordinator_port` | Worker: coordinator TCP port (default 9100) |
| `--cluster_init_timeout` | Init window in seconds: worker retries + coordinator accept (default 300) |
| `--cluster_node_weights` | Comma-separated per-node throughput weights (overrides SM*clock auto-weighting) |
| `--cluster_headroom` | Per-node headroom percent (default 10) |

The transport backend is fixed to TCP. It is exposed only as the config field `MPQSConfig::transport` (default `"tcp"`); there is **no `--transport` CLI flag**.

### Setup Requirements

- **Build on all nodes** -- each node needs a compiled binary for its GPU architecture.
- **Sync code via git** -- push from development node, pull on all others. Do not use rsync.
- **SSH key-based access** -- workers are typically launched via `ssh ${USER}@<hostname>`.
- **Same LAN** -- coordinator IP must be routable from all workers.

### Launch Scripts

Pre-configured launch scripts are available in `tools/cluster/`:

| Script | Description |
|--------|-------------|
| `rtx_cluster_default_launch.sh` | 2-node RTX, default ~80d validation |
| `rtx_cluster_launch.sh` | 2-node RTX, RSA-100 |
| `jetson_cluster_launch.sh` | 2-node Jetson, RSA-100 |
| `rtx_4node_rsa100_launch.sh` | 4-node heterogeneous, RSA-100 |
| `rtx_4node_rsa110_launch.sh` | 4-node heterogeneous, RSA-110 |
| `jetson_rsa110_overnight_launch.sh` | 2-node Jetson overnight RSA-110 |

For detailed configuration, troubleshooting, and architecture, see [CLUSTER.md](CLUSTER.md).

---

## Performance Tuning Tips

1. **Always enable large primes for inputs > 80 digits.** The single large prime variant dramatically increases relation yield. A good starting bound is `10 * fb_bound` to `100 * fb_bound`.

2. **Use `--autotune` for serious runs.** The autotune system finds significantly better parameters than the built-in heuristics, especially for inputs near the boundary between parameter tiers.

3. **Use `--estimate_only` to preview runtime** before committing to a full factorization. This runs a short sieve probe and extrapolates. In cluster mode, it runs the full cluster topology with a time limit (`--probe_timeout`, default 120s) and estimates total runtime including all nodes. Increase `--probe_timeout` for slow GPUs (e.g., `--probe_timeout 600` on Jetson).

4. **Watch buffer fill warnings.** The pipeline logs warnings when buffers approach capacity. If you see `witness_near_full` or overflow warnings, increase `--lp1_max_witnesses` or `--accum_buf_size`.

5. **Batch sieving** (`--sieve_batch_size N`) can improve throughput by overlapping GPU sieving and postprocessing. Try values of 64-256.

6. **Pin important parameters.** When using `--autotune`, explicitly set parameters you know are good (e.g., `--fb_bound`, `--lp1_bound`) and let autotune optimize the rest. Pinned parameters are preserved by autotune.

7. **Save your work.** Use `--disk_io` or `--sieve_only` to persist relations. The sieve stage is typically the longest; saving relations lets you re-run matrix/linalg/sqrt without re-sieving.

8. **Use GPU matrix preprocessing for solo runs.** With `--matrix_backend gpu`, the packed GPU preprocessing pipeline (M9v2) accelerates matrix construction by running singleton removal, merge execution, and character column computation entirely on the GPU. This is most beneficial at RSA-110+ scale where CPU merge time dominates. The GPU backend is only available in solo mode with device-resident relation data.

---

## Troubleshooting

### "CUDA Error: invalid configuration argument"
The sieve kernel launch parameters are infeasible for your GPU. This can happen with `--params` values that exceed the GPU's thread/block limits. Use `--autotune` to find valid parameters, or remove `--params` to use defaults.

### "Insufficient relations"
The sieve collected fewer relations than the factor base size. Possible causes:
- Sieve ran with LP disabled for a large input. Enable `--lp1_bound`.
- `--target_rels` was set too low.
- The `--persistent_buf_size` was smaller than `--target_rels` (pipeline cannot store enough relations).

### "Failed to load SoA relations"
`--linalg_only` requires a prior `--sieve_only` run with the same `--dir` path. Ensure `<work_dir>/relations.soa` exists.

### Pipeline hangs during sieve stage
Known issue: Stage 3 (sieve) can occasionally hang on certain inputs around 90 digits. If the progress indicator stalls for an extended period, terminate and retry with different parameters or `--autotune`.

### Out of GPU memory
Large factor bases (F > 5M) with large witness tables (> 4M) can exhaust GPU memory. Reduce `--lp1_max_witnesses`, `--accum_buf_size`, or `--persistent_buf_size`. Consider reducing `--fb_bound` if possible.

### Autotune produces worse parameters
This can happen when the truncated probe is too short to be representative. Try:
- `--autotune_max_iter 5` (more optimization iterations)
- `--autotune_timeout 600` (more time budget)
- Using `--autotune_stage0 --autotune_stage2 --autotune_stage3` to skip kernel benchmarking if parameters are already good.
