# Tests (`tests/`)

Main driver binary and auxiliary test utilities for the MPQS pipeline.

## Files

| File | Lines | Build Target | Purpose |
|------|-------|-------------|---------|
| `cuda-mpqs.cpp` | ~700 | `cuda-mpqs` | Primary driver: CLI parsing, orchestrator invocation, factor verification, bootstrap mode |
| `sqrt_benchmark.cpp` | 135 | `bench_sqrt` (`EXCLUDE_FROM_ALL`) | **Stale**: uses old AoS `Relation` API; kept for reference but not functional with current SoA pipeline |
| `sieving_benchmark.cpp` | 317 | *(not in build)* | **Stale**: development-only isolated sieve + postprocessing via low-level API; no build target |
| `mpqs_analyzer.html` | -- | -- | Browser-based analysis/visualization artifact |

## CLI Reference (cuda-mpqs)

### Problem Definition

| Flag | Argument | Description |
|------|----------|-------------|
| `--N` | `<decimal>` | Custom integer to factor |
| `--RSA100` .. `--RSA155` | -- | Preset RSA challenge numbers (100–155 digits) |
| `--device` | `<id>` | GPU device (default: 0) |
| `--dir` | `<path>` | Work directory (default: `./mpqs_work`) |
| `--disk_io` | -- | Enable disk serialization between stages |

### Tuning and Sieving

| Flag | Argument | Description |
|------|----------|-------------|
| `--fb_bound` | `<n>` | Factor base bound F (0 = auto) |
| `--sieve_bound` | `<n>` | Sieve interval half-width M (0 = auto) |
| `--lp1_bound` | `<n>` | Large prime bound (0 = disabled) |
| `--lp1_max_witnesses` | `<SIZE>` | Max LP witness capacity; accepts K/M suffix, snaps to power of 2 (default: 1M) |
| `--target_rels` | `<n>` | Target relation count (0 = auto) |
| `--dedup_safety_factor` | `<F>` | Dedup oversample margin (default: 1.05; auto 1.35 for <80d). Warns outside [1.0, 2.0] |
| `--sieve_batch_size` | `<n>` | Batch GPU sieving (0 = legacy host-driven) |
| `--cuda_graph_unroll` | `<n>` | Capture N sieve batches as a CUDA graph for replay (0 = disabled, default 0). Must be even (rounded up if odd); capped at 16 |
| `--lp_interval` | `<n>` | LP processing frequency (0 = auto/adaptive, N>0 = every N batches) |
| `--sieve_gms_blocks` | `<n>` | MetaSieve CUDA blocks (0 = auto) |
| `--probe_timeout` | `<sec>` | Hard timeout for sieve probes (default: 120.0) |
| `--sieve_max_relations` | `<n>` | Stop sieve after N relations; K/M/B/T suffix (0 = disabled) |
| `--sieve_max_batches` | `<n>` | Stop sieve after N batch iterations (0 = disabled) |
| `--sieve_truncate_continue` | -- | Continue pipeline (matrix/BW/sqrt) after truncation |
| `--params` | `<p1,...,p8>` | Custom 8-element sieve parameter tuple |

`--sieve_hc_dim <n>` (hypercube dimension) is parsed but intentionally omitted from the help text.

### Buffer Sizing

All buffer size flags accept K/M suffixes (base-1024, e.g. `512K`, `4M`).

| Flag | Default | Description |
|------|---------|-------------|
| `--accum_buf_size` | `max(4096, batch_size·2048)` | Accumulation buffer capacity |
| `--partial_buf_size` | `= accum` (1×) | Partial (LP staging) buffer; only when `--lp1_bound > 0` |
| `--persistent_buf_size` | `target·2 + accum` | Persistent relation store |
| `--lp1_combined_buf` | 32K | LP match output buffer |
| `--lp1_hash_bits` | auto | LP hash table directory bits |

### Execution Modes

| Flag | Mode |
|------|------|
| `--full` | Full pipeline (default) |
| `--sieve_only` | Sieve + write relations to disk |
| `--linalg_only` | Load relations + matrix + solve + sqrt |
| `--matrix_only` | Load v2 relations → matrix preprocessing + BW + sqrt (no sieve) |
| `--sqrt_only` | Load kernel vectors + sqrt (**BROKEN** — use `--linalg_only` instead) |
| `--param_test` | Parameter exploration (exits after sieve) |
| `--sqrt_legacy` | Force CPU sqrt path (debug/benchmark; default: GPU batched) |
| `--sqrt_diagnostic` | Log extra sqrt diagnostics (HalveExponents validity, solution diversity) |
| `--estimate_only` | Run truncated sieve probe + print runtime estimate, then exit |

`--autotune_only` (sets `AUTOTUNE_ONLY` mode) and `--autotune_bootstrap` (bootstrap mode) are documented in the Autotune section.

### Matrix Construction

| Flag | Argument | Description |
|------|----------|-------------|
| `--matrix_mode` | `<legacy\|preprocess>` | Matrix construction mode (default: auto) |
| `--matrix_backend` | `<cpu\|gpu\|auto>` | Preprocessing backend (default: cpu; auto → gpu if available and >10K rows) |
| `--lp_preprocess_threshold` | `<F>` | LP fraction for auto preprocess-mode detection (default: 0.55) |
| `--lp_matrix_threshold` | `<F>` | **Deprecated** alias for `--lp_preprocess_threshold` (propagates only if the latter is unset) |
| `--truncation_factor` | `<F>` | Post-GF(2) row truncation enable flag (>0 enabled, 0 disabled; default: 1.05). Actual target is excess-based — see `--matrix_truncation_excess` |
| `--matrix_truncation_excess` | `<N>` | Excess rows above `(n_cols + n_extra_cols)` (default: 200) |
| `--compact_cycles` | `<N>` | Max compact-merge cycles, GPU backend (default: 5; 0 = single pass / no compaction) |
| `--matrix_gf2_floor_factor` | `<F>` | M12-S2: stop compact-merge when GF(2) cols fall below `factor × initial_gf2_cols` ([0.0,1.0], default: 0.5) |
| `--matrix_gf2_min_floor` | `<N>` | M12-S2: absolute minimum GF(2) column floor (default: 8192) |
| `--partial_subsample` | `<F>` | Subsample partials/LP-combined for `--matrix_only` experiments ([0.0,1.0], default: 1.0) |
| `--smooth_subsample` | `<F>` | Subsample pure smooths (LP-combined always kept) for `--matrix_only` experiments ([0.0,1.0], default: 1.0) |

`--matrix_only` is listed in the Execution Modes table above.

### Cluster Mode

Flag names and defaults mirror [`cluster.md`](cluster.md) and the [cluster usage guide](../../CLUSTER.md).

| Flag | Argument | Description |
|------|----------|-------------|
| `--cluster_mode` | `<solo\|coordinator\|worker>` | Cluster role (default: solo) |
| `--coordinator_host` | `<host>` | Coordinator hostname/IP (worker mode) |
| `--coordinator_port` | `<port>` | Coordinator TCP port (default: 9100) |
| `--listen_port` | `<port>` | Listen port (coordinator mode, default: 9100) |
| `--expected_workers` | `<n>` | Number of workers to accept (coordinator, default: 0) |
| `--cluster_init_timeout` | `<sec>` | Init window: worker retries + coordinator accept (default: 300) |
| `--cluster_node_weights` | `<w1,...>` | Comma-separated per-node throughput weights (overrides SM×clock) |
| `--cluster_headroom` | `<pct>` | Per-node headroom percent (default: 10) |

The `MPQSConfig::transport` field ("tcp", default) exists in `orchestrator.h` but is **not** wired to a CLI flag — there is no `--transport` parser entry; TCP is the only transport.

### Autotune

| Flag | Argument | Description |
|------|----------|-------------|
| `--autotune` | -- | Enable autotune before sieving (in FULL_PIPELINE mode) |
| `--autotune_only` | -- | Run autotune only, print results, exit (sets `AUTOTUNE_ONLY` mode) |
| `--autotune_stage{0-3}` | -- | Enable specific autotune stage (overrides defaults; can combine multiple) |
| `--autotune_max_iter` | `<n>` | Max autotune iterations (default: 2) |
| `--autotune_timeout` | `<sec>` | Autotune wall-clock timeout in seconds (default: 300) |
| `--autotune_history` | `<path>` | History file path (default: `<work_dir>/autotune_history.json`) |
| `--autotune_benign_history` | `<path>` | Benign history file path (default: `<work_dir>/benign_history.json`) |
| `--autotune_no_history` | -- | Disable parameter history load/save |
| `--autotune_candidates` | `<path>` | Candidates file for bootstrap mode (one decimal per line) |
| `--autotune_bootstrap` | -- | Run bootstrap: factor all candidates from file, build history |

When explicit `--autotune_stage{0-3}` flags are provided, only those stages run (otherwise all 4 stages are enabled by default).

### Block Wiedemann

| Flag | Default | Description |
|------|---------|-------------|
| `--bw_m` | 256 | Block width m |
| `--bw_n` | 256 | Block width n |

### Logging

Console severity uses signed levels: `-4` = result, `-1` = info, `0` = stats, `1` = debug (range `[LOG_RESULT .. LOG_DEBUG_3]`).

| Flag | Argument | Effect |
|------|----------|--------|
| `--verbose` | -- | Set console level to `LOG_STATS` (0): statistics + buffer telemetry |
| `--debug` | -- | Set console level to `LOG_DEBUG_1` (1): per-batch debug telemetry |
| `--log_level` | `<N>` | Set console verbosity threshold directly (`-4`..`1`; default info) |
| `--mute` | -- | Suppress all console output except the factorization result |
| `--log_file` | `<path>` | Write full log (`DEBUG_3`) to file |
| `--error_log` | `<path>` | Write warnings/errors only to a file sink |
| `--log_csv` | -- | Emit `--log_file` output in CSV format |
| `--log_date` | -- | Include date in log timestamps |
| `--log_no_time` | -- | Suppress timestamps entirely |
| `--log_no_stage` | -- | Suppress `[Stage N: Name]` prefix |
| `--log_show_rank` | -- | Show `[Rank N]` prefix (multi-process/cluster) |
| `--log_wrap` | `<N>` | Console line-wrap width (default: 120; 0 = disabled) |
| `--version` | -- | Print version (`cuda-mpqs X.Y.Z (lingen …, git …, built …)`) and exit |
| `--help` | -- | Print usage and exit |

### Debug (SIEVING_DEBUG_FLAG only)

| Flag | Description |
|------|-------------|
| `--metaSnapshot <k>` | Snapshot metaSieve buckets at sieve step k |
| `--metaP <idx>` | Filter metaSnapshot to prime index |
| `--metaO <o>` | Filter metaSnapshot to offset |
| `--sasSnapshot <k>` | Snapshot candidate relations at step k |

## Special Execution Paths

### Estimate-Only Mode (`--estimate_only`)

Runs a truncated sieve probe via `autotune::estimateRuntime()` and prints a breakdown of estimated sieve, matrix, and linear algebra times, plus throughput and LP fill projection. Exits with code 0 on success.

### Bootstrap Mode (`--autotune_bootstrap`)

Requires `--autotune_candidates <file>`. Loads candidate composites (one decimal per line, `#` comments and blank lines skipped), sorts by bit-length ascending, and factors each with `FULL_PIPELINE`. Autotune history is saved after each successful factorization, building a history database for future autotune lookups.

## Built-in Test Numbers

| Name | Bits | Digits | Approx. Time (RTX 5070 Ti) |
|------|------|--------|-----------------------------|
| Default | ~280 | ~80 | ~25 s |
| RSA-100 | 330 | 101 | ~4 min |
| RSA-110 | 364 | 111 | -- |
| RSA-120 | 397 | 120 | -- |
| RSA-129 | 426 | 129 | -- |
| RSA-130 | 430 | 130 | -- |
| RSA-140 | 463 | 140 | -- |
| RSA-150 | 496 | 150 | -- |
| RSA-155 | 512 | 155 | -- |

When no `--N` or `--RSA*` flag is given, the default ~80-digit composite is used.

## Build

```cmake
add_executable(cuda-mpqs cuda-mpqs.cpp)
set_target_properties(cuda-mpqs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(cuda-mpqs PRIVATE
    mpqs_orchestrator mpqs_sieve mpqs_postproc mpqs_matrix
    mpqs_sqrt lingen_ops cudampqs_build_flags)
```

`bench_sqrt` is built with `EXCLUDE_FROM_ALL` (not part of the default build). It links only `mpqs_sqrt`, `mpqs_common`, and `cudampqs_build_flags`. It accepts up to three positional arguments: `<bit_size>` (default 256), `<fb_size>` (default 1000), `<num_relations>` (default 1100).

`sieving_benchmark.cpp` has no `add_executable` entry in `CMakeLists.txt` and is not part of any build target. It is a development-only file that drives the low-level `DeviceSievingController` + `DevicePostProcessingController` API directly, bypassing the orchestrator.

## Test Data Files

| File | Description |
|------|-------------|
| `candidates.txt` | Semiprime test composites (two prime factors) for standard benchmarking |
| `non-rsa-candidates.txt` | Multi-factor composites: 3/4 coprime factors, prime powers, mixed p^2×q×r forms (60–90 digits) |

`multi_factor_candidates.txt` was removed — it was byte-identical to `non-rsa-candidates.txt`.

## Factor Verification

After `FULL_PIPELINE` (or `SQRT_ONLY`) runs, `verify_factors()` multiplies all returned factors using `uint512::mult()` and checks the product equals N. Returns exit code 0 on success, 1 on failure or if no factors are found. Other modes (`SIEVE_ONLY`, `LINALG_ONLY`, `MATRIX_ONLY`, `PARAM_TEST`, `AUTOTUNE_ONLY`) report partial-pipeline success without a factor check.

`SQRT_ONLY` is **broken** (no kernel-vector loader exists) — use `--linalg_only` or `--full` for a verified end-to-end run. The cluster `--estimate_only` path returns immediately after `Run()` with no factor check.
