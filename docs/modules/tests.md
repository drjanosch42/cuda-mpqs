# Tests (`tests/`)

Main driver binary and auxiliary test utilities for the MPQS pipeline.

## Files

| File | Lines | Build Target | Purpose |
|------|-------|-------------|---------|
| `cuda-mpqs.cpp` | ~1450 | `cuda-mpqs` | Primary driver: CLI parsing, orchestrator invocation, factor verification, bootstrap mode |
| `tools/sqrt_failure/relation_validator.cu` | -- | `relation_validator` | Standalone host-side relation + large-prime validator (see below) |
| `sqrt_benchmark.cpp` | 142 | `bench_sqrt` (`EXCLUDE_FROM_ALL`) | **Stale**: uses old AoS `Relation` API; kept for reference but not functional with current SoA pipeline |
| `sieving_benchmark.cpp` | 324 | *(not in build)* | **Stale**: development-only isolated sieve + postprocessing via low-level API; no build target |
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
| `--dump_matrix` | -- | Diagnostic: dump finalized matrix (CSR binary + column legend) to work dir |
| `--dump_kernel_vectors` | -- | Diagnostic: enable BW solution writer + dump original-relation-space kernel vectors |
| `--dump_combine_provenance` | -- | Diagnostic: serialize LP-combine constituents to `combine_provenance.bin` (opt-in; no effect on the factorization path when unset) |

### Tuning and Sieving

| Flag | Argument | Description |
|------|----------|-------------|
| `--fb_bound` | `<n>` | Factor base bound F (0 = auto) |
| `--sieve_bound` | `<n>` | Sieve interval half-width M (0 = auto). **Silently power-of-two snapped** (ties round down); a non-pow2 request emits a `[CLI] Warning: ... snapped to M=` line on `std::cerr` (`tests/cuda-mpqs.cpp:718-733` — `std::cerr`, not the logger, because HPCLogger is not yet initialised during parsing) |
| `--sieve_hc_dim` | `<n>` | SIQS hypercube dimension (0 = auto) |
| `--lp1_bound` | `<n>` | Large prime bound (0 = disabled) |
| `--lp1_max_witnesses` | `<SIZE>` | Max LP witness capacity; accepts K/M suffix, snaps to power of 2 (default: 0 = auto-derived) |
| `--target_rels` | `<n>` | Target relation count (0 = auto) |
| `--dedup_safety_factor` | `<F>` | Dedup oversample margin (default: 1.05; auto 1.35 for <80d). Warns outside [1.0, 2.0] |
| `--sieve_batch_size` | `<n>` | Batch GPU sieving (0 = legacy host-driven) |
| `--cuda_graph_unroll` | `<n>` | Capture N batches as a CUDA graph for replay (0 = disabled, default 0). What each batch contributes depends on `--cuda_graph_capture`. Only odd values **> 1** are rounded up — `1` is a supported, unrounded probe value; capped at 16 |
| `--cuda_graph_capture` | `<sieve\|postproc\|full>` | CUDA-graph capture scope (default: `full` solo / `postproc` cluster). `sieve` = sieve kernels only (pre-v1.0.6 body, rollback path); `postproc` = + batch trial division; `full` = + GPU LP (solo only; downgraded to `postproc` with a warning in cluster). Inert at `--cuda_graph_unroll 0`. ⚠ `MPQS_LP_DIAG=1` forces the achieved scope down to `sieve` regardless of the request (`resolveGraphCaptureScope`, `include/graph_capture_scope.h:78-81`) — the LP diagnostic block issues `cudaMalloc`/synchronous `cudaMemcpy`, which return CUDA 900/906 inside a stream capture |
| `--cuda_graph_lp_stride` | `<k>` | In-graph LP cadence: one dispatch every K-th captured batch (default 1 = per batch, as `--cuda_graph_unroll 0` does; 0 or ≥ unroll = one per replay, the pre-v1.0.6 cadence). Only meaningful when LP is captured |
| `--lp_interval` | `<n>` | LP processing frequency (default: 1; 0 = auto/adaptive, N>0 = every N batches) |
| `--sieve_gms_blocks` | `<n>` | MetaSieve CUDA blocks (0 = auto) |
| `--sieve_meta_cycle_cap` | `<n>` | Cap `num_activeBlocksPerCycle` in the meta-sieve (ablation knob; 0 = off) |
| `--sieve_gather_block_dim` | `<n>` | GATHER-kernel blockDim occupancy override; power of two in [32, 1024]; 0 = off (loader default) |
| `--sieve_block_size` | `<n>` | **v1.0.6, NARROW BATCH + `--params` only.** Overrides `gs_conf.sievingBlockSize` (otherwise derived from the smem budget). 0 = off; N > 0 must be a power of two ≥ 256 (`admissibleSieveBlockSize`, `sieve_memory_model.h`). ⚠ Whenever `SB` changes, `numIntervals` (`--params` field 2) must change with it so `numIntervals × SB ≥ 2M` — enforced pre-sieve by `narrowBatchCoverageOk`. Rejected loudly on legacy (`--sieve_batch_size 0`), on the wide path, without `--params`, and with any `--autotune*` |
| `--sieve_big_prime_start` | `<n>` | **v1.0.6, NARROW BATCH + `--params` only.** Overrides `gs_conf.bigPrimeStartIndex` (the GATHER/SCATTER factor-base split; otherwise `SB/32`). 0 = off; N > 0 must be **> 32** (`= midPrimeStartIndex`; at N ≤ 32 the mid-prime range inverts and the band is silently dropped). Power-of-two **not** required. Same four rejections as above |
| `--sieve_bucket_overflow_stats` | -- | **v1.0.6**, default off. Also emit the `[BucketOverflow]` stats line on the **narrow** path (wide always emits it) — previously `getBucketOverflowStats()` returned `false` for every narrow run. Costs one DtoH copy + sync on the siever stream per ~5 s stats tick; needs `--verbose` |
| `--bucket_size_factor` | `<F>` | Wide-path bucket sizing: `globalBucketSize = F·SB` (F=0.5 == legacy SB/2; 0 = off/legacy). Rejects F < 0 |
| `--sieve_accumulator` | `<auto\|u8\|u16>` | Sieve log-accumulator width (default auto: dispatch predicate decides; u8/u16 force narrow/wide) |
| `--wide_accum` | `<auto\|u8sat\|u16>` | Wide-regime accumulator variant (default auto: saturating-uint8 iff the exactness gate holds, else uint16) |
| `--autotune_probe_polys` | `<n>` | Wide-autotune survivors/sec probe sample size (# distinct staged polynomials; 0 = auto-scale by N) |
| `--probe_timeout` | `<sec>` | Hard timeout for sieve probes (default: 120.0) |
| `--sieve_max_relations` | `<n>` | Stop sieve after N relations; K/M/B/T suffix (0 = disabled) |
| `--sieve_max_batches` | `<n>` | Stop sieve after N batch iterations (0 = disabled) |
| `--sieve_truncate_continue` | -- | Continue pipeline (matrix/BW/sqrt) after truncation |
| `--params` | `<p1,...,p8>` | Custom 8-element sieve parameter tuple. **v1.0.6:** the narrow *batch* path accepts **non-power-of-two** `{np, metaGridDim, sasGridDim, npbptb}`, so the SCATTER/GATHER grids can be pinned to the device's SM count; legacy and wide keep mandatory pow2 and reject such tuples loudly. Certified by the `admissible_geometry` CTest |

Seven numeric-bound flags parse **decimal** suffixes via `parse_suffixed_uint64` (K = 10³, M = 10⁶,
B = 10⁹, T = 10¹²) — `--fb_bound`, `--sieve_bound`, `--lp1_bound`, `--lp1_max_witnesses`,
`--sieve_max_relations`, `--matrix_lp1_bound`, `--matrix_max_rows`. Only `--sieve_bound`
(pow2-snap) and `--lp1_max_witnesses` (pow2-snap) pass `snap_pow2 = true`. This is distinct from the
base-1024 `parse_size` used by the buffer-size flags below.

### Sieve Checkpointing (default-off)

| Flag | Argument | Description |
|------|----------|-------------|
| `--checkpoint_interval` | `<sec>` | Wall-seconds between mid-sieve checkpoints (0 = disabled) |
| `--checkpoint_batches` | `<n>` | Alternative interval in sieve batches (fires first if both set; 0 = disabled) |
| `--checkpoint_dir` | `<path>` | Checkpoint directory (default `""` → `<work_dir>/checkpoint`) |
| `--resume` | -- | Load `sieve.ckpt` from `checkpoint_dir` and continue; warns and starts fresh if absent |

### Buffer Sizing

All buffer size flags accept K/M suffixes (base-1024, e.g. `512K`, `4M`).

| Flag | Default | Description |
|------|---------|-------------|
| `--accum_buf_size` | `max(4096, batch_size·2048)` | Accumulation buffer capacity |
| `--partial_buf_size` | `= accum` (1×) | Partial (LP staging) buffer; only when `--lp1_bound > 0` |
| `--persistent_buf_size` | `target·2 + accum` | Persistent relation store |
| `--lp1_combined_buf` | 0 = auto (32768) | LP match output buffer |
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
| `--sqrt_diagnostic` | Log solution-diversity statistics (distinct BW solutions by hash) at `LOG_INFO`. The per-solution nontrivial-GCD rate (`LOG_DEBUG_1`) and HalveExponents validity (`LOG_WARNING`) are logged unconditionally regardless of this flag — pair with `--debug --log_file` to capture them |
| `--estimate_only` | Run truncated sieve probe + print runtime estimate, then exit |

`--autotune_only` (sets `AUTOTUNE_ONLY` mode) and `--autotune_bootstrap` (bootstrap mode) are documented in the Autotune section.

### Matrix Construction

| Flag | Argument | Description |
|------|----------|-------------|
| `--matrix_mode` | `<legacy\|preprocess>` | Matrix construction mode (default: auto, which **resolves to legacy** for normal runs; preprocess only via explicit flag or `--matrix_only`) |
| `--char_mode` | `<norm\|branch\|none>` | Character-column symbol (default: none). `none` = zero char cols; `norm` = legacy NORM symbol; `branch` = branch-fixed field-element symbol. See [matrix.md](matrix.md) |
| `--matrix_backend` | `<cpu\|gpu\|auto>` | Preprocessing backend (default: cpu; auto → gpu if available and >10K rows) |
| `--lp_preprocess_threshold` | `<F>` | **Deprecated / inert**: the LP-fraction auto-preprocess switch was removed (default still parsed: 0.55, no effect) |
| `--lp_matrix_threshold` | `<F>` | **Deprecated** alias for `--lp_preprocess_threshold` (also inert; propagates only if the latter is unset) |
| `--truncation_factor` | `<F>` | Post-GF(2) row truncation enable flag (>0 enabled, 0 disabled; default: 1.05). Actual target is excess-based — see `--matrix_truncation_excess` |
| `--matrix_truncation_excess` | `<N>` | Excess rows above `(n_cols + n_extra_cols)` (default: 200) |
| `--compact_cycles` | `<N>` | Max compact-merge cycles, GPU backend (default: 5; 0 = single pass / no compaction) |
| `--matrix_gf2_floor_factor` | `<F>` | Stop compact-merge when GF(2) cols fall below `factor × initial_gf2_cols` ([0.0,1.0], default: 0.5) |
| `--matrix_gf2_min_floor` | `<N>` | Absolute minimum GF(2) column floor (default: 8192) |
| `--partial_subsample` | `<F>` | Subsample partials/LP-combined for `--matrix_only` experiments ([0.0,1.0], default: 1.0) |
| `--smooth_subsample` | `<F>` | Subsample pure smooths (LP-combined always kept) for `--matrix_only` experiments ([0.0,1.0], default: 1.0) |
| `--matrix_max_rows` | `<N>` | Cap the legacy relation batch at the first `N` rows before matrix construction (suffix-drop, K/M/B/T suffix; default 0 = off). Keeps the padded square dimension `max(rows, cols) ≤ 2^24−1` so TiledCOO-256 stays admissible in the SpMM autotuner. ⚠ It binds on **rows** as well as columns — check both dimensions |
| `--matrix_lp1_bound` | `<L>` | `--matrix_only` LP-magnitude down-filter (K/M/B/T suffix): drop LP-combined relations with large prime > L after `.v2` load; pure smooths never dropped (default: 0 = inert) |
| `--truncation_min_rows` | `<N>` | Skip CPU-preprocess truncation when the reduced matrix has ≤ N rows (default: 5,000,000) |
| `--preprocess_lp_materialize_max` | `<F>` | Max combined-smooth LP fraction above which preprocess skips materializing raw-1-partial 2-cycle rows (default: 0.45) |
| `--merge_max_weight` | `<K>` | Diagnostic: CPU-preprocess `mergeHigherWeight` k_max (default: 10; 2 disables weight≥3 merges) |
| `--force_preprocess` | -- | Diagnostic: force the expand+merge path even with 0 raw partials |

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
| `--cluster_pool_oversize` | `<F>` | a-value pool over-provisioning multiplier (default: 1.0) |

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
| `--bw_m` | 256 | Block width m (`include/orchestrator.h:292`; the submodule's own `BWSolverConfig` default is 64). Adaptively downscaled to 64 (matrix dim < 4,000) or 128 (dim < 16,000) **only when neither `--bw_m` nor `--bw_n` is pinned** |
| `--bw_n` | 256 | Block width n (`include/orchestrator.h:293`); same adaptive rule |
| `--bw_max_solutions <N>` | `-1` (ALL) | Stop BW Stage-3 reconstruction after `N` solutions; wires the submodule's `stage3_max_solutions`. ⚠ A solution-**COUNT** stop, **not** a batch cap — an under-delivering config still *enters* a later reconstruction batch and can burn hours there for zero yield |
| `--bw_checkpoint_dir <path>` | `""` (off) | Save the Krylov S-sequence / lingen Π / final solutions at stage boundaries under `<path>/bw*`. **Save only** — it never loads without `--bw_resume` |
| `--bw_resume` | off | Load previously-**completed**-stage artifacts from `--bw_checkpoint_dir` and skip those stages. **Stage boundaries only:** resumes from the last completed stage, never mid-stage, so a crash inside Stage 3 restarts Stage 3 from scratch. **NOT validated end-to-end** — only the *save* side has ever run in production |

Parsed at `tests/cuda-mpqs.cpp:926-945`, wired at `src/orchestrator/orchestrator.cpp:6894-6920`;
config fields `include/orchestrator.h:292-296`. See [linalg.md](linalg.md) for the block-width
measurements (128 is a 2.687× LA regression, 512 OOMs in Stage 3) and the checkpoint semantics.

### Logging

Console severity uses signed levels (`src/common/logger/hpc_logger.h:20-28`): `-4` = `LOG_RESULT`,
`-3` = `LOG_ERROR_CRITICAL` / `LOG_ERROR`, `-2` = `LOG_WARNING` (`LOG_ERROR_MAJOR` is an alias),
`-1` = `LOG_INFO` (default), `0` = `LOG_STATS`, `1` = `LOG_DEBUG_1`, `2` = `LOG_DEBUG_2`,
`3` = `LOG_DEBUG_3`.

| Flag | Argument | Effect |
|------|----------|--------|
| `--verbose` | -- | Set console level to `LOG_STATS` (0): statistics + buffer telemetry |
| `--debug` | -- | Set console level to `LOG_DEBUG_1` (1): per-batch debug telemetry |
| `--log_level` | `<N>` | Set console verbosity threshold directly. The parser accepts the **full** `[LOG_RESULT .. LOG_DEBUG_3]` = `-4`..`3` range (`tests/cuda-mpqs.cpp:1030`); the built-in `--help` text (`:340`) only advertises `-4`..`1`. ⚠ `--log_level 2` (`LOG_DEBUG_2`) is required to see the BW Stage-3 Horner per-degree progress line — `--debug` (level 1) does **not** show it |
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

## Standalone Validation Tool: `relation_validator`

`tools/sqrt_failure/relation_validator.cu` builds the `relation_validator` executable: a host-side,
CPU-only exhaustive correctness checker for a saved relations file (`.v2` or `.soa`, loaded via
`detect_and_deserialize`; also accepts a `sieve.ckpt` checkpoint file or a checkpoint directory,
and takes an optional `--out <summary.json>`). It independently re-derives every relation from scratch — it does not
trust what the GPU sieve/postprocessing recorded — and, crucially, runs a **deterministic primality
test on every recorded large prime**. Per relation (smooths and partials) it checks:

1. **Algebraic identity** — recompute `Q = sqrt_Q² − N`, confirm the recorded sign, and confirm `|Q| == 2^v2 · Π fb[idx]^count · large_prime`.
2. **Completeness / missed-factor** — independently trial-divide `|Q|` by the entire factor base and confirm the re-derived factorization matches; flags any FB prime dividing the recorded large prime (the "composite masquerading as a large prime" pathology).
3. **Large-prime primality** (partials) — deterministic Miller–Rabin (BPSW fallback for very large values); flags every composite large prime.
4. **Range** — confirm `max(factor_base) < large_prime ≤ lp_bound`.

Compiled as CUDA (for the `__host__ __device__` math headers) but launches no kernels; parallelised
with OpenMP. Not registered as a CTest target (it takes a relations-file argument).

## CTest Targets (16)

**Sixteen** regression tests are registered with CTest (`add_test` in `tests/CMakeLists.txt`;
counted directly from that file, 2026-08-25 — the previous count of 13 predated
`admissible_geometry`, `sieve_geometry_overrides` and `graph_capture_scope` and is **superseded**).
All are compiled as CUDA (for the `__host__ __device__` math headers) and are deterministic. All but
two launch no kernels and need no CUDA device; the exceptions are `packed_char_device_parity`
(drives the real M9v2 packed GPU pipeline) and `oom_guard` (queries device properties) — both of
which **skip cleanly with exit 0** when no CUDA device is present.

**Branch-fixed character columns, Stages 1–6** (7 tests):

| Test (`add_test` name) | Stage | Certifies |
|------|-------|-----------|
| `nt_primitives_u64` | 1 | `Tonelli_Shanks_u64` / `jacobi_u64` / `is_prime_u64` (sieve) + `uint512::mod_uint64` against a 128-bit oracle and the uint32 implementations |
| `aux_prime_selection` | 2 | `selectAuxPrimes()` under `NORM` (byte-identical legacy walk) and `BRANCH` (q > lp1_bound, fixed Tonelli root) |
| `branch_char_bit` | 3 | `branchCharBit` genus-correctness vs. a Python reference, the F2 homomorphism, and host==device parity |
| `char_bits_persist` | 4 | Birth-capture formula parity, `relation_io` v2 char-bit round-trip (and char-less back-compat), cluster wire round-trip |
| `char_xor_propagation` | 5 | CPU XOR propagation: LP-combination XOR, merge-tree reduction XOR, per-relation adapter, append-after-reduction structure |
| `packed_char_gather` / `packed_char_device_parity` | 6 | M9v2 packed propagation parity (host mirror + real GPU pipeline) vs. the CPU merge-tree oracle |

The Stage-3 fixture (`branch_char_fixture.h`) is **pre-generated and committed** under
`tools/sqrt_failure/`; the build consumes the checked-in header directly. To refresh it, re-run
`gen_branch_char_fixture.py` in the development tree (it imports a genus-correct Python reference
that lives there only) and commit the result. *(Corrected 2026-08-25: the earlier claim that it is
"regenerated at build time" is **superseded** — `tests/CMakeLists.txt:81-86` says the opposite.)*

**Autotune / wide-path guards** (2 tests, sources under `tools/autotune/`):

| Test | Certifies |
|------|-----------|
| `oom_guard` | Autotune OOM-guard decision logic: `sieve_memory_model.h` estimator + 0.80 budget + `KernelLaunchValidator::fitsTotalFootprint` + seed-reduction loop. Asserts no clamps at pinned M=131072 and that a synthetic over-budget config fires the guard. Skips cleanly without a CUDA device |
| `wide_num_polys_clamp` | `clampWideNumPolys()` in the custom apply path: an autotune/pinned winner can never re-inflate `num_polysPerSieveCall` past 512 on the wide path; byte-for-byte no-op on narrow |

**Sieve launch-geometry guards** (2 tests, added in v1.0.6):

| Test | Source | Certifies |
|------|--------|-----------|
| `admissible_geometry` | `tools/autotune/test_admissible_geometry.cu` | **v1.0.6** SM-aligned launch geometry: the per-parameter power-of-two policy, the exact-partition / GATHER-decomposition invariants (G1/G2/G3) and the admissible-set rounding of `sasGridDim`. Runs the **real** `KernelLaunchValidator` through its explicit-`DeviceLimits` constructor, so it is deterministic and needs no CUDA device. Asserts (A) the legacy/wide rule set is **bit-identical to the pre-v1.0.6 predicate** over a pow2 sweep, (B) the SM-aligned tuples are **accepted on the narrow batch path and rejected in legacy/wide mode**, (C) every deliberately-broken tuple is rejected in **both** modes with a diagnostic naming the real invariant, (D) `admissibleSasGridDim` reproduces next-pow2-then-clamp on pow2 `np` |
| `sieve_geometry_overrides` | `tools/sieve/test_sieve_geometry_overrides.cu` | **v1.0.6** narrow-batch geometry overrides (`--sieve_block_size` / `--sieve_big_prime_start`) and the narrow-batch interval-coverage invariant they make reachable. Runs the **real** production helpers from `src/sieve/sieve_memory_model.h` (`narrowBatchCoverageOk`, `admissibleSieveBlockSize`, `admissibleBigPrimeStart`) — the same functions `validateConfigs()` and the CLI call, so the test cannot drift from the shipped rule. Pure host arithmetic; no device required. Asserts (A) the coverage guard is a **no-op on every shipped narrow-batch geometry**, (B) the half-sieve case (`intervals=8, SB=32768, M=262144`) is **rejected** and its `numIntervals=16` fix accepted, (C) over-coverage (`SB == M`, the autotune M-sweep) stays legal, (D) the admissible sets accept the intended values and reject 0-adjacent / non-pow2 / ≤ `midPrimeStartIndex` inputs, (E) the coverage arithmetic is **64-bit and cannot wrap into a false pass** |

**Checkpoint / cluster resume / wire format** (4 tests, sources under `tools/sqrt_failure/`):

| Test | Certifies |
|------|-----------|
| `checkpoint_io` | `deserialize_v2` trailing-bytes tolerance; `writeCheckpointAtomic`/`readCheckpoint` round-trip; `.prev` retention, torn-footer rejection, `loadLatestCheckpoint` fallback |
| `work_pool_cursor` | `WorkPool::completedPrefixCursor()` returns the completed contiguous prefix (min over in-flight ∪ returned), not `nextCursor()` |
| `cluster_resume` | `computeResumeTrim` per-node initial-range trim (+ re-sieve-last-hypercube guard), `clusterResumeTopologyOk` N2 topology guard, re-inject ordering (`addRelations` rebuilds dedup before partial combines) |
| `work_assign_hash` | v2 `WORK_ASSIGN` wire format: FB-hash round-trip with fb_size-independent payload (the 64 MiB frame-cap fix), `generateFactorBase(N, F)` regen-equivalence, mismatch detection (fail-loud worker path) |

**Graph-capture scope** (1 test, v1.0.6, source `tools/sqrt_failure/test_graph_capture_scope.cu`):

| Test | Certifies |
|------|-----------|
| `graph_capture_scope` | Resolution of `--cuda_graph_capture`. Pure-logic host test over plain scalars — the resolver is a free inline function in `include/graph_capture_scope.h`, so it links `mpqs_common` only (no orchestrator/cluster/sieve pull-in). Enforces the operator constraint that **cluster mode NEVER captures GPU large-prime matching** |

## Built-in Test Numbers

| Name | Bits | Digits | Approx. Time (RTX 5070 Ti) |
|------|------|--------|-----------------------------|
| Default | 262 | 79 | ~25 s |
| RSA-100 | 330 | 100 | ~4 min |
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
    CUDA_SEPARABLE_COMPILATION ON)          # tests/CMakeLists.txt:9-11
target_link_libraries(cuda-mpqs PRIVATE
    mpqs_orchestrator cudampqs_cluster mpqs_autotune
    mpqs_sieve mpqs_postproc mpqs_matrix
    mpqs_sqrt lingen_ops cudampqs_build_flags)
```

⚠ The `cuda-mpqs` target does **not** set `CUDA_RESOLVE_DEVICE_SYMBOLS` explicitly — it inherits it
from `set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)` at the repository root (`CMakeLists.txt:7`), which
initialises the property on every target created afterwards. Every other target in
`tests/CMakeLists.txt` *does* set it explicitly.

`bench_sqrt` is built with `EXCLUDE_FROM_ALL` (not part of the default build). It links only `mpqs_sqrt`, `mpqs_common`, and `cudampqs_build_flags`. It accepts up to three positional arguments: `<bit_size>` (default 256), `<fb_size>` (default 1000), `<num_relations>` (default 1100).

`sieving_benchmark.cpp` has no `add_executable` entry in `CMakeLists.txt` and is not part of any build target. It is a development-only file that drives the low-level `DeviceSievingController` + `DevicePostProcessingController` API directly, bypassing the orchestrator.

## Test Data Files

| File | Description |
|------|-------------|
| `candidates.txt` | Semiprime test composites (two prime factors) for standard benchmarking |
| `non-rsa-candidates.txt` | Large multi-factor composites: 3/4 coprime factors, prime powers, mixed p^2×q×r forms (60–90 digits; `tools/generate_test_composites_large.py`, seed=2026) |
| `multi_factor_candidates.txt` | Small multi-factor composites for M10 BCD (coprime refinement) testing (`tools/generate_test_composites.py`, seed=42) |

## Factor Verification

After `FULL_PIPELINE` (or `SQRT_ONLY`) runs, `verify_factors()` multiplies all returned factors using `uint512::mult()` and checks the product equals N. Returns exit code 0 on success, 1 on failure or if no factors are found. Other modes (`SIEVE_ONLY`, `LINALG_ONLY`, `MATRIX_ONLY`, `PARAM_TEST`, `AUTOTUNE_ONLY`) report partial-pipeline success without a factor check.

`SQRT_ONLY` is **broken** (no kernel-vector loader exists) — use `--linalg_only` or `--full` for a verified end-to-end run. The cluster `--estimate_only` path returns immediately after `Run()` with no factor check.
