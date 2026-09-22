# Orchestrator Module (`src/orchestrator/`, `include/`)

Central pipeline driver. Coordinates the 5-stage MPQS factorization pipeline (plus the optional autotune stage), manages configuration, execution modes, cluster topology (solo/coordinator/worker), sieve checkpointing, and inter-stage data flow.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `include/orchestrator.h` | ~840 | `MPQSOrchestrator` class, `MPQSConfig` struct, `ExecutionMode` enum, `SieveProgressTracker`, `LPFillProjector`, `TruncatedSieveResult` |
| `src/orchestrator/orchestrator.cpp` | ~8430 | Pipeline implementation, stage dispatch, truncated sieve probes, coordinator `networkLoop()` (Thread A), checkpoint/resume flows |
| `src/orchestrator/CMakeLists.txt` | 29 | Static library `mpqs_orchestrator` |

Relation disk I/O (`serialize_v1/v2`) lives in `src/common/relation_io.{h,cpp}`; the checkpoint
file format helpers live in `src/common/sieve_checkpoint.{h,cpp}` (both linked via `mpqs_common`).

## MPQSConfig

### Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `N` | `uint512` | -- | Integer to factor |
| `device_id` | `uint32_t` | 0 | GPU device index |
| `node_id` | `uint32_t` | 0 | MPI-style node rank (logging) |
| `mode` | `ExecutionMode` | `FULL_PIPELINE` | Pipeline execution mode |
| `disk_io` | `bool` | false | Enable disk serialization between stages |
| `work_dir` | `string` | `./mpqs_work` | Working directory for disk I/O |
| `silent` | `bool` | false | Suppress constructor/destructor log output (used by autotune probes to prevent header pollution) |

### Diagnostic Dump Fields (sqrt-failure investigation; strictly opt-in)

All three are ADDITIVE: when false, the standard path is byte-for-byte unchanged.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dump_matrix` | `bool` | false | Dump finalized `matrix_A_` (CSR binary + human-readable column legend) to `work_dir`. CLI: `--dump_matrix` |
| `dump_kernel_vectors` | `bool` | false | Enable the BW writer (`bw_config.stage3_save_solutions`) AND dump the final `kernel_solutions_` (original-relation space) plus a sidecar documenting row space + bit→relation map. CLI: `--dump_kernel_vectors` |
| `dump_combine_provenance` | `bool` | false | Capture the two constituents (probe root, witness root, signs, `val_2_exps`, LP) of every LP-combined relation BEFORE the slab purge; serialize to `combine_provenance.bin`. CLI: `--dump_combine_provenance` |

### Tuning / Sieve Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_tune_parameters` | `bool` | true | If true, `determineParams()` overrides F/M from N |
| `lp1_variation` | `bool` | false | LP variant flag (overridden by auto-tune when enabled) |
| `fb_bound` | `uint32_t` | 0 (auto) | Factor base bound F |
| `sieve_bound` | `uint32_t` | 0 (auto) | Sieve interval half-width M |
| `lp1_bound` | `uint64_t` | 0 (disabled) | Large prime bound; 0 disables LP variant |
| `lp1_max_witness_capacity` | `uint64_t` | 0 (auto) | Max LP witness entries in slab hash. Auto: derived in `initLargePrimes()`, scaled with the LP-to-FB ratio and rounded to a power of 2 (Jetson default: 4M) |
| `lp_interval` | `uint32_t` | 1 | LP processing frequency: 0 = auto/adaptive, N > 0 = process LP every N batches. CLI: `--lp_interval` |
| `target_relations` | `uint32_t` | 0 (auto: FB+5%+64) | Target relation count |
| `dedup_safety_factor` | `double` | 1.05 | Oversample margin for dedup: collect `target × factor` relations. Auto-set to 1.35 for inputs < 80 digits. CLI: `--dedup_safety_factor` |
| `sieve_batch_size` | `uint32_t` | 0 (auto) | Batch GPU sieving; 0 = auto-calculate |
| `sieve_gms_num_blocks` | `uint32_t` | 0 (auto) | MetaSieve CUDA blocks |
| `sieve_hcube_dimension` | `uint32_t` | 0 (auto) | Hypercube dimension for polynomial construction |
| `sieve_accumulator_mode` | `int` | 0 (auto) | Sieve log-accumulator width: 0 = auto (`use_wide` dispatch predicate decides), 1 = force uint8, 2 = force uint16. CLI: `--sieve_accumulator auto\|u8\|u16` |
| `sieve_wide_accum_mode` | `int` | 0 (auto) | Wide-regime accumulator variant: 0 = auto (saturating-uint8 iff the exactness gate `APV_max−threshold ≤ 254` holds, else uint16), 1 = force u8sat (gate-honoured), 2 = force uint16. Only consulted in the wide regime. CLI: `--wide_accum auto\|u8sat\|u16` |
| `sieve_meta_cycle_cap` | `uint32_t` | 0 (OFF) | A2 meta-sieve SCATTER locality knob: caps `num_activeBlocksPerCycle` and raises `num_metaSieveCycles` correspondingly, bounding each thread's bucket-write spread independent of M. 0 = exact legacy geometry. CLI: `--sieve_meta_cycle_cap` |
| `sieve_gather_block_dim` | `uint32_t` | 0 (OFF) | A/B knob for the GATHER (sieve-and-scan) kernel's blockDim (`ss_conf.num_threadsPerBlock`). Result-invariant performance knob (only occupancy changes); power of two in [32, 1024]. 0 = loader-derived (currently 256). CLI: `--sieve_gather_block_dim` |
| `sieve_bucket_size_factor` | `double` | 0.0 (OFF) | Decouples large-prime bucket capacity from the legacy `globalBucketSize = SB/2`: F>0 sizes it `F·SB` (0.5 reproduces legacy; 1.0 doubles it). VRAM budget + config validator both account for the resized bucket. CLI: `--bucket_size_factor` |
| `sieve_block_size` | `uint32_t` | 0 (OFF) | **v1.0.6, narrow-batch + `--params` only.** Overrides `gs_conf.sievingBlockSize` **inside `loadPartialCustomConfig` only** (`device_sieving_controller.cpp:1539-1561`); `loadStandardConfig` is deliberately untouched, so without `--params` the flag would be inert — hence the CLI rejection below. 0 = OFF: the loader derives `SB = min(M, pow2leq(3·maxSharedMemPerBlock/4))`, byte-identical to v1.0.6. Admissible at the CLI: 0, or a **power of two ≥ 256** (`mpqs::sieve::admissibleSieveBlockSize`, `src/sieve/sieve_memory_model.h`); pow2 because the GATHER offset mask is `SB−1` and `validateConfigs` `POW2_CHECK`s the field. The upper bounds (`≤ M`, and the shared-memory sum) are **not** checkable at parse time and are enforced downstream by `LEQ_CHECK(ss_conf.sharedMemReq, maxSharedMemPerBlock)` + the narrow-batch occupancy preflight. `log2_sievingBlockSize` and `computeGlobalBucketSize(SB)` both follow the override automatically. ⚠ Lowering `SB` shrinks a launch's interval coverage, so `numIntervals` (`--params` field 2) must rise in step — see the coverage invariant below. CLI: `--sieve_block_size` |
| `sieve_big_prime_start` | `uint32_t` | 0 (OFF) | **v1.0.6, narrow-batch + `--params` only.** Overrides `gs_conf.bigPrimeStartIndex` — the GATHER/SCATTER factor-base split — which the loader otherwise ties **unconditionally** to `sievingBlockSize/32` (`device_sieving_controller.cpp:1588-1610`), so an `SB` override would silently migrate primes between the two paths as a confounded second change. 0 = OFF (derived `SB/32`, byte-identical). Admissible at the CLI: 0, or **strictly greater than 32** (`admissibleBigPrimeStart`); 32 is `gs_conf.midPrimeStartIndex`, and the mid-prime loops run `[midPrimeStart, bPSI)`, so `N ≤ 32` inverts the range and drops the whole mid-prime band silently. **Power-of-two is deliberately NOT required** (every consumer is a grid-stride loop and the GATHER shared-memory layout is plain pointer arithmetic). `N ≤ fb_size` and the smem sum are enforced downstream. CLI: `--sieve_big_prime_start` |
| `sieve_bucket_overflow_stats` | `bool` | false (OFF) | **v1.0.6, diagnostic only, no argument.** Relaxes the previously wide-only host reader `DeviceSievingController::getBucketOverflowStats()` (`device_sieving_controller.cpp:203-226`) so the **narrow** path also emits the `[BucketOverflow]` stats line (`orchestrator.cpp:3084-3095`). The device data always existed on both widths — the SCATTER kernel's bit-31 overflow flag is width-agnostic — only the host reader was gated, because it costs a `cudaMemcpyAsync` + `cudaStreamSynchronize` **on the siever stream** at the ~5 s stats cadence and narrow production is a zero-sync double-buffered pipeline. Measured cost at production geometry ≤ 0.09 % of the sieve wall. Wide behaviour is unaffected either way. Unlike the two overrides above it does **not** require `--params`. CLI: `--sieve_bucket_overflow_stats` |
| `autotune_probe_polys` | `uint32_t` | 0 (auto) | Wide-autotune survivors/sec probe sample size (# distinct polynomials staged; every candidate re-sieves the same sample). 0 = auto-scale by N. Wide path only. CLI: `--autotune_probe_polys` |
| `cuda_graph_unroll` | `uint32_t` | 0 (disabled) | Capture N batches as a CUDA graph for replay. Only odd values **> 1** are rounded up (`tests/cuda-mpqs.cpp`: `if (val > 1 && val % 2 != 0)`); `1` is a supported, unrounded probe value; capped at 16. Recommended: 2 or 4. CLI: `--cuda_graph_unroll` |
| `graph_capture_scope` | `GraphCaptureScope` | `AUTO` → `FULL` (solo) / `POSTPROC` (cluster) | How much of the per-batch body the graph captures: `SIEVE` (sieve kernels only — the pre-v1.0.6 body and the rollback path), `POSTPROC` (+ batch trial division), `FULL` (+ GPU LP, solo only; requested in cluster it is downgraded to `POSTPROC` with a `LOG_WARNING`). `MPQS_LP_DIAG=1` forces `SIEVE`. Inert when `cuda_graph_unroll == 0`. Resolution rules live in `include/graph_capture_scope.h`. CLI: `--cuda_graph_capture {sieve\|postproc\|full}` |
| `graph_lp_stride` | `uint32_t` | 1 | In-graph LP cadence: one dispatch every K-th captured batch. `1` = per batch (what `--cuda_graph_unroll 0` has always done); `0` or `>= cuda_graph_unroll` = one dispatch per replay (the pre-v1.0.6 graph cadence). Honoured only when LP is actually captured. CLI: `--cuda_graph_lp_stride` |
| `probe_timeout` | `double` | 120.0 | Hard timeout (seconds) for `TruncatedSieveRun()`. CLI: `--probe_timeout` |
| `estimate_only` | `bool` | false | Run truncated sieve in current topology, print runtime estimate, exit. CLI: `--estimate_only` |
| `sieve_max_relations` | `uint64_t` | 0 (disabled) | Truncation: stop sieve after N relations (coordinator-only in cluster mode — counts the pooled total). CLI: `--sieve_max_relations` |
| `sieve_max_batches` | `uint64_t` | 0 (disabled) | Truncation: stop sieve after N batch iterations. CLI: `--sieve_max_batches` |
| `sieve_truncate_continue` | `bool` | false | If true, continue the pipeline after a truncated sieve |
| `useParams` | `bool` | false | Use custom sieve parameter tuple |
| `params[8]` | `uint32_t[8]` | all 0 | Custom sieve parameters (passed to `loadPartialCustomConfig`) |
| `useParams11` / `params11[11]` | `bool` / `uint32_t[11]` | false / all 0 | **v1.0.7**, CLI `--params11`. `SieveParam` order: the eight `params` fields, then `sievingBlockSize`, `bigPrimeStartIndex`, `midPrimeStartIndex`. Converted by `makeDynamicParamSet()` (`orchestrator.cpp:134`; a 0 entry = unset) and loaded by `loadPartialCustomConfigDynamic()` in `SieveStage()` and `TruncatedSieveRun()`, taking precedence over `params` (and so over an autotune or history tuple). ⚠ The preflight, the LP `sasGridDim` floor and the legacy `sasBlockDim` cap below are keyed on `useParams` only and do not see `params11`; the small-N adaptive branch runs ahead of it |
| `param_test` | `bool` | false | **v1.0.7**, CLI `--param_test`. Runs the tuning-complex search at the start of `SieveStage()` and exits the process. A flag, not an `ExecutionMode` — `mode` stays `FULL_PIPELINE` |
| `param_test_radius` | `uint32_t` | 3 | **v1.0.7**, CLI `--param_test_radius` (≥ 1). Maximum face dimension of the tuning complex |
| `sieve_offsets_global` | `bool` | false | **v1.0.7**, CLI `--sieve_offsets_global`. Pushed via `setOffsetsInGlobal()` in both setter blocks; narrow only |
| `pinned_params` | `map<string,bool>` | empty | Records which config fields were explicitly set via CLI. `isPinned(name)` is checked by auto-apply, autotune, and the Jetson/small-N default blocks before overriding any user-provided value |

**Scope enforcement for the v1.0.6 geometry overrides — nothing is ever silently floored, clamped or downgraded.**
`sieve_block_size` / `sieve_big_prime_start` are rejected at three levels:

1. **CLI cross-flag guards** (`tests/cuda-mpqs.cpp`, the `cross-flag guards` block, evaluated after the
   whole `argv` sweep so flag order does not matter — each `exit(1)`s naming the reason):
   (a) without `--params` (both are consumed only inside `loadPartialCustomConfig`, so they would be
   inert); (b) combined with any `--autotune*` (the autotuner is deliberately blind to them — it
   re-enters `loadPartialCustomConfig` per probe with tuple-derived interval counts that the override
   would desynchronise from `SB`, and the autotune-side `KernelLaunchValidator` derives its own `SB`
   and never sees the override); (c) combined with `--sieve_batch_size 0` (legacy mode).
2. **CLI admissible-set predicates** — `admissibleSieveBlockSize` / `admissibleBigPrimeStart`, both in
   `src/sieve/sieve_memory_model.h` so the CLI and the host unit test share one statement of each rule.
3. **`validateConfigs()` re-assertion** (`device_sieving_controller.cpp:1979-1996`): re-rejects on the
   **wide** (uint16/u8sat) accumulator and in **legacy** mode with `LOG_ERROR_CRITICAL`. This runs after
   `setSievingBatchSize()`, which is the first point at which the batch/legacy predicate is well-defined.
   The loaders additionally gate both overrides on `!use_wide_accumulator_` at the point of use, so a
   wide run cannot reach an overridden field at all; the `validateConfigs` check is what makes the
   attempt *audible* rather than quietly inert.

**Narrow-batch interval-coverage invariant (v1.0.6, `narrowBatchCoverageOk`; ⚠ superseded in v1.0.7 — see the note at the end of this paragraph).**
Independent of the overrides and **not** gated on them: on the narrow batch path
`num_sievingBlocksPerSieveCall × sievingBlockSize ≥ 2M` is now checked in `validateConfigs()` and aborts
pre-sieve with `LOG_ERROR_CRITICAL`, naming the required `numIntervals`. `runSievingBatch()` never
advances `ds_params.startIndex`, so `gs_conf.num_sievingBlockBatches` — which the loader's older `[C1]`
guard multiplies into its coverage test — does not multiply this coverage; a config with
`intervals × SB == M` passed `[C1]` while silently sieving only `[-M, 0)`. `≥` and not `==`, because
over-coverage is legitimate (the autotune M-sweep produces it). Latent in every binary v1.0.5 and earlier; a
deliberate, recorded behaviour change for smaller-shared-memory devices that are half-sieving today.
Uses `uint64` arithmetic (the product overflows `uint32` at `M ≥ 2^31`).
**v1.0.7:** `validateConfigs()` (`device_sieving_controller.cpp:2800`) replaces this with an **equality** `numIntervals × SB == 2M` on **every narrow run, batch and legacy** — over-coverage is now rejected as well (it is fast per position and nearly barren, which a timing probe would reward), so the autotune M-sweep's over-covering probes no longer pass. `narrowBatchCoverageOk()` is no longer called in production.

### Buffer Sizing Overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `accum_buffer_size` | `uint64_t` | 0 (auto: `max(4096, batch_size·2048)`) | Accumulation buffer capacity |
| `partial_buffer_size` | `uint64_t` | 0 (auto: `= accum`, 1×) | Partial (LP staging) buffer |
| `persistent_buffer_size` | `uint64_t` | 0 (auto: `target·2 + accum`) | Persistent relation store |
| `lp1_max_combined_output` | `uint64_t` | 0 (auto: 32768) | LP match output buffer |
| `lp1_hash_bits` | `uint32_t` | 0 (auto-derived) | LP hash table directory bits |

### Autotune Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `autotune_enabled` | `bool` | false | Enable autotune before sieving |
| `autotune_config` | `AutotuneConfig` | defaults | Autotune stage enables, timeout, history, etc. |
| `autotune_stages_run` | `vector<uint32_t>` | empty | Populated after autotune: which stages ran |
| `autotune_confidence` | `double` | 0.0 | Populated after autotune: final confidence score |

### Matrix Preprocessing Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `matrix_mode` | `MatrixMode` | `AUTO` | Matrix construction mode. `AUTO`: **resolves to LEGACY for normal runs** (the old LP-fraction auto-switch was removed — preprocessing degrades the obstructed high-LP regime). `LEGACY`: projected (F+2)-column matrix. `PREPROCESS`: expanded (F+2+L)-column merge/filter pipeline (engages only via explicit flag, or AUTO in `MATRIX_ONLY` replay with raw partials). CLI: `--matrix_mode legacy\|preprocess` |
| `char_mode` | `matrix::CharMode` | `NONE` | Character-column symbol. `NONE`: zero char cols (default). `NORM`: legacy genus-blind NORM symbol. `BRANCH`: branch-fixed field-element symbol (aux primes `> lp1_bound`, fixed Tonelli roots). Never auto-enabled — set only by CLI. See [matrix.md](matrix.md). CLI: `--char_mode norm\|branch\|none` |
| `matrix_backend` | `int` | 0 (CPU) | Preprocessing backend: 0 = CPU, 1 = GPU, 2 = auto (GPU if available and >10K rows). CLI: `--matrix_backend cpu\|gpu\|auto` |
| `lp_preprocess_threshold` | `double` | 0.55 | **DEPRECATED / INERT.** Formerly the LP fraction above which AUTO selected PREPROCESS; the auto-switch was removed. Still parsed, no effect. CLI: `--lp_preprocess_threshold` |
| `lp_matrix_threshold` | `double` | 0.01 | **DEPRECATED** alias for `lp_preprocess_threshold` (also inert). CLI: `--lp_matrix_threshold` |
| `partial_subsample` | `double` | 1.0 | Fraction of partials/LP-combined to retain in `MATRIX_ONLY`. Range [0.0, 1.0]; 1.0 = no subsampling. CLI: `--partial_subsample` |
| `smooth_subsample` | `double` | 1.0 | Fraction of pure smooths (`large_primes ≤ 1`) to retain in `MATRIX_ONLY`. LP-combined relations are always retained. Range [0.0, 1.0]; 1.0 = no subsampling. CLI: `--smooth_subsample` |
| `matrix_lp1_bound` | `uint64_t` | 0 (inert) | `MATRIX_ONLY` LP-magnitude down-filter: after `.v2` load, DROP every LP-combined relation (and raw partial) whose stored large prime exceeds this value — reproducing the relation set a sieve at bound L would have produced, with NO re-sieve. Pure smooths (`large_primes ≤ 1`) are never dropped. Distinct from `--lp1_bound` (the sieve bound, restored from metadata). Composes with `--partial_subsample` (filter first). Suffix-aware K/M/B/T parse. CLI: `--matrix_lp1_bound` |
| `merge_max_weight` | `uint32_t` | 10 | DIAGNOSTIC: max column weight for higher-weight merges (preprocess CPU path, `mergeHigherWeight` k_max). Default 10 = no behavior change; 2 disables all weight≥3 multi-cycle merges. CLI: `--merge_max_weight` |
| `force_preprocess` | `bool` | false | DIAGNOSTIC: force the preprocess expand+merge path even with 0 raw partials (normally the orchestrator force-legacies in that case). CLI: `--force_preprocess` |
| `preprocess_lp_materialize_max` | `double` | 0.45 | Facet-3 gate: max combined-smooth LP fraction at which the preprocess path materializes matched raw-1-partial 2-cycle rows (above it they pin the BW kernel to the trivial genus → 0% nontrivial). 1.0 = never skip (pre-fix); 0.0 = always skip. CLI: `--preprocess_lp_materialize_max` |
| `truncation_min_rows` | `uint32_t` | 5000000 | Facet-2 size gate: skip CPU-preprocess truncation when the reduced matrix has ≤ this many rows (BW-tractable untruncated). CLI: `--truncation_min_rows` |
| `truncation_factor` | `double` | 1.05 | Matrix truncation on/off switch: > 0 = enabled, 0 = disabled. Actual target is excess-based: `n_cols + n_extra_cols + matrix_truncation_excess`. Retained as backward-compatible CLI toggle. CLI: `--truncation_factor` |
| `matrix_truncation_excess` | `uint32_t` | 200 | Excess rows above `(n_cols + n_extra_cols)` after truncation; controls how overdetermined the post-augmentation matrix is. CLI: `--matrix_truncation_excess` |
| `compact_cycles` | `uint32_t` | 5 | Max compact-merge cycles (GPU backend only). 0 = single pass (no compaction, reverts to pre-M10 behavior). CLI: `--compact_cycles` |
| `matrix_gf2_floor_factor` | `double` | 0.5 | GF(2) column-diversity floor: stop compact-merge cycles when surviving GF(2) cols fall below `max(matrix_gf2_min_floor, factor × initial_gf2_cols)`. CLI: `--matrix_gf2_floor_factor` |
| `matrix_gf2_min_floor` | `uint32_t` | 8192 | Absolute lower bound on the GF(2) column floor; prevents early termination on small test matrices. CLI: `--matrix_gf2_min_floor` |

### Linear Algebra Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bw_m` | `uint32_t` | 256 | Block Wiedemann block width m |
| `bw_n` | `uint32_t` | 256 | Block Wiedemann block width n |

### Square Root Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sqrt_legacy` | `bool` | false | If true, use CPU `Perform()` loop; default uses GPU batched path |
| `sqrt_diagnostic` | `bool` | false | If true, log solution-diversity statistics (distinct BW solutions by hash) at `LOG_INFO`. Per-solution nontrivial-GCD rate (`LOG_DEBUG_1`) and HalveExponents validity (`LOG_WARNING`) are logged unconditionally, independent of this flag. CLI: `--sqrt_diagnostic` |

### Checkpoint Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `checkpoint_interval_sec` | `uint32_t` | 0 (OFF) | Wall-seconds between mid-sieve checkpoints. 0 = disabled (complete no-op). CLI: `--checkpoint_interval <T>` |
| `checkpoint_batches` | `uint32_t` | 0 (OFF) | Alternative interval in sieve batches; fires first if both set. 0 = disabled. CLI: `--checkpoint_batches <N>` |
| `checkpoint_dir` | `string` | `""` → `work_dir/checkpoint` | Directory for checkpoint files. On the cluster: set to a run-stable (non-jobid) scratch path. CLI: `--checkpoint_dir <path>` |
| `resume` | `bool` | false | If true and a valid `sieve.ckpt` exists in `checkpoint_dir`, load it and continue; else start fresh with a warning. CLI: `--resume` |

`checkpoint_interval_sec == 0 && !resume` ⇒ every new code path is skipped and all outputs
are identical to a pre-feature run. The feature is never auto-enabled.

### Cluster Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cluster_mode` | `ClusterMode` | `SOLO` | Execution topology: `SOLO`, `COORDINATOR`, or `WORKER`. CLI: `--cluster_mode solo\|coordinator\|worker` |
| `data_tap` | `cluster::DataTap*` | `nullptr` | Injected DataTap pointer; set before `SieveStage()`. `nullptr` = solo mode |
| `listen_port` | `uint16_t` | 9100 | TCP port for coordinator to accept worker connections. CLI: `--listen_port` |
| `expected_workers` | `uint32_t` | 0 | Number of remote workers to wait for (coordinator). 0 = M1 (no workers). CLI: `--expected_workers` |
| `coordinator_host` | `string` | (empty) | Coordinator hostname or IP (worker only). CLI: `--coordinator_host` |
| `coordinator_port` | `uint16_t` | 9100 | Coordinator TCP port (worker only). CLI: `--coordinator_port` |
| `cluster_init_timeout` | `uint32_t` | 300 | Seconds: worker retry window + coordinator accept timeout. CLI: `--cluster_init_timeout` |
| `cluster_node_weights` | `string` | (auto) | Comma-separated per-node throughput weights; overrides SM×clock auto-balance. CLI: `--cluster_node_weights` |
| `cluster_headroom` | `double` | 10.0 | Per-node headroom percent (0 = exact assignment, default 10%). CLI: `--cluster_headroom` |
| `cluster_pool_oversize` | `double` | 1.0 | Coordinator-only multiplier enlarging the on-demand OVERFLOW pool of polynomial windows (pure index space drawn only after a node exhausts its initial range with the target unmet — over-sizing is essentially free; the run stops at the relation cap). Does NOT affect initial contiguous ranges. CLI: `--cluster_pool_oversize` |
| `transport` | `string` | `"tcp"` | Communication backend selector (future: `"mpi"`, `"gpi2"`) |
| `poly_range_start` | `uint64_t` | 0 | First a-index for this node (0 = natural start; set from `WORK_ASSIGN`/`CHUNK_ASSIGN`) |
| `poly_range_count` | `uint64_t` | 0 | Number of a-values in this node's contiguous range |
| `received_snapshot` | `AFactorsSnapshot` | -- | A-factor walk snapshot received via `WORK_ASSIGN` (workers only, M3 extension) |

### Embedded Component Configs

`PostProcConfig` (`pp_config`), `LargePrimeConfig` (`lp_config`), `BWSolverConfig` (`bw_config`) -- populated dynamically during `SieveStage()` and `LinearAlgebraStage()`.

## Public Interface

```cpp
explicit MPQSOrchestrator(const MPQSConfig& config);
~MPQSOrchestrator();

void Run();                                    // Execute pipeline per config_.mode
std::vector<mpqs::uint512> GetFactors() const; // Retrieve non-trivial factors

// --- Public for autotune probe orchestrator pattern ---
void TuningStage();
uint32_t getFactorBaseSize() const;
const sieve::factoringData& getFactoringData() const;

// Truncated sieve probe (runtime estimation)
TruncatedSieveResult TruncatedSieveRun(
    double frac = 0.12,
    double eta_convergence_threshold = 0.05,
    uint32_t min_eta_samples = 3,
    double min_progress_frac = 0.03);
```

Constructor binds GPU device via `CUDA_CHECK_FATAL(cudaSetDevice)` — a fatal error aborts with `std::runtime_error` if the device is unavailable. Logs hardware specs (suppressed when `config_.silent`), and creates `work_dir` if `disk_io` is set. Sets `f_data_.N = config_.N`. Destructor log also suppressed in silent mode.

## Pipeline Stages

### Stage 1: TuningStage

If `auto_tune_parameters`, calls `determineParams(&f_data_)` to heuristically set F/M from N. Then: `generateFactorBase(&f_data_)` (Tonelli-Shanks), `init_a_factors(&f_data_)`. Auto-calculates `target_relations` as `FB_size + 5% + 64` when zero.

### Autotune (between Tuning and Sieve)

When mode is `AUTOTUNE_ONLY`, or when `autotune_enabled` and mode is `FULL_PIPELINE` or `SIEVE_ONLY` (the SIEVE_ONLY autotune gate), the orchestrator instantiates `AutotuneController(autotune_config, config_, f_data_)` and calls `run()`. Even without `--autotune`, `AutoApplyController` applies history-based parameters after `TuningStage()` (zero GPU probes; disable with `--autotune_no_history`). The controller runs up to 4 stages (projection, kernel params, runtime estimation, sieve params) using truncated sieve probes with `TruncatedSieveRun()`. Results are written back to `config_` (stages run, confidence score). In `AUTOTUNE_ONLY` mode, the pipeline returns immediately after autotune completes.

Autotune probes use `silent = true` on sub-orchestrators to suppress log header/footer noise.

### Sieve Init Helpers

Three private helpers are shared by `SieveStage()` and `TruncatedSieveRun()` to eliminate initialization duplication:

- **`initPostProcessorConfig(accum_override)`** — Builds and returns a `PostProcConfig` from `config_`/`f_data_`. Sets accumulation buffer (CLI override → `accum_override` → default `max(4096, batch_size·2048)`), partial buffer (CLI override → 1× accum, floor-clamped to accum when LP active), persistent buffer (CLI override → `target·2 + accum`, floor-clamped if too small). `TruncatedSieveRun` passes `accum_override = 8192` for frequent probe granularity.
- **`initLargePrimes()`** — Configures `LargePrimeConfig` (witness capacity, hash bits, combined output, sort mode) from `config_`, constructs `largeprime_`, and calls `initiate()`. No-op if `lp1_bound == 0`.
- **`logBufferWarnings()`** — Emits fire-once near-full warnings (accum ≥ 90%, partial ≥ 80%, persistent ≥ 95%, witness ≥ 85%) and per-delta LP overflow warnings (slab, witness, output). Called from both batch and legacy sieve loops.

**Siever-side overrides must be pushed before `initiate()`.** Both `SieveStage()` (`orchestrator.cpp:4142-4151`) and `TruncatedSieveRun()` (`orchestrator.cpp:6427-6434`) call the same setter block on the freshly constructed `DeviceSievingController` — `setAccumulatorMode`, `setWideAccumMode`, `setMetaCycleCap`, `setGatherBlockDim`, `setBucketSizeFactor`, (v1.0.6) `setSievingBlockSizeOverride` / `setBigPrimeStartOverride` / `setNarrowOverflowStats`, and (v1.0.7) `setOffsetsInGlobal` — because every one of them is consumed by the config loader, which runs after `initiate()`.

### Stage 2: SieveStage (three dispatch paths)

**Initialization (shared):**
1. `DeviceSievingController::initiate(f_data_)`. If `lp1_bound > 0`, calls `setThresholdOverride(lp1_bound)`.
2. If `sieve_batch_size > 0`, calls `setSievingBatchSize` and `allocateBatchBuffers`.
3. **v1.0.7 `param_test` short-circuit** (`orchestrator.cpp:4173`): loads the seed (`loadPartialCustomConfigDynamic(params11)` if `useParams11`, else `loadStandardConfig()`), `loadData()`, `updateState()`, then `runParamTest(f_data_, seed, param_test_radius)`, which ends the process with `exit(0)` (or `exit(1)` on a rejected seed); on the wide path it logs and returns, and `SieveStage()` returns with no relations. Then the **`PARAM_TEST_LEGACY` short-circuit** (≤ v1.0.6 `PARAM_TEST`): `loadStandardConfig()` + `runParamTestLegacy(f_data_)`, logs the best timing and returns.
4. **sasGridDim auto-correction** (belt-and-suspenders): when LP is active with custom params, computes `min_sas = ceil(subCubeSize × numIntervals / 64)`, rounds up to next power-of-2, and raises `params[6]` if below minimum.
5. **sasBlockDim cap**: in legacy mode (`sieve_batch_size == 0`), caps `params[7]` to 1024 to match `__launch_bounds__(1024)` on sieveAndScanKernel.
6. Config loading: small-N adaptive config if `!useParams && max_polys < 64`; else `loadPartialCustomConfigDynamic(params11)` if `useParams11`, else `loadPartialCustomConfig(params...)` if `useParams`; else `loadStandardConfig`. Validates and calls `loadData` (which, since v1.0.7, also builds the small-prime mask).
7. **Preflight kernel launch check** (`preflightKernelLaunch`, `orchestrator.cpp:4320-4331`): when `useParams`, rejects infeasible kernel launch configurations before entering the sieve loop. Throws `std::runtime_error` on failure.
   **v1.0.6:** `SieveStage()` first computes `relaxed_geometry = (config_.sieve_batch_size > 0) && !siever_->isWideAccumulator()` — exactly `validateConfigs()`' own predicate, read from the siever whose wide/narrow decision was made in `initiate()` — and passes it as `preflightKernelLaunch`'s `allow_nonpow2_geometry` argument, so a pinned **SM-aligned (non-power-of-two)** `{np, metaGridDim, sasGridDim}` tuple passes preflight on the narrow batch path. Legacy and wide runs pass `false` = the mandatory-pow2 rule set.
   **Probe paths keep the mandatory-pow2 rule set, and that is self-consistent — not a gap.**
   `TruncatedSieveRun()` (`orchestrator.cpp:6390`) calls `preflightKernelLaunch` (`:6403`) without
   `allow_nonpow2_geometry`, so it takes the default `false`. That is correct: its sole caller is
   `estimateRuntime()` in `src/autotune/runtime_estimator.cpp`, which unconditionally sets
   `cfg.sieve_batch_size = 0;  // force legacy loop` (`:64`) before constructing the probe
   orchestrator, and `TruncatedSieveRun` correspondingly never calls `setSievingBatchSize`
   (`orchestrator.cpp:6439`, "Force legacy loop — no batch mode for probes"). The probe genuinely
   runs the **legacy** path, where mandatory power-of-two is the applicable rule — the same rule
   `validateConfigs()` applies there via its own `relaxed_geometry` predicate.
   ⚠ **Documented interaction (v1.0.6):** because the estimator's probes are legacy, a run that pins
   an **SM-aligned (non-power-of-two)** `--params` tuple *and* uses the runtime estimator / FL-optimizer
   probes will have every such probe skipped — `runtime_estimator.cpp` runs its own preflight
   (`:87-89`), also without the flag, and on failure emits
   `LOG_WARNING "[RuntimeEstimator] Skipping probe: preflight failed — …"` and returns the
   `total_est_sec = 1e9` sentinel. SM-aligned geometry is therefore usable for the production sieve
   but not for estimator-based runtime projection; this is a consequence of the decision to keep
   non-pow2 geometry `--params`-only and out of the autotuner, not a defect.
8. `initPostProcessorConfig()` builds `PostProcConfig`; persistent buffer is floor-clamped to `target_relations + accum` if undersized.
9. `initLargePrimes()` initializes LP variant (no-op if `lp1_bound == 0`).

**PATH 1: Batch** (`sieve_batch_size > 0`)

Zero-sync GPU pipeline with CUDA event DAG. Adaptive convergence:
- `dedup_margin = max(256, target × (dedup_safety_factor − 1.0))` (5% at the 1.05 default, matching the cluster accumulator), `relation_cap = target + dedup_margin`
- `postprocessor_->getPersistentBatch()->setTargetCap(relation_cap)` caps device-side writes
- `postprocessor_->setPredictionParams(target, lp_telemetry)` enables yield-rate prediction

Loop condition: `*h_pinned_persistent_count < target_relations && !h_prediction_result->should_terminate`

Per iteration:
1. `cudaStreamWaitEvent` on buffer's `safe_to_write_event`
2. `siever_->setPostProcessingLinks(buffer)`, `prepareSievingBatch`, `runSievingBatch`
3. `cudaEventRecord` on `safe_to_read_event`
4. `postprocessor_->updatePredictionSteps`, `processBatchBufferedCandidates`
5. Every N batches (if LP): sync both streams, `largeprime_->processAndCommit`, `resetPartialBatch`, `resyncPersistentDualCounter`, re-establish DAG invariant. N is adaptive (initial 10, calibrated from ETA).
6. Telemetry (every 10 batches): poll stale pinned counters, `logSieveProgress`, buffer fill monitoring with near-full warnings, LP overflow delta tracking

Post-loop: sync streams, flush stragglers (if `*pinned_counter > 0`, call `processBatchBufferedCandidates`). Final LP flush of remaining partials.

**PATH 1a: CUDA graph replay** (`cuda_graph_unroll > 0`)

The captured body is `graph_N` unrolled iterations of the PATH 1 body — not just the sieve kernels — with the **two accumulation buffers alternating inside the graph**: `processBatchBufferedCandidates()` toggles the host index once per captured batch **at capture time**, so batch `i` is permanently bound to buffer `(start + i) % 2` and batch `i+2`'s `safe_to_write` wait is a genuine internal graph edge. What is captured depends on the resolved scope (`sieve` / `postproc` / `full`; solo defaults to `full`, cluster to `postproc`).

- **Replays are serialized.** A graph launched into a stream is one stream work item ("each launch is ordered behind both any previous work in stream and any previous launches of graphExec"), so the overlap gained is *intra*-replay: batches `0 … graph_N−2` hide their post-processing under their successors' sieve, while the last batch's post-processing is an exposed tail. In solo at scope `full` (even `graph_N`) that tail is **deferred into the next replay** as a root of the post-processing branch, and the one un-post-processed buffer is drained after the graph loop.
- **Solo at scope `full`: no host synchronize between replays.** Solo at scope `postproc`/`sieve` keeps one (the retained between-replay LP block runs on `proc_stream`, which the graph does not order against); cluster keeps one, expressed as a wait on a completion event recorded **outside** the capture followed by `cudaStreamSynchronize(extract_stream)` — extraction needs exact counters. The predicate is `cap.keep_host_sync = is_cluster || !capture_lp`.
- **Staging.** One device slot-set (written and read only on the launch stream, so in-stream order already separates replay `k`'s reads from replay `k+1`'s writes) but **two pinned index sets**, alternating behind a `stage_done` event, because the host `memcpy` into pinned staging would otherwise race a still-pending H2D that reads it.
- **The host's pinned progress counters lag by at most one replay**, so the stop condition can overshoot slightly; the overshoot is bounded by the existing `relation_cap` (measured +0.33 % at RSA-100 / `graph_N = 4`, well inside the cap).
- **After the graph loop** the launch stream is quiesced and every event the capture recorded is re-recorded from a non-capturing stream before the standard loop takes over for the final stretch — an event whose last record was inside a capture is unusable from the host until re-recorded.
- **Skipped entirely** when `CUDA_LAUNCH_BLOCKING=1`, in legacy (non-batch) mode, when `sieve_max_batches < graph_N`, or when the target is too small; `MPQS_LP_DIAG=1` forces the scope down to `sieve` (its diagnostic block issues calls that are illegal inside a capture). A capture that fails is retried once at scope `sieve` and otherwise falls back to the standard loop.
- **Escape hatches:** `--cuda_graph_capture sieve` restores the pre-v1.0.6 captured body per run; `--cuda_graph_unroll 0` bypasses graphs entirely and is byte-for-byte unchanged. ⚠ **The "byte-for-byte unchanged" half of that sentence is SUPERSEDED (2026-08-25):** the *code path* is unchanged, but `--cuda_graph_unroll 0` is **not** a bit-identity baseline on the solo GPU-LP path — it is **±1 nondeterministic** on `LP combined`, `Total (deduped)` and `Cumulative LP full`, measured on both an RTX 5070 Ti and an A100 and on the *pre-change* baseline binary as well as after, so it is not a v1.0.6/b/c regression. `Sieved full`, `Duplicates`, `Batches processed` and the whole `[Config]` block are exact. **Consequence: arm-to-arm work identity must be judged at ±1, never by exact equality** — an exact-equality acceptance criterion cries wolf. The jitter itself is unexplained.

**PATH 2: Legacy + LP** (`sieve_batch_size == 0`, `lp1_bound > 0`)

Host-driven loop: per step `siever_->updateState`, `sieveFullCube`, `postprocessor_->accumulate`. When buffer full: `processBufferedCandidates`, `consolidateToPersistent`, `largeprime_->processAndCommit`, `resetPartialBatch`. Async telemetry from `largeprime_->getTelemetry()` on generation-ticket change drives `logSieveProgress`. Exits when `getPersistentCount() >= target_relations`.

**PATH 3: Legacy no-LP** (`sieve_batch_size == 0`, `lp1_bound == 0`)

Same host-driven loop; when buffer full: `processBufferedCandidates`, `consolidateToPersistent`. Logs progress every 200 new relations. Exits when `getPersistentCount() >= target_relations`.

**Post-loop (all paths):**
Legacy paths: `postprocessor_->flush()`, optional final LP commit. All paths: clear sticky CUDA error via `cudaGetLastError`, then `finalizePersistentToHost()` — end-of-sieve drain → device `deduplicatePersistentBatch()` → deficit-guard re-sieve if the post-dedup count falls below `fb_size + 64` (`kDeficitFloorAuto`) → copy into `host_relations_soa_`/`host_partials_soa_`. A solo resume instead runs `finalizeResumeUnion()` (loaded ∪ new union merge, see Sieve Checkpointing below).

Cleanup: `siever_->clearSievingBuffers()`, `postprocessor_->clearBuffers()`, `largeprime_->clearBuffers()` (if active). Persistent batch is **not** cleared here — `MatrixStage` reads it.

### Stage 3: MatrixStage

Two construction paths:
- **Normal**: `matrix_constructor_->constructFromSoA(postprocessor_->getPersistentBatch()->get_view(), count, csr_matrix)` using the live device batch
- **LINALG_ONLY fallback**: allocates `RelationBatch temp_batch`, calls `temp_batch.uploadFromHost(host_relations_soa_)`, then constructs from that view

⚠ **Known defect (pre-existing, unfixed — re-verified against source and STILL OPEN in v1.0.6):** the
Normal path is taken whenever `postprocessor_` exists (`orchestrator.cpp:6737-6744`) and reads the
**device** batch, but a solo `--resume` merges loaded ∪ new ∪ cross-checkpoint combines into
`host_relations_soa_` only (`finalizeResumeUnion`, defined `orchestrator.cpp:3995`, called from the
end-of-sieve path at `orchestrator.cpp:6102`, reporting at `:4067`). A resumed FULL_PIPELINE run therefore matrices the new
leg alone (measured: `final merged set: 265,639` vs `Processing 112617 relations from Device SoA` →
112,617 × 190,999, under-determined → 0 Block-Wiedemann solutions → no factors), while the same relations
replayed through `--sieve_only` + `--matrix_only` (which uploads the host SoA) give
`System Overdetermined. Excess: 76083`, 205 solutions and factors. Reproduces at `--cuda_graph_unroll 0`
and `4` and on the pre-v1.0.6 binary — it is not a graph-capture regression. Workaround: resume with
`--sieve_only` and run the matrix stage separately. Fix direction: take the host-SoA upload path when
`resume_active_`.

CSR columns = `f_data_.size + 2` (sign + exponent-of-2). Post-construction: `ValidateHostMatrixCSR`, then `matrix_A_ = ConvertFromCSR(csr_matrix)`. Verifies system is overdetermined (`n_rows > n_cols`). Calls `postprocessor_->clearPersistentBuffer()` on success.

### Stage 4: LinearAlgebraStage

Configures `BWSolverConfig`: `solve_transposed=true` (left kernel), `stage2_gpu_mode=true`, `stage1_gpu_batch_size=8`, `checkpoint_prefix = work_dir + "/bw"`, `m_block = bw_m`, `n_block = bw_n`. Pads matrix to square via `lingen::pad_to_square(matrix_A_)`. Instantiates `BlockWiedemannSolver` and calls `.Solve()`. Stores kernel vectors in `kernel_solutions_` and retains device-side view via `linalg_solver_->get_device_solutions()`.

### Stage 5: SquareRootStage

Instantiates `SquareRootRefinement(config_.N)`. Two paths:

**GPU batched path (default, `sqrt_legacy == false`):**
1. `sqrt_solver_->ComputeXBatchedGPU(sol_view, host_relations_soa_)` — uploads `sqrt_Q`, computes X[i] for all solutions on GPU
2. `sqrt_solver_->ComputeYBatchedGPU(sol_view, host_relations_soa_, factorBase)` — computes Y[i] on GPU
3. Verifies X²≡Y² (mod N) for each i via `mpqs::math::modpow`
4. `sqrt_solver_->BatchedGCD(d_X, d_Y, n)` — extracts factors on GPU; checks non-triviality (factor ≠ 1 and ≠ N); validates F1×F2 == N
5. If no nontrivial GPU result: falls back to CPU loop (see below)

**CPU legacy path (`sqrt_legacy == true` or GPU fallback):**
Iterates `kernel_solutions_`. For each: `sqrt_solver_->Perform(solution, host_relations_soa_, factorBase)` returns (F1, F2). Checks non-triviality, validates F1×F2 == N via `uint512::mult`. Stops on first verified non-trivial factorization.

## Sticky CUDA Error Clearing

Two sites call `cudaDeviceSynchronize()` + `cudaGetLastError()` to drain sticky CUDA errors:
1. **TruncatedSieveRun entry**: clears errors from prior stages (e.g., Stage 1 kernel benchmarking) before the probe.
2. **TruncatedSieveRun cleanup**: clears errors after GPU resource destruction so the next probe starts clean.

The SieveStage post-loop also calls `cudaGetLastError()` to clear any sticky error before deduplication.

## SieveProgressTracker (public inner struct)

Implements sliding-window quadratic ETA extrapolation (FP64):

```cpp
struct SieveProgressTracker {
    struct Sample { double t; double y; };
    std::deque<Sample> history;      // ring buffer, max WINDOW_K=24 points
    double current_eta_sec = 0.0;
    uint64_t last_logged_count = 0;

    void addSample(double elapsed_sec, uint64_t relation_count, uint64_t target);
    bool hasETA() const { return history.size() >= 3; }
};
```

`addSample`: fits a rate model r(t) = r₀ + α·t via OLS on consecutive-sample rates over a sliding window. Integrates to y(t) = C + r₀·t + (α/2)·t² with integration constant C fitted via least-squares. Solves the quadratic (α/2)·ΔT² + r_now·ΔT - remaining = 0 for ETA. Falls back to linear extrapolation if α ≤ 0. Output is EMA-smoothed.

`logSieveProgress(tracker, rel_count, target, elapsed_sec, lp_active, witnesses, lp_full_rels)`: unified progress line called by both batch and legacy loops. Calls `tracker.addSample`, emits throughput (LOG_DEBUG_1 when LP active), progress percentage and rel/s (LOG_INFO), and ETA (LOG_INFO, once ≥ 3 samples and target not yet reached).

## LPFillProjector (public inner class)

Linear projection of LP witness hash table fill at estimated run completion. Collects (time, witness_count) samples and fits a linear model `w(t) = α*t + β` via least-squares over a sliding window of 8 samples. Uses the SieveProgressTracker ETA to extrapolate witness count at run end.

```cpp
struct Projection {
    double   projected_fill_pct;     // Projected fill % at estimated run end
    uint64_t projected_witnesses;    // Projected witness count
    uint64_t recommended_capacity;   // Next-power-of-2 of (projected × 1.25)
    bool     overflow_likely;        // projected_fill_pct > 95%
    bool     oversized;              // projected_fill_pct < 20%
};
```

## TruncatedSieveResult (public inner struct)

Result of a truncated sieve probe run, containing all telemetry needed for runtime estimation:

| Field | Type | Description |
|-------|------|-------------|
| `progress_tracker` | `SieveProgressTracker` | ETA state at probe end |
| `buffer_fill` | `BufferFillHistory` | Buffer peak/capacity history |
| `lp_fill` | `LPFillHistory` | LP hash table fill history |
| `lp_projector` | `LPFillProjector` | Witness fill projection state |
| `elapsed_sec` | `double` | Wall-clock probe duration |
| `relations_found` | `uint64_t` | Relations accumulated |
| `target_relations` | `uint64_t` | Full run target |
| `steps_executed` | `uint64_t` | Sieve steps completed |
| `eta_reliable` | `bool` | True if ≥ 6 ETA samples and tracker converged |
| `converged_early` | `bool` | True if probe exited via ETA convergence |

## Relation Disk Serialization (`src/common/relation_io.{h,cpp}`)

Two formats, both under namespace `mpqs::io`, auto-detected on load via `detect_and_deserialize()`:

- **v1** (`serialize_v1` / `deserialize_v1`, magic `MPQS_SOA\0`): single `HostRelationBatch`
  with projected LP values — flat SoA vectors (`sqrt_Q`, `signs`, `val_2_exps`, `large_primes`,
  `factor_offsets`, `factor_indices`, `factor_counts`), each length-prefixed. Written to
  `{work_dir}/relations.soa`; loaded by `LINALG_ONLY`.
- **v2** (`serialize_v2` / `deserialize_v2`, magic `MPQS_V2\0`, version=2, flags bitfield): full
  smooths + raw partials + `V2Metadata` (N, factor base, `lp_bound`, `sieve_bound`; plus a
  flag-guarded branch-char extension — `aux_primes`, `t_s`, `r`, `has_char_bits` — appended at
  the end of the fixed metadata block so char-less files parse byte-for-byte unchanged). Written
  to `{work_dir}/relations.v2`; loaded by `MATRIX_ONLY`. Also the payload format of the mid-sieve
  checkpoint `sieve.ckpt` (see below).

BW checkpoints: `{work_dir}/bw*`.

## Execution Modes

| Mode | Stages Executed | Disk I/O |
|------|-----------------|----------|
| `FULL_PIPELINE` | Tuning [+ Autotune] + Sieve + Matrix + LinAlg + Sqrt | Optional save after sieve (`disk_io`) |
| `SIEVE_ONLY` | Tuning [+ Autotune] + Sieve | Save required: writes BOTH `relations.soa` (v1) and `relations.v2` (v2, smooths + partials + metadata), then returns |
| `LINALG_ONLY` | Tuning + Matrix + LinAlg + Sqrt | Load required (reads `relations.soa`, v1) |
| `MATRIX_ONLY` | Load v2 relations [+ `matrix_lp1_bound` L-filter + subsampling] → Matrix → BW → Sqrt | Load required (reads `relations.v2`), no sieve |
| `SQRT_ONLY` | *(not implemented)* | Logs `LOG_ERROR_CRITICAL` and returns immediately — no disk loader for kernel solutions or factor base |
| `PARAM_TEST_LEGACY` | Tuning + Sieve init (calls `runParamTestLegacy`, then returns). CLI `--param_test_legacy`; this was `PARAM_TEST` / `--param_test` up to v1.0.6 | None |
| `AUTOTUNE_ONLY` | Tuning + Autotune | None (prints results, returns) |

Note: `LINALG_ONLY` always runs `TuningStage` to reconstruct the factor base needed for matrix column indexing.

**v1.0.7 `--param_test` is not a mode.** It sets `MPQSConfig::param_test`; the run is a `FULL_PIPELINE` run (Tuning [+ Autotune], history auto-apply eligible) that ends inside `SieveStage()` when the search calls `exit(0)`. The superseded grid search is the mode `PARAM_TEST_LEGACY`, which is also what `shouldAutoApply()` and the `TuningStage` gate now name.

## Sieve Checkpointing

### Overview

Mid-sieve relation checkpointing enables a killed or wall-clock-expired sieve to resume
from disk instead of re-sieving from zero. Default-off: `checkpoint_interval_sec == 0 &&
!resume` is a complete no-op with identical outputs. See the [Checkpoint Fields](#checkpoint-fields)
table for all four CLI flags.

### Checkpoint Artifact (`sieve.ckpt`)

`<checkpoint_dir>/sieve.ckpt` layout:

```
[ serialize_v2 payload ]   -- smooths (HostRelationBatch) + partials; verbatim v2 format
[ progress trailer ]       -- magic "MPQS_CKPT", ckpt_schema_version, global_a_index (u64),
                           --   target_relations, loaded_smooths_raw, loaded_smooths_dedup,
                           --   loaded_partials, lp1_bound, sieve_bound, N (64B),
                           --   cluster_section_present (u8), elapsed_sieve_sec
[ cluster block ]          -- (when cluster_section_present=1): completedPrefixCursor (u64)
                           --   + per-node initial-range high-water array (count + values)
[ fixed EOF footer ]       -- magic "MPQS_CKFT", trailer_offset (u64), trailer_len (u64),
                           --   ckpt_schema_version (u32)
```

**Graph path (`cuda_graph_unroll > 0`):** the replay loop calls `checkpointWillFire()` and, only when
that returns true, `quiesceForCheckpoint()` — `cudaStreamSynchronize` on the launch stream (which waits
for every graph node, including the forked post-processing branch) and then on `proc_stream` — before
`maybeCheckpoint()`. Without it the snapshot can be torn on the LP-off path, where
`copyPersistentToHost`'s `cudaStreamSynchronize(pp_stream)` is a no-op once post-processing is a graph
node. Gating on "checkpointing is *enabled*" instead of "a checkpoint is about to *fire*" would
reintroduce a per-replay synchronize on every checkpoint-enabled production run, so the predicate is the
fire condition; `maybeCheckpoint()` calls the same predicate, so there is exactly one copy of the logic.
On the deferred-tail path (solo, scope `full`, even `graph_N`) the value handed to `maybeCheckpoint()` is
`current_step − sieve_batch_size`, because one batch's candidates are not yet post-processed at a replay
boundary.

The footer magic at EOF is the completeness sentinel: a torn write never has it.
Write protocol: unlink stale `.tmp` → `serialize_v2` to `.tmp` (intra-FS) → append
trailer+cluster block+footer → `fsync` → rename live to `.prev` → rename `.tmp` to live →
`fsync` directory. One prior generation is retained as `sieve.ckpt.prev`. Completely distinct
from the matrix-handoff `relations.v2` (format and write path unchanged).

### Solo Resume Flow

On `--resume` at `SieveStage` entry with a valid `sieve.ckpt`:

1. Load `sieve.ckpt`; validate footer magic, trailer `N == config N`, and
   `loaded_smooths_dedup == host_relations_soa_.num_relations`.
2. `saveSnapshot()` + `resetAndAdvanceTo(trailer.global_a_index)` — continue the polynomial
   walk from the first un-sieved a-index.
3. **Effective target:** `max(0, target_relations − loaded_smooths_raw)` — applied to all
   three termination paths (pinned-count stop test, yield-prediction `should_terminate`, and
   device target cap `setTargetCap`).
   ⚠ **Known defect (pre-existing, unfixed — re-verified against source and STILL OPEN in v1.0.6):**
   `loaded_smooths_raw` is the **pre-dedup device** count (`maybeCheckpoint`, defined
   `orchestrator.cpp:3822`, trailer write at `:3872`), which mid-run with LP on runs far ahead of
   the useful count because the live device batch still holds duplicate LP appends. Measured at
   RSA-100 (`--cuda_graph_unroll 0`, killed at 45 s): trailer `loaded_smooths_raw = 201,197` vs
   `loaded_smooths_dedup = 19,390` — 10.4× — against `target_relations = 200,610`, so the resume
   short-circuits (`effective_target=0 … loaded >= target: short-circuit (finalize only)`,
   `orchestrator.cpp:4446-4453`) and finalizes 19,390 relations in 0.4 s. LP-off checkpoints are
   unaffected (74,919 vs 74,590). The quantity the union actually contributes is
   `loaded_smooths_dedup`, which is what the resume should subtract.
   Much of that 10.4× ratio came from the pinned index-staging overwrite race (a solo GPU-LP defect
   present in every binary up to and including v1.0.5, **fixed in v1.0.6**), which re-sieved the final batch ~6×; the
   trailer nonetheless still records the wrong quantity, so the defect stands on its own.
4. Sieve the new leg; device `deduplicatePersistentBatch()` runs at end-of-sieve for the new
   leg only (B1: never mid-loop).
5. **End-of-sieve union dedup:** host-merge `loaded ∪ (device-deduped new)` via the shared
   `computeRelationHash`; re-assert `fb_size+64` on the union (trigger more sieving if short).
6. **Final host LP re-match:** over `loaded_partials ∪ new-un-combined-partials` via the shared
   `cpu_lp` combiner, after the union hash-set is built; skipped cleanly if `lp1_bound == 0`.
7. Write `relations.v2` normally. The resumed output is as valid as a single uninterrupted run.

`--resume` with no checkpoint present → warn and start fresh (identical to a normal run).
`loaded >= target` short-circuit: skip the sieve loop but still run the union-merge + LP
re-match + `relations.v2` write.

### Cluster Resume Flow

Coordinator-only (workers are stateless; they reconnect and request work as usual).
All restore steps happen **before Thread A starts and before any `requestWork`/`checkoutWork`**
(the `setCursor` startup-only contract).

1. Load + validate `sieve.ckpt` (cluster block must be present; topology guard checks
   `node_count` match and overflow-pool prefix bounds; reject → start fresh).
2. Re-inject smooths via `RelationAccumulator::addRelations` (rebuilds `accumulated_` and
   the `seen_` dedup set).
3. Re-feed partials via `CPULargePrimeTable::insertAndMatch` (rebuilds `table_`, re-emits
   deduped combines) + buffer into `cluster_raw_partials_`.
4. `WorkPool::setCursor(completedPrefixCursor)` — restores the overflow pool to the completed
   contiguous prefix (NOT `nextCursor()`, which would drop in-flight/returned chunks and
   re-introduce the `4d20d7b` pool exhaustion).
5. Re-issue each node's initial `WORK_ASSIGN` trimmed to
   `[orig_start + hw_node, orig_count − hw_node)` from the per-node initial-range high-water
   array. Fully-complete nodes re-sieve the last hypercube H (dedup-safe boundary guard).
6. Thread A proceeds; the cluster sieves only the remaining a-space (overflow tail above the
   prefix + trimmed initial tails) and tops up to target.

### Production sbatch Wiring

- **Run-stable `CKPT_DIR`:** keyed on `RUN_TAG` (NOT `SLURM_JOB_ID`), so a resubmitted job
  with a fresh jobid finds the prior run's `sieve.ckpt` and auto-resumes.
  `CKPT_DIR="$RUN_BASE/cuda-mpqs/${RUN_TAG}_ckpt"`.
- **Resume detection:** `if [ -s "$CKPT_DIR/sieve.ckpt" ]; then COORD_RESUME="--resume"; fi`
  before the Phase-1 `srun`.
- **Coordinator-only flags:** `--checkpoint_dir "$CKPT_DIR" --checkpoint_interval 1800
  $COORD_RESUME` passed to rank 0 only; workers are unchanged.
- `relations.v2` (the Phase-2 `--matrix_only` handoff) is written only at sieve completion,
  unchanged. `sieve.ckpt` is an internal resume artifact only.

## Internal State

| Member | Type | Purpose |
|--------|------|---------|
| `config_` | `MPQSConfig` | Runtime configuration (mutated during tuning/autotune) |
| `f_data_` | `sieve::factoringData` | Factor base, polynomial state |
| `is_jetson_` | `bool` | SM 8.7 or unified-memory < 12 GB detected at runtime |
| `data_tap_` | `cluster::DataTap*` | Copied from `config_` at `SieveStage()` entry; `nullptr` = solo |
| `host_relations_soa_` | `structures::HostRelationBatch` | Downloaded relations (SoA) |
| `host_partials_soa_` | `structures::HostRelationBatch` | Raw partials (LP > 1) for the expanded-matrix path and `.v2` output |
| `matrix_A_` | `HostMatrix` | Sparse GF(2) relation matrix (jagged rows) |
| `kernel_solutions_` | `vector<vector<uint64_t>>` | Packed kernel vectors from BW solver |
| `result_factors_` | `vector<uint512>` | Final non-trivial factors |
| `siever_` | `unique_ptr<DeviceSievingController>` | GPU polynomial siever |
| `postprocessor_` | `unique_ptr<DevicePostProcessingController>` | Trial division and batch accumulation |
| `largeprime_` | `unique_ptr<LargePrimeVariant>` | Single large prime matching (null if disabled) |
| `matrix_constructor_` | `unique_ptr<MatrixConstructor>` | CSR matrix builder |
| `linalg_solver_` | `unique_ptr<BlockWiedemannSolver>` | BW solver (holds device solutions until Sqrt) |
| `sqrt_solver_` | `unique_ptr<SquareRootRefinement>` | Congruence-of-squares factor extraction |
| `buffer_fill_history_` | `BufferFillHistory` | Peak/capacity tracking for all pipeline buffers |
| `lp_fill_history_` | `LPFillHistory` | LP hash table fill tracking |
| `lp_projector_` | `LPFillProjector` | Witness fill projection for adaptive sizing |
| `adaptive_lp_batch_interval_` | `uint32_t` | LP processing interval (initial 10, calibrated from ETA) |
| `cluster_*` members | various | Coordinator-only (null in solo/worker): `cluster_queue_`, `cluster_accumulator_`, `cluster_handoff_`, `cluster_cpu_lp_`, `cluster_channel_`, `cluster_thread_a_`, `comm_backend_`, `cluster_work_pool_`, `cluster_scheduler_`, `cluster_raw_partials_`, per-node initial-range tracking (`cluster_initial_ranges_`, `cluster_node0_initial_hw_`), and resume state (`cluster_resume_*`) — see [cluster.md](cluster.md) |
| checkpoint state | various | Solo mid-sieve checkpoint/resume: `last_checkpoint_time_`, `ckpt_scratch_*` (copy-only snapshot buffers), `resume_active_`, `resume_loaded_raw_`, `resume_global_a_index_`, `ckpt_loaded_*` (held for the end-of-sieve union merge) |
| preprocessing state | various | `merge_tree_`, `preproc_row_map_`, `used_expanded_matrix_`, `used_packed_pipeline_`, `preproc_v2_result_`, `precomputed_lp_y_` (Montgomery-domain LP Y-contributions) |
| branch-char state | various | `branch_aux_primes_`/`branch_aux_t_s_` (+ device copies), selected once by `initBranchCharData()` under `--char_mode branch` |

## Dependencies

Links (PUBLIC): `mpqs_autotune`, `mpqs_common`, `mpqs_sieve`, `mpqs_postproc`, `mpqs_largeprimes`, `mpqs_matrix`, `lingen_ops`, `mpqs_sqrt`, `cudampqs_build_flags`. The cluster include directory (`src/cluster/`) is added via `target_include_directories`; cluster types are forward-declared in `orchestrator.h`. `cudampqs_cluster` is linked at the binary level (`tests/CMakeLists.txt`), not inside `mpqs_orchestrator` itself.
