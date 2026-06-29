# Orchestrator Module (`src/orchestrator/`, `include/`)

Central pipeline driver. Coordinates the 5-stage MPQS factorization pipeline, manages configuration, execution modes, autotune integration, and inter-stage data flow.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `include/orchestrator.h` | ~400 | `MPQSOrchestrator` class, `MPQSConfig` struct, `ExecutionMode` enum, `SieveProgressTracker`, `LPFillProjector`, `TruncatedSieveResult` |
| `src/orchestrator/orchestrator.cpp` | ~2450 | Pipeline implementation, SoA serialization, stage dispatch, truncated sieve probes |
| `src/orchestrator/CMakeLists.txt` | 23 | Static library `mpqs_orchestrator` |

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

### Tuning / Sieve Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_tune_parameters` | `bool` | true | If true, `determineParams()` overrides F/M from N |
| `lp1_variation` | `bool` | false | LP variant flag (overridden by auto-tune when enabled) |
| `fb_bound` | `uint32_t` | 0 (auto) | Factor base bound F |
| `sieve_bound` | `uint32_t` | 0 (auto) | Sieve interval half-width M |
| `lp1_bound` | `uint64_t` | 0 (disabled) | Large prime bound; 0 disables LP variant |
| `lp1_max_witness_capacity` | `uint64_t` | 0 (auto: 1M) | Max LP witness entries in slab hash |
| `lp1_sort_bound` | `uint32_t` | 0 (disabled) | Enables second sort in LP processing |
| `lp_interval` | `uint32_t` | 1 | LP processing frequency: 0 = auto/adaptive, N > 0 = process LP every N batches. CLI: `--lp_interval` |
| `target_relations` | `uint32_t` | 0 (auto: FB+5%+64) | Target relation count |
| `dedup_safety_factor` | `double` | 1.05 | Oversample margin for dedup: collect `target × factor` relations. Auto-set to 1.35 for inputs < 80 digits. CLI: `--dedup_safety_factor` |
| `sieve_batch_size` | `uint32_t` | 0 (auto) | Batch GPU sieving; 0 = auto-calculate |
| `sieve_gms_num_blocks` | `uint32_t` | 0 (auto) | MetaSieve CUDA blocks |
| `sieve_hcube_dimension` | `uint32_t` | 0 (auto) | Hypercube dimension for polynomial construction |
| `cuda_graph_unroll` | `uint32_t` | 0 (disabled) | Capture N batches as a CUDA graph for replay. Must be even (double-buffer constraint). Recommended: 2 or 4. CLI: `--cuda_graph_unroll` |
| `probe_timeout` | `double` | 120.0 | Hard timeout (seconds) for `TruncatedSieveRun()`. CLI: `--probe_timeout` |
| `estimate_only` | `bool` | false | Run truncated sieve in current topology, print runtime estimate, exit. CLI: `--estimate_only` |
| `useParams` | `bool` | false | Use custom sieve parameter tuple |
| `params[8]` | `uint32_t[8]` | all 0 | Custom sieve parameters (passed to `loadPartialCustomConfig`) |

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
| `truncation_factor` | `double` | 1.05 | Matrix truncation on/off switch: > 0 = enabled, 0 = disabled. Actual target is excess-based (M12-S1): `n_cols + n_extra_cols + matrix_truncation_excess`. Retained as backward-compatible CLI toggle. CLI: `--truncation_factor` |
| `matrix_truncation_excess` | `uint32_t` | 200 | Excess rows above `(n_cols + n_extra_cols)` after truncation; controls how overdetermined the post-augmentation matrix is. CLI: `--matrix_truncation_excess` |
| `compact_cycles` | `uint32_t` | 5 | Max compact-merge cycles (GPU backend only). 0 = single pass (no compaction, reverts to pre-M10 behavior). CLI: `--compact_cycles` |
| `matrix_gf2_floor_factor` | `double` | 0.5 | M12-S2 GF(2) column-diversity floor: stop compact-merge cycles when surviving GF(2) cols fall below `max(matrix_gf2_min_floor, factor × initial_gf2_cols)`. CLI: `--matrix_gf2_floor_factor` |
| `matrix_gf2_min_floor` | `uint32_t` | 8192 | M12-S2 absolute lower bound on the GF(2) column floor; prevents early termination on small test matrices. CLI: `--matrix_gf2_min_floor` |

### Linear Algebra Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bw_m` | `uint32_t` | 256 | Block Wiedemann block width m |
| `bw_n` | `uint32_t` | 256 | Block Wiedemann block width n |

### Square Root Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sqrt_legacy` | `bool` | false | If true, use CPU `Perform()` loop; default uses GPU batched path |
| `sqrt_diagnostic` | `bool` | false | If true, log extra sqrt diagnostics: per-solution nontrivial-GCD rate (`k/n`) per Block-Wiedemann solution, HalveExponents validity, solution diversity (at `LOG_DEBUG_1`). CLI: `--sqrt_diagnostic` |

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
| `poly_range_start` | `uint64_t` | 0 | First a-index for this node (0 = natural start; set from `WORK_ASSIGN`/`CHUNK_ASSIGN`) |
| `poly_range_count` | `uint64_t` | 0 | Number of a-values in this node's contiguous range |

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

When `autotune_enabled` or mode is `AUTOTUNE_ONLY`, the orchestrator instantiates `AutotuneController(autotune_config, config_, f_data_)` and calls `run()`. The controller runs up to 4 stages (projection, kernel params, runtime estimation, sieve params) using truncated sieve probes with `TruncatedSieveRun()`. Results are written back to `config_` (stages run, confidence score). In `AUTOTUNE_ONLY` mode, the pipeline returns immediately after autotune completes.

Autotune probes use `silent = true` on sub-orchestrators to suppress log header/footer noise.

### Sieve Init Helpers

Three private helpers are shared by `SieveStage()` and `TruncatedSieveRun()` to eliminate initialization duplication:

- **`initPostProcessorConfig(accum_override)`** — Builds and returns a `PostProcConfig` from `config_`/`f_data_`. Sets accumulation buffer (CLI override → `accum_override` → default `max(4096, batch_size·2048)`), partial buffer (CLI override → 1× accum, floor-clamped to accum when LP active), persistent buffer (CLI override → `target·2 + accum`, floor-clamped if too small). `TruncatedSieveRun` passes `accum_override = 8192` for frequent probe granularity.
- **`initLargePrimes()`** — Configures `LargePrimeConfig` (witness capacity, hash bits, combined output, sort mode) from `config_`, constructs `largeprime_`, and calls `initiate()`. No-op if `lp1_bound == 0`.
- **`logBufferWarnings()`** — Emits fire-once near-full warnings (accum ≥ 90%, partial ≥ 80%, persistent ≥ 95%, witness ≥ 85%) and per-delta LP overflow warnings (slab, witness, output). Called from both batch and legacy sieve loops.

### Stage 2: SieveStage (three dispatch paths)

**Initialization (shared):**
1. `DeviceSievingController::initiate(f_data_)`. If `lp1_bound > 0`, calls `setThresholdOverride(lp1_bound)`.
2. If `sieve_batch_size > 0`, calls `setSievingBatchSize` and `allocateBatchBuffers`.
3. `PARAM_TEST` short-circuit: calls `siever_->runParamTest(f_data_)` then exits.
4. **sasGridDim auto-correction** (belt-and-suspenders): when LP is active with custom params, computes `min_sas = ceil(subCubeSize × numIntervals / 64)`, rounds up to next power-of-2, and raises `params[6]` if below minimum.
5. **sasBlockDim cap**: in legacy mode (`sieve_batch_size == 0`), caps `params[7]` to 1024 to match `__launch_bounds__(1024)` on sieveAndScanKernel.
6. Config loading: `loadPartialCustomConfig(params...)` if `useParams`, else `loadStandardConfig`. Validates and calls `loadData`.
7. **Preflight kernel launch check** (`preflightKernelLaunch`): when `useParams`, rejects infeasible kernel launch configurations before entering the sieve loop. Throws `std::runtime_error` on failure.
8. `initPostProcessorConfig()` builds `PostProcConfig`; persistent buffer is floor-clamped to `target_relations + accum` if undersized.
9. `initLargePrimes()` initializes LP variant (no-op if `lp1_bound == 0`).

**PATH 1: Batch** (`sieve_batch_size > 0`)

Zero-sync GPU pipeline with CUDA event DAG. Adaptive convergence:
- `dedup_margin = max(256, 3% × target_relations)`, `relation_cap = target_relations + dedup_margin`
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

**PATH 2: Legacy + LP** (`sieve_batch_size == 0`, `lp1_bound > 0`)

Host-driven loop: per step `siever_->updateState`, `sieveFullCube`, `postprocessor_->accumulate`. When buffer full: `processBufferedCandidates`, `consolidateToPersistent`, `largeprime_->processAndCommit`, `resetPartialBatch`. Async telemetry from `largeprime_->getTelemetry()` on generation-ticket change drives `logSieveProgress`. Exits when `getPersistentCount() >= target_relations`.

**PATH 3: Legacy no-LP** (`sieve_batch_size == 0`, `lp1_bound == 0`)

Same host-driven loop; when buffer full: `processBufferedCandidates`, `consolidateToPersistent`. Logs progress every 200 new relations. Exits when `getPersistentCount() >= target_relations`.

**Post-loop (all paths):**
Legacy paths: `postprocessor_->flush()`, optional final LP commit. All paths: clear sticky CUDA error via `cudaGetLastError`, `deduplicatePersistentBatch`, `d_persistent->moveToHost(host_relations_soa_)`.

Cleanup: `siever_->clearSievingBuffers()`, `postprocessor_->clearBuffers()`, `largeprime_->clearBuffers()` (if active). Persistent batch is **not** cleared here — `MatrixStage` reads it.

### Stage 3: MatrixStage

Two construction paths:
- **Normal**: `matrix_constructor_->constructFromSoA(postprocessor_->getPersistentBatch()->get_view(), count, csr_matrix)` using the live device batch
- **LINALG_ONLY fallback**: allocates `RelationBatch temp_batch`, calls `temp_batch.uploadFromHost(host_relations_soa_)`, then constructs from that view

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

## SoA Disk Serialization

Binary format (anonymous namespace helpers `serialize_soa` / `deserialize_soa`):

| Offset | Content |
|--------|---------|
| 0 | Magic `MPQS_SOA` (8 bytes) |
| 8 | `num_relations` (uint64) |
| 16 | `num_factors` (uint64) |
| 24+ | 7 flat vectors, each preceded by uint64 element count |

Vectors in order: `sqrt_Q`, `signs`, `val_2_exps`, `large_primes`, `factor_offsets`, `factor_indices`, `factor_counts`.

File path: `{work_dir}/relations.soa`. BW checkpoints: `{work_dir}/bw*`.

## Execution Modes

| Mode | Stages Executed | Disk I/O |
|------|-----------------|----------|
| `FULL_PIPELINE` | Tuning [+ Autotune] + Sieve + Matrix + LinAlg + Sqrt | Optional save after sieve |
| `SIEVE_ONLY` | Tuning + Sieve | Save required (writes `.soa`), then returns |
| `LINALG_ONLY` | Tuning + Matrix + LinAlg + Sqrt | Load required (reads `.soa`) |
| `MATRIX_ONLY` | Load v2 relations → Matrix → BW → Sqrt | Load required (reads `.v2`), no sieve |
| `SQRT_ONLY` | *(not implemented)* | Logs error and returns immediately — no disk loader for kernel solutions or factor base |
| `PARAM_TEST` | Tuning + Sieve init (calls `runParamTest`, then returns) | None |
| `AUTOTUNE_ONLY` | Tuning + Autotune | None (prints results, returns) |

Note: `LINALG_ONLY` always runs `TuningStage` to reconstruct the factor base needed for matrix column indexing.

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
- See `tools/cluster/rsa140_a100_4node_pc2.sbatch` (production) and
  `tools/cluster/rsa130_a100_2node_resume_smoke_pc2.sbatch` (2-node resume smoke, ~20 min).

## Internal State

| Member | Type | Purpose |
|--------|------|---------|
| `config_` | `MPQSConfig` | Runtime configuration (mutated during tuning/autotune) |
| `f_data_` | `sieve::factoringData` | Factor base, polynomial state |
| `host_relations_soa_` | `structures::HostRelationBatch` | Downloaded relations (SoA) |
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

## Dependencies

Links (PUBLIC): `mpqs_autotune`, `mpqs_common`, `mpqs_sieve`, `mpqs_postproc`, `mpqs_largeprimes`, `mpqs_matrix`, `lingen_ops`, `mpqs_sqrt`, `cudampqs_build_flags`. The cluster include directory (`src/cluster/`) is added via `target_include_directories`; cluster types are forward-declared in `orchestrator.h`. `cudampqs_cluster` is linked at the binary level (`tests/CMakeLists.txt`), not inside `mpqs_orchestrator` itself.
