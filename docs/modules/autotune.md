# Autotune Module (`src/autotune/`)

Automatic parameter optimization for the MPQS pipeline. Determines optimal factor base bound (F), sieve interval (M), large prime bound (L), and GPU kernel launch parameters via a 4-stage optimization loop: projection from history, kernel micro-benchmarking, runtime estimation from truncated sieve probes, and joint (F,L) convex optimization with gradient descent.

Namespace: `mpqs::autotune`.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `autotune.h` | 128 | `AutotuneController` class, `AutotuneConfig`, `AutotuneResult` |
| `autotune.cpp` | 922 | Controller implementation: 4-stage loop, history I/O, convergence, buffer recommendations |
| `autotune_types.h` | 126 | Shared types: `RuntimeEstimate`, `SieveOptimizationResult`, `SieveSearchBounds`, `L_function()` |
| `autotune_projection.h` | 92 | `ParameterProjector` class, `ProjectedParams` struct |
| `autotune_projection.cpp` | 484 | 4-tier projection cascade: exact match, interpolation, extrapolation, theory fallback |
| `kernel_param_optimizer.h` | 70 | `optimizeKernelLaunchParams()`, candidate value arrays, heuristic defaults |
| `kernel_param_optimizer.cpp` | 181 | Seeded coordinate descent over 8 kernel launch parameters |
| `kernel_launch_validator.h` | 121 | `KernelLaunchValidator` class, `PreflightResult`, `Params8`, `ParamIndex` enum |
| `kernel_launch_validator.cpp` | 390 | 6-check validation pipeline, enumeration, preflight with LP-aware auto-correction |
| `runtime_estimator.h` | 33 | `estimateRuntime()` free function |
| `runtime_estimator.cpp` | 269 | Truncated sieve probe via ephemeral orchestrator, ETA extrapolation, confidence scoring |
| `sieve_optimizer.h` | 181 | `SieveParameterOptimizer` class with joint (F,L) convex optimizer |
| `sieve_optimizer.cpp` | 714 | Three-phase optimizer: L-sweep (A), 3×3 grid (B), gradient descent (C) |
| `cost_models.h` | 28 | `estimateMatrixTime()`, `estimateLinalgTime()`, `deriveSieveSearchBounds()` |
| `cost_models.cpp` | 50 | Heuristic power-law cost models calibrated on RTX 5070 Ti |
| `autotune_history.h` | 116 | `HistoryStore` class, `HistoryEntry` struct, `sha256_hex()` |
| `autotune_history.cpp` | 789 | JSON persistence, FIPS 180-4 SHA-256, k-nearest-neighbor lookup with F5 two-pass filter |
| `auto_apply.h` | 70 | `AutoApplyController` class, `AutoApplyResult` struct |
| `auto_apply.cpp` | 360 | History-based parameter application without GPU probes |
| `benign_history.h` | 60 | `BenignHistoryStore` class, `BenignHistoryEntry` struct |
| `benign_history.cpp` | 369 | Cross-GPU hardcoded parameter defaults (17 entries: 12 desktop + 5 Jetson Orin), JSON persistence |
| `json_reader.h` | ~322 | Shared cursor-based recursive-descent JSON parser; used by `HistoryStore::load` and `BenignHistoryStore::load` |
| `memory_estimator.h` | ~45 | `memory_costs` namespace (per-element byte costs for on-device buffers), `kMinPartialBufferSize` floor constant |
| `CMakeLists.txt` | 28 | Static library `mpqs_autotune`, links `mpqs_common`, `mpqs_sieve` |

Total: ~5,581 lines across 23 source files (21 previously documented + `json_reader.h` + `memory_estimator.h`).

## Architecture Overview

```
AutotuneController::run()
  -> loadHistory()               -- short-circuit if exact match with confidence > 0.95
  for iter = 0 .. max_iterations:
    -> Stage 0: Projection       -- initial parameter guess from history or theory
    -> Stage 1: Kernel Params    -- micro-benchmark 8-tuple via coordinate descent
    -> Stage 2: Runtime Est.     -- truncated sieve probe -> ETA extrapolation
    -> Stage 3: Sieve Params     -- joint (F,L) optimizer: L-sweep, 3x3 grid, gradient descent
    -> convergence check         -- |delta(total_est)| / prev < 2% -> stop
  -> saveHistory()
```

The controller iterates up to `max_iterations` (default 2) times through Stages 0-3. Each stage refines a different parameter subspace. The loop terminates early on convergence (< 2% improvement), timeout (default 300s), or history hit.

Autotuning is skipped for N < 70 digits (`kMinAutotuneDigits = 70`, ~232 bits) — heuristic parameters are sufficient at this scale.

## Auto-Apply: Zero-Probe History Application

Even without `--autotune`, the pipeline implicitly loads history via `AutoApplyController` after `TuningStage()` on every run (disable with `--autotune_no_history`). Auto-apply performs zero GPU probes — it cascades through 4 tiers to select parameters:

| Tier | Source | Confidence | Description |
|------|--------|------------|-------------|
| 1 | Exact history match | entry's confidence | N_hash found in per-GPU history |
| 2 | K-nearest neighbors | 0.6 | IDW interpolation by bit-length (3 neighbors) |
| 3 | Benign history | entry's confidence | Cross-GPU baseline from digit-range lookup |
| 4 | TuningStage defaults | — | Leave heuristic defaults unchanged |

### isPinned Protection

The orchestrator's 6-tier small-N parameter table (bits < 200) pins its F and M selections via `config_.pinned_params["fb_bound"] = true`. This prevents auto-apply from overriding known-good small-N parameters with stale history entries from larger inputs. All auto-apply and autotune parameter assignments check `config_.isPinned(field)` before writing.

### M Propagation

When auto-apply changes M (sieve_bound), the orchestrator propagates the change to `f_data_.M` and recomputes `a_target = sqrt(2N) / M` to maintain polynomial coefficient consistency. This triggers `init_a_factors()` to regenerate polynomial coefficients.

### Buffer Cascade

Auto-apply also merges buffer parameters (witness capacity, partial buffer, accum buffer) using the same tier cascade, with a minimum partial buffer floor of 65,536 and a batch-mode floor of 8× accum.

## Stage 0: Parameter Projection (`autotune_projection.cpp`)

**Purpose:** Produce an initial parameter guess from historical data or theoretical heuristics, avoiding cold-start from scratch.

**Class:** `ParameterProjector` -- constructed from a `HistoryStore`, fits OLS models in the constructor.

### Cold-Start Enrichment

When the history is empty, Stage 0 injects a synthetic `HistoryEntry` from the benign history store, seeded with the matching digit-range record. This ensures the projector always has at least one data point for interpolation.

### 4-Tier Projection Cascade

| Tier | Source | Confidence | Condition |
|------|--------|------------|-----------|
| 1 | Exact match | `entry.confidence` | N_hash found in history |
| 2 | Interpolation | `min(c_lo, c_hi) * gap_factor * sparsity_factor` | Bracketing entries exist (one below, one above target bits) |
| 3 | Extrapolation | `avg_conf * exp(-delta/30)` | OLS model valid, delta <= 40 bits from nearest |
| 4 | Theory fallback | 0.0 | No or insufficient history |

**L-space coordinate:** All interpolation/extrapolation operates in L-space:
$$u(\text{bits}) = \sqrt{\text{bits} \cdot \ln 2 \cdot \ln(\text{bits} \cdot \ln 2)}$$
This is the natural interpolation variable for subexponential scaling (the quadratic sieve complexity function).

**OLS model (Tiers 2-3):** Fits `ln(param) = ln_c + alpha * u` via ordinary least squares on history entries. Separate models for `fb_bound` and `lp1_bound`.

**Theory fallback (Tier 4):** Mirrors the 8 bit-length ranges from `primeAlgorithms.cu:determineParams()`:

| Bits | F (fb_bound) | Notes |
|------|-------------|-------|
| < 180 | 60,000 | Small composites |
| 180-219 | 350,000 | |
| 220-249 | 900,000 | |
| 250-339 | 3,000,000 | RSA-100 (330 bits) |
| 340-369 | 4,000,000 | RSA-110 (364 bits) |
| 370-399 | 5,000,000 | RSA-120 (397 bits) |
| 400-439 | 7,500,000 | RSA-130 (430 bits) |
| >= 440 | 10,000,000 | RSA-140+ |

Default M = 2^18 (262144). LP seeded at `10 * F` for inputs >= 200 bits.

**LP feedback loop breaker:** For N >= 290 bits (~88 digits), if projection yields L=0, Stage 0 overrides to `L = F * 50`. This prevents a self-reinforcing loop where LP is empirically beneficial but probes can't observe the benefit (birthday-paradox scaling) so L=0 persists in history.

**Kernel param projection:** For GPU-matching neighbors, constant params (indices 0,1,2,3,5,7) use modal values; variable params (indices 4,6 -- `metaGridDim`, `sasGridDim`) use inverse-distance-weighted interpolation in bit-length.

**Search radius:** Maps confidence to search bounds for Stage 3:
$$r(\text{conf}) = 0.05 + 0.45 \cdot (1 - \text{conf})$$
High confidence (1.0) -> 5% radius; zero confidence -> 50% radius.

## Stage 1: Kernel Parameter Optimization (`kernel_param_optimizer.cpp`)

**Purpose:** Find the optimal 8-parameter kernel launch configuration via seeded coordinate descent with GPU micro-benchmarking.

### The 8 Parameters (`Params8`)

| Index | Name | Symbol | Kernel Config | Candidate Values |
|-------|------|--------|--------------|-----------------|
| 0 | `subCubeSize` | P_SUB_CUBE_SIZE | `gs_conf.num_polysPerSieveCall` | {128, 256, 512, 1024} |
| 1 | `numIntervals` | P_NUM_INTERVALS | `gs_conf.num_sievingBlocksPerSieveCall` | {4, 8, 16, 32} |
| 2 | `polyBlockSize` | P_POLY_BLOCK_SIZE | `gms_conf.polyBlockSize` | {1, 2, 4, 8, 16, 32} |
| 3 | `blocksPerCycle` | P_BLOCKS_PER_CYC | `gms_conf.num_activeBlocksPerCycle` | {4, 8, 16, 32} |
| 4 | `metaGridDim` | P_META_GRID_DIM | `gms_conf.num_threadBlocks` | {32, 64, 128, 256} |
| 5 | `metaBlockDim` | P_META_BLOCK_DIM | `gms_conf.num_threadsPerBlock` | {256, 512, 1024} |
| 6 | `sasGridDim` | P_SAS_GRID_DIM | `ss_conf.num_threadBlocks` | {32, 128, 256, 512} |
| 7 | `sasBlockDim` | P_SAS_BLOCK_DIM | `ss_conf.num_threadsPerBlock` | {256, 512, 1024} |

All values must be powers of 2.

### Algorithm

1. **Phase 1 (Seed):** Start from `HEURISTIC_DEFAULTS = {512, 8, 4, 8, 256, 1024, 256, 1024}`. Clamp to feasible region (e.g., `subCubeSize <= 2^(shc_dim-1)`, `blocksPerCycle <= numIntervals`). Validate; if infeasible, fall back to first valid enumerated config.

2. **Phase 2 (Weak params):** Coordinate descent over weakly-convergent parameters in order: `sasGridDim` (6), `metaGridDim` (4), `polyBlockSize` (2). Each candidate is validated, then timed via `DeviceSievingController::evaluateConfig()` with 2 subcubes (coarse). Improvement threshold: 2% (`EPSILON`).

3. **Phase 2.5 (Thorough, optional):** If `thorough=true`, also sweeps strongly-convergent params: `sasBlockDim` (7), `metaBlockDim` (5), `numIntervals` (1), `blocksPerCycle` (3), `subCubeSize` (0).

4. **Phase 3 (Verify):** Re-evaluate winning config with 4 subcubes (fine) for more accurate timing.

**Post-optimization:** Clears sticky CUDA errors (`cudaDeviceSynchronize()` + `cudaGetLastError()`) that may accumulate from failed kernel launches during benchmarking. Runs a defense-in-depth preflight check on the winning config before applying to the pipeline.

### Skip Logic

Stage 1 is skipped on iterations > 0 when both F and M have changed by < 10% since the last Stage 1 run, since kernel launch parameters are weakly sensitive to small F/M changes.

## Kernel Launch Validator (`kernel_launch_validator.cpp`)

**Purpose:** Pure-arithmetic feasibility checker for the 8-parameter tuple. No GPU launches, no CUDA streams. Only the constructor calls `cudaGetDeviceProperties()`.

### 6-Check Validation Pipeline

Ordered from cheapest to most expensive:

| Check | Rule | Source |
|-------|------|--------|
| 1. Power-of-2 | All 8 params are powers of 2 | `validateConfigs:864-872` |
| 2. Arithmetic | `subCubeSize <= 2^(shc_dim-1)`, `blocksPerCycle <= numIntervals`, `metaGridDim * polyBlockSize <= subCubeSize`, derived `num_polyBlocksPerTB` is pow2, exact decomposition equalities | `validateConfigs:874-885` |
| 3. Shared memory | `blocksPerCycle * polyBlockSize * 4B <= maxSharedMem`, `sievingBlockSize + 3*1024*4B <= maxSharedMem` | `validateConfigs:889-890` |
| 4. Global memory | `subCubeSize * numIntervals * globalBucketSize * 8B <= 3/4 * totalGlobalMem` | `validateConfigs:888` |
| 5. Device limits | `metaBlockDim`, `sasBlockDim <= maxThreadsPerBlock`; grid dims <= `maxGridSize[0]` | CUDA runtime |
| 6. Non-zero derived | `num_sievingBlockBatches > 0`, `num_subCubes > 0`, `num_polyBlocksPerTB > 0`, `num_metaSieveCycles > 0` | `validateConfigs:892-895` |

**`enumerateValidConfigs()`:** Brute-force over all candidate value combinations (~50k iterations), filtering through the validator. Returns valid configs in < 1 ms.

### Preflight Check

`preflightKernelLaunch()` is the entry point for runtime validation. Two overloads:

1. **Raw `Params8` overload:** Builds `SieveConstants` from device properties and factoring dimensions, validates via `KernelLaunchValidator`.

2. **`MPQSConfig&` overload (mutating):** Includes two auto-correction passes before validation:
   - **LP-aware sasGridDim floor:** When `lp1_bound > 0`, computes `min_sas = ceil_pow2(subCubeSize * numIntervals / 64)` to prevent candidate buffer overflow. Auto-corrects `params[6]` upward if needed.
   - **sasBlockDim `__launch_bounds__` cap:** When `sieve_batch_size == 0` (legacy mode), caps `params[7]` at 1024 to match the `__launch_bounds__(1024)` annotation on `sieveAndScanKernel`.

## Stage 2: Runtime Estimation (`runtime_estimator.cpp`)

**Purpose:** Evaluate a parameter set by running a truncated sieve probe on an ephemeral orchestrator, extrapolating total pipeline time.

### Probe Mechanism

1. Clone `base_config`, set `silent=true`, create temp directory, force legacy loop (`sieve_batch_size=0`).
2. Clear sticky CUDA errors.
3. Construct ephemeral `MPQSOrchestrator`, run `TuningStage()`.
4. Preflight check: skip probe if kernel params infeasible (returns sentinel with `total_est_sec = 1e9`).
5. Run `TruncatedSieveRun(truncation_frac, eta_threshold, min_samples)`:
   - Default truncation: 12% of target relations.
   - ETA convergence: stop early if relative spread < 5% across 3+ samples.
6. Extract telemetry; destroy orchestrator (RAII); clean up temp directory.

### Time Estimation

- **Sieve time:** If ETA reliable (quadratic fit with positive curvature, >= 6 samples): `total = eta_remaining + elapsed`. Otherwise: linear extrapolation from throughput.
- **LP yield correction:** Birthday-paradox projection model. Projects total witnesses at full sieve scale from observed linear influx rate, then estimates LP contribution via `E[matches] ~ W^2 / (2*B)` where W = projected witnesses and B = hash table size. LP fraction clamped to [0, 0.35]; sieve time discounted by `(1 - lp_frac)`.
- **Matrix time:** `T_matrix = 9.49e-7 * fb_size^1.1` (calibrated: 0.3s at fb_size=100k on RTX 5070 Ti).
- **LinAlg time:** `T_linalg = 8.5e-6 * fb_size^1.17` (empirical fit from 75d-95d campaign + RSA-100; 3% median error vs. prior N^2 model at 173%).
- **Total:** `T = T_sieve + T_matrix + T_linalg`.

### Buffer Recommendations

Each probe produces buffer sizing recommendations based on observed telemetry:

| Buffer | Recommendation Logic |
|--------|---------------------|
| Witness capacity | `projected_witnesses * 1.3`, rounded to power-of-2, capped at 2^24 |
| Partial buffer | Scaled from default 4M by `projected_witnesses / default_cap`, capped at 4× |
| Accum buffer | `accum_peak * 1.5`, rounded to power-of-2, min 64K, capped at 2^20 |
| LP output | Doubled to 64K if output overflows detected |
| Hash bits | `current_bits + 2` if slab overflows detected (4× more directory buckets) |

Stage 3 merges its buffer recs with Stage 2's via `max()` to avoid regression from shorter probes.

### Confidence Scoring

| Condition | Penalty |
|-----------|---------|
| < 6 ETA samples | x0.6 |
| < 3 ETA samples | = 0.0 (unusable) |
| > 100 overflow events | x0.8 |
| < 5% progress, not converged early | x0.5 |

## Stage 3: Joint (F,L) Convex Optimization (`sieve_optimizer.cpp`)

**Purpose:** Minimize total estimated pipeline time over (F, M, L) via a three-phase joint optimizer with warm-start adaptation.

### Algorithm Overview

```
optimize(warm_start, bounds):
  -> M optimization (exhaustive over powers of 2)
  -> Phase A: 1D L-sweep at fixed F (multiplicative search, α=1.2)
  -> Phase B: 3×3 grid exploration in (F,L) space
  -> [Phase C: log-space gradient descent with learning rate decay]
```

### M Optimization

Enumerate all powers of 2 in `[2^M_min_log2, 2^M_max_log2]` at the warm-start (F, L). Fix M at the best-performing value. M is optimized first because it is cheaply searchable (5 candidates at most) and weakly coupled to F and L.

### Phase A: 1D L-sweep

Sweeps L at fixed F to find the optimal LP bound:
1. Evaluate warm-start L₀
2. Always evaluate L=0 (LP disabled) as an alternative
3. If starting from L=0, trial `L = F×50` (canonical LP bound)
4. Directional sweep with multiplicative steps (α=1.2), accelerating
5. Stop when convexity violated (next step worsens T)

**L constraint:** L capped at `min(bounds.lp_hi, F²)`. Sub-minimum nonzero L (below `L_min = max(1M, 10*F)`) snaps to L=0.

### Phase B: 3×3 Grid Exploration

Builds a grid around (F_center, L_center) with `effective_alpha = 1 + (alpha-1) * delta_scale`:
- F axis: `[F/eff_alpha, F, F*eff_alpha]`, clamped to `[bounds.fb_lo, F_max_effective]`
- L axis: `[L/eff_alpha, L, L*eff_alpha]` (or `[0, F×50, F×50*eff_alpha]` if L=0)

All 9 grid points evaluated via `cachedEstimate()`. Optional edge extension probes one step beyond if the minimum sits at a grid edge.

**F runaway prevention:** `F_max_effective = min(bounds.fb_hi, f_max_multiplier × F_heuristic)` where `f_max_multiplier = 2.0`. Prevents the optimizer from pushing F beyond 2× the heuristic default, which causes enormous BW matrices with catastrophic LinAlg cost.

### Phase C: Gradient Descent

Log-space gradient descent over (F, L):
1. Estimate gradient from the current 3×3 grid via central differences averaged over rows/columns
2. Convert to dimensionless gradient: `g_F = F * dT/dF`, `g_L = L * dT/dL`
3. Update: `F_new = F * exp(-η * g_F / ||g||)`, `L_new = L * exp(-η * g_L / ||g||)`
4. Enforce constraints (F bounds, L ≤ F², L ≥ L_min)
5. If improvement: accept; else halve learning rate
6. Rebuild 3×3 grid with decaying resolution (`delta_scale *= alpha_decay`)
7. Decay learning rate (`η *= lambda`)

**Convergence:** Exit when averaged gradient norm / T_best < epsilon (2%) over a sliding window of `grad_avg_window = 3` iterations, or after `max_gd_iterations = 8`.

**Warm-start adaptation:** Phase C behavior adapts to projection confidence:
- confidence ≥ 0.8 and relative gradient < 5%: skip Phase C entirely
- confidence ≥ 0.5 and relative gradient < 10%: reduce to 3 iterations, smaller step
- Otherwise: full exploration

### Probe Caching

`cachedEstimate()` memoizes `(F_rounded, M, L_rounded) -> RuntimeEstimate`, avoiding redundant truncated sieve probes. Quantization: F to nearest 1000, L to nearest 1,000,000.

### Budget Controls

- `max_total_probes = 40` (raised from 12 for the joint optimizer)
- `wall_clock_timeout_sec` = remaining autotune budget (skips if < 30s)
- Convergence penalty: if not converged, `confidence *= 0.5`

### Search Bounds (`deriveSieveSearchBounds()`)

Derived from the auto-tuned or projected F:

| Parameter | Lower | Upper |
|-----------|-------|-------|
| fb_bound | `0.3 * F` | `min(3.0 * F, 2.0 * F_heuristic)` |
| M (log2) | 15 (32768) | 19 (524288) |
| lp1_bound | 0 (LP-off candidate) | `min(50*F, 500M)` |

The M upper bound is capped at 2^19 to prevent GPU hangs at very large sieve intervals. The F upper bound is capped at 2× F_heuristic to prevent runaway.

## Cost Models (`cost_models.cpp`)

Power-law heuristics calibrated on RTX 5070 Ti benchmarks:

| Stage | Model | Calibration Point |
|-------|-------|-------------------|
| Matrix | `T = 9.49e-7 * fb_size^1.1` | T(100k) = 0.3s |
| LinAlg (BW) | `T = 8.5e-6 * fb_size^1.17` | T(301k) ≈ 25s (RSA-100) |

The LinAlg model uses exponent 1.17 (sub-quadratic) reflecting empirical BW scaling: Stage 2 (lingen) dominates with O(L^1.65), GPU SpMM throughput saturates. This achieves 3% median error vs. the prior N^2 model's 173%.

## Complexity Function

The L-function `L(N)` for the quadratic sieve:

$$L(N) = \exp\bigl(\sqrt{\ln N \cdot \ln\ln N}\bigr)$$

Used for parameter scaling: `F ~ L(N)^alpha` where `alpha ~= 0.707` theoretically. Available as `L_function(bits)` and `L_alpha(bits, alpha)` in `autotune_types.h`.

## History System (`autotune_history.cpp`)

### Storage

JSON file with version 1 schema. Entries keyed by `N_hash_sha256` (FIPS 180-4 SHA-256, self-contained implementation). Atomic writes via temp file + rename.

### HistoryEntry Fields

| Group | Field | Type | Description |
|-------|-------|------|-------------|
| Identity | `N_decimal`, `N_hash_sha256` | string | Full N and its SHA-256 |
| | `digit_count`, `bit_length` | uint32 | N size metrics |
| Params | `fb_bound`, `sieve_bound`, `lp1_bound` | uint32/64 | Optimal sieve parameters |
| | `kernel_params[8]` | uint32[8] | Optimal kernel launch tuple |
| | `recommended_witness_capacity/partial_buffer/accum_buffer` | uint64 | Buffer sizing recommendations |
| Perf | `sieve_time_sec`, `total_time_sec`, `relations_per_sec` | double | Measured performance |
| | `total_relations`, `lp_witnesses`, `lp_combined_relations` | uint64 | Relation counts |
| | `witness_peak/capacity/fill_pct`, `overflow_events` | mixed | Buffer telemetry |
| | `accum_peak`, `partial_peak`, `persistent_peak` | uint64 | Buffer high-water marks |
| Env | `gpu_name`, `gpu_compute_capability`, `cuda_version` | string | Hardware fingerprint |
| Meta | `timestamp`, `autotune_stages_run`, `confidence` | mixed | Provenance |

### Quality Filters (Applied at Load Time)

| Filter | Rule | Action |
|--------|------|--------|
| F1 | Degenerate kernel params (128,4,4,4,...) | Remove entry |
| F2 | Partial buffer < 65536 | Clamp to 65536 |
| F3 | Confidence < 0.3 | Remove entry |
| F5 | Heuristic-only entries | Deprioritized in K-nearest (two-pass filter) |

### K-Nearest Lookup with F5 Two-Pass Filter

`findKNearest(bit_length, k)` implements a two-pass quality filter:

1. **Pass 1 (strict):** Separate entries into autotuned (non-empty `stages_run` or confidence > 0.5) and heuristic-only pools. Rank autotuned entries by bit-length distance.
2. **Pass 2 (relaxation):** If fewer than K autotuned entries pass, backfill from heuristic entries sorted by distance. This prevents sparse history from returning zero neighbors while still preferring validated data.

### Lookup Operations

| Method | Complexity | Description |
|--------|-----------|-------------|
| `findExact(N_hash)` | O(1) | Hash map lookup |
| `findKNearest(bits, k)` | O(n) | k-nearest by bit-length distance (two-pass F5 filter) |
| `findByBitRange(lo, hi)` | O(n) | Linear scan (entries sorted by bit_length) |
| `upsert(entry)` | O(n log n) | Insert/replace + re-sort |

**Upsert policy:** Replace existing entry if new entry has higher confidence, or equal confidence with lower `total_time_sec`.

### History Short-Circuit

In `AutotuneController::run()`, if an exact match with confidence > 0.95 is found and the GPU name matches, all cached params (including kernel params and buffer recommendations) are applied directly and the entire optimization loop is skipped (`TermReason::HISTORY_HIT`).

## Benign History (`benign_history.cpp`)

Cross-GPU parameter defaults keyed by digit range. Provides fallback parameters when no per-GPU history exists.

### Hardcoded Defaults (17 entries: 12 desktop + 5 Jetson Orin, 27-115 digits)

Desktop entries (RTX-class, `conf=1.0` for validated rows; take priority over Jetson entries on non-Jetson hardware due to load-order invariant in `findByDigits`):

| Digit Range | Bits | F | M | L | Confidence |
|-------------|------|---|---|---|------------|
| 27-31 | 96 | 25,000 | 16,384 | 0 | 0.5 |
| 37-41 | 129 | 100,000 | 131,072 | 0 | 0.5 |
| 47-51 | 162 | 120,000 | 131,072 | 0 | 0.5 |
| 57-61 | 195 | 200,000 | 262,144 | 0 | 0.5 |
| 62-66 | 212 | 500,000 | 262,144 | 0 | 0.5 |
| 67-71 | 228 | 300,000 | 262,144 | 0 | 1.0 |
| 72-76 | 245 | 700,000 | 65,536 | 0 | 1.0 |
| 77-81 | 261 | 700,000 | 65,536 | 0 | 1.0 |
| 82-86 | 276 | 1,500,000 | 262,144 | 0 | 1.0 |
| 87-95 | 300 | 3,000,000 | 262,144 | 300M | 1.0 |
| 95-105 | 332 | 7,000,000 | 262,144 | 1T | 1.0 |
| 105-115 | 364 | 9,000,000 | 262,144 | 1T | 1.0 |

Jetson Orin Nano (SM 8.7) entries (`conf=0.5`; appended after desktop block so desktop entries match first on RTX-class hardware):

| Digit Range | Bits | F | M | L | Confidence |
|-------------|------|---|---|---|------------|
| 67-71 | 233 | 300,000 | 262,144 | 0 | 0.5 |
| 77-81 | 266 | 700,000 | 65,536 | 0 | 0.5 |
| 82-86 | 282 | 1,500,000 | 262,144 | 0 | 0.5 |
| 87-95 | 299 | 3,000,000 | 262,144 | 100M | 0.5 |
| 95-105 | 332 | 7,000,000 | 262,144 | 500M | 0.5 |

These entries cover composites from ~27 digits (96 bits) through ~115 digits (364 bits, RSA-110) with no digit gaps in the desktop block. Desktop entries validated by the small-N campaign sweep (RTX 5070 Ti); RSA-110 entry validated at 1048s full pipeline. Jetson Orin entries are extrapolated from desktop params with reduced LP bounds for 1 MB L2 / 8 GB memory.

### Lookup

- `findByDigits(digit_count)` — range match on `[digit_count_lo, digit_count_hi]`
- `findByBits(bit_length)` — nearest by `bit_length` field
- `upsert(entry)` — replace if digit range overlaps

JSON-backed with atomic save via temp file + rename.

## Convergence and Termination

### Per-Iteration Convergence

After each iteration, compares `total_est_sec` with previous iteration:

```
improvement = |prev - current| / prev
if improvement < convergence_threshold (2%):
    terminate with CONVERGED
```

### Termination Reasons

| Reason | Condition |
|--------|-----------|
| `CONVERGED` | Inter-iteration improvement < 2% |
| `MAX_ITER` | Reached `max_iterations` (default 2) |
| `TIMEOUT` | Wall-clock exceeds `timeout_sec` (default 300s) |
| `HISTORY_HIT` | Exact N match with confidence > 0.95 |
| `ERROR` | Unrecoverable error |

## Integration with Orchestrator

The orchestrator invokes autotuning during `TuningStage()` when enabled. The `AutotuneController` takes mutable references to both `MPQSConfig` and `factoringData`, modifying them in place:

1. `applyToConfig(fb, M, lp)` sets `pipeline_config_.fb_bound`, `sieve_bound`, `lp1_bound`. If F changed, calls `regenerateFactorBase()` (which runs `generateFactorBase()` + `init_a_factors()`).
2. Stage 1 writes optimized kernel params to `pipeline_config_.params[0..7]` and sets `pipeline_config_.useParams = true`.
3. After autotuning, the orchestrator proceeds with the tuned config through Sieve, Matrix, LinAlg, and Sqrt stages.

### Orchestrator-Side Adaptations

The orchestrator applies several small-N and performance adaptations that complement autotuning:

**Adaptive BW block sizes (LinearAlgebraStage):** When not user-pinned:
- Matrix dimension < 4,000: m=n=64
- Matrix dimension < 16,000: m=n=128
- Matrix dimension ≥ 16,000: m=n=256 (default)

**Skip BW SpMM autotuning:** For matrices with < 100,000 columns, SpMM autotuning is disabled (`autotune_tune_spmm = false`). The autotuning overhead dominates computation at this scale.

**Accumulate buffer scaling for small N:** When `target_relations < 16,384` and no explicit buffer size is set, the accumulate buffer is scaled to `max(4096, target_relations * 4)` rounded to the next power of 2. This prevents catastrophic over-collection where the default 524K buffer overshoots a 1.5K-relation target by 200×.

### AutotuneConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_stage0..3` | bool | true | Enable/disable individual stages |
| `max_iterations` | uint32 | 2 | Outer loop iterations |
| `timeout_sec` | double | 300.0 | Wall-clock budget (seconds) |
| `convergence_threshold` | double | 0.02 | 2% relative improvement to stop |
| `thorough` | bool | false | Sweep strongly-convergent kernel params |
| `truncated_run_target_pct` | double | 0.12 | Probe truncation fraction (12%) |
| `truncated_run_min_samples` | uint32 | 10 | Minimum ETA samples per probe |
| `prefer_legacy_loop` | bool | true | Force legacy mode for probes |
| `history_file` | string | "" | Path to per-GPU JSON history file |
| `benign_history_file` | string | "" | Path to cross-GPU benign history (empty = auto) |
| `load_history` / `save_history` | bool | true | Enable history I/O |
| `verbose` | bool | false | Extra logging |
| `candidates_file` | string | "" | Path to candidates.txt for bootstrap |
| `bootstrap` | bool | false | Bootstrap mode (--autotune_bootstrap) |

## Recent Fixes (Debug Campaign, March 2026)

### Fix 1: CUDA Error Clearing

Three-layer defense clears sticky CUDA errors that accumulate from Stage 1 kernel benchmarking:
- After Stage 1 siever teardown (`autotune.cpp`)
- Before probe orchestrator construction (`runtime_estimator.cpp`)
- At `TruncatedSieveRun` entry (`orchestrator.cpp`)

### Fix 2: Silent Mode for Probes

`MPQSConfig::silent` flag suppresses orchestrator constructor/destructor log noise during Stage 2/3 probes. Set by `runtime_estimator.cpp`.

### Fix 3: LP Cold-Start Bias

- Theory fallback seeds `lp1_bound = 10*F` for inputs >= 200 bits (previously always 0).
- LP yield correction: birthday-paradox model replaces the simpler `projected_fill_pct` discount.
- L=0 and LP golden section evaluated independently (not mutually exclusive).
- LP feedback loop breaker: L=0 projection overridden for N >= 290 bits.

### Fix 4: M_max_log2 Cap

Capped at 19 (M = 524288) to prevent GPU hangs at large sieve intervals.

### Fix 5: sasBlockDim Register Pressure

Root cause: `sieveAndScanKernel` (legacy) compiled to 68 registers/thread. At 1024 threads: 68 x 1024 = 69,632 > 65,536 SM register file -> `cudaErrorLaunchOutOfResources` -> unchecked -> hang.

Resolution: `__launch_bounds__(1024)` annotation on both `sieveAndScanKernel` and `sieveAndScanBatchKernel` forces the compiler to target <= 64 registers/thread. Preflight cap updated to 1024 as a safety net matching the annotation.

### Fix 6: sasGridDim Auto-Correction

When LP is active, preflight auto-corrects `sasGridDim` to `ceil_pow2(subCubeSize * numIntervals / 64)` to prevent per-block candidate buffer overflow (`maxRelationsPerBlock=64`).

### Fix 7: LinAlg Cost Model (commit `26267cc`)

Replaced the quadratic BW model (`T = 1.64e-9 * fb_size^2.0`) with an empirical sub-quadratic fit (`T = 8.5e-6 * fb_size^1.17`). Achieves 3% median error on the 75d-95d campaign + RSA-100 benchmarks, vs. 173% median error for the prior model. The sub-quadratic exponent reflects lingen dominance and GPU SpMM throughput saturation.

## Dependencies

Links (PUBLIC): `mpqs_common`, `cudampqs_build_flags`, `mpqs_sieve`.

PRIVATE include paths for `runtime_estimator.cpp` (which constructs ephemeral `MPQSOrchestrator`): postprocessing, largeprimes, matrix, sqrt, linalg. No link dependency on `mpqs_orchestrator` to avoid circular CMake dependency -- all symbols resolve at final executable link time.
