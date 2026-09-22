# Autotune Module (`src/autotune/`)

Automatic parameter optimization for the MPQS pipeline. Determines optimal factor base bound (F), sieve interval (M), large prime bound (L), and GPU kernel launch parameters via a 4-stage optimization loop: projection from history, kernel micro-benchmarking, runtime estimation from truncated sieve probes, and joint (F,L) convex optimization with gradient descent.

Namespace: `mpqs::autotune`.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `autotune.h` | 140 | `AutotuneController` class, `AutotuneConfig`, `AutotuneResult` |
| `autotune.cpp` | 1238 | Controller implementation: 4-stage loop, history I/O, convergence, buffer recommendations, wide floor-gate/no-signal handling |
| `autotune_types.h` | 130 | Shared types: `RuntimeEstimate`, `SieveOptimizationResult`, `SieveSearchBounds`, `L_function()` |
| `autotune_projection.h` | 96 | `ParameterProjector` class, `ProjectedParams` struct |
| `autotune_projection.cpp` | 488 | 4-tier projection cascade: exact match, interpolation, extrapolation, theory fallback |
| `kernel_param_optimizer.h` | 120 | `optimizeKernelLaunchParams()`, candidate value arrays (narrow + wide), heuristic defaults, wide probe constants |
| `kernel_param_optimizer.cpp` | 464 | Seeded coordinate descent over 8 kernel launch parameters; wide (uint16) survivors/sec search with floor gate |
| `kernel_launch_validator.h` | 192 | `KernelLaunchValidator` class (device-id **and** explicit-`DeviceLimits` constructors), `PreflightResult`, `Params8`, `ParamIndex` enum, `SieveConstants` (incl. `allow_nonpow2_geometry`), `admissibleSasGridDim()` |
| `kernel_launch_validator.cpp` | 551 | 6-check validation pipeline (**per-parameter** power-of-two policy since v1.0.6), enumeration, preflight with LP-aware auto-correction |
| `runtime_estimator.h` | 37 | `estimateRuntime()` free function |
| `runtime_estimator.cpp` | 273 | Truncated sieve probe via ephemeral orchestrator, ETA extrapolation, confidence scoring |
| `sieve_optimizer.h` | 185 | `SieveParameterOptimizer` class with joint (F,L) convex optimizer |
| `sieve_optimizer.cpp` | 718 | Three-phase optimizer: L-sweep (A), 3×3 grid (B), gradient descent (C) |
| `cost_models.h` | 32 | `estimateMatrixTime()`, `estimateLinalgTime()`, `deriveSieveSearchBounds()` |
| `cost_models.cpp` | 54 | Heuristic power-law cost models calibrated on RTX 5070 Ti |
| `autotune_history.h` | 120 | `HistoryStore` class, `HistoryEntry` struct, `sha256_hex()` |
| `autotune_history.cpp` | 530 | JSON persistence, FIPS 180-4 SHA-256, k-nearest-neighbor lookup with F5 two-pass filter |
| `auto_apply.h` | 74 | `AutoApplyController` class, `AutoApplyResult` struct |
| `auto_apply.cpp` | 387 | History-based parameter application without GPU probes |
| `benign_history.h` | 64 | `BenignHistoryStore` class, `BenignHistoryEntry` struct |
| `benign_history.cpp` | 243 | Cross-GPU hardcoded parameter defaults (17 entries: 12 desktop + 5 Jetson Orin), JSON persistence |
| `json_reader.h` | 321 | Shared cursor-based recursive-descent JSON parser; used by `HistoryStore::load` and `BenignHistoryStore::load` |
| `memory_estimator.h` | 53 | `memory_costs` namespace (per-element byte costs for on-device buffers), `kMinPartialBufferSize` floor constant |
| `CMakeLists.txt` | 29 | Static library `mpqs_autotune`, links `mpqs_common`, `mpqs_sieve` |

Total: ~6,500 lines across 23 source files (`wc -l src/autotune/*` = 6,539 including `CMakeLists.txt`, 2026-08-25).

Three committed unit tests live outside this directory but test only this module — all registered in
`tests/CMakeLists.txt` and all pure host arithmetic (no sieve kernel is launched):

| Source (`tools/autotune/`) | Lines | ctest target | Covers |
|---|---|---|---|
| `test_admissible_geometry.cu` | 220 | **`admissible_geometry`** (`tests/CMakeLists.txt:240-248`) | the v1.0.6 per-parameter pow2 policy + `admissibleSasGridDim`. Drives the real `KernelLaunchValidator` through its explicit-`DeviceLimits` constructor, so it needs no CUDA device. |
| `test_oom_guard.cu` | 216 | **`oom_guard`** (`:204-212`) | the Stage-1 total-footprint OOM guard: no-regression at pinned `M = 131,072`, and the guard firing (candidate skip + seed clamp) against a synthetic tiny free-VRAM budget. Reads device properties only. |
| `test_wide_num_polys_clamp.cu` | 116 | **`wide_num_polys_clamp`** (`:222-230`) | `clampWideNumPolys()` — the wide `num_polys <= kWideNumPolysCap = 512` cap on the `loadPartialCustomConfig` apply path (autotune / pinned tuple / AutoApply). |

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

Auto-apply also merges buffer parameters (witness capacity, partial buffer, accum buffer) using the same tier cascade, with a minimum partial buffer floor of 65,536 (`kMinPartialBufferSize`, `memory_estimator.h`) and a partial ≥ 8× accum floor (all modes, not just batch).

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
| 1 | `numIntervals` | P_NUM_INTERVALS | `gs_conf.num_sievingBlocksPerSieveCall` | {1, 2, 4, 8, 16, 32} |
| 2 | `polyBlockSize` | P_POLY_BLOCK_SIZE | `gms_conf.polyBlockSize` | {1, 2, 4, 8, 16, 32} |
| 3 | `blocksPerCycle` | P_BLOCKS_PER_CYC | `gms_conf.num_activeBlocksPerCycle` | {1, 2, 4, 8, 16, 32} |
| 4 | `metaGridDim` | P_META_GRID_DIM | `gms_conf.num_threadBlocks` | {32, 64, 128, 256} |
| 5 | `metaBlockDim` | P_META_BLOCK_DIM | `gms_conf.num_threadsPerBlock` | {256, 512, 1024} |
| 6 | `sasGridDim` | P_SAS_GRID_DIM | `ss_conf.num_threadBlocks` | {32, 128, 256, 512} |
| 7 | `sasBlockDim` | P_SAS_BLOCK_DIM | `ss_conf.num_threadsPerBlock` | {256, 512, 1024} |

**Power-of-two policy (statement updated 2026-08-25; the pre-v1.0.6 sentence "All values must be
powers of 2" is superseded as a blanket rule, but remains exactly true of everything the autotuner
itself can produce).**

*The search space above is unchanged.* `CANDIDATE_VALUES_0..7` and `HEURISTIC_DEFAULTS`
(`kernel_param_optimizer.h:38-47`) are byte-for-byte identical to v1.0.5 (`7c154c9`) — verified by
`git diff 7c154c9..HEAD -- src/autotune/kernel_param_optimizer.h` returning empty — and every value
in them is a power of two. Nothing under `src/autotune/` proposes, projects, or auto-applies an
SM-aligned (non-power-of-two) tuple; leaving the autotuner alone was a deliberate v1.0.6 decision.
**The autotuner cannot find a non-power-of-two geometry.** Automatic discovery is parked, unscheduled.

What v1.0.6 changed is only the *validator's* rule set: on the **narrow batch** sieve path the
`KernelLaunchValidator` will now ACCEPT a non-power-of-two `subCubeSize` (0), `metaGridDim` (4) and
`sasGridDim` (6) **if a caller supplies one**. The only such caller is a pinned `--params` on the
command line (`tests/cuda-mpqs.cpp:738-760` parses the 8-tuple with no power-of-two check) — or an
`autotune_history.json` entry that recorded one. See
[SM-Aligned (Non-Power-of-Two) Geometry](#sm-aligned-non-power-of-two-geometry-v106) below.

**v1.0.7 — nothing under `src/autotune/` changed, but three sieve-side changes reach it.**
(1) Every tuple the autotuner applies, projects or auto-applies goes through
`loadPartialCustomConfig`, which now sets `midPrimeStartIndex = 96` and runs the small-prime mask,
so an autotuned run sieves differently from the same tuple on v1.0.6 (a bare run through
`loadStandardConfig` keeps `midPrimeStartIndex = 32`). (2) `validateConfigs()` now demands
`numIntervals × SB == 2M` exactly on **every** narrow run, legacy included, so over-covering probes
(the M-sweep driving `SB = min(M, …) == M` at a fixed interval count) and legacy probes with
`numIntervals × SB < 2M` — both legal in v1.0.6 — are now rejected; expect more skipped probes on
small-shared-memory devices. (3) The new `--param_test` search (sieve module,
`DeviceSievingController::runParamTest`) is **separate from the autotuner**: it searches `SB`,
`bigPrimeStartIndex` and `midPrimeStartIndex` in addition to the `Params8` axes, prints its winner
as a `--params11` line, and writes no `autotune_history.json` entry. The autotuner cannot produce or
record an 11-field tuple; a `--params11` pin takes precedence over any autotune or history tuple.

### Algorithm

1. **Phase 1 (Seed):** Start from `HEURISTIC_DEFAULTS = {512, 8, 4, 8, 256, 1024, 256, 1024}`. Clamp to feasible region (e.g., `subCubeSize <= 2^(shc_dim-1)`, `blocksPerCycle <= numIntervals`). Validate; if infeasible, fall back to first valid enumerated config.

2. **Phase 2 (Weak params):** Coordinate descent over weakly-convergent parameters in order: `sasGridDim` (6), `metaGridDim` (4), `polyBlockSize` (2). Each candidate is validated, then timed via `DeviceSievingController::evaluateConfig()` with 2 subcubes (coarse). Improvement threshold: 2% (`EPSILON`).

3. **Phase 2.5 (Thorough, optional):** If `thorough=true`, also sweeps strongly-convergent params: `sasBlockDim` (7), `metaBlockDim` (5), `numIntervals` (1), `blocksPerCycle` (3), `subCubeSize` (0).

4. **Phase 3 (Verify):** Re-evaluate winning config with 4 subcubes (fine) for more accurate timing.

**Post-optimization:** Clears sticky CUDA errors (`cudaDeviceSynchronize()` + `cudaGetLastError()`) that may accumulate from failed kernel launches during benchmarking. Runs a defense-in-depth preflight check on the winning config before applying to the pipeline.

**Stage-1 OOM guard:** the controller passes `non_sieve_bytes` = `computePostprocessingLpBytes()` (postprocessing/LP device footprint + CUDA-context reserve, computed from `memory_estimator.h` per-element costs) into `optimizeKernelLaunchParams()`; when > 0 the optimizer skips/clamps any candidate — and gates its own seed eval — whose complete footprint exceeds 0.80 of free VRAM (`fitsTotalFootprint()`, `kernel_launch_validator.cpp:295-314`, over `estimateSieveFootprint()` + `sieveBucketBudget()`).

*Root cause it fixed (commits `62043ed` / `04e1ab6`):* the historical bucket-memory expression multiplied **three `uint32_t` fields** (`num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize`) in 32-bit **before** the trailing `* sizeof(uint64_t)` promoted to 64-bit, so for buckets ≥ 4 GB the product **wrapped** — at the `M = 262K` seed it wrapped to 0, which made the "halve `num_polys` until the bucket fits" reduction loop a **no-op** and let an over-budget config through unremarked. The formula had been hand-rolled in five drifted copies; it now lives once in `src/sieve/sieve_memory_model.h` (`bucketEntriesBytes()` / `estimateSieveFootprint()` / `sieveBucketBudget()` / `reduceNumPolysToBudget()`), all products promoted to `uint64_t`, and every site — `loadStandardConfig`'s reduction loop, `validateConfigs`' `LEQ_CHECK`, `checkGlobalMem`, its `diagnose()` message, `printBufferRecommendations` **and** this guard — reads the same `kSieveBudgetNum/kSieveBudgetDen = 4/5` constant, so the autotuner can never pick a config the production checks then reject. Regression test: ctest **`oom_guard`**.

### Wide-Path (uint16) Autotune

When the siever resolved to the wide accumulator (`siever.isWideAccumulator()`, i.e. the
RSA-150/155 uint16 — or saturating-uint8 — regime), Stage 1 switches from the
narrow µs-timing objective to a **survivors/sec rate objective** that actually drives the
committed `sieveAndScanBatchKernelWide` geometry (`loadPartialCustomConfig`-wide, matched to
u8sat when selected). The narrow (uint8) path is byte-for-byte unchanged. All discriminators
live in `KernelParamResult`: `objective_is_rate` (true ⇒ `timing_us` is survivors/sec,
HIGHER is better), `beats_floor`, `floor_score`.

- **Survivors/sec harness:** each candidate re-sieves the same staged polynomial sample
  (`--autotune_probe_polys`, 0 = auto-scale by N) over a wall-clock window
  (`WIDE_PROBE_WINDOW_SEC` = 10 s, shrinking toward `WIDE_PROBE_WINDOW_FLOOR_SEC` = 6 s under a
  tight `probe_budget_sec`), after a discarded `WIDE_PROBE_WARMUP_SEC` = 2.5 s warm-up (JIT,
  clock ramp, caches). When the budget is exhausted the search stops adding candidates but the
  floor is ALWAYS still timed.
- **Floor gate:** `loadStandardConfig`-wide is timed as candidate #0
  (`sieveMiniStandardWide()`, the `floor_score`); the tuned winner is applied only if its rate
  ≥ `floor · (1 + WIDE_FLOOR_MARGIN)` (margin 0.05). Otherwise the controller keeps the
  loadStandardConfig defaults (`useParams = false`) — provable parity; closes the −73%
  W-track regression class.
- **No-signal guard:** a measured 0 survivors/s across every candidate AND the floor (e.g. LP
  off / too-tight threshold at this scale) means the autotune got no signal: warn and keep
  loadStandardConfig instead of silently reverting to a meaningless floor
  (`stages[1].notes = "wide no-signal: keeping loadStandardConfig"`).
- **Occupancy-seeded search:** the search is seeded AT the known-good wide default (not
  descended from `HEURISTIC_DEFAULTS`, whose geometry region is wrong for wide); the GATHER
  blockDim seed comes from `siever.wideGatherOccupancyBlockDim()` (cudaOccupancy API; falls
  back to the standard value when unavailable). The wide sweep uses its own index lists —
  `WIDE_WEAK_PARAM_INDICES = {7, 4, 0}` (GATHER blockDim, SCATTER grid, subCubeSize/num_polys)
  always, `WIDE_STRONG_PARAM_INDICES = {5}` (SCATTER blockDim) thorough-only — and
  `WIDE_CANDIDATE_VALUES_0 = {128, 256, 512}` for subCubeSize (1024 dropped:
  `clampWideNumPolys` caps num_polys at 512, so it would probe a redundant geometry).
  `numIntervals` (1) is INERT on wide (`loadPartialCustomConfig`-wide overrides it to
  `2M/SB_wide`) and is excluded from both lists.

Validated on RTX 5070 Ti: floor gate fired on the −73% regression scenario; +38.5% wide efficacy
at RSA-100; narrow byte-identical (RSA-100 ~85 s record reproduced on the same binary).

### Skip Logic

Stage 1 is skipped on iterations > 0 when both F and M have changed by < 10% since the last Stage 1 run, since kernel launch parameters are weakly sensitive to small F/M changes. The gate and the recorded last-run values key on **`f_data_.F` / `f_data_.M`** — the F/M Stage 1 actually consumes — NOT on `config_.fb_bound`/`sieve_bound` (which are legitimately 0 in auto mode; keying on them zeroed the recorded values and made Stage 1 run twice per autotune, applying the noisier second-iteration winner — a ~17% self-inflicted loss, fixed `8005da9`).

## Kernel Launch Validator (`kernel_launch_validator.cpp`)

**Purpose:** Pure-arithmetic feasibility checker for the 8-parameter tuple. No GPU launches, no CUDA streams. Only the constructor calls `cudaGetDeviceProperties()`.

### 6-Check Validation Pipeline

Ordered from cheapest to most expensive:

| Check | Rule | Source |
|-------|------|--------|
| 1. Power-of-2 | **Per parameter** (`pow2Required(idx)`, `kernel_launch_validator.cpp:166-169`): every index must be **non-zero**; all 8 must be pow2 in the legacy/wide regime; on the narrow batch path indices 0/4/6 (`subCubeSize`, `metaGridDim`, `sasGridDim`) are exempt. *(Superseded 2026-08-25: the old "all 8 params are powers of 2" was unconditional.)* | `validateConfigs` `POW2_CHECK` / `POW2_UNLESS_CHECK`, `device_sieving_controller.cpp:1900,2029,2054-2069` |
| 2. Arithmetic | `subCubeSize <= 2^(shc_dim-1)`, `blocksPerCycle <= numIntervals`, `metaGridDim <= subCubeSize`, `metaGridDim * polyBlockSize <= subCubeSize`; **G3** `sasGridDim <= subCubeSize`; **G1** `sasGridDim` divides `subCubeSize`; **G2** the GATHER chunk `subCubeSize/sasGridDim` is a power of two; **SCATTER exact partition** `metaGridDim*polyBlockSize` divides `subCubeSize` — checked **before** deriving `num_polyBlocksPerTB`; derived `num_polyBlocksPerTB > 0`, and pow2 **only when `pow2Required` holds** (legacy/wide); exact decomposition equalities | `validateConfigs` G1/G2/G3 + V5, `device_sieving_controller.cpp:2071-2087,2093-2102` |
| 3. Shared memory | `blocksPerCycle * polyBlockSize * 4B <= maxSharedMem`; `sievingBlockSize * accumulatorBytes + 3 * bigPrimeStartIndex * 4B <= maxSharedMem`, where `bigPrimeStartIndex = sievingBlockSize / 32` and `accumulatorBytes` = 1 (narrow / u8sat) or 2 (uint16 wide). *(Corrected 2026-08-26: the old second term `sievingBlockSize + 3*1024*4B` hardcoded a `bigPrimeStartIndex` of 1024 and omitted the accumulator width; neither has been constant since the wide fork.)* | `checkSharedMem`, `kernel_launch_validator.cpp:260-269`; `bigPrimeStartIndex`/`accumulatorBytes` set in `buildSieveConstants`, `:437-444`. Mirrored production-side by `validateConfigs`' `EQUAL_CHECK` on `ss_conf.sharedMemReq` + `LEQ_CHECK` vs `maxSharedMemPerBlock`, `device_sieving_controller.cpp:2097-2100,2120-2121` |
| 4. Global memory | `bucketEntriesBytes(subCubeSize, numIntervals, globalBucketSize)` = `subCubeSize * numIntervals * globalBucketSize * 8B` `<= sieveBucketBudget(totalGlobalMem, 0, kSieveBudgetNum, kSieveBudgetDen)` = **0.80 × totalGlobalMem**. *(Corrected 2026-08-26: the documented `3/4` is superseded — `kSieveBudgetNum/kSieveBudgetDen` flipped 3/4 → 4/5 in the memory-model refactor, `src/sieve/sieve_memory_model.h:64-65`. The comparison is integer-exact by design; a `double` 0.80 can flip the seed reduction loop's last halving.)* | `checkGlobalMem`, `kernel_launch_validator.cpp:277-289`; production twin `validateConfigs` `LEQ_CHECK`, `device_sieving_controller.cpp:2119` |
| 5. Device limits | `metaBlockDim`, `sasBlockDim <= maxThreadsPerBlock`; grid dims <= `maxGridSize[0]` | CUDA runtime |
| 6. Non-zero derived | **Exactness first** (v1.0.6): `metaGridDim*polyBlockSize != 0` and divides `subCubeSize`; `blocksPerCycle != 0` and divides `numIntervals`; then `num_sievingBlockBatches > 0`, `num_subCubes > 0`, `num_polyBlocksPerTB > 0`, `num_metaSieveCycles > 0` (`checkNonZeroDerived`, `kernel_launch_validator.cpp:339-366`) | `validateConfigs` |

**`enumerateValidConfigs()`:** Brute-force over all candidate value combinations, filtering through the validator. Returns valid configs in < 1 ms. The trip count is **4·6·6·6·4·3·4·3 = 124,416**, not the ~50k previously documented (corrected 2026-08-26). ⚠ The eight value lists are written out **inline** in the loop nest (`kernel_launch_validator.cpp:400-407`) rather than read from `CANDIDATE_VALUES_0..7`; they currently agree value-for-value with `kernel_param_optimizer.h:40-47`, but the duplication is unguarded and would drift silently.

### Preflight Check

`preflightKernelLaunch()` is the entry point for runtime validation. Two overloads, **both of which
gained a trailing `bool allow_nonpow2_geometry = false` in v1.0.6** (`kernel_launch_validator.h:156-172`):

1. **Raw `Params8` overload:** Builds `SieveConstants` from device properties and factoring dimensions, validates via `KernelLaunchValidator`.
   Signature: `(params, shc_dim, M, device_id, use_wide = false, use_u8sat = false, allow_nonpow2_geometry = false)`.

2. **`MPQSConfig&` overload (mutating):** `(config, shc_dim, M, allow_nonpow2_geometry = false)`. Includes three auto-correction passes before validation:
   - **LP-aware sasGridDim floor:** When `lp1_bound > 0`, computes `min_sas = ceil(subCubeSize * numIntervals / 64)` and then rounds it up to the smallest **admissible** grid via `admissibleSasGridDim(subCubeSize, min_sas)` — see below. Auto-corrects `params[6]` upward if needed. *(Superseded 2026-08-25: before v1.0.6 this step was an inline bit-smear "round up to the next power of two"; it is now the `admissibleSasGridDim` call at `kernel_launch_validator.cpp:511-513`. On a power-of-two `subCubeSize` the new call reproduces the old behaviour byte-for-byte.)*
   - **sasGridDim > subCubeSize clamp:** clamps `params[6]` down to `subCubeSize` with a `LOG_WARNING` (zero-work sieve iterations otherwise).
   - **sasBlockDim `__launch_bounds__` cap:** When `sieve_batch_size == 0` (legacy mode), caps `params[7]` at 1024 to match the `__launch_bounds__(1024)` annotation on `sieveAndScanKernel`.

   Note the raw-overload delegation at `kernel_launch_validator.cpp:545-547` passes `use_wide = false, use_u8sat = false` explicitly — the `MPQSConfig&` overload has always been a **narrow-path** preflight.

### SM-Aligned (Non-Power-of-Two) Geometry (v1.0.6)

Added in v1.0.6; the entire `src/autotune/` delta between v1.0.5 and v1.0.6 is confined to `kernel_launch_validator.{h,cpp}` (`git diff --numstat 7c154c9..HEAD -- src/autotune/`
= +123/-28 in the `.cpp`, +36/-3 in the `.h`; every other file in this directory is untouched).
The motivation is the sieve, not the autotuner: setting the SCATTER/GATHER grids to the device's
SM count exactly requires non-power-of-two values, which the validator previously rejected
outright.

**The consequence of leaving the autotuner pow2.** `CANDIDATE_VALUES_4` tops out at
`metaGridDim = 256` and every entry is a power of two, so on a 108-SM A100 the pow2 winner
`metaGridDim = 64` runs SCATTER at **0.590 waves/SM — 44 of 108 SMs idle**, for a kernel that is
~34 % of GPU time. The exact-wave alternative (1.000 waves/SM) is reachable **only by hand**, via a
pinned `--params` tuple such as `864,8,8,8,108,1024,864,1024`; the autotuner will never propose,
project or auto-apply it, and SM counts appear nowhere in `src/autotune/` as a geometry input
(`DeviceLimits::multiProcessorCount` is populated at `kernel_launch_validator.cpp:31` and then read
by nothing). Automatic discovery is a named but **unscheduled** follow-up; leaving the autotuner
alone was a deliberate decision, not an oversight. (The waves/SM figures are a measurement result,
not a source fact.)

**Gate.** `SieveConstants::allow_nonpow2_geometry` (default `false`, `kernel_launch_validator.h:37-42`).
`buildSieveConstants(shc_dim, M, maxSharedMemPerBlock, use_wide, use_u8sat, allow_nonpow2_geometry)`
stores `allow_nonpow2_geometry && !use_wide` — the relaxation is **force-disabled on the wide path
whatever the caller asks** (`kernel_launch_validator.cpp:445-448`).

**Per-parameter policy.** `KernelLaunchValidator::pow2Required(uint32_t idx)` (private,
`kernel_launch_validator.cpp:166-169`) returns `true` for every index unless
`allow_nonpow2_geometry`, in which case it frees exactly
`{P_SUB_CUBE_SIZE (0), P_META_GRID_DIM (4), P_SAS_GRID_DIM (6)}`. `checkPow2()` now rejects **0 for
every index** (previously folded into the same test) and applies pow2 per parameter. `diagnose()`
carries matching messages, including a distinct `param[i]=0 (must be > 0)`.

**What replaces the pow2 guarantee.** In the power-of-two world, divisibility was free; the
relaxed world must state it. `checkArithmeticConstraints()` (`kernel_launch_validator.cpp:203-251`)
now enforces, mirroring `validateConfigs`:

| Id | Invariant | Why |
|----|-----------|-----|
| G1 | `sasGridDim` divides `subCubeSize` | a non-divisor GATHER grid overlaps `polyIdPrefix`es AND leaves a polynomial tail ungathered — silently wrong relations, no error |
| G2 | chunk `= subCubeSize / sasGridDim` is a power of two | the batch GATHER composes `polyIdPrefix \| gray(poly)`; the OR equals addition only for a pow2 chunk |
| G3 | `sasGridDim <= subCubeSize` | otherwise a zero trip count = silent zero-relation sieve |
| — | `metaGridDim * polyBlockSize` divides `subCubeSize`, **checked before** deriving `num_polyBlocksPerTB` | SCATTER has a fixed trip count and no `if (id < n)` guard; a floored quotient silently under- or over-covers |

`num_polyBlocksPerTB` must still be a power of two **in the legacy/wide regimes only**
(`pow2Required(P_SUB_CUBE_SIZE)`); on the narrow batch path no kernel reads its log2, so exactness
is the whole requirement. G1/G2/G3 are tautologies for power-of-two tuples, so **no previously
valid tuple changes verdict**.

**`admissibleSasGridDim(uint32_t np, uint32_t min_sas)`** — new free function
(`kernel_launch_validator.cpp:378-390`). The admissible GATHER grids are exactly
`{ np / 2^j : 2^j | np }` (G1 + G2). It walks that set from the smallest upward and returns the
first value `>= min_sas`, clamping to `np` (chunk 1) when `min_sas` exceeds every admissible grid;
`np == 0` returns 0. For a power-of-two `np` the admissible set *is* the power-of-two ladder
`{1, 2, ..., np}`, so it reproduces "round up to the next power of two, then clamp to `np`" exactly.
Example from the source comment: `np = 864, min_sas = 108 -> 108`, not 128.

**Explicit-`DeviceLimits` constructor.** `KernelLaunchValidator(const DeviceLimits&, const SieveConstants&)`
(`kernel_launch_validator.h:77-80`, header-inline) performs identical checks with **no CUDA call**, so
the policy can be unit-tested deterministically on a machine with no GPU. This is what the
`admissible_geometry` ctest target uses.

**Who passes `true` — the caller matrix (verified 2026-08-25):**

| Call site | `allow_nonpow2_geometry` |
|-----------|--------------------------|
| `SieveStage()`, `src/orchestrator/orchestrator.cpp:4320-4325` | `(config_.sieve_batch_size > 0) && !siever_->isWideAccumulator()` — the **only** `true` |
| `TruncatedSieveRun()`, `src/orchestrator/orchestrator.cpp:6403-6404` | not passed -> default `false` |
| `estimateRuntime()` probe, `src/autotune/runtime_estimator.cpp:87-89` | not passed -> default `false` |
| Stage-1 winner preflight, `src/autotune/autotune.cpp:623-625` | not passed -> default `false` |
| `optimizeKernelLaunchParams()`'s own validator, `src/autotune/kernel_param_optimizer.cpp:103-106` | `buildSieveConstants(...)` called without the argument -> `false` |

The predicate in `SieveStage()` is exactly `validateConfigs`' `relaxed_geometry`
(`src/sieve/device_sieving_controller.cpp:1960-1962`), which gates the `POW2_UNLESS_CHECK` macro
(`:2029-2031`) and the SM-aligned narrow launch-feasibility/occupancy block (`:2170`).

⚠ **Caveat — the autotune probe paths run under the mandatory-pow2 rule set.** Neither
`TruncatedSieveRun()` nor `estimateRuntime()` forwards the flag, so a pinned SM-aligned tuple fails
preflight there: `TruncatedSieveRun` emits `LOG_WARNING "Preflight failed: ... -- returning zero
relations"` and returns an empty result, and `estimateRuntime` returns the
`confidence = 0.0 / total_est_sec = 1e9` sentinel. In practice this is **self-consistent rather than
a bug**, because `estimateRuntime` unconditionally forces `cfg.sieve_batch_size = 0`
(`runtime_estimator.cpp:64`) and its ephemeral orchestrator is the only caller of
`TruncatedSieveRun`, so the probe genuinely runs in legacy mode, where mandatory pow2 is correct.
The practical consequence is that **combining a pinned SM-aligned `--params` with `--autotune`
makes every Stage-2/3 probe return the 1e9 sentinel.** Do not do it — see [Pinning, Determinism and History Caveats](#pinning-determinism-and-history-caveats) below.

## Pinning, Determinism and History Caveats

Three operational facts that follow from the code above and matter whenever this module's output
is benchmarked or transported between binaries.

**1. In-run Stage 1 is NON-DETERMINISTIC — pin `--params` for any A/B.** Identically-configured
runs (`--autotune_stage1 --autotune_no_history --autotune_max_iter 2`, empty `mpqs_work`) can
explore different numbers of configurations and land on **different winners**. Measured on A100 at
RSA-100: five identically-configured arms explored **16 / 17 / 18** configurations and produced
**two** winners — `512,8,8,8,64,1024,512,1024` (probe 67.3 ms) and `512,8,4,8,128,1024,512,1024`
(71.5 ms), differing **only** in `polyBlockSize` and `metaGridDim`. In the real run the loser
sieves **SCATTER 15.5 % slower** (3.915 vs 3.390 ms) with GATHER identical to 0.03 % — up to ~7.8 s
of RSA-100 sieve wall, larger than most effects such arms are built to measure. Any benchmark,
probe or roofline A/B must therefore pin
`--params` rather than re-autotune per arm. Note also that `AutoApplyController` can freeze either
winner into `autotune_history.json`. *(This is an empirical measurement result, not a property
readable from the source.)*

**2. History files are not version-portable across the v1.0.6 geometry change.** `saveHistory()`
copies `pipeline_config_.params[0..7]` into the entry verbatim
(`autotune.cpp:873-875`), and no load-time quality filter tests for powers of two — F1
(`autotune_history.cpp:324-327`) only rejects the degenerate `128,4,4,4` prefix. `AutoApplyController`
then re-applies a matching entry's tuple verbatim and sets `useParams = true`
(`auto_apply.cpp:71-81`). So an `autotune_history.json` written by a v1.0.6-or-newer binary can carry a
non-power-of-two tuple; an older binary that auto-applies it will fail its all-8-pow2 preflight and
abort **loudly** (`SieveStage` throws on `!pf.feasible`). Escape when downgrading: delete the
history file, or pass `--autotune_no_history`.

**3. The v1.0.6 narrow-batch sieve geometry overrides are forbidden with autotune, by design.**
`--sieve_block_size` and `--sieve_big_prime_start` are rejected at the CLI with `exit(1)` if
combined with `--autotune` / `--autotune_stage*` (`tests/cuda-mpqs.cpp:1123-1130`); they also
require `--params` and reject legacy mode (`:1107-1137`). The reasons given in-source: the
autotuner re-enters `loadPartialCustomConfig` per probe with tuple-derived interval counts that the
override would silently desynchronise from `SB`, and the autotune-side `KernelLaunchValidator`
derives its own `sievingBlockSize` in `buildSieveConstants` and never sees the override at all.
**Nothing in `src/autotune/` is aware of these knobs.**

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

- `max_total_probes = 40` (raised from 12 for the joint optimizer; `sieve_optimizer.h:45`)
- `wall_clock_timeout_sec` = remaining autotune budget (skips if < 30s)
- Convergence penalty: if not converged, `confidence *= 0.5`

⚠ **Known issue — the probe budget is not honoured as a hard cap.** In practice the joint (F,L)
optimizer issues **~160 truncated-sieve probes against the ~40 target, a 2.5–3.6× overhead**. This
is a standing measured defect, not a source-readable one — nothing in `sieve_optimizer.cpp` states
it — so treat `max_total_probes = 40` as a *nominal* budget when sizing `--autotune_timeout`.

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
| `candidates_file` | string | "" | Path to candidates.txt for bootstrap |
| `bootstrap` | bool | false | Bootstrap mode (--autotune_bootstrap) |

*(Corrected 2026-08-26: a `verbose` row was listed here; `AutotuneConfig` has no such field —
`autotune.h:22-40` is the complete struct.)* `thorough` has **no CLI flag**, so Phase 2.5 never
runs in a shipped build; its only reader is `autotune.cpp:540`.

`autotune_probe_polys` is **not** an `AutotuneConfig` field — it lives on `MPQSConfig`
(`include/orchestrator.h:250`) and reaches the siever via
`setAutotuneProbePolys()` (`autotune.cpp:405`).

### CLI Flags (parser: `tests/cuda-mpqs.cpp`)

Verified against the live parser 2026-08-26. The ✓ column marks the flags that set
`config.autotune_enabled = true` — that field, not the stage selection, is what the v1.0.6
geometry-override guard tests (see
[Pinning, Determinism and History Caveats](#pinning-determinism-and-history-caveats)).

| Flag | Line | Sets | Enables autotune |
|------|------|------|------------------|
| `--autotune` | 968 | `config.autotune_enabled` | ✓ |
| `--autotune_only` | 971-973 | `mode = AUTOTUNE_ONLY` + enabled | ✓ |
| `--autotune_stage0` | 975-979 | `enable_stage0` (explicit) | ✓ |
| `--autotune_stage1` | 980-984 | `enable_stage1` (explicit) | ✓ |
| `--autotune_stage2` | 985-989 | `enable_stage2` (explicit) | ✓ |
| `--autotune_stage3` | 990-994 | `enable_stage3` (explicit) | ✓ |
| `--autotune_max_iter <N>` | 995-997 | `max_iterations` | — |
| `--autotune_timeout <sec>` | 998-1000 | `timeout_sec` | — |
| `--autotune_history <path>` | 1001-1003 | `history_file` | — |
| `--autotune_benign_history <path>` | 1004-1006 | `benign_history_file` | — |
| `--autotune_no_history` | 1007-1010 | `load_history = save_history = false` | — |
| `--autotune_candidates <path>` | 1011-1013 | `candidates_file` | — |
| `--autotune_bootstrap` | 1014-1016 | `bootstrap` | — |
| `--autotune_probe_polys <N>` | 850-855 | `MPQSConfig::autotune_probe_polys` (wide probe sample; 0 = auto-scale by N) | — |

⚠ **`--autotune_stage*` are SELECTIVE, not additive.** Naming any one of them sets
`has_explicit_autotune_stages`, and the post-parse fixup (`tests/cuda-mpqs.cpp:1095-1101`) then
assigns **all four** `enable_stageN` from the explicit flags — so `--autotune_stage1` alone
*disables* Stages 0, 2 and 3 rather than adding Stage 1 to the defaults.

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

When LP is active, preflight auto-corrects `sasGridDim` upward to prevent per-block candidate
buffer overflow (`maxRelationsPerBlock=64`).

*(Refined 2026-08-25, v1.0.6 — the original wording `ceil_pow2(subCubeSize * numIntervals / 64)`
is superseded.)* The floor is still `min_sas = ceil(subCubeSize * numIntervals / 64)`, but the
rounding is now `admissibleSasGridDim(subCubeSize, min_sas)` — the smallest divisor of
`subCubeSize` whose quotient is a power of two. On a power-of-two `subCubeSize` this is
byte-for-byte the old `ceil_pow2`-then-clamp; on an SM-aligned `subCubeSize` it lands on a grid the
GATHER decomposition can actually use.

### Fix 7: LinAlg Cost Model (commit `26267cc`)

Replaced the quadratic BW model (`T = 1.64e-9 * fb_size^2.0`) with an empirical sub-quadratic fit (`T = 8.5e-6 * fb_size^1.17`). Achieves 3% median error on the 75d-95d campaign + RSA-100 benchmarks, vs. 173% median error for the prior model. The sub-quadratic exponent reflects lingen dominance and GPU SpMM throughput saturation.

## Dependencies

Links (PUBLIC): `mpqs_common`, `cudampqs_build_flags`, `mpqs_sieve`.

PRIVATE include paths for `runtime_estimator.cpp` (which constructs ephemeral `MPQSOrchestrator`): postprocessing, largeprimes, matrix, sqrt, linalg. No link dependency on `mpqs_orchestrator` to avoid circular CMake dependency -- all symbols resolve at final executable link time.
