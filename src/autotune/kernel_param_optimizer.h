// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once
#include "kernel_launch_validator.h"
#include <cstdint>
#include <array>
#include <string>

// Forward declarations
namespace mpqs::sieve {
    class DeviceSievingController;
    struct factoringData;
}

namespace mpqs::autotune {

/// Result of kernel launch parameter optimization
struct KernelParamResult {
    Params8   params;          ///< optimal 8-tuple
    float     timing_us;       ///< verified score. NARROW (uint8): timing in µs, LOWER is better.
                               ///< WIDE (uint16, objective_is_rate==true): candidate survivors/sec,
                               ///< HIGHER is better. Positive ⇒ a valid measurement in both cases.
    uint32_t  configs_tested;  ///< number of configs evaluated
    bool      from_cache;      ///< true if result came from history DB lookup
    // ----- Wide-path autotune discriminators. Defaults keep the narrow path's
    //       existing contract byte-for-byte (rate objective off, floor trivially passed). -----
    bool      objective_is_rate = false;  ///< WIDE: `timing_us` is a survivors/sec RATE (higher=better)
    bool      beats_floor       = true;   ///< WIDE: winner_rate ≥ floor_rate·(1+margin)? (floor gate)
    float     floor_score       = 0.0f;   ///< WIDE: loadStandardConfig-wide survivors/sec (the floor)
};

/// Heuristic defaults based on convergence analysis.
/// Order: {subCubeSize, numIntervals, polyBlockSize, blocksPerCycle,
///          metaGridDim, metaBlockDim, sasGridDim, sasBlockDim}
constexpr Params8 HEURISTIC_DEFAULTS = {512, 8, 4, 8, 256, 1024, 256, 1024};

/// Candidate values for each parameter (power-of-two sequences)
constexpr uint32_t CANDIDATE_VALUES_0[] = {128, 256, 512, 1024};  // subCubeSize
constexpr uint32_t CANDIDATE_VALUES_1[] = {1, 2, 4, 8, 16, 32};         // numIntervals
constexpr uint32_t CANDIDATE_VALUES_2[] = {1, 2, 4, 8, 16, 32};         // polyBlockSize
constexpr uint32_t CANDIDATE_VALUES_3[] = {1, 2, 4, 8, 16, 32};         // blocksPerCycle
constexpr uint32_t CANDIDATE_VALUES_4[] = {32, 64, 128, 256};     // metaGridDim
constexpr uint32_t CANDIDATE_VALUES_5[] = {256, 512, 1024};       // metaBlockDim
constexpr uint32_t CANDIDATE_VALUES_6[] = {32, 128, 256, 512};    // sasGridDim
constexpr uint32_t CANDIDATE_VALUES_7[] = {256, 512, 1024};       // sasBlockDim

/// Weakly-convergent parameter indices (sweep order: sasGridDim, metaGridDim, polyBlockSize)
constexpr uint32_t WEAK_PARAM_INDICES[]   = {6, 4, 2};
/// Strongly-convergent parameter indices
constexpr uint32_t STRONG_PARAM_INDICES[] = {7, 5, 1, 3, 0};

// ----- WIDE / uint16 only: occupancy-seeded search space. -----------------------------
// On wide, the throughput-relevant launch dims are GATHER blockDim (P_SAS_BLOCK_DIM, the
// biggest lever — 25%→~100% occupancy), the SCATTER grid (P_META_GRID_DIM), num_polys /
// subCubeSize (P_SUB_CUBE_SIZE), and — secondarily — the SCATTER blockDim (P_META_BLOCK_DIM).
// numIntervals (pidx 1) is INERT (loadPartialCustomConfig-wide overrides it to 2M/SB_wide), so
// it is deliberately EXCLUDED from both wide lists (skipping it saves probe budget). The narrow
// (uint8) path is untouched — it keeps WEAK_/STRONG_PARAM_INDICES and CANDIDATE_VALUES_* above.
/// WIDE subCubeSize candidates: DROP 1024 (clampWideNumPolys caps num_polys at 512, so a 1024
/// candidate would clamp to 512 → a redundant probe of the same geometry). Belt-and-suspenders.
constexpr uint32_t WIDE_CANDIDATE_VALUES_0[] = {128, 256, 512};
/// WIDE Phase-2 coordinate-descent sweep order (always run): GATHER blockDim (7), SCATTER grid
/// (4), subCubeSize / num_polys (0).
constexpr uint32_t WIDE_WEAK_PARAM_INDICES[]   = {7, 4, 0};
/// WIDE Phase-2.5 thorough-only sweep: SCATTER blockDim (5).
constexpr uint32_t WIDE_STRONG_PARAM_INDICES[] = {5};

/// Mini-benchmark subcube counts
constexpr uint32_t NUM_SUBCUBES_COARSE = 2;  ///< Phase 2 (fast, ~0.2 ms per eval)
constexpr uint32_t NUM_SUBCUBES_FINE   = 4;  ///< Phase 3 (accurate, ~0.4 ms per eval)

/// Convergence threshold: improvement must exceed this fraction to continue
constexpr float EPSILON = 0.02f;

// ----- Wide-path (uint16) probe constants (unused on the narrow path) -----
/// Wall-clock window (seconds) over which the wide probe measures survivors/sec.
constexpr double WIDE_PROBE_WINDOW_SEC       = 10.0;
/// Floor the window shrinks to when the autotune budget is tight (still statistically stable).
constexpr double WIDE_PROBE_WINDOW_FLOOR_SEC = 6.0;
/// Warm-up prefix (seconds) discarded before each wide measurement (JIT, clock ramp, caches).
constexpr double WIDE_PROBE_WARMUP_SEC       = 2.5;
/// Floor-gate margin: accept the search winner only if its rate ≥ floor·(1+this).
constexpr float  WIDE_FLOOR_MARGIN           = 0.05f;

/// Run kernel launch parameter optimization via seeded coordinate descent.
///
/// Algorithm: seed from HEURISTIC_DEFAULTS, sweep 3 weakly-convergent params
/// (sasGridDim, metaGridDim, polyBlockSize), optionally sweep 5 strongly-convergent
/// params (thorough mode), then verify best with longer mini-benchmark.
///
/// Preconditions: siever must have initiate(), loadStandardConfig(),
///                loadData(), updateState() already called.
///
/// @param siever    Initialized DeviceSievingController
/// @param f_data    Factoring data with factor base populated
/// @param device_id CUDA device ordinal
/// @param thorough  If true, also search strongly-convergent params
/// @param non_sieve_bytes  OOM-guard: postprocessing/LP footprint + CUDA-context
///                  reserve. When > 0, the optimizer skips/clamps any candidate (and
///                  gates its own seed eval) whose COMPLETE footprint exceeds the 0.80
///                  budget of free VRAM. 0 (default) => candidate footprint guard
///                  inactive (the bucket-only isValid gate still applies).
/// @param probe_budget_sec  WIDE only: remaining autotune wall-clock budget for
///                  Stage-1. Each wide candidate runs a ~warmup+window (~12.5 s) probe, so
///                  the optimizer shrinks the window toward WIDE_PROBE_WINDOW_FLOOR_SEC (and,
///                  if truly out of budget, stops adding search candidates) — but ALWAYS still
///                  times the loadStandardConfig-wide floor. 0 (default) => unbounded.
///                  Ignored on the narrow path (byte-for-byte unchanged there).
/// @return Optimal params and verified score (µs on narrow / survivors/sec on wide)
KernelParamResult optimizeKernelLaunchParams(
    mpqs::sieve::DeviceSievingController& siever,
    const mpqs::sieve::factoringData& f_data,
    int device_id,
    bool thorough = false,
    uint64_t non_sieve_bytes = 0,
    double probe_budget_sec = 0.0);

} // namespace mpqs::autotune
