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
    float     timing_us;       ///< verified timing (microseconds)
    uint32_t  configs_tested;  ///< number of configs evaluated
    bool      from_cache;      ///< true if result came from history DB lookup
};

/// Heuristic defaults based on convergence analysis (M2 plan §2.2).
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

/// Mini-benchmark subcube counts
constexpr uint32_t NUM_SUBCUBES_COARSE = 2;  ///< Phase 2 (fast, ~0.2 ms per eval)
constexpr uint32_t NUM_SUBCUBES_FINE   = 4;  ///< Phase 3 (accurate, ~0.4 ms per eval)

/// Convergence threshold: improvement must exceed this fraction to continue
constexpr float EPSILON = 0.02f;

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
/// @return Optimal params and verified timing
KernelParamResult optimizeKernelLaunchParams(
    mpqs::sieve::DeviceSievingController& siever,
    const mpqs::sieve::factoringData& f_data,
    int device_id,
    bool thorough = false);

} // namespace mpqs::autotune
