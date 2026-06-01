// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "kernel_param_optimizer.h"
#include "kernel_launch_validator.h"
#include "device_sieving_controller.h"
#include "common.h"
#include "hpc_logger.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace mpqs::autotune {

// ---------------------------------------------------------------------------
// Helper: candidate value array accessor
// ---------------------------------------------------------------------------

namespace {

struct CandidateRange {
    const uint32_t* values;
    uint32_t count;
};

CandidateRange getCandidates(uint32_t param_idx) {
    switch (param_idx) {
        case 0: return {CANDIDATE_VALUES_0, 4};
        case 1: return {CANDIDATE_VALUES_1, 4};
        case 2: return {CANDIDATE_VALUES_2, 4};
        case 3: return {CANDIDATE_VALUES_3, 4};
        case 4: return {CANDIDATE_VALUES_4, 4};
        case 5: return {CANDIDATE_VALUES_5, 3};
        case 6: return {CANDIDATE_VALUES_6, 4};
        case 7: return {CANDIDATE_VALUES_7, 3};
        default: return {nullptr, 0};
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Seeded coordinate descent optimizer
// ---------------------------------------------------------------------------

KernelParamResult optimizeKernelLaunchParams(
    mpqs::sieve::DeviceSievingController& siever,
    const mpqs::sieve::factoringData& f_data,
    int device_id,
    bool thorough)
{
    // 1. Build validator from device properties and factoring data
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    SieveConstants sc = buildSieveConstants(
        f_data.a_factors.size(), f_data.M, prop.sharedMemPerBlock);
    KernelLaunchValidator validator(device_id, sc);

    uint32_t configs_tested = 0;

    // 2. Phase 1: Seed selection — start from heuristic defaults, clamp to feasible
    Params8 best = HEURISTIC_DEFAULTS;

    // Clamp subCubeSize to max feasible: subCubeSize <= 2^(shc_dim - 1)
    uint32_t maxSubCube = 1u << (sc.shc_dim - 1);
    if (best[P_SUB_CUBE_SIZE] > maxSubCube) best[P_SUB_CUBE_SIZE] = maxSubCube;

    // Clamp numIntervals: numIntervals <= (2*M) / sievingBlockSize
    uint32_t maxIntervals = (2 * sc.M) / sc.sievingBlockSize;
    if (best[P_NUM_INTERVALS] > maxIntervals) best[P_NUM_INTERVALS] = maxIntervals;

    // Clamp blocksPerCycle <= numIntervals
    if (best[P_BLOCKS_PER_CYC] > best[P_NUM_INTERVALS])
        best[P_BLOCKS_PER_CYC] = best[P_NUM_INTERVALS];

    // Clamp metaGridDim: metaGridDim * polyBlockSize <= subCubeSize
    while (best[P_META_GRID_DIM] * best[P_POLY_BLOCK_SIZE] > best[P_SUB_CUBE_SIZE]
           && best[P_META_GRID_DIM] > 1)
        best[P_META_GRID_DIM] >>= 1;

    // Validate seed; if infeasible, fall back to first valid enumerated config
    if (!validator.isValid(best)) {
        auto all_valid = enumerateValidConfigs(validator);
        if (all_valid.empty()) {
            LOG(LOG_ERROR_CRITICAL) << "[Autotune] No valid kernel launch configs found!";
            return KernelParamResult{{}, 0.0f, 0, false};
        }
        best = all_valid[0];
    }

    // 3. Evaluate seed with coarse mini-benchmark
    bool reload_needed = true;
    float best_time = siever.evaluateConfig(best, NUM_SUBCUBES_COARSE, reload_needed);
    if (best_time < 0.0f) {
        LOG(LOG_ERROR_CRITICAL) << "[Autotune] Seed config evaluation failed!";
        return KernelParamResult{best, 0.0f, 0, false};
    }
    ++configs_tested;

    LOG(LOG_DEBUG_1) << "[Autotune] Phase 1 seed: timing = " << best_time << " us";

    // 4. Phase 2: Coordinate descent on weakly-convergent parameters
    //    Sweep sasGridDim (6), metaGridDim (4), polyBlockSize (2).
    //    These have nearly independent, unimodal 1D slices.
    bool improved = true;
    while (improved) {
        improved = false;
        for (uint32_t pidx : WEAK_PARAM_INDICES) {
            auto [values, count] = getCandidates(pidx);
            for (uint32_t v = 0; v < count; ++v) {
                if (values[v] == best[pidx]) continue;

                Params8 candidate = best;
                candidate[pidx] = values[v];
                if (!validator.isValid(candidate)) continue;

                // Always reload — comparing candidate against 'best' is wrong.
                // 'best' tracks the logically optimal config, but buffers are
                // sized for the last loadSievingDataParamTest() call, which may
                // differ (e.g., after a sweep where no candidate improved).
                reload_needed = true;

                float t = siever.evaluateConfig(candidate, NUM_SUBCUBES_COARSE, reload_needed);
                ++configs_tested;

                if (t > 0.0f && t < best_time * (1.0f - EPSILON)) {
                    best = candidate;
                    best_time = t;
                    improved = true;
                    LOG(LOG_DEBUG_2) << "[Autotune] Phase 2: param[" << pidx
                                     << "] = " << values[v] << " -> " << t << " us";
                }
            }
        }
    }

    LOG(LOG_DEBUG_1) << "[Autotune] Phase 2 complete: "
                     << configs_tested << " configs tested, best = " << best_time << " us";

    // 4.5 Optional: sweep strongly-convergent params (thorough mode).
    //     These normally don't need searching, but new GPU architectures may differ.
    if (thorough) {
        for (uint32_t pidx : STRONG_PARAM_INDICES) {
            auto [values, count] = getCandidates(pidx);
            for (uint32_t v = 0; v < count; ++v) {
                if (values[v] == best[pidx]) continue;

                Params8 candidate = best;
                candidate[pidx] = values[v];
                if (!validator.isValid(candidate)) continue;

                // Always reload — 'best' tracks the logically optimal config,
                // not the config buffers were last allocated for (see Phase 2).
                reload_needed = true;
                float t = siever.evaluateConfig(candidate, NUM_SUBCUBES_COARSE, reload_needed);
                ++configs_tested;

                if (t > 0.0f && t < best_time * (1.0f - EPSILON)) {
                    best = candidate;
                    best_time = t;
                    LOG(LOG_DEBUG_2) << "[Autotune] Phase 2.5 (thorough): param[" << pidx
                                     << "] = " << values[v] << " -> " << t << " us";
                }
            }
        }
    }

    // 5. Phase 3: Fine verification with longer mini-benchmark
    reload_needed = true;  // ensure buffers match final config
    float verified_time = siever.evaluateConfig(best, NUM_SUBCUBES_FINE, reload_needed);
    ++configs_tested;

    if (verified_time < 0.0f) verified_time = best_time;  // fallback

    LOG(LOG_INFO) << "[Autotune] Stage 1 complete: ("
                  << best[0] << "," << best[1] << "," << best[2] << ","
                  << best[3] << "," << best[4] << "," << best[5] << ","
                  << best[6] << "," << best[7] << ") = "
                  << verified_time << " us (" << configs_tested << " configs tested)";

    return KernelParamResult{best, verified_time, configs_tested, false};
}

} // namespace mpqs::autotune
