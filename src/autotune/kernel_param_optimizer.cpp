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
    bool thorough,
    uint64_t non_sieve_bytes)
{
    // 1. Build validator from device properties and factoring data
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    SieveConstants sc = buildSieveConstants(
        f_data.a_factors.size(), f_data.M, prop.sharedMemPerBlock);
    KernelLaunchValidator validator(device_id, sc);

    uint32_t configs_tested = 0;

    // OOM guard (S2): a candidate's COMPLETE device footprint (estimateSieveFootprint
    // + the postprocessing/LP + context reserve in non_sieve_bytes) must fit the 0.80
    // budget of free VRAM. Amortize ONE cudaMemGetInfo for the whole Stage-1 — free is
    // ~constant across the sweep (loadSievingDataParamTest frees the bucket before
    // realloc, and FB/primeData stay resident from the initial loadData). The guard is
    // active only when non_sieve_bytes > 0 (set by runStage1_KernelParams) AND the
    // cudaMemGetInfo read succeeds; otherwise the bucket-only isValid gate still applies.
    const uint64_t guard_fb_size = (uint64_t)f_data.factorBase.size();
    uint64_t guard_free_vram = 0;
    bool     guard_active = false;
    if (non_sieve_bytes > 0) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b > 0) {
            guard_free_vram = free_b;
            guard_active = true;
        }
    }
    uint32_t footprint_skips = 0;  ///< candidates skipped by the total-footprint guard
    // Local: candidate fits the total-footprint budget (true when the guard is inactive).
    auto fitsFootprint = [&](const Params8& cand, uint64_t* est_out) -> bool {
        if (!guard_active) return true;
        return validator.fitsTotalFootprint(cand, guard_fb_size, guard_free_vram,
                                            non_sieve_bytes, est_out);
    };

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

    // OOM guard (S2, design §2.4(B)/m2): gate the optimizer's OWN seed `best` through
    // the total-footprint check before the first evaluateConfig (→ loadSievingDataParamTest
    // → kernel.cu:613). `best` passed only the bucket-only isValid above; this is a
    // distinct siever from the runStage1 seed guard, so it needs its own gate. If `best`
    // is over-budget, reduce it to the first feasible enumerated config (footprint-filtered),
    // guaranteeing a survivor. The seed bucket is small for validated configs, so this is a
    // no-op there (logged only when it binds).
    if (guard_active) {
        uint64_t est_total = 0;
        if (!fitsFootprint(best, &est_total)) {
            auto all_valid = enumerateValidConfigs(validator);
            bool found = false;
            for (const auto& cand : all_valid) {
                if (fitsFootprint(cand, nullptr)) { best = cand; found = true; break; }
            }
            LOG(LOG_DEBUG_1) << "[Autotune][OOM-guard] seed eval over budget (est total "
                             << (est_total / (1024 * 1024)) << "MB); "
                             << (found ? "reduced to first feasible enumerated config"
                                       : "NO feasible config — proceeding with smallest seed");
            // If none fit (genuinely under-provisioned), keep `best` as the
            // smallest-bucket enumerated config so evaluateConfig still attempts the
            // minimum; the runStage1 seed guard's fallback already covered the seed path.
            if (!found && !all_valid.empty()) best = all_valid[0];
        }
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
                // OOM guard (S2): skip candidates whose COMPLETE footprint exceeds the
                // 0.80 budget (additive to the bucket-only isValid above). No-op when
                // the guard is inactive or the candidate fits.
                {
                    uint64_t est_total = 0;
                    if (!fitsFootprint(candidate, &est_total)) {
                        ++footprint_skips;
                        LOG(LOG_DEBUG_2) << "[Autotune][OOM-guard] skip candidate param["
                                         << pidx << "]=" << values[v] << " (est total "
                                         << (est_total / (1024 * 1024)) << "MB > budget)";
                        continue;
                    }
                }

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
                // OOM guard (S2): same total-footprint skip as Phase 2.
                {
                    uint64_t est_total = 0;
                    if (!fitsFootprint(candidate, &est_total)) {
                        ++footprint_skips;
                        LOG(LOG_DEBUG_2) << "[Autotune][OOM-guard] skip candidate (thorough) param["
                                         << pidx << "]=" << values[v] << " (est total "
                                         << (est_total / (1024 * 1024)) << "MB > budget)";
                        continue;
                    }
                }

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

    // OOM-guard summary (S2): how many candidates the total-footprint guard skipped.
    // 0 at the validated operating points (no-regression assertion / committed CTest).
    if (guard_active) {
        LOG(LOG_INFO) << "[Autotune][OOM-guard] candidate footprint skips: " << footprint_skips;
    }

    return KernelParamResult{best, verified_time, configs_tested, false};
}

} // namespace mpqs::autotune
