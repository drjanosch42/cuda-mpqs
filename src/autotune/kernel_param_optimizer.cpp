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
#include <chrono>   // S3/S4 wide-path per-candidate wall-clock budget management
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

CandidateRange getCandidates(uint32_t param_idx, bool use_wide = false) {
    // S5 (WIDE only): subCubeSize (pidx 0) drops the 1024 candidate — clampWideNumPolys caps
    // num_polys at 512, so probing 1024 re-tests the 512 geometry. Every other dim shares the
    // narrow candidate arrays (the wide sweep just visits a SUBSET of indices; see the
    // WIDE_*_PARAM_INDICES lists), so the arrays themselves are unchanged for them.
    if (use_wide && param_idx == 0) return {WIDE_CANDIDATE_VALUES_0, 3};
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
    uint64_t non_sieve_bytes,
    double probe_budget_sec)
{
    // 1. Build validator from device properties and factoring data.
    //    S2 (wide-autotune foundation): build the WIDE geometry when the siever resolved to
    //    the uint16 accumulator, so the pre-filter (isValid / OOM guard) ranks each config as
    //    it will actually run. Narrow (uint8) -> use_wide=false -> byte-identical to pre-S2.
    const bool use_wide = siever.isWideAccumulator();

    // ----- S3/S4 (WIDE only): the objective is a survivors/sec RATE (higher=better) measured
    //       over a wall-clock window that auto-scales and honours the autotune budget. All of
    //       the following is gated on use_wide; the NARROW path never touches it and keeps its
    //       isolated-kernel-µs objective + lower-is-better accept byte-for-byte. -----
    using wclk = std::chrono::steady_clock;
    const auto wide_t0 = wclk::now();
    const bool wide_budgeted = use_wide && probe_budget_sec > 0.0;
    auto wideElapsed = [&]() {
        return std::chrono::duration<double>(wclk::now() - wide_t0).count();
    };
    // Reserve enough tail budget for the mandatory S4 floor eval (warmup + a floor window).
    const double wide_floor_reserve = WIDE_PROBE_WARMUP_SEC + WIDE_PROBE_WINDOW_FLOOR_SEC + 5.0;
    // Set this candidate's window from the remaining budget. Returns false ⇒ out of budget:
    // stop adding search candidates (the floor is still timed afterward). On narrow: no-op/true.
    auto prepareWideWindow = [&]() -> bool {
        if (!use_wide) return true;
        double window = WIDE_PROBE_WINDOW_SEC;
        if (wide_budgeted) {
            const double remaining   = probe_budget_sec - wideElapsed();
            const double this_window = (remaining - wide_floor_reserve) - WIDE_PROBE_WARMUP_SEC;
            if (this_window < WIDE_PROBE_WINDOW_FLOOR_SEC) return false;  // reserve the floor
            window = std::min(WIDE_PROBE_WINDOW_SEC, this_window);
            window = std::max(WIDE_PROBE_WINDOW_FLOOR_SEC, window);
        }
        siever.setProbeWindow(window, WIDE_PROBE_WARMUP_SEC);
        return true;
    };
    // WIDE accept: rate, higher-is-better. NARROW accept: µs, lower-is-better (unchanged).
    auto wideBetter = [&](float t, float best) {
        return use_wide ? (t > 0.0f && t > best * (1.0f + EPSILON))
                        : (t > 0.0f && t < best * (1.0f - EPSILON));
    };
    bool wide_budget_stop = false;  // set once the search runs out of budget (floor still runs)

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    SieveConstants sc = buildSieveConstants(
        f_data.a_factors.size(), f_data.M, prop.sharedMemPerBlock, use_wide,
        siever.isWideU8Sat());  // Option A: match the saturating-uint8 wide geometry when selected
    KernelLaunchValidator validator(device_id, sc);

    // On wide, loadPartialCustomConfig OVERRIDES num_sievingBlocksPerSieveCall to 2M/SB_wide
    // (the [C1] single-batch coverage invariant), so the tuple's numIntervals (p[1]) is inert.
    // Canonicalize p[1] to that value on every candidate the validator sees, so the pre-filter
    // matches the runtime geometry. 0 only if sievingBlockSize is 0 (never for a real device).
    const uint32_t canonical_intervals =
        (use_wide && sc.sievingBlockSize) ? (2u * sc.M) / sc.sievingBlockSize : 0u;

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

    // ----- S5 (WIDE only): occupancy-seeded start. -------------------------------------------
    // Descending from HEURISTIC_DEFAULTS on wide starts in the WRONG geometry region (SCATTER
    // blockDim 1024, no A1 SM-aware SCATTER grid, GATHER blockDim hardcoded 1024) — so the search
    // never reaches, hence never beats, the shipping loadStandardConfig-wide default, and the S4
    // floor gate always falls back. Instead SEED the search AT the known-good default: read the
    // live loadStandardConfig-wide tuple (the autotune caller ran loadStandardConfig() +
    // loadData() before this optimizer, so gs/gms/ss ARE the standard-wide geometry) and override
    // ONLY the GATHER blockDim to the occupancy knee of sieveAndScanBatchKernelWide (API-chosen,
    // not hardcoded — H100 1/SM→1024, A100 2/SM→1024=100%). The descent then explores the wide
    // dims OUTWARD from a config that already matches the floor except for the occupancy win.
    // Narrow keeps HEURISTIC_DEFAULTS verbatim (this whole block is use_wide-gated).
    if (use_wide) {
        const auto gs_std  = siever.getGeneralConfig();
        const auto gms_std = siever.getMetaSieveConfig();
        const auto ss_std  = siever.getSieveAndScanConfig();
        // Guard: only adopt the live tuple if the siever is actually configured (all real
        // callers ran loadStandardConfig first; a 0 here means it was not — keep HEURISTIC).
        if (gs_std.num_polysPerSieveCall != 0 && gms_std.num_threadBlocks != 0
            && gms_std.num_threadsPerBlock != 0 && ss_std.num_threadBlocks != 0
            && ss_std.num_threadsPerBlock != 0 && gms_std.polyBlockSize != 0
            && gms_std.num_activeBlocksPerCycle != 0) {
            best[P_SUB_CUBE_SIZE]   = gs_std.num_polysPerSieveCall;         // == num_polys (<=512 on wide)
            best[P_NUM_INTERVALS]   = gs_std.num_sievingBlocksPerSieveCall; // canonicalized below (inert on wide)
            best[P_POLY_BLOCK_SIZE] = gms_std.polyBlockSize;
            best[P_BLOCKS_PER_CYC]  = gms_std.num_activeBlocksPerCycle;     // divides runtime 2M/SB_wide (single meta cycle)
            best[P_META_GRID_DIM]   = gms_std.num_threadBlocks;            // A1 SM-aware SCATTER grid
            best[P_META_BLOCK_DIM]  = gms_std.num_threadsPerBlock;         // standard SCATTER blockDim (256)
            best[P_SAS_GRID_DIM]    = ss_std.num_threadBlocks;            // standard GATHER grid (min(256,num_polys))
            best[P_SAS_BLOCK_DIM]   = ss_std.num_threadsPerBlock;         // standard GATHER blockDim — overridden next
            static constexpr uint32_t kWideGatherBlockDims[] = {256u, 512u, 1024u};
            const uint32_t occ_bd = siever.wideGatherOccupancyBlockDim(kWideGatherBlockDims, 3);
            if (occ_bd != 0) best[P_SAS_BLOCK_DIM] = occ_bd;
            LOG(LOG_INFO) << "[Autotune][wide-seed] occupancy-optimal GATHER blockDim="
                          << best[P_SAS_BLOCK_DIM]
                          << (occ_bd != 0 ? " (cudaOccupancy)" : " (API unavailable; kept standard)")
                          << "; seeded at loadStandardConfig-wide: num_polys=" << best[P_SUB_CUBE_SIZE]
                          << " SCATTER(grid=" << best[P_META_GRID_DIM]
                          << ",blockDim=" << best[P_META_BLOCK_DIM]
                          << ") GATHER(grid=" << best[P_SAS_GRID_DIM]
                          << ",blockDim=" << best[P_SAS_BLOCK_DIM]
                          << ") blocksPerCycle=" << best[P_BLOCKS_PER_CYC]
                          << " polyBlockSize=" << best[P_POLY_BLOCK_SIZE];
        } else {
            LOG(LOG_WARNING) << "[Autotune][wide-seed] siever not standard-configured; "
                                "seeding from HEURISTIC_DEFAULTS (occupancy seed skipped)";
        }
    }

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

    // Wide: force the seed's numIntervals to the runtime-canonical value (see above).
    if (use_wide) best[P_NUM_INTERVALS] = canonical_intervals;

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

    // 3. Evaluate seed with coarse mini-benchmark.
    //    WIDE (S3): set the probe window first. If we are already budget-tight, evaluate the
    //    seed at the floor window (it is the mandatory search baseline) and skip the sweep,
    //    heading straight to the S4 floor gate.
    if (use_wide && !prepareWideWindow()) {
        siever.setProbeWindow(WIDE_PROBE_WINDOW_FLOOR_SEC, WIDE_PROBE_WARMUP_SEC);
        wide_budget_stop = true;
    }
    bool reload_needed = true;
    float best_time = siever.evaluateConfig(best, NUM_SUBCUBES_COARSE, reload_needed);
    if (best_time < 0.0f) {
        LOG(LOG_ERROR_CRITICAL) << "[Autotune] Seed config evaluation failed!";
        return KernelParamResult{best, 0.0f, 0, false};
    }
    ++configs_tested;

    LOG(LOG_DEBUG_1) << "[Autotune] Phase 1 seed: "
                     << (use_wide ? "rate = " : "timing = ") << best_time
                     << (use_wide ? " survivors/s" : " us");

    // 4. Phase 2: Coordinate descent on weakly-convergent parameters
    //    NARROW: sweep sasGridDim (6), metaGridDim (4), polyBlockSize (2).
    //    WIDE (S5): sweep GATHER blockDim (7), SCATTER grid (4), subCubeSize/num_polys (0) —
    //    the throughput-relevant wide dims — starting from the occupancy-seeded standard-wide
    //    config; the inert numIntervals (1) is not in the wide list, so it is never swept.
    //    These have nearly independent, unimodal 1D slices.
    const uint32_t* weak_idx = use_wide ? WIDE_WEAK_PARAM_INDICES : WEAK_PARAM_INDICES;
    const uint32_t  weak_cnt = use_wide
        ? (uint32_t)(sizeof(WIDE_WEAK_PARAM_INDICES) / sizeof(uint32_t))
        : (uint32_t)(sizeof(WEAK_PARAM_INDICES) / sizeof(uint32_t));
    bool improved = true;
    while (improved && !wide_budget_stop) {
        improved = false;
        for (uint32_t wi = 0; wi < weak_cnt; ++wi) {
            const uint32_t pidx = weak_idx[wi];
            if (wide_budget_stop) break;
            auto [values, count] = getCandidates(pidx, use_wide);
            for (uint32_t v = 0; v < count; ++v) {
                if (values[v] == best[pidx]) continue;

                Params8 candidate = best;
                candidate[pidx] = values[v];
                // Wide: numIntervals is inert (runtime overrides it); keep it canonical so the
                // pre-filter and the real config agree (incl. the pidx==1 thorough sweep).
                if (use_wide) candidate[P_NUM_INTERVALS] = canonical_intervals;
                if (!validator.isValid(candidate)) continue;
                // WIDE (S3): size this candidate's probe window from the remaining budget;
                // out of budget ⇒ stop searching (the S4 floor is still timed afterward).
                if (use_wide && !prepareWideWindow()) { wide_budget_stop = true; break; }
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

                if (wideBetter(t, best_time)) {
                    best = candidate;
                    best_time = t;
                    improved = true;
                    LOG(LOG_DEBUG_2) << "[Autotune] Phase 2: param[" << pidx
                                     << "] = " << values[v] << " -> " << t
                                     << (use_wide ? " survivors/s" : " us");
                }
            }
        }
    }

    LOG(LOG_DEBUG_1) << "[Autotune] Phase 2 complete: "
                     << configs_tested << " configs tested, best = " << best_time << " us";

    // 4.5 Optional: sweep strongly-convergent params (thorough mode).
    //     NARROW: {sasBlockDim, metaBlockDim, numIntervals, blocksPerCycle, subCubeSize}.
    //     WIDE (S5): {metaBlockDim} only — every other wide-relevant dim is already in Phase 2,
    //     and numIntervals (1) is inert on wide. These normally don't need searching, but new
    //     GPU architectures may differ.
    const uint32_t* strong_idx = use_wide ? WIDE_STRONG_PARAM_INDICES : STRONG_PARAM_INDICES;
    const uint32_t  strong_cnt = use_wide
        ? (uint32_t)(sizeof(WIDE_STRONG_PARAM_INDICES) / sizeof(uint32_t))
        : (uint32_t)(sizeof(STRONG_PARAM_INDICES) / sizeof(uint32_t));
    if (thorough && !wide_budget_stop) {
        for (uint32_t si = 0; si < strong_cnt; ++si) {
            const uint32_t pidx = strong_idx[si];
            if (wide_budget_stop) break;
            auto [values, count] = getCandidates(pidx, use_wide);
            for (uint32_t v = 0; v < count; ++v) {
                if (values[v] == best[pidx]) continue;

                Params8 candidate = best;
                candidate[pidx] = values[v];
                if (use_wide) candidate[P_NUM_INTERVALS] = canonical_intervals;  // inert on wide (see Phase 2)
                if (!validator.isValid(candidate)) continue;
                // WIDE (S3): budget-size this candidate's window (see Phase 2).
                if (use_wide && !prepareWideWindow()) { wide_budget_stop = true; break; }
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

                if (wideBetter(t, best_time)) {
                    best = candidate;
                    best_time = t;
                    LOG(LOG_DEBUG_2) << "[Autotune] Phase 2.5 (thorough): param[" << pidx
                                     << "] = " << values[v] << " -> " << t
                                     << (use_wide ? " survivors/s" : " us");
                }
            }
        }
    }

    // 5. Phase 3: Fine verification with a longer mini-benchmark.
    //    WIDE (S3/S4): measure the winner AND the loadStandardConfig-wide floor with ONE
    //    consistent decision window, so the S4 accept comparison is apples-to-apples. Choose it
    //    from the remaining budget: the full window if two more evals (verify + floor) fit, else
    //    the floor window (the floor MUST still be timed even over budget — the parity guarantee).
    if (use_wide) {
        double dw = WIDE_PROBE_WINDOW_SEC;
        if (wide_budgeted) {
            const double remaining = probe_budget_sec - wideElapsed();
            const double per_eval  = remaining / 2.0 - WIDE_PROBE_WARMUP_SEC;  // verify + floor
            dw = std::max(WIDE_PROBE_WINDOW_FLOOR_SEC,
                          std::min(WIDE_PROBE_WINDOW_SEC, per_eval));
        }
        siever.setProbeWindow(dw, WIDE_PROBE_WARMUP_SEC);
    }
    reload_needed = true;  // ensure buffers match final config
    float verified_time = siever.evaluateConfig(best, NUM_SUBCUBES_FINE, reload_needed);
    ++configs_tested;

    if (verified_time < 0.0f) verified_time = best_time;  // fallback

    // 5b. S4 FLOOR GATE (WIDE only). Time loadStandardConfig-wide (the exact config that ships
    //     when autotune is OFF) under the SAME decision window, and accept the search winner IFF
    //     its rate ≥ floor·(1+WIDE_FLOOR_MARGIN). Otherwise the caller keeps loadStandardConfig
    //     verbatim (useParams=false) — worst case is provably parity with the shipping wide
    //     default, so the wide-autotune −73% regression cannot recur.
    bool  beats_floor = true;
    float floor_score = 0.0f;
    if (use_wide) {
        floor_score = siever.sieveMiniStandardWide();  // survivors/sec of the standard wide config
        ++configs_tested;
        if (floor_score > 0.0f) {
            beats_floor = (verified_time >= floor_score * (1.0f + WIDE_FLOOR_MARGIN));
            const double beat_pct = 100.0 * ((double)verified_time / (double)floor_score - 1.0);
            LOG(LOG_INFO) << "[Autotune][floor] winner " << verified_time
                          << " survivors/s vs loadStandardConfig-wide floor " << floor_score
                          << " survivors/s (winner " << (beats_floor ? "beats" : "does NOT beat")
                          << " floor by " << beat_pct << "%, required margin +"
                          << (100.0f * WIDE_FLOOR_MARGIN) << "%)";
        } else if (verified_time > 0.0f) {
            // Floor unmeasurable (setup/validation issue) but the winner has a REAL positive
            // rate: do NOT block it, but flag that no parity floor was enforced this run.
            LOG(LOG_WARNING) << "[Autotune][floor] floor rate unavailable; applying winner on "
                                "positive measured rate (no parity floor enforced this run)";
        } else {
            // No-signal condition: BOTH the winner and the floor measured a non-positive
            // rate (0 survivors/s — e.g. LP off / too-tight threshold at this scale, so no
            // full smooths survived any candidate). Treat as "not beaten" so the caller keeps
            // loadStandardConfig — do NOT imply a positive verdict on a zero rate.
            beats_floor = false;
            LOG(LOG_WARNING) << "[Autotune][floor] no signal: winner rate " << verified_time
                             << " survivors/s and floor rate unavailable -> not applying "
                                "(keeping loadStandardConfig)";
        }
    }

    LOG(LOG_INFO) << "[Autotune] Stage 1 complete: ("
                  << best[0] << "," << best[1] << "," << best[2] << ","
                  << best[3] << "," << best[4] << "," << best[5] << ","
                  << best[6] << "," << best[7] << ") = "
                  << verified_time << (use_wide ? " survivors/s (" : " us (")
                  << configs_tested << " configs tested)";

    // OOM-guard summary (S2): how many candidates the total-footprint guard skipped.
    // 0 at the validated operating points (no-regression assertion / committed CTest).
    if (guard_active) {
        LOG(LOG_INFO) << "[Autotune][OOM-guard] candidate footprint skips: " << footprint_skips;
    }

    KernelParamResult result{best, verified_time, configs_tested, false};
    result.objective_is_rate = use_wide;   // WIDE: timing_us is a survivors/sec RATE
    result.beats_floor       = beats_floor;
    result.floor_score       = floor_score;
    return result;
}

} // namespace mpqs::autotune
