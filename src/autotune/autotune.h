// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

#include "autotune_history.h"
#include "benign_history.h"
#include "autotune_projection.h"
#include "autotune_types.h"
#include "kernel_launch_validator.h"  // for Params8

namespace mpqs { struct MPQSConfig; }
namespace mpqs::sieve { struct factoringData; }

namespace mpqs::autotune {

struct AutotuneConfig {
    bool enable_stage0 = true;
    bool enable_stage1 = true;
    bool enable_stage2 = true;
    bool enable_stage3 = true;
    uint32_t max_iterations = 2;
    double   timeout_sec    = 300.0;
    double   convergence_threshold = 0.02;
    bool     thorough = false;
    double   truncated_run_target_pct = 0.12;
    uint32_t truncated_run_min_samples = 10;
    bool     prefer_legacy_loop = true;
    std::string history_file;
    std::string benign_history_file;  // Path to benign (cross-GPU) history (empty = auto)
    bool        load_history  = true;
    bool        save_history  = true;
    std::string candidates_file;    // Path to candidates.txt for bootstrap (empty = none)
    bool        bootstrap = false;  // --autotune_bootstrap: run bootstrap mode
};

struct AutotuneResult {
    bool     improved = false;
    uint32_t iterations_run = 0;
    double   total_time_sec = 0.0;
    double   initial_throughput_est = 0.0;
    double   final_throughput_est   = 0.0;
    uint32_t best_params[8] = {};
    uint32_t best_F = 0;
    uint32_t best_M = 0;
    uint64_t best_lp1_bound = 0;
    struct StageSummary {
        bool    ran = false;
        double  time_sec = 0.0;
        double  improvement_pct = 0.0;
        std::string notes;
    };
    StageSummary stages[4];
    enum class TermReason { CONVERGED, MAX_ITER, TIMEOUT, HISTORY_HIT, ERROR };
    TermReason termination = TermReason::MAX_ITER;
};

class AutotuneController {
public:
    explicit AutotuneController(const AutotuneConfig& config,
                                MPQSConfig& pipeline_config,
                                mpqs::sieve::factoringData& f_data);
    bool run();
    const AutotuneResult& getResult() const;
    std::vector<uint32_t> stagesRun() const;
    double finalConfidence() const;

private:
    void runStage0_Projection();
    void runStage1_KernelParams();
    void runStage2_RuntimeEstimation();
    void runStage3_SieveParams();
    bool loadHistory();
    bool saveHistory();
    bool hasTimedOut() const;
    double elapsed() const;

    AutotuneConfig  atcfg_;
    MPQSConfig&     pipeline_config_;
    mpqs::sieve::factoringData& f_data_;
    AutotuneResult  result_;
    std::chrono::high_resolution_clock::time_point start_time_;

    // --- N metadata (computed once in constructor) ---
    HistoryStore history_;
    BenignHistoryStore benign_history_;
    std::string  N_hash_;           // SHA-256 of N decimal string
    std::string  gpu_name_;         // cudaDeviceProp::name
    std::string  gpu_cc_;           // "major.minor" compute capability
    std::string  cuda_version_;     // "major.minor" CUDA runtime version
    uint32_t     N_bits_   = 0;     // f_data_.N bit-length
    uint32_t     N_digits_ = 0;     // f_data_.N digit count

    // --- Stage 0 output (consumed by Stages 1-3) ---
    ProjectedParams projected_;
    double projection_confidence_ = 0.0;

    // --- Stage 1 output ---
    Params8  best_kernel_params_ = {};
    float    best_kernel_timing_us_ = 0.0f;

    // --- Stage 2/3 state ---
    RuntimeEstimate   last_estimate_;
    SieveSearchBounds search_bounds_;

    // --- Tracking for Stage 1 skip logic (F/M at last Stage 1 run) ---
    uint32_t F_at_last_stage1_ = 0;
    uint32_t M_at_last_stage1_ = 0;

    // --- Convergence tracking: total_est_sec from previous iteration ---
    double prev_total_est_sec_ = 0.0;

    // --- Stage 3 convergence tracking (skip re-run if converged) ---
    bool stage3_converged_ = false;

    // --- Heuristic F from TuningStage (used for F-cap in joint optimizer) ---
    uint32_t heuristic_fb_bound_ = 0;

    // --- Helpers ---
    void regenerateFactorBase();
    void applyToConfig(uint32_t fb_bound, uint32_t sieve_bound, uint64_t lp1_bound);
    void applyBufferRecommendations();
    void printBufferRecommendations();
};

} // namespace mpqs::autotune
