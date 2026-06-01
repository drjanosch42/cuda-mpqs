// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "autotune.h"
#include "orchestrator.h"            // MPQSConfig full definition
#include "hpc_logger.h"
#include "autotune_history.h"
#include "autotune_types.h"
#include "kernel_launch_validator.h"
#include "kernel_param_optimizer.h"
#include "device_sieving_controller.h"
#include "prime_algorithms.h"         // generateFactorBase, init_a_factors
#include "memory_estimator.h"          // shared GPU memory cost constants

#include "runtime_estimator.h"
#include "sieve_optimizer.h"
#include "autotune_projection.h"
#include "cost_models.h"

#include <cuda_runtime.h>            // cudaGetDeviceProperties, cudaRuntimeGetVersion
#include <cmath>
#include <iomanip>

namespace mpqs::autotune {

using Clock = std::chrono::high_resolution_clock;
inline double duration(std::chrono::high_resolution_clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

/// Relative change between two values; returns 1.0 if b == 0.
static double relChange(double a, double b) {
    return (b > 0.0) ? std::abs(a - b) / b : 1.0;
}

AutotuneController::AutotuneController(const AutotuneConfig& config,
                                       MPQSConfig& pipeline_config,
                                       mpqs::sieve::factoringData& f_data)
    : atcfg_(config)
    , pipeline_config_(pipeline_config)
    , f_data_(f_data)
    , result_{}
    , start_time_{}
{
    // Compute N metadata for history lookup and projection
    N_bits_   = f_data_.N.msb() + 1;
    N_digits_ = f_data_.N.to_string().size();
    N_hash_   = sha256_hex(f_data_.N.to_string());

    // Query GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, pipeline_config_.device_id);
    gpu_name_ = prop.name;
    gpu_cc_   = std::to_string(prop.major) + "." + std::to_string(prop.minor);

    int cuda_rt_version = 0;
    cudaRuntimeGetVersion(&cuda_rt_version);
    cuda_version_ = std::to_string(cuda_rt_version / 1000) + "."
                  + std::to_string((cuda_rt_version % 1000) / 10);

    LOG(LOG_DEBUG_1) << "N: " << N_digits_ << " digits, "
                     << N_bits_ << " bits, hash=" << N_hash_.substr(0, 12) << "..."
                     << " GPU=" << gpu_name_ << " CC=" << gpu_cc_
                     << " CUDA=" << cuda_version_;
}

void AutotuneController::regenerateFactorBase() {
    f_data_.F = pipeline_config_.fb_bound;
    generateFactorBase(&f_data_);
    init_a_factors(&f_data_);
    pipeline_config_.target_relations = f_data_.size + (f_data_.size / 20) + 64;
}

void AutotuneController::applyToConfig(uint32_t fb, uint32_t M, uint64_t lp) {
    bool F_changed = false;
    if (fb > 0 && !pipeline_config_.isPinned("fb_bound")) {
        F_changed = (pipeline_config_.fb_bound != fb);
        pipeline_config_.fb_bound = fb;
    }
    if (M > 0 && !pipeline_config_.isPinned("sieve_bound"))
        pipeline_config_.sieve_bound = M;
    if (!pipeline_config_.isPinned("lp1_bound"))
        pipeline_config_.lp1_bound = lp;

    if (F_changed) regenerateFactorBase();
}

bool AutotuneController::run() {
    LOG_SCOPED_MODULE("Autotune");
    LOG_SET_STAGE(LOG_STAGE_AUTOTUNE);

    // Skip autotuning for small N — heuristic parameters are sufficient below 70 digits
    constexpr uint32_t kMinAutotuneDigits = 70;  // ~232 bits
    if (N_digits_ < kMinAutotuneDigits) {
        LOG(LOG_WARNING) << "Skipped: N is " << N_digits_
                         << " digits (< " << kMinAutotuneDigits
                         << ") — using heuristic parameters";
        result_.termination = AutotuneResult::TermReason::CONVERGED;
        result_.total_time_sec = 0.0;
        return false;
    }

    LOG(LOG_INFO) << "Starting autotune procedure";
    start_time_ = std::chrono::high_resolution_clock::now();

    // History short-circuit
    if (atcfg_.load_history && loadHistory()) {
        // loadHistory() applied F, M, L, kernel_params. Now also apply buffer recs.
        const HistoryEntry* exact = history_.findExact(N_hash_);
        if (exact) {
            last_estimate_.recommended_witness_capacity =
                exact->optimal_params.recommended_witness_capacity;
            last_estimate_.recommended_partial_buffer =
                exact->optimal_params.recommended_partial_buffer;
            last_estimate_.recommended_accum_buffer =
                exact->optimal_params.recommended_accum_buffer;
            // Set total_est_sec nonzero so applyBufferRecommendations() doesn't bail
            last_estimate_.total_est_sec =
                exact->measured_performance.total_time_sec > 0
                ? exact->measured_performance.total_time_sec : 1.0;
        }
        applyBufferRecommendations();

        result_.total_time_sec = elapsed();
        return result_.improved;
    }

    // Capture heuristic F (from TuningStage) as the baseline for F-cap.
    heuristic_fb_bound_ = pipeline_config_.fb_bound;

    // Track global best across iterations to prevent regression
    double best_total_est_sec = std::numeric_limits<double>::max();
    MPQSConfig best_config;
    bool have_best = false;

    for (uint32_t iter = 0; iter < atcfg_.max_iterations; ++iter) {
        // Clear any sticky CUDA errors from previous iteration's probes
        cudaDeviceSynchronize();
        cudaGetLastError();

        if (hasTimedOut()) {
            result_.termination = AutotuneResult::TermReason::TIMEOUT;
            break;
        }

        LOG(LOG_DEBUG_1) << "--- Iteration " << iter << " ---"
                         << " F=" << pipeline_config_.fb_bound
                         << " M=" << pipeline_config_.sieve_bound
                         << " L=" << pipeline_config_.lp1_bound;

        // Stage 0: Projection (first iteration only)
        if (atcfg_.enable_stage0 && iter == 0)
            runStage0_Projection();

        if (hasTimedOut()) {
            result_.termination = AutotuneResult::TermReason::TIMEOUT;
            LOG(LOG_INFO) << "Timeout after Stage 0, iter " << iter;
            result_.iterations_run = iter + 1;
            break;
        }

        // Stage 1: Kernel launch params (skip if F/M unchanged)
        if (atcfg_.enable_stage1) {
            bool skip_stage1 = false;
            if (iter > 0 && best_kernel_timing_us_ > 0.0f
                && F_at_last_stage1_ > 0 && M_at_last_stage1_ > 0) {
                double F_change = relChange(pipeline_config_.fb_bound, F_at_last_stage1_);
                double M_change = relChange(pipeline_config_.sieve_bound, M_at_last_stage1_);
                if (F_change < 0.10 && M_change < 0.10) {
                    skip_stage1 = true;
                    LOG(LOG_DEBUG_1) << "Stage 1: Skipped (F/M unchanged)";
                    result_.stages[1].ran = true;
                    result_.stages[1].notes = "skipped (F/M unchanged)";
                }
            }
            if (!skip_stage1) {
                runStage1_KernelParams();
                F_at_last_stage1_ = pipeline_config_.fb_bound;
                M_at_last_stage1_ = pipeline_config_.sieve_bound;
            }
        }

        if (hasTimedOut()) {
            result_.termination = AutotuneResult::TermReason::TIMEOUT;
            LOG(LOG_INFO) << "Timeout after Stage 1, iter " << iter;
            result_.iterations_run = iter + 1;
            break;
        }

        // Stage 2: Runtime estimation (skip on iter>0 when Stage 3 active)
        if (atcfg_.enable_stage2) {
            bool skip_stage2 = (iter > 0 && atcfg_.enable_stage3);
            if (skip_stage2) {
                LOG(LOG_DEBUG_1) << "Stage 2: Skipped (Stage 3 active, iter > 0)";
                result_.stages[2].notes = "skipped (Stage 3 measures throughput)";
            } else {
                runStage2_RuntimeEstimation();
                // Capture initial throughput on first iteration
                if (iter == 0 && result_.initial_throughput_est == 0.0)
                    result_.initial_throughput_est = last_estimate_.relations_per_sec;
            }
        }

        if (hasTimedOut()) {
            result_.termination = AutotuneResult::TermReason::TIMEOUT;
            LOG(LOG_INFO) << "Timeout after Stage 2, iter " << iter;
            result_.iterations_run = iter + 1;
            break;
        }

        // Stage 3: Sieve parameter sweep (skip if already converged on prior iteration)
        if (atcfg_.enable_stage3) {
            if (iter > 0 && stage3_converged_) {
                LOG(LOG_DEBUG_1) << "Stage 3: Skipped (converged on prior iteration)";
                result_.stages[3].notes = "skipped (prior convergence)";
            } else {
                runStage3_SieveParams();
            }
        }

        // Track global best: snapshot config if this iteration improved
        if (last_estimate_.total_est_sec > 0.0
            && last_estimate_.total_est_sec < best_total_est_sec) {
            best_total_est_sec = last_estimate_.total_est_sec;
            best_config = pipeline_config_;
            have_best = true;
        }

        // Convergence check: compare total_est_sec across iterations
        if (iter > 0 && last_estimate_.total_est_sec > 0.0
            && prev_total_est_sec_ > 0.0) {
            double improvement = std::abs(prev_total_est_sec_ - last_estimate_.total_est_sec)
                               / prev_total_est_sec_;
            LOG(LOG_DEBUG_1) << "Iter " << iter
                             << ": total_est " << std::fixed << std::setprecision(1)
                             << last_estimate_.total_est_sec << "s"
                             << " (prev " << prev_total_est_sec_ << "s"
                             << ", delta " << std::setprecision(3) << improvement * 100.0 << "%)";
            if (improvement < atcfg_.convergence_threshold) {
                result_.termination = AutotuneResult::TermReason::CONVERGED;
                LOG(LOG_INFO) << "Converged at iter " << iter
                              << ": delta=" << std::fixed << std::setprecision(3)
                              << improvement * 100.0 << "% < "
                              << atcfg_.convergence_threshold * 100.0 << "% threshold";
                result_.iterations_run = iter + 1;
                break;
            }
        }
        prev_total_est_sec_ = last_estimate_.total_est_sec;

        // Per-iteration verbose summary
        LOG(LOG_DEBUG_1) << "Iter " << iter << " complete:"
                         << " F=" << pipeline_config_.fb_bound
                         << " M=" << pipeline_config_.sieve_bound
                         << " L=" << pipeline_config_.lp1_bound
                         << " total_est=" << std::fixed << std::setprecision(1)
                         << last_estimate_.total_est_sec << "s"
                         << " throughput=" << std::setprecision(1)
                         << last_estimate_.relations_per_sec << " rels/s";

        result_.iterations_run = iter + 1;
    }

    // Restore best iteration's config if final iteration regressed
    if (have_best && best_total_est_sec < last_estimate_.total_est_sec) {
        LOG(LOG_WARNING) << "Restoring best iteration (T_est="
                         << std::fixed << std::setprecision(1) << best_total_est_sec
                         << "s vs final " << last_estimate_.total_est_sec << "s)";
        pipeline_config_ = best_config;
    }

    if (atcfg_.save_history)
        saveHistory();

    // Apply buffer recommendations to pipeline config
    applyBufferRecommendations();
    printBufferRecommendations();

    result_.total_time_sec = elapsed();

    static const char* term_names[] = { "CONVERGED", "MAX_ITER", "TIMEOUT", "HISTORY_HIT", "ERROR" };
    LOG(LOG_INFO) << "Complete: " << result_.iterations_run
                  << " iterations, " << result_.total_time_sec << "s elapsed"
                  << " (" << term_names[static_cast<int>(result_.termination)] << ")";

    return result_.improved;
}

// ---------------------------------------------------------------------------
// Stage stubs
// ---------------------------------------------------------------------------

void AutotuneController::runStage0_Projection() {
    auto t0 = Clock::now();

    // Load benign history if not yet loaded
    if (benign_history_.size() == 0 && !atcfg_.benign_history_file.empty()) {
        if (!benign_history_.load(atcfg_.benign_history_file)) {
            benign_history_.loadDefaults();
            benign_history_.save(atcfg_.benign_history_file);
            LOG(LOG_DEBUG_1) << "Seeded benign history with "
                             << benign_history_.size() << " defaults";
        }
    }

    // Cold start: enrich history with synthetic entry from benign data
    if (history_.size() == 0) {
        const auto* benign = benign_history_.findByDigits(N_digits_);
        if (benign) {
            HistoryEntry synthetic;
            synthetic.N_decimal     = "";  // Not a real N
            synthetic.N_hash_sha256 = "benign_" + std::to_string(benign->bit_length);
            synthetic.digit_count   = (benign->digit_count_lo + benign->digit_count_hi) / 2;
            synthetic.bit_length    = benign->bit_length;
            synthetic.optimal_params.fb_bound    = benign->fb_bound;
            synthetic.optimal_params.sieve_bound = benign->sieve_bound;
            synthetic.optimal_params.lp1_bound   = benign->lp1_bound;
            // kernel_params all zero — triggers auto-calculate
            synthetic.measured_performance = {};
            synthetic.environment.gpu_name = "benign";  // Never matches for kernel params
            synthetic.confidence = benign->confidence;   // 0.4 — lower than local data
            synthetic.timestamp  = iso8601_now();

            history_.upsert(synthetic);

            LOG(LOG_INFO) << "Stage 0: Cold start — injected benign entry for "
                          << synthetic.digit_count << "-digit range"
                          << " (F=" << benign->fb_bound
                          << " M=" << benign->sieve_bound
                          << " L=" << benign->lp1_bound
                          << " confidence=" << benign->confidence << ")";
        }
    }

    // 1. Build projector from loaded history
    ParameterProjector projector(history_);

    // 2. Project
    projected_ = projector.project(N_bits_, N_digits_, N_hash_, gpu_name_);
    projection_confidence_ = projected_.confidence;

    // Fix 4: Break LP feedback loop — don't inherit L=0 for large N.
    // When projection gives L=0 for N ≥ 290 bits (≈88 digits), LP is
    // empirically beneficial but probes can't observe it (birthday-paradox
    // scaling). Override with F*50 heuristic to seed coordinate descent.
    if (N_bits_ >= 290 && projected_.lp1_bound == 0) {
        projected_.lp1_bound = static_cast<uint64_t>(projected_.fb_bound) * 50;
        LOG(LOG_DEBUG_1) << "Overriding L=0 for " << N_bits_
                         << "-bit N: LP bound set to F*50 = " << projected_.lp1_bound;
    }

    // 3. Apply projected sieve params to pipeline config
    applyToConfig(projected_.fb_bound, projected_.sieve_bound, projected_.lp1_bound);

    // 4. Derive search bounds from projection for Stage 3
    search_bounds_ = deriveSieveSearchBounds(
        pipeline_config_.fb_bound, pipeline_config_.sieve_bound,
        heuristic_fb_bound_);
    // Narrow bounds if confidence is high
    if (projected_.confidence > 0.6) {
        double r = projected_.fb_bound_search_radius_pct;
        search_bounds_.fb_lo = static_cast<uint32_t>(pipeline_config_.fb_bound * (1.0 - r));
        search_bounds_.fb_hi = static_cast<uint32_t>(pipeline_config_.fb_bound * (1.0 + r));
    }

    // 5. Log
    LOG(LOG_INFO) << "Stage 0: Projected F=" << projected_.fb_bound
                  << " M=" << projected_.sieve_bound
                  << " L=" << projected_.lp1_bound
                  << " (source=" << static_cast<int>(projected_.source)
                  << ", confidence=" << projected_.confidence << ")";

    // 6. Record
    result_.stages[0].ran = true;
    result_.stages[0].time_sec = duration(Clock::now() - t0);
    result_.stages[0].notes = "source=" + std::to_string(static_cast<int>(projected_.source))
                            + " confidence=" + std::to_string(projected_.confidence);
}

void AutotuneController::runStage1_KernelParams() {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Create ephemeral DeviceSievingController for mini-benchmarking
    auto siever = std::make_unique<mpqs::sieve::DeviceSievingController>(
        pipeline_config_.device_id);
    siever->initiate(f_data_);
    siever->loadStandardConfig();
    siever->loadData();
    siever->updateState();

    // 2. Run coordinate descent optimizer
    auto kp_result = optimizeKernelLaunchParams(
        *siever, f_data_, pipeline_config_.device_id, atcfg_.thorough);

    // 3. Tear down siever — reset unique_ptr so destructor runs exactly once
    //    (clearSievingBuffers() + destructor = double-free, same as TruncatedSieveRun bug)
    siever.reset();

    // Clear sticky CUDA errors from Stage 1 kernel benchmarking
    cudaDeviceSynchronize();
    cudaGetLastError();

    // 4. Defense in depth: verify the winning config passes preflight
    //    before applying to the pipeline. Should never fail (optimizer uses
    //    isValid() internally), but guards against validator/optimizer bugs.
    if (kp_result.timing_us > 0.0f) {
        auto pf = preflightKernelLaunch(kp_result.params,
            static_cast<uint32_t>(f_data_.a_factors.size()), f_data_.M,
            pipeline_config_.device_id);
        if (!pf.feasible) {
            LOG(LOG_WARNING) << "Stage 1 winner failed preflight: "
                          << pf.reason
                          << " -- falling back to heuristic defaults";
            pipeline_config_.useParams = false;
            result_.stages[1].notes = "preflight fallback: " + pf.reason;
            result_.stages[1].ran = true;
            result_.stages[1].time_sec = duration(Clock::now() - t0);
            return;
        }

        // Preflight passed — apply optimized params
        for (int i = 0; i < 8; ++i)
            pipeline_config_.params[i] = kp_result.params[i];
        pipeline_config_.useParams = true;
        best_kernel_params_ = kp_result.params;
        best_kernel_timing_us_ = kp_result.timing_us;
    }

    // 5. Log
    LOG(LOG_INFO) << "Stage 1: ("
                  << kp_result.params[0] << "," << kp_result.params[1] << ","
                  << kp_result.params[2] << "," << kp_result.params[3] << ","
                  << kp_result.params[4] << "," << kp_result.params[5] << ","
                  << kp_result.params[6] << "," << kp_result.params[7] << ") = "
                  << kp_result.timing_us << " us ("
                  << kp_result.configs_tested << " configs)";

    // 6. Record stage result
    result_.stages[1].ran = true;
    result_.stages[1].time_sec =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
    std::copy(kp_result.params.begin(), kp_result.params.end(), result_.best_params);
}

void AutotuneController::runStage2_RuntimeEstimation() {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Build probe config. Always carry forward Stage 1 kernel params so
    // probes reflect the optimized launch configuration regardless of LP status.
    MPQSConfig probe_config = pipeline_config_;

    last_estimate_ = estimateRuntime(
        probe_config,
        pipeline_config_.fb_bound,
        pipeline_config_.sieve_bound,
        pipeline_config_.lp1_bound,
        atcfg_.truncated_run_target_pct,
        0.05,  // eta_convergence_threshold
        static_cast<uint32_t>(atcfg_.truncated_run_min_samples));

    // Store throughput for convergence check
    result_.final_throughput_est = last_estimate_.relations_per_sec;

    // Capture initial throughput on first iteration (before Stage 3 modifies anything)
    if (result_.iterations_run == 0 && result_.initial_throughput_est == 0.0) {
        result_.initial_throughput_est = last_estimate_.relations_per_sec;
    }

    // Log
    LOG(LOG_INFO) << "Stage 2: ETA=" << std::fixed << std::setprecision(1)
                  << last_estimate_.sieve_total_sec << "s"
                  << " (sieve=" << last_estimate_.sieve_total_sec
                  << " matrix=" << last_estimate_.matrix_est_sec
                  << " linalg=" << last_estimate_.linalg_est_sec
                  << " total=" << last_estimate_.total_est_sec << "s)"
                  << " throughput=" << last_estimate_.relations_per_sec << " rels/s"
                  << " confidence=" << std::setprecision(2) << last_estimate_.confidence;

    // Record
    result_.stages[2].ran = true;
    auto t1 = std::chrono::high_resolution_clock::now();
    result_.stages[2].time_sec =
        std::chrono::duration<double>(t1 - t0).count();
}

void AutotuneController::runStage3_SieveParams() {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Warm start from current pipeline config
    auto warm_start = std::make_tuple(
        pipeline_config_.fb_bound,
        pipeline_config_.sieve_bound,
        pipeline_config_.lp1_bound);

    // Configure the optimizer
    SieveParameterOptimizer::Config opt_cfg;
    opt_cfg.max_rounds = 3;                                    // legacy, unused by joint optimizer
    opt_cfg.max_probes_per_axis = 6;                           // legacy, unused by joint optimizer
    opt_cfg.max_total_probes = 40;                             // raised for joint optimizer (was 12)
    opt_cfg.convergence_epsilon = atcfg_.convergence_threshold;
    opt_cfg.truncation_frac = atcfg_.truncated_run_target_pct;
    opt_cfg.eta_convergence_threshold = 0.05;
    opt_cfg.min_eta_samples = 3;

    // Joint (F,L) optimizer fields
    opt_cfg.F_heuristic = heuristic_fb_bound_;
    opt_cfg.warm_start_confidence = projection_confidence_;

    // Compute remaining wall-clock budget; skip if insufficient
    double elapsed_total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time_).count();
    double remaining = atcfg_.timeout_sec - elapsed_total;
    if (remaining < 30.0) {
        LOG(LOG_INFO) << "Stage 3: Skipped (< 30s remaining)";
        result_.stages[3].ran = true;
        result_.stages[3].notes = "skipped (timeout budget)";
        return;
    }
    opt_cfg.wall_clock_timeout_sec = remaining;

    // Construct optimizer and run coordinate descent.
    // Always carry forward Stage 1 kernel params so probes reflect the
    // optimized launch configuration regardless of LP status.
    MPQSConfig probe_config = pipeline_config_;

    SieveParameterOptimizer optimizer(opt_cfg);
    auto sopt = optimizer.optimize(probe_config, warm_start, search_bounds_);

    // Always mark Stage 3 as converged after the joint (F,L) optimizer runs.
    // The joint optimizer's internal Phase C gradient descent with convergence
    // detection already exhausts the search space.  Re-running Stage 3 in a
    // subsequent outer iteration would create a fresh SieveParameterOptimizer
    // (resetting its probe cache), doubling the probe budget (~80 → ~160) with
    // no benefit.  The original `sopt.converged` flag only tracks Phase C's
    // internal convergence — even when false (e.g. probe-limit exit), repeating
    // Stage 3 from scratch is wasteful.
    stage3_converged_ = true;

    // Apply results to pipeline config (handles factor base regeneration if F changed)
    applyToConfig(sopt.optimal_fb_bound, sopt.optimal_sieve_bound, sopt.optimal_lp1_bound);

    // Update result fields
    result_.best_F = sopt.optimal_fb_bound;
    result_.best_M = sopt.optimal_sieve_bound;
    result_.best_lp1_bound = sopt.optimal_lp1_bound;
    result_.improved = (sopt.estimate.total_est_sec < last_estimate_.total_est_sec);

    // Compute improvement BEFORE overwriting last_estimate_ (otherwise ratio = 1.0)
    result_.stages[3].improvement_pct =
        (last_estimate_.total_est_sec > 0)
        ? 100.0 * (1.0 - sopt.estimate.total_est_sec / last_estimate_.total_est_sec)
        : 0.0;

    // Update throughput from the optimizer's best estimate
    if (sopt.estimate.relations_per_sec > 0)
        result_.final_throughput_est = sopt.estimate.relations_per_sec;

    // Preserve Stage 2 buffer recs before Stage 3 overwrites last_estimate_.
    // Stage 3's winning probe may be too short for LP projection.
    uint64_t s2_witness = last_estimate_.recommended_witness_capacity;
    uint64_t s2_partial = last_estimate_.recommended_partial_buffer;
    uint64_t s2_accum   = last_estimate_.recommended_accum_buffer;

    last_estimate_ = sopt.estimate;

    // Merge: take max of Stage 2 and Stage 3 buffer recs
    last_estimate_.recommended_witness_capacity =
        std::max(last_estimate_.recommended_witness_capacity, s2_witness);
    last_estimate_.recommended_partial_buffer =
        std::max(last_estimate_.recommended_partial_buffer, s2_partial);
    last_estimate_.recommended_accum_buffer =
        std::max(last_estimate_.recommended_accum_buffer, s2_accum);

    // Narrow search bounds for next iteration (+/-20% around new optimum)
    search_bounds_.fb_lo = static_cast<uint32_t>(sopt.optimal_fb_bound * 0.8);
    search_bounds_.fb_hi = static_cast<uint32_t>(sopt.optimal_fb_bound * 1.2);

    // Log results
    LOG(LOG_INFO) << "Stage 3: F=" << sopt.optimal_fb_bound
                  << " M=" << sopt.optimal_sieve_bound
                  << " L=" << sopt.optimal_lp1_bound
                  << " T_est=" << std::fixed << std::setprecision(1)
                  << sopt.estimate.total_est_sec << "s"
                  << " (" << sopt.total_probes << " probes: "
                  << "A=" << sopt.phase_a_probes
                  << " B=" << sopt.phase_b_probes
                  << " C=" << sopt.phase_c_probes
                  << (sopt.phase_c_skipped ? " [skipped]" : "")
                  << ", " << (sopt.converged ? "converged" : "not converged") << ")";

    // Record stage timing
    result_.stages[3].ran = true;
    result_.stages[3].time_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// History stubs
// ---------------------------------------------------------------------------

bool AutotuneController::loadHistory() {
    if (atcfg_.history_file.empty()) return false;

    if (!history_.load(atcfg_.history_file)) return false;

    LOG(LOG_DEBUG_1) << "Loaded " << history_.size()
                     << " history entries from " << atcfg_.history_file;

    // Check for exact match with high confidence
    const HistoryEntry* exact = history_.findExact(N_hash_);
    if (exact && exact->confidence > 0.95) {
        // Apply cached parameters
        pipeline_config_.fb_bound    = exact->optimal_params.fb_bound;
        pipeline_config_.sieve_bound = exact->optimal_params.sieve_bound;
        pipeline_config_.lp1_bound   = exact->optimal_params.lp1_bound;

        // Apply kernel params if on same GPU
        if (exact->environment.gpu_name == gpu_name_) {
            for (int i = 0; i < 8; ++i)
                pipeline_config_.params[i] = exact->optimal_params.kernel_params[i];
            pipeline_config_.useParams = true;
        }

        regenerateFactorBase();

        result_.termination = AutotuneResult::TermReason::HISTORY_HIT;
        result_.best_F = exact->optimal_params.fb_bound;
        result_.best_M = exact->optimal_params.sieve_bound;
        result_.best_lp1_bound = exact->optimal_params.lp1_bound;
        std::copy(std::begin(exact->optimal_params.kernel_params),
                  std::end(exact->optimal_params.kernel_params),
                  result_.best_params);
        result_.improved = true;

        LOG(LOG_INFO) << "History hit: exact match for N ("
                      << exact->digit_count << " digits), confidence="
                      << exact->confidence << ". Skipping optimization.";
        return true;   // Signals run() to short-circuit
    }

    return false;  // History loaded but no high-confidence exact match
}

bool AutotuneController::saveHistory() {
    if (atcfg_.history_file.empty()) return false;

    HistoryEntry entry;
    entry.N_decimal       = f_data_.N.to_string();
    entry.N_hash_sha256   = N_hash_;
    entry.digit_count     = N_digits_;
    entry.bit_length      = N_bits_;

    entry.optimal_params.fb_bound    = pipeline_config_.fb_bound;
    entry.optimal_params.sieve_bound = pipeline_config_.sieve_bound;
    entry.optimal_params.lp1_bound   = pipeline_config_.lp1_bound;
    std::copy(std::begin(pipeline_config_.params),
              std::end(pipeline_config_.params),
              entry.optimal_params.kernel_params);

    // Buffer recommendations from autotune probes
    entry.optimal_params.recommended_witness_capacity = last_estimate_.recommended_witness_capacity;
    entry.optimal_params.recommended_partial_buffer   = last_estimate_.recommended_partial_buffer;
    entry.optimal_params.recommended_accum_buffer     = last_estimate_.recommended_accum_buffer;

    // Measured performance from the last estimate (Stage 2/3 probe)
    entry.measured_performance.sieve_time_sec    = last_estimate_.sieve_total_sec;
    entry.measured_performance.total_time_sec    = last_estimate_.total_est_sec;
    entry.measured_performance.relations_per_sec = last_estimate_.relations_per_sec;
    entry.measured_performance.total_relations   = 0;  // Not known yet (haven't run full sieve)
    entry.measured_performance.lp_witnesses      = 0;
    entry.measured_performance.lp_combined_relations = 0;

    entry.environment.gpu_name                = gpu_name_;
    entry.environment.gpu_compute_capability  = gpu_cc_;
    entry.environment.cuda_version            = cuda_version_;

    entry.timestamp            = iso8601_now();
    entry.autotune_stages_run  = stagesRun();
    entry.confidence           = finalConfidence();

    history_.upsert(entry);
    if (history_.save(atcfg_.history_file)) {
        LOG(LOG_INFO) << "Saved history ("
                      << history_.size() << " entries) to "
                      << atcfg_.history_file;
        return true;
    }
    LOG(LOG_ERROR_MAJOR) << "Failed to save history to "
                   << atcfg_.history_file;
    return false;
}

// ---------------------------------------------------------------------------
// Convergence and timing
// ---------------------------------------------------------------------------

bool AutotuneController::hasTimedOut() const {
    return elapsed() >= atcfg_.timeout_sec;
}

double AutotuneController::elapsed() const {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time_).count();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const AutotuneResult& AutotuneController::getResult() const {
    return result_;
}

std::vector<uint32_t> AutotuneController::stagesRun() const {
    std::vector<uint32_t> v;
    for (uint32_t i = 0; i < 4; ++i)
        if (result_.stages[i].ran)
            v.push_back(i);
    return v;
}

double AutotuneController::finalConfidence() const {
    // Combine projection confidence with optimization result confidence
    double conf = last_estimate_.confidence;
    if (conf == 0.0 && projection_confidence_ > 0.0)
        conf = projection_confidence_;
    // If no stages ran (stub state), return 0
    return conf;
}

// ---------------------------------------------------------------------------
// Buffer recommendation application
// ---------------------------------------------------------------------------

void AutotuneController::applyBufferRecommendations() {
    // Only apply if autotune actually produced estimates
    if (last_estimate_.total_est_sec <= 0.0) return;

    // History-based inflation: check if prior entry had overflows or high fill
    const HistoryEntry* prior = history_.findExact(N_hash_);
    if (prior) {
        uint64_t prior_overflow = prior->measured_performance.overflow_events;
        double prior_fill = prior->measured_performance.witness_fill_pct;

        if (prior_overflow > 0 || prior_fill > 85.0) {
            uint64_t inflated = static_cast<uint64_t>(
                prior->measured_performance.witness_capacity * 1.5);
            if (inflated > last_estimate_.recommended_witness_capacity)
                last_estimate_.recommended_witness_capacity = inflated;
            LOG(LOG_INFO) << "Inflated witness recommendation to "
                          << last_estimate_.recommended_witness_capacity
                          << " (prior overflow=" << prior_overflow
                          << ", fill=" << prior_fill << "%)";
        }
    }

    // Benign history fallback: if no probe-based recommendation, use benign data
    if (last_estimate_.recommended_witness_capacity == 0
        || last_estimate_.recommended_partial_buffer == 0) {
        const auto* benign_entry = benign_history_.findByDigits(N_digits_);
        if (benign_entry) {
            if (last_estimate_.recommended_witness_capacity == 0
                && benign_entry->recommended_witness_capacity > 0) {
                last_estimate_.recommended_witness_capacity =
                    benign_entry->recommended_witness_capacity;
                LOG(LOG_DEBUG_1) << "Benign fallback: witness capacity = "
                                 << benign_entry->recommended_witness_capacity;
            }
            if (last_estimate_.recommended_partial_buffer == 0
                && benign_entry->recommended_partial_buffer > 0) {
                last_estimate_.recommended_partial_buffer =
                    benign_entry->recommended_partial_buffer;
                LOG(LOG_DEBUG_1) << "Benign fallback: partial buffer = "
                                 << benign_entry->recommended_partial_buffer;
            }
        }
    }

    // Apply to pipeline config (only if NOT pinned by CLI override)
    if (!pipeline_config_.isPinned("lp1_max_witness_capacity")
        && last_estimate_.recommended_witness_capacity > 0) {
        pipeline_config_.lp1_max_witness_capacity =
            last_estimate_.recommended_witness_capacity;
        LOG(LOG_INFO) << "Set witness capacity = "
                      << pipeline_config_.lp1_max_witness_capacity;
    }
    if (!pipeline_config_.isPinned("partial_buffer_size")
        && last_estimate_.recommended_partial_buffer > 0) {
        pipeline_config_.partial_buffer_size =
            last_estimate_.recommended_partial_buffer;
        LOG(LOG_INFO) << "Set partial buffer = "
                      << pipeline_config_.partial_buffer_size;
    }
    // Batch mode needs larger partial buffer (drains every N batches, not every purge)
    if (pipeline_config_.sieve_batch_size > 0
        && !pipeline_config_.isPinned("partial_buffer_size")
        && pipeline_config_.partial_buffer_size > 0) {
        // Ensure partial buffer is at least 8× accum for batch mode's higher inter-drain fill
        uint64_t accum = pipeline_config_.accum_buffer_size > 0
            ? pipeline_config_.accum_buffer_size : uint64_t{524288};
        uint64_t batch_floor = 8ULL * accum;
        if (pipeline_config_.partial_buffer_size < batch_floor) {
            pipeline_config_.partial_buffer_size = batch_floor;
            LOG(LOG_DEBUG_1) << "Batch-mode partial buffer floor: "
                             << pipeline_config_.partial_buffer_size;
        }
    }
    if (!pipeline_config_.isPinned("accum_buffer_size")
        && last_estimate_.recommended_accum_buffer > 0) {
        pipeline_config_.accum_buffer_size =
            last_estimate_.recommended_accum_buffer;
    }
    if (!pipeline_config_.isPinned("lp1_max_combined_output")
        && last_estimate_.recommended_lp_output > 0) {
        pipeline_config_.lp1_max_combined_output =
            last_estimate_.recommended_lp_output;
        LOG(LOG_INFO) << "Set LP output buffer = "
                      << pipeline_config_.lp1_max_combined_output;
    }
    if (!pipeline_config_.isPinned("lp1_hash_bits")
        && last_estimate_.recommended_hash_bits > 0) {
        pipeline_config_.lp1_hash_bits =
            last_estimate_.recommended_hash_bits;
        LOG(LOG_INFO) << "Set hash bits = "
                      << pipeline_config_.lp1_hash_bits;
    }
}

// ---------------------------------------------------------------------------
// Buffer recommendation summary
// ---------------------------------------------------------------------------

void AutotuneController::printBufferRecommendations() {
    // Memory cost constants are shared with AutoApplyController::computeMemoryEstimate
    // — see memory_estimator.h for definitions and audit references.
    using memory_costs::DENSE_CANDIDATE_BYTES;
    using memory_costs::SOA_PER_REL_BYTES;
    using memory_costs::WITNESS_PAYLOAD_BYTES;
    using memory_costs::WITNESS_DIR_BYTES;
    using memory_costs::LP_PIPELINE_BYTES;

    // Resolve buffer sizes to their effective values (0 = use defaults)
    uint64_t accum   = pipeline_config_.accum_buffer_size > 0
                     ? pipeline_config_.accum_buffer_size : 524288ULL;
    uint64_t partial = pipeline_config_.partial_buffer_size > 0
                     ? pipeline_config_.partial_buffer_size : accum;
    uint64_t persist = pipeline_config_.persistent_buffer_size > 0
                     ? pipeline_config_.persistent_buffer_size
                     : pipeline_config_.target_relations + accum;
    uint64_t witness = pipeline_config_.lp1_max_witness_capacity > 0
                     ? pipeline_config_.lp1_max_witness_capacity : (1ULL << 20);
    uint64_t lp_out  = pipeline_config_.lp1_max_combined_output > 0
                     ? pipeline_config_.lp1_max_combined_output : 32768ULL;
    uint32_t hbits   = pipeline_config_.lp1_hash_bits;
    if (hbits == 0 && witness > 0) {
        int log2_w = 63 - __builtin_clzll(witness > 0 ? witness : 1);
        hbits = (log2_w > 4) ? (log2_w - 4) : 4;
    }

    bool lp_active = pipeline_config_.lp1_bound > 0;

    // Compute total estimated GPU memory
    size_t mem = 0;
    mem += 2 * accum * DENSE_CANDIDATE_BYTES;  // Accum double-buffer
    mem += accum * SOA_PER_REL_BYTES;           // Full batch SoA (transient)
    mem += persist * SOA_PER_REL_BYTES;         // Persistent SoA
    if (lp_active) {
        mem += partial * SOA_PER_REL_BYTES;     // Partial SoA
        mem += (1ULL << hbits) * WITNESS_DIR_BYTES;  // Hash directory
        mem += witness * WITNESS_PAYLOAD_BYTES;      // Payload slabs
        mem += witness * SOA_PER_REL_BYTES;          // Witness SoA
        mem += lp_out * SOA_PER_REL_BYTES;           // Output SoA
        mem += partial * LP_PIPELINE_BYTES;          // LP pipeline transient
    }
    size_t mem_mb = mem / (1024 * 1024);

    // Compute "default" memory for savings comparison
    size_t def_accum = 524288, def_partial = def_accum;
    size_t def_persist = pipeline_config_.target_relations + def_accum;
    size_t def_witness = 1ULL << 20, def_lp_out = 32768;
    uint32_t def_hbits = 16;
    size_t def_mem = 0;
    def_mem += 2 * def_accum * DENSE_CANDIDATE_BYTES;
    def_mem += def_accum * SOA_PER_REL_BYTES;
    def_mem += def_persist * SOA_PER_REL_BYTES;
    if (lp_active) {
        def_mem += def_partial * SOA_PER_REL_BYTES;
        def_mem += (1ULL << def_hbits) * WITNESS_DIR_BYTES;
        def_mem += def_witness * WITNESS_PAYLOAD_BYTES;
        def_mem += def_witness * SOA_PER_REL_BYTES;
        def_mem += def_lp_out * SOA_PER_REL_BYTES;
        def_mem += def_partial * LP_PIPELINE_BYTES;
    }
    size_t def_mem_mb = def_mem / (1024 * 1024);

    int savings_pct = (def_mem_mb > 0)
        ? static_cast<int>(100.0 * (1.0 - static_cast<double>(mem_mb) / def_mem_mb))
        : 0;

    auto provenance = [&](const std::string& field) -> std::string {
        if (pipeline_config_.isPinned(field)) return "user";
        // If the field was set by autotune (differs from default), it's from probe/history
        return "auto";
    };

    LOG(LOG_DEBUG_1) << "Buffer recommendations:";
    LOG(LOG_DEBUG_1) << "  accum_buffer_size:        " << accum
                     << "  [" << provenance("accum_buffer_size") << "]  "
                     << "2 × " << accum / 1024 << "K × " << DENSE_CANDIDATE_BYTES
                     << "B = " << (2 * accum * DENSE_CANDIDATE_BYTES) / (1024*1024) << " MB";
    if (lp_active) {
        LOG(LOG_DEBUG_1) << "  partial_buffer_size:      " << partial
                         << "  [" << provenance("partial_buffer_size") << "]  "
                         << partial / 1024 << "K × " << SOA_PER_REL_BYTES
                         << "B = " << (partial * SOA_PER_REL_BYTES) / (1024*1024) << " MB";
    }
    LOG(LOG_DEBUG_1) << "  persistent_buffer_size:   " << persist
                     << "  [derived: target + accum]  "
                     << persist / 1024 << "K × " << SOA_PER_REL_BYTES
                     << "B = " << (persist * SOA_PER_REL_BYTES) / (1024*1024) << " MB";
    if (lp_active) {
        LOG(LOG_DEBUG_1) << "  lp1_max_witness_capacity: " << witness
                         << "  [" << provenance("lp1_max_witness_capacity") << "]  "
                         << witness / 1024 << "K × "
                         << (WITNESS_PAYLOAD_BYTES + SOA_PER_REL_BYTES)
                         << "B = " << (witness * (WITNESS_PAYLOAD_BYTES + SOA_PER_REL_BYTES)) / (1024*1024) << " MB";
        LOG(LOG_DEBUG_1) << "  lp1_max_combined_output:  " << lp_out
                         << "  [" << provenance("lp1_max_combined_output") << "]  "
                         << lp_out / 1024 << "K × " << SOA_PER_REL_BYTES
                         << "B = " << (lp_out * SOA_PER_REL_BYTES) / (1024*1024) << " MB";
        LOG(LOG_DEBUG_1) << "  lp1_hash_bits:            " << hbits
                         << "  [" << provenance("lp1_hash_bits") << "]  "
                         << "2^" << hbits << " × " << WITNESS_DIR_BYTES
                         << "B = " << ((1ULL << hbits) * WITNESS_DIR_BYTES) / (1024*1024) << " MB";
    }
    LOG(LOG_DEBUG_1) << "  Estimated total GPU memory: ~" << mem_mb << " MB"
                     << " (vs ~" << def_mem_mb << " MB at defaults: "
                     << savings_pct << "% savings)";
}

} // namespace mpqs::autotune
