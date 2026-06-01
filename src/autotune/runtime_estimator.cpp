// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "runtime_estimator.h"
#include "cost_models.h"
#include "kernel_launch_validator.h"  // PreflightResult, preflightKernelLaunch
#include "orchestrator.h"   // Full include needed — constructs MPQSOrchestrator

#include <cuda_runtime.h>
#include <filesystem>
#include <random>
#include <iomanip>
#include <sstream>

namespace mpqs::autotune {

namespace {

/// Generate a unique temp directory path for probe isolation.
std::string generateProbeTempDir() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream oss;
    oss << "/tmp/mpqs_autotune_" << std::hex << dist(gen);
    return oss.str();
}

/// Remove probe temp directory (best-effort, non-throwing).
void cleanupProbeTempDir(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

/// Compute confidence score from probe diagnostics.
/// Spec: *0.6 if <6 samples, =0.0 if <3, *0.8 if >100 overflows,
///       *0.5 if <5% progress and not converged early.
double computeConfidence(uint32_t eta_samples, uint64_t overflow_events,
                         double truncation_progress, bool converged_early) {
    double confidence = 1.0;
    if (eta_samples < 6)        confidence *= 0.6;
    if (eta_samples < 3)        return 0.0;   // unusable
    if (overflow_events > 100)  confidence *= 0.8;
    if (truncation_progress < 0.05 && !converged_early)
                                confidence *= 0.5;
    return confidence;
}

} // anonymous namespace

RuntimeEstimate estimateRuntime(
    const mpqs::MPQSConfig& base_config,
    uint32_t fb_bound, uint32_t sieve_bound, uint64_t lp1_bound,
    double truncation_frac, double eta_convergence_threshold,
    uint32_t min_eta_samples)
{
    // 1. Build probe config
    MPQSConfig cfg = base_config;
    cfg.fb_bound = fb_bound;
    cfg.sieve_bound = sieve_bound;
    cfg.lp1_bound = lp1_bound;
    cfg.auto_tune_parameters = true;   // let determineParams fill zeros
    cfg.sieve_batch_size = 0;          // force legacy loop
    cfg.work_dir = generateProbeTempDir();
    cfg.disk_io = false;
    cfg.silent = true;  // Suppress probe orchestrator init/shutdown log noise

    RuntimeEstimate est{};
    est.fb_bound = fb_bound;
    est.sieve_bound = sieve_bound;
    est.lp1_bound = lp1_bound;

    try {
        // 2. Create ephemeral orchestrator (RAII scope)
        {
            // Clear sticky CUDA errors before probe
            cudaDeviceSynchronize();
            cudaGetLastError();

            MPQSOrchestrator probe(cfg);
            probe.TuningStage();

            // Preflight check: skip probe if kernel params are infeasible
            if (cfg.useParams) {
                const auto& fd = probe.getFactoringData();
                auto pf = preflightKernelLaunch(cfg,
                    static_cast<uint32_t>(fd.a_factors.size()), fd.M);
                if (!pf.feasible) {
                    LOG(LOG_WARNING) << "[RuntimeEstimator] Skipping probe: preflight failed — "
                                  << pf.reason;
                    RuntimeEstimate sentinel{};
                    sentinel.confidence = 0.0;
                    sentinel.total_est_sec = 1e9;
                    sentinel.relations_per_sec = 0.0;
                    // probe destructor fires on scope exit; clean up temp dir manually
                    cleanupProbeTempDir(cfg.work_dir);
                    return sentinel;
                }
            }

            // 3. Truncated sieve (may exit early via ETA convergence)
            auto result = probe.TruncatedSieveRun(
                truncation_frac, eta_convergence_threshold, min_eta_samples);
            uint32_t fb_size = probe.getFactorBaseSize();

            est.fb_size = fb_size;

            // Sieve time estimate: quadratic ETA if reliable, linear fallback otherwise
            if (result.eta_reliable) {
                // current_eta_sec = remaining time from quadratic fit.
                // Total = elapsed + remaining.
                est.sieve_total_sec = result.progress_tracker.current_eta_sec
                                    + result.elapsed_sec;
            } else {
                // Linear extrapolation from observed throughput
                double rels_per_sec = (result.elapsed_sec > 0.0)
                    ? static_cast<double>(result.relations_found) / result.elapsed_sec
                    : 1.0;
                est.sieve_total_sec = static_cast<double>(result.target_relations)
                                    / rels_per_sec;
            }
            est.sieve_eta_sec = est.sieve_total_sec;
            est.relations_per_sec = (result.elapsed_sec > 0.0)
                ? static_cast<double>(result.relations_found) / result.elapsed_sec
                : 0.0;

            // LP yield correction: birthday-paradox projection model.
            //
            // Short probes see linear witness accumulation but LP combined
            // relations scale quadratically (matches ~ W^2/2B, birthday paradox).
            // We project total witnesses at full scale using observed influx rate,
            // then estimate LP contribution via the birthday-paradox formula.
            if (lp1_bound > 0 && result.lp_projector.ready()) {
                auto proj = result.lp_projector.project(est.sieve_total_sec);

                // Witness arrival rate from probe (linear, measurable in short probes)
                double witness_rate = (result.elapsed_sec > 0.01)
                    ? proj.projected_witnesses / est.sieve_total_sec
                    : 0.0;

                if (witness_rate > 0.0) {
                    // Project total witnesses at full sieve scale
                    double total_witnesses = witness_rate * est.sieve_total_sec;

                    // Birthday denominator B = number of distinct LP values.
                    // Primes in [F, L] ≈ π(L) - π(F), with π(x) ≈ x/ln(x).
                    double pi_L = (lp1_bound > 10) ? lp1_bound / std::log(static_cast<double>(lp1_bound)) : 0.0;
                    double pi_F = (fb_bound > 10)  ? fb_bound  / std::log(static_cast<double>(fb_bound))  : 0.0;
                    double B = std::max(1.0, pi_L - pi_F);

                    // Birthday-paradox expected matches: E[matches] ~ W^2 / (2*B)
                    // Each match yields one combined relation (1-partial + 1-partial → full)
                    double expected_matches = (total_witnesses * total_witnesses)
                                            / (2.0 * B);

                    // Target relations for this config
                    double target = static_cast<double>(result.target_relations);

                    // LP contribution as fraction of target
                    // Clamp to [0, 0.35] — LP rarely contributes >35% of relations
                    double lp_frac = (target > 0.0)
                        ? std::clamp(expected_matches / target, 0.0, 0.35)
                        : 0.0;

                    // Apply discount: LP relations reduce the sieve time needed
                    // to collect the remaining (1 - lp_frac) full relations
                    est.sieve_total_sec *= (1.0 - lp_frac);

                    LOG(LOG_DEBUG_1) << "[RuntimeEstimator] LP birthday model:"
                                     << " witness_rate=" << std::fixed << std::setprecision(1)
                                     << witness_rate << "/s"
                                     << " projected_W=" << static_cast<uint64_t>(total_witnesses)
                                     << " prime_range_B=" << static_cast<uint64_t>(B)
                                     << " expected_matches=" << static_cast<uint64_t>(expected_matches)
                                     << " lp_frac=" << std::setprecision(3) << lp_frac;
                }
            }

            // Matrix + LinAlg heuristic models
            est.matrix_est_sec = estimateMatrixTime(fb_size);
            est.linalg_est_sec = estimateLinalgTime(fb_size);

            // Combined
            est.total_est_sec = est.sieve_total_sec + est.matrix_est_sec
                              + est.linalg_est_sec;

            // Diagnostics
            est.truncation_progress = (result.target_relations > 0)
                ? static_cast<double>(result.relations_found)
                  / static_cast<double>(result.target_relations)
                : 0.0;
            est.eta_samples = static_cast<uint32_t>(
                result.progress_tracker.history.size());
            est.eta_reliable = result.eta_reliable;

            // LP diagnostics
            if (lp1_bound > 0 && result.lp_projector.ready()) {
                auto proj = result.lp_projector.project(est.sieve_total_sec);
                est.projected_witness_fill_pct = proj.projected_fill_pct;

                // Buffer recommendations
                // Witness capacity: projected witnesses * 1.3 safety margin, rounded to power of 2
                uint64_t rec_witnesses = static_cast<uint64_t>(proj.projected_witnesses * 1.3);
                if (rec_witnesses > 0) {
                    uint64_t clz_arg = rec_witnesses > 1 ? rec_witnesses - 1 : 1;
                    int log2_w = 64 - __builtin_clzll(clz_arg);
                    log2_w = std::min(log2_w, 24);  // OOM cap: 2^24 = 16M entries (~256 MB)
                    est.recommended_witness_capacity = 1ULL << log2_w;
                }

                // Partial buffer: scale based on projected witness pressure.
                // More witnesses → more 1-partials → more partial buffer pressure.
                // Scale up from default 4M if projected witnesses exceed default capacity (1M).
                uint64_t default_witness_cap = 1ULL << 20;  // 1M (current default)
                if (proj.projected_witnesses > default_witness_cap) {
                    double scale = static_cast<double>(proj.projected_witnesses) / default_witness_cap;
                    scale = std::clamp(scale, 1.0, 4.0);  // Cap at 4x = 16M partial buffer
                    est.recommended_partial_buffer = static_cast<uint64_t>(4194304 * scale);
                }
            }

            // Accum buffer recommendation: stationary peak from probe, with safety margin.
            // Accum peak is independent of LP and run length — probe peak ≈ full-run peak.
            if (result.buffer_fill.accum_max > 0) {
                uint64_t rec_accum = static_cast<uint64_t>(result.buffer_fill.accum_max * 1.5);
                rec_accum = std::max(rec_accum, uint64_t{65536});  // Minimum 64K to saturate GPU sorting
                // Round up to next power of 2 for alignment
                uint64_t clz_arg_a = rec_accum > 1 ? rec_accum - 1 : 1;
                int log2_a = 64 - __builtin_clzll(clz_arg_a);
                log2_a = std::min(log2_a, 20);  // Cap at 1M (336 MB per buffer × 2)
                est.recommended_accum_buffer = 1ULL << log2_a;
            }

            est.overflow_events = result.lp_fill.total_slab_overflows
                                + result.lp_fill.total_witness_overflows
                                + result.lp_fill.total_output_overflows;

            // LP output buffer: double if output overflows detected during probe
            if (result.lp_fill.total_output_overflows > 0) {
                est.recommended_lp_output = 65536;  // 2× default 32K
            }

            // Hash bits: increase if slab overflows detected (indicates bucket crowding)
            if (result.lp_fill.total_slab_overflows > 0) {
                // Current hash_bits = log2(witness_cap) - 4.
                // Add 2 to reduce collision rate (4× more directory buckets).
                int current_bits = 63 - __builtin_clzll(
                    result.lp_fill.witness_capacity > 0
                        ? result.lp_fill.witness_capacity : 1ULL);
                current_bits = (current_bits > 4) ? (current_bits - 4) : 4;
                est.recommended_hash_bits = static_cast<uint32_t>(current_bits + 2);
            }

            // Confidence scoring
            est.confidence = computeConfidence(
                est.eta_samples, est.overflow_events,
                est.truncation_progress, result.converged_early);
        }
        // Probe destroyed here — GPU buffers freed via RAII
    } catch (...) {
        // Ensure temp dir cleanup on exception path
        cleanupProbeTempDir(cfg.work_dir);
        throw;
    }

    // 5. Cleanup temp directory
    cleanupProbeTempDir(cfg.work_dir);

    return est;
}

} // namespace mpqs::autotune
