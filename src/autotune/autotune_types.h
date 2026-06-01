// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cstdint>
#include <cmath>
#include <vector>

namespace mpqs::autotune {

/// Per-stage wall-clock timing for the autotune pipeline.
/// Uses seconds to match M1's AutotuneResult::StageSummary::time_sec.
struct AutotuneStageTiming {
    double stage0_sec = 0.0;   // Parameter projection (M4)
    double stage1_sec = 0.0;   // Kernel launch param optimization (M2)
    double stage2_sec = 0.0;   // Runtime estimation (M3 subroutine)
    double stage3_sec = 0.0;   // Sieve parameter optimization (M3)
    double total_sec  = 0.0;   // Sum of all stages
};

/// L(N) complexity function for the quadratic sieve.
/// L(N) = exp(sqrt(ln(N) * ln(ln(N)))), with c=1 normalization.
/// @param bits  Bit-length of N (i.e., floor(log2(N)) + 1).
/// @return L(N) as a double.
inline double L_function(uint32_t bits) {
    double ln_N = bits * std::log(2.0);           // ln(N) ≈ bits * ln(2)
    double ln_ln_N = std::log(ln_N);              // ln(ln(N))
    return std::exp(std::sqrt(ln_N * ln_ln_N));   // exp(sqrt(ln(N)*ln(ln(N))))
}

/// L(N)^alpha for parameter scaling.
/// Used to model F ~ L(N)^alpha where alpha ≈ 0.707 theoretically.
/// @param bits  Bit-length of N.
/// @param alpha Scaling exponent.
inline double L_alpha(uint32_t bits, double alpha) {
    double ln_N = bits * std::log(2.0);
    double ln_ln_N = std::log(ln_N);
    return std::exp(alpha * std::sqrt(ln_N * ln_ln_N));
}

/// Compute search radius percentage from confidence.
/// Maps confidence ∈ [0,1] to search radius ∈ [5%, 50%].
/// High confidence → narrow search; low confidence → wide search.
/// @param confidence  Confidence score from ParameterProjector.
/// @return Search radius as a fraction (e.g., 0.05 = 5%).
inline double searchRadius(double confidence) {
    return 0.05 + 0.45 * (1.0 - confidence);
}

// ---------------------------------------------------------------------------
// M3 additions: runtime estimation types, optimization result, search bounds
// ---------------------------------------------------------------------------

/// Estimated runtime for a single parameter set, produced by estimateRuntime().
struct RuntimeEstimate {
    // Sieve (from truncated run)
    double sieve_eta_sec = 0.0;         ///< Projected remaining sieve time (quadratic fit)
    double sieve_total_sec = 0.0;       ///< Full sieve time estimate (extrapolated)
    double relations_per_sec = 0.0;     ///< Observed throughput

    // Matrix + LinAlg (heuristic models)
    double matrix_est_sec = 0.0;        ///< Estimated matrix construction time
    double linalg_est_sec = 0.0;        ///< Estimated Block Wiedemann time

    // Combined
    double total_est_sec = 0.0;         ///< sieve_total_sec + matrix_est_sec + linalg_est_sec

    // Diagnostics
    uint32_t fb_size = 0;               ///< Factor base size (determines matrix dimensions)
    double truncation_progress = 0.0;   ///< Fraction of target completed in probe
    uint32_t eta_samples = 0;           ///< Number of samples in quadratic fit
    bool eta_reliable = false;          ///< >= 6 samples and positive curvature

    // LP diagnostics (zero if LP disabled)
    double projected_witness_fill_pct = 0.0;
    uint64_t overflow_events = 0;       ///< Total slab + witness + output overflows

    // The parameters that were evaluated
    uint32_t fb_bound = 0;
    uint32_t sieve_bound = 0;
    uint64_t lp1_bound = 0;

    // Confidence: 0.0 (unreliable) to 1.0 (high confidence)
    double confidence = 0.0;

    // Buffer recommendations (computed by estimateRuntime, applied by AutotuneController)
    uint64_t recommended_witness_capacity = 0;  // 0 = no recommendation
    uint64_t recommended_partial_buffer   = 0;  // 0 = no recommendation
    uint64_t recommended_accum_buffer     = 0;  // 0 = no recommendation
    uint64_t recommended_lp_output        = 0;  // 0 = no recommendation
    uint32_t recommended_hash_bits        = 0;  // 0 = no recommendation
};

/// Result of the sieve parameter optimization (coordinate descent).
struct SieveOptimizationResult {
    uint32_t optimal_fb_bound = 0;
    uint32_t optimal_sieve_bound = 0;
    uint64_t optimal_lp1_bound = 0;
    RuntimeEstimate estimate;

    // Optimization metadata
    uint32_t rounds_completed = 0;
    uint32_t total_probes = 0;
    double wall_clock_sec = 0.0;
    bool converged = false;

    // Full probe history (for logging / debugging)
    std::vector<RuntimeEstimate> probe_history;

    // Phase-level tracking (joint optimizer)
    uint32_t phase_a_probes = 0;           ///< Probes used by Phase A (L-sweep)
    uint32_t phase_b_probes = 0;           ///< Probes used by Phase B (3x3 grid)
    uint32_t phase_c_probes = 0;           ///< Probes used by Phase C (gradient descent)
    uint32_t phase_c_iterations = 0;       ///< Gradient descent iterations completed
    bool phase_c_skipped = false;          ///< True if warm-start adaptation skipped Phase C
};

/// Search bounds for coordinate descent over (F, M, L).
struct SieveSearchBounds {
    uint32_t fb_lo = 0;               ///< Factor base bound lower
    uint32_t fb_hi = 0;               ///< Factor base bound upper
    uint32_t M_min_log2 = 15;         ///< log2(sieve_bound) range lower
    uint32_t M_max_log2 = 22;         ///< log2(sieve_bound) range upper
    uint64_t lp_lo = 0;               ///< LP bound lower (0 = LP-disabled is a candidate)
    uint64_t lp_hi = 0;               ///< LP bound upper
};

} // namespace mpqs::autotune
