// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "autotune_history.h"
#include <string>
#include <vector>
#include <cstdint>

namespace mpqs::autotune {

/// Output of Stage 0 parameter projection: initial parameter guess with
/// confidence score and search radii for M2/M3 optimization stages.
struct ProjectedParams {
    uint32_t fb_bound    = 0;
    uint32_t sieve_bound = 0;
    uint64_t lp1_bound   = 0;
    uint32_t kernel_params[8] = {};

    double confidence = 0.0;   // 0.0–1.0
    enum class Source {
        EXACT_MATCH,           // History contains this exact N
        INTERPOLATION,         // Bracketed by history entries
        EXTRAPOLATION,         // Beyond history range
        THEORY_FALLBACK        // No or insufficient history
    } source = Source::THEORY_FALLBACK;

    // Suggested search radii for M2/M3 (narrower if confidence is high).
    double fb_bound_search_radius_pct    = 0.50;
    double sieve_bound_search_radius_pct = 0.50;
    double lp1_bound_search_radius_pct   = 0.50;
};

/// Stage 0 projector: given N's bit-length and a history database, produce an
/// initial parameter guess via a 4-tier cascade (exact → interpolate →
/// extrapolate → theory fallback). All computation is pure CPU, O(n) in
/// history size, completes in microseconds.
class ParameterProjector {
public:
    /// Construct from history store. Fits OLS models from history data.
    explicit ParameterProjector(const HistoryStore& store);

    /// Project optimal parameters for the given N.
    /// @param bits      Bit-length of N.
    /// @param digits    Digit count of N.
    /// @param N_hash    SHA-256 hash of N decimal string (for exact lookup).
    /// @param gpu_name  Current GPU name (for kernel param filtering).
    ProjectedParams project(
        uint32_t bits, uint32_t digits,
        const std::string& N_hash,
        const std::string& gpu_name) const;

private:
    const HistoryStore& store_;

    /// Fitted OLS model: ln(param) = ln_c + alpha * u
    /// where u = sqrt(bits * ln2 * ln(bits * ln2)).
    struct FittedModel {
        double ln_c  = 0.0;   // intercept
        double alpha = 0.0;   // slope
        bool valid   = false;
    };
    FittedModel fb_model_;     // For fb_bound
    FittedModel lp_model_;     // For lp1_bound (fit on LP-active entries only)

    /// Fit OLS models from history. Called once in constructor.
    void fitModels();

    /// Tier 2: Interpolation between bracketing entries in L-space.
    ProjectedParams interpolate(
        uint32_t bits,
        const std::vector<const HistoryEntry*>& neighbors,
        const std::string& gpu_name) const;

    /// Tier 3: Extrapolation using fitted OLS model.
    ProjectedParams extrapolate(
        uint32_t bits,
        const std::vector<const HistoryEntry*>& neighbors,
        const std::string& gpu_name) const;

    /// Tier 4: Theory fallback mirroring determineParams() heuristic (all 8 ranges).
    ProjectedParams theoryFallback(uint32_t bits) const;

    /// Project kernel launch parameters from neighbors.
    /// Constant params (0,1,2,3,5,7): modal value from GPU-matching neighbors.
    /// Variable params (4,6): interpolated linearly in bit-length.
    void projectKernelParams(
        ProjectedParams& out,
        uint32_t bits,
        const std::vector<const HistoryEntry*>& neighbors,
        const std::string& gpu_name) const;
};

} // namespace mpqs::autotune
