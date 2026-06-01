// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "autotune_projection.h"
#include "autotune_types.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace mpqs::autotune {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

/// L-space coordinate: u(bits) = sqrt(bits * ln(2) * ln(bits * ln(2))).
/// This is the natural interpolation variable for subexponential scaling.
inline double u_coord(uint32_t bits) {
    double ln_N = bits * std::log(2.0);
    return std::sqrt(ln_N * std::log(ln_N));
}

/// Round to nearest power of 2.
inline uint32_t nearestPow2(double x) {
    if (x <= 1.0) return 1;
    double log2_x = std::log2(x);
    uint32_t exp = static_cast<uint32_t>(std::round(log2_x));
    if (exp > 30) exp = 30;  // cap at 2^30 to avoid overflow
    return 1u << exp;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ParameterProjector::ParameterProjector(const HistoryStore& store)
    : store_(store)
{
    fitModels();
}

// ---------------------------------------------------------------------------
// fitModels — OLS: ln(param) = a + b*u
// ---------------------------------------------------------------------------

void ParameterProjector::fitModels() {
    const auto& entries = store_.entries();

    // --- fb_bound model ---
    // Require >= 2 entries with fb_bound > 0
    {
        double sum_u = 0.0, sum_y = 0.0, sum_uu = 0.0, sum_uy = 0.0;
        size_t n = 0;
        for (const auto& e : entries) {
            if (e.optimal_params.fb_bound == 0) continue;
            double u = u_coord(e.bit_length);
            double y = std::log(static_cast<double>(e.optimal_params.fb_bound));
            sum_u  += u;
            sum_y  += y;
            sum_uu += u * u;
            sum_uy += u * y;
            ++n;
        }
        if (n >= 2) {
            double denom = static_cast<double>(n) * sum_uu - sum_u * sum_u;
            if (std::abs(denom) > 1e-12) {
                double b = (static_cast<double>(n) * sum_uy - sum_u * sum_y) / denom;
                double a = (sum_y - b * sum_u) / static_cast<double>(n);
                fb_model_ = {a, b, true};
            }
        }
    }

    // --- lp1_bound model (LP-active entries only) ---
    {
        double sum_u = 0.0, sum_y = 0.0, sum_uu = 0.0, sum_uy = 0.0;
        size_t n = 0;
        for (const auto& e : entries) {
            if (e.optimal_params.lp1_bound == 0) continue;
            double u = u_coord(e.bit_length);
            double y = std::log(static_cast<double>(e.optimal_params.lp1_bound));
            sum_u  += u;
            sum_y  += y;
            sum_uu += u * u;
            sum_uy += u * y;
            ++n;
        }
        if (n >= 2) {
            double denom = static_cast<double>(n) * sum_uu - sum_u * sum_u;
            if (std::abs(denom) > 1e-12) {
                double b = (static_cast<double>(n) * sum_uy - sum_u * sum_y) / denom;
                double a = (sum_y - b * sum_u) / static_cast<double>(n);
                lp_model_ = {a, b, true};
            }
        }
    }
}

// ---------------------------------------------------------------------------
// project — 4-tier cascade
// ---------------------------------------------------------------------------

ProjectedParams ParameterProjector::project(
    uint32_t bits, uint32_t digits,
    const std::string& N_hash,
    const std::string& gpu_name) const
{
    (void)digits;  // used for context only; projection keys on bits

    // Tier 1: Exact match
    const HistoryEntry* exact = store_.findExact(N_hash);
    if (exact) {
        ProjectedParams p{};
        p.fb_bound    = exact->optimal_params.fb_bound;
        p.sieve_bound = exact->optimal_params.sieve_bound;
        p.lp1_bound   = exact->optimal_params.lp1_bound;
        std::memcpy(p.kernel_params, exact->optimal_params.kernel_params,
                     sizeof(p.kernel_params));
        p.confidence  = exact->confidence;
        p.source      = ProjectedParams::Source::EXACT_MATCH;
        p.fb_bound_search_radius_pct    = 0.0;
        p.sieve_bound_search_radius_pct = 0.0;
        p.lp1_bound_search_radius_pct   = 0.0;
        return p;
    }

    // Find k-nearest neighbors
    auto neighbors = store_.findKNearest(bits, 10);
    if (neighbors.empty()) {
        return theoryFallback(bits);
    }

    // Check for bracketing entries (one below, one at-or-above target bits)
    bool has_below = false, has_above = false;
    for (const auto* e : neighbors) {
        if (e->bit_length <= bits) has_below = true;
        if (e->bit_length >= bits) has_above = true;
    }

    if (has_below && has_above && neighbors.size() >= 2) {
        // Tier 2: Interpolation
        return interpolate(bits, neighbors, gpu_name);
    }

    if (neighbors.size() >= 2 && fb_model_.valid) {
        // Tier 3: Extrapolation — cap at 40 bits beyond nearest entry
        uint32_t nearest_bits = neighbors.front()->bit_length;
        uint32_t delta = (bits > nearest_bits)
                         ? bits - nearest_bits
                         : nearest_bits - bits;
        if (delta <= 40) {
            return extrapolate(bits, neighbors, gpu_name);
        }
    }

    // Tier 4: Theory fallback
    return theoryFallback(bits);
}

// ---------------------------------------------------------------------------
// theoryFallback — mirror ALL 8 ranges from primeAlgorithms.cu:356-373
// ---------------------------------------------------------------------------

ProjectedParams ParameterProjector::theoryFallback(uint32_t bits) const {
    ProjectedParams p{};

    // Mirror determineParams() heuristic — all 8 bit-length ranges
    if      (bits < 180) p.fb_bound = 60000;
    else if (bits < 220) p.fb_bound = 350000;
    else if (bits < 250) p.fb_bound = 900000;
    else if (bits < 340) p.fb_bound = 3000000;     // RSA-100 (330 bits)
    else if (bits < 370) p.fb_bound = 4000000;     // RSA-110 (364 bits)
    else if (bits < 400) p.fb_bound = 5000000;     // RSA-120 (397 bits)
    else if (bits < 440) p.fb_bound = 7500000;     // RSA-130 (430 bits)
    else                 p.fb_bound = 10000000;     // RSA-140+

    p.sieve_bound = 262144;   // 2^18, current default
    // Seed LP for large inputs where LP historically helps
    p.lp1_bound   = (bits >= 200)
        ? static_cast<uint64_t>(p.fb_bound) * 10
        : 0;

    // Clamp LP bound to F² (safety invariant)
    if (p.lp1_bound > 0) {
        uint64_t f_squared = static_cast<uint64_t>(p.fb_bound) * static_cast<uint64_t>(p.fb_bound);
        if (p.lp1_bound > f_squared) {
            LOG(LOG_DEBUG_1) << "[Projection] Clamping theory LP bound from "
                             << p.lp1_bound << " to F²=" << f_squared;
            p.lp1_bound = f_squared;
        }
    }

    // Kernel params all zero → auto-calculate
    std::memset(p.kernel_params, 0, sizeof(p.kernel_params));

    p.confidence = 0.0;
    p.source     = ProjectedParams::Source::THEORY_FALLBACK;
    p.fb_bound_search_radius_pct    = 0.50;
    p.sieve_bound_search_radius_pct = 0.50;
    p.lp1_bound_search_radius_pct   = 0.50;
    return p;
}

// ---------------------------------------------------------------------------
// interpolate — L-space weighted interpolation
// ---------------------------------------------------------------------------

ProjectedParams ParameterProjector::interpolate(
    uint32_t bits,
    const std::vector<const HistoryEntry*>& neighbors,
    const std::string& gpu_name) const
{
    // Select nearest entry below and nearest entry above the target.
    // Among ties at same bit_length, prefer highest confidence.
    const HistoryEntry* e_lo = nullptr;
    const HistoryEntry* e_hi = nullptr;

    for (const auto* e : neighbors) {
        if (e->bit_length <= bits) {
            if (!e_lo ||
                e->bit_length > e_lo->bit_length ||
                (e->bit_length == e_lo->bit_length && e->confidence > e_lo->confidence))
                e_lo = e;
        }
        if (e->bit_length >= bits) {
            if (!e_hi ||
                e->bit_length < e_hi->bit_length ||
                (e->bit_length == e_hi->bit_length && e->confidence > e_hi->confidence))
                e_hi = e;
        }
    }

    // Should always have both (caller checked), but be defensive
    if (!e_lo || !e_hi) return theoryFallback(bits);

    // Handle exact bit-length match (both point to same or same-bits entry)
    if (e_lo->bit_length == e_hi->bit_length) {
        // Pick the higher-confidence entry
        const HistoryEntry* best = (e_lo->confidence >= e_hi->confidence) ? e_lo : e_hi;
        ProjectedParams p{};
        p.fb_bound    = best->optimal_params.fb_bound;
        p.sieve_bound = best->optimal_params.sieve_bound;
        p.lp1_bound   = best->optimal_params.lp1_bound;
        p.confidence  = best->confidence *
                        std::min(1.0, static_cast<double>(store_.size()) / 5.0);
        p.source = ProjectedParams::Source::INTERPOLATION;
        double r = searchRadius(p.confidence);
        p.fb_bound_search_radius_pct    = r;
        p.sieve_bound_search_radius_pct = r;
        p.lp1_bound_search_radius_pct   = r;
        projectKernelParams(p, bits, neighbors, gpu_name);
        return p;
    }

    // Compute L-space coordinates
    double u_lo     = u_coord(e_lo->bit_length);
    double u_hi     = u_coord(e_hi->bit_length);
    double u_target = u_coord(bits);

    // Interpolation weight t ∈ [0,1]
    double t = (u_hi > u_lo) ? (u_target - u_lo) / (u_hi - u_lo) : 0.5;
    t = std::clamp(t, 0.0, 1.0);

    ProjectedParams p{};

    // fb_bound — interpolate in log-space
    double ln_F_lo = std::log(static_cast<double>(e_lo->optimal_params.fb_bound));
    double ln_F_hi = std::log(static_cast<double>(e_hi->optimal_params.fb_bound));
    double ln_F    = (1.0 - t) * ln_F_lo + t * ln_F_hi;
    p.fb_bound = static_cast<uint32_t>(std::round(std::exp(ln_F)));

    // sieve_bound — interpolate in log2-space, round to power of 2
    double log2_M_lo = std::log2(static_cast<double>(e_lo->optimal_params.sieve_bound));
    double log2_M_hi = std::log2(static_cast<double>(e_hi->optimal_params.sieve_bound));
    double log2_M    = (1.0 - t) * log2_M_lo + t * log2_M_hi;
    p.sieve_bound = nearestPow2(std::pow(2.0, log2_M));

    // lp1_bound — interpolate in log-space if both active; handle mixed/inactive
    uint64_t lp_lo = e_lo->optimal_params.lp1_bound;
    uint64_t lp_hi = e_hi->optimal_params.lp1_bound;
    if (lp_lo > 0 && lp_hi > 0) {
        double ln_lp_lo = std::log(static_cast<double>(lp_lo));
        double ln_lp_hi = std::log(static_cast<double>(lp_hi));
        double ln_lp    = (1.0 - t) * ln_lp_lo + t * ln_lp_hi;
        p.lp1_bound = static_cast<uint64_t>(std::round(std::exp(ln_lp)));
    } else if (lp_lo > 0) {
        // Scale active entry's ratio by projected F
        double ratio = static_cast<double>(lp_lo) /
                       static_cast<double>(e_lo->optimal_params.fb_bound);
        p.lp1_bound = static_cast<uint64_t>(std::round(ratio * p.fb_bound));
    } else if (lp_hi > 0) {
        double ratio = static_cast<double>(lp_hi) /
                       static_cast<double>(e_hi->optimal_params.fb_bound);
        p.lp1_bound = static_cast<uint64_t>(std::round(ratio * p.fb_bound));
    } else {
        p.lp1_bound = 0;
    }

    // Clamp LP bound to F² (prevents runaway extrapolation in log-space)
    if (p.lp1_bound > 0) {
        uint64_t f_squared = static_cast<uint64_t>(p.fb_bound) * static_cast<uint64_t>(p.fb_bound);
        if (p.lp1_bound > f_squared) {
            LOG(LOG_DEBUG_1) << "[Projection] Clamping interpolated LP bound from "
                             << p.lp1_bound << " to F²=" << f_squared;
            p.lp1_bound = f_squared;
        }
    }

    // Confidence: min(c_lo, c_hi) * max(0, 1 - gap/20) * min(1, size/5)
    uint32_t gap = e_hi->bit_length - e_lo->bit_length;
    double gap_factor = std::max(0.0, 1.0 - static_cast<double>(gap) / 20.0);
    double sparsity_factor = std::min(1.0, static_cast<double>(store_.size()) / 5.0);
    p.confidence = std::min(e_lo->confidence, e_hi->confidence) * gap_factor * sparsity_factor;

    p.source = ProjectedParams::Source::INTERPOLATION;
    double r = searchRadius(p.confidence);
    p.fb_bound_search_radius_pct    = r;
    p.sieve_bound_search_radius_pct = r;
    p.lp1_bound_search_radius_pct   = r;

    projectKernelParams(p, bits, neighbors, gpu_name);
    return p;
}

// ---------------------------------------------------------------------------
// extrapolate — OLS model evaluation
// ---------------------------------------------------------------------------

ProjectedParams ParameterProjector::extrapolate(
    uint32_t bits,
    const std::vector<const HistoryEntry*>& neighbors,
    const std::string& gpu_name) const
{
    ProjectedParams p{};
    double u_target = u_coord(bits);

    // fb_bound from fitted model
    if (fb_model_.valid) {
        double ln_F = fb_model_.ln_c + fb_model_.alpha * u_target;
        p.fb_bound = static_cast<uint32_t>(std::round(std::exp(ln_F)));
    } else {
        // Fallback: use nearest entry's value
        p.fb_bound = neighbors.front()->optimal_params.fb_bound;
    }

    // sieve_bound: use unanimous value if all entries agree, else 2^18 default
    {
        bool all_same = true;
        uint32_t first_M = neighbors.front()->optimal_params.sieve_bound;
        for (const auto* e : neighbors) {
            if (e->optimal_params.sieve_bound != first_M) {
                all_same = false;
                break;
            }
        }
        p.sieve_bound = all_same ? first_M : 262144u;
    }

    // lp1_bound from fitted model or median ratio
    if (lp_model_.valid) {
        double ln_lp = lp_model_.ln_c + lp_model_.alpha * u_target;
        p.lp1_bound = static_cast<uint64_t>(std::round(std::exp(ln_lp)));
    } else {
        // Collect lp1/F ratios from LP-active entries
        std::vector<double> ratios;
        for (const auto* e : neighbors) {
            if (e->optimal_params.lp1_bound > 0 && e->optimal_params.fb_bound > 0) {
                ratios.push_back(static_cast<double>(e->optimal_params.lp1_bound) /
                                 static_cast<double>(e->optimal_params.fb_bound));
            }
        }
        if (ratios.size() >= 3) {
            // Median ratio
            std::sort(ratios.begin(), ratios.end());
            double median = ratios[ratios.size() / 2];
            p.lp1_bound = static_cast<uint64_t>(std::round(median * p.fb_bound));
        } else {
            p.lp1_bound = 0;  // insufficient LP data
        }
    }

    // Clamp LP bound to F² (prevents runaway extrapolation in log-space)
    if (p.lp1_bound > 0) {
        uint64_t f_squared = static_cast<uint64_t>(p.fb_bound) * static_cast<uint64_t>(p.fb_bound);
        if (p.lp1_bound > f_squared) {
            LOG(LOG_DEBUG_1) << "[Projection] Clamping extrapolated LP bound from "
                             << p.lp1_bound << " to F²=" << f_squared;
            p.lp1_bound = f_squared;
        }
    }

    // Confidence: avg_confidence * exp(-delta_bits / 30)
    double sum_conf = 0.0;
    for (const auto* e : neighbors) sum_conf += e->confidence;
    double avg_conf = sum_conf / static_cast<double>(neighbors.size());

    uint32_t nearest_bits = neighbors.front()->bit_length;
    uint32_t delta = (bits > nearest_bits)
                     ? bits - nearest_bits
                     : nearest_bits - bits;
    p.confidence = avg_conf * std::exp(-static_cast<double>(delta) / 30.0);

    p.source = ProjectedParams::Source::EXTRAPOLATION;
    double r = searchRadius(p.confidence);
    p.fb_bound_search_radius_pct    = r;
    p.sieve_bound_search_radius_pct = r;
    p.lp1_bound_search_radius_pct   = r;

    projectKernelParams(p, bits, neighbors, gpu_name);
    return p;
}

// ---------------------------------------------------------------------------
// projectKernelParams — modal for constant, interpolated for variable
// ---------------------------------------------------------------------------

void ParameterProjector::projectKernelParams(
    ProjectedParams& out,
    uint32_t bits,
    const std::vector<const HistoryEntry*>& neighbors,
    const std::string& gpu_name) const
{
    // Collect GPU-matching neighbors
    std::vector<const HistoryEntry*> gpu_matches;
    for (const auto* e : neighbors) {
        if (e->environment.gpu_name == gpu_name)
            gpu_matches.push_back(e);
    }

    if (gpu_matches.empty()) {
        // No GPU match — all kernel params auto (0)
        std::memset(out.kernel_params, 0, sizeof(out.kernel_params));
        return;
    }

    // Constant params (indices 0,1,2,3,5,7): modal value
    constexpr int constant_indices[] = {0, 1, 2, 3, 5, 7};
    for (int idx : constant_indices) {
        std::unordered_map<uint32_t, int> freq;
        for (const auto* e : gpu_matches)
            ++freq[e->optimal_params.kernel_params[idx]];

        uint32_t best_val = 0;
        int best_count = 0;
        for (const auto& [val, count] : freq) {
            if (count > best_count || (count == best_count && val > best_val)) {
                best_val = val;
                best_count = count;
            }
        }
        out.kernel_params[idx] = best_val;
    }

    // Variable params (indices 4,6 — metaGridDim, sasGridDim):
    // inverse-distance weighted interpolation in bit-length.
    constexpr int variable_indices[] = {4, 6};
    for (int idx : variable_indices) {
        if (gpu_matches.size() < 2) {
            out.kernel_params[idx] = 0;  // auto-calculate
            continue;
        }

        double sum_w = 0.0, sum_wv = 0.0;
        for (const auto* e : gpu_matches) {
            uint32_t val = e->optimal_params.kernel_params[idx];
            if (val == 0) continue;  // skip auto entries
            double dist = std::abs(static_cast<int>(e->bit_length) -
                                   static_cast<int>(bits));
            double w = 1.0 / (1.0 + dist);
            sum_w  += w;
            sum_wv += w * static_cast<double>(val);
        }
        if (sum_w > 0.0) {
            out.kernel_params[idx] = static_cast<uint32_t>(std::round(sum_wv / sum_w));
        } else {
            out.kernel_params[idx] = 0;
        }
    }
}

} // namespace mpqs::autotune
