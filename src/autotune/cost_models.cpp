// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "cost_models.h"
#include <cmath>
#include <algorithm>

namespace mpqs::autotune {

// Calibration constants (RTX 5070 Ti benchmarks)
// T_matrix(100000) = alpha_mat * 100000^1.1 = 0.3s
// 100000^1.1 ≈ 316228, so alpha_mat = 0.3 / 316228 ≈ 9.49e-7
static constexpr double ALPHA_MAT = 9.49e-7;   // 0.3 / pow(100000, 1.1)

// T_linalg: power law fit from campaign data (75d-95d) + RSA-100.
// Exponent 1.17 reflects sub-quadratic empirical BW scaling:
//   Stage 2 (lingen) dominates with O(L^1.65), L ~ N/128.
//   GPU SpMM throughput saturates; AutoTune adds sub-linear overhead.
// Calibration: T_la(301485) ~ 25s (RSA-100, RTX 5070 Ti, m=n=256).
static constexpr double ALPHA_BW      = 8.5e-6;
static constexpr double BW_EXPONENT   = 1.17;

double estimateMatrixTime(uint32_t fb_size) {
    if (fb_size == 0) return 0.0;
    return ALPHA_MAT * std::pow(static_cast<double>(fb_size), 1.1);
}

double estimateLinalgTime(uint32_t fb_size) {
    if (fb_size == 0) return 0.0;
    return ALPHA_BW * std::pow(static_cast<double>(fb_size), BW_EXPONENT);
}

SieveSearchBounds deriveSieveSearchBounds(uint32_t F_auto, uint32_t /*M_auto*/,
                                          uint32_t F_heuristic,
                                          double f_max_multiplier) {
    SieveSearchBounds b;
    b.fb_lo = static_cast<uint32_t>(0.3 * F_auto);
    b.fb_hi = static_cast<uint32_t>(3.0 * F_auto);
    // F runaway prevention: cap fb_hi at f_max_multiplier × F_heuristic.
    // Prevents optimizer from pushing F beyond 2× the heuristic default,
    // which causes enormous BW matrices with catastrophic LA cost.
    if (F_heuristic > 0) {
        uint32_t f_cap = static_cast<uint32_t>(f_max_multiplier * F_heuristic);
        b.fb_hi = std::min(b.fb_hi, f_cap);
    }
    b.M_min_log2 = 15;  // 2^15 = 32768
    b.M_max_log2 = 19;  // 2^19 = 524288 — cap to avoid large-M GPU hangs
    b.lp_lo = 0;        // LP-disabled is always a candidate
    b.lp_hi = std::min(static_cast<uint64_t>(F_auto) * 50, uint64_t{500'000'000});
    return b;
}

} // namespace mpqs::autotune
