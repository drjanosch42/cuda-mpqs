// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "autotune_types.h"
#include <cstdint>

// Forward-declare to avoid pulling in orchestrator.h
namespace mpqs { struct MPQSConfig; }

namespace mpqs::autotune {

/// Heuristic matrix construction time: T = alpha_mat * fb_size^1.1
/// Calibrated: estimateMatrixTime(100000) ~= 0.3s (RSA-100 on RTX 5070 Ti).
double estimateMatrixTime(uint32_t fb_size);

/// Heuristic Block Wiedemann time: T = alpha_bw * fb_size^1.17
/// Empirical fit from 75d-95d campaign + RSA-100 benchmarks (RTX 5070 Ti).
/// Sub-quadratic scaling reflects GPU SpMM saturation + lingen dominance.
double estimateLinalgTime(uint32_t fb_size);

/// Derive search bounds from base config's auto-tuned defaults.
/// Uses fb_bound from config (or determineParams default) as center point:
///   fb_lo = 0.3 * F_auto,  fb_hi = 3.0 * F_auto
///   lp_hi = min(F_auto * 50, 500'000'000)
SieveSearchBounds deriveSieveSearchBounds(uint32_t F_auto, uint32_t M_auto,
                                          uint32_t F_heuristic = 0,
                                          double f_max_multiplier = 2.0);

} // namespace mpqs::autotune
