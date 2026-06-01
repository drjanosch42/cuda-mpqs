// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "autotune_types.h"

// Forward-declare to avoid circular include (orchestrator.h -> autotune.h)
namespace mpqs { struct MPQSConfig; }

namespace mpqs::autotune {

/// Evaluate a parameter set by running a truncated sieve on an ephemeral orchestrator.
///
/// Constructs a fresh MPQSOrchestrator with the given parameters, runs TuningStage()
/// + TruncatedSieveRun(), extracts telemetry, and returns a RuntimeEstimate.
/// The probe orchestrator is destroyed (RAII) and its temp directory cleaned up.
///
/// @param base_config  Base configuration (N, device_id, etc. preserved)
/// @param fb_bound     Factor base bound to evaluate (0 = auto)
/// @param sieve_bound  Sieve interval half-width (0 = auto, must be power of 2)
/// @param lp1_bound    Large prime bound (0 = disabled)
/// @param truncation_frac  Hard ceiling fraction (default 0.12)
/// @param eta_convergence_threshold  Relative spread for early exit (default 0.05)
/// @param min_eta_samples  Minimum ETA samples before convergence (default 3)
RuntimeEstimate estimateRuntime(
    const mpqs::MPQSConfig& base_config,
    uint32_t fb_bound,
    uint32_t sieve_bound,
    uint64_t lp1_bound,
    double truncation_frac = 0.12,
    double eta_convergence_threshold = 0.05,
    uint32_t min_eta_samples = 3
);

} // namespace mpqs::autotune
