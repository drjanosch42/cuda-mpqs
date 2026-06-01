// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cstdint>
#include <vector>

// These are in the global 'mpqs' namespace
#include "uint512.cuh"
#include "math_utils.cuh"
#include "logger/hpc_logger.h"

namespace mpqs {
namespace sieve {

/**
 * @brief The primary state object for the Sieving Phase.
 * Contains the number to be factored, the factor base,
 * and polynomial parameters.
 */
struct factoringData {
    // ========================================================================
    // Global Constants (mpqs::uint512)
    // ========================================================================
    mpqs::uint512 N;          // Number to be factored
    mpqs::uint512 a_target;   // Target magnitude for polynomial coefficient 'a'
    mpqs::uint512 a;          // Current 'a' coefficient

    // B values for solving congruences.
    std::vector<mpqs::uint512> B_values;

    // ========================================================================
    // Factor Base (Strict 32-bit)
    // ========================================================================
    std::vector<uint32_t> factorBase; // Primes p
    std::vector<uint32_t> rootN;      // Roots r where r^2 = N (mod p)

    // ========================================================================
    // Polynomial State
    // ========================================================================
    std::vector<uint32_t> a_factors; // Indices of primes forming 'a'

    uint32_t current_a_index;
    uint32_t max_a_index;
    uint32_t lowerHalfStart;
    uint32_t upperHalfStart;

    // ========================================================================
    // Sieve Configuration
    // ========================================================================
    uint32_t M;    // Sieve interval
    uint32_t F;    // Factor Base bound
    uint32_t size; // Factor Base actual size
};

/// Snapshot of initial a-factor state after init_a_factors().
/// Immutable for the lifetime of a factorization. Used by resetAndAdvanceTo()
/// to reconstruct polynomial state at any global a-index.
/// One of 2 permitted submodule changes (Spec Section 11.1).
struct AFactorsSnapshot {
    std::vector<uint32_t> a_factors;    ///< Canonical (all-even) indices
    uint32_t lowerHalfStart = 0;
    uint32_t upperHalfStart = 0;
    uint32_t shc_dim = 0;               ///< = a_factors.size()
};

} // namespace sieve
} // namespace mpqs
