// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include <vector>
#include <cstdint>
#include <random>

#include "mpqs_structures.h" // For Relation struct
#include "uint512.cuh"

namespace mpqs {
namespace test {

/**
 * @brief Container for the synthetic factorization problem.
 */
struct FakeData {
    /// @brief The composite number N = p * q
    mpqs::uint512 N;
    
    /// @brief The secret factors (p and q are congruent to 3 mod 4)
    mpqs::uint512 p;
    mpqs::uint512 q;

    /// @brief The factor base (Odd primes only)
    std::vector<uint32_t> factor_base;

    /// @brief A set of relations where (ax+b)^2 = Smooth mod N
    std::vector<mpqs::structures::Relation> relations;

    /// @brief A bitmask indicating which relations form the solution.
    /// For this fake generator, usually all generated relations are perfect squares.
    std::vector<uint64_t> solution_bits;
};

/**
 * @brief Synthetic Factorization Problem Generator.
 * * Generates N = p*q and a set of relations that are mathematically consistent.
 * Used to benchmark the Sqrt Step arithmetic without running the Sieve/Linear Algebra.
 */
class FakeRelationGenerator {
public:
    /**
     * @brief Constructor.
     * @param bit_size Approximate bit length of N.
     * @param fb_size Number of odd primes in the factor base.
     */
    FakeRelationGenerator(int bit_size, int fb_size);

    /**
     * @brief Generates the fake problem instance.
     * @param num_relations Number of relations to generate.
     * @return Fully populated FakeData struct.
     */
    FakeData generate(int num_relations);

private:
    // =========================================================================
    // Helpers
    // =========================================================================

    mpqs::uint512 get_random_uint512(int bits);
    mpqs::uint512 get_random_prime_3mod4(int bits);
    bool is_prime(const mpqs::uint512& n, int rounds = 5);

    // =========================================================================
    // Internal State
    // =========================================================================

    int bit_size_;
    int fb_size_;
    std::vector<uint32_t> factor_base_cache_; // Odd primes
    std::mt19937_64 rng_;
};

} // namespace test
} // namespace mpqs
