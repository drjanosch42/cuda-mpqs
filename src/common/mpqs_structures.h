// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cstdint>
#include <functional>
#include "uint512.cuh"

namespace mpqs {

/// Hash functor for unsigned __int128 keys in std::unordered_map/set.
struct Hash128 {
    size_t operator()(unsigned __int128 v) const {
        uint64_t lo = static_cast<uint64_t>(v);
        uint64_t hi = static_cast<uint64_t>(v >> 64);
        return std::hash<uint64_t>{}(lo)
             ^ (std::hash<uint64_t>{}(hi) * 0x9e3779b97f4a7c15ULL);
    }
};

namespace structures {

/**
 * @brief Represents a single relation found by the sieve.
 * Mathematical meaning: (ax + b)^2 = Product(factors) * (-1)^sign * 2^val2 mod N
 */
struct Relation {
    uint64_t relation_index; // Global ID of this relation

    // LHS parameters: (ax + b)
    mpqs::uint512 a; 
    mpqs::uint512 b; 
    int64_t x;       // Sieve offset

    // RHS factorization data
    uint8_t sign_of_Q;         // -1 or +1
    uint32_t val_2_exp;       // Exponent of factor 2
    
    struct Factor {
        uint32_t p_index; // Index in the Factor Base
        uint8_t count;    // Multiplicity
    };

    uint32_t num_factors;
    /// @brief Fixed capacity factor list.
    /// Sufficient for 512-bit numbers (avg factors ~15-20, max < 64).
    /// Contains the factorization of (ax+b)^2-N up to sign and powers of 2.
    /// I.e. it incorporates the sieved prime factors + prime factors of a.
    Factor factors[64];
    /// 48 should be sufficient:
    /// Product of first 32 primes: 169 bits
    /// Product of first 48 primes: 288 bits
    /// Product of first 64 primes: 417 bits 
    
    /// @brief The remainder after dividing out all FB factors.
    /// If 1, it is a full relation.
    /// If > 1, it is a Large Prime (up to 128 bits).
    /// If 0, it indicates an error/invalid LP.
    unsigned __int128 large_prime_remainder; 
};

} // namespace structures
} // namespace mpqs

