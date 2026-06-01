// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace mpqs {
namespace math {

// 1. Type Definition
// ------------------
// NVCC and GCC/Clang support __int128 natively on 64-bit systems.
typedef unsigned __int128 uint128_t;

// 2. Safe Modular Arithmetic Helpers
// ----------------------------------

/**
 * @brief Computes (a + b) % m.
 * Safe against overflow of a+b.
 * Assumes a < m and b < m.
 */
__host__ __device__ __forceinline__ 
uint64_t add_mod(uint64_t a, uint64_t b, uint64_t m) {
    // uint64_t is sufficient if we check for wrap manually, 
    // but using u128 is cleaner and likely compiles to carry-flag instructions.
    return (uint64_t)((uint128_t(a) + b) % m);
}

/**
 * @brief Computes (a - b) % m.
 * Safe against underflow (unsigned wrapping).
 * Assumes a < m and b < m.
 */
__host__ __device__ __forceinline__ 
uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t m) {
    // If a >= b, standard subtraction.
    // If a < b, we need (a + m) - b.
    return (a >= b) ? (a - b) : (m - (b - a));
}

/**
 * @brief Computes (a * b) % m.
 * ESSENTIAL: Handles cases where a*b exceeds 2^64.
 * This is critical for Hensel Lifting (mod p^2) where p^2 > 2^32.
 */
__host__ __device__ __forceinline__ 
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (uint64_t)((uint128_t(a) * b) % m);
}

/**
 * @brief Computes (base ^ exp) % m.
 * Type-optimized: arguments are u64, intermediate math is u128.
 */
__host__ __device__ __forceinline__ 
uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t res = 1;
    base %= m;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod(res, base, m);
        base = mul_mod(base, base, m);
        exp /= 2;
    }
    return res;
}

} // namespace math
} // namespace mpqs