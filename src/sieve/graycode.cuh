// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

namespace mpqs {
namespace sieve {

// ============================================================================
// Device Inline Functions (Gray Codes)
// ============================================================================

/**
 * @brief Returns the Gray code of an index.
 * Formula: G(i) = i ^ (i >> 1).
 *
 * Properties:
 * - Two successive Gray codes differ by exactly one bit.
 * - Used to iterate through combinations with minimal changes.
 */
__host__ __device__ __forceinline__
uint32_t gray(uint32_t index){
    return index ^ (index >> 1);
}

/**
 * @brief Determines the index of the bit that changes between steps.
 *
 * Calculates which bit flips when moving from gray(index) to gray(index+1).
 * This corresponds to the number of trailing zeros in the XOR difference.
 */
__host__ __device__ __forceinline__
uint32_t advanceGray(uint32_t index){
    // gray(index+1) ^ gray(index) always has exactly one bit set.
    // We want the index of that bit.
    // mpqs::ctz32 wraps __ffs(x)-1 on device and __builtin_ctz on host.
    return mpqs::ctz32(gray(index+1) ^ gray(index));
    // return __ffs((gray(index+1)^gray(index)))-1;
}

/*
inline uint32_t advanceGray(uint32_t index) {
#ifdef __CUDA_ARCH__
    return __ffs(gray(index + 1) ^ gray(index)) - 1;
#else
    return __builtin_ctz(gray(index + 1) ^ gray(index));
#endif
}
*/


__host__ __device__ __forceinline__
uint32_t grayBitToFlip(uint32_t index1,uint32_t index2){
    return mpqs::ctz32(gray(index1) ^ gray(index2));
    // return __ffs((gray(index1)^gray(index2)))-1;
}

} // namespace sieve
} // namespace mpqs
