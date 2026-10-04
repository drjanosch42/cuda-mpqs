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


/// One Gray-code transition: which bit flipped, and whether it is set afterwards.
struct GrayFlip {
    uint32_t bit;
    bool     nowSet;
};

/**
 * @brief One step of a cyclic Gray walk over a block of blockSize codes (blockSize a power of two).
 *
 * Moves grayCode from gray((walkIndex-1) mod blockSize) to gray(walkIndex mod blockSize) and reports
 * the flip. gray(j) ^ gray(j-1) = 1 << ctz(j), and the wrap gray(blockSize-1) = blockSize/2 ->
 * gray(0) = 0 flips the top bit. With x = walkIndex mod blockSize both are ctz(x | blockSize/2):
 * for x != 0 the lowest set bit of x is at most blockSize/2.
 *
 * blockMask = blockSize - 1, blockHalf = max(blockSize / 2, 1) (for blockSize == 1 there is nothing
 * to flip; the step then reports bit 0).
 */
__host__ __device__ __forceinline__
GrayFlip advanceGrayCyclic(uint32_t walkIndex, uint32_t blockMask, uint32_t blockHalf, uint32_t& grayCode){
    GrayFlip flip;
    flip.bit = mpqs::ctz32((walkIndex & blockMask) | blockHalf);
    grayCode ^= 1u << flip.bit;
    flip.nowSet = (grayCode >> flip.bit) & 1u;
    return flip;
}

__host__ __device__ __forceinline__
uint32_t grayBitToFlip(uint32_t index1,uint32_t index2){
    return mpqs::ctz32(gray(index1) ^ gray(index2));
    // return __ffs((gray(index1)^gray(index2)))-1;
}

} // namespace sieve
} // namespace mpqs
