// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "uint512.cuh"

namespace mpqs {
namespace math {

/**
 * @brief Montgomery Arithmetic Context for uint512.
 * * Handles conversion to/from Montgomery form and arithmetic operations (mul, sqr, pow).
 * * Mathematical properties:
 * - R = 2^512
 * - N is the modulus (must be odd).
 * - n_prime = -N^-1 mod 2^32 (used for digit-level reduction).
 * - r2 = R^2 mod N (used for conversion into Montgomery form).
 */
class Montgomery {
public:
    mpqs::uint512 N;
    mpqs::uint512 R2;   // R^2 mod N
    uint32_t n_prime;   // -1/N mod 2^32

    // =========================================================================
    // Constructor / Setup
    // =========================================================================

    __host__ __device__ Montgomery() : n_prime(0) {}

    /**
     * @brief Initializes the context for a specific modulus N.
     * Computes the necessary constants R^2 and n_prime.
     */
    __host__ __device__ explicit Montgomery(const mpqs::uint512& modulus) : N(modulus) {
        // 1. Compute n_prime = -N^-1 mod 2^32
        uint32_t n0 = N.limbs[0];
        uint32_t inv = 1;
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 5; i++) inv *= (2 - n0 * inv);
        n_prime = -inv;

        // 2. Compute R^2 mod N
        // We want R^2 = 2^1024 mod N.
        // We start with 1 and double it 1024 times, reducing mod N each time.
        // This is slower than the previous shortcut but guarantees correctness for any N size.

	// We compute 2^511 mod N.
        mpqs::uint512 val = mpqs::uint512((uint32_t)0);
	val.limbs[15] = (uint32_t)(1<<31);
	val.mod(N);
	// We double val mod N to get 2^512 mod N.
	val.double_mod(N);
	// We square val to get 2^1024, this self-referencing call is save.
	val.mul_mod(val, N);
        R2 = val;
    }    

    // =========================================================================
    // Conversions
    // =========================================================================

    /**
     * @brief Converts a standard residue to Montgomery form.
     * Computes: res = a * R mod N
     */
    __host__ __device__ mpqs::uint512 transform(const mpqs::uint512& a) const {
        // mont_mul(a, R^2) = a * R^2 * R^-1 = a * R
        return mul(a, R2);
    }

    /**
     * @brief Converts a Montgomery residue back to standard form.
     * Computes: res = a * R^-1 mod N
     */
    __host__ __device__ mpqs::uint512 reduce(const mpqs::uint512& a) const {
        // mont_mul(a, 1) = a * 1 * R^-1 = a * R^-1
        mpqs::uint512 one((uint32_t)1);
        return mul(a, one);
    }

    // =========================================================================
    // Arithmetic
    // =========================================================================

    /**
     * @brief Montgomery Multiplication.
     * Computes: res = (a * b * R^-1) mod N
     * Implementation: CIOS (Coarsely Integrated Operand Scanning).
     */
    __host__ __device__ mpqs::uint512 mul(const mpqs::uint512& a, const mpqs::uint512& b) const {
        mpqs::uint512 T; // Initialized to 0
        uint32_t extra = 0; // The 17th limb (carry out of 512 bits)

        // Outer Loop: Scan limbs of A
        #if defined(__NVCC__) && defined(__CUDA_ARCH__)
            #pragma unroll
        #endif      
        for (int i = 0; i < 16; i++) {
            uint32_t a_i = a.limbs[i];

            // 1. Calculate reduction factor u for this row
            // u = (T[0] + a_i * b[0]) * n_prime mod 2^32
            // Note: T.limbs[0] is the current lowest word of the accumulator
            uint32_t u = (T.limbs[0] + a_i * b.limbs[0]) * n_prime;

            // 2. Inner Loop: Compute T + a_i*B + u*N
            // We do this in one pass to avoid storing the full 1024-bit product.
            unsigned __int128 carry = 0;

            #if defined(__NVCC__) && defined(__CUDA_ARCH__)
                #pragma unroll
            #endif      
            for (int j = 0; j < 16; j++) {
                // Calculation: T[j] + A[i]*B[j] + u*N[j] + carry
                unsigned __int128 sum = (unsigned __int128)T.limbs[j]
                                      + (unsigned __int128)a_i * b.limbs[j]
                                      + (unsigned __int128)u * N.limbs[j]
                                      + carry;
                
                // Shift down: result for position j goes into position j-1
                // effectively dividing by 2^32
                if (j > 0) T.limbs[j-1] = (uint32_t)sum;

                carry = sum >> 32;
            }

            // 3. Handle the carry out of the top (the new value for T[15] and extra)
            // The previous 'extra' is conceptually at position 16.
            // We shift it down to 15.
            unsigned __int128 sum_top = (unsigned __int128)extra + carry;
            T.limbs[15] = (uint32_t)sum_top;
            extra = (uint32_t)(sum_top >> 32);
        }

        // 4. Final Conditional Subtraction
        // If (extra:T) >= N, subtract N.
        // Since we are mod N, and the result is bounded < 2N, one subtraction is sufficient.
        if (extra != 0 || T >= N) {
            T.sub(N);
        }

        return T;
    }

    /**
     * @brief Montgomery Squaring.
     * Computes: res = (a * a * R^-1) mod N
     */
    __host__ __device__ mpqs::uint512 sqr(const mpqs::uint512& a) const {
        return mul(a, a);
    }

    /**
     * @brief Montgomery Modular Exponentiation.
     * Computes: res = (base ^ exp) mod N (Result in Montgomery form)
     * @param base_mont Base in Montgomery form.
     * @param exp Standard integer exponent (NOT Montgomery form).
     * @return Result in Montgomery form.
     */
    __host__ __device__ mpqs::uint512 pow(mpqs::uint512 base_mont, mpqs::uint512 exp) const {
        // 1. Initialize result to Montgomery 1 (which is R mod N)
        // transform(1) == mul(1, R2) == 1 * R^2 * R^-1 = R
        mpqs::uint512 one((uint32_t)1);
        mpqs::uint512 res = transform(one); 

        // 2. Binary Exponentiation
        // We use the existing limbs array for bit scanning
        while (!exp.is_zero()) {
            if (exp.limbs[0] & 1) {
                res = mul(res, base_mont);
            }
            base_mont = sqr(base_mont);
            exp.rshift(1);
        }
        return res;
    }
};

} // namespace math
} // namespace mpqs
