// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "uint512.cuh"

namespace mpqs {
namespace math {

// =============================================================================
// Modular Exponentiation
// =============================================================================

/**
 * @brief Computes (base ^ exp) % mod.
 * Uses right-to-left binary method.
 */
__host__ __device__ inline uint512 modpow(uint512 base, uint512 exp, const uint512& mod) {
    uint512 res((uint32_t)1);

    // Pre-reduce base to simplify loop logic
    base.mod(mod);

    while (!exp.is_zero()) {
        // If LSB is 1, multiply result
        if (exp.limbs[0] & 1) {
            res.mul_mod(base, mod);
        }
        // Square base
        base.mul_mod(base, mod);
        
        // exp >>= 1
        exp.rshift(1);
    }
    return res;
}

/**
 * @brief Computes (base ^ exp) % mod.
 * Uses right-to-left binary method.
 */
__host__ __device__ inline uint512 modpow(uint512 base, uint32_t exp, const uint512& mod) {
    uint512 res((uint32_t)1);
    
    // Pre-reduce base to simplify loop logic
    base.mod(mod);

    while (exp) {
        // If LSB is 1, multiply result
        if (exp & 1) {
            res.mul_mod(base, mod);
        }
        // Square base
        base.mul_mod(base, mod);
        
        exp >>= 1;
    }
    return res;
}
 

// =============================================================================
// Modular Inverse (Unsigned Extended Euclidean)
// =============================================================================

/**
 * @brief Computes the modular multiplicative inverse a^-1 mod m.
 * 
 * Algorithm: Extended Euclidean Algorithm adjusted for strict unsigned arithmetic.
 * Returns 0 if gcd(a, m) != 1 (inverse does not exist).
 */
__host__ __device__ inline uint512 mod_inverse(const uint512& a_in, const uint512& m) {
    uint512 r0 = a_in;
    uint512 r1 = m;
    uint512 u0((uint32_t)1);
    uint512 u1((uint32_t)0);

    // Loop until remainder is 0
    while (!r1.is_zero()) {
        uint512 q = r0;
        uint512 r_next;
        
        // q = r0 / r1, r_next = r0 % r1
        q.div_mod_core(r1, &r_next);

        // Update coefficients: u_next = u0 - q*u1 (mod m)
        // Calculation: u_next = (u0 - (q * u1) % m) % m
        
        // 1. Term T = (q * u1) % m
        uint512 term = q;
        term.mul_mod(u1, m);

        // 2. Subtraction with Modulo Wrap handling
        // We want (u0 - term) mod m.
        // If u0 < term, we perform (u0 + m) - term.
        uint512 u_next = u0;
        if (u_next < term) {
            u_next.add(m);
        }
        u_next.sub(term);

        // Shift states
        r0 = r1; 
        r1 = r_next;
        u0 = u1; 
        u1 = u_next;
    }

    // Check GCD
    if (!r0.is_one()) {
        return uint512((uint32_t)0); // Error: Not Coprime
    }

    return u0;
}

/**
 * @brief Computes the modular multiplicative inverse a^-1 mod m (Mixed Type).
 * 
 * Optimization: Since m is 32-bit, we reduce 'a' immediately and perform 
 * the Extended Euclidean Algorithm using native registers instead of BigInt arrays.
 * 
 * @param a_in The large integer to invert.
 * @param m The 32-bit modulus.
 * @return The inverse as a uint512 (containing a 32-bit value), or 0 if not coprime.
 */
__host__ __device__ inline uint32_t mod_inverse(const uint512& a_in, uint32_t m) {
    // 1. Immediate Reduction: Reduce 512-bit input to 32-bit.
    uint32_t r0 = a_in.mod_uint32(m);
    uint32_t r1 = m;

    uint32_t u0 = 1;
    uint32_t u1 = 0;

    // 2. 32-bit Extended Euclidean Algorithm
    // Matches logic of the BigInt version but uses CPU/GPU native registers.
    while (r1 != 0) {
        uint32_t q = r0 / r1;
        uint32_t r_next = r0 % r1;

        // Update coefficients: u_next = u0 - q*u1 (mod m)
        // Use uint64_t for intermediate multiplication to prevent overflow
        uint32_t term = (uint32_t)(((uint64_t)q * u1) % m);

        uint32_t u_next;
        if (u0 < term) {
            // Handle wrap-around (negative result in signed logic)
            u_next = (u0 + m) - term;
        } else {
            u_next = u0 - term;
        }

        // Shift states
        r0 = r1;
        r1 = r_next;
        u0 = u1;
        u1 = u_next;
    }

    // 3. Check GCD
    if (r0 != 1) {
        return 0; // Error: Not Coprime
    }

    return u0;
}

// =============================================================================
// GCD
// =============================================================================

__host__ __device__ inline uint512 gcd(uint512 a, uint512 b) {
    while (!b.is_zero()) {
        uint512 t = b;
        a.mod(b);
        b = a;
        a = t;
    }
    return a;
}

// =============================================================================
// Number Theoretic Helpers (Legendre, Sqrt)
// =============================================================================

/**
 * @brief Computes Legendre Symbol (a / p).
 * Returns:
 *  0 : if a % p == 0
 *  1 : if a is quadratic residue mod p
 *  p-1 : if a is non-residue (represents -1 in unsigned arithmetic)
 */
__host__ __device__ inline uint512 legendre_symbol(uint512 a, const uint512& p) {
    a.mod(p);
    if (a.is_zero()) return uint512((uint32_t)0);

    // Euler's Criterion: a^((p-1)/2) mod p
    uint512 exp = p;
    exp.sub(uint512((uint32_t)1));
    exp.rshift(1); // (p-1)/2

    return modpow(a, exp, p);
}

/**
 * @brief Computes Square Root Modulo p (where p = 3 mod 4).
 * Returns r such that r^2 = n (mod p).
 * Returns 0 if n is not a quadratic residue.
 */
__host__ __device__ inline uint512 sqrt_mod_3mod4(uint512 n, const uint512& p) {
    // 1. Verify p % 4 == 3
    if ((p.limbs[0] & 3) != 3) return uint512((uint32_t)0); // Error

    // 2. Verify Quadratic Residue
    uint512 ls = legendre_symbol(n, p);
    if (!ls.is_one()) return uint512((uint32_t)0); // Error

    // 3. r = n^((p+1)/4) mod p
    uint512 exp = p;
    exp.add(uint512((uint32_t)1));
    exp.rshift(2); // (p+1)/4

    return modpow(n, exp, p);
}

/**
 * @brief Chinese Remainder Theorem for 2 equations.
 * x = r1 mod m1
 * x = r2 mod m2
 * Assumes gcd(m1, m2) = 1.
 */
__host__ __device__ inline uint512 crt_2(uint512 r1, uint512 m1, uint512 r2, uint512 m2) {
    // M = m1 * m2
    uint512 M = m1;
    M.mult(m2); // Note: M must fit in 512 bits for uint512 CRT

    // y1 = m2^-1 mod m1
    uint512 y1 = mod_inverse(m2, m1);
    
    // y2 = m1^-1 mod m2
    uint512 y2 = mod_inverse(m1, m2);

    // term1 = (r1 * m2 * y1) % M
    uint512 term1 = r1;
    term1.mul_mod(m2, M);    
    term1.mul_mod(y1, M);

    // term2 = (r2 * m1 * y2) % M
    uint512 term2 = r2;
    term2.mul_mod(m1, M);
    term2.mul_mod(y2, M);

    // res = (term1 + term2) % M
    term1.add(term2);
    term1.mod(M);
    
    return term1;
}


// =============================================================================
// Math Helper: Q(x) = (ax+b)^2 - N Calculation (Unsigned Strict)
// =============================================================================
// CONTRACT: |a| and |b| are small relative to N, N < 2^512.
// FORMALLY: -2^511 < (ax+b)^2 - N < 2^511
// This is guaranteed by our construction of a,b, and x.
__device__ __forceinline__ 
void calculateQ_unsigned(mpqs::uint512 a, mpqs::uint512 b, int32_t x, mpqs::uint512 N, 
                       mpqs::uint512& Q_out, int8_t& sign_out) 
{
    // Compute term = ax + b (algebraically).
    // We exploit uint512 underflow.
    mpqs::uint512 ax = a;
    uint32_t abs_x = (x < 0) ? -x : x;
    ax.mult_uint32(abs_x);

    mpqs::uint512 temp; // Stores ax + b "mod 2^512"
    
    if (x >= 0) {
        temp = ax;
        temp.add(b);
    } else {
        temp = b;
        temp.sub(ax);
    }
    // Compute |ax + b|
    int8_t sign;
    mpqs::uint512 abs = temp.abs_twos_complement(sign);

    // 2. Square: (ax+b)^2 < 2^512 + epsilon, epsilon > 0 only if 2^511 < N.
    abs.mult(abs);

    // 3. Subtract N: (ax+b)^2 - N
    abs.sub(N); // lies strictly between -2^511 and 2^511 by our contract
    // Therefore we may use the 512th bit to safely determine the sign and
    // the absolute value with abs_twos_complement.
    Q_out = abs.abs_twos_complement(sign_out);
}

// =============================================================================
// Math Helper: sqrt{|Q(x)|} = |ax+b| Calculation
// =============================================================================

// CONTRACT: |a| and |b| are small relative to N, N < 2^512.
// FORMALLY: |ax+b| < sqrt{N} (|ax+b| < 2^511 - 1 is actually sufficient)
//
// 5-arg form: additionally returns the SIGN of the genuine (signed) (ax+b) via
// sign_axb_out (+1 or -1, as produced by abs_twos_complement). The sign is needed
// at relation birth by the branch-fixed character path (Stage 4): the
// branch-fixed symbol reads the SIGNED field element ((ax+b) - t_s) mod q, so the
// sign discarded when reducing to sqrt_out = |ax+b| must be recovered here.
__host__ __device__ __forceinline__
void calculate_sqrt_of_QX(mpqs::uint512 a, mpqs::uint512 b, int32_t x,
                          mpqs::uint512& sqrt_out, int8_t& sign_axb_out)
{
    // Compute term = ax + b (algebraically) as above.
    // We exploit uint512 underflow.
    mpqs::uint512 ax = a;
    uint32_t abs_x = (x < 0) ? -x : x;
    ax.mult_uint32(abs_x);

    mpqs::uint512 temp; // Stores intermediate values

    if (x >= 0) {
        // b + ax
        temp = ax;
        temp.add(b);
    } else {
        // b - ax
        temp = b;
        temp.sub(ax);
    }
    sqrt_out = temp.abs_twos_complement(sign_axb_out);
}

// 3-arg form (legacy): identical to the 5-arg form but discards the (ax+b) sign.
// Existing callers (the genus-blind NORM path) are unaffected.
__host__ __device__ __forceinline__
void calculate_sqrt_of_QX(mpqs::uint512 a, mpqs::uint512 b, int32_t x, mpqs::uint512& sqrt_out)
{
    int8_t sign;
    calculate_sqrt_of_QX(a, b, x, sqrt_out, sign);
}

} // namespace math
} // namespace mpqs
