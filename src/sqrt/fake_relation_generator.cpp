// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "fake_relation_generator.h"
#include "math_utils.cuh" // For modpow, crt_2, etc.

#include <iostream>
#include <algorithm>
#include <vector>

namespace mpqs {
namespace test {

using mpqs::uint512;

// =============================================================================
// Constructor & Setup
// =============================================================================

FakeRelationGenerator::FakeRelationGenerator(int bit_size, int fb_size)
    : bit_size_(bit_size), fb_size_(fb_size) 
{
    std::random_device rd;
    rng_.seed(rd());

    // 1. Pre-calculate Factor Base (Odd primes only)
    factor_base_cache_.reserve(fb_size_);
    
    // Simple sieve or trial division for small primes
    int candidate = 3; // Start from 3
    while (factor_base_cache_.size() < (size_t)fb_size_) {
        bool is_p = true;
        for (int k = 2; k * k <= candidate; ++k) {
            if (candidate % k == 0) {
                is_p = false;
                break;
            }
        }
        if (is_p) {
            factor_base_cache_.push_back(candidate);
        }
        candidate += 2;
    }
}

// =============================================================================
// Main Generation Logic
// =============================================================================

FakeData FakeRelationGenerator::generate(int num_relations) {
    FakeData data;
    
    // 1. Generate Factors p, q (Blum Integers: p = 3 mod 4)
    // This allows easy square roots using exponentiation.
    data.p = get_random_prime_3mod4(bit_size_ / 2);
    do {
        data.q = get_random_prime_3mod4(bit_size_ / 2);
    } while (data.p == data.q);

    // N = p * q
    data.N = data.p;
    data.N.mult(data.q); // N = p * q
    
    // Copy FB
    data.factor_base = factor_base_cache_;
    
    // Resize solution bits (packed uint64)
    size_t num_u64 = (num_relations + 63) / 64;
    data.solution_bits.resize(num_u64, 0);

    // 2. Generate Relations
    // Strategy: Construct the RHS (factors) first, then solve for LHS (ax+b).
    // To make validation trivial, we make EVERY generated relation a perfect square.
    // This ensures ComputeY works regardless of which relations are selected.
    
    for (int k = 0; k < num_relations; ++k) {
        mpqs::structures::Relation rel;
        rel.relation_index = k;
        rel.large_prime_remainder = 0; // No large primes in fake data

        // A. Construct RHS (Smooth Value)
        // -----------------------------------------------------------------
        // We pick random EVEN exponents so the RHS is a quadratic residue.
        
        mpqs::uint512 rhs_val((uint32_t)1);
        
        // Handle -1: Exponent must be even -> (-1)^2k = 1.
        // So effectively -1 contributes nothing to the value, but we can 
        // set the flag and pretend the exponent is 0 or 2.
        // Let's assume sign is positive for simplicity of Sqrt, 
        // or randomly pick sign and ensure total exponent is even.
        // For simplicity: Always positive RHS for this benchmark.
        rel.sign_of_Q = 1; 

        // Handle 2:
        uint64_t exp2 = (rng_() % 4) * 2; // 0, 2, 4, 6...
        rel.val_2_exp = exp2;
        if (exp2 > 0) {
	    mpqs::uint512 two((uint32_t)2);
	    mpqs::uint512 exp((uint64_t)exp2);
            mpqs::uint512 term = mpqs::math::modpow(two, exp, data.N);
            rhs_val.mul_mod(term, data.N);
        }

        // Handle Odd Factors
        int num_factors = rng_() % 10 + 1; // 1 to 10 distinct prime factors
        rel.num_factors = 0;

        for (int f = 0; f < num_factors; ++f) {
            int fb_idx = rng_() % fb_size_;
            // Ensure even exponent (2, 4, 6)
            uint8_t count = (uint8_t)((rng_() % 3 + 1) * 2); 
            
            // Add to struct
            if (rel.num_factors < 64) {
                rel.factors[rel.num_factors].p_index = fb_idx;
                rel.factors[rel.num_factors].count = count;
                rel.num_factors++;
            }

            // Multiply into RHS value
            mpqs::uint512 p(factor_base_cache_[fb_idx]);
            mpqs::uint512 exp((uint32_t)count);
            mpqs::uint512 term = mpqs::math::modpow(p, exp, data.N);
            rhs_val.mul_mod(term, data.N);
        }

        // B. Solve for LHS: (ax + b)^2 = RHS mod N
        // -----------------------------------------------------------------
        // 1. Compute root = sqrt(RHS) mod N
        // Since RHS is a constructed square and p,q = 3 mod 4:
        // r_p = rhs^((p+1)/4) mod p
        // r_q = rhs^((q+1)/4) mod q
        mpqs::uint512 r_p = mpqs::math::sqrt_mod_3mod4(rhs_val, data.p);
        mpqs::uint512 r_q = mpqs::math::sqrt_mod_3mod4(rhs_val, data.q);
        
        mpqs::uint512 root = mpqs::math::crt_2(r_p, data.p, r_q, data.q);

        // 2. We need (ax + b) = root mod N.
        // To stress test ComputeX, we pick random 'a' and 'x' and solve for 'b'.
        // a is usually small (polynomial coefficient).
        uint64_t a_small = (rng_() % 0xFFFFFF) + 1; // Ensure a != 0
        rel.a = mpqs::uint512(a_small);
        
        // x is usually the sieve offset (can be positive or negative).
        // Let's pick a random 64-bit int for x.
        int64_t x_val = (int64_t)rng_(); // Random 64-bit
        rel.x = x_val; // Store in relation

        // Calculate term = a * x
        mpqs::uint512 val_a = rel.a;
        
        // Helper to calculate a*x mod N safely using the logic we discussed
        mpqs::uint512 val_ax;
        
        val_ax = val_a;
        val_ax.mul_add_mod_signed(x_val, uint512((uint32_t)0), data.N); // Compute a*x mod N

        // We want (ax + b) = root mod N  =>  b = (root - ax) mod N
        mpqs::uint512 val_b = root;
        val_b.sub_mod(val_ax, data.N);
        
        rel.b = val_b;

        data.relations.push_back(rel);

        // Mark this relation as part of the solution
        data.solution_bits[k / 64] |= (1ULL << (k % 64));
    }

    return data;
}

// =============================================================================
// Math Helpers
// =============================================================================

mpqs::uint512 FakeRelationGenerator::get_random_uint512(int bits) {
    mpqs::uint512 r;
    // Fill limbs with random data
    for (int i = 0; i < 16; ++i) {
        r.limbs[i] = (uint32_t)rng_();
    }
    
    // For simplicity, just handling full limbs or reducing the top.
    // This generator is roughly sufficient for producing p, q.
    // Ensure MSB is set for correct size
    int top_limb = (bits - 1) / 32;
    int top_bit  = (bits - 1) % 32;
    
    // Clear limbs above size
    for (int i = top_limb + 1; i < 16; ++i) r.limbs[i] = 0;
    
    // Mask top limb
    uint32_t mask = (1ULL << (top_bit + 1)) - 1;
    r.limbs[top_limb] &= mask;
    r.limbs[top_limb] |= (1U << top_bit); // Set MSB
    
    // Ensure odd
    r.limbs[0] |= 1;
    
    return r;
}

bool FakeRelationGenerator::is_prime(const mpqs::uint512& n, int rounds) {
    // Miller-Rabin test
    // We can assume math_utils might have one, but implementing a basic one here
    // or relying on the one used in previous phases.
    // For brevity, assuming a basic implementation or external reference.
    // Here is a quick implementation:
    
    if (n.limbs[0] % 2 == 0) return false;
    
    mpqs::uint512 d = n;
    d.sub(mpqs::uint512((uint32_t)1)); // d = n - 1
    int s = 0;
    
    // while (d % 2 == 0) { d /= 2; s++; }
    while ((d.limbs[0] & 1) == 0) {
        d.rshift(1);
        s++;
    }
    
    for (int i = 0; i < rounds; ++i) {
        // Pick random a in [2, n-2]
        mpqs::uint512 a = get_random_uint512(32); // Small base is fine
        // ensure a < n-1
        
        mpqs::uint512 x = mpqs::math::modpow(a, d, n);
        
        mpqs::uint512 n_minus_1 = n;
        n_minus_1.sub(mpqs::uint512((uint32_t)1));

        if (x.is_one() || x == n_minus_1) continue;
        
        bool composite = true;
        for (int r = 1; r < s; ++r) {
            x.mul_mod(x, n);
            if (x == n_minus_1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

mpqs::uint512 FakeRelationGenerator::get_random_prime_3mod4(int bits) {
    while (true) {
        mpqs::uint512 p = get_random_uint512(bits);
        // Ensure p % 4 == 3
        p.limbs[0] |= 3; 
        
        if (is_prime(p)) return p;
    }
}

} // namespace test
} // namespace mpqs
