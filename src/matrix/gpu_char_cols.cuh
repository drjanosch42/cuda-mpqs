// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_char_cols.cuh
// GPU-accelerated standard character column computation (M8c).
#pragma once

#include "character_columns.h"  // CharacterColumns, structures::HostRelationBatch
#include "uint128_helper.cuh"   // mpqs::math::sub_mod (u128-safe modular subtraction)
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Jacobi symbol (a|n) for odd n > 0.
/// Algorithm 1.4.12 from Cohen, "A Course in Computational Algebraic Number Theory".
/// Returns +1, -1, or 0 (when gcd(a,n) > 1).
/// Prerequisite: n must be odd and positive — enforced by callers (aux primes are odd).
/// Host+device compatible; results are bit-identical to character_columns.cpp::jacobi_symbol().
__host__ __device__ inline int jacobi_symbol_dev(uint32_t a, uint32_t n) {
    a = a % n;
    int t = 1;
    while (a != 0) {
        // Remove all factors of 2 from a, flipping t when n ≡ 3 or 5 (mod 8).
        while ((a & 1u) == 0) {
            a >>= 1;
            uint32_t r = n & 7u;
            if (r == 3 || r == 5) t = -t;
        }
        // Quadratic reciprocity: swap a and n, flip t when both ≡ 3 (mod 4).
        uint32_t tmp = a; a = n; n = tmp;
        if ((a & 3u) == 3 && (n & 3u) == 3) t = -t;
        a = a % n;
    }
    return (n == 1) ? t : 0;
}

/// 64-bit Jacobi symbol (a|n) for odd n > 0. Returns +1, -1, or 0 (gcd(a,n) > 1).
/// Inlined here (rather than calling mpqs::sieve::jacobi_u64) so the branch-fixed
/// character bit is evaluable from device code without RDC linkage to mpqs_sieve.
/// MUST stay bit-identical to mpqs::sieve::jacobi_u64 (src/sieve/prime_algorithms.cu);
/// both implement Cohen Alg. 1.4.12. Branch-mode aux primes exceed uint32_t
/// (q > lp1_bound ~1e11), so the 32-bit jacobi_symbol_dev above cannot be reused.
__host__ __device__ inline int jacobi_symbol_dev_u64(uint64_t a, uint64_t n) {
    // n must be odd and positive (aux primes are odd primes); enforced by callers.
    a %= n;
    int t = 1;
    while (a != 0) {
        // Remove all factors of 2 from a, flipping t when n ≡ 3 or 5 (mod 8).
        while ((a & 1ull) == 0) {
            a >>= 1;
            uint64_t r = n & 7ull;
            if (r == 3 || r == 5) t = -t;
        }
        // Quadratic reciprocity: swap a and n, flip t when both ≡ 3 (mod 4).
        uint64_t tmp = a; a = n; n = tmp;
        if ((a & 3ull) == 3 && (n & 3ull) == 3) t = -t;
        a %= n;
    }
    return (n == 1) ? t : 0;
}

/// Branch-fixed field-element character bit (characters.md §3; Stage 3).
///
/// The genus-correct symbol substitutes the fixed Tonelli root t_s for the formal
/// √N in the field element α = (ax+b) − √N and reads the rational Legendre symbol:
///     bit = [ Legendre( ((ax+b) − t_s) / q ) == −1 ]   (0 when the symbol is +1 or 0).
/// Locking t_s (t_s² ≡ N mod q) pins one prime ideal q above q, making the symbol a
/// genus character (branch-sensitive) rather than the genus-blind NORM symbol.
///
/// Inputs are already reduced mod the 64-bit aux prime q:
///   axb_mod_q — the SIGNED (ax+b) reduced into [0,q). The caller (Stage 4, at
///               relation birth) forms it sign-aware:
///                   m = |ax+b| % q;
///                   axb_mod_q = (sign == -1) ? sub_mod(0, m, q) : m;
///               (uint512::mod_uint64 gives |ax+b| % q; sub_mod handles the sign).
///   t_s       — fixed Tonelli root in [0,q) (CharacterColumnComputer::tS()).
///   q         — odd 64-bit aux prime (> lp1_bound in branch mode).
/// diff = ((ax+b) − t_s) mod q via mpqs::math::sub_mod (u128-safe, no mod-mul).
///
/// Multiplicative homomorphism ⇒ additive over F2: the bit of a product field
/// element equals the XOR of the factor bits, so combined relations (Stage 5) XOR
/// the raw per-partial bits — never re-evaluate the symbol on the combined element.
///
/// __host__ __device__: host use at relation birth (Stage 4); device use in the
/// GPU character path (Stage 6).
__host__ __device__ inline int branchCharBit(uint64_t axb_mod_q, uint64_t t_s, uint64_t q) {
    uint64_t diff = mpqs::math::sub_mod(axb_mod_q, t_s, q);   // ((ax+b) − t_s) mod q
    return (jacobi_symbol_dev_u64(diff, q) == -1) ? 1 : 0;
}

/// GPU-accelerated standard character column computation.
/// Drop-in replacement for CharacterColumnComputer::compute() in the legacy matrix path.
///
/// This is the genus-blind NORM symbol path (--char_mode norm, the default).
/// Launches one thread per relation. Each thread evaluates k Jacobi symbols via
/// jacobi_symbol_dev(Q_i mod q, q) where Q_i mod q = (sqrt_Q[i]^2 - N) mod q.
/// Results are packed as uint32_t bitmasks (bit j set ↔ char col j is 1),
/// then unpacked into column-major CharacterColumns format.
///
/// The branch-fixed field-element symbol (--char_mode branch) is the distinct
/// genus character ((signed (ax+b) − t_s) / q) provided by branchCharBit() above;
/// it is wired into the relation-birth/combine/GPU paths in later stages.
///
/// Output is bit-identical to CharacterColumnComputer::compute() for all inputs.
///
/// @param batch       Source relation batch (must have valid sqrt_Q).
/// @param aux_primes  k auxiliary primes (from CharacterColumnComputer::auxPrimes()).
/// @param n_mod_q     N mod q for each aux prime (from CharacterColumnComputer::nModQ()).
/// @return CharacterColumns identical to CharacterColumnComputer::compute().
CharacterColumns gpuComputeCharacterColumns(
    const structures::HostRelationBatch& batch,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q);

/// GPU character columns from device-resident sqrt_Q (M9d).
/// Avoids the sqrt_Q D→H→D round-trip in the legacy MatrixStage path.
/// Reads sqrt_Q directly from the device RelationBatchView pointer.
///
/// @param d_sqrt_Q     Device pointer to sqrt_Q array (from RelationBatchView).
/// @param n_rels       Number of relations.
/// @param aux_primes   k auxiliary primes (host vector).
/// @param n_mod_q      N mod q for each aux prime (host vector).
/// @return CharacterColumns identical to the HostRelationBatch overload.
CharacterColumns gpuComputeCharacterColumns_device(
    const uint512* d_sqrt_Q,
    uint32_t n_rels,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q);

} // namespace matrix
} // namespace mpqs
