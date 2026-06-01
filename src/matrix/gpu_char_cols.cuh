// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_char_cols.cuh
// GPU-accelerated standard character column computation (M8c).
#pragma once

#include "character_columns.h"  // CharacterColumns, structures::HostRelationBatch
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

/// GPU-accelerated standard character column computation.
/// Drop-in replacement for CharacterColumnComputer::compute() in the legacy matrix path.
///
/// Launches one thread per relation. Each thread evaluates k Jacobi symbols via
/// jacobi_symbol_dev(Q_i mod q, q) where Q_i mod q = (sqrt_Q[i]^2 - N) mod q.
/// Results are packed as uint32_t bitmasks (bit j set ↔ char col j is 1),
/// then unpacked into column-major CharacterColumns format.
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
