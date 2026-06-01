// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_product_char_packed.cuh
//
// GPU product character columns from merged sqrt_Q values (M9f).
// No Montgomery recomputation needed — sqrt_Q products were computed
// during M9e merges. Evaluates 32 Jacobi symbols per row:
//   Q_i mod q = (sqrt_Q[i]^2 - N) mod q → jacobi(Q_i mod q, q)
// Same kernel structure as char_col_kernel (CC1) in gpu_char_cols.cu,
// reading from merged sqrt_Q instead of original relation sqrt_Q.

#pragma once

#include "character_columns.h"  // CharacterColumns
#include "uint512.cuh"          // uint512
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// GPU product character columns from merged sqrt_Q values (M9f).
///
/// No Montgomery recomputation needed — sqrt_Q products were computed
/// during M9e merges. Evaluates 32 Jacobi symbols per row:
///   Q_i mod q = (sqrt_Q[i]^2 - N) mod q → jacobi(Q_i mod q, q)
///
/// Same kernel structure as char_col_kernel (CC1) in gpu_char_cols.cu,
/// reading from merged sqrt_Q instead of original relation sqrt_Q.
///
/// @param d_merged_sqrt_Q  Device pointer to merged sqrt_Q array [n_rows].
/// @param n_rows           Number of merged (alive) rows.
/// @param aux_primes       k auxiliary primes (host, from CharacterColumnComputer).
/// @param n_mod_q          N mod q for each aux prime (host).
/// @return CharacterColumns with k columns, each of length n_rows.
CharacterColumns gpuProductCharCols_packed(
    const uint512* d_merged_sqrt_Q,
    uint32_t n_rows,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q);

} // namespace matrix
} // namespace mpqs
