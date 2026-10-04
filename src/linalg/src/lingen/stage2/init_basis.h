// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/**
 * @file init_basis.h
 * @brief Stage 2 initialization basis (Coppersmith/Thomé start) and F_init.
 *
 * Given the Krylov sequence S = (S_0, ..., S_{L-1}), S_i ∈ GF(2)^{m×n}, the
 * initialization scans the candidate columns v_{i,j} = S_i[:, j] ∈ GF(2)^m in
 * strict lexicographic order (i, j) and greedily accepts every candidate that is
 * linearly independent of the previously accepted ones, stopping at rank m.
 *
 *   pairs = ((i_0, j_0), ..., (i_{m-1}, j_{m-1}))   (accepted candidates, in order)
 *   t0    = i_{m-1} + 1                              (first t with rank(S_0..S_{t-1}) = m)
 *
 * and the initial generator F ∈ GF(2)[x]^{dim×dim}, dim = m + n, deg F ≤ t0:
 *
 *   F[0][j][j]            = 1   for j < n      (identity on the top n rows)
 *   F[t0 - i_k][n + k][j_k] = 1   for k < m      (monomials x^{t0 - i_k})
 *   gamma                 = (t0, ..., t0) ∈ Z^dim
 *
 * If rank m is not reached over all L coefficients the result is the degenerate
 * start {t0 = 0, pairs = ∅}: F = identity on the top n rows only, gamma = 0.
 *
 * Two implementations with bit-identical outputs are provided:
 *  - the CPU reference (find_init_basis_cpu / build_f_init_cpu), and
 *  - a device path (launch_find_init_basis_gpu / launch_build_f_init_gpu) that
 *    reads S directly from device memory, writes F_init / gamma directly into
 *    the solver's device buffers, and returns only a 16-byte meta record.
 *    The equivalence proof is in the documentation of k_find_init_basis
 *    (init_basis.cu).
 *
 * Layouts (identical on host and device):
 *  - S_i is m×n row-major, nw = ⌈n/64⌉ words per row; entry (r, j) is bit j&63 of
 *    word i·m·nw + r·nw + (j>>6).
 *  - F coefficient d is dim×dim row-major, stride_F = ⌈dim/64⌉ words per row,
 *    coefficient stride mat_stride = dim·stride_F words; entry (r, c) of F[d] is
 *    bit c&63 of word d·mat_stride + r·stride_F + (c>>6).
 *  - Pivot mask: mw = ⌈m/64⌉ words, bit p set iff p is a pivot row of the basis.
 */

#include <cstdint>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace lingen {
namespace stage2 {

/// Result of the initialization-basis scan.
struct InitBasisResult {
    int t0 = 0;                                ///< loop start; 0 on rank failure
    int rank = 0;                              ///< rank reached (m on success)
    std::vector<std::pair<int, int>> pairs;    ///< accepted (i, j), in scan order; empty on failure
    std::vector<uint64_t> pivot_mask;          ///< mw words; used for the failure log only
};

/// Layout of the device meta record written by the find kernel.
enum InitBasisMeta : int {
    INIT_META_T0     = 0,   ///< t0 (0 on failure)
    INIT_META_RANK   = 1,   ///< rank reached
    INIT_META_STATUS = 2,   ///< 0 = rank m reached, 1 = rank condition failed
    INIT_META_WORDS  = 4    ///< record size in ints (16 bytes; slot 3 reserved)
};

/// Largest m supported by the device path (MW ≤ 8 words of 64 bits).
constexpr int INIT_BASIS_GPU_MAX_M = 512;

/**
 * @brief CPU reference scan (the original BasecaseSolver::find_initialization_basis).
 * Does not log; see log_init_rank_failure().
 * @param S  L coefficients, each m·⌈n/64⌉ words.
 */
InitBasisResult find_init_basis_cpu(const std::vector<std::vector<uint64_t>>& S, int m, int n);

/**
 * @brief CPU reference F_init / gamma (the original BasecaseSolver::build_f_init).
 * On failure (pairs empty) F holds the identity on the top n rows only.
 * @param F      out: t0+1 coefficients of dim·⌈dim/64⌉ words each.
 * @param gamma  out: dim entries, all t0.
 */
void build_f_init_cpu(const InitBasisResult& init, int m, int n,
                      std::vector<std::vector<uint64_t>>& F, std::vector<int>& gamma);

/**
 * @brief Emits the two LOG_ERROR_CRITICAL lines of a rank failure
 * ("Rank condition failed! ..." and "Missing pivot indices: [...]"),
 * exactly as the CPU reference always has.
 */
void log_init_rank_failure(int rank, int m, const std::vector<uint64_t>& pivot_mask);

/// @brief True iff the device path supports (m, n): 1 ≤ m ≤ 512, n ≥ 1.
bool init_basis_gpu_supported(int m, int n);

/**
 * @brief Device scan: one warp, S read in place, basis in shared memory.
 *
 * Writes d_meta[0..3] = {t0, rank, status, 0}; d_pairs[2k], d_pairs[2k+1] = (i_k, j_k)
 * for k < rank; d_pivmask[0..mw) = pivot mask (written on failure only).
 * All outputs are device memory; nothing is synchronized.
 *
 * @param d_S  len coefficients in the S layout above.
 * @pre init_basis_gpu_supported(m, n).
 */
void launch_find_init_basis_gpu(const uint64_t* d_S, int len, int m, int n,
                                int* d_meta, int* d_pairs, uint64_t* d_pivmask,
                                cudaStream_t stream);

/**
 * @brief Device F_init / gamma: ORs the identity and the m monomials into d_F0
 * (which must be zero-initialized over t0+1 coefficients) and sets d_Gamma[r] = t0.
 * Reads t0 / status from d_meta, so it has no host dependency on the scan.
 *
 * @param d_F0     coefficient 0 of the solver's F buffer (dim×dim layout above).
 * @param d_Gamma  dim ints.
 */
void launch_build_f_init_gpu(uint64_t* d_F0, int* d_Gamma, const int* d_meta,
                             const int* d_pairs, int m, int n, cudaStream_t stream);

} // namespace stage2
} // namespace lingen
