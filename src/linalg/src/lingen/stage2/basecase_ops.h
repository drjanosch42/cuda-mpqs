// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "lingen/types.h"

namespace lingen {
namespace stage2 {

/**
 * @brief Structure representing a column operation (elimination).
 * Target column ^= Source column.
 */
struct ColOp {
    uint16_t target;
    uint16_t source;
};

/**
 * @brief Low-level kernels for the Block Berlekamp-Massey algorithm.
 */
namespace ops {

/**
 * @brief Computes the discrepancy matrix Delta at time t.
 * 
 * Delta = coeff_t( S(x) * Pi(x) )
 * 
 * Note: S is N x N, Pi is M x M (where M=2N). 
 * This computes the interaction of S with the top N rows of Pi.
 * 
 * @param S_dev   Input sequence S (N x N binary matrices).
 * @param S_len   Length of S.
 * @param Pi_dev  Generator matrix Pi (M x M binary matrices).
 * @param Pi_len  Current length (degree+1) of Pi.
 * @param Delta   Output buffer (M uint64s, representing M columns of N bits).
 * @param t       Current time step.
 * @param stream  CUDA Stream.
 */
template<int N, int M>
void compute_discrepancy(
    const PolyMatrixView<N>& S_dev,
    const PolyMatrixView<M>& Pi_dev,
    uint64_t* Delta,
    int t,
    cudaStream_t stream
);

/**
 * @brief Applies the elimination matrix Tau to Pi.
 * 
 * Pi(x) <- Pi(x) * Tau
 * 
 * @param Pi_dev    Generator matrix.
 * @param ops_dev   List of column operations (on Device).
 * @param n_ops     Number of operations.
 * @param stream    CUDA Stream.
 */
template<int N, int M>
void apply_update(
    PolyMatrixView<M> Pi_dev,
    const ColOp* ops_dev,
    int n_ops,
    cudaStream_t stream
);

/**
 * @brief Shifts specific columns of Pi by x.
 * 
 * For columns j where shift_mask has bit j set: Col_j(x) <- Col_j(x) * x.
 * This increases the polynomial degree of those columns by 1.
 * 
 * @param Pi_dev        Generator matrix.
 * @param shift_mask_lo Mask for cols 0..63.
 * @param shift_mask_hi Mask for cols 64..127 (if M=128).
 * @param stream        CUDA Stream.
 */
template<int N, int M>
void apply_shift(
    PolyMatrixView<M> Pi_dev,
    uint64_t shift_mask_lo,
    uint64_t shift_mask_hi,
    cudaStream_t stream
);

} // namespace ops
} // namespace stage2
} // namespace lingen
