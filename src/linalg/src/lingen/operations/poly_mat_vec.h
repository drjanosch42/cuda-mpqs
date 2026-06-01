// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "lingen/types.h"

namespace lingen {

/**
 * @brief Execution Backend Selector.
 * Allows runtime switching between optimized GPU kernels and reference CPU code
 * for debugging numerical issues in Stage 2.
 */
enum class PolyMatVecBackend {
    GPU_Kernel,
    CPU_Reference
};

/**
 * @brief Operations for applying a Polynomial Matrix to a Vector Series.
 * 
 * Computes convolution: W[t] = sum_{k=0}^{deg} Pi[k] * V[t-k]
 * 
 * Dimensions:
 *  - M: Dimension of the square polynomial matrix Pi (e.g., 64, 128).
 *  - G: Effective dimension of the vector (G <= M). Used for padding support.
 * 
 * Memory Layout:
 *  - Pi: PolyMatrixView (contiguous [Coeff][Row][Word]).
 *  - V, W: Contiguous arrays of vector words.
 */
class PolyMatVec {
public:
    /**
     * @brief Computes W = Pi(x) * V (Right Multiplication).
     * 
     * Used in the recursive step of MSLGDC and reconstruction.
     * 
     * @param Pi_dev  Polynomial matrix on Device (Length Pi_len).
     * @param V_dev   Input vector series on Device (Length Lin).
     * @param W_dev   Output vector series on Device (Length Lout).
     * @param backend GPU or CPU reference.
     */
    template<int M, int G>
    static void apply_right(
        const PolyMatrixView<M>& Pi_dev,
        const uint64_t* V_dev, int Lin,
        uint64_t* W_dev, int Lout,
        PolyMatVecBackend backend = PolyMatVecBackend::GPU_Kernel,
        cudaStream_t stream = 0
    );

    /**
     * @brief Computes W = V * Pi(x) (Left Multiplication).
     * 
     * Often computed as W^T = Pi^T * V^T, or via specialized kernels.
     */
    template<int M, int G>
    static void apply_left(
        const PolyMatrixView<M>& Pi_dev,
        const uint64_t* V_dev, int Lin,
        uint64_t* W_dev, int Lout,
        PolyMatVecBackend backend = PolyMatVecBackend::GPU_Kernel,
        cudaStream_t stream = 0
    );

    /**
     * @brief Computes transpose of a polynomial matrix: B(x) = A(x)^T.
     * 
     * Needed for Left application if using the transpose trick.
     * @param dst Output buffer (must be allocated).
     * @param src Input view.
     */
    template<int M>
    static void transpose(
        PolyMatrixView<M> dst,
        PolyMatrixView<M> src,
        cudaStream_t stream = 0
    );
};

} // namespace lingen
