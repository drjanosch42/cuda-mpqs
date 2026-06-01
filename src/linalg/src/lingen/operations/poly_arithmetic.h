// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "lingen/types.h" // For PolyMatrixView

namespace lingen {

/**
 * @brief Basic Polynomial Arithmetic Operations.
 * 
 * Provides GPU kernels and CPU reference implementations for:
 *  - C(x) = A(x) ^ B(x)
 *  - C(x) = 0
 *  - C(x) = A(x)
 *  - C(x) = A(x) * B(x) (Naive CPU only, for validation)
 */

// --- GPU Interface ---

/**
 * @brief C(x) ^= A(x).  (Accumulating XOR / Addition).
 * 
 * Performs element-wise XOR on the matrix coefficients.
 * Processing length is min(C.len, A.len).
 */
template<int N>
void poly_add_gpu(PolyMatrixView<N> C, PolyMatrixView<N> A, cudaStream_t stream = 0);

/**
 * @brief C(x) = 0.
 */
template<int N>
void poly_zero_gpu(PolyMatrixView<N> C, cudaStream_t stream = 0);

/**
 * @brief C(x) = A(x). (Copy).
 */
template<int N>
void poly_copy_gpu(PolyMatrixView<N> C, PolyMatrixView<N> A, cudaStream_t stream = 0);

// --- CPU Reference Interface ---

/**
 * @brief CPU Reference: C(x) ^= A(x).
 */
template<int N>
void poly_add_cpu(PolyMatrixView<N> C, PolyMatrixView<N> A);

/**
 * @brief CPU Reference: C(x) = A(x) * B(x) (Naive convolution).
 * 
 * Used to validate Karatsuba.
 * Assumes C is zeroed initially.
 * C should have length len(A) + len(B) - 1.
 */
template<int N>
void poly_mul_naive_cpu(PolyMatrixView<N> C, PolyMatrixView<N> A, PolyMatrixView<N> B);

} // namespace lingen
