// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace lingen {

/**
 * @brief Dense Matrix Multiplication over GF(2) using the MPB (Multiple Matrices Per Block) strategy.
 * 
 * Computes C = A * B for a batch of matrices.
 * 
 * Mathematical Definition:
 *   Let A, B be N x N binary matrices.
 *   C[i, j] = XOR_{k=0}^{N-1} (A[i, k] AND B[k, j])
 * 
 * Memory Layout:
 *   Matrices are stored continuously in row-major order (bits packed into uint64_t).
 *   Stride (bytes) = N * (N / 8).
 *   Stride (words) = N * (N / 64).
 * 
 *   The 'batch_stride' parameter allows processing polynomial coefficients where
 *   matrices might be interleaved or contiguous.
 */

// --- GPU Interface ---

/**
 * @brief Launch the optimized CUDA kernel for batch MatMul.
 * 
 * This wrapper selects the optimal "Matrices Per Block" (MPB) configuration
 * based on N (32, 64, 128, 256, 512).
 * 
 * @param N            Dimension of matrices (must be 32, 64, 128, 256, 512).
 * @param A            Pointer to input A (Device memory).
 * @param B            Pointer to input B (Device memory).
 * @param C            Pointer to output C (Device memory).
 * @param num_matrices Number of matrix multiplications to perform.
 * @param stream       CUDA stream.
 */
void matmul_gf2_gpu(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream = 0);

/**
 * @brief Variant of batch MatMul where matrix A is broadcast (shared) across all batch items.
 * 
 * Computes C[k] = A * B[k] for k in 0..num_matrices-1.
 * A is a single N x N matrix.
 * B and C are arrays of num_matrices matrices.
 */
void matmul_gf2_gpu_broadcast_A(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream = 0);

/**
 * @brief Legacy implementation
 */
void matmul_gf2_gpu_legacy(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream = 0);
  
// --- CPU Reference Interface ---

/**
 * @brief CPU Reference implementation of Batch MatMul GF(2).
 * 
 * Intended for validation and debugging.
 * 
 * @param N            Dimension.
 * @param A            Host pointer A.
 * @param B            Host pointer B.
 * @param C            Host pointer C.
 * @param num_matrices Number of matrices.
 */
void matmul_gf2_cpu(int N, const void* A, const void* B, void* C, size_t num_matrices);

} // namespace lingen
