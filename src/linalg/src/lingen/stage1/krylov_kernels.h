// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lingen {
namespace stage1 {

/**
 * @brief Batched Projection: S_i = X^T * V_i for i in [0, BATCH_SIZE).
 * 
 * Computes 'count' projections simultaneously to reuse X loaded from memory.
 * 
 * Dimensions:
 *  X: N x m (Row-major packed bits)
 *  V_buf: Contiguous buffer of 'count' vectors V_i, each N x n.
 *  S_buf: Contiguous buffer of 'count' matrices S_i, each m x n.
 * 
 * @param N Total rows in X and V.
 * @param count Number of vectors in the batch (<= BATCH_CAPACITY).
 * @param X Device pointer to X.
 * @param V_buf Device pointer to start of V buffer.
 * @param V_stride_bytes Bytes to skip to get from V_i to V_{i+1}.
 * @param S_buf Device pointer to start of S buffer.
 * @param S_stride_bytes Bytes to skip to get from S_i to S_{i+1}.
 */
template<int M_BLOCK, int N_BLOCK>
void launch_krylov_batch_proj(
    int N_rows, 
    int count,
    const uint64_t* X, 
    const uint64_t* V_buf, 
    size_t V_stride_bytes,
    uint64_t* S_buf, 
    size_t S_stride_bytes,
    cudaStream_t stream
);

/**
 * @brief Per-batch 64×64 sub-block bit-transpose of packed GF(2) matrices.
 *
 * Transposes 'count' matrices from (src_rows × src_cols) to (src_cols × src_rows)
 * using shared-memory 64×64 bit-transpose sub-blocks.
 *
 * @param src Source buffer (count contiguous matrices, row-major packed uint64_t)
 * @param dst Destination buffer (count contiguous transposed matrices)
 * @param count Number of matrices in the batch
 * @param src_rows Rows per source matrix (m_block_)
 * @param src_cols Columns per source matrix (n_block_)
 * @param src_stride_words Words per row in source: (src_cols + 63) / 64
 * @param dst_stride_words Words per row in dest: (src_rows + 63) / 64
 * @param stream CUDA stream for kernel launch
 */
void launch_transpose_batch(
    const uint64_t* src, uint64_t* dst,
    int count, int src_rows, int src_cols,
    size_t src_stride_words, size_t dst_stride_words,
    cudaStream_t stream);

} // namespace stage1
} // namespace lingen
