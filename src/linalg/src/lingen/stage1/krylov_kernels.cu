// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage1/krylov_kernels.h"
#include <cstdio>

namespace lingen {
namespace stage1 {

// Atomic XOR helper
// Uses native atomicXor for unsigned long long if available (SM 6.0+),
// otherwise falls back to a CAS loop.
__device__ inline void atomicXor64(uint64_t* address, uint64_t val) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long val_ull = (unsigned long long)val;

#if __CUDA_ARCH__ >= 600
    atomicXor(addr_as_ull, val_ull);
#else
    unsigned long long old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, val_ull ^ assumed);
    } while (assumed != old);
#endif
}
 
/**
 * @brief Batch Projection Kernel
 * 
 * Strategy:
 *  - Grid loops over rows k of N.
 *  - Block Y dimension iterates over the batch (0..count-1).
 *  - Block X dimension iterates over bits of X (rows of S) (0..M-1).
 *  - Each thread accumulates one row of one S matrix (S_batchIdx[row_r]).
 *  - X[k] is broadcast to all batch-processing warps to save bandwidth.
 * 
 * Note: To maximize occupancy, we handle multiple batch items per block if possible,
 * or map block.y to batch index.
 */
template<int M, int N>
__global__ void kernel_krylov_batch_proj(
    int N_rows,
    int count,
    const uint64_t* __restrict__ X,
    const uint64_t* __restrict__ V_buf,
    size_t V_stride_words,
    uint64_t* __restrict__ S_buf,
    size_t S_stride_words
) {
    // Indices
    int r = threadIdx.x; // Row of S (0..M-1), corresponding to bit of X
    int b = blockIdx.y;  // Batch index (0..count-1)

    if (r >= M || b >= count) return;

    // Stride Setup
    constexpr int WORDS_N = N / 64;
    constexpr int WORDS_M = M / 64;
    
    // Pointer to this batch's V and Output S
    const uint64_t* V_ptr = V_buf + b * V_stride_words;
    
    // Local accumulator for S_b[r] (Row r of S_b)
    uint64_t acc[WORDS_N];
    #pragma unroll
    for (int w = 0; w < WORDS_N; ++w) acc[w] = 0;

    // Identify bit in X
    int x_word_idx = r / 64;
    int x_bit_idx  = r & 63; //r % 64;
    uint64_t x_mask = 1ULL << x_bit_idx;

    // Grid-stride loop over N
    for (int k = blockIdx.x; k < N_rows; k += gridDim.x) {
        // Load X[k] (all batches share this load)
        // Note: L1 cache should handle the broadcast if blocks are scheduled together
        uint64_t x_val = X[k * WORDS_M + x_word_idx];

        if (x_val & x_mask) {
            // XOR V_b[k] into acc
            // V is packed: Row k starts at k * WORDS_N
            const uint64_t* v_row = V_ptr + k * WORDS_N;
            #pragma unroll
            for (int w = 0; w < WORDS_N; ++w) {
                acc[w] ^= v_row[w];
            }
        }
    }

    // Atomic Reduction to global memory
    // S_buf is [Batch0][Batch1]...
    // Each S is M x N packed. Row r is at S_base + r * WORDS_N.
    uint64_t* S_base = S_buf + b * S_stride_words;
    uint64_t* S_row  = S_base + r * WORDS_N;

    #pragma unroll
    for (int w = 0; w < WORDS_N; ++w) {
        if (acc[w]) {
            atomicXor64(&S_row[w], acc[w]);
        }
    }
}

template<int M, int N>
void launch_krylov_batch_proj(
    int N_rows, 
    int count,
    const uint64_t* X, 
    const uint64_t* V_buf, 
    size_t V_stride_bytes,
    uint64_t* S_buf, 
    size_t S_stride_bytes,
    cudaStream_t stream
) {
    if (count == 0) return;

    // Configuration
    // Thread Block: X=M (one thread per row of S), Y=1 (one batch item per block)
    // Grid: X=Cover N, Y=Batch Count
    
    int threads_x = M;
    if (threads_x < 32) threads_x = 32; // Warp size min
    
    // We can handle multiple batch items per block to improve X-locality?
    // For now, mapping 1 block -> 1 batch item is simplest and correct.
    // X is cached in L2.
    
    dim3 block(threads_x, 1);
    
    // Grid X: Enough to cover N, but not too many to avoid atomic contention
    int grid_x = 128; 
    
    // Grid Y: One per batch item
    dim3 grid(grid_x, count);

    size_t v_stride_words = V_stride_bytes / 8;
    size_t s_stride_words = S_stride_bytes / 8;

    kernel_krylov_batch_proj<M, N><<<grid, block, 0, stream>>>(
        N_rows, count, X, V_buf, v_stride_words, S_buf, s_stride_words
    );
}

// Explicit Instantiations for common sizes
#define INSTANTIATE(M, N) \
    template void launch_krylov_batch_proj<M, N>(int, int, const uint64_t*, const uint64_t*, size_t, uint64_t*, size_t, cudaStream_t);

INSTANTIATE(64, 64)
INSTANTIATE(64, 128)
INSTANTIATE(64, 256)
INSTANTIATE(64, 512)
INSTANTIATE(128, 64)
INSTANTIATE(128, 128)
INSTANTIATE(128, 256)
INSTANTIATE(128, 512)
INSTANTIATE(256, 64)
INSTANTIATE(256, 128)
INSTANTIATE(256, 256)
INSTANTIATE(256, 512)
INSTANTIATE(512, 64)
INSTANTIATE(512, 128)
INSTANTIATE(512, 256)
INSTANTIATE(512, 512)

// =============================================================================
// Per-batch 64x64 sub-block bit-transpose kernel
// =============================================================================

__global__ void kernel_transpose_batch(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int count,                // number of matrices in batch
    int src_rows,             // m_block_
    int src_cols,             // n_block_
    size_t src_stride_words,  // (src_cols + 63) / 64
    size_t dst_stride_words   // (src_rows + 63) / 64
) {
    __shared__ uint64_t smem[64];

    int mat_idx = blockIdx.z;
    if (mat_idx >= count) return;

    // Decompose blockIdx.x into sub-block (br, bc)
    int num_bc = (src_cols + 63) / 64;
    int br = blockIdx.x / num_bc;     // row block index
    int bc = blockIdx.x % num_bc;     // col block index
    int t = threadIdx.x;              // 0..63

    // Pointers to this matrix
    size_t src_mat_words = (size_t)src_rows * src_stride_words;
    size_t dst_mat_words = (size_t)src_cols * dst_stride_words;
    const uint64_t* src_mat = src + mat_idx * src_mat_words;
    uint64_t* dst_mat = dst + mat_idx * dst_mat_words;

    // Load: thread t reads row (br*64 + t), column-word bc
    int src_r = br * 64 + t;
    smem[t] = (src_r < src_rows) ? src_mat[src_r * src_stride_words + bc] : 0;
    __syncthreads();

    // Transpose: build row (bc*64 + t) from column t of smem
    uint64_t result = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (smem[i] & (1ULL << t))
            result |= (1ULL << i);
    }

    // Store: row (bc*64 + t), column-word br
    int dst_r = bc * 64 + t;
    if (dst_r < src_cols)
        dst_mat[dst_r * dst_stride_words + br] = result;
}

void launch_transpose_batch(
    const uint64_t* src, uint64_t* dst,
    int count, int src_rows, int src_cols,
    size_t src_stride_words, size_t dst_stride_words,
    cudaStream_t stream)
{
    int num_br = (src_rows + 63) / 64;
    int num_bc = (src_cols + 63) / 64;
    dim3 block(64);
    dim3 grid(num_br * num_bc, 1, count);

    kernel_transpose_batch<<<grid, block, 0, stream>>>(
        src, dst, count, src_rows, src_cols,
        src_stride_words, dst_stride_words);
}

} // namespace stage1
} // namespace lingen
