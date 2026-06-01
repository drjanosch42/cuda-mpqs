// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "kernels.h"
#include "vec_type.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// =================================================================================
// 3. Optimized Kernels (Using VecType)
// =================================================================================

// --- Kernel: Tiled COO (Writeback) ---
template<int VEC_BITS>
__global__ void kernel_tiled_coo_writeback(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ coords,
    const uint32_t* __restrict__ block_ptrs,
    row_idx_t row_start_offset,
    row_idx_t num_blocks,
    row_idx_t total_rows,
    uint32_t rows_per_block,
    int col_bits                  // number of bits for column index in packing
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    uint32_t block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    extern __shared__ __align__(16) char smem[];
    VecT* s_C = (VecT*)smem;

    uint32_t tid = threadIdx.x;
    uint32_t col_mask = (1u << col_bits) - 1;

    // Init Shared
    for (uint32_t i = tid; i < rows_per_block; i += blockDim.x) {
        s_C[i] = VecType<VEC_BITS>::zero();
    }
    __syncthreads();

    uint32_t start_idx = block_ptrs[block_id];
    uint32_t end_idx = block_ptrs[block_id + 1];

    // Stream
    for (uint32_t i = start_idx + tid; i < end_idx; i += blockDim.x) {
        uint32_t packed = coords[i];
        uint32_t r_local = packed >> col_bits;
        uint32_t c_idx = packed & col_mask;

        VecT val = VecType<VEC_BITS>::load(&V[c_idx]);

        if (r_local < rows_per_block) {
            VecType<VEC_BITS>::atomic_xor_shared(&s_C[r_local], val);
        }
    }
    __syncthreads();

    // Writeback
    for (uint32_t i = tid; i < rows_per_block; i += blockDim.x) {
        row_idx_t global_row = row_start_offset + block_id * rows_per_block + i;
        if (global_row < total_rows) {
             VecT current = VecType<VEC_BITS>::load(&C[global_row]);
             VecType<VEC_BITS>::store(&C[global_row], VecType<VEC_BITS>::xor_val(current, s_C[i]));
        }
    }
}

// --- Kernel: Tiled COO (Unrolled 2x) ---
template<int VEC_BITS>
__global__ void kernel_tiled_coo_writeback_unrolled(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ coords,
    const uint32_t* __restrict__ block_ptrs,
    row_idx_t row_start_offset,
    row_idx_t num_blocks,
    row_idx_t total_rows,
    uint32_t rows_per_block,
    int col_bits                  // number of bits for column index in packing
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    uint32_t block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    extern __shared__ __align__(16) char smem[];
    VecT* s_C = (VecT*)smem;

    uint32_t tid = threadIdx.x;
    uint32_t col_mask = (1u << col_bits) - 1;

    for (uint32_t i = tid; i < rows_per_block; i += blockDim.x) {
        s_C[i] = VecType<VEC_BITS>::zero();
    }
    __syncthreads();

    uint32_t start_idx = block_ptrs[block_id];
    uint32_t end_idx = block_ptrs[block_id + 1];

    uint32_t i = start_idx + tid;
    uint32_t stride = blockDim.x * 2;

    for (; i + blockDim.x < end_idx; i += stride) {
        uint32_t p1 = coords[i];
        uint32_t p2 = coords[i + blockDim.x];

        uint32_t r1 = p1 >> col_bits;     uint32_t c1 = p1 & col_mask;
        uint32_t r2 = p2 >> col_bits;     uint32_t c2 = p2 & col_mask;

        VecT v1 = VecType<VEC_BITS>::load(&V[c1]);
        VecT v2 = VecType<VEC_BITS>::load(&V[c2]);

        if (r1 < rows_per_block) VecType<VEC_BITS>::atomic_xor_shared(&s_C[r1], v1);
        if (r2 < rows_per_block) VecType<VEC_BITS>::atomic_xor_shared(&s_C[r2], v2);
    }

    if (i < end_idx) {
        uint32_t p1 = coords[i];
        uint32_t r1 = p1 >> col_bits;     uint32_t c1 = p1 & col_mask;
        VecT v1 = VecType<VEC_BITS>::load(&V[c1]);
        if (r1 < rows_per_block) VecType<VEC_BITS>::atomic_xor_shared(&s_C[r1], v1);
    }

    __syncthreads();

    for (uint32_t k = tid; k < rows_per_block; k += blockDim.x) {
        row_idx_t global_row = row_start_offset + block_id * rows_per_block + k;
        if (global_row < total_rows) {
             VecT current = VecType<VEC_BITS>::load(&C[global_row]);
             VecType<VEC_BITS>::store(&C[global_row], VecType<VEC_BITS>::xor_val(current, s_C[k]));
        }
    }
}

// --- Kernel: Dense Bitslice ---
template<int VEC_BITS>
__global__ void kernel_dense_bitslice(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ A_dense,
    idx_t n_dense_cols,
    row_idx_t n_rows
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t row = (row_idx_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    
    VecT acc = VecType<VEC_BITS>::zero();
    idx_t words_per_row = (n_dense_cols + 31) / 32;
    
    for (idx_t w = 0; w < words_per_row; ++w) {
        uint32_t a_word = A_dense[row * words_per_row + w];
        if (a_word == 0) continue;
        idx_t base_col = w * 32;
        #pragma unroll
        for (int bit = 0; bit < 32; ++bit) {
            if ((a_word >> bit) & 1) {
                idx_t col = base_col + bit;
                if (col < n_dense_cols) acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[col]));
            }
        }
    }
    VecT current = VecType<VEC_BITS>::load(&C[row]);
    VecType<VEC_BITS>::store(&C[row], VecType<VEC_BITS>::xor_val(current, acc));
}

// --- Kernel: Heavy (Warp CSR) ---
template<int VEC_BITS>
__launch_bounds__(256)
__global__ void kernel_heavy_warp_csr(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ indices,
    row_idx_t n_heavy_rows
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    int warp_id = threadIdx.x >> 5; 
    int lane_id = threadIdx.x & 31; 
    row_idx_t row = blockIdx.x * 8 + warp_id;
    if (row >= n_heavy_rows) return;
    
    uint32_t start = offsets[row];
    uint32_t end = offsets[row + 1];
    
    VecT my_acc = VecType<VEC_BITS>::zero();
    uint32_t i = start + lane_id;
    
    // Unroll manually
    while (i + 96 < end) {
        idx_t c0 = indices[i]; idx_t c1 = indices[i+32]; idx_t c2 = indices[i+64]; idx_t c3 = indices[i+96];
        my_acc = VecType<VEC_BITS>::xor_val(my_acc, VecType<VEC_BITS>::xor_val(VecType<VEC_BITS>::load(&V[c0]), VecType<VEC_BITS>::load(&V[c1])));
        my_acc = VecType<VEC_BITS>::xor_val(my_acc, VecType<VEC_BITS>::xor_val(VecType<VEC_BITS>::load(&V[c2]), VecType<VEC_BITS>::load(&V[c3])));
        i += 128;
    }
    for (; i < end; i += 32) my_acc = VecType<VEC_BITS>::xor_val(my_acc, VecType<VEC_BITS>::load(&V[indices[i]]));
    
    // Warp Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        VecT other = VecType<VEC_BITS>::shfl_down(my_acc, offset);
        my_acc = VecType<VEC_BITS>::xor_val(my_acc, other);
    }
    
    if (lane_id == 0) {
        VecT current = VecType<VEC_BITS>::load(&C[row]);
        VecType<VEC_BITS>::store(&C[row], VecType<VEC_BITS>::xor_val(current, my_acc));
    }
}

// --- Kernel: Sparse Thread-per-Row CSR (optimized: __ldg + 4× unroll) ---
// Thread-per-row design: each thread processes one complete row.  Optimal for
// ultra-sparse rows (weight 1–30) where a warp-per-row approach wastes 31/32
// threads.  Column indices are direct (no serial delta decoding), allowing the
// GPU to pipeline V[] loads independently.
//
// Optimizations over the naive version:
//   1. __ldg() on col_ind[] and V[] — routes through read-only texture cache
//   2. 4× unrolled inner loop with individual __ldg index loads
//   3. Pair-wise XOR reduces dependency chain depth from 4 to 2
template<int VEC_BITS>
__launch_bounds__(256)
__global__ void kernel_sparse_warp_csr(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ row_ptr,    // CSR row pointers (zero-based)
    const uint32_t* __restrict__ col_ind,    // CSR column indices
    row_idx_t start_row,                     // global row offset for C indexing
    row_idx_t n_rows                         // number of rows in this segment
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t row = (row_idx_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t start = __ldg(&row_ptr[row]);
    uint32_t end   = __ldg(&row_ptr[row + 1]);
    uint32_t nnz   = end - start;

    VecT acc = VecType<VEC_BITS>::zero();

    // --- 4× unrolled path: process 4 NNZ per iteration ---
    uint32_t i = start;
    uint32_t unrolled_end = start + (nnz & ~3u);  // round down to multiple of 4

    for (; i < unrolled_end; i += 4) {
        // Load 4 column indices via __ldg for read-only cache hint
        uint32_t c0 = __ldg(&col_ind[i]);
        uint32_t c1 = __ldg(&col_ind[i + 1]);
        uint32_t c2 = __ldg(&col_ind[i + 2]);
        uint32_t c3 = __ldg(&col_ind[i + 3]);

        // Load V[] elements and XOR in pairs to reduce dependency chain depth
        VecT v0 = VecType<VEC_BITS>::load(&V[c0]);
        VecT v1 = VecType<VEC_BITS>::load(&V[c1]);
        VecT v2 = VecType<VEC_BITS>::load(&V[c2]);
        VecT v3 = VecType<VEC_BITS>::load(&V[c3]);

        VecT pair01 = VecType<VEC_BITS>::xor_val(v0, v1);
        VecT pair23 = VecType<VEC_BITS>::xor_val(v2, v3);
        acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::xor_val(pair01, pair23));
    }

    // --- Scalar tail: remaining 0–3 elements ---
    for (; i < end; i++) {
        acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[__ldg(&col_ind[i])]));
    }

    // Writeback: XOR accumulator into output
    row_idx_t global_row = start_row + row;
    VecT current = VecType<VEC_BITS>::load(&C[global_row]);
    VecType<VEC_BITS>::store(&C[global_row], VecType<VEC_BITS>::xor_val(current, acc));
}

// --- Kernel: Heavy (Bitmap) ---
template<int VEC_BITS>
__global__ void kernel_heavy_bitmap(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ heavy_bitmaps,
    idx_t n_sparse_cols,
    idx_t dense_offset
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t row = blockIdx.x;
    VecT acc = VecType<VEC_BITS>::zero();
    idx_t words_per_row = (n_sparse_cols + 31) / 32;
    idx_t tid = threadIdx.x;
    idx_t stride = blockDim.x;
    
    // Accumulate
    for (idx_t w = tid; w < words_per_row; w += stride) {
        uint32_t mask = heavy_bitmaps[row * words_per_row + w];
        if (mask == 0) continue;
        idx_t base_col = dense_offset + w * 32;
        #pragma unroll
        for (int bit = 0; bit < 32; ++bit) {
            if ((mask >> bit) & 1) acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[base_col + bit]));
        }
    }
    
    extern __shared__ __align__(16) char smem[];
    VecT* s_acc = (VecT*)smem;
    s_acc[tid] = acc;
    __syncthreads();
    
    // Block Reduction with Volatile Safety
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Use volatile pointer to force read from shared memory
            volatile VecT* v_ptr = (volatile VecT*)s_acc;
            
            // Note: Direct assignment from volatile struct to non-volatile triggers read.
            // We read 'mine' and 'other' carefully.
            // Since VecT is a struct, C++ doesn't support 'volatile struct' assignment well directly in all compilers.
            // We use a safe cast strategy:
            VecT mine = *(VecT*)&v_ptr[tid];
            VecT other = *(VecT*)&v_ptr[tid + s];
            
            s_acc[tid] = VecType<VEC_BITS>::xor_val(mine, other);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        VecT current = VecType<VEC_BITS>::load(&C[row]);
        VecType<VEC_BITS>::store(&C[row], VecType<VEC_BITS>::xor_val(current, s_acc[0]));
    }
}

// --- Kernel: Sparse Golomb ---
template<int VEC_BITS>
__global__ void kernel_sparse_golomb(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ rem_stream,
    const uint32_t* __restrict__ quo_stream,
    const uint32_t* __restrict__ row_rem_offsets,
    const uint32_t* __restrict__ row_quo_offsets,
    const uint32_t* __restrict__ warp_params, 
    idx_t dense_offset,
    row_idx_t total_rows,     
    row_idx_t active_row_count, 
    row_idx_t row_start_offset
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t local_idx = (row_idx_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= active_row_count) return;
    
    row_idx_t row = local_idx + row_start_offset;
    int k = warp_params[row >> 5]; 
    uint32_t r_idx = row_rem_offsets[row];
    uint32_t r_end = row_rem_offsets[row + 1]; 
    uint32_t q_bit_idx = row_quo_offsets[row];
    
    idx_t prev_col = dense_offset - 1;
    VecT acc = VecType<VEC_BITS>::load(&C[row]);
    
    for (uint32_t i = r_idx; i < r_end; ++i) {
        uint32_t remainder = rem_stream[i]; 
        uint32_t quotient = 0;
        while (true) {
            uint32_t word_idx = q_bit_idx >> 5;
            uint32_t bit_offset = q_bit_idx & 31;
            uint32_t word = quo_stream[word_idx];
            bool bit = (word >> bit_offset) & 1;
            q_bit_idx++; 
            if (bit) quotient++; else break;
        }
        idx_t delta = (quotient << k) + remainder + 1;
        idx_t col = prev_col + delta;
        prev_col = col;
        acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[col]));
    }
    VecType<VEC_BITS>::store(&C[row], acc);
}

// --- Kernel: Sparse Delta-16 ---
// n_cols gates escape handling: when n_cols <= 65535 no escape pairs exist in
// the stream, so the check is skipped entirely (uniform branch, zero cost).
template<int VEC_BITS>
__global__ void kernel_sparse_delta_16(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint16_t* __restrict__ stream,
    const uint32_t* __restrict__ offsets,
    idx_t dense_offset,
    row_idx_t active_rows,
    row_idx_t row_start_offset,
    idx_t n_cols
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= active_rows) return;

    row_idx_t row = local_row + row_start_offset;
    uint32_t start = offsets[row];
    uint32_t end = offsets[row + 1];

    idx_t col = dense_offset - 1;
    VecT acc = VecType<VEC_BITS>::load(&C[row]);

    for (uint32_t i = start; i < end; ++i) {
        uint16_t delta = stream[i];
        col += delta;
        // Escape sequence: (0xFFFF, 0x0000) = "advance col by 0xFFFF, no data".
        // Uniform branch on n_cols — compiler hoists comparison out of loop.
        if (n_cols > 65535 && delta == 0xFFFF && i + 1 < end && stream[i + 1] == 0x0000) {
            ++i;  // consume the 0x0000 marker
            continue;
        }
        if (delta != 0) {
            acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[col]));
        }
    }
    VecType<VEC_BITS>::store(&C[row], acc);
}

// --- Kernel: Sparse PForDelta ---
template<int VEC_BITS>
__global__ void kernel_sparse_pfor(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint8_t*  __restrict__ data,
    const uint16_t* __restrict__ meta,
    const uint32_t* __restrict__ block_byte_offsets,
    const uint32_t* __restrict__ row_block_starts,
    idx_t dense_offset,
    row_idx_t active_rows,
    row_idx_t row_start_offset
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    row_idx_t local_row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (local_row >= active_rows) return;
    
    uint32_t lane_id = threadIdx.x & 31;
    row_idx_t row = local_row + row_start_offset;

    uint32_t blk_start = row_block_starts[row];
    uint32_t blk_end   = row_block_starts[row + 1];

    idx_t current_col = dense_offset - 1;
    VecT acc = VecType<VEC_BITS>::zero();

    for (uint32_t b = blk_start; b < blk_end; ++b) {
        uint16_t m = 0;
        uint32_t data_offset = 0;
        if (lane_id == 0) {
            m = meta[b];
            data_offset = block_byte_offsets[b];
        }
        m = __shfl_sync(0xFFFFFFFF, m, 0);
        data_offset = __shfl_sync(0xFFFFFFFF, data_offset, 0);

        uint32_t count = m >> 8;
        uint32_t type  = m & 0x3;
        const uint8_t* ptr = data + data_offset;

        uint32_t total_delta = 0;

        for (uint32_t i = 0; i < count; i += 32) {
            uint32_t idx = i + lane_id;
            uint32_t delta = 0;

            if (idx < count) {
                if (type == 0) delta = (uint32_t)ptr[idx];
                else if (type == 1) delta = (uint32_t)((const uint16_t*)ptr)[idx];
                else delta = ((const uint32_t*)ptr)[idx];
            }

            uint32_t scan_delta = delta;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                uint32_t n = __shfl_up_sync(0xFFFFFFFF, scan_delta, offset);
                if (lane_id >= offset) scan_delta += n;
            }

            if (idx < count) {
                idx_t col = current_col + total_delta + scan_delta;
                acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[col]));
            }

            uint32_t chunk_sum = __shfl_sync(0xFFFFFFFF, scan_delta, 31);
            total_delta += chunk_sum;
        }
        current_col += total_delta;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        VecT other = VecType<VEC_BITS>::shfl_down(acc, offset);
        acc = VecType<VEC_BITS>::xor_val(acc, other);
    }

    if (lane_id == 0) {
        VecT current = VecType<VEC_BITS>::load(&C[row]);
        VecType<VEC_BITS>::store(&C[row], VecType<VEC_BITS>::xor_val(current, acc));
    }
}

// --- Kernel: Bit-Exact PForDelta ---
template<int VEC_BITS>
__global__ void kernel_sparse_pfor_bit_exact(
    void* __restrict__ C_ptr,
    const void* __restrict__ V_ptr,
    const uint32_t* __restrict__ data,
    const uint32_t* __restrict__ meta_bits,
    const uint32_t* __restrict__ meta_ex_start,
    const uint32_t* __restrict__ meta_data_start,
    const uint32_t* __restrict__ exceptions,
    const uint8_t*  __restrict__ exception_idx,
    const uint32_t* __restrict__ row_block_starts,
    idx_t dense_offset,
    row_idx_t active_rows,
    row_idx_t row_start_offset
) {
    using VecT = typename VecType<VEC_BITS>::Type;
    VecT* C = (VecT*)C_ptr;
    const VecT* V = (const VecT*)V_ptr;

    extern __shared__ uint32_t smem_deltas[];
    
    row_idx_t local_row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (local_row >= active_rows) return;
    
    uint32_t lane_id = threadIdx.x & 31;
    uint32_t warp_id = threadIdx.x >> 5; 
    row_idx_t row = local_row + row_start_offset;
    
    volatile uint32_t* my_deltas = smem_deltas + warp_id * 128;

    uint32_t blk_start = row_block_starts[row];
    uint32_t blk_end   = row_block_starts[row + 1];

    idx_t current_col = dense_offset - 1;
    VecT acc = VecType<VEC_BITS>::zero();

    for (uint32_t b_idx = blk_start; b_idx < blk_end; ++b_idx) {
        uint32_t m_bits = meta_bits[b_idx];
        uint32_t d_start = meta_data_start[b_idx];
        
        uint32_t b = m_bits & 0xFF;
        uint32_t ex_count = (m_bits >> 8) & 0xFF;
        uint32_t count = m_bits >> 16;
        
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t i = chunk * 32 + lane_id;
            uint32_t bit_pos = i * b;
            uint32_t word_idx = bit_pos >> 5;
            uint32_t bit_rem = bit_pos & 31;
            
            uint32_t w0 = data[d_start + word_idx];
            uint32_t w1 = data[d_start + word_idx + 1]; 
            
            uint32_t val = __funnelshift_r(w0, w1, bit_rem);
            val &= (b == 32) ? 0xFFFFFFFF : ((1 << b) - 1);
            
            my_deltas[i] = val;
        }
        
        uint32_t ex_start = meta_ex_start[b_idx];
        for (uint32_t i = lane_id; i < ex_count; i += 32) {
            uint8_t idx = exception_idx[ex_start + i];
            uint32_t val = exceptions[ex_start + i];
            my_deltas[idx] = val;
        }
        __syncwarp();

        uint32_t total_delta = 0;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint32_t i = chunk * 32 + lane_id;
            uint32_t delta = (i < count) ? my_deltas[i] : 0;
            
            uint32_t scan_delta = delta;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                uint32_t n = __shfl_up_sync(0xFFFFFFFF, scan_delta, offset);
                if (lane_id >= offset) scan_delta += n;
            }
            
            if (i < count) {
                idx_t col = current_col + total_delta + scan_delta;
                acc = VecType<VEC_BITS>::xor_val(acc, VecType<VEC_BITS>::load(&V[col]));
            }
            
            uint32_t chunk_sum = __shfl_sync(0xFFFFFFFF, scan_delta, 31);
            total_delta += chunk_sum;
        }
        current_col += total_delta;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        VecT other = VecType<VEC_BITS>::shfl_down(acc, offset);
        acc = VecType<VEC_BITS>::xor_val(acc, other);
    }

    if (lane_id == 0) {
        VecT current = VecType<VEC_BITS>::load(&C[row]);
        VecType<VEC_BITS>::store(&C[row], VecType<VEC_BITS>::xor_val(current, acc));
    }
}

// =================================================================================
// 4. Host Functions & Dispatch
// =================================================================================

// Helper to clean up cache flush code
static uint64_t* d_flush_buffer = nullptr;
static size_t flush_size_bytes = 0;

__global__ void kernel_flush_l2(uint64_t* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < size; i += stride) {
        data[i] += 1; 
    }
}

void SpMMKernels::flush_l2_cache() {
    if (!d_flush_buffer) {
        int device;
        cudaGetDevice(&device);
        int l2_size;
        cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
        // 2x L2 size ensures full cache flush; clamp to [2 MB, 128 MB]
        flush_size_bytes = static_cast<size_t>(std::clamp(
            l2_size * 2,
            2 * 1024 * 1024,        // 2 MB floor (prevents zero-size on query error)
            128 * 1024 * 1024        // 128 MB cap (preserves RTX behavior)
        ));
        CUDA_CHECK(cudaMalloc(&d_flush_buffer, flush_size_bytes));
        CUDA_CHECK(cudaMemset(d_flush_buffer, 0, flush_size_bytes));
    }
    size_t num_elements = flush_size_bytes / sizeof(uint64_t);
    int threads = 256;
    int blocks = static_cast<int>((num_elements + threads - 1) / threads);
    kernel_flush_l2<<<blocks, threads>>>(d_flush_buffer, num_elements);
    cudaDeviceSynchronize();
}

void SpMMKernels::cleanup_l2_cache() {
    if (d_flush_buffer) {
        cudaFree(d_flush_buffer);
        d_flush_buffer = nullptr;
    }
}

DeviceMatrix SpMMKernels::allocate_and_copy(const CompressedMatrix& host_data) {
    DeviceMatrix d_mat;
    d_mat.n_dense_cols = host_data.n_dense_cols;
    d_mat.n_rows = host_data.permutation.size();
    d_mat.n_heavy_rows = host_data.n_heavy_rows;
    d_mat.n_sparse_cols = host_data.n_sparse_cols;

    d_mat.tiled_row_block_size = host_data.tiled_row_block_size;

    auto safe_copy = [](void** dest, const void* src, size_t bytes) {
        if (bytes > 0) {
            CUDA_CHECK(cudaMalloc(dest, bytes));
            CUDA_CHECK(cudaMemcpy(*dest, src, bytes, cudaMemcpyHostToDevice));
        }
    };

    safe_copy((void**)&d_mat.d_dense_values, host_data.dense_values.data(), host_data.dense_values.size() * 4);
    safe_copy((void**)&d_mat.d_heavy_csr_offsets, host_data.heavy_csr_offsets.data(), host_data.heavy_csr_offsets.size() * 4);
    safe_copy((void**)&d_mat.d_heavy_csr_indices, host_data.heavy_csr_indices.data(), host_data.heavy_csr_indices.size() * 4);
    safe_copy((void**)&d_mat.d_heavy_tiled_coords, host_data.heavy_tiled_coords.data(), host_data.heavy_tiled_coords.size() * 4);
    safe_copy((void**)&d_mat.d_heavy_tiled_ptr, host_data.heavy_tiled_ptr.data(), host_data.heavy_tiled_ptr.size() * 4);
    d_mat.heavy_tiled_size = host_data.heavy_tiled_ptr.empty() ? 0 : host_data.heavy_tiled_ptr.size() - 1;
    
    safe_copy((void**)&d_mat.d_heavy_bitmaps, host_data.heavy_bitmaps.data(), host_data.heavy_bitmaps.size() * 4);

    safe_copy((void**)&d_mat.d_stream_remainders, host_data.stream_remainders.data(), host_data.stream_remainders.size() * 4);
    safe_copy((void**)&d_mat.d_stream_quotients, host_data.stream_quotients.data(), host_data.stream_quotients.size() * 4);
    safe_copy((void**)&d_mat.d_warp_rice_params, host_data.warp_rice_params.data(), host_data.warp_rice_params.size() * 4);
    safe_copy((void**)&d_mat.d_row_offsets_rem, host_data.row_rem_offsets.data(), host_data.row_rem_offsets.size() * 4);
    safe_copy((void**)&d_mat.d_row_offsets_quo, host_data.row_quo_offsets.data(), host_data.row_quo_offsets.size() * 4);

    safe_copy((void**)&d_mat.d_sparse_tiled_coords, host_data.sparse_tiled_coords.data(), host_data.sparse_tiled_coords.size() * 4);
    safe_copy((void**)&d_mat.d_sparse_tiled_ptr, host_data.sparse_tiled_ptr.data(), host_data.sparse_tiled_ptr.size() * 4);
    d_mat.sparse_tiled_size = host_data.sparse_tiled_ptr.empty() ? 0 : host_data.sparse_tiled_ptr.size() - 1;

    safe_copy((void**)&d_mat.d_delta_16_stream, host_data.delta_16_stream.data(), host_data.delta_16_stream.size() * 2);
    safe_copy((void**)&d_mat.d_delta_16_offsets, host_data.delta_16_offsets.data(), host_data.delta_16_offsets.size() * 4);

    safe_copy((void**)&d_mat.d_pfor_data, host_data.pfor_data.data(), host_data.pfor_data.size());
    safe_copy((void**)&d_mat.d_pfor_meta, host_data.pfor_meta.data(), host_data.pfor_meta.size() * 2);
    safe_copy((void**)&d_mat.d_pfor_block_byte_offsets, host_data.pfor_block_byte_offsets.data(), host_data.pfor_block_byte_offsets.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_row_block_starts, host_data.pfor_row_block_starts.data(), host_data.pfor_row_block_starts.size() * 4);

    safe_copy((void**)&d_mat.d_pfor_be_data, host_data.pfor_be_data.data(), host_data.pfor_be_data.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_be_meta_bits, host_data.pfor_be_meta_bits.data(), host_data.pfor_be_meta_bits.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_be_meta_ex_start, host_data.pfor_be_meta_ex_start.data(), host_data.pfor_be_meta_ex_start.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_be_meta_data_start, host_data.pfor_be_meta_data_start.data(), host_data.pfor_be_meta_data_start.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_be_exceptions, host_data.pfor_be_exceptions.data(), host_data.pfor_be_exceptions.size() * 4);
    safe_copy((void**)&d_mat.d_pfor_be_exception_idx, host_data.pfor_be_exception_idx.data(), host_data.pfor_be_exception_idx.size());
    safe_copy((void**)&d_mat.d_pfor_be_row_block_starts, host_data.pfor_be_row_block_starts.data(), host_data.pfor_be_row_block_starts.size() * 4);

    return d_mat;
}

void SpMMKernels::free_matrix(DeviceMatrix& mat) {
    // Helper macro to free if not null
    #define FREE_SAFE(ptr) if(ptr) cudaFree(ptr)
    FREE_SAFE(mat.d_dense_values);
    FREE_SAFE(mat.d_heavy_bitmaps);
    FREE_SAFE(mat.d_heavy_csr_offsets);
    FREE_SAFE(mat.d_heavy_csr_indices);
    FREE_SAFE(mat.d_heavy_tiled_coords);
    FREE_SAFE(mat.d_heavy_tiled_ptr);
    FREE_SAFE(mat.d_stream_remainders);
    FREE_SAFE(mat.d_stream_quotients);
    FREE_SAFE(mat.d_warp_rice_params);
    FREE_SAFE(mat.d_row_offsets_rem);
    FREE_SAFE(mat.d_row_offsets_quo);
    FREE_SAFE(mat.d_sparse_tiled_coords);
    FREE_SAFE(mat.d_sparse_tiled_ptr);
    FREE_SAFE(mat.d_delta_16_stream);
    FREE_SAFE(mat.d_delta_16_offsets);
    FREE_SAFE(mat.d_pfor_data);
    FREE_SAFE(mat.d_pfor_meta);
    FREE_SAFE(mat.d_pfor_block_byte_offsets);
    FREE_SAFE(mat.d_pfor_row_block_starts);
    FREE_SAFE(mat.d_pfor_be_data);
    FREE_SAFE(mat.d_pfor_be_meta_bits);
    FREE_SAFE(mat.d_pfor_be_meta_ex_start);
    FREE_SAFE(mat.d_pfor_be_meta_data_start);
    FREE_SAFE(mat.d_pfor_be_exceptions);
    FREE_SAFE(mat.d_pfor_be_exception_idx);
    FREE_SAFE(mat.d_pfor_be_row_block_starts);
    if (mat.warp_csr_row_ptr) { cudaFree(const_cast<uint32_t*>(mat.warp_csr_row_ptr)); mat.warp_csr_row_ptr = nullptr; }
    if (mat.warp_csr_col_ind) { cudaFree(const_cast<uint32_t*>(mat.warp_csr_col_ind)); mat.warp_csr_col_ind = nullptr; }
    mat = {};
}

// --- Implementation Helper (Replaces Lambda) ---
template<int BITS>
void run_spmm_impl(const DeviceMatrix& mat, void* d_C_void, const void* d_V_void, SpMMConfig config, cudaStream_t stream) {
    using VecT = typename VecType<BITS>::Type;
    VecT* d_C = (VecT*)d_C_void;
    const VecT* d_V = (const VecT*)d_V_void;
    dim3 block_256(256);

    // 1. Dense Bitslice
    if (config.enable_dense_bitslice && mat.d_dense_values) {
        dim3 grid((mat.n_rows + 255) / 256);
        kernel_dense_bitslice<BITS><<<grid, block_256, 0, stream>>>(
            (void*)d_C, (const void*)d_V, mat.d_dense_values, mat.n_dense_cols, mat.n_rows);
    }

    // 2. Heavy Rows
    if (mat.n_heavy_rows > 0) {
        if (config.enable_heavy_tiled_coo && mat.d_heavy_tiled_coords) {
             uint32_t rows_per_blk = mat.tiled_row_block_size;
             size_t shmem = rows_per_blk * sizeof(VecT);
             dim3 grid(mat.heavy_tiled_size);
             kernel_tiled_coo_writeback<BITS><<<grid, block_256, shmem, stream>>>(
                 (void*)d_C, (const void*)d_V, mat.d_heavy_tiled_coords, mat.d_heavy_tiled_ptr, 0, mat.heavy_tiled_size, mat.n_rows, rows_per_blk, mat.tiled_col_bits);
        }
        else if (config.enable_heavy_warp_csr && mat.d_heavy_csr_offsets) {
             dim3 grid((mat.n_heavy_rows + 7) / 8);
             kernel_heavy_warp_csr<BITS><<<grid, block_256, 0, stream>>>(
                 (void*)d_C, (const void*)d_V, mat.d_heavy_csr_offsets, mat.d_heavy_csr_indices, mat.n_heavy_rows);
        }
        else if (config.enable_heavy_bitmap && mat.d_heavy_bitmaps) {
            dim3 grid(mat.n_heavy_rows);
            size_t shmem = 256 * sizeof(VecT); 
            kernel_heavy_bitmap<BITS><<<grid, block_256, shmem, stream>>>(
                (void*)d_C, (const void*)d_V, mat.d_heavy_bitmaps, mat.n_sparse_cols, mat.n_dense_cols);
        }
    }

    // 3. Sparse Rows
    if (config.enable_sparse) {
        row_idx_t sparse_rows_count = mat.n_rows - mat.n_heavy_rows;
        if (sparse_rows_count > 0) {
            if (config.enable_sparse_warp_csr && mat.warp_csr_row_ptr) {
                dim3 grid((mat.warp_csr_n_rows + 255) / 256);
                kernel_sparse_warp_csr<BITS><<<grid, block_256, 0, stream>>>(
                    (void*)d_C, (const void*)d_V,
                    mat.warp_csr_row_ptr, mat.warp_csr_col_ind,
                    mat.warp_csr_start_row, mat.warp_csr_n_rows);
            }
            else if (config.enable_sparse_tiled_coo_unrolled && mat.d_sparse_tiled_coords) {
                 uint32_t rows_per_blk = mat.tiled_row_block_size;
                 size_t shmem = rows_per_blk * sizeof(VecT);
                 dim3 grid(mat.sparse_tiled_size);
                 kernel_tiled_coo_writeback_unrolled<BITS><<<grid, block_256, shmem, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_sparse_tiled_coords, mat.d_sparse_tiled_ptr, mat.n_heavy_rows, mat.sparse_tiled_size, mat.n_rows, rows_per_blk, mat.tiled_col_bits);
            }
            else if (config.enable_sparse_tiled_coo && mat.d_sparse_tiled_coords) {
                 uint32_t rows_per_blk = mat.tiled_row_block_size;
                 size_t shmem = rows_per_blk * sizeof(VecT);
                 dim3 grid(mat.sparse_tiled_size);
                 kernel_tiled_coo_writeback<BITS><<<grid, block_256, shmem, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_sparse_tiled_coords, mat.d_sparse_tiled_ptr, mat.n_heavy_rows, mat.sparse_tiled_size, mat.n_rows, rows_per_blk, mat.tiled_col_bits);
            }
            else if (config.enable_sparse_pfor && mat.d_pfor_data) {
                 dim3 grid((sparse_rows_count + 7) / 8);
                 kernel_sparse_pfor<BITS><<<grid, block_256, 0, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_pfor_data, mat.d_pfor_meta, mat.d_pfor_block_byte_offsets, mat.d_pfor_row_block_starts, mat.n_dense_cols, sparse_rows_count, mat.n_heavy_rows);
            }
            else if (config.enable_sparse_pfor_bit_exact && mat.d_pfor_be_data) {
                 dim3 grid((sparse_rows_count + 7) / 8);
                 size_t smem = 256 * 4 * 4; 
                 kernel_sparse_pfor_bit_exact<BITS><<<grid, block_256, smem, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_pfor_be_data, mat.d_pfor_be_meta_bits, mat.d_pfor_be_meta_ex_start, mat.d_pfor_be_meta_data_start, mat.d_pfor_be_exceptions, mat.d_pfor_be_exception_idx, mat.d_pfor_be_row_block_starts, mat.n_dense_cols, sparse_rows_count, mat.n_heavy_rows);
            }
            else if (config.enable_sparse_delta_16 && mat.d_delta_16_stream) {
                 dim3 grid((sparse_rows_count + 255) / 256);
                 kernel_sparse_delta_16<BITS><<<grid, block_256, 0, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_delta_16_stream, mat.d_delta_16_offsets, mat.n_dense_cols, sparse_rows_count, mat.n_heavy_rows, mat.n_sparse_cols);
            }
            else if (mat.d_stream_remainders != nullptr) {
                 dim3 grid((sparse_rows_count + 255) / 256);
                 kernel_sparse_golomb<BITS><<<grid, block_256, 0, stream>>>(
                     (void*)d_C, (const void*)d_V, mat.d_stream_remainders, mat.d_stream_quotients, mat.d_row_offsets_rem, mat.d_row_offsets_quo, mat.d_warp_rice_params, mat.n_dense_cols, mat.n_rows, sparse_rows_count, mat.n_heavy_rows);
            }
        }
    }
}

void SpMMKernels::run_spmm(const DeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream) {
    if (mat.n_rows == 0) return;

    // Use standard switch instead of templated lambda
    switch (config.vector_width_bits) {
        case 32:  run_spmm_impl<32>(mat, d_C, d_V, config, stream); break;
        case 64:  run_spmm_impl<64>(mat, d_C, d_V, config, stream); break;
        case 128: run_spmm_impl<128>(mat, d_C, d_V, config, stream); break;
        case 256: run_spmm_impl<256>(mat, d_C, d_V, config, stream); break;
        case 512: run_spmm_impl<512>(mat, d_C, d_V, config, stream); break;
        default:  throw std::runtime_error("Unsupported vector width");
    }
}

// Slice wrappers remain largely similar, just forwarding to run_spmm
SlicedDeviceMatrix SpMMKernels::allocate_sliced(const std::vector<CompressedMatrix>& host_slices) {
    SlicedDeviceMatrix sdm;
    for(const auto& slice : host_slices) {
        sdm.slices.push_back(allocate_and_copy(slice));
    }
    return sdm;
}

void SpMMKernels::free_sliced(SlicedDeviceMatrix& mat) {
    for(auto& s : mat.slices) SpMMKernels::free_matrix(s);
    mat.slices.clear();
}

void SpMMKernels::run_spmm_sliced(const SlicedDeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream) {
    for (const auto& slice_mat : mat.slices) {
        run_spmm(slice_mat, d_C, d_V, config, stream);
    }
}

void SpMMKernels::run_hybrid(const HybridDeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream) {
    launch_m4rm_full(mat.m4rm_ctx, d_V, d_C, config.vector_width_bits, stream); 
    
    size_t row_size_bytes = (config.vector_width_bits / 8);
    size_t m4rm_offset_bytes = mat.m4rm_ctx.num_dense_rows * row_size_bytes;
    void* d_C_sparse = (void*)((uint8_t*)d_C + m4rm_offset_bytes);
    
    run_spmm(mat.sparse_mat, d_C_sparse, d_V, config, stream);
}

// =================================================================================
// GPU Preprocessing Implementations
// =================================================================================

// 1. Helper Kernels (Must be defined at file scope)

__global__ void k_calc_row_lengths(const uint32_t* offsets, uint32_t* lengths, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        lengths[idx] = offsets[idx+1] - offsets[idx];
    }
}

// Templated Gather Kernel: Supports 32, 64, 128, 256, 512 bits
template<int BITS>
__global__ void k_permute_gather_tmpl(const void* src, void* dst, const uint32_t* map, size_t n) {
    using VecT = typename VecType<BITS>::Type;
    const VecT* s = (const VecT*)src;
    VecT* d = (VecT*)dst;
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = s[map[idx]];
    }
}

// Expands row pointers into a coordinate list of row indices.
__global__ void k_expand_rows_global(const uint32_t* row_ptr, uint32_t* out_rows, size_t num_rows) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_rows) {
        uint32_t start = row_ptr[r];
        uint32_t end = row_ptr[r+1];
        for (uint32_t k = start; k < end; ++k) {
            out_rows[k] = (uint32_t)r;
        }
    }
}

// Computes histogram of row indices.
__global__ void k_histo_global(const uint32_t* rows, uint32_t* counts, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&counts[rows[i] + 1], 1);
    }
}

// 2. Host Wrapper Implementations

uint32_t* gpu_compute_density_permutation(const uint32_t* d_row_offsets, size_t n_rows, bool ascending) {
    uint32_t* d_perm;
    uint32_t* d_lengths;
    CUDA_CHECK(cudaMalloc(&d_perm, n_rows * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lengths, n_rows * sizeof(uint32_t)));

    dim3 block(256);
    dim3 grid((n_rows + 255) / 256);
    k_calc_row_lengths<<<grid, block>>>(d_row_offsets, d_lengths, n_rows);

    thrust::device_ptr<uint32_t> p_perm(d_perm);
    thrust::device_ptr<uint32_t> p_lengths(d_lengths);
    thrust::sequence(p_perm, p_perm + n_rows);

    if (ascending) {
        thrust::sort_by_key(p_lengths, p_lengths + n_rows, p_perm);
    } else {
        thrust::sort_by_key(p_lengths, p_lengths + n_rows, p_perm, thrust::greater<uint32_t>());
    }

    CUDA_CHECK(cudaFree(d_lengths));
    return d_perm;
}

// Correct Generic Implementation
void gpu_permute_vector(const void* d_src, void* d_dst, const uint32_t* d_map, size_t num_elements, int bit_width, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    
    switch (bit_width) {
        case 32:  k_permute_gather_tmpl<32><<<grid, block, 0, stream>>>(d_src, d_dst, d_map, num_elements); break;
        case 64:  k_permute_gather_tmpl<64><<<grid, block, 0, stream>>>(d_src, d_dst, d_map, num_elements); break;
        case 128: k_permute_gather_tmpl<128><<<grid, block, 0, stream>>>(d_src, d_dst, d_map, num_elements); break;
        case 256: k_permute_gather_tmpl<256><<<grid, block, 0, stream>>>(d_src, d_dst, d_map, num_elements); break;
        case 512: k_permute_gather_tmpl<512><<<grid, block, 0, stream>>>(d_src, d_dst, d_map, num_elements); break;
        default: throw std::runtime_error("Unsupported bit width for permutation");
    }
}

void gpu_transpose_csr(
    size_t n_rows, size_t n_cols, size_t nnz,
    const uint32_t* d_csr_row_ptr, const uint32_t* d_csr_col_ind,
    uint32_t** d_out_row_ptr, uint32_t** d_out_col_ind
) {
    uint32_t* d_new_rows;
    uint32_t* d_new_cols;
    CUDA_CHECK(cudaMalloc(&d_new_rows, nnz * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_new_cols, nnz * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_new_rows, d_csr_col_ind, nnz * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    k_expand_rows_global<<<(n_rows + 255)/256, 256>>>(d_csr_row_ptr, d_new_cols, n_rows);

    thrust::device_ptr<uint32_t> p_keys(d_new_rows);
    thrust::device_ptr<uint32_t> p_vals(d_new_cols);
    thrust::sort_by_key(p_keys, p_keys + nnz, p_vals);

    uint32_t* d_at_ptr;
    CUDA_CHECK(cudaMalloc(&d_at_ptr, (n_cols + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_at_ptr, 0, (n_cols + 1) * sizeof(uint32_t)));
    
    k_histo_global<<<(nnz + 255)/256, 256>>>(d_new_rows, d_at_ptr, nnz);
    
    thrust::device_ptr<uint32_t> p_ptr(d_at_ptr);
    thrust::inclusive_scan(p_ptr, p_ptr + n_cols + 1, p_ptr);

    *d_out_row_ptr = d_at_ptr;
    *d_out_col_ind = d_new_cols; 
    
    CUDA_CHECK(cudaFree(d_new_rows));
}
