// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage2/basecase_ops.h"
#include <cstdio>

namespace lingen {
namespace stage2 {
namespace ops {

// =================================================================================
// KERNELS
// =================================================================================

template<int N, int M>
__global__ void k_compute_discrepancy(
    const uint64_t* __restrict__ S_data, 
    int S_len,
    const uint64_t* __restrict__ Pi_data, 
    int Pi_len,
    uint64_t* __restrict__ Delta, 
    int t
) {
    int col_idx = blockIdx.x; 
    if (col_idx >= M) return;
    if (threadIdx.x > 0) return; 

    uint64_t acc = 0;
    
    for (int k = 0; k <= t; ++k) {
        if (k >= Pi_len) break;
        int s_idx = t - k;
        if (s_idx >= S_len) continue;

        const uint64_t* pi_mat = Pi_data + (size_t)k * (size_t(M)*M/64);
        
        uint64_t pi_col_vec = 0;
        int word_idx = col_idx / 64;
        int bit_idx  = col_idx % 64;

        for (int r = 0; r < N; ++r) { 
            uint64_t row_word = pi_mat[r * (M/64) + word_idx];
            if ((row_word >> bit_idx) & 1) {
                pi_col_vec |= (1ULL << r);
            }
        }

        if (pi_col_vec != 0) {
            const uint64_t* s_mat = S_data + (size_t)s_idx * (size_t(N)*N/64);
            uint64_t res_word = 0;
            
            if (N == 64) {
                for (int r = 0; r < N; ++r) { 
                    uint64_t s_row = s_mat[r]; 
                    if (__popcll(s_row & pi_col_vec) & 1) {
                        res_word |= (1ULL << r);
                    }
                }
            } else {
                for (int r = 0; r < N; ++r) {
                    // Fix: Removed unused 'pop' variable
                    uint64_t s_row = s_mat[r * (N/64)]; 
                    if (__popcll(s_row & pi_col_vec) & 1) res_word |= (1ULL << r);
                }
            }
            acc ^= res_word;
        }
    }
    Delta[col_idx] = acc;
}

template<int N, int M>
__global__ void k_apply_update(
    uint64_t* __restrict__ Pi_data,
    int Pi_len,
    const ColOp* __restrict__ ops,
    int n_ops,
    const uint32_t* __restrict__ shift_mask 
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    if (k >= Pi_len) return;

    const size_t mat_stride = size_t(M) * M / 64;
    uint64_t* pi_mat = Pi_data + k * mat_stride;

    for (int r = 0; r < M; ++r) {
        if (M == 128) {
            uint64_t row_w0 = pi_mat[r * 2 + 0];
            uint64_t row_w1 = pi_mat[r * 2 + 1];

            for (int i = 0; i < n_ops; ++i) {
                int src = ops[i].source;
                int dst = ops[i].target;
                
                bool bit_set = false;
                if (src < 64) bit_set = (row_w0 >> src) & 1;
                else          bit_set = (row_w1 >> (src-64)) & 1;

                if (bit_set) {
                    if (dst < 64) row_w0 ^= (1ULL << dst);
                    else          row_w1 ^= (1ULL << (dst-64));
                }
            }
            pi_mat[r * 2 + 0] = row_w0;
            pi_mat[r * 2 + 1] = row_w1;
        } else if (M == 64) {
            uint64_t row = pi_mat[r];
            for (int i = 0; i < n_ops; ++i) {
                if ((row >> ops[i].source) & 1) {
                    row ^= (1ULL << ops[i].target);
                }
            }
            pi_mat[r] = row;
        }
    }
}

template<int N, int M>
__global__ void k_apply_shift(
    uint64_t* __restrict__ Pi_data,
    int Pi_len,
    uint64_t shift_mask_lo, 
    uint64_t shift_mask_hi 
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;

    const size_t mat_stride = size_t(M) * M / 64;
    
    for (int k = Pi_len - 1; k > 0; --k) {
        uint64_t* curr = Pi_data + k * mat_stride;
        uint64_t* prev = Pi_data + (k-1) * mat_stride;
        
        if (M == 128) {
            uint64_t c_w0 = curr[r*2+0];
            uint64_t c_w1 = curr[r*2+1];
            uint64_t p_w0 = prev[r*2+0];
            uint64_t p_w1 = prev[r*2+1];

            curr[r*2+0] = (c_w0 & ~shift_mask_lo) | (p_w0 & shift_mask_lo);
            curr[r*2+1] = (c_w1 & ~shift_mask_hi) | (p_w1 & shift_mask_hi);
        } else if (M == 64) {
            uint64_t c = curr[r];
            uint64_t p = prev[r];
            curr[r] = (c & ~shift_mask_lo) | (p & shift_mask_lo);
        }
    }
    
    uint64_t* mat0 = Pi_data;
    if (M == 128) {
        mat0[r*2+0] &= ~shift_mask_lo;
        mat0[r*2+1] &= ~shift_mask_hi;
    } else if (M == 64) {
        mat0[r] &= ~shift_mask_lo;
    }
}

// =================================================================================
// WRAPPERS
// =================================================================================

template<int N, int M>
void compute_discrepancy(const PolyMatrixView<N>& S_dev, const PolyMatrixView<M>& Pi_dev, uint64_t* Delta, int t, cudaStream_t stream) {
    k_compute_discrepancy<N, M><<<M, 1, 0, stream>>>(S_dev.data, (int)S_dev.length, Pi_dev.data, (int)Pi_dev.length, Delta, t);
}

template<int N, int M>
void apply_update(PolyMatrixView<M> Pi_dev, const ColOp* ops_dev, int n_ops, cudaStream_t stream) {
    int threads = 256;
    int blocks = (Pi_dev.length + threads - 1) / threads;
    k_apply_update<N, M><<<blocks, threads, 0, stream>>>(Pi_dev.data, (int)Pi_dev.length, ops_dev, n_ops, nullptr);
}

template<int N, int M>
void apply_shift(PolyMatrixView<M> Pi_dev, uint64_t shift_mask_lo, uint64_t shift_mask_hi, cudaStream_t stream) {
    int threads = 32;
    int blocks = (M + threads - 1) / threads;
    k_apply_shift<N, M><<<blocks, threads, 0, stream>>>(Pi_dev.data, (int)Pi_dev.length, shift_mask_lo, shift_mask_hi);
}

// Instantiations
template void compute_discrepancy<64, 128>(const PolyMatrixView<64>&, const PolyMatrixView<128>&, uint64_t*, int, cudaStream_t);
template void apply_update<64, 128>(PolyMatrixView<128>, const ColOp*, int, cudaStream_t);
template void apply_shift<64, 128>(PolyMatrixView<128>, uint64_t, uint64_t, cudaStream_t);

} // namespace ops
} // namespace stage2
} // namespace lingen
