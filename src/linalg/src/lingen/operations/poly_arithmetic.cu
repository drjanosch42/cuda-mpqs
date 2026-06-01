// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/operations/poly_arithmetic.h"
#include "hpc_logger.h"
#include <vector>
#include <cstring>

namespace lingen {

// =================================================================================
// GPU KERNELS
// =================================================================================

/**
 * @brief Kernel: Element-wise XOR of uint4 vectors.
 * 
 * Treats the polynomial data as a contiguous array of 128-bit words (uint4)
 * to maximize memory bandwidth.
 */
__global__ void kernel_poly_add_xor_u4(uint4* dst, const uint4* src, size_t n_u4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_u4) {
        uint4 a = dst[idx];
        uint4 b = src[idx];
        a.x ^= b.x; 
        a.y ^= b.y; 
        a.z ^= b.z; 
        a.w ^= b.w;
        dst[idx] = a;
    }
}

// Fallback for tail elements if total size is not divisible by 16 bytes
__global__ void kernel_poly_add_xor_u64(uint64_t* dst, const uint64_t* src, size_t n_u64) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_u64) {
        dst[idx] ^= src[idx];
    }
}

// =================================================================================
// GPU IMPLEMENTATIONS
// =================================================================================

template<int N>
void poly_add_gpu(PolyMatrixView<N> dst, PolyMatrixView<N> src, cudaStream_t stream) {
    size_t len = (dst.length < src.length) ? dst.length : src.length;
    size_t total_bytes = len * PolyMatrixView<N>::MAT_SIZE_BYTES;
    
    // Check alignment for uint4 (16 bytes)
    if (total_bytes % 16 == 0) {
        size_t n_u4 = total_bytes / 16;
        int threads = 256;
        int blocks = (n_u4 + threads - 1) / threads;
        kernel_poly_add_xor_u4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<uint4*>(dst.data),
            reinterpret_cast<const uint4*>(src.data),
            n_u4
        );
    } else {
        // Fallback to uint64 (should strictly not happen for N>=64 as 64*64/8 = 512 bytes)
        size_t n_u64 = total_bytes / 8;
        int threads = 256;
        int blocks = (n_u64 + threads - 1) / threads;
        kernel_poly_add_xor_u64<<<blocks, threads, 0, stream>>>(dst.data, src.data, n_u64);
    }
}

template<int N>
void poly_zero_gpu(PolyMatrixView<N> view, cudaStream_t stream) {
    size_t total_bytes = view.length * PolyMatrixView<N>::MAT_SIZE_BYTES;
    cudaMemsetAsync(view.data, 0, total_bytes, stream);
}

template<int N>
void poly_copy_gpu(PolyMatrixView<N> dst, PolyMatrixView<N> src, cudaStream_t stream) {
    size_t len = (dst.length < src.length) ? dst.length : src.length;
    size_t total_bytes = len * PolyMatrixView<N>::MAT_SIZE_BYTES;
    cudaMemcpyAsync(dst.data, src.data, total_bytes, cudaMemcpyDeviceToDevice, stream);
}

// =================================================================================
// CPU REFERENCE IMPLEMENTATIONS
// =================================================================================

template<int N>
void poly_add_cpu(PolyMatrixView<N> dst, PolyMatrixView<N> src) {
    // Note: PolyMatrixView pointers in this context point to HOST memory
    // if calling this function. However, the struct is designed for Device pointers.
    // **CRITICAL**: The caller must ensure 'data' pointers are valid host pointers.
    
    size_t len = (dst.length < src.length) ? dst.length : src.length;
    size_t total_u64 = len * PolyMatrixView<N>::MAT_STRIDE_WORDS;
    
    for(size_t i=0; i<total_u64; ++i) {
        dst.data[i] ^= src.data[i];
    }
}

// Helpers for naive CPU mul
static inline bool get_bit(const uint64_t* mat, int N, int r, int c) {
    int word_idx = r * (N / 64) + (c / 64);
    int bit_idx = c % 64;
    return (mat[word_idx] >> bit_idx) & 1ULL;
}

static inline void xor_row(uint64_t* dest, const uint64_t* src, int words) {
    for (int i = 0; i < words; ++i) dest[i] ^= src[i];
}

template<int N>
void poly_mul_naive_cpu(PolyMatrixView<N> C, PolyMatrixView<N> A, PolyMatrixView<N> B) {
    // Assumes A.data, B.data, C.data are HOST pointers.
    
    size_t words_per_row = N / 64;
    size_t words_per_mat = N * words_per_row;

    // Clear C
    std::memset(C.data, 0, C.length * words_per_mat * 8);

    for (size_t i = 0; i < A.length; ++i) {
        const uint64_t* A_mat = A.coeff(i);
        
        for (size_t j = 0; j < B.length; ++j) {
            if (i + j >= C.length) continue;
            
            const uint64_t* B_mat = B.coeff(j);
            uint64_t* C_mat = C.coeff(i + j);
            
            // C[i+j] += A[i] * B[j]
            for (int r = 0; r < N; ++r) {
                for (int k = 0; k < N; ++k) {
                    if (get_bit(A_mat, N, r, k)) {
                        // C_row[r] ^= B_row[k]
                        uint64_t* c_row = C_mat + r * words_per_row;
                        const uint64_t* b_row = B_mat + k * words_per_row;
                        xor_row(c_row, b_row, words_per_row);
                    }
                }
            }
        }
    }
}

// =================================================================================
// EXPLICIT INSTANTIATIONS
// =================================================================================

// N=32
template void poly_add_gpu<32>(PolyMatrixView<32>, PolyMatrixView<32>, cudaStream_t);
template void poly_zero_gpu<32>(PolyMatrixView<32>, cudaStream_t);
template void poly_copy_gpu<32>(PolyMatrixView<32>, PolyMatrixView<32>, cudaStream_t);

// N=64
template void poly_add_gpu<64>(PolyMatrixView<64>, PolyMatrixView<64>, cudaStream_t);
template void poly_zero_gpu<64>(PolyMatrixView<64>, cudaStream_t);
template void poly_copy_gpu<64>(PolyMatrixView<64>, PolyMatrixView<64>, cudaStream_t);
template void poly_add_cpu<64>(PolyMatrixView<64>, PolyMatrixView<64>);
template void poly_mul_naive_cpu<64>(PolyMatrixView<64>, PolyMatrixView<64>, PolyMatrixView<64>);

// N=128
template void poly_add_gpu<128>(PolyMatrixView<128>, PolyMatrixView<128>, cudaStream_t);
template void poly_zero_gpu<128>(PolyMatrixView<128>, cudaStream_t);
template void poly_copy_gpu<128>(PolyMatrixView<128>, PolyMatrixView<128>, cudaStream_t);
template void poly_add_cpu<128>(PolyMatrixView<128>, PolyMatrixView<128>);
template void poly_mul_naive_cpu<128>(PolyMatrixView<128>, PolyMatrixView<128>, PolyMatrixView<128>);

// N=256
template void poly_add_gpu<256>(PolyMatrixView<256>, PolyMatrixView<256>, cudaStream_t);
template void poly_zero_gpu<256>(PolyMatrixView<256>, cudaStream_t);
template void poly_copy_gpu<256>(PolyMatrixView<256>, PolyMatrixView<256>, cudaStream_t);
template void poly_add_cpu<256>(PolyMatrixView<256>, PolyMatrixView<256>);
template void poly_mul_naive_cpu<256>(PolyMatrixView<256>, PolyMatrixView<256>, PolyMatrixView<256>);

// N=512
template void poly_add_gpu<512>(PolyMatrixView<512>, PolyMatrixView<512>, cudaStream_t);
template void poly_zero_gpu<512>(PolyMatrixView<512>, cudaStream_t);
template void poly_copy_gpu<512>(PolyMatrixView<512>, PolyMatrixView<512>, cudaStream_t);
template void poly_add_cpu<512>(PolyMatrixView<512>, PolyMatrixView<512>);
template void poly_mul_naive_cpu<512>(PolyMatrixView<512>, PolyMatrixView<512>, PolyMatrixView<512>);

} // namespace lingen