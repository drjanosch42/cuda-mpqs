// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

#include "vec_type.h"

#include "lingen/types.h"

#include "lingen/operations/poly_arithmetic.h"
#include "lingen/operations/matmul_gf2.h" 

namespace lingen {

#ifndef LINGEN_LEAF_MODE
#define LINGEN_LEAF_MODE 1
#endif

#ifndef LINGEN_ENABLE_KPB2
#define LINGEN_ENABLE_KPB2 0
#endif

#ifdef __CUDACC__

// --- Fused Naive Convolution Kernel (Robust 64-bit Path) ---
template <int N, int MPB>
__global__ void k_naive_conv_fused(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    int len_A,
    int len_B
) {
    // 1. Identification
    int k = blockIdx.x; // The index of the output coefficient C[k]
    int len_C = len_A + len_B - 1;
    if (k >= len_C) return;

    // 2. Convolution Bounds
    // We want A[i] * B[j] where i + j = k
    int i_min = max(0, k - len_B + 1);
    int i_max = min(k, len_A - 1);

    // 3. Accumulator (Local Registers)
    using Vec = VecType<N>;
    using T = typename Vec::Type;
    T acc = Vec::zero();

    // 4. Matrix Dimensions
    constexpr int ROW_BYTES = N / 8;
    constexpr int MAT_BYTES = N * ROW_BYTES;
    constexpr int WORDS_PER_ROW = N / 64;

    const uint8_t* A_base_ptr = reinterpret_cast<const uint8_t*>(A_ptr);
    const uint8_t* B_base_ptr = reinterpret_cast<const uint8_t*>(B_ptr);

    // 5. Convolution Loop
    for (int i = i_min; i <= i_max; ++i) {
        int j = k - i;
        
        const uint64_t* A_mat = reinterpret_cast<const uint64_t*>(A_base_ptr + i * MAT_BYTES);
        const uint64_t* B_mat = reinterpret_cast<const uint64_t*>(B_base_ptr + j * MAT_BYTES);
        
        uint64_t a_words[WORDS_PER_ROW];
        int row_idx = threadIdx.x;
        
        #pragma unroll
        for(int w=0; w<WORDS_PER_ROW; ++w) {
            a_words[w] = A_mat[row_idx * WORDS_PER_ROW + w];
        }

        #pragma unroll
        for (int w = 0; w < WORDS_PER_ROW; ++w) {
            uint64_t word = a_words[w];
            int bit_base = w * 64;

            #pragma unroll
            for (int bit = 0; bit < 64; ++bit) {
                if ((word >> bit) & 1ULL) {
                    const uint8_t* b_row_addr = reinterpret_cast<const uint8_t*>(B_mat) + (bit_base + bit) * ROW_BYTES;
                    T b_val = Vec::load(reinterpret_cast<const T*>(b_row_addr));
                    acc = Vec::xor_val(acc, b_val);
                }
            }
        }
    }

    uint8_t* C_mat_ptr = reinterpret_cast<uint8_t*>(C_ptr) + k * MAT_BYTES;
    T* C_row_ptr = reinterpret_cast<T*>(C_mat_ptr);
    Vec::store(&C_row_ptr[threadIdx.x], acc);
}

// --- Fused Naive Convolution Kernel (B cached in shared memory) ---
template <int N>
__global__ __launch_bounds__(N)
void k_naive_conv_fused_smemB(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    int len_A,
    int len_B
) {
    using Vec = VecType<N>;
    using T   = typename Vec::Type;

    constexpr int ROW_BYTES     = N / 8;
    constexpr int MAT_BYTES     = N * ROW_BYTES;
    constexpr int BITS_PER_LIMB = 64;
    constexpr int NUM_LIMBS     = N / BITS_PER_LIMB;

    const int k = (int)blockIdx.x;
    const int len_C = len_A + len_B - 1;
    if (k >= len_C) return;

    const int i_min = max(0, k - len_B + 1);
    const int i_max = min(k, len_A - 1);
    const int row_idx = (int)threadIdx.x; 

    __shared__ T B_shmem[N];
    T acc = Vec::zero();

    const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A_ptr);
    const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B_ptr);

    union RegView {
        T vec;
        uint64_t u64[NUM_LIMBS];
    };

    for (int i = i_min; i <= i_max; ++i) {
        const int j = k - i;

        const uint8_t* A_mat_bytes = A_base + (size_t)i * MAT_BYTES;
        const uint8_t* B_mat_bytes = B_base + (size_t)j * MAT_BYTES;

        const T* B_typed = reinterpret_cast<const T*>(B_mat_bytes);
        B_shmem[row_idx] = Vec::load(&B_typed[row_idx]);
        __syncthreads();

        const T* A_typed = reinterpret_cast<const T*>(A_mat_bytes);
        T a_row = Vec::load(&A_typed[row_idx]);

        RegView a_view;
        a_view.vec = a_row;

        #pragma unroll
        for (int limb = 0; limb < NUM_LIMBS; ++limb) {
            uint64_t word = a_view.u64[limb];
            const int k_base = limb * BITS_PER_LIMB;

            #pragma unroll
            for (int bit = 0; bit < BITS_PER_LIMB; ++bit) {
                if ((word >> bit) & 1ULL) {
                    acc = Vec::xor_val(acc, B_shmem[k_base + bit]);
                }
            }
        }
        __syncthreads();
    }

    uint8_t* C_base = reinterpret_cast<uint8_t*>(C_ptr);
    T* C_typed = reinterpret_cast<T*>(C_base + (size_t)k * MAT_BYTES);
    Vec::store(&C_typed[row_idx], acc);
}

// --- Fused Naive Convolution Leaf (B cached in shared memory), MPB outputs per block ---
template <int N, int MPB>
__global__ __launch_bounds__(N * MPB)
void k_naive_conv_fused_smemB_mpb(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    int len_A,
    int len_B
) {
    using Vec = VecType<N>;
    using T   = typename Vec::Type;

    constexpr int ROW_BYTES     = N / 8;
    constexpr int MAT_BYTES     = N * ROW_BYTES;
    constexpr int BITS_PER_LIMB = 64;
    constexpr int NUM_LIMBS     = N / BITS_PER_LIMB;

    const int row_idx   = (int)threadIdx.x; 
    const int local_k   = (int)threadIdx.y; 
    const int k         = (int)blockIdx.x * MPB + local_k;
    const int len_C     = len_A + len_B - 1;
    const bool is_active = (k < len_C);

    __shared__ T B_shmem[MPB][N];

    T acc = Vec::zero();

    const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A_ptr);
    const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B_ptr);

    union RegView {
        T vec;
        uint64_t u64[NUM_LIMBS];
    };

    for (int i = 0; i < len_A; ++i) {
        const int j = k - i;

        if (is_active && (j >= 0) && (j < len_B)) {
            const uint8_t* B_mat_bytes = B_base + (size_t)j * MAT_BYTES;
            const T* B_typed = reinterpret_cast<const T*>(B_mat_bytes);
            B_shmem[local_k][row_idx] = Vec::load(&B_typed[row_idx]);
        } else {
            B_shmem[local_k][row_idx] = Vec::zero();
        }
        __syncthreads();

        if (is_active && (j >= 0) && (j < len_B)) {
            const uint8_t* A_mat_bytes = A_base + (size_t)i * MAT_BYTES;
            const T* A_typed = reinterpret_cast<const T*>(A_mat_bytes);
            const T a_row = Vec::load(&A_typed[row_idx]);

            RegView a_view;
            a_view.vec = a_row;

            #pragma unroll
            for (int limb = 0; limb < NUM_LIMBS; ++limb) {
                const uint64_t word = a_view.u64[limb];
                const int k_base = limb * BITS_PER_LIMB;

                #pragma unroll
                for (int bit = 0; bit < BITS_PER_LIMB; ++bit) {
                    if ((word >> bit) & 1ULL) {
                        acc = Vec::xor_val(acc, B_shmem[local_k][k_base + bit]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (is_active) {
        uint8_t* C_base = reinterpret_cast<uint8_t*>(C_ptr);
        T* C_typed = reinterpret_cast<T*>(C_base + (size_t)k * MAT_BYTES);
        Vec::store(&C_typed[row_idx], acc);
    }
}

#endif

enum class PolyMulLeafKind : int {
    NaiveFused_Global = 0,
    NaiveFused_SmemB  = 1,
    NaiveFused_SmemB_MPB = 2,
};

struct KaratsubaTuneConfig {
    int threshold = 32;
    PolyMulLeafKind leaf_kind = PolyMulLeafKind::NaiveFused_SmemB;
    int leaf_mpb = 1; 
    bool prefer_smemB = true; 
};

#ifdef __CUDACC__
 
// Internal helper: launch the selected leaf.
template<int N>
inline void karatsuba_launch_leaf(
    cudaStream_t stream,
    PolyMatrixView<N> C,
    PolyMatrixView<N> A,
    PolyMatrixView<N> B,
    PolyMulLeafKind leaf_kind
) {
    const int grid = (int)(A.length + B.length - 1);

    switch (leaf_kind) {
        case PolyMulLeafKind::NaiveFused_Global:
            k_naive_conv_fused<N, 1><<<grid, N, 0, stream>>>(
                A.data, B.data, C.data, (int)A.length, (int)B.length
            );
            break;

        case PolyMulLeafKind::NaiveFused_SmemB:
            k_naive_conv_fused_smemB<N><<<grid, N, 0, stream>>>(
                A.data, B.data, C.data, (int)A.length, (int)B.length
            );
            break;

        case PolyMulLeafKind::NaiveFused_SmemB_MPB:
            if constexpr (N == 64) {
                constexpr int MPB = 8;
                const dim3 block(N, MPB, 1);
                const int len_C = (int)(A.length + B.length - 1);
                const int grid = (len_C + MPB - 1) / MPB;
                k_naive_conv_fused_smemB_mpb<64, MPB><<<grid, block, 0, stream>>>(
                      A.data, B.data, C.data, (int)A.length, (int)B.length
                );
            } else if constexpr (N == 128) {
                constexpr int MPB = 4;
                const dim3 block(N, MPB, 1);
                const int len_C = (int)(A.length + B.length - 1);
                const int grid = (len_C + MPB - 1) / MPB;
                k_naive_conv_fused_smemB_mpb<128, MPB><<<grid, block, 0, stream>>>(
                      A.data, B.data, C.data, (int)A.length, (int)B.length
                );
            } else if constexpr (N == 256) {
                constexpr int MPB = 2;
                const dim3 block(N, MPB, 1);
                const int len_C = (int)(A.length + B.length - 1);
                const int grid = (len_C + MPB - 1) / MPB;
                k_naive_conv_fused_smemB_mpb<256, MPB><<<grid, block, 0, stream>>>(
                      A.data, B.data, C.data, (int)A.length, (int)B.length
                );
            } else if constexpr (N == 512) {
                constexpr int MPB = 1;
                const dim3 block(N, MPB, 1);
                const int len_C = (int)(A.length + B.length - 1);
                const int grid = (len_C + MPB - 1) / MPB;
                k_naive_conv_fused_smemB_mpb<512, MPB><<<grid, block, 0, stream>>>(
                      A.data, B.data, C.data, (int)A.length, (int)B.length
                );
            } else {
                k_naive_conv_fused_smemB<N><<<grid, N, 0, stream>>>(
                    A.data, B.data, C.data, (int)A.length, (int)B.length
                );
            }
            break;
    }
}

template<int N>
static inline void launch_leaf_conv(cudaStream_t stream,
                                    PolyMatrixView<N> C,
                                    PolyMatrixView<N> A,
                                    PolyMatrixView<N> B)
{
    const int len_A = (int)A.length;
    const int len_B = (int)B.length;
    const int len_C = len_A + len_B - 1;

#if (LINGEN_LEAF_MODE == 0)
    if (N == 64)      k_naive_conv_fused<64,  1><<<len_C,  64, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 128) k_naive_conv_fused<128, 1><<<len_C, 128, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 256) k_naive_conv_fused<256, 1><<<len_C, 256, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 512) k_naive_conv_fused<512, 1><<<len_C, 512, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    return;
#elif (LINGEN_LEAF_MODE == 2)
    if (N == 64)      k_naive_conv_fused_smemB<64><<<len_C,  64, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 128) k_naive_conv_fused_smemB<128><<<len_C, 128, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 256) k_naive_conv_fused_smemB<256><<<len_C, 256, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    else if (N == 512) k_naive_conv_fused_smemB<512><<<len_C, 512, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
    return;
#else
    if constexpr (N >= 256) {
        k_naive_conv_fused_smemB<N><<<len_C, N, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
        return;
    } else {
        k_naive_conv_fused<N, 1><<<len_C, N, 0, stream>>>(A.data, B.data, C.data, len_A, len_B);
        return;
    }
#endif
} 

// Forward declare tuned version
template<int N>
inline void karatsuba_mul_tuned(
    cudaStream_t stream,
    PolyMatrixView<N> C,
    PolyMatrixView<N> A,
    PolyMatrixView<N> B,
    uint64_t* workspace,
    const KaratsubaTuneConfig& cfg
);

template<int N>
void karatsuba_mul(
    cudaStream_t stream,
    PolyMatrixView<N> C,
    PolyMatrixView<N> A,
    PolyMatrixView<N> B,
    uint64_t* workspace,
    int threshold = 32
) {
    if (A.length == 0 || B.length == 0) return;

    // Base case
    if (A.length < (size_t)threshold || B.length < (size_t)threshold) {
        launch_leaf_conv<N>(stream, C, A, B);
        return;
    }

    const size_t min_len = (A.length < B.length) ? A.length : B.length;
    size_t k = (min_len + 1) / 2;

    if (k == 0 || k >= A.length || k >= B.length) {
        launch_leaf_conv<N>(stream, C, A, B);
        return;
    }

    PolyMatrixView<N> A0 = A.split(0, k);
    PolyMatrixView<N> A1 = A.split(k, A.length - k);
    PolyMatrixView<N> B0 = B.split(0, k);
    PolyMatrixView<N> B1 = B.split(k, B.length - k);

    // Z0
    const size_t z0_len = A0.length + B0.length - 1;
    PolyMatrixView<N> T0_view = C.split(0, z0_len);
    karatsuba_mul(stream, T0_view, A0, B0, workspace, threshold);

    // Z2
    size_t z2_len = 0;
    if (A1.length > 0 && B1.length > 0) z2_len = A1.length + B1.length - 1;

    if (2 * k >= C.length) z2_len = 0;
    if (2 * k + z2_len > C.length) z2_len = (C.length > 2 * k) ? (C.length - 2 * k) : 0;

    PolyMatrixView<N> T1_view;
    if (z2_len > 0) {
        T1_view = C.split(2 * k, z2_len);
    } else {
        T1_view = PolyMatrixView<N>(nullptr, 0);
    }

    // Workspace
    const size_t asum_len = (A0.length > A1.length) ? A0.length : A1.length;
    const size_t bsum_len = (B0.length > B1.length) ? B0.length : B1.length;
    const size_t bytes_Asum = asum_len * PolyMatrixView<N>::MAT_SIZE_BYTES;
    const size_t bytes_Bsum = bsum_len * PolyMatrixView<N>::MAT_SIZE_BYTES;

    uint64_t* Asum_ptr = workspace;
    uint64_t* Bsum_ptr = workspace + (bytes_Asum / 8);
    uint64_t* Tmix_ptr = Bsum_ptr + (bytes_Bsum / 8);

    const size_t tmix_len = asum_len + bsum_len - 1;
    const size_t bytes_Tmix = tmix_len * PolyMatrixView<N>::MAT_SIZE_BYTES;

    uint64_t* next_workspace = Tmix_ptr + (bytes_Tmix / 8);

    if (z2_len > 0) {
        karatsuba_mul(stream, T1_view, A1, B1, next_workspace, threshold);
    }

    // Zero Gap
    const size_t gap_start = z0_len;
    const size_t gap_end   = (2 * k < C.length) ? (2 * k) : C.length;
    if (gap_start < gap_end) {
        PolyMatrixView<N> gap_view = C.split(gap_start, gap_end - gap_start);
        poly_zero_gpu(gap_view, stream);
    }

    // Tmix
    PolyMatrixView<N> Asum(Asum_ptr, asum_len);
    poly_zero_gpu(Asum, stream);
    poly_copy_gpu(Asum, A0, stream);
    poly_add_gpu(Asum, A1, stream);

    PolyMatrixView<N> Bsum(Bsum_ptr, bsum_len);
    poly_zero_gpu(Bsum, stream);
    poly_copy_gpu(Bsum, B0, stream);
    if (B1.length > 0) poly_add_gpu(Bsum, B1, stream);

    PolyMatrixView<N> Tmix(Tmix_ptr, tmix_len);
    karatsuba_mul(stream, Tmix, Asum, Bsum, next_workspace, threshold);

    // Recombine
    PolyMatrixView<N> T0_src = C.split(0, z0_len);
    poly_add_gpu(Tmix, T0_src, stream);

    if (z2_len > 0) {
        PolyMatrixView<N> T1_src = C.split(2 * k, z2_len);
        poly_add_gpu(Tmix, T1_src, stream);
    }

    size_t cmid_len = tmix_len;
    if (k < C.length) {
        if (k + cmid_len > C.length) cmid_len = C.length - k;
        PolyMatrixView<N> C_mid = C.split(k, cmid_len);
        PolyMatrixView<N> Tmix_clamped = Tmix.split(0, cmid_len);
        poly_add_gpu(C_mid, Tmix_clamped, stream);
    }
}

template<int N>
inline void karatsuba_mul_tuned(
    cudaStream_t stream,
    PolyMatrixView<N> C,
    PolyMatrixView<N> A,
    PolyMatrixView<N> B,
    uint64_t* workspace,
    const KaratsubaTuneConfig& cfg
) {
    if (A.length == 0 || B.length == 0) return;

    if (A.length < (size_t)cfg.threshold || B.length < (size_t)cfg.threshold) {
        karatsuba_launch_leaf<N>(stream, C, A, B, cfg.leaf_kind);
        return;
    }
    karatsuba_mul<N>(stream, C, A, B, workspace, cfg.threshold);
}

template<int N>
size_t karatsuba_workspace_size(size_t max_degree) {
    return max_degree * 8 * PolyMatrixView<N>::MAT_SIZE_BYTES; 
}

#endif
 
} // namespace lingen
