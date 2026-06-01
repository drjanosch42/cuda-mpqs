// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/operations/matmul_gf2.h"
#include "vec_type.h"       // Assumed to be in include path
#include "hpc_logger.h"
#include <cstdio>
#include <vector>

namespace lingen {

// =================================================================================
// HELPERS
// =================================================================================

__device__ __forceinline__ bool get_bit_device(const void* mat, size_t stride_bytes, int r, int c) {
    const uint8_t* row_base = reinterpret_cast<const uint8_t*>(mat) + r * stride_bytes;
    const uint64_t* row_u64 = reinterpret_cast<const uint64_t*>(row_base);
    return (row_u64[c / 64] >> (c % 64)) & 1ULL;
}  

// =================================================================================
// SPECIALIZED KERNELS (Powers of 2)
// =================================================================================

/**
 * @brief CUDA Kernel: C = A * B over GF(2) with explicit strides.
 * Specialized for N in {32, 64, 128, 256, 512}.
 */
template <int N, int MPB>
__global__ 
__launch_bounds__(N * MPB) 
void kernel_matmul_gf2_strided(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    size_t stride_A,
    size_t stride_B,
    size_t stride_C,
    size_t num_matrices
) {
    using Vec = VecType<N>;
    using T = typename Vec::Type;

    __shared__ T B_shmem[MPB * N];

    const int row_idx = threadIdx.x;       
    const int local_mat_idx = threadIdx.y; 
    
    const size_t global_mat_idx = blockIdx.x * MPB + local_mat_idx;
    const bool is_active = global_mat_idx < num_matrices;

    const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A_ptr) + global_mat_idx * stride_A;
    const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B_ptr) + global_mat_idx * stride_B;
    uint8_t* C_base       = reinterpret_cast<uint8_t*>(C_ptr)       + global_mat_idx * stride_C;

    T b_row_local = Vec::zero();
    if (is_active) {
        const T* B_typed = reinterpret_cast<const T*>(B_base);
        b_row_local = Vec::load(&B_typed[row_idx]);
    }
    B_shmem[local_mat_idx * N + row_idx] = b_row_local;

    T a_row = Vec::zero();
    if (is_active) {
        const T* A_typed = reinterpret_cast<const T*>(A_base);
        a_row = Vec::load(&A_typed[row_idx]);
    }

    __syncthreads();

    if (!is_active) return;

    T acc = Vec::zero();
    const T* B_local_shmem = &B_shmem[local_mat_idx * N];

    constexpr int BITS_PER_LIMB = 64;
    constexpr int NUM_LIMBS = N / BITS_PER_LIMB;
    
    union RegView { T vec; uint64_t u64[NUM_LIMBS]; };
    RegView a_view; a_view.vec = a_row;

    #pragma unroll
    for (int limb = 0; limb < NUM_LIMBS; ++limb) {
        uint64_t word = a_view.u64[limb];
        int k_base = limb * BITS_PER_LIMB;
        #pragma unroll
        for (int bit = 0; bit < BITS_PER_LIMB; ++bit) {
            if ((word >> bit) & 1ULL) {
                acc = Vec::xor_val(acc, B_local_shmem[k_base + bit]);
            }
        }
    }

    T* C_typed = reinterpret_cast<T*>(C_base);
    Vec::store(&C_typed[row_idx], acc);
}

// =================================================================================
// GENERIC TILED KERNEL (Any N <= 1024)
// =================================================================================

/**
 * @brief Generic Tiled MatMul for GF(2).
 * 
 * Supports any dimension N (passed as N_bits) up to MAX_W * 64.
 * Uses shared memory tiling on the inner dimension K to support arbitrary sizes efficiently.
 * 
 * Config:
 *  - 1 Block per Matrix (MPB=1).
 *  - Threads map to output rows.
 *  - Tiling on K (inner dimension).
 * 
 * @tparam MAX_W Maximum width in uint64 words (16 = 1024 bits).
 * @tparam TILE_K Blocking factor for inner loop (rows of B in shared mem).
 */
template <int MAX_W, int TILE_K>
__global__ void kernel_matmul_gf2_tiled_generic(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    size_t stride_A,
    size_t stride_B,
    size_t stride_C, // Bytes
    int N_bits,
    int width_words, // stride in words
    size_t num_matrices
) {
    size_t mat_idx = blockIdx.y;
    if (mat_idx >= num_matrices) return;

    // Base pointers for this matrix
    const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A_ptr) + mat_idx * stride_A;
    const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B_ptr) + mat_idx * stride_B;
    uint8_t* C_base       = reinterpret_cast<uint8_t*>(C_ptr)       + mat_idx * stride_C;

    // Output row index
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    bool active_row = (r < N_bits);

    // Accumulator (Registers)
    uint64_t acc[MAX_W];
    #pragma unroll
    for(int w=0; w<MAX_W; ++w) acc[w] = 0;

    // Shared Memory Tile: [TILE_K rows][width_words]
    // We flatten it to avoid dynamic array issues or stride mismatch
    // Size: TILE_K * width_words * 8 bytes.
    // For MAX_W=16, TILE_K=32 -> 4KB.
    extern __shared__ uint64_t B_tile[]; 
    // Layout: B_tile[k_local * width_words + w]

    // Loop over inner dimension (K) in chunks
    for (int k_base = 0; k_base < N_bits; k_base += TILE_K) {
        int limit = (k_base + TILE_K <= N_bits) ? TILE_K : (N_bits - k_base);
        
        // 1. Cooperative Load of B tile
        int total_words_in_tile = limit * width_words;
        for (int i = threadIdx.x; i < total_words_in_tile; i += blockDim.x) {
             // We treat B_base as uint64_t* for loading
             // B_base points to start of B matrix.
             // We need row (k_base + row_local).
             int row_local = i / width_words;
             int w_local   = i % width_words;
             int row_global = k_base + row_local;
             
             // Address calculation assuming row-major packed
             const uint64_t* B_row_ptr = reinterpret_cast<const uint64_t*>(B_base) + row_global * width_words;
             B_tile[i] = B_row_ptr[w_local];
        }
        __syncthreads();

        // 2. Compute Partial Product
        if (active_row) {
             // For each bit k in the current tile
             for (int k = 0; k < limit; ++k) {
                 int k_global = k_base + k;
                 // Check if A[r, k_global] is set
                 if (get_bit_device(A_base, width_words * 8, r, k_global)) {
                     // XOR row k from shared memory into acc
                     // Optimization: Loop unrolling for common small widths?
                     // Compiler usually handles this if width_words is small const, but here it's runtime.
                     // But MAX_W is small bound.
                     const uint64_t* b_row_shm = &B_tile[k * width_words];
                     for (int w = 0; w < width_words; ++w) {
                         acc[w] ^= b_row_shm[w];
                     }
                 }
             }
        }
        __syncthreads();
    }

    // 3. Store Result
    if (active_row) {
        uint64_t* C_row_ptr = reinterpret_cast<uint64_t*>(C_base) + r * width_words;
        for (int w = 0; w < width_words; ++w) C_row_ptr[w] = acc[w];
    }
}

// =================================================================================
// DISPATCH LOGIC
// =================================================================================

template<int N, int MPB>
void launch_specialized(const void* A, const void* B, void* C, size_t sA, size_t sB, size_t sC, size_t n, cudaStream_t stream) {
    dim3 block(N, MPB);
    size_t grid = (n + MPB - 1) / MPB;
    kernel_matmul_gf2_strided<N, MPB><<<grid, block, 0, stream>>>(A, B, C, sA, sB, sC, n);
}

void dispatch_matmul_generic(int N, const void* A, const void* B, void* C, size_t sA, size_t sB, size_t sC, size_t n_mats, cudaStream_t stream) {
    // Determine grid
    // One block per matrix (Grid.y), Threads cover rows (Grid.x * Block.x)
    // We use Block.x = 256.
    int block_dim = 256;
    int grid_dim_x = (N + block_dim - 1) / block_dim;
    dim3 grid(grid_dim_x, n_mats);
    
    int width_words = (N + 63) / 64;
    
    // Shared mem size: TILE_K * width_words * 8
    constexpr int TILE_K = 32;
    constexpr int MAX_W = 16; // Supports up to 1024 bits
    
    size_t shmem = TILE_K * width_words * sizeof(uint64_t);
    
    kernel_matmul_gf2_tiled_generic<MAX_W, TILE_K><<<grid, block_dim, shmem, stream>>>(
        A, B, C, sA, sB, sC, N, width_words, n_mats
    );
}

void matmul_gf2_gpu(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream) {
    size_t row_bytes = N / 8;
    size_t matrix_bytes = row_bytes * N;
    
    // Specialized kernels require N to be a supported power of 2 AND strides to be packed tightly.
    // If we add support for strides in specialized kernels, we can use them.
    // The current specialized kernel `kernel_matmul_gf2_strided` DOES support explicit strides.
    // However, it relies on VecType<N>, which only exists for 32, 64, 128, 256, 512.
    
    bool specialized = (N == 32 || N == 64 || N == 128 || N == 256 || N == 512);
    
    if (specialized) {
        if (N == 32)       launch_specialized<32, 16>(A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 64)  launch_specialized<64, 8>(A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 128) launch_specialized<128, 4>(A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 256) launch_specialized<256, 2>(A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 512) launch_specialized<512, 1>(A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
    } else {
        dispatch_matmul_generic(N, A, B, C, matrix_bytes, matrix_bytes, matrix_bytes, num_matrices, stream);
    }
}

void matmul_gf2_gpu_broadcast_A(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream) {
    size_t row_bytes = (size_t)(N + 63)/64 * 8; // Align to 8 bytes logic from PackedBitMatView
    size_t matrix_bytes = row_bytes * N;
    
    // Check if N matches power of 2 AND packed stride matches N/8.
    // The generalized stride logic `(N+63)/64 * 8` might differ from `N/8` if N not multiple of 64 (unlikely for N power of 2).
    // But generalized case N e.g. 192 -> row_bytes 24.
    // Specialized kernels use `VecType` loads which might assume alignment or width.
    // Safest to use specialized ONLY for standard powers of 2.
    
    bool specialized = (N == 32 || N == 64 || N == 128 || N == 256 || N == 512);
    
    if (specialized) {
        if (N == 32)       launch_specialized<32, 16>(A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 64)  launch_specialized<64, 8>(A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 128) launch_specialized<128, 4>(A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 256) launch_specialized<256, 2>(A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
        else if (N == 512) launch_specialized<512, 1>(A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
    } else {
        // Generic path for sums like 128+64=192
        dispatch_matmul_generic(N, A, B, C, 0, matrix_bytes, matrix_bytes, num_matrices, stream);
    }
} 

// =================================================================================
// GPU KERNELS (LEGACY)
// =================================================================================

/**
 * @brief CUDA Kernel: C = A * B over GF(2)
 * 
 * Strategy:
 *   - Each Thread Block handles 'MPB' output matrices.
 *   - Shared Memory: Loads 'MPB' matrices of B into Shared Memory.
 *     Size = MPB * N * sizeof(T).
 *   - Registers: Each thread holds one row of A.
 *   - Computation: Threads iterate over N bits of A-row. If bit set, XOR corresponding B-row from Shared.
 * 
 * Thread Block Layout:
 *   dim3(N, MPB)
 *   threadIdx.x (0..N-1): Represents the Row Index 'i' being computed for C.
 *   threadIdx.y (0..MPB-1): Represents which Matrix in the batch (local index).
 */
template <int N, int MPB>
__global__ 
__launch_bounds__(N * MPB) 
void kernel_matmul_gf2_legacy(
    const void* __restrict__ A_ptr,
    const void* __restrict__ B_ptr,
    void* __restrict__ C_ptr,
    size_t batch_stride_bytes,
    size_t num_matrices
) {
    using Vec = VecType<N>;
    using T = typename Vec::Type;

    // Shared memory: MPB matrices of B. Each matrix is N rows of type T.
    __shared__ T B_shmem[MPB * N];

    const int row_idx = threadIdx.x;       // Row 0..N-1
    const int local_mat_idx = threadIdx.y; // Local matrix 0..MPB-1
    
    // Global Matrix Index
    const size_t global_mat_idx = blockIdx.x * MPB + local_mat_idx;
    const bool is_active = global_mat_idx < num_matrices;

    // Calculate base pointers
    const uint8_t* A_base = reinterpret_cast<const uint8_t*>(A_ptr) + global_mat_idx * batch_stride_bytes;
    const uint8_t* B_base = reinterpret_cast<const uint8_t*>(B_ptr) + global_mat_idx * batch_stride_bytes;
    uint8_t* C_base       = reinterpret_cast<uint8_t*>(C_ptr)       + global_mat_idx * batch_stride_bytes;

    // 1. Cooperative Load of B into Shared Memory
    //    Each thread (row_idx) loads exactly one row of its assigned matrix (local_mat_idx)
    T b_row_local = Vec::zero();
    if (is_active) {
        const T* B_typed = reinterpret_cast<const T*>(B_base);
        b_row_local = Vec::load(&B_typed[row_idx]);
    }
    
    // Store to shared memory
    B_shmem[local_mat_idx * N + row_idx] = b_row_local;

    // 2. Load Row of A (Global Memory)
    //    Each thread computes C[row_idx], so it needs A[row_idx]
    T a_row = Vec::zero();
    if (is_active) {
        const T* A_typed = reinterpret_cast<const T*>(A_base);
        a_row = Vec::load(&A_typed[row_idx]);
    }

    // Barrier: Wait for B to be fully in Shared Memory
    __syncthreads();

    if (!is_active) return;

    // 3. Compute C[row_idx] = A[row_idx] * B
    T acc = Vec::zero();
    const T* B_local_shmem = &B_shmem[local_mat_idx * N];

    constexpr int BITS_PER_LIMB = 64;
    constexpr int NUM_LIMBS = N / BITS_PER_LIMB;
    
    union RegView {
        T vec;
        uint64_t u64[NUM_LIMBS];
    };

    RegView a_view;
    a_view.vec = a_row;

    #pragma unroll
    for (int limb = 0; limb < NUM_LIMBS; ++limb) {
        uint64_t word = a_view.u64[limb];
        int k_base = limb * BITS_PER_LIMB;

        #pragma unroll
        for (int bit = 0; bit < BITS_PER_LIMB; ++bit) {
            // Check bit 'k' of A[row_idx]
            if ((word >> bit) & 1ULL) {
                // acc ^= B[k]
                acc = Vec::xor_val(acc, B_local_shmem[k_base + bit]);
            }
        }
    }

    // 4. Store result
    T* C_typed = reinterpret_cast<T*>(C_base);
    Vec::store(&C_typed[row_idx], acc);
}

// =================================================================================
// GPU DISPATCHER (LEGACY)
// =================================================================================

 void matmul_gf2_gpu_legacy(int N, const void* A, const void* B, void* C, size_t num_matrices, cudaStream_t stream) {
    size_t row_bytes = N / 8;
    size_t matrix_bytes = row_bytes * N;

    // Heuristics for MPB based on N to maximize occupancy/register usage
    if (N == 32) {
        constexpr int MPB = 16;
        dim3 block(N, MPB);
        size_t grid = (num_matrices + MPB - 1) / MPB;
        kernel_matmul_gf2_legacy<32, MPB><<<grid, block, 0, stream>>>(A, B, C, matrix_bytes, num_matrices);
    } 
    else if (N == 64) {
        constexpr int MPB = 8;
        dim3 block(N, MPB);
        size_t grid = (num_matrices + MPB - 1) / MPB;
        kernel_matmul_gf2_legacy<64, MPB><<<grid, block, 0, stream>>>(A, B, C, matrix_bytes, num_matrices);
    } 
    else if (N == 128) {
        constexpr int MPB = 4;
        dim3 block(N, MPB);
        size_t grid = (num_matrices + MPB - 1) / MPB;
        kernel_matmul_gf2_legacy<128, MPB><<<grid, block, 0, stream>>>(A, B, C, matrix_bytes, num_matrices);
    } 
    else if (N == 256) {
        constexpr int MPB = 2;
        dim3 block(N, MPB);
        size_t grid = (num_matrices + MPB - 1) / MPB;
        kernel_matmul_gf2_legacy<256, MPB><<<grid, block, 0, stream>>>(A, B, C, matrix_bytes, num_matrices);
    } 
    else if (N == 512) {
        constexpr int MPB = 1;
        dim3 block(N, MPB);
        size_t grid = (num_matrices + MPB - 1) / MPB;
        kernel_matmul_gf2_legacy<512, MPB><<<grid, block, 0, stream>>>(A, B, C, matrix_bytes, num_matrices);
    } 
    else {
        LOG(LOG_ERROR_CRITICAL) << "[MatMulGPU] Unsupported N=" << N << std::endl;
        std::abort();
    }
}

// =================================================================================
// CPU REFERENCE
// =================================================================================

static inline bool get_bit(const uint64_t* mat, int N, int r, int c) {
    int word_idx = r * (N / 64) + (c / 64);
    int bit_idx = c % 64;
    return (mat[word_idx] >> bit_idx) & 1ULL;
}

static inline void xor_row(uint64_t* dest, const uint64_t* src, int words) {
    for (int i = 0; i < words; ++i) dest[i] ^= src[i];
}

void matmul_gf2_cpu(int N, const void* A_ptr, const void* B_ptr, void* C_ptr, size_t num_matrices) {
    size_t words_per_row = N / 64;
    size_t words_per_mat = N * words_per_row;
    
    const uint64_t* A = reinterpret_cast<const uint64_t*>(A_ptr);
    const uint64_t* B = reinterpret_cast<const uint64_t*>(B_ptr);
    uint64_t* C = reinterpret_cast<uint64_t*>(C_ptr);

    for (size_t m = 0; m < num_matrices; ++m) {
        const uint64_t* A_mat = A + m * words_per_mat;
        const uint64_t* B_mat = B + m * words_per_mat;
        uint64_t* C_mat       = C + m * words_per_mat;

        // Clear C
        std::fill(C_mat, C_mat + words_per_mat, 0);

        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                if (get_bit(A_mat, N, i, k)) {
                    // C[i] ^= B[k]
                    uint64_t* c_row = C_mat + i * words_per_row;
                    const uint64_t* b_row = B_mat + k * words_per_row;
                    xor_row(c_row, b_row, words_per_row);
                }
            }
        }
    }
}

} // namespace lingen
