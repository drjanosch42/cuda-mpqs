// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "sqrt_step.h"
#include "math_utils.cuh"
#include "montgomery.cuh"
#include "logger/hpc_logger.h"
#include "bw_solution_view.h"
#include "cuda_check.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <map>

namespace mpqs {
namespace sqrt {

// =============================================================================
// Local Math Helpers
// =============================================================================

namespace {
    
    // Unpack bit-vector from Block Wiedemann/Linear Algebra output
    std::vector<uint8_t> unpack_bits_local(const std::vector<uint64_t>& packed, size_t n_bits) {
        std::vector<uint8_t> unpacked;
        unpacked.reserve(n_bits);
        for (size_t i = 0; i < n_bits; ++i) {
            if ((packed[i / 64] >> (i % 64)) & 1) unpacked.push_back(1);
            else unpacked.push_back(0);
        }
        return unpacked;
    }

} // anonymous namespace

// =============================================================================
// Tuning Constants (M9)
// =============================================================================

constexpr uint32_t SQRT_CHUNK_SIZE      = 256;  // Relations per chunk for parallel reduction
constexpr uint32_t SQRT_TRANSFORM_BLOCK = 256;  // Threads per block for transform kernels
constexpr uint32_t SQRT_REDUCE_BLOCK    = 256;  // Threads per block for final reduction (must be power of 2)
constexpr uint32_t SQRT_EXP_THREADS     = 256;  // Threads per block for ExponentiateY (up from 128)
constexpr uint32_t SQRT_CSR_BLOCK       = 256;  // Threads per block for CSR scatter
constexpr uint32_t SQRT_WINDOW_BITS     = 4;    // Window size for windowed exponentiation
constexpr uint32_t SQRT_WINDOW_THRESHOLD= 16;   // Use windowed exp for exponents > this

// Post-factorization refinement constants (M10)
constexpr int REFINE_BLOCK_SIZE  = 128;  // Threads per block (≥ max solutions for compaction)
constexpr int MAX_POOL           = 64;   // Max distinct factors in refinement pool
constexpr int MAX_REFINE_ROUNDS  = 8;    // Convergence guaranteed in O(k²) ≪ 8

// =============================================================================
// BatchedComputeX Kernel (M2 — legacy single-block version)
// =============================================================================

#ifdef SQRT_LEGACY_KERNELS
/**
 * @brief Batched Montgomery product: X[j] = Π_{i ∈ S_j} sqrt_Q[i] (mod N).
 *
 * One thread per solution. Thread 0 loads each sqrt_Q[i] from global memory and
 * Montgomery-transforms it into shared memory; all threads then selectively
 * multiply into their private accumulator based on their solution bit-mask.
 * Two __syncthreads() barriers per iteration guard the shared memory broadcast.
 *
 * @param d_sqrt_Q      [num_relations] sqrt_Q values in standard form.
 * @param solutions     Packed solution bit-matrix (device-resident, kernel-passable).
 * @param d_X_out       [num_solutions] output — X values in standard form.
 * @param mont          Montgomery context (132 bytes, passed by value).
 * @param num_relations Number of relations to iterate.
 */
__global__ void BatchedComputeX(
    const mpqs::uint512*         __restrict__ d_sqrt_Q,
    lingen::BWKernelSolutionView              solutions,
    mpqs::uint512*               __restrict__ d_X_out,
    mpqs::math::Montgomery                    mont,
    uint32_t                                  num_relations
) {
    const uint32_t sol_idx = threadIdx.x;
    if (sol_idx >= solutions.num_solutions) return;

    // Initialize accumulator to Montgomery(1) = R mod N
    mpqs::uint512 X_mont = mont.transform(mpqs::uint512((uint32_t)1));

    // Use raw aligned storage to avoid CUDA warning #20054 (no dynamic init for __shared__)
    __shared__ alignas(alignof(mpqs::uint512)) unsigned char sq_mont_shared_buf[sizeof(mpqs::uint512)];
    mpqs::uint512& sq_mont_shared = *reinterpret_cast<mpqs::uint512*>(sq_mont_shared_buf);

    for (uint32_t i = 0; i < num_relations; ++i) {
        // Thread 0 loads and transforms sqrt_Q[i] — one transform per relation
        if (threadIdx.x == 0) {
            sq_mont_shared = mont.transform(d_sqrt_Q[i]);
        }
        __syncthreads();  // Ensure sq_mont_shared is written before any thread reads it

        // Each thread checks its solution bit for relation i
        const uint32_t word_idx = i / 64;
        const uint32_t bit_idx  = i % 64;
        uint64_t mask_word = solutions.d_data[sol_idx * solutions.words_per_vec + word_idx];

        if ((mask_word >> bit_idx) & 1) {
            X_mont = mont.mul(X_mont, sq_mont_shared);
        }
        __syncthreads();  // Guard shared memory before next iteration overwrites it
    }

    // Reduce from Montgomery domain to standard form and store
    d_X_out[sol_idx] = mont.reduce(X_mont);
}
#endif // SQRT_LEGACY_KERNELS

// =============================================================================
// TransformSqrtQ Kernel (M2 Phase 1)
// =============================================================================

/**
 * @brief Pre-transform sqrt_Q values from standard form to Montgomery form.
 *
 * Embarrassingly parallel: one thread per sqrt_Q value. This replaces the
 * serialized transform in BatchedComputeX where thread 0 transformed each
 * value inside the inner loop.
 *
 * Grid: <<<ceil(K / SQRT_TRANSFORM_BLOCK), SQRT_TRANSFORM_BLOCK>>>
 * Shared memory: none.
 *
 * @param d_sqrt_Q      [K] sqrt_Q values in standard form (input).
 * @param d_sqrt_Q_mont [K] Montgomery-transformed values (output).
 * @param mont          Montgomery context (passed by value).
 * @param K             Number of relations.
 */
__global__ void TransformSqrtQ(
    const mpqs::uint512* __restrict__ d_sqrt_Q,
    mpqs::uint512*       __restrict__ d_sqrt_Q_mont,
    mpqs::math::Montgomery            mont,
    uint32_t                          K
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    d_sqrt_Q_mont[idx] = mont.transform(d_sqrt_Q[idx]);
}

// =============================================================================
// TransformLargePrimes Kernel (M3 Phase 1 / Tile 3.1)
// =============================================================================

/**
 * @brief Embarrassingly parallel large-prime transform: __int128 -> Montgomery uint512.
 *
 * Each thread converts one large prime cofactor to Montgomery form. If lp <= 1
 * (no large prime, or identity), stores Montgomery(1) so the value is neutral
 * in subsequent Montgomery products.
 *
 * Grid: <<<ceil(K / SQRT_TRANSFORM_BLOCK), SQRT_TRANSFORM_BLOCK>>>
 * Shared memory: none.
 *
 * @param d_large_primes  [K] Large prime cofactors in __int128 form.
 * @param d_lp_mont       [K] Output: Montgomery-transformed large primes.
 * @param mont            Montgomery context (passed by value).
 * @param K               Number of relations.
 */
__global__ void TransformLargePrimes(
    const unsigned __int128* __restrict__ d_large_primes,
    mpqs::uint512*           __restrict__ d_lp_mont,
    mpqs::math::Montgomery               mont,
    uint32_t                              K
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    unsigned __int128 lp = d_large_primes[idx];
    if (lp > 1) {
        d_lp_mont[idx] = mont.transform(mpqs::uint512(lp));
    } else {
        // Identity element — neutral in Montgomery multiplication
        d_lp_mont[idx] = mont.transform(mpqs::uint512((uint32_t)1));
    }
}

// =============================================================================
// TransformFactorBase Kernel (M9 Tile 4.2)
// =============================================================================

/**
 * @brief Pre-transform factor base primes from uint32 to Montgomery form.
 *
 * Embarrassingly parallel: one thread per factor base prime. Run once per
 * factor base allocation (guarded by fb_transformed_ flag in host code).
 * Allows BatchedExponentiateY to read pre-transformed values directly,
 * eliminating one mont.transform() call per (thread × prime) in Phase 3.
 *
 * Grid: <<<ceil(fb_size / SQRT_TRANSFORM_BLOCK), SQRT_TRANSFORM_BLOCK>>>
 * Shared memory: none.
 *
 * @param d_factor_base       [fb_size] Factor base primes in uint32 form.
 * @param d_factor_base_mont  [fb_size] Montgomery-transformed output (uint512).
 * @param mont                Montgomery context (passed by value).
 * @param fb_size             Factor base size |FB|.
 */
__global__ void TransformFactorBase(
    const uint32_t*      __restrict__ d_factor_base,
    mpqs::uint512*       __restrict__ d_factor_base_mont,
    mpqs::math::Montgomery            mont,
    uint32_t                          fb_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fb_size) return;
    d_factor_base_mont[idx] = mont.transform(mpqs::uint512(d_factor_base[idx]));
}

// =============================================================================
// ComputeX_ChunkReduce Kernel (M2 Phase 2)
// =============================================================================

/**
 * @brief Block-level partial products for parallel ComputeX.
 *
 * 2D grid: (n_chunks, n_solutions). Each block handles SQRT_CHUNK_SIZE
 * consecutive relations for one solution. Threads load pre-transformed
 * Montgomery values from d_sqrt_Q_mont, mask with the solution bit (using
 * Montgomery identity mont.transform(1) for unselected or out-of-range
 * relations), and perform a shared-memory tree reduction to produce one
 * partial product per (chunk, solution) pair.
 *
 * Shared memory: SQRT_CHUNK_SIZE * sizeof(uint512) = 16 KB.
 *
 * @param d_sqrt_Q_mont [K] Montgomery-transformed sqrt_Q (from TransformSqrtQ).
 * @param solutions     Packed solution bit-matrix (device-resident, kernel-passable).
 * @param d_partials    [n_chunks * n_solutions] Output partial products (chunk-major).
 * @param mont          Montgomery context (passed by value).
 * @param K             Number of relations.
 * @param chunk_size    Block size / chunk width (must equal blockDim.x).
 */
__global__ void ComputeX_ChunkReduce(
    const mpqs::uint512*              __restrict__ d_sqrt_Q_mont,
    lingen::BWKernelSolutionView                   solutions,
    mpqs::uint512*                    __restrict__ d_partials,
    mpqs::math::Montgomery                         mont,
    uint32_t                                       K,
    uint32_t                                       chunk_size
) {
    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t sol_idx   = blockIdx.y;
    const uint32_t tid       = threadIdx.x;

    if (sol_idx >= solutions.num_solutions) return;

    // Global relation index for this thread
    const uint32_t rel_idx = chunk_idx * chunk_size + tid;

    // Load Montgomery(1) as identity; load transformed value if in range AND selected
    mpqs::uint512 val = mont.transform(mpqs::uint512((uint32_t)1));
    if (rel_idx < K) {
        // Check solution bit for this relation
        const uint32_t word_idx = rel_idx / 64;
        const uint32_t bit_idx  = rel_idx % 64;
        uint64_t mask_word = solutions.d_data[sol_idx * solutions.words_per_vec + word_idx];
        if ((mask_word >> bit_idx) & 1) {
            val = d_sqrt_Q_mont[rel_idx];
        }
    }

    // Tree reduction in shared memory
    extern __shared__ unsigned char smem_chunk[];
    mpqs::uint512* shared = reinterpret_cast<mpqs::uint512*>(smem_chunk);
    shared[tid] = val;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = mont.mul(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0 writes partial product (chunk-major layout)
    if (tid == 0) {
        d_partials[chunk_idx * solutions.num_solutions + sol_idx] = shared[0];
    }
}

// =============================================================================
// FinalReduce Kernel (M2 Phase 3 / reusable)
// =============================================================================

/**
 * @brief Reduce n_chunks partial products into one final value per solution.
 *
 * Grid: one block per solution (blockIdx.x = solution index).
 * Threads partition the n_chunks values via grid-stride loop, then tree-reduce
 * in shared memory.
 *
 * Reusable for LP product reduction in Milestone 3:
 *   - do_reduce=true  → output in standard form (mont.reduce applied)
 *   - do_reduce=false → output stays in Montgomery form
 *
 * Block size MUST be a power of 2 for correct tree reduction.
 * Shared memory: blockDim.x * sizeof(uint512) bytes.
 *
 * Partial layout: d_partials[chunk * n_solutions + sol] (chunk-major for
 * coalesced access when all solutions read the same chunk index).
 *
 * @param d_partials   [n_chunks * n_solutions] Partial products.
 * @param d_output     [n_solutions] Output — one reduced value per solution.
 * @param mont         Montgomery context (passed by value).
 * @param n_chunks     Number of chunks (partial products per solution).
 * @param n_solutions  Number of solutions.
 * @param do_reduce    If true, apply mont.reduce() to final result (standard form).
 *                     If false, leave in Montgomery form (for LP products).
 */
__global__ void FinalReduce(
    const mpqs::uint512* __restrict__ d_partials,
    mpqs::uint512*       __restrict__ d_output,
    mpqs::math::Montgomery            mont,
    uint32_t                          n_chunks,
    uint32_t                          n_solutions,
    bool                              do_reduce
) {
    const uint32_t sol_idx = blockIdx.x;
    const uint32_t tid     = threadIdx.x;

    if (sol_idx >= n_solutions) return;

    // Grid-stride accumulation of partial products for this solution.
    // Montgomery identity = R mod N = mont.transform(1).
    mpqs::uint512 acc = mont.transform(mpqs::uint512((uint32_t)1));
    for (uint32_t c = tid; c < n_chunks; c += blockDim.x) {
        acc = mont.mul(acc, d_partials[c * n_solutions + sol_idx]);
    }

    // Tree reduction in shared memory (block size must be power of 2)
    extern __shared__ unsigned char smem_final[];
    mpqs::uint512* shared = reinterpret_cast<mpqs::uint512*>(smem_final);
    shared[tid] = acc;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = mont.mul(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_output[sol_idx] = do_reduce ? mont.reduce(shared[0]) : shared[0];
    }
}

// =============================================================================
// BatchedComputeY Kernels (M3)
// =============================================================================

#ifdef SQRT_LEGACY_KERNELS
/**
 * @brief Phase 1 (legacy): Accumulate exponents across selected relations.
 *
 * Single-block kernel: one thread per solution (n ≤ 64), iterating all
 * relations sequentially.  Superseded by AccumExponents_Parallel + parallel
 * LP product pipeline (M9 Tile 3.3/3.4).  Kept for debugging reference.
 */
__global__ void BatchedAccumulateExponents(
    lingen::BWKernelSolutionView              solutions,
    const uint8_t*           __restrict__ d_signs,
    const int32_t*           __restrict__ d_val_2_exps,
    const unsigned __int128* __restrict__ d_large_primes,
    const uint64_t*          __restrict__ d_factor_offsets,
    const uint32_t*          __restrict__ d_factor_indices,
    const uint8_t*           __restrict__ d_factor_counts,
    uint32_t*                __restrict__ d_exp_matrix,
    uint32_t*                __restrict__ d_sign_counts,
    uint32_t*                __restrict__ d_exp2_accum,
    mpqs::uint512*           __restrict__ d_Y_lp_mont,
    mpqs::math::Montgomery                    mont,
    uint32_t                                  num_relations,
    uint32_t                                  fb_size
) {
    const uint32_t sol_idx = threadIdx.x;
    if (sol_idx >= solutions.num_solutions) return;

    // Thread-local accumulators (flushed to global memory at end)
    uint32_t sign_count = 0;
    uint32_t exp2_acc   = 0;
    mpqs::uint512 lp_mont = mont.transform(mpqs::uint512((uint32_t)1));

    for (uint32_t i = 0; i < num_relations; ++i) {
        // Check solution bit for relation i
        const uint32_t word_idx = i / 64;
        const uint32_t bit_idx  = i % 64;
        uint64_t mask_word = solutions.d_data[sol_idx * solutions.words_per_vec + word_idx];
        if (!((mask_word >> bit_idx) & 1)) continue;

        // Sign: count negative relations
        if (d_signs[i] != (uint8_t)1) sign_count++;

        // Power of 2
        exp2_acc += static_cast<uint32_t>(d_val_2_exps[i]);

        // Large prime: multiply L into Montgomery accumulator (L¹ contribution)
        unsigned __int128 lp = d_large_primes[i];
        if (lp > 1) {
            mpqs::uint512 lp_val(lp);
            mpqs::uint512 term = mont.transform(lp_val);
            lp_mont = mont.mul(lp_mont, term);
        }

        // CSR factors: accumulate exponents into this solution's row
        uint64_t csr_start = d_factor_offsets[i];
        uint64_t csr_end   = d_factor_offsets[i + 1];
        for (uint64_t k = csr_start; k < csr_end; ++k) {
            uint32_t fb_idx = d_factor_indices[k];
            uint8_t  count  = d_factor_counts[k];
            d_exp_matrix[sol_idx * fb_size + fb_idx] += count;
        }
    }

    // Flush accumulators to global memory
    d_sign_counts[sol_idx] = sign_count;
    d_exp2_accum[sol_idx]  = exp2_acc;
    d_Y_lp_mont[sol_idx]   = lp_mont;
}
#endif // SQRT_LEGACY_KERNELS

// =============================================================================
// AccumExponents_Parallel Kernel (M9 Tile 3.3)
// =============================================================================

/**
 * @brief Phase 1 (parallel): Scatter CSR exponents + accumulate sign/exp2.
 *
 * 2D grid: <<<dim3(n_chunks, n_solutions), SQRT_CHUNK_SIZE>>>. Each thread
 * handles one relation: checks the solution bit, and if selected, scatters
 * CSR exponents via atomicAdd to d_exp_matrix, and contributes sign count
 * and val_2_exp via atomicAdd to per-solution accumulators.
 *
 * No shared memory required — pure global scatter via atomics.
 * All three target arrays MUST be zero-initialized before launch (caller).
 *
 * @param solutions        Packed solution bit-matrix (device, kernel-passable view).
 * @param d_signs          [K] Relation signs (1 = positive).
 * @param d_val_2_exps     [K] Power-of-2 exponent per relation.
 * @param d_factor_offsets [K+1] CSR row pointers.
 * @param d_factor_indices [nnz] CSR column indices (factor base index).
 * @param d_factor_counts  [nnz] CSR values (exponent count per factor entry).
 * @param d_exp_matrix     [n × fb_size] Output exponent matrix (atomicAdd target).
 * @param d_sign_counts    [n] Output sign accumulator (atomicAdd target).
 * @param d_exp2_accum     [n] Output power-of-2 accumulator (atomicAdd target).
 * @param K                Number of relations.
 * @param fb_size          Factor base size |FB|.
 * @param chunk_size       Relations per chunk (must equal blockDim.x).
 */
__global__ void AccumExponents_Parallel(
    lingen::BWKernelSolutionView               solutions,
    const uint8_t*           __restrict__ d_signs,
    const int32_t*           __restrict__ d_val_2_exps,
    const uint64_t*          __restrict__ d_factor_offsets,
    const uint32_t*          __restrict__ d_factor_indices,
    const uint8_t*           __restrict__ d_factor_counts,
    uint32_t*                __restrict__ d_exp_matrix,
    uint32_t*                __restrict__ d_sign_counts,
    uint32_t*                __restrict__ d_exp2_accum,
    uint32_t                              K,
    uint32_t                              fb_size,
    uint32_t                              chunk_size
) {
    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t sol_idx   = blockIdx.y;
    const uint32_t tid       = threadIdx.x;

    if (sol_idx >= solutions.num_solutions) return;

    // Global relation index for this thread
    const uint32_t rel_idx = chunk_idx * chunk_size + tid;
    if (rel_idx >= K) return;

    // Check solution bit for this relation
    const uint32_t word_idx = rel_idx / 64;
    const uint32_t bit_idx  = rel_idx % 64;
    uint64_t mask_word = solutions.d_data[sol_idx * solutions.words_per_vec + word_idx];
    if (!((mask_word >> bit_idx) & 1)) return;

    // Sign: count negative relations (sign != 1 means negative)
    if (d_signs[rel_idx] != (uint8_t)1)
        atomicAdd(&d_sign_counts[sol_idx], 1u);

    // Power-of-2 exponent accumulation
    atomicAdd(&d_exp2_accum[sol_idx], static_cast<uint32_t>(d_val_2_exps[rel_idx]));

    // CSR factor scatter: accumulate exponents into this solution's row
    uint64_t csr_start = d_factor_offsets[rel_idx];
    uint64_t csr_end   = d_factor_offsets[rel_idx + 1];
    for (uint64_t k = csr_start; k < csr_end; ++k) {
        uint32_t fb_idx = d_factor_indices[k];
        uint8_t  count  = d_factor_counts[k];
        atomicAdd(&d_exp_matrix[sol_idx * fb_size + fb_idx], (uint32_t)count);
    }
}

/**
 * @brief Phase 2: Assert parity of accumulated exponents and halve them.
 *
 * 2D grid: x-dimension over FB primes, y-dimension over solutions.
 * Thread with p_idx == 0 additionally checks sign and exp2 parity.
 * Invalid solutions (odd exponents) are flagged in d_valid.
 * Sign counts are parity-checked but NOT halved (used as sign_count/2 in Phase 3).
 *
 * @param d_exp_matrix  [n × |FB|] Exponent matrix (halved in place).
 * @param d_sign_counts [n] Sign counts (parity-checked, NOT halved).
 * @param d_exp2_accum  [n] Power-of-2 accumulators (halved in place).
 * @param d_valid       [n] Validity flags (set to 0 on parity error).
 * @param n_solutions   Number of solutions.
 * @param fb_size       Factor base size.
 */
__global__ void HalveExponents(
    uint32_t*       __restrict__ d_exp_matrix,
    const uint32_t* __restrict__ d_sign_counts,
    uint32_t*       __restrict__ d_exp2_accum,
    uint8_t*        __restrict__ d_valid,
    uint32_t                     n_solutions,
    uint32_t                     fb_size
) {
    const uint32_t sol_idx = blockIdx.y;
    const uint32_t p_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (sol_idx >= n_solutions) return;

    // First thread per solution: check sign and exp2 parity, halve exp2
    if (p_idx == 0) {
        if (d_sign_counts[sol_idx] & 1) d_valid[sol_idx] = 0;
        uint32_t e2 = d_exp2_accum[sol_idx];
        if (e2 & 1) d_valid[sol_idx] = 0;
        d_exp2_accum[sol_idx] = e2 >> 1;
    }

    if (p_idx >= fb_size) return;

    uint32_t idx = sol_idx * fb_size + p_idx;
    uint32_t val = d_exp_matrix[idx];
    if (val & 1) d_valid[sol_idx] = 0;
    d_exp_matrix[idx] = val >> 1;
}

/**
 * @brief Phase 3: Compute Y[j] = (-1)^s · 2^e₂ · Π pᵢ^(eᵢ/2) · LP (mod N).
 *
 * One block per solution (blockIdx.x = solution index).  Within each block,
 * threads partition the factor base primes in a grid-stride loop, each computing
 * a partial Montgomery product via Montgomery::pow().  Products are tree-reduced
 * in shared memory.  Thread 0 then applies sign, factor-2, and large-prime
 * contributions before the final Montgomery::reduce().
 *
 * Block size MUST be a power of 2 for correct tree reduction.
 * Shared memory: blockDim.x × sizeof(uint512) bytes.
 *
 * @param d_exp_matrix   [n × |FB|] Halved exponents.
 * @param d_sign_counts  [n] Sign counts (un-halved; used as sign_count/2).
 * @param d_exp2_accum   [n] Halved power-of-2 exponents.
 * @param d_Y_lp_mont    [n] Large-prime Montgomery accumulators.
 * @param d_factor_base_mont  [|FB|] Factor base primes in Montgomery form (pre-transformed).
 * @param d_valid             [n] Validity flags.
 * @param d_Y_out             [n] Output Y values in standard form.
 * @param mont                Montgomery context.
 * @param N                   Modulus (for negate_mod_inplace).
 * @param n_solutions         Number of solutions.
 * @param fb_size             Factor base size.
 */
__global__ void BatchedExponentiateY(
    const uint32_t*      __restrict__ d_exp_matrix,
    const uint32_t*      __restrict__ d_sign_counts,
    const uint32_t*      __restrict__ d_exp2_accum,
    const mpqs::uint512* __restrict__ d_Y_lp_mont,
    const mpqs::uint512* __restrict__ d_factor_base_mont,
    const uint8_t*       __restrict__ d_valid,
    mpqs::uint512*       __restrict__ d_Y_out,
    mpqs::math::Montgomery                    mont,
    mpqs::uint512                             N,
    uint32_t                                  n_solutions,
    uint32_t                                  fb_size
) {
    const uint32_t sol_idx  = blockIdx.x;
    if (sol_idx >= n_solutions || d_valid[sol_idx] == 0) return;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;

    // Each thread computes a partial Montgomery product over its subset of FB primes
    mpqs::uint512 partial = mont.transform(mpqs::uint512((uint32_t)1));

    for (uint32_t p = tid; p < fb_size; p += nthreads) {
        uint32_t exp_val = d_exp_matrix[sol_idx * fb_size + p];
        if (exp_val == 0) continue;
        // Pre-transformed by TransformFactorBase (Tile 4.2) — no transform call needed
        mpqs::uint512 base = d_factor_base_mont[p];
        if (exp_val == 1) {
            // Fast path: skip mont.pow() — saves 2 Montgomery muls per exp==1 prime
            // (~30-50% of non-zero primes have exponent 1 after halving)
            partial = mont.mul(partial, base);
        } else if (exp_val <= SQRT_WINDOW_THRESHOLD) {
            // Standard binary exponentiation for small exponents
            mpqs::uint512 term = mont.pow(base, mpqs::uint512((uint32_t)exp_val));
            partial = mont.mul(partial, term);
        } else {
            // 4-bit windowed exponentiation for large exponents (saves ~25% of muls)
            // Precompute table: table[i] = base^(i+1) for i in [0, 2^W - 2]
            constexpr uint32_t TABLE_SIZE = (1u << SQRT_WINDOW_BITS) - 1;  // 15
            mpqs::uint512 table[TABLE_SIZE];
            table[0] = base;
            for (uint32_t i = 1; i < TABLE_SIZE; ++i)
                table[i] = mont.mul(table[i - 1], base);

            // Scan exponent in W-bit windows from MSB to LSB
            mpqs::uint512 result = mont.transform(mpqs::uint512((uint32_t)1));
            uint32_t e = exp_val;
            // Find highest set bit, round up to next multiple of window size
            int bit_pos = 31 - __clz(e);
            bit_pos = ((bit_pos + SQRT_WINDOW_BITS) & ~(SQRT_WINDOW_BITS - 1)) - 1;

            for (int pos = bit_pos; pos >= (int)(SQRT_WINDOW_BITS - 1); pos -= SQRT_WINDOW_BITS) {
                // Square W times
                for (uint32_t s = 0; s < SQRT_WINDOW_BITS; ++s)
                    result = mont.mul(result, result);
                // Extract W-bit window and multiply by precomputed power
                uint32_t window = (e >> (pos - (SQRT_WINDOW_BITS - 1))) & ((1u << SQRT_WINDOW_BITS) - 1);
                if (window > 0)
                    result = mont.mul(result, table[window - 1]);
            }
            partial = mont.mul(partial, result);
        }
    }

    // Tree reduction via shared memory (nthreads must be power of 2)
    extern __shared__ unsigned char smem_y_buf[];
    mpqs::uint512* shared_prods = reinterpret_cast<mpqs::uint512*>(smem_y_buf);
    shared_prods[tid] = partial;
    __syncthreads();

    for (uint32_t stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_prods[tid] = mont.mul(shared_prods[tid], shared_prods[tid + stride]);
        }
        __syncthreads();
    }

    // Thread 0: apply sign, factor-2, large primes, and reduce to standard form
    if (tid == 0) {
        mpqs::uint512 Y_mont = shared_prods[0];

        // Sign: (-1)^(sign_count / 2)  — matches CPU oracle line 233
        if ((d_sign_counts[sol_idx] / 2) & 1) {
            Y_mont.negate_mod_inplace(N);
        }

        // Factor 2: 2^(exp2_halved)
        uint32_t exp2_half = d_exp2_accum[sol_idx];
        if (exp2_half > 0) {
            mpqs::uint512 base2 = mont.transform(mpqs::uint512((uint32_t)2));
            mpqs::uint512 term2 = mont.pow(base2, mpqs::uint512((uint32_t)exp2_half));
            Y_mont = mont.mul(Y_mont, term2);
        }

        // Large primes: merge accumulated LP Montgomery product
        Y_mont = mont.mul(Y_mont, d_Y_lp_mont[sol_idx]);

        // Reduce from Montgomery domain and store
        d_Y_out[sol_idx] = mont.reduce(Y_mont);
    }
}

// =============================================================================
// ApplyLPCorrection Kernel (M5)
// =============================================================================

/// Multiply Y[sol] by a precomputed Montgomery-domain LP correction value.
/// Used by the expanded-matrix path where LP Y-contributions are precomputed
/// on the CPU and uploaded, bypassing the standard LP accumulation pipeline.
__global__ void ApplyLPCorrectionKernel(
    mpqs::uint512*       __restrict__ d_Y,
    const mpqs::uint512* __restrict__ d_correction_mont,
    mpqs::math::Montgomery            mont,
    uint32_t                          n_solutions
) {
    const uint32_t sol = blockIdx.x * blockDim.x + threadIdx.x;
    if (sol >= n_solutions) return;
    // d_Y is in standard form (post-reduce). Transform → multiply → reduce.
    mpqs::uint512 Y_mont = mont.transform(d_Y[sol]);
    Y_mont = mont.mul(Y_mont, d_correction_mont[sol]);
    d_Y[sol] = mont.reduce(Y_mont);
}

// =============================================================================
// BatchedGCD Kernel (M4)
// =============================================================================

/**
 * @brief Batched factor extraction: gcd(|X[j]-Y[j]|, N) and gcd(X[j]+Y[j], N).
 *
 * One thread per solution (n ≤ 64). Each thread independently computes:
 *   1. diff = |X[j] - Y[j]|  (explicit compare to avoid unsigned underflow wrap)
 *   2. f1 = gcd(diff, N)
 *   3. If f1 nontrivial → store and return
 *   4. sum = X[j] + Y[j]     (wrapping at 512 bits, matching CPU behavior)
 *   5. f2 = gcd(sum, N)
 *   6. If f2 nontrivial → store and return
 *   7. Otherwise → trivial (status = 0)
 *
 * Uses mpqs::math::gcd() from math_utils.cuh (__host__ __device__ Euclidean GCD).
 *
 * @param d_X             [n] X values in standard form.
 * @param d_Y             [n] Y values in standard form.
 * @param N               The number being factored.
 * @param d_factors       [n] Output: nontrivial factor candidate per solution.
 * @param d_factor_status [n] Output: 0 = trivial, 1 = via |X-Y|, 2 = via X+Y.
 * @param n               Number of solutions.
 */
__global__ void BatchedGCDKernel(
    const mpqs::uint512* __restrict__ d_X,
    const mpqs::uint512* __restrict__ d_Y,
    const mpqs::uint512               N,
    mpqs::uint512*       __restrict__ d_factors,
    int*                 __restrict__ d_factor_status,
    int                              n
) {
    const int j = threadIdx.x;
    if (j >= n) return;

    mpqs::uint512 Xj = d_X[j];
    mpqs::uint512 Yj = d_Y[j];

    // 1. Absolute difference: |X - Y| with explicit comparison (uint512 wraps on underflow)
    mpqs::uint512 diff;
    if (Xj < Yj) {
        diff = Yj; diff.sub(Xj);
    } else {
        diff = Xj; diff.sub(Yj);
    }

    // 2. f1 = gcd(|X - Y|, N)
    mpqs::uint512 f1 = mpqs::math::gcd(diff, N);
    bool f1_trivial = f1.is_one() || (f1 == N);

    if (!f1_trivial) {
        d_factors[j] = f1;
        d_factor_status[j] = 1;
        return;
    }

    // 3. Sum: X + Y (wrapping at 512 bits, matching CPU)
    mpqs::uint512 sum = Xj;
    sum.add(Yj);

    // 4. f2 = gcd(X + Y, N)
    mpqs::uint512 f2 = mpqs::math::gcd(sum, N);
    bool f2_trivial = f2.is_one() || (f2 == N);

    if (!f2_trivial) {
        d_factors[j] = f2;
        d_factor_status[j] = 2;
        return;
    }

    // No nontrivial factor from this solution
    d_factor_status[j] = 0;
}

// =============================================================================
// Triangular Index Decoding Helper (M10)
// =============================================================================

/// Maps a linear pair index p ∈ [0, m*(m-1)/2) to (i, j) with 0 ≤ i < j < m.
/// Simple loop — for m ≤ 64, the cost is negligible vs the ~1000-cycle GCD.
__device__ void decode_triangular(int p, int m, int* i, int* j) {
    int row = 0, acc = 0;
    while (acc + (m - 1 - row) <= p) {
        acc += (m - 1 - row);
        row++;
    }
    *i = row;
    *j = row + 1 + (p - acc);
}

// =============================================================================
// RefineFactorsKernel — Post-Factorization Coprime Refinement (M10)
// =============================================================================

/**
 * @brief Extracts the finest factorization from BatchedGCDKernel output.
 *
 * Three-phase single-block kernel:
 *   Phase 0: Compact non-trivial factors, sort, deduplicate → shared pool[m].
 *   Phase 1: Iterative pairwise GCD refinement until pool is pairwise coprime.
 *   Phase 2: Write refined factors to global memory.
 *
 * For semiprimes (N = pq): m = 1, zero refinement rounds → ~2μs.
 * For multi-factor N: converges in O(k²) rounds where k = distinct prime powers.
 *
 * Grid:  <<<1, REFINE_BLOCK_SIZE>>>  (REFINE_BLOCK_SIZE = 128)
 * Shared: pool[MAX_POOL] + scratch[MAX_POOL*3] + counters ≈ 16.4 KB
 *
 * @param d_factors        [n] Factor candidates from BatchedGCDKernel.
 * @param d_factor_status  [n] Status: 0=trivial, 1=via|X-Y|, 2=via X+Y.
 * @param N                Number being factored.
 * @param d_refined        [MAX_POOL] Output: refined pairwise-coprime factors.
 * @param d_refined_count  Output: number of refined factors.
 * @param n                Number of solutions.
 */
__global__ __launch_bounds__(REFINE_BLOCK_SIZE) void RefineFactorsKernel(
    const mpqs::uint512* __restrict__ d_factors,
    const int*           __restrict__ d_factor_status,
    const mpqs::uint512               N,
    mpqs::uint512*       __restrict__ d_refined,
    int*                 __restrict__ d_refined_count,
    int                              n
) {
    // ---- Shared memory layout (manual partitioning of extern smem) ----
    extern __shared__ unsigned char smem_refine[];
    // pool:    MAX_POOL * sizeof(uint512)     = 4096 bytes  [offset 0]
    // scratch: MAX_POOL * 3 * sizeof(uint512) = 12288 bytes [offset 4096]
    // counters: 3 * sizeof(int)               = 12 bytes    [offset 16384]
    mpqs::uint512* pool    = reinterpret_cast<mpqs::uint512*>(smem_refine);
    mpqs::uint512* scratch = reinterpret_cast<mpqs::uint512*>(smem_refine + MAX_POOL * sizeof(mpqs::uint512));
    int* s_pool_count  = reinterpret_cast<int*>(smem_refine + MAX_POOL * 4 * sizeof(mpqs::uint512));
    int* s_changed     = s_pool_count + 1;
    int* s_split_count = s_pool_count + 2;

    const int tid = threadIdx.x;

    // ====================================================================
    // Phase 0: Compact non-trivial factors into pool[], then deduplicate
    // ====================================================================

    // Thread-0 initializes counters
    if (tid == 0) {
        *s_pool_count = 0;
        *s_split_count = 0;
    }
    __syncthreads();

    // Each thread with tid < n checks its factor; non-trivial ones get compacted
    if (tid < n && d_factor_status[tid] != 0) {
        int idx = atomicAdd(s_pool_count, 1);
        if (idx < MAX_POOL) {
            pool[idx] = d_factors[tid];
        }
    }
    __syncthreads();

    // Clamp pool_count to MAX_POOL
    if (tid == 0 && *s_pool_count > MAX_POOL) {
        *s_pool_count = MAX_POOL;
    }
    __syncthreads();

    int m = *s_pool_count;

    // If no non-trivial factors, output empty set and return
    if (m == 0) {
        if (tid == 0) *d_refined_count = 0;
        return;
    }

    // Sort pool (insertion sort — m ≤ 64, single-thread is fine)
    if (tid == 0) {
        for (int i = 1; i < m; ++i) {
            mpqs::uint512 key = pool[i];
            int j = i - 1;
            while (j >= 0 && key < pool[j]) {
                pool[j + 1] = pool[j];
                j--;
            }
            pool[j + 1] = key;
        }

        // Deduplicate adjacent equal elements (pool is sorted)
        int write = 0;
        for (int i = 0; i < m; ++i) {
            if (i > 0 && pool[i] == pool[i - 1]) continue;
            if (pool[i].is_one()) continue;
            if (pool[i] == N) continue;
            pool[write++] = pool[i];
        }
        *s_pool_count = write;
    }
    __syncthreads();

    m = *s_pool_count;
    if (m == 0) {
        if (tid == 0) *d_refined_count = 0;
        return;
    }

    // ====================================================================
    // Phase 1: Iterative pairwise GCD refinement
    // ====================================================================

    for (int round = 0; round < MAX_REFINE_ROUNDS; ++round) {
        if (tid == 0) {
            *s_changed = 0;
            *s_split_count = 0;
        }
        __syncthreads();

        m = *s_pool_count;
        int total_pairs = m * (m - 1) / 2;

        // Each thread takes one pair from the triangular matrix
        if (tid < total_pairs) {
            int i, j;
            decode_triangular(tid, m, &i, &j);

            mpqs::uint512 g = mpqs::math::gcd(pool[i], pool[j]);

            // Check if g reveals a finer split
            if (!g.is_one() && !(g == pool[i]) && !(g == pool[j])) {
                atomicExch(s_changed, 1);
                // Record the split: scratch layout [g, A/g, B/g] per split
                int split_idx = atomicAdd(s_split_count, 1);
                if (split_idx < MAX_POOL) {
                    scratch[split_idx * 3 + 0] = g;
                    mpqs::uint512 a_div_g = pool[i];
                    a_div_g.div(g);
                    scratch[split_idx * 3 + 1] = a_div_g;
                    mpqs::uint512 b_div_g = pool[j];
                    b_div_g.div(g);
                    scratch[split_idx * 3 + 2] = b_div_g;
                }
            }
        }
        __syncthreads();

        if (*s_changed == 0) break;  // Pool is pairwise coprime — convergence

        // Rebuild pool: thread 0 applies splits, deduplicates, removes trivials
        if (tid == 0) {
            int n_splits = *s_split_count;
            if (n_splits > MAX_POOL) n_splits = MAX_POOL;

            // Collect all unique non-trivial fragments
            mpqs::uint512 tmp[MAX_POOL * 2];  // Local array (register/stack, max 8KB)
            int tmp_count = 0;

            // Mark involved pool indices (m ≤ 64 → fits in uint64_t)
            uint64_t involved = 0;
            for (int s = 0; s < n_splits; ++s) {
                mpqs::uint512 g     = scratch[s * 3 + 0];
                mpqs::uint512 a_dg  = scratch[s * 3 + 1];
                mpqs::uint512 b_dg  = scratch[s * 3 + 2];

                // Reconstruct A = g * a_dg, B = g * b_dg to identify involved entries
                mpqs::uint512 A = g; A.mult(a_dg);
                mpqs::uint512 B = g; B.mult(b_dg);

                for (int k = 0; k < m; ++k) {
                    if (pool[k] == A || pool[k] == B) {
                        involved |= (1ULL << k);
                    }
                }

                // Add fragments (if non-trivial and not N)
                if (!g.is_one() && !(g == N) && tmp_count < MAX_POOL * 2)
                    tmp[tmp_count++] = g;
                if (!a_dg.is_one() && !(a_dg == N) && tmp_count < MAX_POOL * 2)
                    tmp[tmp_count++] = a_dg;
                if (!b_dg.is_one() && !(b_dg == N) && tmp_count < MAX_POOL * 2)
                    tmp[tmp_count++] = b_dg;
            }

            // Keep non-involved pool entries
            for (int k = 0; k < m; ++k) {
                if (!(involved & (1ULL << k)) && tmp_count < MAX_POOL * 2) {
                    tmp[tmp_count++] = pool[k];
                }
            }

            // Sort tmp
            for (int i = 1; i < tmp_count; ++i) {
                mpqs::uint512 key = tmp[i];
                int j = i - 1;
                while (j >= 0 && key < tmp[j]) {
                    tmp[j + 1] = tmp[j];
                    j--;
                }
                tmp[j + 1] = key;
            }

            // Deduplicate and filter trivials
            int write = 0;
            for (int i = 0; i < tmp_count; ++i) {
                if (i > 0 && tmp[i] == tmp[i - 1]) continue;
                if (tmp[i].is_one()) continue;
                if (tmp[i] == N) continue;
                if (write < MAX_POOL) pool[write++] = tmp[i];
            }
            *s_pool_count = write;
        }
        __syncthreads();
    }

    // ====================================================================
    // Phase 2: Write refined factors to global memory
    // ====================================================================

    m = *s_pool_count;
    if (tid < m) {
        d_refined[tid] = pool[tid];
    }
    if (tid == 0) {
        *d_refined_count = m;
    }
}

// =============================================================================
// Implementation of SquareRootRefinement
// =============================================================================

SquareRootRefinement::SquareRootRefinement(const mpqs::uint512& n) : N_(n) {
    cudaStreamCreate(&stream_);
}

SquareRootRefinement::~SquareRootRefinement() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    freeDeviceBuffers();
}

void SquareRootRefinement::allocateDeviceBuffers(
    uint32_t max_K, uint32_t max_nnz, uint32_t max_fb, uint32_t max_n
) {
    if (bufs_.allocated
        && max_K <= bufs_.max_K && max_nnz <= bufs_.max_nnz
        && max_fb <= bufs_.max_fb && max_n <= bufs_.max_n) {
        return;  // Current buffers are large enough
    }

    // Free existing buffers if reallocating
    freeDeviceBuffers();

    // Apply 1.5x safety margin for future reuse
    bufs_.max_K   = static_cast<uint32_t>(max_K * 1.5);
    bufs_.max_nnz = static_cast<uint32_t>(max_nnz * 1.5);
    bufs_.max_fb  = static_cast<uint32_t>(max_fb * 1.5);
    bufs_.max_n   = static_cast<uint32_t>(max_n * 1.5);

    // Helper: check cudaMalloc return and log on failure
    auto cuda_check = [](cudaError_t err, const char* name) {
        if (err != cudaSuccess) {
            LOG(LOG_ERROR_CRITICAL) << "cudaMalloc failed for " << name
                                    << ": " << cudaGetErrorString(err);
            throw std::runtime_error(std::string("cudaMalloc failed for ") + name
                + ": " + cudaGetErrorString(err));
        }
    };

    // Relation data
    cuda_check(cudaMalloc(&bufs_.d_sqrt_Q,        bufs_.max_K * sizeof(uint512)),        "d_sqrt_Q");
    cuda_check(cudaMalloc(&bufs_.d_sqrt_Q_mont,   bufs_.max_K * sizeof(uint512)),        "d_sqrt_Q_mont");
    cuda_check(cudaMalloc(&bufs_.d_signs,         bufs_.max_K * sizeof(uint8_t)),         "d_signs");
    cuda_check(cudaMalloc(&bufs_.d_val_2_exps,    bufs_.max_K * sizeof(int32_t)),         "d_val_2_exps");
    cuda_check(cudaMalloc(&bufs_.d_large_primes,  bufs_.max_K * sizeof(unsigned __int128)), "d_large_primes");
    cuda_check(cudaMalloc(&bufs_.d_factor_offsets,(bufs_.max_K + 1) * sizeof(uint64_t)),  "d_factor_offsets");
    cuda_check(cudaMalloc(&bufs_.d_factor_indices,bufs_.max_nnz * sizeof(uint32_t)),      "d_factor_indices");
    cuda_check(cudaMalloc(&bufs_.d_factor_counts, bufs_.max_nnz * sizeof(uint8_t)),       "d_factor_counts");

    // Large prime Montgomery products
    cuda_check(cudaMalloc(&bufs_.d_lp_mont,       bufs_.max_K * sizeof(uint512)),         "d_lp_mont");

    // Factor base
    cuda_check(cudaMalloc(&bufs_.d_factor_base,       bufs_.max_fb * sizeof(uint32_t)),   "d_factor_base");
    cuda_check(cudaMalloc(&bufs_.d_factor_base_mont,  bufs_.max_fb * sizeof(uint512)),    "d_factor_base_mont");

    // Per-solution outputs
    cuda_check(cudaMalloc(&bufs_.d_X,             bufs_.max_n * sizeof(uint512)),          "d_X");
    cuda_check(cudaMalloc(&bufs_.d_Y,             bufs_.max_n * sizeof(uint512)),          "d_Y");
    cuda_check(cudaMalloc(&bufs_.d_exp_matrix,    bufs_.max_n * bufs_.max_fb * sizeof(uint32_t)), "d_exp_matrix");
    cuda_check(cudaMalloc(&bufs_.d_sign_counts,   bufs_.max_n * sizeof(uint32_t)),         "d_sign_counts");
    cuda_check(cudaMalloc(&bufs_.d_exp2_accum,    bufs_.max_n * sizeof(uint32_t)),         "d_exp2_accum");
    cuda_check(cudaMalloc(&bufs_.d_Y_lp_mont,     bufs_.max_n * sizeof(uint512)),          "d_Y_lp_mont");
    cuda_check(cudaMalloc(&bufs_.d_valid,         bufs_.max_n * sizeof(uint8_t)),           "d_valid");

    // Scratch for parallel reductions
    {
        const uint32_t max_chunks = (bufs_.max_K + SQRT_CHUNK_SIZE - 1) / SQRT_CHUNK_SIZE;
        cuda_check(cudaMalloc(&bufs_.d_partials_x,  max_chunks * bufs_.max_n * sizeof(uint512)),  "d_partials_x");
        cuda_check(cudaMalloc(&bufs_.d_partials_lp, max_chunks * bufs_.max_n * sizeof(uint512)),  "d_partials_lp");
    }

    // GCD outputs
    cuda_check(cudaMalloc(&bufs_.d_factors,       bufs_.max_n * sizeof(uint512)),          "d_factors");
    cuda_check(cudaMalloc(&bufs_.d_factor_status, bufs_.max_n * sizeof(int)),              "d_factor_status");

    // Post-factorization refinement output (M10)
    constexpr int REFINE_MAX_POOL = 64;
    cuda_check(cudaMalloc(&bufs_.d_refined_factors, REFINE_MAX_POOL * sizeof(uint512)),    "d_refined_factors");
    cuda_check(cudaMalloc(&bufs_.d_refined_count,   sizeof(int)),                          "d_refined_count");

    bufs_.allocated = true;
}

void SquareRootRefinement::freeDeviceBuffers() {
    if (!bufs_.allocated) return;

    cudaFree(bufs_.d_sqrt_Q);
    cudaFree(bufs_.d_sqrt_Q_mont);
    cudaFree(bufs_.d_signs);
    cudaFree(bufs_.d_val_2_exps);
    cudaFree(bufs_.d_large_primes);
    cudaFree(bufs_.d_factor_offsets);
    cudaFree(bufs_.d_factor_indices);
    cudaFree(bufs_.d_factor_counts);
    cudaFree(bufs_.d_lp_mont);
    cudaFree(bufs_.d_factor_base);
    cudaFree(bufs_.d_factor_base_mont);
    cudaFree(bufs_.d_X);
    cudaFree(bufs_.d_Y);
    cudaFree(bufs_.d_exp_matrix);
    cudaFree(bufs_.d_sign_counts);
    cudaFree(bufs_.d_exp2_accum);
    cudaFree(bufs_.d_Y_lp_mont);
    cudaFree(bufs_.d_valid);
    cudaFree(bufs_.d_partials_x);
    cudaFree(bufs_.d_partials_lp);
    cudaFree(bufs_.d_factors);
    cudaFree(bufs_.d_factor_status);
    cudaFree(bufs_.d_refined_factors);
    cudaFree(bufs_.d_refined_count);

    bufs_ = DeviceBuffers{};  // Reset all pointers to nullptr
    fb_transformed_ = false;  // Factor base must be re-transformed after reallocation
}

// -----------------------------------------------------------------------------
// Compute X (LHS Product)
// -----------------------------------------------------------------------------
mpqs::uint512 SquareRootRefinement::ComputeX(
    const std::vector<uint8_t>& mask,
    const mpqs::structures::HostRelationBatch& batch,
    mpqs::math::Montgomery& mont
) {
    LOG(LOG_DEBUG_2) << "Computing X...";
    mpqs::uint512 X_mont = mont.transform(mpqs::uint512((uint32_t)1));

    size_t n = std::min(mask.size(), batch.num_relations);
    
    for (size_t i = 0; i < n; ++i) {
        if (mask[i]) {
            // In SoA, sqrt_Q already holds |ax+b| (or the combined product)
            // We just need to accumulate it.
            mpqs::uint512 val = batch.sqrt_Q[i];
            
            // Transform to Montgomery and multiply
            mpqs::uint512 term_mont = mont.transform(val);
            X_mont = mont.mul(X_mont, term_mont);
        }
    }
    return mont.reduce(X_mont);
}

// -----------------------------------------------------------------------------
// Compute X Batched GPU — 3-phase parallel pipeline (M2 / Tile 2.4)
// -----------------------------------------------------------------------------
void SquareRootRefinement::ComputeXBatchedGPU(
    const lingen::BWKernelSolutionView& solutions,
    const mpqs::structures::HostRelationBatch& batch
) {
    const uint32_t K = static_cast<uint32_t>(batch.num_relations);
    const uint32_t n = solutions.num_solutions;

    // Ensure device buffers are allocated (lazy, no-op if already sized)
    allocateDeviceBuffers(K, static_cast<uint32_t>(batch.num_factors),
                          /*max_fb=*/0, n);

    // Upload sqrt_Q to device pool buffer (async on stream_)
    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_sqrt_Q, batch.sqrt_Q.data(),
                    K * sizeof(mpqs::uint512), cudaMemcpyHostToDevice, stream_));

    mpqs::math::Montgomery mont(N_);

    // Phase 1: Transform sqrt_Q to Montgomery form (embarrassingly parallel)
    TransformSqrtQ<<<(K + SQRT_TRANSFORM_BLOCK - 1) / SQRT_TRANSFORM_BLOCK,
                     SQRT_TRANSFORM_BLOCK, 0, stream_>>>(
        bufs_.d_sqrt_Q, bufs_.d_sqrt_Q_mont, mont, K);

    // Phase 2: Chunk-level partial products via tree reduction
    const uint32_t n_chunks = (K + SQRT_CHUNK_SIZE - 1) / SQRT_CHUNK_SIZE;
    dim3 grid2(n_chunks, n);
    size_t smem2 = SQRT_CHUNK_SIZE * sizeof(mpqs::uint512);
    ComputeX_ChunkReduce<<<grid2, SQRT_CHUNK_SIZE, smem2, stream_>>>(
        bufs_.d_sqrt_Q_mont, solutions, bufs_.d_partials_x, mont, K, SQRT_CHUNK_SIZE);

    // Phase 3: Final reduction across chunks (do_reduce=true → standard form output)
    size_t smem3 = SQRT_REDUCE_BLOCK * sizeof(mpqs::uint512);
    FinalReduce<<<n, SQRT_REDUCE_BLOCK, smem3, stream_>>>(
        bufs_.d_partials_x, bufs_.d_X, mont, n_chunks, n, /*do_reduce=*/true);

    cudaStreamSynchronize(stream_);

    // Results stay on device — accessible via getDeviceX()
}

// -----------------------------------------------------------------------------
// Compute Y Batched GPU (M3)
// -----------------------------------------------------------------------------
template <typename FBType>
void SquareRootRefinement::ComputeYBatchedGPU(
    const lingen::BWKernelSolutionView& solutions,
    const mpqs::structures::HostRelationBatch& batch,
    const std::vector<FBType>& factor_base
) {
    const uint32_t num_relations = static_cast<uint32_t>(batch.num_relations);
    const uint32_t n_solutions   = solutions.num_solutions;
    const uint32_t fb_size       = static_cast<uint32_t>(factor_base.size());
    const uint32_t nnz           = static_cast<uint32_t>(batch.num_factors);

    // Lazy buffer allocation (no-op if buffers are already large enough)
    allocateDeviceBuffers(num_relations, nnz, fb_size, n_solutions);

    // ---- 1. Upload relation SoA arrays to device (async on stream_) ----

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_signs, batch.signs.data(),
                    num_relations * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_val_2_exps, batch.val_2_exps.data(),
                    num_relations * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_large_primes, batch.large_primes.data(),
                    num_relations * sizeof(unsigned __int128), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_factor_offsets, batch.factor_offsets.data(),
                    (num_relations + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_factor_indices, batch.factor_indices.data(),
                    nnz * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_factor_counts, batch.factor_counts.data(),
                    nnz * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));

    // Convert and upload factor base as uint32_t (async on stream_)
    std::vector<uint32_t> fb_u32(fb_size);
    for (uint32_t i = 0; i < fb_size; ++i)
        fb_u32[i] = static_cast<uint32_t>(factor_base[i]);
    CUDA_CHECK(cudaMemcpyAsync(bufs_.d_factor_base, fb_u32.data(),
                    fb_size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_));

    // Pre-transform factor base to Montgomery form (once per allocation)
    if (!fb_transformed_) {
        mpqs::math::Montgomery mont_fb(N_);
        TransformFactorBase<<<(fb_size + SQRT_TRANSFORM_BLOCK - 1) / SQRT_TRANSFORM_BLOCK,
                              SQRT_TRANSFORM_BLOCK, 0, stream_>>>(
            bufs_.d_factor_base, bufs_.d_factor_base_mont, mont_fb, fb_size);
        fb_transformed_ = true;
    }

    // ---- 2. Zero-initialize accumulation and output buffers ----

    // exp_matrix stride is fb_size (not max_fb), so zero only the active portion
    const size_t exp_matrix_bytes =
        static_cast<size_t>(n_solutions) * fb_size * sizeof(uint32_t);
    cudaMemset(bufs_.d_exp_matrix, 0, exp_matrix_bytes);

    cudaMemset(bufs_.d_sign_counts, 0, n_solutions * sizeof(uint32_t));
    cudaMemset(bufs_.d_exp2_accum,  0, n_solutions * sizeof(uint32_t));

    // d_valid initialized to 1 (all valid before HalveExponents)
    cudaMemset(bufs_.d_valid, 1, n_solutions * sizeof(uint8_t));
    cudaMemset(bufs_.d_Y,     0, n_solutions * sizeof(mpqs::uint512));

    // ---- 3. Phase 1: Parallel LP product + CSR scatter + sign/exp2 ----
    mpqs::math::Montgomery mont(N_);

    const uint32_t n_chunks = (num_relations + SQRT_CHUNK_SIZE - 1) / SQRT_CHUNK_SIZE;
    const size_t smem_chunk = SQRT_CHUNK_SIZE * sizeof(mpqs::uint512);
    const size_t smem_final = SQRT_REDUCE_BLOCK * sizeof(mpqs::uint512);

    // Phase 1a — LP Product (parallel):
    //   TransformLargePrimes → ChunkReduce (over d_lp_mont) → FinalReduce (no reduce)
    TransformLargePrimes<<<(num_relations + SQRT_TRANSFORM_BLOCK - 1) / SQRT_TRANSFORM_BLOCK,
                           SQRT_TRANSFORM_BLOCK, 0, stream_>>>(
        bufs_.d_large_primes, bufs_.d_lp_mont, mont, num_relations);

    ComputeX_ChunkReduce<<<dim3(n_chunks, n_solutions), SQRT_CHUNK_SIZE, smem_chunk, stream_>>>(
        bufs_.d_lp_mont, solutions, bufs_.d_partials_lp, mont, num_relations, SQRT_CHUNK_SIZE);

    FinalReduce<<<n_solutions, SQRT_REDUCE_BLOCK, smem_final, stream_>>>(
        bufs_.d_partials_lp, bufs_.d_Y_lp_mont, mont, n_chunks, n_solutions, /*do_reduce=*/false);

    // Phase 1b — CSR Scatter + Sign/Exp2 (parallel):
    AccumExponents_Parallel<<<dim3(n_chunks, n_solutions), SQRT_CHUNK_SIZE, 0, stream_>>>(
        solutions, bufs_.d_signs, bufs_.d_val_2_exps,
        bufs_.d_factor_offsets, bufs_.d_factor_indices, bufs_.d_factor_counts,
        bufs_.d_exp_matrix, bufs_.d_sign_counts, bufs_.d_exp2_accum,
        num_relations, fb_size, SQRT_CHUNK_SIZE);

    cudaStreamSynchronize(stream_);

    // ---- 4. Phase 2: Halve exponents and assert parity ----
    constexpr uint32_t HALVE_BLOCK_SIZE = 256;
    dim3 halve_grid(
        (fb_size + HALVE_BLOCK_SIZE - 1) / HALVE_BLOCK_SIZE,
        n_solutions);
    HalveExponents<<<halve_grid, HALVE_BLOCK_SIZE, 0, stream_>>>(
        bufs_.d_exp_matrix, bufs_.d_sign_counts, bufs_.d_exp2_accum, bufs_.d_valid,
        n_solutions, fb_size
    );
    cudaStreamSynchronize(stream_);

    // ---- 5. Phase 3: Batched modular exponentiation ----
    //   One block per solution, 256 threads partition FB primes, tree-reduce.
    constexpr uint32_t EXP_THREADS_PER_BLOCK = 256;  // must be power of 2
    const size_t shared_mem = EXP_THREADS_PER_BLOCK * sizeof(mpqs::uint512);
    BatchedExponentiateY<<<n_solutions, EXP_THREADS_PER_BLOCK, shared_mem, stream_>>>(
        bufs_.d_exp_matrix, bufs_.d_sign_counts, bufs_.d_exp2_accum, bufs_.d_Y_lp_mont,
        bufs_.d_factor_base_mont, bufs_.d_valid, bufs_.d_Y, mont, N_, n_solutions, fb_size
    );
    cudaStreamSynchronize(stream_);

    // Results stay on device — accessible via getDeviceY()
}

// -----------------------------------------------------------------------------
// Compute Y (RHS Sqrt)
// -----------------------------------------------------------------------------
template <typename FBType>
mpqs::uint512 SquareRootRefinement::ComputeY(
    const std::vector<uint8_t>& mask, 
    const mpqs::structures::HostRelationBatch& batch,
    const std::vector<FBType>& factor_base,
    mpqs::math::Montgomery& mont
) {
    LOG(LOG_DEBUG_2) << "Computing Y...";
    
    // 1. Accumulate Exponents
    std::vector<uint64_t> total_exponents(factor_base.size(), 0);
    uint64_t total_exp_minus_1 = 0; 
    uint64_t total_exp_2 = 0;       
    
    // We multiply Y by the Large Primes directly as we encounter them.
    // Explanation: A combined relation has RHS = (Factors) * L^2.
    // Sqrt(RHS) = Sqrt(Factors) * L.
    mpqs::uint512 Y_large_primes_mont = mont.transform(mpqs::uint512((uint32_t)1));

    size_t n = std::min(mask.size(), batch.num_relations);

    for (size_t i = 0; i < n; ++i) {
        if (mask[i]) {
            // Sign
	    if (batch.signs[i] != (uint8_t)1) total_exp_minus_1++;
            
            // Factor 2
            total_exp_2 += batch.val_2_exps[i];

            // Large Prime (Combined Relations)
            // If large_prime > 1, it represents a square factor L^2 in the combined relation.
            // We contribute L^1 to the square root.
            unsigned __int128 lp = batch.large_primes[i];
            if (lp > 1) {
	        mpqs::uint512 lp_val(lp);
                mpqs::uint512 term = mont.transform(lp_val);
                Y_large_primes_mont = mont.mul(Y_large_primes_mont, term);
            }

            // CSR Factors
            uint64_t start = batch.factor_offsets[i];
            uint64_t end   = batch.factor_offsets[i+1];
            
            for (uint64_t k = start; k < end; ++k) {
                uint32_t fb_idx = batch.factor_indices[k];
                uint8_t  count  = batch.factor_counts[k];
                
                if (fb_idx < total_exponents.size()) {
                    total_exponents[fb_idx] += count;
                } else {
                     LOG(LOG_ERROR_CRITICAL) << "FB Index out of bounds: " << fb_idx;
                }
            }
        }
    }

    // 2. Compute Product from Exponents
    mpqs::uint512 Y_mont = mont.transform(mpqs::uint512((uint32_t)1));

    if ((total_exp_minus_1 & 1) != 0) {
        LOG(LOG_ERROR_CRITICAL) << "ERROR: Odd sign exponent: " << total_exp_minus_1;
    }
    
    if ((total_exp_2 & 1) != 0) {
        LOG(LOG_ERROR_CRITICAL) << "ERROR: Odd exponent of 2: " << total_exp_2;
    }

    // Sign (-1)
    if ((total_exp_minus_1 / 2) & 1) {
        Y_mont.negate_mod_inplace(N_);
    }

    // Factor 2
    if (total_exp_2 > 0) {
        mpqs::uint512 base = mont.transform(mpqs::uint512((uint32_t)2));
        mpqs::uint512 exp = mpqs::uint512((uint32_t)total_exp_2 / 2);
        mpqs::uint512 term = mont.pow(base, exp);
        Y_mont = mont.mul(Y_mont, term);
    }

    // Factor Base
    for (size_t j = 0; j < factor_base.size(); ++j) {
        if (total_exponents[j] == 0) continue;
	if ((total_exponents[j] & 1) != 0) {
	    LOG(LOG_ERROR_CRITICAL) << "ERROR: Odd exponent of " << j << "th prime " << factor_base[j] << ":" << total_exponents[j];
	}
        mpqs::uint512 p = mpqs::uint512((uint32_t)factor_base[j]);
        mpqs::uint512 base = mont.transform(p);
        mpqs::uint512 exp = mpqs::uint512((uint32_t)total_exponents[j] / 2);
        mpqs::uint512 term = mont.pow(base, exp);
        Y_mont = mont.mul(Y_mont, term);
    }
    
    // Merge Large Primes contribution
    Y_mont = mont.mul(Y_mont, Y_large_primes_mont);

    return mont.reduce(Y_mont);
}

// -----------------------------------------------------------------------------
// Batched GCD (M4)
// -----------------------------------------------------------------------------
std::pair<mpqs::uint512, mpqs::uint512> SquareRootRefinement::BatchedGCD(
    const mpqs::uint512* d_X,
    const mpqs::uint512* d_Y,
    int n
) {
    LOG_SET_MODULE("Sqrt");
    LOG_SET_SUBMODULE("GPU-GCD");
    // Use pool buffers (pre-allocated in DeviceBuffers)
    mpqs::uint512* d_factors = bufs_.d_factors;
    int* d_factor_status = bufs_.d_factor_status;
    cudaMemset(d_factor_status, 0, n * sizeof(int));

    // 2. Launch kernel: one thread per solution, single block
    if (n > 1024) {
        throw std::runtime_error(
            "BatchedGCDKernel: n=" + std::to_string(n) +
            " exceeds CUDA maximum thread count (1024). "
            "Block Wiedemann typically produces <=64 solutions — this indicates an unexpected solver state.");
    }
    BatchedGCDKernel<<<1, n>>>(d_X, d_Y, N_, d_factors, d_factor_status, n);

    // 3. M10: Post-factorization refinement — extract finest factor set
    constexpr size_t smem_refine = MAX_POOL * 4 * sizeof(mpqs::uint512) + 3 * sizeof(int);
    RefineFactorsKernel<<<1, REFINE_BLOCK_SIZE, smem_refine>>>(
        d_factors, d_factor_status, N_,
        bufs_.d_refined_factors, bufs_.d_refined_count, n);
    cudaDeviceSynchronize();

    // 4. Download refined factors (M10)
    int h_refined_count = 0;
    cudaMemcpy(&h_refined_count, bufs_.d_refined_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<mpqs::uint512> h_refined(h_refined_count);
    if (h_refined_count > 0) {
        cudaMemcpy(h_refined.data(), bufs_.d_refined_factors,
                   h_refined_count * sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);
    }

    // 5. Copy raw results to host (for statistics block)
    std::vector<mpqs::uint512> h_factors(n);
    std::vector<int> h_status(n);
    cudaMemcpy(h_factors.data(), d_factors,
               n * sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_status.data(), d_factor_status,
               n * sizeof(int), cudaMemcpyDeviceToHost);

    // Pool buffers — no free needed

    // ---- Statistics: non-trivial count + factorization agreement ----
    {
        int nontrivial_count = 0;
        // Collect distinct factor pairs as {min, max} for order-independent comparison
        // Use a vector of (F1, F2, count) triples; small n makes linear scan fine
        struct FactorPair {
            mpqs::uint512 small;
            mpqs::uint512 large;
            int count;
        };
        std::vector<FactorPair> distinct_pairs;

        for (int j = 0; j < n; ++j) {
            if (h_status[j] != 0) {
                nontrivial_count++;
                mpqs::uint512 f = h_factors[j];
                mpqs::uint512 cofactor = N_;
                cofactor.div(f);

                // Normalize: {smaller, larger}
                mpqs::uint512 f_small = (f < cofactor) ? f : cofactor;
                mpqs::uint512 f_large = (f < cofactor) ? cofactor : f;

                // Check if this pair is already in distinct_pairs
                bool found = false;
                for (auto& dp : distinct_pairs) {
                    if (dp.small == f_small && dp.large == f_large) {
                        dp.count++;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    distinct_pairs.push_back({f_small, f_large, 1});
                }
            }
        }

        double pct = (n > 0) ? 100.0 * nontrivial_count / n : 0.0;
        LOG(LOG_DEBUG_1) << "Non-trivial factorizations: "
                         << nontrivial_count << "/" << n
                         << " (" << std::fixed << std::setprecision(1) << pct << "%)";

        if (nontrivial_count > 0) {
            if (distinct_pairs.size() == 1) {
                // All non-trivial factorizations agree
                LOG(LOG_DEBUG_1) << "All " << nontrivial_count
                                 << " non-trivial factorizations agree: F1 = "
                                 << distinct_pairs[0].small.to_string()
                                 << ", F2 = " << distinct_pairs[0].large.to_string();

            } else {
                // Multiple distinct factorizations
                LOG(LOG_DEBUG_1) << "Found " << distinct_pairs.size()
                                 << " distinct factorizations across "
                                 << nontrivial_count << " non-trivial solutions";
                for (size_t i = 0; i < distinct_pairs.size(); ++i) {
                    LOG(LOG_DEBUG_1) << "  {" << distinct_pairs[i].small.to_string()
                                     << ", " << distinct_pairs[i].large.to_string()
                                     << "} x" << distinct_pairs[i].count;
                }
                LOG(LOG_DEBUG_1) << "Multiple distinct splits detected"
                                 << " -- cross-GCD refinement possible";
            }
        }
    }

    LOG_SET_SUBMODULE("GPU-Refine");
#ifdef MPQS_DEBUG
    // ---- M10: Refinement diagnostics (debug-only) ----
    {
        LOG(LOG_DEBUG_1) << "Refined factor count: " << h_refined_count;
        for (int i = 0; i < h_refined_count; ++i) {
            LOG(LOG_DEBUG_1) << "  F[" << i << "] = " << h_refined[i].to_string();
        }

        // Verify pairwise coprimality on host
        bool coprime_ok = true;
        for (int i = 0; i < h_refined_count && coprime_ok; ++i) {
            for (int j = i + 1; j < h_refined_count; ++j) {
                mpqs::uint512 g = mpqs::math::gcd(h_refined[i], h_refined[j]);
                if (!g.is_one()) {
                    LOG(LOG_ERROR_CRITICAL) << "Coprimality FAILED: gcd(F["
                                           << i << "], F[" << j << "]) = " << g.to_string();
                    coprime_ok = false;
                    break;
                }
            }
        }
        if (coprime_ok && h_refined_count > 1) {
            LOG(LOG_DEBUG_1) << "Pairwise coprimality verified.";
        }

        // Verify product divides N
        if (h_refined_count > 0) {
            mpqs::uint512 product((uint32_t)1);
            for (int i = 0; i < h_refined_count; ++i) {
                product.mult(h_refined[i]);
            }
            // Check product divides N: N mod product == 0
            mpqs::uint512 remainder = N_;
            remainder.mod(product);
            if (remainder.is_zero()) {
                LOG(LOG_DEBUG_1) << "Product of refined factors divides N: OK";
            } else {
                LOG(LOG_ERROR_CRITICAL) << "Product of refined factors does NOT divide N!";
            }
        }
    }
#endif // MPQS_DEBUG

    // 6. Use refined factor set (M10: pairwise coprime)
    if (h_refined_count > 0) {
        // Return the first (smallest) refined factor and its cofactor
        mpqs::uint512 factor = h_refined[0];
        mpqs::uint512 cofactor = N_;
        cofactor.div(factor);

#ifdef MPQS_DEBUG
        // Verify factor * cofactor == N — authoritative check is in orchestrator;
        // this redundant copy is retained only in debug builds.
        {
            mpqs::uint512 product = factor;
            product.mult(cofactor);
            if (product != N_) {
                LOG(LOG_ERROR_CRITICAL) << "FAILURE: factor * cofactor != N";
                // Fall through to legacy scan
            } else {
                LOG(LOG_INFO) << "Refined factor: " << factor.to_string()
                              << " (cofactor: " << cofactor.to_string() << ")";
                return {factor, cofactor};
            }
        }
#else
        LOG(LOG_INFO) << "Refined factor: " << factor.to_string()
                      << " (cofactor: " << cofactor.to_string() << ")";
        return {factor, cofactor};
#endif
    }

    // Fallback: scan raw per-solution factors (should not be needed if refinement works)
    for (int j = 0; j < n; ++j) {
        if (h_status[j] != 0) {
            mpqs::uint512 factor = h_factors[j];
            mpqs::uint512 cofactor = N_;
            cofactor.div(factor);

            mpqs::uint512 product = factor;
            product.mult(cofactor);
            if (product != N_) {
                LOG(LOG_ERROR_CRITICAL) << "FAILURE: factor * cofactor != N for solution " << j;
                continue;
            }

            const char* path = (h_status[j] == 1) ? "|X-Y|" : "X+Y";
            LOG(LOG_INFO) << "Solution " << j
                          << ": nontrivial factor via " << path << " (fallback)";
            return {factor, cofactor};
        }
    }

    LOG(LOG_WARNING) << "No nontrivial factor found across " << n << " solutions.";
    return {mpqs::uint512((uint32_t)1), N_};
}

void SquareRootRefinement::ApplyLPCorrection(
    const mpqs::uint512* d_correction_mont,
    uint32_t n_solutions
) {
    mpqs::math::Montgomery mont(N_);
    constexpr uint32_t BLOCK = 256;
    uint32_t grid = (n_solutions + BLOCK - 1) / BLOCK;
    ApplyLPCorrectionKernel<<<grid, BLOCK, 0, stream_>>>(
        bufs_.d_Y, d_correction_mont, mont, n_solutions);
    cudaStreamSynchronize(stream_);
}

bool SquareRootRefinement::SanityCheck(const mpqs::uint512& X, const mpqs::uint512& Y) {
    mpqs::uint512 X2 = mpqs::math::modpow(X, 2, N_);
    mpqs::uint512 Y2 = mpqs::math::modpow(Y, 2, N_);
    return (X2 == Y2);
}

template <typename FBType>
std::pair<mpqs::uint512, mpqs::uint512> SquareRootRefinement::Perform(
    const std::vector<uint64_t>& solution_bits,
    const mpqs::structures::HostRelationBatch& batch,
    const std::vector<FBType>& factor_base,
    const mpqs::uint512* lp_correction
) {
    LOG_SET_MODULE("Sqrt");
    // 1. Unpack solution mask
    auto mask = unpack_bits_local(solution_bits, batch.num_relations);

    // 2. Compute X and Y
    mpqs::math::Montgomery mont(N_);
    mpqs::uint512 X = ComputeX(mask, batch, mont);

#ifdef MPQS_DEBUG
    // --- M2 GPU Validation: Compare BatchedComputeX against CPU oracle (debug-only) ---
    {
        const size_t wps = (batch.num_relations + 63) / 64;  // words per solution vector
        uint64_t* d_single_sol = nullptr;
        cudaMalloc(&d_single_sol, wps * sizeof(uint64_t));
        cudaMemcpy(d_single_sol, solution_bits.data(),
                   wps * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Construct a single-solution view and run GPU kernel with n=1
        lingen::BWKernelSolutionView single_view{
            d_single_sol, 1,
            static_cast<uint32_t>(wps),
            static_cast<uint32_t>(batch.num_relations)};
        ComputeXBatchedGPU(single_view, batch);

        // Download from device to compare
        mpqs::uint512 gpu_X_val;
        cudaMemcpy(&gpu_X_val, getDeviceX(),
                   sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);

        if (gpu_X_val == X) {
            LOG(LOG_STATS) << "X MATCH: GPU agrees with CPU oracle.";
        } else {
            LOG(LOG_ERROR_CRITICAL) << "X MISMATCH!";
            LOG(LOG_ERROR_CRITICAL) << "  CPU X = " << X.to_string();
            LOG(LOG_ERROR_CRITICAL) << "  GPU X = " << gpu_X_val.to_string();
        }

        cudaFree(d_single_sol);
    }
#endif // MPQS_DEBUG

    mpqs::uint512 Y = ComputeY(mask, batch, factor_base, mont);

#ifdef MPQS_DEBUG
    // --- M3 GPU Validation: Compare BatchedComputeY against CPU oracle (debug-only) ---
    {
        const size_t wps = (batch.num_relations + 63) / 64;
        uint64_t* d_single_sol = nullptr;
        cudaMalloc(&d_single_sol, wps * sizeof(uint64_t));
        cudaMemcpy(d_single_sol, solution_bits.data(),
                   wps * sizeof(uint64_t), cudaMemcpyHostToDevice);

        lingen::BWKernelSolutionView single_view{
            d_single_sol, 1,
            static_cast<uint32_t>(wps),
            static_cast<uint32_t>(batch.num_relations)};
        ComputeYBatchedGPU(single_view, batch, factor_base);

        // Download from device to compare
        mpqs::uint512 gpu_Y_val;
        cudaMemcpy(&gpu_Y_val, getDeviceY(),
                   sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);

        if (gpu_Y_val == Y) {
            LOG(LOG_STATS) << "Y MATCH: GPU agrees with CPU oracle.";
        } else {
            LOG(LOG_ERROR_CRITICAL) << "Y MISMATCH!";
            LOG(LOG_ERROR_CRITICAL) << "  CPU Y = " << Y.to_string();
            LOG(LOG_ERROR_CRITICAL) << "  GPU Y = " << gpu_Y_val.to_string();
        }

        // Verify congruence: X² ≡ Y_gpu² (mod N)
        mpqs::uint512 X2     = mpqs::math::modpow(X, 2, N_);
        mpqs::uint512 Y2_gpu = mpqs::math::modpow(gpu_Y_val, 2, N_);
        if (X2 == Y2_gpu) {
            LOG(LOG_STATS) << "Congruence X^2 == Y^2 (mod N) verified.";
        } else {
            LOG(LOG_ERROR_CRITICAL) << "Congruence FAILED: X^2 != Y^2 (mod N)";
        }

        cudaFree(d_single_sol);
    }
#endif // MPQS_DEBUG

    // Apply precomputed LP Y-correction (expanded-matrix path)
    if (lp_correction != nullptr) {
        mpqs::uint512 Y_mont = mont.transform(Y);
        Y_mont = mont.mul(Y_mont, *lp_correction);  // Already in Montgomery domain
        Y = mont.reduce(Y_mont);
    }

    if (Y.is_zero()) {
         LOG(LOG_ERROR_MAJOR) << "Failed to compute Y (Invalid relation set).";
         return {mpqs::uint512((uint32_t)0), mpqs::uint512((uint32_t)0)};
    }

    // 3. Sanity Check
    if (!SanityCheck(X, Y)) {
        LOG(LOG_ERROR_CRITICAL) << "FATAL: X^2 != Y^2 mod N";

        static bool validation_executed = false;
        if (!validation_executed) {
            validation_executed = true;
            std::vector<uint32_t> fb_u32_validate(factor_base.size());
            for (size_t i = 0; i < factor_base.size(); ++i)
                fb_u32_validate[i] = static_cast<uint32_t>(factor_base[i]);
            mpqs::structures::RelationBatch::validate_host_batch(
                batch,
                fb_u32_validate,
                N_
            );
        }
	
        return {mpqs::uint512((uint32_t)0), mpqs::uint512((uint32_t)0)};
    }

    // 4. GCD Steps to find factors
    // We compute gcd(|X - Y|, N). 
    // Since mpqs::uint512 sub wraps on underflow (behaves as 2's complement), 
    // and we want mathematical GCD, we explicitly compute absolute difference.
    
    mpqs::uint512 diff;
    if (X < Y) {
        diff = Y; 
        diff.sub(X); // Y - X
    } else {
        diff = X; 
        diff.sub(Y); // X - Y
    }
    
    mpqs::uint512 factor1 = mpqs::math::gcd(diff, N_);

    // Factor2 = GCD(X + Y, N)
    mpqs::uint512 sum = X;
    sum.add(Y);
    mpqs::uint512 factor2 = mpqs::math::gcd(sum, N_);

    LOG(LOG_STATS) << "Candidates: " << factor1.to_string() << ", " << factor2.to_string();

    // 5. Filter Trivial Factors
    bool f1_trivial = factor1.is_one() || (factor1 == N_);
    bool f2_trivial = factor2.is_one() || (factor2 == N_);

    if (!f1_trivial) {
        mpqs::uint512 co = N_;
        co.div(factor1); // Assumes exact division
        return {factor1, co};
    }

    if (!f2_trivial) {
        mpqs::uint512 co = N_;
        co.div(factor2);
        return {factor2, co};
    }

    LOG(LOG_WARNING) << "Found only trivial factors (1, N).";
    return {mpqs::uint512((uint32_t)1), N_};
}

// =============================================================================
// Explicit Template Instantiation
// =============================================================================

// Instantiate
template std::pair<mpqs::uint512, mpqs::uint512>
mpqs::sqrt::SquareRootRefinement::Perform<uint32_t>(
    const std::vector<uint64_t>&,
    const mpqs::structures::HostRelationBatch&,
    const std::vector<uint32_t>&,
    const mpqs::uint512*
);

template void
mpqs::sqrt::SquareRootRefinement::ComputeYBatchedGPU<uint32_t>(
    const lingen::BWKernelSolutionView&,
    const mpqs::structures::HostRelationBatch&,
    const std::vector<uint32_t>&
);

} // namespace sqrt
} // namespace mpqs
