// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include <vector>
#include <cstdint>
#include <utility> // for std::pair

#include "mpqs_soa.h"        // For SOA Relation structs
#include "montgomery.cuh"    // For Montgomery Class definition
#include "uint512.cuh"
#include "math_utils.cuh"     // For sqrt{Q(X)} kernel
#include "bw_solution_view.h"

namespace mpqs {
namespace sqrt {

/// Pre-allocated device buffer pool for the sqrt GPU pipeline.
/// Allocated lazily on first GPU call, reused across invocations, freed in destructor.
struct DeviceBuffers {
    // Relation data (sized for max_K)
    uint512*             d_sqrt_Q       = nullptr;  // [max_K]
    uint512*             d_sqrt_Q_mont  = nullptr;  // [max_K] Montgomery-transformed sqrt_Q
    uint8_t*             d_signs        = nullptr;  // [max_K]
    int32_t*             d_val_2_exps   = nullptr;  // [max_K]
    unsigned __int128*   d_large_primes = nullptr;  // [max_K]
    uint64_t*            d_factor_offsets = nullptr; // [max_K + 1]
    uint32_t*            d_factor_indices = nullptr; // [max_nnz]
    uint8_t*             d_factor_counts  = nullptr; // [max_nnz]

    // Large prime Montgomery products (sized for max_K)
    uint512*             d_lp_mont      = nullptr;  // [max_K] Montgomery-transformed large primes

    // Factor base (sized for max_fb)
    uint32_t*            d_factor_base      = nullptr;  // [max_fb]
    uint512*             d_factor_base_mont = nullptr;  // [max_fb] Montgomery-transformed

    // Per-solution outputs (sized for max_n)
    uint512*             d_X            = nullptr;  // [max_n]
    uint512*             d_Y            = nullptr;  // [max_n]
    uint32_t*            d_exp_matrix   = nullptr;  // [max_n * max_fb]
    uint32_t*            d_sign_counts  = nullptr;  // [max_n]
    uint32_t*            d_exp2_accum   = nullptr;  // [max_n]
    uint512*             d_Y_lp_mont    = nullptr;  // [max_n]
    uint8_t*             d_valid        = nullptr;  // [max_n]

    // Scratch for parallel reductions
    uint512*             d_partials_x   = nullptr;  // [max_chunks * max_n] chunk partial products for ComputeX
    uint512*             d_partials_lp  = nullptr;  // [max_chunks * max_n] chunk partial products for LP reduction

    // GCD outputs
    uint512*             d_factors      = nullptr;  // [max_n]
    int*                 d_factor_status = nullptr; // [max_n]

    // Post-factorization refinement output (M10)
    uint512*  d_refined_factors = nullptr;  // [MAX_POOL=64] refined pairwise-coprime factors
    int*      d_refined_count   = nullptr;  // scalar: number of refined factors

    // Sizing metadata
    uint32_t max_K   = 0;
    uint32_t max_nnz = 0;
    uint32_t max_fb  = 0;
    uint32_t max_n   = 0;

    bool allocated = false;
};

/**
 * @brief Square Root Refinement Step for MPQS.
 * 
 * Performs the final factorization derivation:
 * 1. Takes a kernel vector (solution_bits) from the linear algebra step.
 * 2. Selects the corresponding subset of relations.
 * 3. Computes X = Product(LHS) mod N.
 * 4. Computes Y = Sqrt(Product(RHS)) mod N.
 *    - Product(RHS) is a perfect square by construction of the kernel.
 *    - Y is computed by halving the exponents of the prime factors.
 * 5. Verifies X^2 == Y^2 mod N.
 * 6. Computes gcd(X - Y, N) and gcd(X + Y, N) to find non-trivial factors.
 */
class SquareRootRefinement {
public:
    explicit SquareRootRefinement(const mpqs::uint512& n);
    ~SquareRootRefinement();

    /**
     * @brief Execute the square root step for a specific solution vector.
     * * @param solution_bits Packed bitmask from Block Wiedemann.
     * @param batch The SoA container holding ALL relations (consistent with matrix row indices).
     * @param factor_base The factor base primes.
     */
    template <typename FBType>
    std::pair<mpqs::uint512, mpqs::uint512> Perform(
        const std::vector<uint64_t>& solution_bits,
        const mpqs::structures::HostRelationBatch& batch,
        const std::vector<FBType>& factor_base,
        const mpqs::uint512* lp_correction = nullptr  // Optional: Montgomery-domain LP correction
    );

    /**
     * @brief GPU-accelerated batched X computation for all solutions simultaneously.
     *
     * Computes X[j] = Π sqrt_Q[i] (mod N) for all i where solution j selects relation i,
     * for j in [0, num_solutions). Results stay on device in bufs_.d_X — retrieve via
     * getDeviceX(). Uses the BatchedComputeX CUDA kernel with the "read once, broadcast
     * to n" paradigm.
     *
     * @param solutions  Device-resident packed solution bit-matrix (kernel-passable view).
     * @param batch      Host-side relation data (sqrt_Q uploaded to device internally).
     */
    void ComputeXBatchedGPU(
        const lingen::BWKernelSolutionView& solutions,
        const mpqs::structures::HostRelationBatch& batch
    );

    /// Device pointer to X results (valid after ComputeXBatchedGPU returns)
    const mpqs::uint512* getDeviceX() const { return bufs_.d_X; }

    /**
     * @brief GPU-accelerated batched Y computation for all solutions simultaneously.
     *
     * Three-phase algorithm:
     *   Phase 1 — Accumulate sign, power-of-2, CSR factor, and large-prime contributions
     *             (one thread per solution, iterating all relations).
     *   Phase 2 — Assert parity of all accumulated exponents and halve them.
     *   Phase 3 — Compute Y[j] = (-1)^s · 2^e₂ · Π pᵢ^(eᵢ/2) · LP (mod N)
     *             via Montgomery modular exponentiation with intra-block parallel
     *             reduction (one block per solution, threads partition FB primes).
     *
     * Results stay on device in bufs_.d_Y — retrieve via getDeviceY().
     *
     * @param solutions   Device-resident packed solution bit-matrix (kernel-passable view).
     * @param batch       Host-side relation data (uploaded to device internally).
     * @param factor_base Factor base primes.
     */
    template <typename FBType>
    void ComputeYBatchedGPU(
        const lingen::BWKernelSolutionView& solutions,
        const mpqs::structures::HostRelationBatch& batch,
        const std::vector<FBType>& factor_base
    );

    /// Device pointer to Y results (valid after ComputeYBatchedGPU returns)
    const mpqs::uint512* getDeviceY() const { return bufs_.d_Y; }

    /// Device pointer to per-solution validity flags (valid after ComputeYBatchedGPU returns).
    /// Entry j is 0 if HalveExponents detected an odd accumulated exponent for solution j
    /// (meaning Y[j] is zeroed out and the congruence X²≡Y² cannot hold).
    const uint8_t* getDeviceValid() const { return bufs_.d_valid; }

    /**
     * @brief Batched GPU factor extraction via binary GCD.
     *
     * Given n (X, Y) pairs on the device (from BatchedComputeX/Y),
     * computes gcd(|X[j]-Y[j]|, N) and gcd(X[j]+Y[j], N) for all j,
     * returns the first nontrivial factor pair {F1, F2} where F1*F2 = N.
     *
     * @param d_X Device pointer to n X values (standard form, from M2).
     * @param d_Y Device pointer to n Y values (standard form, from M3).
     * @param n   Number of solutions.
     * @return {F1, F2} if a nontrivial factor is found, {1, N} otherwise.
     */
    std::pair<mpqs::uint512, mpqs::uint512> BatchedGCD(
        const mpqs::uint512* d_X,
        const mpqs::uint512* d_Y,
        int n
    );

    /// Apply precomputed LP Y-correction to device Y values (expanded-matrix path).
    /// d_correction_mont points to n_solutions uint512 values in Montgomery domain.
    /// Must be called AFTER ComputeYBatchedGPU and BEFORE BatchedGCD.
    void ApplyLPCorrection(const mpqs::uint512* d_correction_mont, uint32_t n_solutions);

    /// Pre-allocate device buffer pool. Called automatically by ComputeX/ComputeY, but may be
    /// called explicitly with complete sizing to avoid mid-pipeline reallocation.
    void allocateDeviceBuffers(uint32_t max_K, uint32_t max_nnz,
                               uint32_t max_fb, uint32_t max_n);

private:
    mpqs::uint512 ComputeX(
        const std::vector<uint8_t>& mask,
        const mpqs::structures::HostRelationBatch& batch,
        mpqs::math::Montgomery& mont
    );

    template <typename FBType>
    mpqs::uint512 ComputeY(
        const std::vector<uint8_t>& mask,
        const mpqs::structures::HostRelationBatch& batch,
        const std::vector<FBType>& factor_base,
        mpqs::math::Montgomery& mont
    );

    bool SanityCheck(const mpqs::uint512& X, const mpqs::uint512& Y);

    mpqs::uint512 N_;

    DeviceBuffers bufs_;
    cudaStream_t stream_ = nullptr;  // Async stream for H2D transfers and kernel launches
    bool fb_transformed_ = false;  // true after TransformFactorBase ran for current allocation
    void freeDeviceBuffers();
};

} // namespace sqrt
} // namespace mpqs
