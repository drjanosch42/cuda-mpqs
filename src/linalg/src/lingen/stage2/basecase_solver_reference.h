// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
/**
 * @brief CPU-based Basecase Reference Solver for Block Wiedemann (Coppersmith/Thomé).
 * 
 * Implements the iterative Block Berlekamp-Massey algorithm.
 * Validated against reference Python implementation.
 * 
 * REFERENCE ORACLE STATUS:
 * This class acts as the ground truth for the square N=64 case.
 * Do not modify the pivoting, shifting, or initialization logic without 
 * ensuring it remains bit-identical to the Python verification script
 * for m=n=64 and all other choices m,n in 32, 64, 128, 256, 512.
 */

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "lingen/types.h"

namespace lingen {
namespace stage2 {

/**
 * @brief CPU-based Basecase Solver for Block Wiedemann (Coppersmith/Thomé).
 * 
 * Implements the iterative Block Berlekamp-Massey algorithm.
 * Validated against reference Python implementation.
 * 
 * REFERENCE ORACLE STATUS:
 * This class acts as the ground truth for all cases.
 * Do not modify the pivoting, shifting, or initialization logic without 
 * ensuring it remains bit-identical to the Python verification script
 * for m=n=64 and all other choices m,n in 64, 128, 256, 512.
 */
class BasecaseSolverLegacy {
public:
    /**
     * @brief Constructs the legacy solver for the rectangular case (m x n sequence).
     * 
     * @param m Number of rows in the sequence matrices S_k (from X projection).
     * @param n Number of columns in the sequence matrices S_k (from Z projection).
     * @param delta Algorithm parameter delta (usually L/2).
     * 
     * Note: The generator matrix Pi will have dimension dim = m + n.
     */
    BasecaseSolverLegacy(int m, int n, int delta);
  
    /**
     * @brief Legacy wrapper for square sequences (m = n = block_dim).
     * @param block_dim The block size n.
     * @param delta Algorithm parameter.
     */  
    BasecaseSolverLegacy(int block_dim, int delta);

    /* Destructor */
    ~BasecaseSolverLegacy();

    /**
     * @brief Main Entry Point for Generic Legacy Solver.
     * 
     * @param d_S Device pointer to the flat sequence S.
     *            Layout: contiguous S_0, S_1, ..., S_{len-1}.
     *            Each S_k is an m x n row-major bit-packed matrix.
     * @param len Sequence length (number of matrices).
     * @param stream CUDA stream for downloads.
     * @param verify_annihilation_gpu Enable optimized GPU-based annihilation check.
     * @param verify_annihilation_legacy Enable legacy CPU-based annihilation check.
     */
    void solve(const uint64_t* d_S, int len, cudaStream_t stream, 
               bool verify_annihilation_gpu = true, bool verify_annihilation_legacy = false);

    const std::vector<uint64_t>& get_pi_host() const { return pi_data_; }
    const std::vector<int>& get_gamma() const { return gamma_; }

private:
    int m_;     ///< Rows of S (number of X vectors)
    int n_;     ///< Cols of S (number of Z vectors)
    int dim_;   ///< Dimension of Generator Matrix Pi (m + n)  
    int delta_;

    std::vector<uint64_t> pi_data_;
    std::vector<int> gamma_;

    bool get_bit(const uint64_t* data, int rows, int cols, int r, int c) const;
    void xor_bit(uint64_t* data, int rows, int cols, int r, int c);
    void set_bit(uint64_t* data, int rows, int cols, int r, int c, bool val);

    bool check_annihilation(const uint64_t* d_S, int len_S, cudaStream_t stream);
    // Legacy annihilation check
    bool check_annihilation_legacy(const std::vector<std::vector<uint64_t>>& S, 
                                   const std::vector<std::vector<uint64_t>>& Pi);
                            
    void log_matrix_snippet(const std::string& label, const std::vector<uint64_t>& mat, int rows, int cols, int r_lim=4, int c_lim=4);

    // --- General Initialization Helpers ---
    struct InitResult {
        int t0;
        std::vector<std::pair<int, int>> basis_pairs;
    };

    struct FInitResult {
        std::vector<std::vector<uint64_t>> F_poly;
        std::vector<int> gamma;
    };

    InitResult find_initialization_basis(const std::vector<std::vector<uint64_t>>& S_host);
    FInitResult build_f_init(const InitResult& init);

    // --- General Discrepancy computation ---
    std::vector<uint64_t> compute_discrepancy(
        const std::vector<std::vector<uint64_t>>& F_poly,
        const std::vector<std::vector<uint64_t>>& S_host,
        int t
    );

    // --- Generalized Elimination & Shift ---
    struct StepResult {
        std::vector<uint64_t> tau;           // dim x dim (packed)
        std::vector<uint64_t> reduced_delta; // dim x m   (packed)
    };

    StepResult compute_elimination_step(const std::vector<uint64_t>& delta);
    std::vector<bool> compute_shift_vector(const std::vector<uint64_t>& reduced_delta);
    
    std::vector<std::vector<uint64_t>> apply_update(
        const std::vector<std::vector<uint64_t>>& F_poly,
        const std::vector<uint64_t>& tau,
        const std::vector<bool>& will_shift
    );
};

} // namespace stage2
} // namespace lingen
