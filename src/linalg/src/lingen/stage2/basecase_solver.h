// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "lingen/types.h"
#include "bw_solver.h"

namespace lingen {
namespace stage2 {

/**
 * @brief CPU/GPU Basecase Solver for Block Wiedemann (Coppersmith/Thomé).
 * 
 * DESIGN NOTE: HYBRID ARCHITECTURE FOR HPC
 * 
 * To achieve high performance while guaranteeing bitwise equivalence to the reference 
 * (Python/Legacy) implementation, this solver adopts a hybrid approach:
 * 
 * 1. ORACLE DECISIONS (CPU Resident):
 *    - Shift Selection: Determining which rows to shift (x -> x*z).
 *      Reason: Depends on gamma values and exact pivot choices.
 *    - Gamma Updates: Simple integer increments based on shift vector.
 * 
 * 2. BULK TRANSFORMS (GPU Offloadable):
 *    - Discrepancy Computation: Delta = F(x) * S(x)^T.
 *    - Elimination: Gaussian elimination on Delta to produce Tau and ReducedDelta.
 *      Operation: Pivot search (stable sort) + Row operations (XOR).
 *    - Generator Update: F_next(x) = Tau * F(x) (with shifts).
 * 
 * 3. VERIFICATION STRATEGY:
 *    - The solver supports dual execution paths.
 *    - Critical checkpoints (Pi, Gamma) are hashed to ensure the optimized path 
 *      never diverges from the canonical logic.
 */
class BasecaseSolver {
public:
    /**
     * @brief Constructs the solver using the Stage 2 configuration.
     * 
     * @param config A populated BWStage2Config structure.
     */
    explicit BasecaseSolver(const BWStage2Config& config);
  
    /* Destructor */
    ~BasecaseSolver();

    /**
     * @brief Main Entry Point for Generic Solver.
     * 
     * @param d_S Device pointer to the flat sequence S.
     *            Layout: contiguous S_0, S_1, ..., S_{len-1}.
     *            Each S_k is an m x n row-major bit-packed matrix.
     * @param stream CUDA stream for downloads (and offload in hybrid mode).
     * 
     * Verification flags and length are now taken from the config passed at construction.
     */
    void solve(const uint64_t* d_S, cudaStream_t stream);  

    const std::vector<uint64_t>& get_pi_host() const { return pi_data_; }
    const std::vector<int>& get_gamma() const { return gamma_; }

    /**
     * @brief Computes a deterministic hash of the current solver state (Pi and Gamma).
     * Used for regression testing and side-by-side verification.
     */
    uint64_t compute_state_hash() const;

private:
    // Stored configuration
    BWStage2Config config_;

    //int m_;     ///< Rows of S (number of X vectors)
    //int n_;     ///< Cols of S (number of Z vectors)
    int dim_;   ///< Dimension of Generator Matrix Pi (m + n)
    int delta_;
    int cached_elim_max_blocks_per_SM_ = -1; ///< Cached occupancy for k_elimination kernel
    int cached_shifts_max_blocks_per_SM_ = -1; ///< Cached occupancy for k_compute_shifts kernel
    // ExecutionMode mode_;

    std::vector<uint64_t> pi_data_;
    std::vector<int> gamma_;

    // Internal Helpers
    bool get_bit(const uint64_t* data, int rows, int cols, int r, int c) const;
    void xor_bit(uint64_t* data, int rows, int cols, int r, int c);
    void set_bit(uint64_t* data, int rows, int cols, int r, int c, bool val);

    bool check_annihilation(const uint64_t* d_S, int len_S, cudaStream_t stream);
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

    // --- GPU Discrepancy computation ---
    // Computes directly to device buffer for GPU internal pipeline
    void compute_discrepancy_gpu_to_buffer(
        PackedBitPolyView F_view,
        const uint64_t* d_S,
        int t,
        uint64_t* d_Delta_out,
	cudaStream_t stream = 0
    );

    // --- Generalized Elimination & Shift ---
    struct StepResult {
        std::vector<uint64_t> tau;           // dim x dim (packed)
        std::vector<uint64_t> reduced_delta; // dim x m   (packed)
    };

    // CPU implementation
    StepResult compute_elimination_step(const std::vector<uint64_t>& delta);

    // --- GPU Elimination ---
    // GPU Elimination: Computes Tau and ReducedDelta on device buffers.
    // NOTE: Does NOT download results. Callers must copy d_ReducedDelta/d_Tau if needed.
    void compute_elimination_step_gpu(
        const uint64_t* d_Delta,
        const int* d_Gamma,
        uint64_t* d_Tau,
        uint64_t* d_ReducedDelta,
        cudaStream_t stream
    );

    // GPU Shift Selection
    void compute_shift_vector_gpu(
        const uint64_t* d_ReducedDelta,
        const int* d_Gamma,
        int* d_Shifts, // Output: 0 or 1 per row
        cudaStream_t stream
    );

    // GPU Gamma Update
    void update_gamma_gpu(
        int* d_Gamma,
        const int* d_Shifts,
        cudaStream_t stream
    );  
  
    std::vector<bool> compute_shift_vector(const std::vector<uint64_t>& reduced_delta);
    
    std::vector<std::vector<uint64_t>> apply_update(
        const std::vector<std::vector<uint64_t>>& F_poly,
        const std::vector<uint64_t>& tau,
        const std::vector<bool>& will_shift
    );
};

} // namespace stage2
} // namespace lingen
