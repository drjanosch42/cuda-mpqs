// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <vector>
#include <deque>
#include <set>
#include <cstdint>
#include <map>
#include <cuda_runtime.h>
#include "bw_spmm_interface.h"
#include "lingen/operations/bw_operator.h"
#include "bw_solver.h"

namespace lingen {
namespace stage3 {

struct OracleEntry {
    uint64_t hash_horner; // Hash of w after Horner evaluation (before stripping)
    uint64_t hash_final;  // Hash of w after valuation stripping (final solution candidate)
};

class SolutionReconstructor {
public:
    /**
     * @brief Construct a new Solution Reconstructor.
     * 
     * @param op The linear operator A or A^T.
     * @param nrows Number of rows in the matrix.
     * @param m_block The left block size (m).
     * @param n_block The right block size (n).
     * @param use_transpose If true, reconstructs for kernel of A^T.
     * @param batch_mode If true, enables parallelized batch execution logic.
     */
     SolutionReconstructor(const lingen::BwOperator& op, const BWStage3Config& config);
  
    ~SolutionReconstructor();

    /**
     * @brief Initializes the search state.
     * 
     * Uploads the starting block Z (hY) to GPU.
     * Analyzes the generator matrix Pi to identify and rank candidate columns.
     * If batch_mode is enabled, constructs the batch queue.
     * 
     * @param hY The starting block Z (from Stage 1).
     * @param hPi The generator polynomial matrix (from Stage 2).
     * @param stream CUDA stream.
     */
    void Init(const std::vector<uint64_t>& hY,
              const std::vector<uint64_t>& hPi,
              cudaStream_t stream);

    /**
     * @brief Searches for the next linearly independent solution.
     * 
     * Resumes the search from the last checked candidate.
     * In batch mode, this may trigger the processing of a new batch of candidates.
     * 
     * @param out_solution Output buffer for the found vector (if any).
     * @param stream CUDA stream.
     * @return true if a new independent solution was found.
     * @return false if the search space is exhausted.
     */
    bool FindNext(std::vector<uint64_t>& out_solution, cudaStream_t stream);

    /**
     * @brief Returns the number of linearly independent solutions found so far.
     */
    size_t GetBasisSize() const { return basis_struct_.size(); }

    /**
     * @brief CPU Oracle Validation Interface
     */
    std::map<int, OracleEntry> GetOracleData() const { return recorded_data_; }
    void SetOracleData(const std::map<int, OracleEntry>& data) { oracle_data_ = data; }

    void VerifyAgainstOracle(int candidate_col_idx, 
                             const uint64_t* d_w_horner, 
                             const uint64_t* d_w_final,
                             cudaStream_t stream);
  
private:
    struct Candidate {
        int degree;
        int col_idx;
        bool operator<(const Candidate& other) const {
            if (degree != other.degree) return degree < other.degree;
            return col_idx < other.col_idx;
        }
    };

    struct BatchCandidate {
        std::vector<Candidate> candidates;
        int max_degree;
    };

    // Execution Paths
    bool FindNextLegacy(std::vector<uint64_t>& out_solution, cudaStream_t stream);
    bool FindNextBatch(std::vector<uint64_t>& out_solution, cudaStream_t stream);
    void RunBatchStripping(const BatchCandidate& batch, cudaStream_t stream);

    // Core Logic
    bool ProcessCandidate(const Candidate& cand, std::vector<uint64_t>& out_solution, cudaStream_t stream);
    bool ProcessFoundSolution(const Candidate& cand, uint64_t* d_sol_vec, cudaStream_t stream);

    static void construct_batch_matrix(
        std::vector<uint64_t>& out_U, 
        int n_block, 
        const std::vector<uint64_t>& hPi, 
        int M, 
        int k, 
        const std::vector<Candidate>& batch
    );

    bool strip_valuation(uint64_t* d_w, uint64_t* d_tmp, int max_steps, cudaStream_t stream);
    bool is_zero_vector(const uint64_t* d_ptr, cudaStream_t stream);
    //void apply_operator(void* d_out, const void* d_in);
    uint64_t hash_device_vector(const uint64_t* d_ptr, cudaStream_t stream);

    struct BasisVector {
        int pivot; 
        std::vector<uint64_t> vec;
    };
    bool insert_into_basis(std::vector<BasisVector>& basis, std::vector<uint64_t>& candidate_vec);

    BWStage3Config config_;
    const BwOperator& op_; //formerly BlockWiedemannSpMM& spmm_;

    // Redundant local copies
    int nrows_;
    int nrows_aligned_; // Aligned row count for safe block operations
    int m_block_;
    int n_block_;
    bool use_transpose_;
    bool batch_mode_;
    
    // Oracle State
    std::map<int, OracleEntry> recorded_data_;
    std::map<int, OracleEntry> oracle_data_;
    std::set<int> missing_candidates_;
    
    // Generator Data
    const std::vector<uint64_t>* ptr_hPi_ = nullptr;
    int M_ = 0;

    // Queues
    std::vector<Candidate> candidate_queue_;
    size_t queue_cursor_ = 0;
    std::deque<BatchCandidate> batch_queue_;
    std::deque<std::vector<uint64_t>> solution_buffer_;
    std::vector<BasisVector> basis_struct_;

    // Device buffers
    uint64_t* d_Z_ = nullptr;      
    uint64_t* d_W_acc_ = nullptr;   
    uint64_t* d_W_batch_ = nullptr; 
    uint64_t* d_Tmp_ = nullptr; 
    uint64_t* d_u_vec_ = nullptr;   
    
    uint64_t* d_U_small_ = nullptr;    
    uint64_t* d_U_expanded_ = nullptr; 
    uint64_t* d_MatMul_Res_ = nullptr; 
    uint64_t* d_extracted_sol_ = nullptr; 
    
    // History & Stripping buffers
    uint64_t* d_history_ = nullptr;
    int* d_col_status_ = nullptr; 
    int* d_found_idx_ = nullptr;  
    
    int history_depth_ = 64;
    int check_interval_ = 16;

    std::vector<uint64_t> h_chk_buf_;

#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010
    // CUDA Graph state for Horner projection chain
    cudaGraph_t       proj_graph_      = nullptr;
    cudaGraphExec_t   proj_graph_exec_ = nullptr;
    bool              proj_graph_captured_ = false;

    void destroy_proj_graph();
#endif
#endif
};

} // namespace stage3
} // namespace lingen
