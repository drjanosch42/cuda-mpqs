// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "lingen/operations/bw_operator.h"

namespace lingen {
namespace stage1 {

/**
 * @brief Generates the Block Krylov Sequence S_k = X^T * B^(k+1) * Z.
 * 
 * Optimized GPU Implementation:
 *  - Double-buffered pipeline.
 *  - Batched Projection Kernel (reuses X for multiple V).
 *  - Maximizes GPU Compute / Copy overlap.
 */
class KrylovSequenceGenerator {
public:
    /**
     * @brief Construct a new Krylov Sequence Generator.
     * 
     * @param op       The linear operator B.
     * @param m_block  Blocking factor for X (rows of S).
     * @param n_block  Blocking factor for Z (cols of S).
     * @param batch_size Circular buffer size (default 64).
     */
    KrylovSequenceGenerator(const lingen::BwOperator& op, int m_block, int n_block,
                            int batch_size = 64, bool transpose_output = false,
                            bool use_cuda_graph = false);
    ~KrylovSequenceGenerator();

    /**
     * @brief Execute the generation loop.
     */
    void generate(
        int length,
        const std::vector<uint64_t>& h_X,
        const std::vector<uint64_t>& h_Z,
        std::vector<uint64_t>& out_seq,
        cudaStream_t stream = 0,
        uint64_t* d_S_target = nullptr
    );

private:
    const BwOperator& op_;
    int m_block_;
    int n_block_;
    int batch_size_;
    bool use_cuda_graph_ = false;

    size_t dim_rows_;
    size_t x_bytes_;       // Size of X
    size_t v_bytes_;       // Size of one vector V
    size_t s_mat_bytes_;   // Size of one result matrix S
    bool transpose_output_ = false;
    size_t s_transposed_mat_bytes_;   // n_block_ × ceil(m_block_/64) × 8
    size_t s_output_mat_bytes_;       // whichever layout reaches the host

    // -- GPU Memory --
    uint64_t* dX_ = nullptr;
    uint64_t* dV_ = nullptr; // State Vector V
    uint64_t* dW_ = nullptr; // Temp Vector W

    // -- Double Buffering --
    // We use 2 sets of buffers. Set 0 is filling while Set 1 is downloading.
    // 1. V-Buffer: Holds 'batch_size' vectors V. Used for batch projection.
    uint64_t* d_V_buf_[2] = {nullptr, nullptr};
    
    // 2a. S-Buffer: Holds 'batch_size' projection matrices S.
    uint64_t* d_S_buf_[2] = {nullptr, nullptr};

    // 2b. S-Transposed Buffer: Holds transposed projection matrices (Feature A).
    uint64_t* d_S_transposed_buf_[2] = {nullptr, nullptr};
    
    // 3. Host Buffer: Pinned memory for downloads.
    uint64_t* h_S_buf_[2] = {nullptr, nullptr};

    // -- Streams & Events --
    cudaStream_t compute_stream_;
    cudaStream_t copy_stream_;
    cudaEvent_t compute_done_[2]; // Signaled when Compute on Buf[i] is done
    cudaEvent_t copy_done_[2];    // Signaled when Copy on Buf[i] is done

    // Helper to dispatch the correct template kernel
    void dispatch_batch_proj(int count, int buf_idx);

#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010
    // -- Phase 1: Inner loop graph (2-iteration unrolled) --
    cudaGraph_t     inner_loop_graph_ = nullptr;
    cudaGraphExec_t inner_loop_exec_  = nullptr;
    bool            inner_loop_captured_ = false;

    void capture_inner_loop_graph();
    void destroy_inner_loop_graph();

    // -- Phase 2: Batch tail graph (one per buffer index) --
    cudaGraph_t     batch_tail_graph_[2] = {};
    cudaGraphExec_t batch_tail_exec_[2]  = {};
    bool            batch_tail_captured_[2] = {false, false};

    void capture_batch_tail(int count, int buf_idx);
    void destroy_batch_tail_graphs();
#endif
#endif
};

} // namespace stage1
} // namespace lingen
