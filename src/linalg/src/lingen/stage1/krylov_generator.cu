// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/stage1/krylov_generator.h"
#include "lingen/stage1/krylov_kernels.h"
#include "hpc_logger.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace lingen {
namespace stage1 {

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "CUDA Error: " << cudaGetErrorString(err) << " at " __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)
#endif

KrylovSequenceGenerator::KrylovSequenceGenerator(const lingen::BwOperator& op,
						 int m_block, int n_block,
						 int batch_size, bool transpose_output,
						 bool use_cuda_graph)
    : op_(op), m_block_(m_block), n_block_(n_block), batch_size_(batch_size),
      use_cuda_graph_(use_cuda_graph), transpose_output_(transpose_output)
{
    dim_rows_ = op_.dim_bits();

    size_t x_stride = (m_block_ + 63) / 64;
    size_t v_stride = (n_block_ + 63) / 64;

    x_bytes_ = dim_rows_ * x_stride * sizeof(uint64_t);
    v_bytes_ = dim_rows_ * v_stride * sizeof(uint64_t);
    s_mat_bytes_ = m_block_ * v_stride * sizeof(uint64_t);

    // Transpose-aware sizes
    s_transposed_mat_bytes_ = n_block_ * ((m_block_ + 63) / 64) * sizeof(uint64_t);
    s_output_mat_bytes_ = transpose_output_ ? s_transposed_mat_bytes_ : s_mat_bytes_;

    // Allocations
    CHECK_CUDA(cudaMalloc(&dX_, x_bytes_));
    CHECK_CUDA(cudaMalloc(&dV_, v_bytes_));
    CHECK_CUDA(cudaMalloc(&dW_, v_bytes_));

    size_t v_buf_size = (size_t)batch_size_ * v_bytes_;
    size_t s_buf_size = (size_t)batch_size_ * s_mat_bytes_;

    for(int i=0; i<2; ++i) {
        CHECK_CUDA(cudaMalloc(&d_V_buf_[i], v_buf_size));
        CHECK_CUDA(cudaMalloc(&d_S_buf_[i], s_buf_size));
        CHECK_CUDA(cudaMallocHost(&h_S_buf_[i], (size_t)batch_size_ * s_output_mat_bytes_));
        CHECK_CUDA(cudaEventCreate(&compute_done_[i]));
        CHECK_CUDA(cudaEventCreate(&copy_done_[i]));
    }

    // Allocate transpose device buffers (only when needed)
    if (transpose_output_) {
        size_t s_trans_buf_size = (size_t)batch_size_ * s_transposed_mat_bytes_;
        for (int i = 0; i < 2; ++i)
            CHECK_CUDA(cudaMalloc(&d_S_transposed_buf_[i], s_trans_buf_size));
    }

    CHECK_CUDA(cudaStreamCreate(&compute_stream_));
    CHECK_CUDA(cudaStreamCreate(&copy_stream_));

    // Clear initial state
    CHECK_CUDA(cudaMemset(dX_, 0, x_bytes_));
    CHECK_CUDA(cudaMemset(dV_, 0, v_bytes_));
    CHECK_CUDA(cudaMemset(dW_, 0, v_bytes_));

    double total_mb = (x_bytes_ + 2*v_bytes_ + 2*v_buf_size + 2*s_buf_size) / (1024.0 * 1024.0);
    if (transpose_output_)
        total_mb += 2.0 * batch_size_ * s_transposed_mat_bytes_ / (1024.0 * 1024.0);
    LOG(LOG_STATS) << "[KrylovGen] Init: M=" << m_block_ << " N=" << n_block_
                  << " Batch=" << batch_size_
                  << (transpose_output_ ? " (transpose ON)" : "");
    LOG(LOG_STATS) << "[KrylovGen] GPU Memory: " << total_mb << " MB";
}

KrylovSequenceGenerator::~KrylovSequenceGenerator() {
#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010
    destroy_inner_loop_graph();
    destroy_batch_tail_graphs();
#endif
#endif
    cudaFree(dX_);
    cudaFree(dV_);
    cudaFree(dW_);
    for(int i=0; i<2; ++i) {
        cudaFree(d_V_buf_[i]);
        cudaFree(d_S_buf_[i]);
        if (d_S_transposed_buf_[i]) cudaFree(d_S_transposed_buf_[i]);
        cudaFreeHost(h_S_buf_[i]);
        cudaEventDestroy(compute_done_[i]);
        cudaEventDestroy(copy_done_[i]);
    }
    cudaStreamDestroy(compute_stream_);
    cudaStreamDestroy(copy_stream_);
}

void KrylovSequenceGenerator::dispatch_batch_proj(int count, int buf_idx) {
    // Select Kernel based on M, N
    // This dispatcher covers the required combinations
    // We assume M, N are one of 64, 128, 256, 512.
    
    #define LAUNCH_KERNEL(M, N) \
        launch_krylov_batch_proj<M, N>( \
            dim_rows_, count, dX_, d_V_buf_[buf_idx], v_bytes_, \
            d_S_buf_[buf_idx], s_mat_bytes_, compute_stream_)

    if (m_block_ == 64 && n_block_ == 64) { LAUNCH_KERNEL(64, 64); }
    else if (m_block_ == 64 && n_block_ == 128) { LAUNCH_KERNEL(64, 128); }
    else if (m_block_ == 64 && n_block_ == 256) { LAUNCH_KERNEL(64, 256); }
    else if (m_block_ == 64 && n_block_ == 256) { LAUNCH_KERNEL(64, 512); }
    else if (m_block_ == 128 && n_block_ == 64) { LAUNCH_KERNEL(128, 64); }
    else if (m_block_ == 128 && n_block_ == 128) { LAUNCH_KERNEL(128, 128); }
    else if (m_block_ == 128 && n_block_ == 256) { LAUNCH_KERNEL(128, 256); }
    else if (m_block_ == 128 && n_block_ == 256) { LAUNCH_KERNEL(128, 512); }
    else if (m_block_ == 256 && n_block_ == 64) { LAUNCH_KERNEL(256, 64); }
    else if (m_block_ == 256 && n_block_ == 128) { LAUNCH_KERNEL(256, 128); }
    else if (m_block_ == 256 && n_block_ == 256) { LAUNCH_KERNEL(256, 256); }
    else if (m_block_ == 256 && n_block_ == 256) { LAUNCH_KERNEL(256, 512); }
    else if (m_block_ == 512 && n_block_ == 64) { LAUNCH_KERNEL(512, 64); }
    else if (m_block_ == 512 && n_block_ == 128) { LAUNCH_KERNEL(512, 128); }
    else if (m_block_ == 512 && n_block_ == 256) { LAUNCH_KERNEL(512, 256); }
    else if (m_block_ == 512 && n_block_ == 512) { LAUNCH_KERNEL(512, 512); }
    else {
        LOG(LOG_ERROR_CRITICAL) << "[KrylovGen] Uninstantiated Kernel M=" << m_block_ << " N=" << n_block_;
        throw std::runtime_error("KrylovGen: uninstantiated kernel for given M/N block sizes");
    }
    #undef LAUNCH_KERNEL
}

// =============================================================================
// CUDA Graph Capture/Destroy Methods (Phases 1 & 2)
// =============================================================================

#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010

void KrylovSequenceGenerator::capture_inner_loop_graph() {
    cudaError_t err = cudaStreamBeginCapture(compute_stream_,
                                              cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        LOG(LOG_WARNING) << "[KrylovGen] Graph capture begin failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        use_cuda_graph_ = false;
        return;
    }

    // --- Even iteration: V is input, W is output ---
    CHECK_CUDA(cudaMemsetAsync(dW_, 0, v_bytes_, compute_stream_));
    op_.mul(dW_, dV_, compute_stream_);

    // --- Odd iteration: W is input, V is output ---
    CHECK_CUDA(cudaMemsetAsync(dV_, 0, v_bytes_, compute_stream_));
    op_.mul(dV_, dW_, compute_stream_);

    cudaGraph_t graph;
    err = cudaStreamEndCapture(compute_stream_, &graph);
    if (err != cudaSuccess || graph == nullptr) {
        LOG(LOG_WARNING) << "[KrylovGen] Inner loop graph capture failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        use_cuda_graph_ = false;
        return;
    }

    err = cudaGraphInstantiate(&inner_loop_exec_, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        LOG(LOG_WARNING) << "[KrylovGen] Inner loop graph instantiation failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        inner_loop_exec_ = nullptr;
        use_cuda_graph_ = false;
        return;
    }

    inner_loop_captured_ = true;
    LOG(LOG_DEBUG_2) << "[KrylovGen] Inner loop graph captured (2-iteration unroll)";
}

void KrylovSequenceGenerator::destroy_inner_loop_graph() {
    if (inner_loop_captured_) {
        cudaGraphExecDestroy(inner_loop_exec_);
        inner_loop_captured_ = false;
        inner_loop_exec_ = nullptr;
    }
}

void KrylovSequenceGenerator::capture_batch_tail(int count, int buf_idx) {
    cudaError_t err = cudaStreamBeginCapture(compute_stream_,
                                              cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        LOG(LOG_WARNING) << "[KrylovGen] Batch tail graph capture begin failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        use_cuda_graph_ = false;
        return;
    }

    // Op #1: clear S buffer
    CHECK_CUDA(cudaMemsetAsync(d_S_buf_[buf_idx], 0,
                                count * s_mat_bytes_, compute_stream_));

    // Op #2: batch projection kernel
    dispatch_batch_proj(count, buf_idx);

    // Op #3: transpose (conditional — captured only if transpose_output_)
    if (transpose_output_) {
        size_t src_stride_words = (n_block_ + 63) / 64;
        size_t dst_stride_words = (m_block_ + 63) / 64;
        launch_transpose_batch(
            d_S_buf_[buf_idx], d_S_transposed_buf_[buf_idx],
            count, m_block_, n_block_,
            src_stride_words, dst_stride_words,
            compute_stream_);
    }

    // NOTE: D2D copy to d_S_target is NOT in the graph — offset varies per batch.

    cudaGraph_t graph;
    err = cudaStreamEndCapture(compute_stream_, &graph);
    if (err != cudaSuccess || graph == nullptr) {
        LOG(LOG_WARNING) << "[KrylovGen] Batch tail graph capture failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        use_cuda_graph_ = false;
        return;
    }

    err = cudaGraphInstantiate(&batch_tail_exec_[buf_idx], graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        LOG(LOG_WARNING) << "[KrylovGen] Batch tail graph instantiation failed: "
                         << cudaGetErrorString(err) << ". Falling back to direct launches.";
        batch_tail_exec_[buf_idx] = nullptr;
        use_cuda_graph_ = false;
        return;
    }

    batch_tail_captured_[buf_idx] = true;
    LOG(LOG_DEBUG_2) << "[KrylovGen] Batch tail graph captured for buf_idx=" << buf_idx;
}

void KrylovSequenceGenerator::destroy_batch_tail_graphs() {
    for (int i = 0; i < 2; ++i) {
        if (batch_tail_captured_[i]) {
            cudaGraphExecDestroy(batch_tail_exec_[i]);
            batch_tail_captured_[i] = false;
            batch_tail_exec_[i] = nullptr;
        }
    }
}

#endif // CUDART_VERSION >= 10010
#endif // BW_ENABLE_CUDA_GRAPHS

void KrylovSequenceGenerator::generate(
    int length,
    const std::vector<uint64_t>& h_X,
    const std::vector<uint64_t>& h_Z,
    std::vector<uint64_t>& out_seq,
    cudaStream_t user_stream, // Unused, we use internal streams
    uint64_t* d_S_target
) {
    LOG(LOG_STATS) << "[KrylovGen] Generating " << length << " terms (Pipeline Batch=" << batch_size_ << ")...";

    // 1. Initial Uploads
    //    Use compute_stream_ for uploads
    CHECK_CUDA(cudaMemcpyAsync(dX_, h_X.data(), x_bytes_, cudaMemcpyHostToDevice, compute_stream_));
    CHECK_CUDA(cudaMemcpyAsync(dW_, h_Z.data(), v_bytes_, cudaMemcpyHostToDevice, compute_stream_));
    
    // V_0 = B * Z
    CHECK_CUDA(cudaMemsetAsync(dV_, 0, v_bytes_, compute_stream_));
    op_.mul(dV_, dW_, compute_stream_); // dV = Op * dW
    
    // Resize Output (use output layout — transposed or native)
    size_t s_words = s_output_mat_bytes_ / sizeof(uint64_t);
    out_seq.resize(length * s_words);

    // Initial sync to ensure setup is done
    CHECK_CUDA(cudaStreamSynchronize(compute_stream_));

    // Pipeline Loop
    // We iterate in steps of batch_size.
    int total_batches = (length + batch_size_ - 1) / batch_size_;

    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        int buf_idx = batch_idx % 2;
        int t_start = batch_idx * batch_size_;
        int count = std::min(batch_size_, length - t_start);

        // --- Stream 0: COMPUTE ---
        // Wait until previous Copy on this buffer is done
        if (batch_idx >= 2) {
             CHECK_CUDA(cudaStreamWaitEvent(compute_stream_, copy_done_[buf_idx], 0));
        }

        // ============ INNER LOOP: Krylov iteration V <- B*V ============
        {
            bool used_graph_inner = false;
#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010
            if (use_cuda_graph_ && count == batch_size_ && count >= 2) {
                // Capture on first full batch
                if (!inner_loop_captured_) {
                    capture_inner_loop_graph();
                }
                if (inner_loop_captured_) {
                    used_graph_inner = true;
                    int pairs = count / 2;

                    for (int p = 0; p < pairs; ++p) {
                        int i_even = 2 * p;
                        int i_odd  = 2 * p + 1;

                        // Snapshot V_{2k} -> V_buf[i_even]
                        CHECK_CUDA(cudaMemcpyAsync(
                            d_V_buf_[buf_idx] + i_even * (v_bytes_ / 8),
                            dV_, v_bytes_,
                            cudaMemcpyDeviceToDevice, compute_stream_));

                        // 2-iteration graph: dV_ -> dW_ (V_{2k+1}), dV_ (V_{2k+2})
                        CHECK_CUDA(cudaGraphLaunch(inner_loop_exec_, compute_stream_));

                        // Snapshot V_{2k+1} from dW_ (preserved after graph)
                        CHECK_CUDA(cudaMemcpyAsync(
                            d_V_buf_[buf_idx] + i_odd * (v_bytes_ / 8),
                            dW_, v_bytes_,
                            cudaMemcpyDeviceToDevice, compute_stream_));
                    }

                    // Handle odd remainder (count is odd)
                    if (count % 2 != 0) {
                        int i_last = count - 1;
                        CHECK_CUDA(cudaMemcpyAsync(
                            d_V_buf_[buf_idx] + i_last * (v_bytes_ / 8),
                            dV_, v_bytes_,
                            cudaMemcpyDeviceToDevice, compute_stream_));
                        CHECK_CUDA(cudaMemsetAsync(dW_, 0, v_bytes_, compute_stream_));
                        op_.mul(dW_, dV_, compute_stream_);
                        std::swap(dV_, dW_);
                    }
                }
            }
#endif
#endif
            if (!used_graph_inner) {
                // === Direct launch path (original code) ===
                for (int i = 0; i < count; ++i) {
                    CHECK_CUDA(cudaMemcpyAsync(
                        d_V_buf_[buf_idx] + i * (v_bytes_ / 8),
                        dV_, v_bytes_,
                        cudaMemcpyDeviceToDevice, compute_stream_));

                    if (t_start + i < length) {
                        CHECK_CUDA(cudaMemsetAsync(dW_, 0, v_bytes_, compute_stream_));
                        op_.mul(dW_, dV_, compute_stream_);
                        std::swap(dV_, dW_);
                    }
                }
            }
        }

        // ============ BATCH TAIL: projection + transpose + D2D ============
        {
            bool used_graph_tail = false;
#ifdef BW_ENABLE_CUDA_GRAPHS
#if CUDART_VERSION >= 10010
            if (use_cuda_graph_ && count == batch_size_) {
                // Capture on first use of this buf_idx
                if (!batch_tail_captured_[buf_idx]) {
                    capture_batch_tail(count, buf_idx);
                }
                if (batch_tail_captured_[buf_idx]) {
                    used_graph_tail = true;
                    CHECK_CUDA(cudaGraphLaunch(batch_tail_exec_[buf_idx], compute_stream_));
                }
            }
#endif
#endif
            if (!used_graph_tail) {
                // === Direct launch path (fallback / last batch / graph disabled) ===
                CHECK_CUDA(cudaMemsetAsync(d_S_buf_[buf_idx], 0,
                                            count * s_mat_bytes_, compute_stream_));
                dispatch_batch_proj(count, buf_idx);

                if (transpose_output_) {
                    size_t src_stride_words = (n_block_ + 63) / 64;
                    size_t dst_stride_words = (m_block_ + 63) / 64;
                    launch_transpose_batch(
                        d_S_buf_[buf_idx], d_S_transposed_buf_[buf_idx],
                        count, m_block_, n_block_,
                        src_stride_words, dst_stride_words,
                        compute_stream_);
                }
            }
        }

        // D2D accumulation — ALWAYS direct launch (destination offset varies per batch)
        if (d_S_target) {
            uint64_t* copy_src = transpose_output_
                ? d_S_transposed_buf_[buf_idx] : d_S_buf_[buf_idx];
            CHECK_CUDA(cudaMemcpyAsync(
                d_S_target + (size_t)t_start * (s_output_mat_bytes_ / sizeof(uint64_t)),
                copy_src,
                (size_t)count * s_output_mat_bytes_,
                cudaMemcpyDeviceToDevice,
                compute_stream_));
        }

        // Record Compute Done
        CHECK_CUDA(cudaEventRecord(compute_done_[buf_idx], compute_stream_));

        // --- Stream 1: COPY ---
        // Wait for Compute
        CHECK_CUDA(cudaStreamWaitEvent(copy_stream_, compute_done_[buf_idx], 0));

        // Download S-Buffer (from transposed or native source)
        uint64_t* d2h_src = transpose_output_
            ? d_S_transposed_buf_[buf_idx] : d_S_buf_[buf_idx];
        CHECK_CUDA(cudaMemcpyAsync(
            h_S_buf_[buf_idx],
            d2h_src,
            count * s_output_mat_bytes_,
            cudaMemcpyDeviceToHost,
            copy_stream_
        ));
        
        // Record Copy Done
        CHECK_CUDA(cudaEventRecord(copy_done_[buf_idx], copy_stream_));

        // --- HOST: CONSUME PREVIOUS ---
        // If we have a previous batch in flight (batch_idx-1), wait for it and copy to out_seq.
        // The host must wait for the PREVIOUS copy to finish.
        if (batch_idx > 0) {
            int prev_idx = (batch_idx - 1) % 2;
            int prev_t_start = (batch_idx - 1) * batch_size_;
            int prev_count = std::min(batch_size_, length - prev_t_start);
            
            CHECK_CUDA(cudaEventSynchronize(copy_done_[prev_idx]));
            
            // Copy to final std::vector
            uint64_t* dest = out_seq.data() + prev_t_start * s_words;
            std::memcpy(dest, h_S_buf_[prev_idx], prev_count * s_output_mat_bytes_);
        }
        
        if (batch_idx % 2 == 0) {
             LOG(LOG_DEBUG_2) << "[KrylovGen] Batch " << batch_idx << "/" << total_batches;
        }
    }

    // Flush last batch
    int last_idx = (total_batches - 1) % 2;
    int last_t_start = (total_batches - 1) * batch_size_;
    int last_count = std::min(batch_size_, length - last_t_start);
    
    CHECK_CUDA(cudaEventSynchronize(copy_done_[last_idx]));
    uint64_t* dest = out_seq.data() + last_t_start * s_words;
    std::memcpy(dest, h_S_buf_[last_idx], last_count * s_output_mat_bytes_);

    LOG(LOG_STATS) << "[KrylovGen] Generation complete.";
}

} // namespace stage1
} // namespace lingen
