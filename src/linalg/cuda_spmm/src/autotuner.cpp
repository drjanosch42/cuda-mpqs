// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "autotuner.h"
#include "kernels.h"
#include "preprocessing.h"
#include "m4rm_data.h"
#include "hpc_logger.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

std::vector<KernelConfig> SpMMAutoTuner::generate_candidate_configs(int global_vec_width) {
    std::vector<KernelConfig> configs;
    
    // 1. M4RM
    {
        KernelConfig kc; kc.id = KernelID::M4RM; kc.vector_width_bits = global_vec_width;
        kc.params.m4rm_rows = 8; kc.name = "M4RM (8 Rows)";
        configs.push_back(kc);
    }
    
    // 2. Dense Bitslice
    {
        // KernelConfig kc; kc.id = KernelID::Dense_Bitslice; kc.vector_width_bits = global_vec_width;
        // kc.name = "Dense Bitslice";
        // configs.push_back(kc);
    }

    // 3. Warp CSR
    {
        // KernelConfig kc; kc.id = KernelID::Sparse_WarpCSR; kc.vector_width_bits = global_vec_width;
        // kc.name = "Warp CSR";
        // configs.push_back(kc);
    }

    // 4.a Tiled COO
    if (true) {
        KernelConfig kc; kc.id = KernelID::Sparse_TiledCOO; kc.vector_width_bits = global_vec_width;
        kc.params.tiled_block_size = 256; kc.name = "Tiled COO (256)";
        configs.push_back(kc);
    } else
    // 4.b Tiled COO Unrolled (WARNING: Can exhibit rare Heisenbug behavior)
    {
        // KernelConfig kc; kc.id = KernelID::Sparse_TiledCOO_Unrolled; kc.vector_width_bits = global_vec_width;
        // kc.params.tiled_block_size = 256; kc.name = "Tiled COO Unrolled (256)";
        // configs.push_back(kc);
    }

    // 5. Delta-16
    {
        KernelConfig kc; kc.id = KernelID::Sparse_Delta16; kc.vector_width_bits = global_vec_width;
        kc.name = "Delta-16";
        configs.push_back(kc);
    }

    // 6. PForDelta
    {
        KernelConfig kc; kc.id = KernelID::Sparse_PForDelta_BitExact; kc.vector_width_bits = global_vec_width;
        kc.params.pfor_threshold = 0.90f; kc.name = "PForDelta (BE, 0.90)";
        configs.push_back(kc);

        // kc.params.pfor_threshold = 0.98f; kc.name = "PForDelta (BE, 0.98)";
        // configs.push_back(kc);
    }
    // 7. Golomb-Rice
    {
        KernelConfig kc; kc.id = KernelID::Sparse_Golomb; kc.vector_width_bits = global_vec_width;
        kc.name = "Golomb-Rice";
        configs.push_back(kc);
    }
    return configs;
}

SpMMConfig SpMMAutoTuner::config_to_spmm_config(const KernelConfig& kc) {
    SpMMConfig cfg;
    cfg.vector_width_bits = kc.vector_width_bits;
    cfg.enable_dense_bitslice = false;
    cfg.enable_heavy_rows = false;
    cfg.enable_sparse = false;

    switch (kc.id) {
        case KernelID::M4RM: cfg.enable_m4rm = true; cfg.m4rm_rows = kc.params.m4rm_rows; break;
        case KernelID::Dense_Bitslice: 
            cfg.enable_dense_bitslice = true; 
            cfg.enable_sparse = true; 
            cfg.enable_heavy_warp_csr = true; 
            break;
        case KernelID::Sparse_WarpCSR: cfg.enable_sparse = true; cfg.enable_heavy_warp_csr = true; break;
        case KernelID::Sparse_TiledCOO: cfg.enable_sparse = true; cfg.enable_sparse_tiled_coo = true; cfg.tiled_row_block_size = kc.params.tiled_block_size; break;
        case KernelID::Sparse_TiledCOO_Unrolled: cfg.enable_sparse = true; cfg.enable_sparse_tiled_coo_unrolled = true; cfg.tiled_row_block_size = kc.params.tiled_block_size; break;
        case KernelID::Sparse_Delta16: cfg.enable_sparse = true; cfg.enable_sparse_delta_16 = true; break;
        case KernelID::Sparse_PForDelta: cfg.enable_sparse = true; cfg.enable_sparse_pfor = true; cfg.pfor_exception_threshold = kc.params.pfor_threshold; break;
        case KernelID::Sparse_PForDelta_BitExact: cfg.enable_sparse = true; cfg.enable_sparse_pfor_bit_exact = true; cfg.pfor_exception_threshold = kc.params.pfor_threshold; break;
        case KernelID::Sparse_Golomb: cfg.enable_sparse = true; break;
        default: break;
    }
    return cfg;
}

std::vector<SpMMAutoTuner::AtomicBlock> SpMMAutoTuner::create_atomic_blocks(
    const HostMatrix& mat, 
    int global_vec_width, 
    const std::vector<KernelConfig>& configs
) {
    std::vector<AtomicBlock> blocks;
    row_idx_t current = 0;
    
    while (current < mat.n_rows) {
        row_idx_t step;
        if (current < 64) step = 8;
        else if (current < 1024) step = current; 
        else step = 65536;

        row_idx_t next = std::min(current + step, mat.n_rows);
        
        AtomicBlock b;
        b.start_row = current;
        b.end_row = next;
        b.nnz = 0;
        for(row_idx_t r=current; r<next; ++r) b.nnz += mat.rows[r].size();
        b.benchmark_times.resize(configs.size(), std::numeric_limits<double>::infinity());
        
        blocks.push_back(b);
        current = next;
    }
    return blocks;
}

void SpMMAutoTuner::benchmark_block(
    const HostMatrix& full_mat, 
    AtomicBlock& block, 
    int global_vec_width,
    const std::vector<KernelConfig>& configs,
    uint8_t* d_flush_buffer,
    uint64_t* d_V,
    uint64_t* d_C
) {
    if (block.nnz == 0) {
        std::fill(block.benchmark_times.begin(), block.benchmark_times.end(), 0.0001); 
        return;
    }

    HostMatrixView slice(full_mat, block.start_row, block.end_row);
    size_t out_bytes = slice.n_rows * (global_vec_width/8);

    // --- Data Caching State ---
    enum class CachedType { NONE, M4RM, STANDARD };
    CachedType cached_type = CachedType::NONE;
    
    // Cached Parameters
    int cached_m4rm_rows = -1;
    idx_t cached_dense_lim = 0;
    float cached_pfor_thresh = -1.0f;
    int cached_tiled_bs = -1;

    // Cached Resources
    M4RMContext m4rm_ctx = {};
    DeviceMatrix dm = {};

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& conf = configs[i];
        
        if (conf.id == KernelID::M4RM && slice.n_rows != 8) {
            block.benchmark_times[i] = std::numeric_limits<double>::infinity();
            continue;
        }

        try {
            double ms = 0.0;
            if (conf.id == KernelID::M4RM) {
                // Check Cache for M4RM
                int req_rows = conf.params.m4rm_rows;
                if (cached_type != CachedType::M4RM || cached_m4rm_rows != req_rows) {
                    // Invalidate and Reload
                    if (cached_type == CachedType::M4RM) free_m4rm_context(m4rm_ctx);
                    if (cached_type == CachedType::STANDARD) SpMMKernels::free_matrix(dm);
                    
                    m4rm_ctx = MatrixPreprocessor::preprocess_m4rm(slice, req_rows);
                    cached_type = CachedType::M4RM;
                    cached_m4rm_rows = req_rows;
                }

                CUDA_CHECK(cudaMemset(d_C, 0, out_bytes));
                if(d_flush_buffer) CUDA_CHECK(cudaMemset(d_flush_buffer, 1, 40*1024*1024));
                
                auto t0 = std::chrono::high_resolution_clock::now();
                int iter = 10; 
                for(int k=0; k<iter; ++k) launch_m4rm_full(m4rm_ctx, d_V, d_C, global_vec_width);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iter;
            } else {
                // Check Cache for Standard/Sparse
                idx_t req_dense = (conf.id == KernelID::Dense_Bitslice) ? 64 : 0;
                float req_pfor = conf.params.pfor_threshold;
                int req_bs = conf.params.tiled_block_size;

                bool is_match = (cached_type == CachedType::STANDARD) &&
                                (cached_dense_lim == req_dense) &&
                                (std::abs(cached_pfor_thresh - req_pfor) < 1e-4) &&
                                (cached_tiled_bs == req_bs);

                if (!is_match) {
                    if (cached_type == CachedType::M4RM) free_m4rm_context(m4rm_ctx);
                    if (cached_type == CachedType::STANDARD) SpMMKernels::free_matrix(dm);

                    CompressedMatrix cm = MatrixPreprocessor::preprocess(
                        slice, req_dense, 0, false, 
                        req_pfor, req_bs, false
                    );
                    dm = SpMMKernels::allocate_and_copy(cm);
                    
                    cached_type = CachedType::STANDARD;
                    cached_dense_lim = req_dense;
                    cached_pfor_thresh = req_pfor;
                    cached_tiled_bs = req_bs;
                }

                if(d_flush_buffer) CUDA_CHECK(cudaMemset(d_flush_buffer, 0xAA, 40*1024*1024));
                
                SpMMConfig cfg = config_to_spmm_config(conf);
                auto t0 = std::chrono::high_resolution_clock::now();
                int iter = 5;
                for(int k=0; k<iter; ++k) SpMMKernels::run_spmm(dm, d_C, d_V, cfg);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iter;
            }
            block.benchmark_times[i] = ms;
        } catch (...) {
            // Force cache flush on error to prevent inconsistent state
            if (cached_type == CachedType::M4RM) free_m4rm_context(m4rm_ctx);
            if (cached_type == CachedType::STANDARD) SpMMKernels::free_matrix(dm);
            cached_type = CachedType::NONE;
            
            block.benchmark_times[i] = std::numeric_limits<double>::infinity();
        }
    }
    // Final cleanup of cached resources
    if (cached_type == CachedType::M4RM) free_m4rm_context(m4rm_ctx);
    if (cached_type == CachedType::STANDARD) SpMMKernels::free_matrix(dm);
}

ExecutionPlan SpMMAutoTuner::tune_global(const HostMatrix& mat, bool is_transposed, int global_vec_width, bool verbose) {
   
    LOG(LOG_DEBUG_1)  << "[AutoTuner] Analyzing " << (is_transposed ? "AT" : "A") 
              << " (" << mat.n_rows << " rows) Width=" << global_vec_width << "..." << std::endl;

    auto configs = generate_candidate_configs(global_vec_width);
    auto blocks = create_atomic_blocks(mat, global_vec_width, configs);

    // Hoist Allocations
    row_idx_t max_block_rows = 0;
    for(const auto& b : blocks) max_block_rows = std::max(max_block_rows, b.end_row - b.start_row);
    
    size_t vec_bytes = mat.n_cols * (global_vec_width / 8);
    size_t max_out_bytes = max_block_rows * (global_vec_width / 8);
    
    uint64_t *d_V, *d_C;
    CUDA_CHECK(cudaMalloc(&d_V, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, max_out_bytes));
    CUDA_CHECK(cudaMemset(d_V, 0xFF, vec_bytes));
    
    size_t flush_size = 40 * 1024 * 1024;
    uint8_t* d_flush_buffer;
    CUDA_CHECK(cudaMalloc(&d_flush_buffer, flush_size));
    
    // Profile
    for (size_t i = 0; i < blocks.size(); ++i) {
        benchmark_block(mat, blocks[i], global_vec_width, configs, d_flush_buffer, d_V, d_C);
        if (((i+1) % 5 == 0) || (i == blocks.size()-1))
	    LOG(LOG_DEBUG_1) << "[AutoTuner] " << (100.0*(float)(i+1)/(float)blocks.size()) << "% blocks processed" << std::endl << std::flush;
    }
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_flush_buffer));

    // DP
    std::vector<double> dp(blocks.size() + 1, std::numeric_limits<double>::infinity());
    std::vector<std::pair<int, int>> path(blocks.size() + 1, {-1, -1});
    dp[0] = 0.0;
    const double KERNEL_OVERHEAD_MS = 0.015;

    for (size_t i = 1; i <= blocks.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            for (int k = 0; k < (int)configs.size(); ++k) {
                if (configs[k].id == KernelID::M4RM) {
                    row_idx_t seg_rows = blocks[i-1].end_row - blocks[j].start_row;
                    if (seg_rows != 8) continue; 
                }

                double exec_time = 0.0;
                bool possible = true;
                
                for (size_t b = j; b < i; ++b) {
                    double t = blocks[b].benchmark_times[k];
                    if (t == std::numeric_limits<double>::infinity()) {
                        possible = false; break;
                    }
                    exec_time += t;
                }
                
                if (!possible) continue;
                double total_cost = dp[j] + exec_time + KERNEL_OVERHEAD_MS;
                if (total_cost < dp[i]) {
                    dp[i] = total_cost;
                    path[i] = {(int)j, k};
                }
            }
        }
    }

    ExecutionPlan plan;
    plan.is_transposed = is_transposed;
    plan.estimated_total_time_ms = dp[blocks.size()];
    
    int curr = blocks.size();
    std::vector<SegmentRecipe> reverse_segs;
    while (curr > 0) {
        int prev = path[curr].first;
        int conf_idx = path[curr].second;
        SegmentRecipe seg;
        seg.start_row = blocks[prev].start_row;
        seg.end_row = blocks[curr-1].end_row;
        seg.best_config = configs[conf_idx];
        
        // Calculate Throughput for Display
        double seg_time = dp[curr] - dp[prev] - KERNEL_OVERHEAD_MS;
        size_t seg_nnz = 0;
        for(int b=prev; b<curr; ++b) seg_nnz += blocks[b].nnz;
        seg.measured_throughput_gnnz = (seg_nnz / 1e9) / (seg_time / 1000.0);

        reverse_segs.push_back(seg);
        curr = prev;
    }
    std::reverse(reverse_segs.begin(), reverse_segs.end());
    plan.segments = reverse_segs;

    // Verbose Output
    if (verbose) {
        LOG(LOG_DEBUG_2) << "[AutoTuner] Optimized Plan:" << std::endl;
        size_t total_nnz = 0;
        for (const auto& s : plan.segments) {
            LOG(LOG_DEBUG_2) << "[AutoTuner]  [" << std::setw(7) << s.start_row << "-" << std::setw(7) << s.end_row << ") " 
                      << std::setw(28) << s.best_config.name 
                      << " | " << std::fixed << std::setprecision(2) << s.measured_throughput_gnnz << " GNNz/s" << std::endl;
            for(row_idx_t r=s.start_row; r<s.end_row; ++r) total_nnz += mat.rows[r].size();
        }
        double est_throughput = (total_nnz / 1e9) / (plan.estimated_total_time_ms / 1000.0);
        LOG(LOG_DEBUG_1) << "[AutoTuner] Estimated Total Throughput: " << est_throughput << " GNNz/s" << std::endl;
    }

    return plan;
}
