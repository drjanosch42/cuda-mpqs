// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "spmm_optimized.h"
#include "preprocessing.h"
#include "autotuner.h"
#include <iostream>
#include <iomanip>
#include <chrono>

OptimizedSpMM::~OptimizedSpMM() { free_resources(); }

void OptimizedSpMM::free_resources() {
    for(auto& seg : segments_) {
        SpMMKernels::free_matrix(seg.device_data);
        free_m4rm_context(seg.m4rm_data);
    }
    segments_.clear();
    for(auto s : streams_) cudaStreamDestroy(s);
    streams_.clear();
    if(event_dense_done_) { cudaEventDestroy(event_dense_done_); event_dense_done_ = nullptr; }
}

static HostMatrix get_slice(const HostMatrix& src, row_idx_t start, row_idx_t count) {
    HostMatrix mat;
    mat.n_rows = count; mat.n_cols = src.n_cols; mat.rows.resize(count);
    for (row_idx_t i = 0; i < count; ++i) mat.rows[i] = src.rows[start + i];
    return mat;
}

static SpMMConfig to_spmm_cfg(const KernelConfig& kc) {
    SpMMConfig cfg;
    cfg.vector_width_bits = kc.vector_width_bits;
    cfg.enable_dense_bitslice = false; cfg.enable_heavy_rows = false; cfg.enable_sparse = false;
    switch (kc.id) {
        case KernelID::M4RM: cfg.enable_m4rm = true; cfg.m4rm_rows = kc.params.m4rm_rows; break;
        case KernelID::Dense_Bitslice: cfg.enable_dense_bitslice = true; cfg.enable_sparse = true; cfg.enable_heavy_warp_csr = true; break;
        case KernelID::Sparse_WarpCSR: cfg.enable_sparse = true; cfg.enable_sparse_warp_csr = true; break;
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

void OptimizedSpMM::compile(const HostMatrix& host_mat, const ExecutionPlan& plan) {
    free_resources();
    if (plan.segments.empty()) return;

    std::vector<SegmentRecipe> merged;
    if (!plan.segments.empty()) {
        SegmentRecipe current = plan.segments[0];
        for (size_t i = 1; i < plan.segments.size(); ++i) {
            const auto& next = plan.segments[i];
            bool is_m4rm = (current.best_config.id == KernelID::M4RM);
            if (!is_m4rm && current.best_config == next.best_config) {
                if (current.end_row == next.start_row) current.end_row = next.end_row;
                else { merged.push_back(current); current = next; }
            } else { merged.push_back(current); current = next; }
        }
        merged.push_back(current);
    }

    // Null out plan's original device_data so a second compile() on the same
    // plan (e.g. verify_AT after unload) sees has_device_data()==false and
    // falls through to CPU fallback instead of using freed pointers.
    // The merged copies already captured the device data above.
    for (const auto& ps : plan.segments) {
        ps.device_data = {};
        ps.m4rm_data   = {};
    }

    for (const auto& recipe : merged) {
        MergedSegment seg;
        seg.config = recipe.best_config;
        seg.start_row = recipe.start_row;
        seg.num_rows = recipe.end_row - recipe.start_row;
        seg.legacy_config = to_spmm_cfg(seg.config);
        seg.is_m4rm = (seg.config.id == KernelID::M4RM);
        seg.is_dense = (seg.config.id == KernelID::Dense_Bitslice);
        seg.is_sparse = (!seg.is_m4rm && !seg.is_dense);

        if (recipe.has_device_data()) {
            // GPU-only path: adopt device data directly (zero re-preprocessing).
            // Transfer ownership — null out recipe's pointers so that if
            // compile() is called again on the same plan (e.g. verify_A/AT
            // re-compiles), has_device_data() returns false and the CPU
            // fallback runs instead of double-freeing device memory.
            seg.device_data = recipe.device_data;
            seg.m4rm_data   = recipe.m4rm_data;
            recipe.device_data = {};
            recipe.m4rm_data   = {};
        } else {
            // Legacy CPU path (unchanged)
            HostMatrix slice;
            if (!plan.density_perm.empty()) {
                // GPU autotuner applied a density permutation on top of the
                // already-sorted host_mat.  Map segment rows back through
                // the permutation so we pick up the correct host rows.
                slice.n_rows = seg.num_rows;
                slice.n_cols = host_mat.n_cols;
                slice.rows.resize(seg.num_rows);
                for (row_idx_t i = 0; i < seg.num_rows; ++i) {
                    row_idx_t orig_row = plan.density_perm[seg.start_row + i];
                    slice.rows[i] = host_mat.rows[orig_row];
                }
            } else {
                slice = get_slice(host_mat, seg.start_row, seg.num_rows);
            }
            if (seg.is_m4rm) seg.m4rm_data = MatrixPreprocessor::preprocess_m4rm(slice, seg.config.params.m4rm_rows);
            else {
                idx_t dense_lim = seg.is_dense ? 64 : 0;
                CompressedMatrix cm = MatrixPreprocessor::preprocess(
                    slice, dense_lim, 0, false,
                    seg.config.params.pfor_threshold,
                    seg.config.params.tiled_block_size, false
                );
                seg.device_data = SpMMKernels::allocate_and_copy(cm);
            }
        }
        segments_.push_back(seg);
    }
    size_t num_streams = segments_.size();
    if (num_streams < 1) num_streams = 1;
    for(size_t i=0; i<num_streams; ++i) { cudaStream_t s; cudaStreamCreate(&s); streams_.push_back(s); }
    cudaEventCreate(&event_dense_done_);
}

void OptimizedSpMM::launch_segment(const MergedSegment& seg, void* d_C, const void* d_V, cudaStream_t stream) {
    int width_bytes = seg.config.vector_width_bits / 8;
    // Calculate offset in bytes using char* or uint8_t* to allow unaligned to 64-bit boundaries
    void* d_C_offset = (void*)((uint8_t*)d_C + seg.start_row * width_bytes);

    if (seg.is_m4rm) {
         launch_m4rm_full(seg.m4rm_data, d_V, d_C_offset, seg.config.vector_width_bits, stream); 
    } else {
         SpMMKernels::run_spmm(seg.device_data, d_C_offset, d_V, seg.legacy_config, stream); 
    }
}

void OptimizedSpMM::execute_with_config(void* d_C, const void* d_V, const LaunchConfig& lc) {
    std::vector<int> idx_dense, idx_sparse;
    for (size_t i = 0; i < segments_.size(); ++i) {
        if (segments_[i].is_m4rm || segments_[i].is_dense)
            idx_dense.push_back(i);
        else
            idx_sparse.push_back(i);
    }

    // Dense + M4RM phase (disjoint row ranges — safe to overlap)
    for (size_t k = 0; k < idx_dense.size(); ++k) {
        int seg_idx = idx_dense[k];
        cudaStream_t s = lc.spawn_dense_group ? streams_[k % streams_.size()] : 0;
        launch_segment(segments_[seg_idx], d_C, d_V, s);
    }
    if (!lc.spawn_dense_sparse_merge) cudaDeviceSynchronize();

    // Sparse phase
    for (size_t k = 0; k < idx_sparse.size(); ++k) {
        int seg_idx = idx_sparse[k];
        int stream_offset = lc.spawn_dense_sparse_merge ? idx_dense.size() : 0;
        cudaStream_t s = lc.spawn_sparse_group ? streams_[(k + stream_offset) % streams_.size()] : 0;
        launch_segment(segments_[seg_idx], d_C, d_V, s);
    }
}

void OptimizedSpMM::execute(void* d_C, const void* d_V) {
    execute_with_config(d_C, d_V, active_config_);
}

void OptimizedSpMM::execute(void* d_C, const void* d_V, cudaStream_t stream) {
    execute_with_config(d_C, d_V, active_config_, stream);
}

void OptimizedSpMM::execute_with_config(void* d_C, const void* d_V,
                                         const LaunchConfig& lc,
                                         cudaStream_t stream) {
    // Stream-aware path: ALL segments dispatched on caller's stream.
    // Segment ordering is guaranteed by CUDA's in-order stream semantics.
    // LaunchConfig multi-stream flags are ignored — single-stream dispatch
    // is required for CUDA graph capture compatibility.
    for (auto& seg : segments_) {
        launch_segment(seg, d_C, d_V, stream);
    }
}

void OptimizedSpMM::tune_execution_strategy(void* d_C, const void* d_V) {
    double best_time = 1e9;
    LaunchConfig best_cfg;
    for (int i = 0; i < 8; ++i) {
        LaunchConfig cfg;
        cfg.spawn_dense_group = (i & 1);        // bit 0
        cfg.spawn_sparse_group = (i & 2);       // bit 1
        cfg.spawn_dense_sparse_merge = (i & 4); // bit 2
        execute_with_config(d_C, d_V, cfg); cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        int iterations = 10;
        for(int k=0; k<iterations; ++k) execute_with_config(d_C, d_V, cfg);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iterations;
        if (ms < best_time) { best_time = ms; best_cfg = cfg; }
    }
    active_config_ = best_cfg;
}
