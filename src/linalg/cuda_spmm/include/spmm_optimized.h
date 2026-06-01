// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "optimizer_common.h"
#include "kernels.h"
#include "m4rm_data.h"
#include <vector>
#include <cuda_runtime.h>

class OptimizedSpMM {
public:
    OptimizedSpMM() = default;
    ~OptimizedSpMM();

    void compile(const HostMatrix& host_mat, const ExecutionPlan& plan);

    // Legacy API (synchronous, uses internal streams + active_config_)
    void execute(void* d_C, const void* d_V);
    void execute_with_config(void* d_C, const void* d_V, const LaunchConfig& config);

    // Stream-aware API (all segments dispatched on caller's stream, no internal sync)
    void execute(void* d_C, const void* d_V, cudaStream_t stream);
    void execute_with_config(void* d_C, const void* d_V, const LaunchConfig& lc, cudaStream_t stream);

    // Benchmarks all strategy combinations and picks the winner
    void tune_execution_strategy(void* d_C, const void* d_V);

    void free_resources();

private:
    struct MergedSegment {
        KernelConfig config;
        row_idx_t start_row;
        row_idx_t num_rows;
        
        DeviceMatrix device_data; 
        M4RMContext m4rm_data;
        SpMMConfig legacy_config; 
        
        // Metadata for fast scheduling
        bool is_m4rm;
        bool is_dense; // Bitslice
        bool is_sparse;
    };

    std::vector<MergedSegment> segments_;
    
    // CUDA Streams
    std::vector<cudaStream_t> streams_;
    cudaEvent_t event_dense_done_ = nullptr;

    // Current best strategy
    LaunchConfig active_config_;

    // Helper to launch a single segment
    void launch_segment(const MergedSegment& seg, void* d_C, const void* d_V, cudaStream_t stream);
};
