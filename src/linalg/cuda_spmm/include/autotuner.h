// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "optimizer_common.h"
#include <vector>

class SpMMAutoTuner {
public:
    static ExecutionPlan tune_global(
        const HostMatrix& mat, 
        bool is_transposed, 
        int global_vec_width,
        bool verbose = false // Default to false
    );

private:
    struct AtomicBlock {
        row_idx_t start_row;
        row_idx_t end_row;
        size_t nnz;
        std::vector<double> benchmark_times; 
    };

    static std::vector<AtomicBlock> create_atomic_blocks(
        const HostMatrix& mat, 
        int global_vec_width, 
        const std::vector<KernelConfig>& configs
    );

    static void benchmark_block(
        const HostMatrix& full_mat, 
        AtomicBlock& block, 
        int global_vec_width,
        const std::vector<KernelConfig>& configs,
        uint8_t* d_flush_buffer,
	uint64_t* d_V,
	uint64_t* d_C
    );

    static std::vector<KernelConfig> generate_candidate_configs(int global_vec_width);
    static SpMMConfig config_to_spmm_config(const KernelConfig& kc);
};
