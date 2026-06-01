// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "optimizer_common.h"
#include "device_csr.h"

class GPUAutoTuner {
public:
    struct Config {
        int  initial_block_size = 8;
        int  max_block_size     = 65536;
        bool enable_m4rm        = true;
        bool enable_tiledcoo    = true;
        bool enable_tiledcoo_unrolled = true;  // M1.2: unrolled TiledCOO variant
        bool enable_delta16     = true;
        bool enable_pfor_be     = true;
        bool enable_warp_csr    = true;
        bool enable_golomb      = false;  // P2-A: disabled — CPU fallback produces wrong format data
        bool allow_cpu_fallback = true;
        uint32_t n_spmm_calls   = 0;   // Expected SpMM invocations for amortization (0 = auto-estimate)
        int sm_count = 0;       // 0 = auto-detect in tune()
        int compute_major = 0;  // 0 = auto-detect
        int compute_minor = 0;  // 0 = auto-detect
    };

    static ExecutionPlan tune(
        const HostMatrix& mat,
        bool is_transposed,
        int global_vec_width,
        const Config& config,
        bool verbose = false
    );

private:
    struct AtomicBlock {
        row_idx_t start_row;
        row_idx_t end_row;
        uint64_t  nnz;
        double    avg_weight;  // nnz / n_rows — for density-based format pruning
        bool      dense_head;  // true → only benchmark M4RM on this block
        std::vector<double> benchmark_times;  // [n_candidates]
    };

    static std::vector<AtomicBlock> create_exponential_blocks(
        const DeviceCSR& csr, int initial_size, int max_size, bool is_transposed);

    static std::vector<KernelConfig> generate_candidate_configs(
        int global_vec_width, const Config& config);
};
