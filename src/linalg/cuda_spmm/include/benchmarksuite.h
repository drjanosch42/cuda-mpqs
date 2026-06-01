// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include <string>
#include <vector>

// =================================================================================
// Data Structures
// =================================================================================

struct MatrixParameter {
    row_idx_t n_rows;
    idx_t n_cols;
    double alpha;
    uint64_t seed;
};

struct BenchmarkScenario {
    std::string name;
    SpMMConfig config;
    bool use_transpose; // true = AT, false = A
    bool verify;        // Enable/Disable CPU verification
};

// =================================================================================
// Benchmark Suite Class
// =================================================================================

class BenchmarkSuite {
public:
    /**
     * @brief Main entry point to run a suite of benchmarks across multiple matrix configurations.
     * 
     * @param params List of matrix parameters to test.
     * @param scenarios List of kernel configurations/scenarios to run for each matrix.
     * @param enable_layer_analysis If true, runs the fine-grained layer analysis (Layers 0-8) after standard scenarios.
     */
    static void run_suite(
        const std::vector<MatrixParameter>& params,
        const std::vector<BenchmarkScenario>& scenarios,
        bool enable_layer_analysis = true
    );

private:
    static void process_single_matrix(
        const MatrixParameter& param,
        const std::vector<BenchmarkScenario>& scenarios,
        bool enable_layer_analysis
    );

    // [New] Dynamic Density Analysis
    static void execute_density_profile(
        const HostMatrix& AT,
        idx_t dense_limit_cols
    );

    // [New] Dense Kernel Scaling Analysis
    static void execute_dense_scaling(
        const HostMatrix& AT
    );

    // [Helper] L2 Cache Flush
    static void flush_l2_cache();
};


