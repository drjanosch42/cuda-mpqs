// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "bw_spmm_interface.h"
#include "bw_version.h"
#include "generator.h"
#include "hpc_logger.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main(int argc, char** argv) {

    // Initialize Logger for standalone debugging
    LogConfig cfg;
    cfg.enable_cout = true;
    cfg.min_severity_cout = LOG_DEBUG_2; // Verbose for debugging
    cfg.enable_file = false;             // No file needed for quick tests
    HPCLogger::Get().Init(cfg);
    
    // Default settings
    row_idx_t n_rows = 200000;
    double alpha = 15.0;
    int bit_width = 64;
    if (argc >= 2) n_rows = std::atoll(argv[1]);
    if (argc >= 3) alpha = std::atof(argv[2]);
    if (argc >= 4) bit_width = std::atof(argv[3]);
    LOG(LOG_INFO) << "[SingleBenchmark] Generating Matrix (" << n_rows << "x" << n_rows << ")..." << std::endl;
    
    // 1. Generate (Now uses GPU by default internally)
    MatrixGenerator gen(12345ULL);
    gen.generate_factor_base(n_rows);
    HostMatrix A = gen.generate_matrix(n_rows, alpha); 

    LOG(LOG_INFO) << "[SingleBenchmark] Initializing Block Wiedemann SpMM Engine " << lingen::version_string() << "..." << std::endl;
    // Padding 0 means default to n_cols
    BlockWiedemannSpMM bw_engine(A, 0, true, true); 

    LOG(LOG_INFO) << "[SingleBenchmark] Optimizing for " << bit_width << "-bit vectors..." << std::endl;
    SpMMPerformanceReport report = bw_engine.setup_and_benchmark(bit_width, bit_width);

    LOG(LOG_INFO) << "[SingleBenchmark] Block Wiedemann Planner Decision:" << std::endl;
    bool use_A_sequence = (report.ratio_A_over_AT >= 1.0);
    
    if (use_A_sequence) {
        LOG(LOG_INFO) << "[SingleBenchmark]    Use Matrix A sequence (A is faster)." << std::endl;
    } else {
        LOG(LOG_INFO) << "[SingleBenchmark]    Use Matrix AT sequence (AT is faster)." << std::endl;
    }

    LOG(LOG_INFO) << "[SingleBenchmark] Ready for Iterations. Executing dummy kernel..." << std::endl;

    // Generate random input vector
    std::vector<uint8_t> h_V = gen.generate_random_vector(n_rows, bit_width);

    // Allocate dummy vectors
    size_t vec_size = (size_t)n_rows * (bit_width / 8);
    uint64_t *d_C, *d_V;
    cudaMalloc(&d_C, vec_size);
    cudaMalloc(&d_V, vec_size);
    
    // Copy random vector to device instead of using a fixed pattern
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vec_size, cudaMemcpyHostToDevice));
    cudaMemset(d_C, 0x00, vec_size);

    // Only run the chosen sequence to avoid auto-loading the purged matrix
    if (use_A_sequence) {
        bw_engine.execute_A(d_C, d_V);
    } else {
        bw_engine.execute_AT(d_C, d_V);
    }
    cudaDeviceSynchronize();

    LOG(LOG_INFO) << "[SingleBenchmark] Execution successful. Cleaning up." << std::endl;

    cudaFree(d_C);
    cudaFree(d_V);

    return 0;
}
