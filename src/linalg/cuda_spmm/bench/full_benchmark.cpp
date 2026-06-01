// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "autotuner.h"
#include "gpu_autotuner.h"
#include "spmm_optimized.h"
#include "generator.h"
#include "verification.h"
#include "hpc_logger.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

double run_hot_benchmark(OptimizedSpMM& spmm, uint64_t* d_C, uint64_t* d_V, int iterations) {
    // 1. Warmup
    spmm.execute(d_C, d_V);
    cudaDeviceSynchronize();

    // 2. Hot Measurement
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        spmm.execute(d_C, d_V);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iterations; 
}

// Ensure matrix is sorted for best performance
HostMatrix sort_matrix_by_density(const HostMatrix& A) {
    std::vector<row_idx_t> p(A.n_rows);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](row_idx_t a, row_idx_t b) { return A.rows[a].size() > A.rows[b].size(); });
    HostMatrix sortedA; sortedA.n_rows = A.n_rows; sortedA.n_cols = A.n_cols; sortedA.rows.resize(A.n_rows);
    for(row_idx_t i=0; i<A.n_rows; ++i) sortedA.rows[i] = A.rows[p[i]];
    return sortedA;
}

int main(int argc, char** argv) {

    // Initialize Logger for standalone debugging
    LogConfig cfg;
    cfg.enable_cout = true;
    cfg.min_severity_cout = LOG_DEBUG_2; // Verbose for debugging
    cfg.enable_file = false;             // No file needed for quick tests
    HPCLogger::Get().Init(cfg);

    // Default settings
    row_idx_t N_ROWS = 200000;
    idx_t N_COLS = 200000;
    double ALPHA = 15.0;
    bool use_gpu_autotuner = true;  // P2-D: GPU autotuner by default

    // Parse flags and positional args
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--legacy") use_gpu_autotuner = false;
        else if (arg == "--gpu-only") use_gpu_autotuner = true;
        else pos_args.push_back(arg);
    }
    if (pos_args.size() >= 1) N_ROWS = std::atoll(pos_args[0].c_str());
    if (pos_args.size() >= 2) N_COLS = std::atoi(pos_args[1].c_str());
    if (pos_args.size() >= 3) ALPHA = std::atof(pos_args[2].c_str());

    MatrixGenerator gen(42);
    gen.generate_factor_base(N_COLS);
    HostMatrix A_raw = gen.generate_matrix(N_ROWS, ALPHA);
    HostMatrix A = sort_matrix_by_density(A_raw);
    HostMatrix AT = MatrixGenerator::transpose(A_raw);

    // List of bit widths to test
    std::vector<int> widths = {32, 64, 128, 256, 512};
    
    struct Result {
        int width;
        double time_A;
        double gnnz_A;  // Speed A
        double time_AT;
        double gnnz_AT; // Speed AT
        double score;   // Optimization metric
    };
    std::vector<Result> results;
    size_t total_nnz = 0;
    for(const auto& r : A.rows) total_nnz += r.size();

    LOG(LOG_INFO) << "[FullBenchnmark] ==========================================================" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  META-AUTOTUNER: Testing Widths {32, 64, 128, 256, 512}" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  AutoTuner: " << (use_gpu_autotuner ? "GPUAutoTuner (--gpu-only)" : "SpMMAutoTuner (--legacy)") << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  Optimization Metric: Maximize (Width * GNNz/s_Combined)" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark] ==========================================================" << std::endl;

    for (int width : widths) {
        LOG(LOG_INFO) << "[FullBenchnmark] >>> Testing Bit Width: " << width << std::endl;
        
        size_t v_bytes = N_COLS * (width / 8);
        size_t c_bytes = N_ROWS * (width / 8);
        uint64_t *d_V, *d_C;
        CUDA_CHECK(cudaMalloc(&d_V, v_bytes));
        CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
        
        // Use a dummy non-random vector for performance measurement for stability
        CUDA_CHECK(cudaMemset(d_V, 0xAA, v_bytes));

        // 1. Optimize A (VERBOSE=true)
        ExecutionPlan planA;
        if (use_gpu_autotuner) {
            GPUAutoTuner::Config gpu_conf;
            planA = GPUAutoTuner::tune(A_raw, false, width, gpu_conf, true);
        } else {
            planA = SpMMAutoTuner::tune_global(A, false, width, true);
        }
        OptimizedSpMM opt_A;
        opt_A.compile(use_gpu_autotuner ? A_raw : A, planA);
        opt_A.tune_execution_strategy(d_C, d_V);

        // 2. Optimize AT (VERBOSE=true)
        ExecutionPlan planAT;
        if (use_gpu_autotuner) {
            GPUAutoTuner::Config gpu_conf;
            planAT = GPUAutoTuner::tune(AT, true, width, gpu_conf, true);
        } else {
            planAT = SpMMAutoTuner::tune_global(AT, true, width, true);
        }
        OptimizedSpMM opt_AT;
        opt_AT.compile(AT, planAT);
        opt_AT.tune_execution_strategy(d_C, d_V);

        // 3. Hot Benchmarks
        double t_A = run_hot_benchmark(opt_A, d_C, d_V, 50);
        double t_AT = run_hot_benchmark(opt_AT, d_C, d_V, 50);

        // 4. Verify (Once per width)
        {
            LOG(LOG_INFO) << "[FullBenchnmark] Verification Width " << width << ": " << std::endl;
            // Generate Random V using the new generator for verification
            std::vector<uint8_t> h_V = gen.generate_random_vector(N_COLS, width);
            CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), v_bytes, cudaMemcpyHostToDevice));

            // Run A on GPU
            CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
            opt_A.execute(d_C, d_V);
            std::vector<uint8_t> h_C(c_bytes);
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

            // Run A on CPU — use density_perm for GPU autotuner path
            std::vector<uint8_t> h_C_ref(c_bytes);
            if (use_gpu_autotuner) {
                Verification::cpu_spmm_gf2(A_raw, h_V.data(), h_C_ref.data(), planA.density_perm, width);
            } else {
                std::vector<row_idx_t> id_perm(N_ROWS); std::iota(id_perm.begin(), id_perm.end(), 0);
                Verification::cpu_spmm_gf2(A, h_V.data(), h_C_ref.data(), id_perm, width);
            }

            if(!Verification::check_results(h_C.data(), h_C_ref.data(), N_ROWS, width)) {
                LOG(LOG_ERROR_CRITICAL) << "[FullBenchnmark] CRITICAL: Verification Failed for Width " << width << std::endl;
                exit(1);
            } else {
	        LOG(LOG_INFO) << "[FullBenchnmark] Verification PASSED" << std::endl;
            }
        }

        // 5. Calculate Metrics
        double gnnz_A  = (double)total_nnz / 1e9 / (t_A / 1000.0);
        double gnnz_AT = (double)total_nnz / 1e9 / (t_AT / 1000.0);
        
        // Score based on combined throughput (harmonic mean logic relative to total time)
        double total_time_s = (t_A + t_AT) / 1000.0;
        double combined_throughput = (double)total_nnz / 1e9 / total_time_s;
        double score = width * combined_throughput; 
        
        results.push_back({width, t_A, gnnz_A, t_AT, gnnz_AT, score});

        cudaFree(d_V); cudaFree(d_C);
    }

    // Report
    LOG(LOG_INFO) << "[FullBenchnmark] ======================================================================================================" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  FINAL RESULTS SUMMARY (Measured Hot Performance)" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark] ======================================================================================================" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  Width |  Time A  | Speed A (G/s) |  Time AT | Speed AT (G/s) |  Sum (ms) | Score (Width*G/s)" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark] -------|----------|---------------|----------|----------------|-----------|-------------------" << std::endl;

    Result winner = results[0];
    for (const auto& r : results) {
        double sum_time = r.time_A + r.time_AT;
        LOG(LOG_INFO) << "[FullBenchnmark]   " << std::setw(3) << r.width << "  | " 
                  << std::setw(8) << std::fixed << std::setprecision(3) << r.time_A << " | " 
                  << std::setw(13) << std::setprecision(2) << r.gnnz_A << " | "
                  << std::setw(8) << std::fixed << std::setprecision(3) << r.time_AT << " | "
                  << std::setw(14) << std::setprecision(2) << r.gnnz_AT << " | " 
                  << std::setw(9) << std::fixed << std::setprecision(3) << sum_time << " | " 
                  << std::scientific << std::setprecision(3) << r.score << std::defaultfloat << std::endl;
        
        if (r.score > winner.score) winner = r;
    }

    LOG(LOG_INFO) << "[FullBenchnmark] ======================================================================================================" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]  >>> OPTIMAL CONFIGURATION FOUND: " << winner.width << " BITS <<<" << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark]      Score: " << std::scientific << winner.score << std::defaultfloat << std::endl;
    LOG(LOG_INFO) << "[FullBenchnmark] ======================================================================================================" << std::endl;

    return 0;
}
