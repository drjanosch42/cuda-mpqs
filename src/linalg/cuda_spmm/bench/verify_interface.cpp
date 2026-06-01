// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "bw_spmm_interface.h"
#include "generator.h"
#include "verification.h"
#include "hpc_logger.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <iomanip>

void random_vector(std::vector<uint64_t>& v, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for(size_t i=0; i<n; ++i) v[i] = rng();
}

bool test_matrix_config(row_idx_t rows, idx_t cols, int bit_width, double alpha) {
    LOG(LOG_INFO) << "[VerifySpmmInterface] ================================================================================================" << std::endl;
    LOG(LOG_INFO) << "[VerifySpmmInterface]  VERIFICATION TEST: " << rows << "x" << cols << " (" << bit_width << "-bit, Alpha=" << alpha << ")" << std::endl;
    LOG(LOG_INFO) << "[VerifySpmmInterface] ================================================================================================" << std::endl;

    MatrixGenerator gen(12345);
    gen.generate_factor_base(cols);
    HostMatrix A = gen.generate_matrix(rows, alpha, true); 

    idx_t padded_cols = (cols + 63) / 64 * 64;
    BlockWiedemannSpMM bw(A, padded_cols, true, true);
    
    // 1. Setup and Get Performance Report
    SpMMPerformanceReport report = bw.setup_and_benchmark(bit_width, bit_width);
    
    // --- Custom Logging (Requirement 1, 2, 3) ---
    LOG(LOG_INFO) << "[VerifySpmmInterface] System Statistics:" << std::endl;
    
    // Memory
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Memory Host]   A: " << format_bytes(bw.get_host_memory_bytes(false)) 
              << " | AT: " << format_bytes(bw.get_host_memory_bytes(true)) << std::endl;
    
    // Note: To get device memory, matrices must be loaded. setup_and_benchmark loads/unloads them.
    // However, we cached the peak usage in the report if possible? No, we added methods to query current state.
    // But setup_and_benchmark unloads them. 
    // Optimization: Reload them momentarily or just rely on the fact that verification requires them loaded.
    // Actually, verification steps below will load them. We can print memory there.
    // Alternatively, I added code in setup_and_benchmark to measure it internally. Not exposed?
    // Let's rely on the verification phase loading to print device stats.

    // Performance
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Throughput]    A: " << std::fixed << std::setprecision(2) << report.throughput_A_gnnz << " GNNz/s"
              << " | AT: " << report.throughput_AT_gnnz << " GNNz/s" << std::endl;
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Time/Op]       A: " << std::fixed << std::setprecision(3) << report.time_A_ms << " ms"
              << " | AT: " << report.time_AT_ms << " ms" << std::endl;

    // Kernels (Need to load to query)
    bw.load_A_to_device(bit_width);
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Kernels A]     " << bw.get_kernel_summary(false) << std::endl;
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Memory Dev A]  " << format_bytes(bw.get_device_memory_bytes(false)) << std::endl;
    bw.unload_A_from_device();

    bw.load_AT_to_device(bit_width);
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Kernels AT]    " << bw.get_kernel_summary(true) << std::endl;
    LOG(LOG_INFO) << "[VerifySpmmInterface]  [Memory Dev AT] " << format_bytes(bw.get_device_memory_bytes(true)) << std::endl;
    bw.unload_AT_from_device();

    // --- Buffer Sizing ---
    size_t v_bytes = padded_cols * (bit_width / 8);
    size_t c_bytes = rows * (bit_width / 8);
    
    std::vector<uint8_t> h_V_bytes = gen.generate_random_vector(padded_cols, bit_width);
    std::vector<uint8_t> h_C_ref_bytes(c_bytes);
    std::vector<uint8_t> h_C_gpu_bytes(c_bytes);
    
    // =========================================================
    // TEST 1: Execute A
    // =========================================================
    LOG(LOG_INFO) << "[VerifySpmmInterface] Testing Sequence A..." << std::endl;
    std::vector<row_idx_t> identity_p(rows);
    std::iota(identity_p.begin(), identity_p.end(), 0);
    std::memset(h_C_ref_bytes.data(), 0, c_bytes);
    Verification::cpu_spmm_gf2(A, h_V_bytes.data(), h_C_ref_bytes.data(), identity_p, bit_width);

    bw.load_A_to_device(bit_width);
    uint64_t *d_V, *d_C;
    CUDA_CHECK(cudaMalloc(&d_V, v_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    
    std::vector<uint8_t> h_V_perm(v_bytes);
    bw.preprocess_domain_A(h_V_bytes.data(), h_V_perm.data(), padded_cols, bit_width);
    CUDA_CHECK(cudaMemcpy(d_V, h_V_perm.data(), v_bytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
    bw.execute_A(d_C, d_V);
    
    std::vector<uint8_t> h_C_perm(c_bytes);
    CUDA_CHECK(cudaMemcpy(h_C_perm.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));
    bw.postprocess_domain_A(h_C_perm.data(), h_C_gpu_bytes.data(), rows, bit_width);
    
    bw.unload_A_from_device();
    cudaFree(d_V); cudaFree(d_C);
    
    bool pass_A = Verification::check_results(h_C_gpu_bytes.data(), h_C_ref_bytes.data(), rows, bit_width);
    {
        if(pass_A) LOG(LOG_INFO) << "[VerifySpmmInterface] [SUCCESS] A Sequence Matches." << std::endl;
        else { LOG(LOG_ERROR_CRITICAL) << "[VerifySpmmInterface] [FAILED] A Sequence Mismatch." << std::endl; return false; }
    }
    // =========================================================
    // TEST 2: Execute AT
    // =========================================================
    LOG(LOG_INFO) << "[VerifySpmmInterface] Testing Sequence AT..." << std::endl;
    size_t w_bytes = rows * (bit_width/8); 
    size_t cat_bytes = padded_cols * (bit_width/8); 
    std::vector<uint8_t> h_W_bytes = gen.generate_random_vector(rows, bit_width);
    
    std::vector<uint8_t> h_Cat_ref(cat_bytes);
    std::vector<uint8_t> h_Cat_gpu(cat_bytes);
    HostMatrix AT = MatrixGenerator::transpose(A);
    std::vector<row_idx_t> identity_pt(AT.n_rows);
    std::iota(identity_pt.begin(), identity_pt.end(), 0);
    
    std::memset(h_Cat_ref.data(), 0, cat_bytes);
    Verification::cpu_spmm_gf2(AT, h_W_bytes.data(), h_Cat_ref.data(), identity_pt, bit_width);

    bw.load_AT_to_device(bit_width);
    CUDA_CHECK(cudaMalloc(&d_V, w_bytes));   
    CUDA_CHECK(cudaMalloc(&d_C, cat_bytes)); 
    
    std::vector<uint8_t> h_W_perm(w_bytes);
    bw.preprocess_domain_AT(h_W_bytes.data(), h_W_perm.data(), rows, bit_width);
    CUDA_CHECK(cudaMemcpy(d_V, h_W_perm.data(), w_bytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_C, 0, cat_bytes));
    bw.execute_AT(d_C, d_V);
    
    std::vector<uint8_t> h_Cat_perm(cat_bytes);
    CUDA_CHECK(cudaMemcpy(h_Cat_perm.data(), d_C, cat_bytes, cudaMemcpyDeviceToHost));
    bw.postprocess_domain_AT(h_Cat_perm.data(), h_Cat_gpu.data(), padded_cols, bit_width);

    bw.unload_AT_from_device();
    cudaFree(d_V); cudaFree(d_C);
    
    bool pass_AT = Verification::check_results(h_Cat_gpu.data(), h_Cat_ref.data(), AT.n_rows, bit_width);
    if(pass_AT) LOG(LOG_INFO) << "[VerifySpmmInterface] [SUCCESS] AT Sequence Matches." << std::endl;
    else { LOG(LOG_ERROR_CRITICAL) << "[VerifySpmmInterface] [FAILED] AT Sequence Mismatch." << std::endl; return false; }

    return true;
}

int main(int argc, char** argv) {

    // Initialize Logger for standalone debugging
    LogConfig cfg;
    cfg.enable_cout = true;
    cfg.min_severity_cout = LOG_DEBUG_2; // Verbose for debugging
    cfg.enable_file = false;             // No file needed for quick tests
    HPCLogger::Get().Init(cfg);

    // Default settings
    row_idx_t n_rows = 200000;
    idx_t n_cols = 200000;
    double alpha = 15.0;
    std::vector<int> bit_widths;

    // Argument Parsing
    if (argc >= 2) n_rows = std::atoll(argv[1]);
    if (argc >= 3) n_cols = std::atoi(argv[2]);
    if (argc >= 4) alpha = std::atof(argv[3]);
    
    if (argc >= 5) {
        for(int i=4; i<argc; ++i) bit_widths.push_back(std::atoi(argv[i]));
    } else {
        bit_widths = {64, 128, 256, 512};
    }

    LOG(LOG_INFO) << "[VerifySpmmInterface] Configuration: Rows=" << n_rows << ", Cols=" << n_cols << ", Alpha=" << alpha << std::endl;
    std::stringstream ss;
    ss << "[VerifySpmmInterface] Bit Widths: ";
    for (int b : bit_widths) {
        ss << b << " ";
    }
    LOG(LOG_INFO) << ss.str();

    bool all_passed = true;
    for(int bits : bit_widths) {
        all_passed &= test_matrix_config(n_rows, n_cols, bits, alpha);
    }

    if (all_passed) {
        LOG(LOG_INFO) << "[VerifySpmmInterface] [FINAL RESULT] ALL TESTS PASSED." << std::endl;
        return 0;
    } else {
        LOG(LOG_ERROR_CRITICAL) << "[VerifySpmmInterface] [FINAL RESULT] SOME TESTS FAILED." << std::endl;
        return 1;
    }
}
