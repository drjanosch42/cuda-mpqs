// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "bw_solver.h"
#include "bw_version.h"
#include "hpc_logger.h"
#include <random>
#include <vector>
#include <algorithm>

using namespace lingen;

// --- Matrix Generation Helper ---
HostMatrix generate_exact_rank_matrix(int N, int rank, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    
    int stride = (N + 63) / 64;
    std::vector<uint64_t> L(N * stride, 0);
    std::vector<uint64_t> U(N * stride, 0);
    
    auto set_bit = [&](std::vector<uint64_t>& m, int r, int c) {
        m[r * stride + (c / 64)] |= (1ULL << (c % 64));
    };
    auto get_bit = [&](const std::vector<uint64_t>& m, int r, int c) {
        return (m[r * stride + (c / 64)] >> (c % 64)) & 1;
    };

    // L (Lower Unit) and U (Upper Unit)
    for (int i = 0; i < N; ++i) {
        set_bit(L, i, i); 
        set_bit(U, i, i);
        for (int j = 0; j < i; ++j) if (dist(rng)) set_bit(L, i, j);
        for (int j = i + 1; j < N; ++j) if (dist(rng)) set_bit(U, i, j);
    }

    // B = L * D * U (D has 'rank' ones)
    std::vector<uint64_t> B(N * stride, 0);
    for(int i=0; i<N; ++i) {
        for(int k=0; k<rank; ++k) {
            if(get_bit(L, i, k)) {
                for(int w=0; w<stride; ++w) B[i*stride+w] ^= U[k*stride+w];
            }
        }
    }

    // Convert to HostMatrix
    HostMatrix A;
    A.n_rows = N; 
    A.n_cols = N;
    A.rows.resize(N);
    
    for(int r=0; r<N; ++r) {
        for(int c=0; c<N; ++c) {
            if(get_bit(B, r, c)) A.rows[r].push_back(c);
        }
    }
    return A;
}

/**
 * @brief Smoke Test / Reference Generator.
 * 
 * Usage:
 *   ./bw_lingen_smoke <device> <prefix>
 *   ./bw_lingen_smoke --golden
 * 
 * Golden Mode:
 *   Runs a deterministic 512x512 (N=512, n=64) test with fixed seed 12345.
 *   Output prefix is "golden".
 *   Enables "reference_check_mode" in solver.
 */
int main(int argc, char** argv) {
    int seed = 12345;
    int device = 0;
    std::string prefix = "test"; 
    bool golden_mode = false;

    int m_block = 64;
    int n_block = 64;

    // Argument Parsing
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--golden") {
            golden_mode = true;
        } else if (arg == "--m" && i+1 < argc) {
            m_block = std::atoi(argv[++i]);
        } else if (arg == "--n" && i+1 < argc) {
            n_block = std::atoi(argv[++i]);
        } else if (i == 1 && argv[i][0] != '-') {
            device = std::atoi(argv[i]);
        } else if (i == 2 && argv[i][0] != '-') {
            prefix = argv[i];
        }
    }
    
    if (m_block == 0) m_block = n_block; // Fallback if user only sets --n

    LogConfig log_cfg;
    log_cfg.enable_cout = true;
    log_cfg.min_severity_cout = LOG_DEBUG_2;
    HPCLogger::Get().Init(log_cfg);

    LOG_SET_STAGE(LOG_STAGE_BW_INITIALIZATION, "LinAlg");
    if (golden_mode) {
        LOG(LOG_INFO) << "=== Block Wiedemann GOLDEN RUN Mode " << lingen::version_string() << " ===";
        LOG(LOG_INFO) << "Setting deterministic parameters (m=n=64) for regression testing.";
        prefix = "golden";
        device = 0; 
        m_block = 64;
        n_block = 64;
    } else {
        LOG(LOG_INFO) << "=== Block Wiedemann Smoke Test " << lingen::version_string() << " ===";
        LOG(LOG_INFO) << "Prefix: " << prefix << ", m=" << m_block << ", n=" << n_block;
    }    

    // 1. Generate Problem
    // Default: N = 4*512 = 2048. Golden: N = 512.
    int N = golden_mode ? 512 : 4096;        
    int Rank = N - 5;   
    HostMatrix A = generate_exact_rank_matrix(N, Rank, seed);

    // Save A for Python verification
    io::BWIOSystem io(prefix);
    io.save_matrix_A(A);

// 2. Configure Solver
    BWSolverConfig config;
    config.device_id = device;
    config.nrows = N;
    config.m_block = m_block; // Set m
    config.n_block = n_block; // Set n
    config.checkpoint_prefix = prefix;
    config.seed = seed;

    // We solve Ax=0 (Right Kernel)
    config.solve_transposed = false;

    // --- Execution Mode ---
    // Enable GPU Coppersmith (Stage 2)
    config.stage2_gpu_mode = true;

    // --- Validation & Verification ---
    // Enable hashing to generate logs for external Python verification
    config.enable_all_hashing = true;
    
    // Enable internal strict Oracle verification.
    // This forces self-consistent "Record -> Verify" loops in Stage 2 and 3.
    config.enable_all_oracle_verification = true;

    // Stage 2: Annihilation Checks
    config.stage2_check_annihilation_gpu = true;
    // Legacy CPU check is extremely slow, enable only for the small Golden run
    config.stage2_check_annihilation_legacy = golden_mode;
    
    // Stage 2: Reference State Checks (Log internal state hashes for regression)
    config.stage2_reference_check = golden_mode;

    // --- Auto-Tune & Pre-Flight ---
    // IMPORTANT: Disable SpMM tuning for Smoke/Golden tests.
    // Tuning might reorder the matrix (permutation), which breaks bit-exact 
    // comparison with the external Python script (which assumes natural ordering).
    config.autotune_tune_spmm = false; 
    config.autotune_tune_poly = false;
    
    // Force Pre-Flight SpMM correctness check (A*x)
    config.autotune_verify_spmm = true;

    // --- Stage Settings ---
    config.stage1_seq_len = 0; // Auto-calculate based on dimensions

    // Request exactly 5 linearly independent solutions
    config.stage3_max_solutions = 5;
    
    // Disable un-permutation since tuning (and thus permutation) is disabled.
    config.stage3_perform_unpermutation = false;

    // 3. Launch
    BlockWiedemannSolver solver(config, A);
    solver.Solve();
    
    LOG(LOG_INFO) << "=== Smoke Test Complete ===";
    LOG(LOG_INFO) << "Verify with: python verify_bw_pipeline.py " << prefix;
    
    return 0;
}


