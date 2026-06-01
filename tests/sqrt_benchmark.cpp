// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file sqrt_benchmark.cpp
 * @brief Test Driver for the Square Root Refinement Step using Synthetic Data.
 * 
 * Benchmarks the final step of the MPQS algorithm:
 * Given a set of relations found by sieving and a dependency vector found by 
 * linear algebra (Block Wiedemann), compute the actual factors of N.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib> // for std::atoi

// Project Headers
#include "logger/hpc_logger.h"
#include "uint512.cuh"
#include "fake_relation_generator.h"
#include "sqrt_step.h"

// Namespace shortcuts
using namespace mpqs;
using namespace mpqs::sqrt;

// Helper to print duration
void print_timer(const std::chrono::steady_clock::time_point& start, const std::string& label) {
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    LOG(LOG_INFO) << label << ": " << ms << " ms";
}

int main(int argc, char** argv) {
    // 1. Configure Logging
    LogConfig log_cfg;
    SinkConfig console;
    console.type = SinkConfig::CONSOLE;
    console.min_severity = LOG_INFO;
    log_cfg.sinks.push_back(console);
    HPCLogger::Get().Init(log_cfg);
    LOG_SET_STAGE(LOG_STAGE_BW_POSTPROCESSING); // Closest stage to sqrt

    // 2. Parse Arguments
    int bit_size = 256;       // Default modulus size (small for fast verification)
    int fb_size = 1000;       // Size of factor base
    int num_relations = 1100; // Relations to generate

    if (argc > 1) bit_size = std::atoi(argv[1]);
    if (argc > 2) fb_size = std::atoi(argv[2]);
    if (argc > 3) num_relations = std::atoi(argv[3]);

    LOG(LOG_INFO) << "==============================================";
    LOG(LOG_INFO) << " MPQS Square Root Step Validation & Benchmark";
    LOG(LOG_INFO) << "==============================================";
    LOG(LOG_INFO) << "Modulus Bits  : " << bit_size;
    LOG(LOG_INFO) << "Factor Base   : " << fb_size << " primes";
    LOG(LOG_INFO) << "Relations     : " << num_relations;

    // 3. Generate Fake Data
    // -------------------------------------------------------------------------
    LOG(LOG_INFO) << "Generating synthetic problem instance...";
    auto gen_start = std::chrono::steady_clock::now();

    mpqs::test::FakeRelationGenerator generator(bit_size, fb_size);
    mpqs::test::FakeData data = generator.generate(num_relations);

    print_timer(gen_start, "Data Generation");

    LOG(LOG_INFO) << "Modulus N: " << data.N.to_string();
    LOG(LOG_INFO) << "Secret P : " << data.p.to_string();
    LOG(LOG_INFO) << "Secret Q : " << data.q.to_string();

    // 4. Run Square Root Step
    // -------------------------------------------------------------------------
    LOG(LOG_INFO) << "Starting Square Root Refinement...";
    
    // Instantiate the solver
    SquareRootRefinement sqrt_solver(data.N);

    auto solve_start = std::chrono::steady_clock::now();

    // Perform the calculation
    // Note: data.factor_base is std::vector<uint32_t> (odd primes only)
    // data.solution_bits mimics the output of Block Wiedemann (all 1s or specific subset)
    std::pair<mpqs::uint512, mpqs::uint512> factors = sqrt_solver.Perform(
        data.solution_bits, 
        data.relations, 
        data.factor_base
    );

    print_timer(solve_start, "Square Root Step");

    // 5. Validate Results
    // -------------------------------------------------------------------------
    bool success = true;

    LOG(LOG_INFO) << "Found Factors:";
    LOG(LOG_INFO) << "  F1: " << factors.first.to_string();
    LOG(LOG_INFO) << "  F2: " << factors.second.to_string();

    // Check 1: Non-Triviality
    if (factors.first.is_one() || factors.first == data.N) {
        LOG(LOG_ERROR_CRITICAL) << "FAILURE: Trivial factor found (1 or N).";
        success = false;
    }

    // Check 2: Correctness (F1 * F2 == N)
    // We use a copy to perform multiplication to avoid modifying the result vars
    mpqs::uint512 product = factors.first;
    product.mult(factors.second); // Wraps at 512 bits, but N fits in 512 bits so it is exact

    if (product != data.N) {
        LOG(LOG_ERROR_CRITICAL) << "FAILURE: F1 * F2 != N";
        LOG(LOG_ERROR_CRITICAL) << "  Computed Product: " << product.to_string();
        success = false;
    } else {
        LOG(LOG_INFO) << "Check: Product matches N.";
    }

    // Check 3: Match Known Seeds (Optional but good for benchmarks)
    // Since N = P * Q, the factors found must be {P, Q} or {Q, P}
    if (success) {
        bool match_p = (factors.first == data.p) || (factors.second == data.p);
        bool match_q = (factors.first == data.q) || (factors.second == data.q);

        if (match_p && match_q) {
            LOG(LOG_INFO) << "SUCCESS: Factors match the secret generator seeds.";
        } else {
            // This is theoretically impossible for N=pq unless GCD failed weirdly
            LOG(LOG_WARNING) << "Result correct (Product=N), but factors distinct from seed?";
        }
    }

    LOG(LOG_INFO) << "=========================================================";
    return success ? 0 : 1;
}

