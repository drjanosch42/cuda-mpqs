// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file sieving_benchmark.cpp
 * @brief Standalone Benchmark and Validation for the GPU Sieving Pipeline.
 * * Replicates the tight loop of the Orchestrator (Siever -> PostProc) 
 * without the overhead of the full Linear Algebra or Sqrt stages.
 * Performs deep mathematical validation on random samples of the output.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>

// Includes from Project
#include "logger/hpc_logger.h"
#include "uint512.cuh"
#include "math_utils.cuh"
#include "common.h"
#include "prime_algorithms.h" // For determineParams, generateFactorBase
#include "device_sieving_controller.h"
#include "postprocessing.h"
#include "fake_relation_generator.h" // Used only to generate a valid N

// For siever output inspection
#include "debug_dump.h"


using namespace mpqs;

// ============================================================================
// Verification Logic
// ============================================================================

/**
 * @brief Mathematically verifies a single relation.
 * Checks: (ax+b)^2 - N == sign * 2^k * Product(factors) * LP
 */
bool verify_relation(const mpqs::structures::Relation& rel, const mpqs::uint512& N, const std::vector<uint32_t>& factorBase) {
    LOG(LOG_DEBUG_1) << "[VerRel] ---- VERIFYING RELATION ----";
    LOG(LOG_DEBUG_1) << "[VerRel] a: " << rel.a.to_string();
    LOG(LOG_DEBUG_1) << "[VerRel] b: " << rel.b.to_string();
    LOG(LOG_DEBUG_1) << "[VerRel] Odd prime divisors count: " << rel.num_factors;
    LOG(LOG_DEBUG_1) << "[VerRel] sign: " << (int)rel.sign_of_Q << ", exp2: " << rel.val_2_exp;

    std::string factor_pairs = "";
    for (uint32_t i = 0; i < rel.num_factors; ++i) {
        factor_pairs += "(" + std::to_string(rel.factors[i].p_index) + ", " 
                            + std::to_string((int)rel.factors[i].count) + ") ";
    }
    LOG(LOG_DEBUG_1) << "[VerRel] " << factor_pairs;
  
    // 1. Calculate LHS = |(ax + b)^2 - N| + the sign of the inner termsign
    mpqs::uint512 lhs;
    int8_t lhs_sign;
    
    mpqs::math::calculateQ_unsigned(rel.a, rel.b, rel.x, N, lhs, lhs_sign);
    
    // 2. Calculate RHS
    // RHS = 2^val_2_exp * Product(p_i) * LP * sign
    mpqs::uint512 rhs((uint32_t)1);

    // 2.1 Power of 2
    for(uint32_t i = 0; i < rel.val_2_exp; ++i) {
        rhs.mult_uint32(2);
    }

    // 2.2 Factors
    for(uint32_t i = 0; i < rel.num_factors; ++i) {
        uint32_t p_idx = rel.factors[i].p_index;
        uint32_t p = factorBase[p_idx];
        uint8_t count = rel.factors[i].count;
        
        for(uint8_t k=0; k<count; ++k) {
            rhs.mult_uint32(p);
        }
    }

    // 2.3 Large Prime Remainder
    if (rel.large_prime_remainder != 1) {
        // large_prime_remainder is u128, might need multiple mults if > u32
        // Assuming simple mult for benchmark (implementing full uint512 * u128 is verbose)
        // Here we hack it: cast to string or u64 parts if u128 support in uint512 is missing.
        // uint512 supports `mult_uint32`.
        unsigned __int128 lp = rel.large_prime_remainder;
        
        // Split into 32-bit chunks for safety
        mpqs::uint512 lp_mpqs;
        // Simple assignment if supported, else manual reconstruction
        uint64_t lo = (uint64_t)lp;
        uint64_t hi = (uint64_t)(lp >> 64);
        
        mpqs::uint512 t_lo(lo);
        mpqs::uint512 t_hi(hi);
        t_hi.lshift(64);
        t_lo.add(t_hi);
        lp_mpqs = t_lo;

        rhs.mult(lp_mpqs);
    }

    bool passed = true;

    // 2.4 Sign handling
    if (lhs_sign != rel.sign_of_Q) {
        LOG(LOG_DEBUG_2) << "[VerRel] SIGN MISMATCH!";
        passed = false;
    }
    if (lhs != rhs) {
        LOG(LOG_DEBUG_2) << "[VerRel] |Q(X)| MISMATCH!";
        passed = false;
    }
    {
        if(passed)
	    LOG(LOG_DEBUG_1) << "[VerRel] RELATION VALID.";
	else
	    LOG(LOG_ERROR_CRITICAL) << "[VerRel] RELATION INVALID!";
    }
    return passed;
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    // 1. Logger Setup
    LogConfig log_cfg;
    SinkConfig console;
    console.type = SinkConfig::CONSOLE;
    console.min_severity = LOG_DEBUG_2;
    log_cfg.sinks.push_back(console);
    HPCLogger::Get().Init(log_cfg);
    LOG_SET_STAGE(LOG_STAGE_SIEVE);

    LOG(LOG_INFO) << "[SieveBench] ==========================================";
    LOG(LOG_INFO) << "[SieveBench]  MPQS Sieving & Post-Processing Benchmark";
    LOG(LOG_INFO) << "[SieveBench] ==========================================";

    // 2. Configuration & Data Generation
    int bit_size = 256; // Default from sqrt_benchmark
    if (argc > 1) bit_size = std::atoi(argv[1]);

    int device_id = 0;

    // LOG(LOG_INFO) << "Generating " << bit_size << "-bit semiprime N...";
    // Use the generator to get a valid N, but we ignore the fake relations
    // mpqs::test::FakeRelationGenerator generator(bit_size, 1000); 
    // auto fake_data = generator.generate(0); // 0 relations needed
    // RSA-100
    //const char* n_str_RSA100 = "1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139";
    //mpqs::uint512 N(n_str_RSA100);
    const char* n_str = "6024065079889642469495026789749787328504528247460180000248150504804066095061017";
    mpqs::uint512 N(n_str);
    
    mpqs::sieve::factoringData f_data;
    f_data.N = N; //fake_data.N;
    LOG(LOG_INFO) << "[SieveBench] N = " << f_data.N.to_string();

    // 3. Tuning (Host)
    LOG(LOG_INFO) << "[SieveBench] Determining parameters...";
    determineParams(&f_data); // Calculate F, M, etc.
    LOG(LOG_INFO) << "[SieveBench] M (Interval)  : " << f_data.M;
    LOG(LOG_INFO) << "[SieveBench] F (FactorBase): " << f_data.F;

    LOG(LOG_INFO) << "[SieveBench] Generating Factor Base...";
    generateFactorBase(&f_data);
    LOG(LOG_INFO) << "[SieveBench] Factor Base Size: " << f_data.size;
    
    LOG(LOG_INFO) << "[SieveBench] Initializing Polynomials...";
    init_a_factors(&f_data);

    // 4. Component Initialization
    LOG(LOG_INFO) << "[SieveBench] Initializing GPU Components...";
    auto siever = std::make_unique<mpqs::sieve::DeviceSievingController>(device_id);
    siever->initiate(f_data);
    siever->loadStandardConfig();
    siever->loadData();
    // Upload 'a', 'B_values', and 'a_factors' to the allocated memory
    siever->updateState();
    
    siever->printConfigs();
    if (!(siever->validateConfigs())) {
        std::cerr << "[SieveBench] Configuration invalid!" << std::endl;
	exit(-1);
    }

    mpqs::postprocessing::PostProcConfig pp_conf;
    // The following two parameters influence the throughput
    // of sieving and batch factorization
    pp_conf.batch_size_threshold = 65536*4;
    pp_conf.batch_purge_threshold = (9*pp_conf.batch_size_threshold) / 10;

    // Hypercube dimension for a
    pp_conf.shc_dim = siever->getFactoringData().a_factors.size();
    // Enable (simultaneous) on device Relation collection.
    pp_conf.persistent_device_buffer_size = siever->getFactoringData().size + 4096;
    uint32_t target = siever->getFactoringData().size + 128;

    // Access device pointers
    auto dev_ptrs = siever->getDevicePointers();

    auto post_proc = std::make_unique<mpqs::postprocessing::DevicePostProcessingController>();
    post_proc->initiate(f_data, dev_ptrs, pp_conf);

    // 5. The Sieving Loop
    LOG(LOG_INFO) << "[SieveBench] Starting Sieve Loop...";
    std::vector<mpqs::structures::Relation> relations;
    relations.reserve(pp_conf.persistent_device_buffer_size);

    int steps = 0;
    int last_purge = 0;
    
    cudaStream_t stream = 0; // Default stream

    auto t_start = std::chrono::high_resolution_clock::now();

    size_t current_a_factors_size = siever->getFactoringData().a_factors.size();

    while ((relations.size() < target) && (steps < 587)) {
        if(steps == 586) { // 586
        
	    LOG(LOG_DEBUG_2) << "[SieveBench] step " << steps << " a factors size: " << siever->getFactoringData().a_factors.size();
            std::string a_factors_str = "";
	    for (uint32_t i = 0; i < siever->getFactoringData().a_factors.size(); ++i) {
	        a_factors_str +=  std::to_string(siever->getFactoringData().a_factors[i]) + ", ";
	    }
	    LOG(LOG_DEBUG_2) << "[SieveBench] step " << steps << " a factors: " << a_factors_str;
	}
     
        // A. Sieve
        siever->sieveStep();
	//cudaDeviceSynchronize();
	
        if(steps == 586) { // 586
	    dumpCandidatesJSON(*siever);
	}
        if(steps == 586) { // 586
	    siever->validateResults(f_data);
	}
	steps++;

        // B. Accumulate
        bool buffer_full = post_proc->accumulate(
            siever->getRawCandidates(),
            siever->getRawCandidateBufferSize(),
            siever->getFactoringData().a,
            siever->getDeviceA_Factors(),
            (uint32_t)siever->getFactoringData().a_factors.size(),
            -(int32_t)siever->getFactoringData().M,
            stream
        );
	if(!((steps-last_purge) % 100) || true)
	    LOG(LOG_DEBUG_1) << "[SieveBench] Step " << steps << ": Relations (host): " << relations.size() << ", accumulated relation candidates: " << post_proc->getAccumulatedCount();

        // C. Process
        if (buffer_full) {
	    LOG(LOG_DEBUG_1) << "[SieveBench] Relation Candidate Buffer Fill threshold triggered at: " << post_proc->getAccumulatedCount() << " relation candidates.";
            post_proc->processBufferedCandidates();
            post_proc->retrieveRelations(relations);
	    last_purge = steps;
        }

        // D. Advance
        siever->advance_a(1);
        siever->updateState();
	size_t new_a_factors_size = siever->getFactoringData().a_factors.size();
	if(new_a_factors_size != current_a_factors_size) {
	    LOG(LOG_DEBUG_2) << "[SieveBench] Step " << steps << ": a_factors size changed from: " << current_a_factors_size << " to " << new_a_factors_size;
	    current_a_factors_size = new_a_factors_size;
	}
    }
    LOG(LOG_INFO) << "[SieveBench] Sieving complete.";

    // Flush remaining
    post_proc->flush();
    post_proc->retrieveRelations(relations);

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(t_end - t_start).count();

    LOG(LOG_INFO) << "[SieveBench] Sieving Loop finished after " << steps << " steps in " << duration << " seconds.";
    LOG(LOG_INFO) << "[SieveBench] Throughput: " << (relations.size() / duration) << " rels/sec; "
		  << (relations.size() / steps) << " rels/step";
    LOG(LOG_INFO) << "[SieveBench] Relations found: " << relations.size();

    // 6. Validation (Random Sampling)
    LOG(LOG_INFO) << "[SieveBench] Validating random sample of relations...";
    
    int sample_count = std::min((int)relations.size(), 50);
    int valid_count = 0;
    
    // Random engine
    std::mt19937 rng(12345);
    std::vector<int> indices(relations.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for(int i=0; i<sample_count; ++i) {
        const auto& rel = relations[indices[i]];
        if (verify_relation(rel, f_data.N, f_data.factorBase)) {
            valid_count++;
        } else {
            LOG(LOG_ERROR_CRITICAL) << "[SieveBench] Verification FAILED for relation ID " << rel.relation_index;
            LOG(LOG_ERROR_CRITICAL) << "[SieveBench]   a: " << rel.a.to_string();
            LOG(LOG_ERROR_CRITICAL) << "[SieveBench]   b: " << rel.b.to_string();
            LOG(LOG_ERROR_CRITICAL) << "[SieveBench]   x: " << rel.x;
        }
    }

    LOG(LOG_INFO) << "[SieveBench] Validation Results: " << valid_count << " / " << sample_count << " passed.";

    // Cleanup
    post_proc->clearBuffers();
    siever->clearSievingBuffers();

    return (valid_count == sample_count) ? 0 : 1;
}
