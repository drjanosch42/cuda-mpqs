// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file bw_solver.cu
 * @brief High-Level Orchestrator for the Block Wiedemann Algorithm over GF(2).
 * 
 * This file implements the three-stage pipeline required to find vectors in the 
 * kernel of a large sparse matrix (or its transpose) over GF(2).
 * 
 * Mathematical Pipeline:
 * 
 * 1. **Stage 1 (Krylov Sequence Generation)**:
 *    Computes the sequence of moments \f$ S_k \in \mathbb{F}_2^{n \times n} \f$:
 *    \f[
 *       S_k = X^T \cdot \mathcal{B}^{k+1} \cdot Z
 *    \f]
 *    where \f$ \mathcal{B} \f$ is the linear operator (either \f$ A \f$ or \f$ A^T \f$),
 *    and \f$ X, Z \f$ are random blocks of size \f$ N \times n \f$.
 * 
 * 2. **Stage 2 (Linear Generator / Lingen)**:
 *    Solves the Block Berlekamp-Massey problem to find a generator polynomial matrix 
 *    \f$ \pi(x) \in \mathbb{F}_2[x]^{m \times m} \f$ (with \f$ m \approx 2n \f$) such that:
 *    \f[
 *       S(x) \cdot \pi(x) \equiv 0 \pmod{x^L}
 *    \f]
 * 
 * 3. **Stage 3 (Solution Reconstruction)**:
 *    For a chosen candidate column \f$ u(x) \f$ from \f$ \pi(x) \f$, computes the 
 *    evaluation:
 *    \f[
 *       w = \sum_{k=0}^{\deg \pi} \mathcal{B}^{\deg \pi - k} \cdot Z \cdot u_k
 *    \f]
 *    The vector \f$ w \f$ is then multiplied by \f$ \mathcal{B} \f$ repeatedly ("stripping valuation")
 *    until \f$ \mathcal{B} w = 0 \f$.
 */

#include "bw_solver.h"
#include "bw_version.h"

// Concrete Implementations
#include "lingen/stage1/krylov_generator.h"
#include "lingen/stage2/basecase_solver.h"
#include "lingen/stage2/basecase_solver_reference.h"
#include "lingen/stage3/solution_reconstructor.h"
#include "lingen/operations/bw_operator.h"
#include "lingen/operations/poly_arith_engine.h"
#include "bw_spmm_interface.h"
#include "generator.h"  // from cuda_spmm for random matrix / random vector generation
#include "lingen/io/hash.h"

#include <chrono>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string>

namespace lingen {

// Helper: build SpMMAutoTuneConfig from solver-level BWAutoTuneConfig.
// Extracted from AutoTune() so RunStage1/RunStage3 can reuse it when
// spmm_ hasn't been created yet (standalone stage execution).
static SpMMAutoTuneConfig build_spmm_autotune_config(const BWAutoTuneConfig& at_cfg) {
    SpMMAutoTuneConfig cfg;
    cfg.gpu_only = at_cfg.gpu_only;
    cfg.gpu_config.initial_block_size = at_cfg.initial_block_size;
    cfg.gpu_config.max_block_size     = at_cfg.max_block_size;
    cfg.gpu_config.enable_tiledcoo    = at_cfg.enable_tiled_coo;
    cfg.gpu_config.enable_delta16     = at_cfg.enable_delta16;
    cfg.gpu_config.enable_pfor_be     = at_cfg.enable_pfor_be;
    cfg.gpu_config.enable_golomb      = at_cfg.enable_golomb;
    return cfg;
}

#define CHECK_SOLVER(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[BWSolver] CUDA Error: " << cudaGetErrorString(err) \
                                << " (" << __FILE__ << ":" << __LINE__ << ")"; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

namespace {
/// @brief Binary header for S sequence files (Feature B).
/// Enables validation of magic, version, dimensions, and payload size on load.
struct SSequenceHeader {
    static constexpr uint64_t MAGIC = 0x425753455153ULL; // "BWSEQS"
    static constexpr uint32_t CURRENT_VERSION = 1;
    uint64_t magic = MAGIC;
    uint32_t version = CURRENT_VERSION;
    uint32_t seq_len = 0;
    uint32_t m_block = 0;
    uint32_t n_block = 0;
    uint64_t data_bytes = 0;  // payload size after header
};
} // anonymous namespace

// -----------------------------------------------------------------------------
// Bit Packing Helpers
// -----------------------------------------------------------------------------
static std::vector<uint8_t> unpack_bits(const std::vector<uint64_t>& packed, int n_bits) {
    std::vector<uint8_t> unpacked(n_bits);
    for(int i=0; i<n_bits; ++i) {
        if((packed[i/64] >> (i%64)) & 1) unpacked[i] = 1;
        else unpacked[i] = 0;
    }
    return unpacked;
}

static std::vector<uint64_t> pack_bits(const std::vector<uint8_t>& unpacked) {
    int n_bits = unpacked.size();
    std::vector<uint64_t> packed((n_bits + 63)/64, 0);
    for(int i=0; i<n_bits; ++i) {
        if(unpacked[i]) packed[i/64] |= (1ULL << (i%64));
    }
    return packed;
}

// -----------------------------------------------------------------------------
// Padding helper
// -----------------------------------------------------------------------------

// Pads a rectangular matrix to be square (D x D) where D = max(rows, cols).
// Block Wiedemann requires a square linear operator for iteration.
HostMatrix pad_to_square(const HostMatrix& in) {
    int D = std::max((int)in.n_rows, (int)in.n_cols);
    if ((int)in.n_rows == D && (int)in.n_cols == D) return in;

    LOG(LOG_DEBUG_1) << "[BWSolver] Padding matrix from " << in.n_rows << "x" << in.n_cols 
                  << " to " << D << "x" << D;

    HostMatrix out = in;
    out.n_rows = D;
    out.n_cols = D;
    if (out.rows.size() < (size_t)D) {
        out.rows.resize(D);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Solver Implementation
// -----------------------------------------------------------------------------

BlockWiedemannSolver::BlockWiedemannSolver(BWSolverConfig config, const HostMatrix& A)
    : cfg_(config), A_host_(A), io_(config.checkpoint_prefix) 
{
    CHECK_SOLVER(cudaSetDevice(cfg_.device_id));
    CHECK_SOLVER(cudaStreamCreate(&stream_));

    // Detect Jetson Orin (SM 8.7) and apply resource-appropriate defaults
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, cfg_.device_id);
        cfg_.is_jetson = (prop.major == 8 && prop.minor == 7);

        if (cfg_.is_jetson) {
            LOG(LOG_STATS) << "[BWSolver] Jetson Orin detected (SM 8.7, "
                          << prop.multiProcessorCount << " SMs, "
                          << (prop.totalGlobalMem >> 20) << " MB)";

            // Default m=n=64 on Jetson — memory analysis shows m=n=128 is too tight on 8 GB
            // (Stage 1 Krylov buffers quadruple: ~114 MB @batch=16 vs ~786 MB @batch=64)
            // Skip if user explicitly set block sizes via CLI.
            if (!cfg_.block_size_pinned) {
                cfg_.m_block = 64;
                cfg_.n_block = 64;
            } else {
                LOG(LOG_STATS) << "[BWSolver] User-pinned block sizes: m=" << cfg_.m_block
                              << ", n=" << cfg_.n_block << " (Jetson default override skipped)";
            }

            // Cap batch size: default 64 allocates ~786 MB for d_V_buf alone @N=750K
            if (cfg_.stage1_gpu_batch_size > 16) {
                cfg_.stage1_gpu_batch_size = 16;
            }

            // Cap autotuner block sizes: 65536-row blocks waste time on 8 SMs
            if (cfg_.autotune_max_block_size > 8192) {
                cfg_.autotune_max_block_size = 8192;
            }

            // Keep S on device — avoids unnecessary GPU->CPU->GPU roundtrip
            // (on Jetson unified memory, cudaMalloc and host malloc share the same DRAM,
            //  but the device pointer avoids address-space translation overhead)
            cfg_.stage1_keep_S_on_device = true;
        }
    }

    // Heuristic for Sequence Length if not provided
    if (cfg_.stage1_seq_len == 0) {
        int n = std::max(1, cfg_.n_block);
        int m = std::max(1, cfg_.m_block);
        // Basic heuristic: ~ 2 * N/n
        // For rectangular (m != n), theoretical bound is N/m + N/n. We add a margin, depending on m and n.
        cfg_.stage1_seq_len = (cfg_.nrows / n) + (cfg_.nrows / m) + (m + n) / 4; 
    }

    LOG(LOG_STATS) << "[BWSolver] Initialized on Device " << cfg_.device_id;
    LOG(LOG_STATS) << "[BWSolver] Problem Size: N=" << cfg_.nrows
                  << " (Padded), Block m=" << cfg_.m_block << ", n=" << cfg_.n_block;
    LOG(LOG_STATS) << "[BWSolver] Target Sequence Length L=" << cfg_.stage1_seq_len;
    LOG(LOG_STATS) << "[BWSolver] Solve Mode: "
                  << (cfg_.solve_transposed ? "Transposed (A^Tx=0)" : "Standard (Ax=0)");

    // Synthesize configurations for all stages including autotuning.
    UpdateAllConfigs();    
}

BlockWiedemannSolver::~BlockWiedemannSolver() {
    cudaFree(d_solutions_);  // no-op if nullptr
    if (d_S_sequence_) { cudaFree(d_S_sequence_); d_S_sequence_ = nullptr; }
    if (stream_) cudaStreamDestroy(stream_);
}

std::unique_ptr<BwOperator> BlockWiedemannSolver::make_operator() const {
    if (cfg_.solve_transposed) {
        spmm_->load_AT_to_device(cfg_.n_block);
        return std::make_unique<BwOperatorAT>(*spmm_, cfg_.nrows);
    } else {
        spmm_->load_A_to_device(cfg_.n_block);
        return std::make_unique<BwOperatorA>(*spmm_, cfg_.nrows);
    }
}

void BlockWiedemannSolver::UpdateAutoTuneConfig() {
    autotune_cfg_.tune_spmm = cfg_.autotune_tune_spmm;
    autotune_cfg_.tune_poly = cfg_.autotune_tune_poly;
    autotune_cfg_.skip_if_present = cfg_.autotune_skip_if_present;
    autotune_cfg_.verify_spmm = cfg_.autotune_verify_spmm;
    autotune_cfg_.tuning_db_path = cfg_.autotune_db_path;
    autotune_cfg_.gpu_only = cfg_.autotune_gpu_only;
    autotune_cfg_.m4rm_rows = cfg_.autotune_m4rm_rows;
    autotune_cfg_.skip_m4rm_benchmark = cfg_.autotune_skip_m4rm_benchmark;
    autotune_cfg_.block_growth = cfg_.autotune_block_growth;
    autotune_cfg_.initial_block_size = cfg_.autotune_initial_block_size;
    autotune_cfg_.max_block_size = cfg_.autotune_max_block_size;
    autotune_cfg_.enable_tiled_coo = cfg_.autotune_enable_tiled_coo;
    autotune_cfg_.enable_delta16 = cfg_.autotune_enable_delta16;
    autotune_cfg_.enable_pfor_be = cfg_.autotune_enable_pfor_be;
    autotune_cfg_.enable_golomb = cfg_.autotune_enable_golomb;
}

void BlockWiedemannSolver::UpdateStage1Config() {
    // 1. Invariants & Global Settings
    stage1_cfg_.nrows = cfg_.nrows;
    stage1_cfg_.m_block = cfg_.m_block;
    stage1_cfg_.n_block = cfg_.n_block;
    stage1_cfg_.solve_transposed = cfg_.solve_transposed;
    stage1_cfg_.seed = cfg_.seed;
    stage1_cfg_.checkpoint_prefix = cfg_.checkpoint_prefix;

    // 2. Stage Specifics
    stage1_cfg_.skip = cfg_.stage1_skip;
    stage1_cfg_.seq_len = cfg_.stage1_seq_len;
    stage1_cfg_.gpu_batch_size = cfg_.stage1_gpu_batch_size;
    stage1_cfg_.prefer_faster_op = cfg_.stage1_prefer_faster_op;
    stage1_cfg_.keep_S_on_device = cfg_.stage1_keep_S_on_device;
    stage1_cfg_.save_S_to_disk = cfg_.stage1_save_S_to_disk;
    stage1_cfg_.S_disk_path = cfg_.stage1_S_disk_path;

    // 3. I/O & Checkpoints
    stage1_cfg_.load_checkpoints = cfg_.stage1_load_checkpoints;
    stage1_cfg_.save_checkpoints = cfg_.stage1_save_checkpoints;
    stage1_cfg_.suffix_X = cfg_.stage1_suffix_X;
    stage1_cfg_.suffix_Y = cfg_.stage1_suffix_Y;
    stage1_cfg_.suffix_S = cfg_.stage1_suffix_S;

    // 4. Hashing & Validation (With Global Overrides)
    stage1_cfg_.compute_hashes = cfg_.enable_all_hashing || cfg_.stage1_compute_hashes;
    stage1_cfg_.hash_validation = cfg_.enable_all_validation || cfg_.stage1_hash_validation;
    
    // 5. CUDA Graph Acceleration
    stage1_cfg_.use_cuda_graph = cfg_.graph_enable;
    stage1_cfg_.graph_min_iterations = cfg_.graph_min_iterations;

    stage1_cfg_.hash_X = cfg_.stage1_hash_X;
    stage1_cfg_.hash_Y = cfg_.stage1_hash_Y;
    stage1_cfg_.hash_S = cfg_.stage1_hash_S;
}

void BlockWiedemannSolver::UpdateStage2Config() {
    // 1. Invariants & Global Settings
    stage2_cfg_.m_block = cfg_.m_block;
    stage2_cfg_.n_block = cfg_.n_block;
    stage2_cfg_.checkpoint_prefix = cfg_.checkpoint_prefix;

    // 2. Stage Specifics
    // stage2_cfg_.skip = cfg_.stage2_skip;
    
    // Default to Stage 1 length if Stage 2 length not explicitly set
    stage2_cfg_.seq_len = (cfg_.stage2_seq_len > 0) ? cfg_.stage2_seq_len : cfg_.stage1_seq_len;
    
    stage2_cfg_.delta = cfg_.stage2_delta;
    stage2_cfg_.gpu_mode = cfg_.stage2_gpu_mode;
    stage2_cfg_.load_S_from_disk = cfg_.stage2_load_S_from_disk;
    stage2_cfg_.S_disk_path = cfg_.stage2_S_disk_path;

    // 3. Verification & Checks (With Global Overrides)
    stage2_cfg_.check_annihilation_gpu = cfg_.stage2_check_annihilation_gpu;
    stage2_cfg_.check_annihilation_legacy = cfg_.stage2_check_annihilation_legacy;
    stage2_cfg_.reference_check = cfg_.stage2_reference_check;
    stage2_cfg_.reference_check_interval = cfg_.stage2_reference_check_interval;
    stage2_cfg_.post_run_legacy_check = cfg_.stage2_post_run_legacy_check;
    
    // Oracle Override
    stage2_cfg_.internal_oracle_verification = cfg_.enable_all_oracle_verification || cfg_.stage2_internal_oracle_verification;

    // 4. Hashing & Validation (With Global Overrides)
    stage2_cfg_.compute_hashes = cfg_.enable_all_hashing || cfg_.stage2_compute_hashes;
    stage2_cfg_.hash_validation = cfg_.enable_all_validation || cfg_.stage2_hash_validation;
    
    stage2_cfg_.expected_hash_S = cfg_.stage2_hash_S;
    stage2_cfg_.expected_hash_Pi = cfg_.stage2_hash_Pi;

    // 5. I/O
    stage2_cfg_.save_checkpoints = cfg_.stage2_save_checkpoints;
    stage2_cfg_.suffix_S = cfg_.stage2_suffix_S;
    stage2_cfg_.suffix_Pi = cfg_.stage2_suffix_Pi;

    // 6. CUDA Graph Acceleration
    stage2_cfg_.use_cuda_graph = cfg_.graph_enable;
    stage2_cfg_.graph_min_iterations = cfg_.graph_min_iterations;
}

void BlockWiedemannSolver::UpdateStage3Config() {
    // 1. Invariants & Global Settings
    stage3_cfg_.nrows = cfg_.nrows;
    stage3_cfg_.m_block = cfg_.m_block;
    stage3_cfg_.n_block = cfg_.n_block;
    stage3_cfg_.solve_transposed = cfg_.solve_transposed;
    stage3_cfg_.checkpoint_prefix = cfg_.checkpoint_prefix;

    // 2. Stage Specifics
    // Note: s3_skip is handled in the RunStage3 logic wrapper, but could be passed here if the struct supported it.
    // Since BWStage3Config is primarily for the Reconstructor class, and skipping happens at Solver level, 
    // we focus on Reconstructor parameters here.
    
    stage3_cfg_.batch_mode = cfg_.stage3_batch_mode;
    stage3_cfg_.max_solutions = cfg_.stage3_max_solutions;
    stage3_cfg_.perform_unpermutation = cfg_.stage3_perform_unpermutation;
    
    // Tuning
    stage3_cfg_.history_depth = cfg_.stage3_history_depth;
    stage3_cfg_.check_interval = cfg_.stage3_check_interval;
    stage3_cfg_.stripping_step_limit = cfg_.stage3_stripping_limit;

    // 3. Verification (With Global Overrides)
    stage3_cfg_.oracle_verification = cfg_.enable_all_oracle_verification || cfg_.stage3_oracle_verification;

    // 4. Hashing & Validation (With Global Overrides)
    stage3_cfg_.compute_hashes = cfg_.enable_all_hashing || cfg_.stage3_compute_hashes;
    stage3_cfg_.hash_validation = cfg_.enable_all_validation || cfg_.stage3_hash_validation;
    
    stage3_cfg_.expected_hash_first_solution = cfg_.stage3_hash_first_solution;

    // 5. I/O
    stage3_cfg_.save_solutions = cfg_.stage3_save_solutions;
    stage3_cfg_.suffix_Y = cfg_.stage3_suffix_Y;
    stage3_cfg_.suffix_Pi = cfg_.stage3_suffix_Pi;
    stage3_cfg_.suffix_solutions = cfg_.stage3_suffix_solutions;

    // 6. CUDA Graph Acceleration
    stage3_cfg_.use_cuda_graph = cfg_.graph_enable;
    stage3_cfg_.graph_min_iterations = cfg_.graph_min_iterations;
}

void BlockWiedemannSolver::UpdateAllConfigs() {
    UpdateAutoTuneConfig();
    UpdateStage1Config();
    UpdateStage2Config();
    UpdateStage3Config();
}

void BlockWiedemannSolver::AutoTune() {

    LOG(LOG_STATS) << "[AutoTune] Configuration:";
    LOG(LOG_STATS) << "[AutoTune]  Tune SpMM: " << (autotune_cfg_.tune_spmm ? "YES" : "NO")
                  << ", Tune Poly: " << (autotune_cfg_.tune_poly ? "YES" : "NO");
    LOG(LOG_STATS) << "[AutoTune]  Skip Existing: " << (autotune_cfg_.skip_if_present ? "YES" : "NO")
                  << ", DB Path: '" << autotune_cfg_.tuning_db_path << "'";
    LOG(LOG_STATS) << "[AutoTune]  Verify SpMM (Pre-Flight): " << (autotune_cfg_.verify_spmm ? "YES" : "NO");
    LOG(LOG_STATS) << "[AutoTune]  GPU-Only Pipeline: " << (autotune_cfg_.gpu_only ? "YES" : "NO");

    if (autotune_cfg_.tune_spmm) {
        LOG(LOG_STATS) << "[BWSolver] Starting SpMM Auto-Tuning and Optimization...";
        uint32_t m = (cfg_.solve_transposed ? cfg_.n_block : cfg_.m_block);
        uint32_t n = (cfg_.solve_transposed ? cfg_.m_block : cfg_.n_block);
        LOG(LOG_DEBUG_1) << "[BWSolver]  m = " << cfg_.m_block << ", n = " << cfg_.n_block
			 << (cfg_.solve_transposed ? ", transposed (AT)" : ", normal (A)");
        
        if (!spmm_) {
            // NOTE: sort_rows=true, sort_cols=true are defaults in SpMM, 
            // confirming we are using the permuted/optimized internal representation.
            spmm_ = std::make_unique<::BlockWiedemannSpMM>(A_host_, cfg_.nrows);
        }
        
        // Build SpMMAutoTuneConfig from the solver-level BWAutoTuneConfig
        SpMMAutoTuneConfig spmm_tune_cfg = build_spmm_autotune_config(autotune_cfg_);

        // This runs the benchmark and internal optimizer
        SpMMPerformanceReport report = spmm_->setup_and_benchmark(m, n, spmm_tune_cfg);
        autotune_report_ = report;   // Persist for Stage 1 operator selection

        // Temporarily load to device to get introspection string
        if (cfg_.solve_transposed) spmm_->load_AT_to_device(cfg_.n_block);
        else spmm_->load_A_to_device(cfg_.n_block);

        std::string kernels = spmm_->get_kernel_summary(cfg_.solve_transposed);

        if (cfg_.solve_transposed) spmm_->unload_AT_from_device();
        else spmm_->unload_A_from_device();

        LOG(LOG_STATS) << "[BWSolver] === SpMM Optimization Report ===";
        LOG(LOG_STATS) << "[BWSolver]  Mode:            " << (cfg_.solve_transposed ? "Transposed (AT)" : "Normal (A)");
        LOG(LOG_STATS) << "[BWSolver]  Vector Widths:   " << report.bit_width_m << " bits, " << report.bit_width_n << " bits";
        LOG(LOG_STATS) << "[BWSolver]  Throughput (A):  " << report.throughput_A_gnnz << " GNNz/s with " << n << " bits";
        LOG(LOG_STATS) << "[BWSolver]  Throughput (AT): " << report.throughput_AT_gnnz << " GNNz/s with " << m << " bits";
        LOG(LOG_STATS) << "[BWSolver]  Memory (GPU):    " << (report.peak_gpu_mem_bytes / (1024*1024)) << " MB";
        LOG(LOG_STATS) << "[BWSolver]  Active Kernels:  " << kernels;
        LOG(LOG_STATS) << "[BWSolver] ================================";

	// --- Invoking internal SpMM Verification for A and AT ---
	if (autotune_cfg_.verify_spmm) {
	    LOG(LOG_STATS) << "[BWSolver] Running Pre-Flight Checks...";
	    bool passed = spmm_->verify_A(n);
	    if(passed) {
	        passed = spmm_->verify_AT(m);
	    }
	    if(!passed) {
	        LOG(LOG_ERROR_CRITICAL) << "[BWSolver] Pre-Flight Checks FAILED. Aborting.";
		throw std::runtime_error("BWSolver pre-flight SpMM verification failed");
	    }
	}
    }
}

void BlockWiedemannSolver::RunStage1() {

    // --- BIT-EXACT CONTRACT ---
    // Object: Sequence S_k (m x n blocks over GF(2))
    // Definition: S_k = X^T * B^(k+1) * Z
    // Notes:
    //  - The exponent is (k+1), meaning S_0 = X^T * (B*Z).
    //  - This matches the Python reference implementation exactly.
    //  - Any deviation (e.g., using B^k) will break the solver.
    // --------------------------

    if(stage1_cfg_.skip) {
        LOG(LOG_INFO) << "[BWStage 1] === Krylov Sequence Generation SKIPPED ===";
	return;
    }
  
    LOG(LOG_INFO) << "[BWStage 1] === Krylov Sequence Generation ===";
    
    // Extensive Configuration Logging
    LOG(LOG_STATS) << "[BWStage 1] Configuration:";
    LOG(LOG_STATS) << "[BWStage 1]  Problem  : N=" << stage1_cfg_.nrows << " m=" << stage1_cfg_.m_block << " n=" << stage1_cfg_.n_block
                  << ", L=" << stage1_cfg_.seq_len << ", Mode=" << (stage1_cfg_.solve_transposed ? "Transposed (Left Kernel)" : "Standard (Right Kernel)");
    LOG(LOG_STATS) << "[BWStage 1]  Tuning    : GPU Batch Size " << stage1_cfg_.gpu_batch_size;
    LOG(LOG_STATS) << "[BWStage 1]  Hashing   : Hash computation " << (stage1_cfg_.compute_hashes ? "ON" : "OFF")
                  << ", Validation " << (stage1_cfg_.hash_validation ? "ON" : "OFF");
    LOG(LOG_STATS) << "[BWStage 1]  I/O Prefix: '" << stage1_cfg_.checkpoint_prefix << "'"
                  << ", Load: " << (stage1_cfg_.load_checkpoints ? "YES" : "NO")
                  << ", Save: " << (stage1_cfg_.save_checkpoints ? "YES" : "NO");

    if (!spmm_) {
        spmm_ = std::make_unique<::BlockWiedemannSpMM>(A_host_, stage1_cfg_.nrows, false, false);
        // Note: Tuning typically handles m/n selection, but we ensure setup here
        uint32_t m_op = (stage1_cfg_.solve_transposed ? stage1_cfg_.n_block : stage1_cfg_.m_block);
        uint32_t n_op = (stage1_cfg_.solve_transposed ? stage1_cfg_.m_block : stage1_cfg_.n_block);
        spmm_->setup_and_benchmark(m_op, n_op, build_spmm_autotune_config(autotune_cfg_));
    }
    
    LOG(LOG_STATS) << "[BWStage 1] Operator mode: " << (stage1_cfg_.solve_transposed ? "AT" : "A");

    // 1. Setup Starting Vectors X and Y (Z)
    // Note: We currently stick to deterministic generation via seed for X/Y. 
    // Future expansion could implement explicit loading via io_.load_vector if needed.
    if (hX_.empty()) {
        LOG(LOG_STATS) << "[BWStage 1] Initializing vectors X and Y (Seed: " << stage1_cfg_.seed << ")...";
        // X is N x m_block
        MatrixGenerator gen(stage1_cfg_.seed);
        auto vX = gen.generate_random_vector(stage1_cfg_.nrows, stage1_cfg_.m_block, true);
        
        // Y (Z) is N x n_block
        MatrixGenerator gen2((stage1_cfg_.seed << 1) + 7);
        auto vY = gen2.generate_random_vector(stage1_cfg_.nrows, stage1_cfg_.n_block, true);
    
        hX_.resize(vX.size()/8); memcpy(hX_.data(), vX.data(), vX.size());
        hY_.resize(vY.size()/8); memcpy(hY_.data(), vY.data(), vY.size());

        if(stage1_cfg_.compute_hashes) {
            uint64_t shashX = fnv1a_hash_uint64(hX_);
            LOG(LOG_STATS) << "[BWStage 1] [HASH] X: " << std::hex << shashX << std::dec;
            uint64_t shashY = fnv1a_hash_uint64(hY_);
            LOG(LOG_STATS) << "[BWStage 1] [HASH] Y: " << std::hex << shashY << std::dec;
            
            if(stage1_cfg_.hash_validation) {
                  if(shashX != stage1_cfg_.hash_X)
		      LOG(LOG_WARNING) << "[BWStage 1] [VALIDATION] X Hash MISMATCH (Expected " << std::hex << stage1_cfg_.hash_X << ")";
                  if(shashY != stage1_cfg_.hash_Y)
		      LOG(LOG_WARNING) << "[BWStage 1] [VALIDATION] Y Hash MISMATCH (Expected " << std::hex << stage1_cfg_.hash_Y << ")";
            }
        }
        
        if (stage1_cfg_.save_checkpoints) {
            io_.save_vector(stage1_cfg_.suffix_X, hX_, stage1_cfg_.nrows, stage1_cfg_.m_block);
            io_.save_vector(stage1_cfg_.suffix_Y, hY_, stage1_cfg_.nrows, stage1_cfg_.n_block);
        }
    }

    LOG_INCREMENT_STAGE(10);

    // 2. Check for Existing Sequence
    // Only attempt load if explicitly enabled in config
    if (stage1_cfg_.load_checkpoints && io_.exists(stage1_cfg_.suffix_S)) {
        int loaded_len = 0;
        // Try loading sequence
        if (io_.load_sequence(hS_, loaded_len, stage1_cfg_.m_block, stage1_cfg_.n_block)) {
             LOG(LOG_STATS) << "[BWStage 1] Loaded sequence S from checkpoint.";

             if(stage1_cfg_.compute_hashes) {
                 uint64_t shashS = fnv1a_hash_uint64(hS_);
                 LOG(LOG_STATS) << "[BWStage 1] [HASH] Sequence S: " << std::hex << shashS << std::dec;
                 if(stage1_cfg_.hash_validation) {
                     if(shashS == stage1_cfg_.hash_S)
                         LOG(LOG_STATS) << "[BWStage 1] [VALIDATION] Sequence S MATCH.";
                     else
                         LOG(LOG_WARNING) << "[BWStage 1] [VALIDATION] Sequence S MISMATCH (Expected " << std::hex << stage1_cfg_.hash_S << ")";
                 }
             }
             return; // Skip computation
        }
    }

    // 3. Compute Sequence — Adaptive Operator Selection
    LOG_INCREMENT_STAGE(10);

    bool use_alt_op = false;
    if (stage1_cfg_.prefer_faster_op &&
        autotune_report_.time_A_ms > 0 && autotune_report_.time_AT_ms > 0) {
        double current_time = cfg_.solve_transposed
            ? autotune_report_.time_AT_ms : autotune_report_.time_A_ms;
        double alt_time = cfg_.solve_transposed
            ? autotune_report_.time_A_ms : autotune_report_.time_AT_ms;

        // Require >10% improvement to justify transpose overhead
        use_alt_op = (alt_time < current_time * 0.9);

        LOG(LOG_STATS) << "[BWStage 1] Operator Selection: current="
                      << (cfg_.solve_transposed ? "AT" : "A")
                      << " (" << std::fixed << std::setprecision(2) << current_time << " ms), alternative="
                      << (cfg_.solve_transposed ? "A" : "AT")
                      << " (" << alt_time << " ms) -> "
                      << (use_alt_op ? "SWAP (using alternative)" : "KEEP (using current)");
    }

    // Create operator — local to Stage 1, does NOT modify make_operator()
    std::unique_ptr<BwOperator> op;
    if (use_alt_op) {
        if (cfg_.solve_transposed) {
            // Switch from AT to A — load A at m_block width
            spmm_->load_A_to_device(cfg_.m_block);
            op = std::make_unique<BwOperatorA>(*spmm_, cfg_.nrows);
        } else {
            // Switch from A to AT — load AT at m_block width
            spmm_->load_AT_to_device(cfg_.m_block);
            op = std::make_unique<BwOperatorAT>(*spmm_, cfg_.nrows);
        }
    } else {
        op = make_operator();
    }

    // Create generator with potentially swapped dimensions
    // transpose_output=true tells the generator to transpose S per-batch on GPU
    int gen_m = use_alt_op ? stage1_cfg_.n_block : stage1_cfg_.m_block;
    int gen_n = use_alt_op ? stage1_cfg_.m_block : stage1_cfg_.n_block;
    stage1::KrylovSequenceGenerator krylov(*op, gen_m, gen_n,
        stage1_cfg_.gpu_batch_size, /*transpose_output=*/use_alt_op,
        /*use_cuda_graph=*/stage1_cfg_.use_cuda_graph);

    LOG_INCREMENT_STAGE(10);

    // Allocate device sequence buffer if keeping S on device (Feature B)
    uint64_t* d_S_target = nullptr;
    if (stage1_cfg_.keep_S_on_device) {
        size_t s_output_mat_bytes = (size_t)stage1_cfg_.m_block
            * ((stage1_cfg_.n_block + 63) / 64) * sizeof(uint64_t);
        d_S_sequence_bytes_ = (size_t)stage1_cfg_.seq_len * s_output_mat_bytes;
        CHECK_SOLVER(cudaMalloc(&d_S_sequence_, d_S_sequence_bytes_));
        d_S_target = d_S_sequence_;

        LOG(LOG_STATS) << "[BWStage 1] S-on-device: allocated "
                      << (d_S_sequence_bytes_ / (1024.0 * 1024.0)) << " MB";
    }

    // Generate — swap X↔Z if using alternative operator
    // GPU transpose (if active) happens inside the pipeline; no post-processing needed
    // d_S_target (if non-null) receives D2D copies inside the pipeline
    if (use_alt_op) {
        krylov.generate(stage1_cfg_.seq_len, hY_, hX_, hS_, stream_, d_S_target);
    } else {
        krylov.generate(stage1_cfg_.seq_len, hX_, hY_, hS_, stream_, d_S_target);
    }
    CHECK_SOLVER(cudaStreamSynchronize(stream_));

    // Optional disk save (Feature B) — writes binary header + payload
    if (stage1_cfg_.save_S_to_disk && !stage1_cfg_.S_disk_path.empty()) {
        LOG(LOG_STATS) << "[BWStage 1] Saving S to disk: " << stage1_cfg_.S_disk_path;
        std::ofstream ofs(stage1_cfg_.S_disk_path, std::ios::binary);
        if (!ofs.is_open()) {
            LOG(LOG_ERROR_MAJOR) << "[BWStage 1] Cannot open S file for writing: " << stage1_cfg_.S_disk_path;
        } else {
            SSequenceHeader header;
            header.seq_len = static_cast<uint32_t>(stage1_cfg_.seq_len);
            header.m_block = static_cast<uint32_t>(stage1_cfg_.m_block);
            header.n_block = static_cast<uint32_t>(stage1_cfg_.n_block);
            header.data_bytes = hS_.size() * sizeof(uint64_t);

            ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
            ofs.write(reinterpret_cast<const char*>(hS_.data()), header.data_bytes);

            if (!ofs.good()) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 1] Write error saving S to disk: " << stage1_cfg_.S_disk_path;
            } else {
                LOG(LOG_STATS) << "[BWStage 1] Saved S to disk: "
                              << (sizeof(header) + header.data_bytes) << " bytes ("
                              << sizeof(header) << " header + " << header.data_bytes << " data)";
            }
        }
    }

    // 4. Post-Computation Validation and Save
    if(stage1_cfg_.compute_hashes) {
        uint64_t shashS = fnv1a_hash_uint64(hS_);
        LOG(LOG_STATS) << "[BWStage 1] [HASH] Sequence S: " << std::hex << shashS << std::dec;
        if(stage1_cfg_.hash_validation) {
            if(shashS == stage1_cfg_.hash_S)
                LOG(LOG_STATS) << "[BWStage 1] [VALIDATION] Sequence S MATCH.";
            else
                LOG(LOG_WARNING) << "[BWStage 1] [VALIDATION] Sequence S MISMATCH (Expected " << std::hex << stage1_cfg_.hash_S << ")";
        }
    }
    
    if (stage1_cfg_.save_checkpoints) {
        io_.save_sequence(hS_, stage1_cfg_.seq_len, stage1_cfg_.m_block, stage1_cfg_.n_block);
    }   
}

void BlockWiedemannSolver::RunStage2() {
    LOG(LOG_INFO) << "[BWStage 2] === Linear Generator Computation ===";

    // --- BIT-EXACT CONTRACT ---
    // Object: Generator Matrix Pi(x) (M x M matrix polynomial, M ~ 2n)
    // Algorithm: Coppersmith's Block Berlekamp-Massey (iterative).
    // Invariants:
    //  1. Initialization: t0 is the first index where rank(S_0...S_{t0-1}) == m.
    //  2. Pivot Selection: First row in a stable-sorted list (by gamma, then index) that has a 1 in current col.
    //  3. Shift Strategy: Exactly m rows must be shifted (x -> x*z) at every step.
    //  4. Trimming: Trailing zero matrices are removed.
    // --------------------------

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    uint64_t* dS = nullptr;
    bool owns_dS = false;   // Track whether we need to free dS

    // Source 1: Load from disk (with header validation and graceful fallback)
    if (stage2_cfg_.load_S_from_disk && !stage2_cfg_.S_disk_path.empty()) {
        bool disk_loaded = false;
        do {  // Single-iteration do-while for break-on-error pattern
            std::ifstream ifs(stage2_cfg_.S_disk_path, std::ios::binary);
            if (!ifs.is_open()) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] Cannot open S file: " << stage2_cfg_.S_disk_path;
                break;
            }

            SSequenceHeader header;
            if (!ifs.read(reinterpret_cast<char*>(&header), sizeof(header))) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] Failed to read S file header: " << stage2_cfg_.S_disk_path;
                break;
            }

            if (header.magic != SSequenceHeader::MAGIC) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] Invalid S file magic: expected 0x"
                               << std::hex << SSequenceHeader::MAGIC << ", got 0x" << header.magic << std::dec;
                break;
            }

            if (header.version != SSequenceHeader::CURRENT_VERSION) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] Unsupported S file version: " << header.version
                               << " (expected " << SSequenceHeader::CURRENT_VERSION << ")";
                break;
            }

            if (header.m_block != static_cast<uint32_t>(cfg_.m_block) ||
                header.n_block != static_cast<uint32_t>(cfg_.n_block)) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] S file dimension mismatch: file has "
                               << header.m_block << "x" << header.n_block
                               << ", solver expects " << cfg_.m_block << "x" << cfg_.n_block;
                break;
            }

            // Validate data_bytes consistency
            size_t expected_words = static_cast<size_t>(header.seq_len)
                * header.m_block * ((header.n_block + 63) / 64);
            size_t expected_bytes = expected_words * sizeof(uint64_t);
            if (header.data_bytes != expected_bytes) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] S file data size mismatch: header says "
                               << header.data_bytes << " bytes, expected " << expected_bytes;
                break;
            }

            // Validate actual file size
            auto pos = ifs.tellg();
            ifs.seekg(0, std::ios::end);
            auto file_end = ifs.tellg();
            ifs.seekg(pos);
            size_t remaining = static_cast<size_t>(file_end - pos);
            if (remaining < header.data_bytes) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] S file truncated: " << remaining
                               << " bytes remaining, expected " << header.data_bytes;
                break;
            }

            // Read payload
            hS_.resize(header.data_bytes / sizeof(uint64_t));
            if (!ifs.read(reinterpret_cast<char*>(hS_.data()), header.data_bytes)) {
                LOG(LOG_ERROR_MAJOR) << "[BWStage 2] Failed to read S data (" << header.data_bytes << " bytes)";
                break;
            }

            // Success — upload to device
            CHECK_SOLVER(cudaMalloc(&dS, header.data_bytes));
            CHECK_SOLVER(cudaMemcpyAsync(dS, hS_.data(), header.data_bytes,
                                          cudaMemcpyHostToDevice, stream_));
            owns_dS = true;
            disk_loaded = true;

            LOG(LOG_STATS) << "[BWStage 2] Loaded S from disk: " << stage2_cfg_.S_disk_path
                          << " (seq_len=" << header.seq_len << ", " << header.data_bytes << " bytes)";
        } while (false);

        if (!disk_loaded) {
            LOG(LOG_WARNING) << "[BWStage 2] Disk load failed, falling back to next source";
        }
    }

    // Source 2: Direct device pointer from Stage 1 (zero-copy!)
    if (!dS && d_S_sequence_) {
        LOG(LOG_STATS) << "[BWStage 2] Using S from device memory (zero-copy from Stage 1)";
        dS = d_S_sequence_;
        owns_dS = false;
    }

    // Source 3: Fallback — upload from host (current behavior)
    if (!dS) {
        LOG(LOG_STATS) << "[BWStage 2] Uploading S from host memory";
        size_t s_bytes = hS_.size() * sizeof(uint64_t);
        CHECK_SOLVER(cudaMalloc(&dS, s_bytes));
        CHECK_SOLVER(cudaMemcpyAsync(dS, hS_.data(), s_bytes,
                                      cudaMemcpyHostToDevice, stream_));
        owns_dS = true;
    }

    // 1. Run Solver
    LOG_INCREMENT_STAGE(10);
    
    // Instantiate with populated config
    stage2::BasecaseSolver solver(stage2_cfg_);
    
    LOG_INCREMENT_STAGE(10);
    // Call solve with just stream (and dS), flags are in config
    solver.solve(dS, stream_);    
    
    LOG_INCREMENT_STAGE(10);
    CHECK_SOLVER(cudaStreamSynchronize(stream_));

    LOG_INCREMENT_STAGE(10);
    hPi_ = solver.get_pi_host();
    hGamma_ = solver.get_gamma();

    if(stage2_cfg_.compute_hashes) {
        // Compute reference hash
        uint64_t shashPi = fnv1a_hash_uint64(hPi_);
        LOG(LOG_STATS) << "[BWSolver] [REF_HASH] Stage 2 Generator Pi: " << std::hex << shashPi << std::dec;
        if(stage2_cfg_.hash_validation) {
            if(shashPi == stage2_cfg_.expected_hash_Pi)
                LOG(LOG_STATS) << "[BWSolver] [REF_HASH] Stage 2 Generator Pi MATCH.";
            else
                LOG(LOG_WARNING) << "[BWSolver] [REF_HASH] Stage 2 Generator Pi MISMATCH.";
        }
    }

    // 2. Post-Run Oracle Check (Legacy)
    if (stage2_cfg_.post_run_legacy_check) {
        LOG_INCREMENT_STAGE(100);
        LOG(LOG_STATS) << "[BWSolver] Running Legacy Oracle Check...";

        uint64_t* dS_oracle;
        size_t s_bytes_oracle = hS_.size() * sizeof(uint64_t);
        CHECK_SOLVER(cudaMalloc(&dS_oracle, s_bytes_oracle));
        CHECK_SOLVER(cudaMemcpyAsync(dS_oracle, hS_.data(), s_bytes_oracle, cudaMemcpyHostToDevice, stream_));

        // Legacy solver still uses raw params for now (reference implementation)
        int delta_oracle = stage2_cfg_.seq_len / 2;
        stage2::BasecaseSolverLegacy oracle(stage2_cfg_.m_block, stage2_cfg_.n_block, delta_oracle);
        
        oracle.solve(dS_oracle, stage2_cfg_.seq_len, stream_, 
                     stage2_cfg_.check_annihilation_gpu, stage2_cfg_.check_annihilation_legacy);
        
        CHECK_SOLVER(cudaStreamSynchronize(stream_));    
       
        const auto& ref_pi = oracle.get_pi_host();
        const auto& ref_gamma = oracle.get_gamma();
        
        bool ok = true;
        if (ref_pi.size() != hPi_.size()) ok = false;
        else if (memcmp(ref_pi.data(), hPi_.data(), hPi_.size()*8) != 0) ok = false;
        
        if (ref_gamma != hGamma_) ok = false;

        if (ok) LOG(LOG_INFO) << "[BWSolver] Oracle Check: PASS (Bit-Exact)";
        else {
            LOG(LOG_ERROR_CRITICAL) << "[BWSolver] Oracle Check: FAIL! Generic solver deviates from Legacy.";
            throw std::runtime_error("BWSolver oracle check failed");
        }
        CHECK_SOLVER(cudaFree(dS_oracle));
    }

    // Free device S resources
    if (owns_dS) {
        CHECK_SOLVER(cudaFree(dS));
    }
    // Release device sequence buffer — no longer needed after Stage 2
    if (d_S_sequence_) {
        CHECK_SOLVER(cudaFree(d_S_sequence_));
        d_S_sequence_ = nullptr;
        d_S_sequence_bytes_ = 0;
    }

    if (stage2_cfg_.save_checkpoints) {
        int dim = stage2_cfg_.m_block + stage2_cfg_.n_block;
        size_t mat_size_words = (size_t)dim * ((dim+63)/64);
        int pi_len = hPi_.size() / mat_size_words;
        io_.save_polynomial(hPi_, pi_len, dim, dim);      
    }
}

void BlockWiedemannSolver::RunStage3() {
    LOG(LOG_INFO) << "[BWStage 3] === Solution Reconstruction ===";

    // --- BIT-EXACT CONTRACT ---
    // Object: Kernel vectors w such that B*w = 0.
    // Reconstruction: w = sum_{k=0}^{deg} B^(deg-k) * Z * u_k
    //  - Note the reversed exponent on B relative to u_k.
    //  - Followed by "valuation stripping": w <- B*w until B*w = 0.
    // --------------------------

    /*
     * Mathematical Note:
     * The reconstruction formula for a vector w in the kernel is:
     * 
     *    w = sum_{k=0}^{deg} B^(deg - k) * Z * u_k
     * 
     * where:
     *  - B is the linear operator (A or A^T).
     *  - Z is the random block from Stage 1.
     *  - u(x) = sum u_k x^k is a column of the generator matrix Pi(x).
     * 
     * This evaluation is followed by "Valuation Stripping":
     *    while (B*w != 0) w = B*w;
     * 
     * We execute this either via a reference serial path (Legacy) or an optimized 
     * parallel path (Batch) that processes multiple u(x) candidates simultaneously.
     */

    if (hPi_.empty()) return;

    if (!spmm_) {
        spmm_ = std::make_unique<::BlockWiedemannSpMM>(A_host_, stage3_cfg_.nrows, false, false);
        uint32_t m = (stage3_cfg_.solve_transposed ? stage3_cfg_.n_block : stage3_cfg_.m_block);
        uint32_t n = (stage3_cfg_.solve_transposed ? stage3_cfg_.m_block : stage3_cfg_.n_block);
        spmm_->setup_and_benchmark(m, n, build_spmm_autotune_config(autotune_cfg_));
    }

    int target = (stage3_cfg_.max_solutions > 0) ? stage3_cfg_.max_solutions : 999999;
    solutions_.clear();

    // -------------------------------------------------------------------------
    // Execution Orchestration
    // -------------------------------------------------------------------------
    
    if (stage3_cfg_.oracle_verification) {
        LOG(LOG_STATS) << "[BWSolver] Oracle Verification Enabled.";
        std::map<int, stage3::OracleEntry> oracle_data;

        // --- PHASE 1: Legacy (Record Mode) ---
        {
            LOG(LOG_STATS) << "[BWSolver] >>> PHASE 1: Recording Oracle (Legacy Path) <<<";
            
            // Create a specific config for the Recording phase
            BWStage3Config oracle_cfg = stage3_cfg_;
            oracle_cfg.batch_mode = false; // Force Legacy
            // Note: In the updated SolutionReconstructor, 'oracle_verification=true' 
            // plus empty oracle_data triggers recording.
            
	    auto op = make_operator();
            stage3::SolutionReconstructor recon_oracle(*op, oracle_cfg);
            recon_oracle.Init(hY_, hPi_, stream_);
            
            int oracle_target = stage3_cfg_.n_block;
            std::vector<uint64_t> sol;
            int count = 0;
            
            while (count < oracle_target) {
                if (recon_oracle.FindNext(sol, stream_)) {
                    count++;
                } else {
                    break;
                }
            }
            oracle_data = recon_oracle.GetOracleData();
            LOG(LOG_STATS) << "[BWSolver] Recorded " << oracle_data.size() << " verification entries from Oracle.";
        } // recon_oracle destroyed, memory freed

        // --- PHASE 2: Optimized (Verify Mode) ---
        {
            LOG(LOG_STATS) << "[BWSolver] >>> PHASE 2: Verifying Optimized Path <<<";

            // Config for verification phase (Batch mode)
            BWStage3Config verify_cfg = stage3_cfg_;
            verify_cfg.batch_mode = true;

	    auto op = make_operator();
            stage3::SolutionReconstructor recon_opt(*op, verify_cfg);
            recon_opt.SetOracleData(oracle_data); // Inject recorded data
            recon_opt.Init(hY_, hPi_, stream_);
	    
            std::vector<uint64_t> sol;
            while (solutions_.size() < (uint64_t)target) {
                // FindNext handles batching and verification internally
                bool found = recon_opt.FindNext(sol, stream_);
                if (found) {
                    if (stage3_cfg_.compute_hashes) {
                        uint64_t shashsol = fnv1a_hash_uint64(sol);
                        LOG(LOG_STATS) << "[BWSolver] [REF_HASH] Stage 3 solution " << solutions_.size() << ": " << std::hex << shashsol << std::dec;
                    }
                    solutions_.push_back(sol);
                } else {
                    break;
                }
            }
            LOG(LOG_STATS) << "[BWSolver] Oracle Verification Phase Complete. Optimized Path Matches Legacy.";
        }
    } 
    else {
        // --- STANDARD EXECUTION (Default: Batch Mode) ---
        auto op = make_operator();
        stage3::SolutionReconstructor recon(*op, stage3_cfg_);
        // formerly stage3::SolutionReconstructor recon(*spmm_, stage3_cfg_);
        recon.Init(hY_, hPi_, stream_);

        LOG(LOG_STATS) << "[BWSolver] Target Solutions: " << (stage3_cfg_.max_solutions > 0 ? std::to_string(target) : "ALL");

        std::vector<uint64_t> sol;
        while (solutions_.size() < (uint64_t)target) {
            bool found = recon.FindNext(sol, stream_);
            if (found) {
                if(stage3_cfg_.compute_hashes) {
                    uint64_t shashsol = fnv1a_hash_uint64(sol);
                    LOG(LOG_STATS) << "[BWSolver] [REF_HASH] Stage 3 solution " << solutions_.size() << ": " << std::hex << shashsol << std::dec;

                    // Golden Check for first solution
                    if(stage3_cfg_.expected_hash_first_solution && solutions_.size() == 0) {
                        if(shashsol == stage3_cfg_.expected_hash_first_solution)
                            LOG(LOG_STATS) << "[BWSolver] [REF_HASH] Stage 3 first solution MATCH.";
                        else
                            LOG(LOG_WARNING) << "[BWSolver] [REF_HASH] Stage 3 solution MISMATCH.";
                    }
                }
                solutions_.push_back(sol);
            } else {
                break;
            }
        }
    }

    LOG_INCREMENT_STAGE(10);
    // Un-permute solutions if SpMM optimization was enabled
    if (stage3_cfg_.perform_unpermutation) {
        LOG(LOG_STATS) << "[BWSolver] Mapping solutions back to original domain...";
        for(size_t i=0; i<solutions_.size(); ++i) {
            std::vector<uint8_t> unpacked = unpack_bits(solutions_[i], stage3_cfg_.nrows);
            std::vector<uint8_t> unpermuted(stage3_cfg_.nrows);
            
            if (stage3_cfg_.solve_transposed) {
                spmm_->postprocess_domain_A(unpacked.data(), unpermuted.data(), stage3_cfg_.nrows, 8);
            } else {
                spmm_->postprocess_domain_AT(unpacked.data(), unpermuted.data(), stage3_cfg_.nrows, 8);
            }
            
            solutions_[i] = pack_bits(unpermuted);
        }
    }

    // --- Upload solutions to device for downstream GPU consumption ---
    if (!solutions_.empty()) {
        size_t words_per_solution = (stage3_cfg_.nrows + 63) / 64;
        size_t total_words = solutions_.size() * words_per_solution;
        CHECK_SOLVER(cudaMalloc(&d_solutions_, total_words * sizeof(uint64_t)));
        for (size_t i = 0; i < solutions_.size(); ++i) {
            CHECK_SOLVER(cudaMemcpy(
                d_solutions_ + i * words_per_solution,
                solutions_[i].data(),
                words_per_solution * sizeof(uint64_t),
                cudaMemcpyHostToDevice));
        }
        num_device_solutions_ = solutions_.size();
        LOG(LOG_DEBUG_1) << "[BWSolver] Uploaded " << num_device_solutions_
                         << " solutions to device (" << total_words * 8 << " bytes)";
    }

    LOG(LOG_INFO) << "[BWSolver] Reconstruction Complete.";
    LOG(LOG_INFO) << "[BWSolver] Total Solutions Found: " << solutions_.size();

    if (stage3_cfg_.save_solutions && !stage3_cfg_.checkpoint_prefix.empty()) {
        io_.save_solutions(solutions_, stage3_cfg_.nrows);
    }    
}
 
void BlockWiedemannSolver::Solve() {
    using clock = std::chrono::high_resolution_clock;
    auto t_start_total = clock::now();

    LOG(LOG_INFO) << "=== " << lingen::version_banner() << " ===";

    LOG_SET_STAGE(LOG_STAGE_BW_AUTOTUNE, "LinAlg");
    auto t0 = clock::now();
    AutoTune();
    double ms_tune = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    LOG(LOG_STATS) << "[BWSolver] AutoTune finished in " << FormatDuration(ms_tune);

    LOG_SET_STAGE(LOG_STAGE_BW_STAGE1, "LinAlg");
    t0 = clock::now();
    RunStage1();
    double ms_stage1 = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    LOG(LOG_STATS) << "[BWSolver] Stage 1 finished in " << FormatDuration(ms_stage1);

    LOG_SET_STAGE(LOG_STAGE_BW_STAGE2, "LinAlg");
    t0 = clock::now();
    if (!cfg_.stage2_skip) RunStage2();
    else LOG(LOG_INFO) << "[BWStage 2] === Linear Generator Computation SKIPPED ===";
    double ms_stage2 = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    LOG(LOG_STATS) << "[BWSolver] Stage 2 finished in " << FormatDuration(ms_stage2);

    LOG_SET_STAGE(LOG_STAGE_BW_STAGE3, "LinAlg");
    t0 = clock::now();
    if (!cfg_.stage3_skip) RunStage3();
    else LOG(LOG_INFO) << "[BWStage 3] === Solution Reconstruction SKIPPED ===";
    double ms_stage3 = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    LOG(LOG_STATS) << "[BWSolver] Stage 3 finished in " << FormatDuration(ms_stage3);

    LOG_SET_STAGE(LOG_STAGE_BW_POSTPROCESSING, "LinAlg");

    double ms_total = std::chrono::duration<double, std::milli>(clock::now() - t_start_total).count();
    double ms_123 = ms_stage1 + ms_stage2 + ms_stage3;

    LOG(LOG_STATS) << "[BWSolver] === Block Wiedemann Timing Summary ===";
    LOG(LOG_STATS) << "[BWSolver]  AutoTune    : " << FormatDuration(ms_tune);
    LOG(LOG_STATS) << "[BWSolver]  Stage 1     : " << FormatDuration(ms_stage1);
    LOG(LOG_STATS) << "[BWSolver]  Stage 2     : " << FormatDuration(ms_stage2);
    LOG(LOG_STATS) << "[BWSolver]  Stage 3     : " << FormatDuration(ms_stage3);
    LOG(LOG_STATS) << "[BWSolver]  ------------------------------";
    LOG(LOG_STATS) << "[BWSolver]  Total (1-3) : " << FormatDuration(ms_123);
    LOG(LOG_STATS) << "[BWSolver]  Total (All) : " << FormatDuration(ms_total);
    LOG(LOG_STATS) << "[BWSolver] ======================================";
} 

} // namespace lingen
