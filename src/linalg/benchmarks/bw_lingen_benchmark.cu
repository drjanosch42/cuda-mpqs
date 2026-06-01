// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/**
 * @file bw_lingen_benchmark.cu
 * @brief Benchmark and Test Driver for the Block Wiedemann Solver.
 * 
 * This application generates a random sparse matrix (mimicking the characteristics
 * of GNFS relation matrices) and executes the full Block Wiedemann pipeline
 * (Stages 1, 2, and 3) to find kernel vectors.
 * 
 * It serves as both a performance benchmark and a verification tool, exposing
 * the full granularity of the solver's configuration via command-line arguments.
 */

#include "bw_solver.h"
#include "bw_version.h"
#include "hpc_logger.h"
#include "generator.h" // Linked from cuda_spmm library
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>

using namespace lingen;

#define CHECK_SOLVER_BENCH(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[Bech] CUDA Error: " << cudaGetErrorString(err) \
                                << " (" << __FILE__ << ":" << __LINE__ << ")"; \
        exit(1); \
    } \
} while(0)


// -----------------------------------------------------------------------------
// Generator Configuration
// -----------------------------------------------------------------------------

struct GeneratorConfig {
    int rows = 10100;    ///< Number of relations (rows)
    int cols = 10000;    ///< Factor base size (cols)
    double alpha = 25.0; ///< Density parameter (Avg NNZ per row approx alpha)
    uint64_t seed = 42;  ///< Generator Seed (Separate from Solver Seed)
    bool verbose = false;
};

// -----------------------------------------------------------------------------
// Argument Parsing Helper
// -----------------------------------------------------------------------------

struct ParsedArgs {
    GeneratorConfig gen_cfg;
    BWSolverConfig solver_cfg;
    
    // Tracking for conflict detection
    std::map<std::string, bool> set_flags;
    
    void mark(const std::string& name) {
        set_flags[name] = true;
    }
    
    bool is_set(const std::string& name) const {
        return set_flags.find(name) != set_flags.end();
    }
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\n--- Generator Parameters ---\n"
              << "  --rows <N>       Number of rows (relations) [Default: 10100]\n"
              << "  --cols <N>       Number of cols (factor base) [Default: 10000]\n"
              << "  --alpha <F>      Density parameter alpha [Default: 25.0]\n"
              << "  --gen_seed <N>   Seed for matrix generation [Default: 43]\n"
              << "\n--- Global Solver Settings ---\n"
              << "  --seed <N>       Solver random seed [Default: 12345]\n"
              << "  --device <ID>    GPU Device ID [Default: 0]\n"
              << "  --m_block <N>    Block size m (left projection / S rows) [Default: 64]\n"
              << "  --n_block <N>    Block size n (starting block / S cols) [Default: 64]\n"
              << "  --out <PATH>     Checkpoint directory/prefix [Default: ""]\n"
              << "  --solutions <N>  Target number of solutions [Default: 5]\n"
              << "  --sequence_length <N>  Global Sequence Length override (L).\n"
              << "\n--- Global Switches ---\n"
              << "  --enable_hashing    Enable result hashing in all stages.\n"
              << "  --enable_validation Validate results against expected hashes in all stages.\n"
              << "  --enable_oracle     Enable strict bit-exact Oracle verification (Implies hashing/gpu).\n"
              << "\n--- Auto-Tune (autotune) ---\n"
              << "  --autotune_skip_spmm   Disable SpMM kernel tuning.\n"
              << "  --autotune_force_check Force Pre-Flight SpMM correctness check.\n"
              << "\n--- Stage 1: Krylov Generation (s1) ---\n"
              << "  --s1_skip              Skip Stage 1 (Load from checkpoint).\n"
              << "  --s1_sequence_length <N>  Set Stage 1 sequence length.\n"
              << "  --s1_gpu_batch_size <N>   Pipeline batch size [Default: 64].\n"
              << "  --s1_enable_hashing    Enable hashing for Stage 1.\n"
              << "  --s1_no_save           Disable saving checkpoints (X, Y, S).\n"
              << "\n--- Stage 2: Linear Generator (s2) ---\n"
              << "  --s2_skip              Skip Stage 2 (Load from checkpoint).\n"
              << "  --s2_sequence_length <N>  Input sequence length to consume.\n"
              << "  --s2_delta <N>         Explicit degree bound delta.\n"
              << "  --s2_cpu_mode          Force CPU-only Basecase (Disable GPU default).\n"
              << "  --s2_verify_gpu        Enable optimized GPU annihilation check.\n"
              << "  --s2_verify_legacy     Enable slow CPU legacy annihilation check.\n"
              << "  --s2_post_run_legacy   Run full Legacy solver after main run for comparison.\n"
              << "  --s2_enable_hashing    Enable hashing for Stage 2.\n"
              << "\n--- Stage 3: Reconstruction (s3) ---\n"
              << "  --s3_skip              Skip Stage 3.\n"
              << "  --s3_serial_mode       Force Legacy Serial Reconstructor (Disable Batch).\n"
              << "  --s3_history_depth <N> Backtracking history depth [Default: 64].\n"
              << "  --s3_check_interval <N> Zero-column check interval [Default: 16].\n"
              << "  --s3_stripping_limit <N> Valuation stripping step limit (0=auto).\n"
              << "  --s3_enable_hashing    Enable hashing for Stage 3.\n"
              << "  --s3_no_save           Disable saving solutions.\n"
              << "\n--- Misc ---\n"
              << "  --verbose        Enable verbose logging.\n"
              << "  --help           Show this help message.\n";
}

ParsedArgs parse_args(int argc, char** argv, LogConfig *log_cfg) {
    ParsedArgs args;
    
    // Set Solver Defaults (different from Struct defaults if needed)
    args.solver_cfg.stage2_gpu_mode = true; // Default to GPU mode for High Performance
    args.solver_cfg.solve_transposed = true; // Default for GNFS (Left Kernel)

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        
        // --- Generator & Standard ---
        if (arg == "--rows" && i+1 < argc) args.gen_cfg.rows = std::atoi(argv[++i]);
        else if (arg == "--cols" && i+1 < argc) args.gen_cfg.cols = std::atoi(argv[++i]);
        else if (arg == "--alpha" && i+1 < argc) args.gen_cfg.alpha = std::atof(argv[++i]);
        else if (arg == "--gen_seed" && i+1 < argc) args.gen_cfg.seed = std::atoll(argv[++i]);
        
        // --- Global Solver ---
        else if (arg == "--seed" && i+1 < argc) args.solver_cfg.seed = std::atoi(argv[++i]);
        else if (arg == "--device" && i+1 < argc) args.solver_cfg.device_id = std::atoi(argv[++i]);
        else if (arg == "--m_block" && i+1 < argc) args.solver_cfg.m_block = std::atoi(argv[++i]);
        else if (arg == "--n_block" && i+1 < argc) args.solver_cfg.n_block = std::atoi(argv[++i]);
        else if (arg == "--out" && i+1 < argc) args.solver_cfg.checkpoint_prefix = argv[++i];
        else if (arg == "--solutions" && i+1 < argc) {
            args.solver_cfg.stage3_max_solutions = std::atoi(argv[++i]);
        }
        else if (arg == "--sequence_length" && i+1 < argc) {
            args.solver_cfg.stage1_seq_len = std::atoi(argv[++i]);
            args.solver_cfg.stage2_seq_len = args.solver_cfg.stage1_seq_len;
            args.mark("global_seq_len");
        }

        // --- Global Switches ---
        else if (arg == "--enable_hashing") {
            args.solver_cfg.enable_all_hashing = true;
            args.mark("global_enable_hashing");
        }
        else if (arg == "--enable_validation") {
            args.solver_cfg.enable_all_validation = true;
            args.mark("global_enable_validation");
        }
        else if (arg == "--enable_oracle" || arg == "--oracle") { // Support legacy alias
            args.solver_cfg.enable_all_oracle_verification = true;
            args.mark("global_enable_oracle");
        }

        // --- AutoTune ---
        else if (arg == "--autotune_skip_spmm") args.solver_cfg.autotune_tune_spmm = false;
        else if (arg == "--autotune_force_check") args.solver_cfg.autotune_verify_spmm = true;

        // --- Stage 1 ---
        else if (arg == "--s1_skip") args.solver_cfg.stage1_skip = true;
        else if (arg == "--s1_sequence_length" && i+1 < argc) {
            args.solver_cfg.stage1_seq_len = std::atoi(argv[++i]);
            args.mark("s1_seq_len");
        }
        else if (arg == "--s1_gpu_batch_size" && i+1 < argc) args.solver_cfg.stage1_gpu_batch_size = std::atoi(argv[++i]);
        else if (arg == "--s1_enable_hashing") args.solver_cfg.stage1_compute_hashes = true;
        else if (arg == "--s1_no_save") args.solver_cfg.stage1_save_checkpoints = false;

        // --- Stage 2 ---
        else if (arg == "--s2_skip") args.solver_cfg.stage2_skip = true;
        else if (arg == "--s2_sequence_length" && i+1 < argc) args.solver_cfg.stage2_seq_len = std::atoi(argv[++i]);
        else if (arg == "--s2_delta" && i+1 < argc) args.solver_cfg.stage2_delta = std::atoi(argv[++i]);
        else if (arg == "--s2_cpu_mode") args.solver_cfg.stage2_gpu_mode = false; // Disable default GPU
        else if (arg == "--s2_verify_gpu") args.solver_cfg.stage2_check_annihilation_gpu = true;
        else if (arg == "--s2_verify_legacy") args.solver_cfg.stage2_check_annihilation_legacy = true;
        else if (arg == "--s2_post_run_legacy") args.solver_cfg.stage2_post_run_legacy_check = true;
        else if (arg == "--s2_enable_hashing") args.solver_cfg.stage2_compute_hashes = true;

        // --- Stage 3 ---
        else if (arg == "--s3_skip") args.solver_cfg.stage3_skip = true;
        else if (arg == "--s3_serial_mode") args.solver_cfg.stage3_batch_mode = false;
        else if (arg == "--s3_history_depth" && i+1 < argc) args.solver_cfg.stage3_history_depth = std::atoi(argv[++i]);
        else if (arg == "--s3_check_interval" && i+1 < argc) args.solver_cfg.stage3_check_interval = std::atoi(argv[++i]);
        else if (arg == "--s3_stripping_limit" && i+1 < argc) args.solver_cfg.stage3_stripping_limit = std::atoi(argv[++i]);
        else if (arg == "--s3_enable_hashing") args.solver_cfg.stage3_compute_hashes = true;
        else if (arg == "--s3_no_save") args.solver_cfg.stage3_save_solutions = false;

        // --- CUDA Graphs ---
        else if (arg == "--graph_enable") args.solver_cfg.graph_enable = true;

        // --- Misc ---
        else if (arg == "--verbose") {
            args.gen_cfg.verbose = true;
            log_cfg->min_severity_cout = LOG_DEBUG_2;
        }
        else if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Logic Validation & Trigger-Down
    // -------------------------------------------------------------------------

    // 1. Conflict Check: Global Seq Len vs Local Seq Len
    if (args.is_set("global_seq_len") && args.is_set("s1_seq_len")) {
        std::cerr << "Error: Contradictory arguments. --sequence_length overrides --s1_sequence_length.\n"
                  << "Please specify only one.\n";
        exit(1);
    }

    // 2. Oracle Trigger Down
    // Oracle implies strict checks, usually requires hashing for comparison and GPU mode for S2 speed vs CPU oracle
    if (args.solver_cfg.enable_all_oracle_verification) {
        LOG(LOG_INFO) << "[ArgParse] Oracle Verification enabled -> Enforcing Hashing and GPU Mode.";
        args.solver_cfg.enable_all_hashing = true; 
        args.solver_cfg.stage2_gpu_mode = true; // Required to verify GPU path against Oracle
        
        // Oracle implies verification of results
        args.solver_cfg.stage2_internal_oracle_verification = true;
        args.solver_cfg.stage3_oracle_verification = true;
    }

    // 3. Global Hashing Trigger
    if (args.solver_cfg.enable_all_hashing) {
        args.solver_cfg.stage1_compute_hashes = true;
        args.solver_cfg.stage2_compute_hashes = true;
        args.solver_cfg.stage3_compute_hashes = true;
    }

    // 4. Global Validation Trigger
    if (args.solver_cfg.enable_all_validation) {
        args.solver_cfg.stage1_hash_validation = true;
        args.solver_cfg.stage2_hash_validation = true;
        args.solver_cfg.stage3_hash_validation = true;
        // Note: Validation requires hashes, which is handled in UpdateConfig logic, 
        // but safe to enforce here too.
        args.solver_cfg.enable_all_hashing = true; 
    }

    return args;
}

// -----------------------------------------------------------------------------
// Matrix Helpers
// -----------------------------------------------------------------------------

void log_matrix_stats(const HostMatrix& mat, const std::string& label) {
    size_t total_nnz = 0;
    size_t min_row = std::numeric_limits<size_t>::max();
    size_t max_row = 0;
    
    for(const auto& r : mat.rows) {
        size_t s = r.size();
        total_nnz += s;
        if(s < min_row) min_row = s;
        if(s > max_row) max_row = s;
    }
    if (mat.n_rows == 0) min_row = 0;

    double avg_weight = (mat.n_rows > 0) ? (double)total_nnz / mat.n_rows : 0.0;
    double density = (mat.n_rows > 0 && mat.n_cols > 0) 
                     ? (double)total_nnz / ((double)mat.n_rows * mat.n_cols) 
                     : 0.0;
    double mem_mb = (double)(total_nnz * 4 + mat.n_rows * 4) / (1024.0 * 1024.0);

    LOG(LOG_INFO) << "[Bench] --- Matrix Statistics: " << label << " ---";
    LOG(LOG_INFO) << "[Bench]  Dimensions:    " << mat.n_rows << " x " << mat.n_cols;
    LOG(LOG_INFO) << "[Bench]  Total NNZ:     " << total_nnz;
    LOG(LOG_INFO) << "[Bench]  Density:       " << std::scientific << std::setprecision(2) << density 
                  << "[Bench] (" << std::fixed << std::setprecision(4) << (density * 100.0) << "%)" << std::defaultfloat;
    LOG(LOG_INFO) << "[Bench]  Row Weights:   Avg=" << std::fixed << std::setprecision(1) << avg_weight 
                  << ", Min=" << min_row << ", Max=" << max_row;
    LOG(LOG_INFO) << "[Bench]  Est. Memory:   " << std::fixed << std::setprecision(2) << mem_mb << " MB";
    LOG(LOG_INFO) << "[Bench] ----------------------------------------------------";
}

// Verifies A^T * x = 0 over GF(2)
bool verify_kernel_AT(const HostMatrix& A_rect, const std::vector<uint64_t>& x_packed) {
    int D = std::max((int)A_rect.n_rows, (int)A_rect.n_cols);
    std::vector<uint8_t> x_vec(D, 0);
    for(int i=0; i < D; ++i) {
        int word_idx = i / 64;
        int bit_idx  = i % 64;
        if (word_idx < (int)x_packed.size()) {
            if ((x_packed[word_idx] >> bit_idx) & 1ULL) x_vec[i] = 1;
        }
    }

    std::vector<uint8_t> y(A_rect.n_cols, 0);
    for(uint32_t r = 0; r < A_rect.n_rows; ++r) {
        if (r >= D || x_vec[r] == 0) continue;
        for(auto c : A_rect.rows[r]) {
            if (c < (idx_t)y.size()) {
                y[c] ^= 1;
            }
        }
    }
    
    int weight = 0;
    for(uint8_t val : y) weight += val;
    
    if (weight == 0) return true;
    else {
        LOG(LOG_ERROR_MAJOR) << "[Bench] Verification FAIL: A^T * x has hamming weight " << weight;
        return false;
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    // 0. Configure Logging
    LogConfig log_cfg;
    log_cfg.enable_cout = true;
    log_cfg.min_severity_cout = LOG_INFO;

    // 1. Parse Arguments
    ParsedArgs args = parse_args(argc, argv, &log_cfg);
    const GeneratorConfig& gen_cfg = args.gen_cfg;
    // Note: args.solver_cfg is populated. We set nrows after matrix gen.

    // 2. Initialize Logging
    HPCLogger::Get().Init(log_cfg);
    LOG_SET_STAGE(LOG_STAGE_BW_INITIALIZATION, "LinAlg");

    LOG(LOG_INFO) << "=== Block Wiedemann Benchmark " << lingen::version_string() << " ===";
    LOG(LOG_INFO) << "  Generator: " << gen_cfg.rows << "x" << gen_cfg.cols 
                  << ", Alpha=" << gen_cfg.alpha << ", Seed=" << gen_cfg.seed;
    LOG(LOG_INFO) << "  Solver Device: " << args.solver_cfg.device_id 
                  << ", m=" << args.solver_cfg.m_block << ", n=" << args.solver_cfg.n_block;
    
    CHECK_SOLVER_BENCH(cudaSetDevice(args.solver_cfg.device_id));
    cudaDeviceProp prop;
    CHECK_SOLVER_BENCH(cudaGetDeviceProperties(&prop, args.solver_cfg.device_id));
    LOG(LOG_INFO) << "--- Device Hardware Specs ---";
    LOG(LOG_INFO) << "  Device Name: " << prop.name;
    LOG(LOG_INFO) << "  Compute Capability: " << prop.major << "." << prop.minor;
    LOG(LOG_INFO) << "  SMs: " << prop.multiProcessorCount << ", Global Mem: " << prop.totalGlobalMem / (1024 * 1024) << " MB";

    // 3. Generate Matrix
    LOG_INCREMENT_STAGE(10);
    LOG(LOG_INFO) << "[Bench] Initializing CUDA MatrixGenerator...";
    MatrixGenerator gen(gen_cfg.seed);
    
    LOG_INCREMENT_STAGE(10);
    LOG(LOG_INFO) << "[Bench] Generating Factor Base (" << gen_cfg.cols << ")...";
    gen.generate_factor_base(gen_cfg.cols);

    LOG_INCREMENT_STAGE(10);
    LOG(LOG_INFO) << "[Bench] Generating Matrix A (" << gen_cfg.rows << "x" << gen_cfg.cols << ")...";
    HostMatrix A_rect = gen.generate_matrix(gen_cfg.rows, gen_cfg.alpha, true);
    log_matrix_stats(A_rect, "Generated A (Rectangular)");

    // 4. Finalize Solver Config
    // Square dimension calculation
    int D = std::max(gen_cfg.rows, gen_cfg.cols);
    args.solver_cfg.nrows = D;
    
    // 5. Prepare Solver Input (Pad to Square)
    LOG_INCREMENT_STAGE(10);
    HostMatrix A_square = pad_to_square(A_rect);
    
    if (!args.solver_cfg.checkpoint_prefix.empty()) {
        io::BWIOSystem io(args.solver_cfg.checkpoint_prefix);
        io.save_matrix_A(A_square);
    }

    LOG_INCREMENT_STAGE(10);
    
    // 6. Run Solver
    BlockWiedemannSolver solver(args.solver_cfg, A_square);
    solver.Solve();

    // 7. Verify Results (Bench side validation)
    // Only verifies if solutions were requested and found
    if (!args.solver_cfg.stage3_skip) {
        const auto& solutions = solver.get_solutions();
        if (solutions.empty()) {
            LOG(LOG_WARNING) << "[Bench] No solutions found (or Stage 3 skipped/failed).";
        } else {
            LOG(LOG_INFO) << "[Bench] Verifying " << solutions.size() << " solutions against A_rect...";
            bool all_ok = true;
            int idx = 0;
            for(const auto& sol : solutions) {
                idx++;
                if (!verify_kernel_AT(A_rect, sol)) {
                    all_ok = false;
                    LOG(LOG_ERROR_CRITICAL) << "[Bench] Solution #" << idx << " is invalid.";
                }
            }
            if (all_ok) {
                LOG(LOG_INFO) << "[Bench] All " << solutions.size() << " solutions verified successfully.";
            } else {
                LOG(LOG_ERROR_CRITICAL) << "[Bench] Verification FAILED.";
                return 1;
            }
        }
    }

    return 0;
}
