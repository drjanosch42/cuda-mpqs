// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <vector>
#include <memory>
#include <string>

#include "lingen/io/bw_io.h"
#include <cuda_runtime.h>
#include "bw_spmm_interface.h"
#include "bw_solution_view.h"

class BlockWiedemannSpMM;

namespace lingen {

struct BwOperator; 

namespace stage1 { class KrylovSequenceGenerator; }
namespace stage2 { class BasecaseSolver; }
namespace stage3 { class SolutionReconstructor; }

/**
 * @brief Global Configuration for the Block Wiedemann Solver.
 * 
 * This structure acts as the "Single Source of Truth" for the entire solver pipeline.
 * It is used to synthesize the specific configurations for AutoTune and Stages 1-3.
 * 
 * Naming Convention:
 *  - No prefix: Mathematical invariants or global topology.
 *  - global_:   Master switches affecting multiple stages (overrides).
 *  - autotune_: Auto-Tune specific settings.
 *  - stage1_:   Stage 1 (Krylov Sequence Generation) specific.
 *  - stage2_:   Stage 2 (Linear Generator / Lingen) specific.
 *  - stage3_:   Stage 3 (Solution Reconstruction) specific.
 */
struct BWSolverConfig {
    // =========================================================================
    // 1. HARDWARE & TOPOLOGY (Invariants)
    // =========================================================================

    /// @brief CUDA Device ID.
    int device_id = 0;

    /// @brief Auto-detected at construction: true if running on Jetson Orin (SM 8.7).
    /// When true, the constructor applies memory-appropriate defaults before UpdateAllConfigs().
    bool is_jetson = false;

    /// @brief Matrix Dimension N (padded).
    /// This defines the size of the vector space. Must match the input matrix.
    int nrows = 0;

    /// @brief Block Size 'm' (Sequence Row Dimension).
    /// Mathematical invariant: Must be consistent across all stages.
    /// Determines rows in S_k and columns in projection block X.
    int m_block = 64;

    /// @brief Block Size 'n' (Sequence Column Dimension).
    /// Mathematical invariant: Must be consistent across all stages.
    /// Determines columns in S_k and columns in starting block Y (Z).
    int n_block = 64;

    /// @brief If true, m_block/n_block were explicitly set by the user (CLI).
    /// Jetson resource-limit overrides will not apply.
    bool block_size_pinned = false;

    // =========================================================================
    // 2. GLOBAL BEHAVIORS (Master Switches)
    // =========================================================================

    /// @brief Solve Transposed Problem (Left Kernel).
    /// If true, the solver finds x such that A^T x = 0. 
    /// Affects operator choice in Stage 1/3 and matrix dimensions.
    bool solve_transposed = false;

    /// @brief Base Random Seed.
    /// Used to initialize PRNGs. Stage-specific seeds are derived deterministically 
    /// from this unless manually overridden in stage configs (future).
    int seed = 12345;

    /// @brief Checkpoint Directory Prefix.
    /// Base path for all I/O. E.g., "/tmp/bw_run1/".
    /// If empty, I/O is typically disabled unless stage flags force it.
    std::string checkpoint_prefix = "";

    /// @brief Master Switch: Compute Hashes.
    /// If true, overrides stage-specific flags to ENABLE hashing everywhere.
    /// Useful for generating regression data.
    bool enable_all_hashing = false;

    /// @brief Master Switch: Validate Hashes.
    /// If true, overrides stage-specific flags to ENABLE validation against 
    /// provided hash values everywhere.
    bool enable_all_validation = false;

    /// @brief Master Switch: Oracle Verification.
    /// If true, enables heavy-duty bit-exact verification against legacy/oracle 
    /// methods in all applicable stages (S2 and S3). Significantly slows execution.
    bool enable_all_oracle_verification = false;

    // =========================================================================
    // 3. AUTO-TUNE CONFIGURATION (autotune_)
    // =========================================================================

    /// @brief Enable SpMM Kernel Tuning.
    /// Benchmarks sparse matrix multiplication variants (blocking, reordering).
    bool autotune_tune_spmm = true;

    /// @brief Enable Polynomial Arithmetic Tuning.
    /// Benchmarks FFT/Multiplication kernels for Stage 2 (Reserved).
    bool autotune_tune_poly = false;

    /// @brief Skip if Tuning Database Exists.
    /// If true, loads parameters from db_path if available, skipping benchmarks.
    bool autotune_skip_if_present = true;

    /// @brief Force SpMM Pre-Flight Verification.
    /// Runs a correctness check (A * x) on the optimized SpMM kernels immediately 
    /// after tuning. Essential for "Pre-Flight" safety.
    bool autotune_verify_spmm = true;

    /// @brief Tuning Database Path.
    std::string autotune_db_path = "bw_tuning.csv";

    // =========================================================================
    // 3b. AUTO-TUNE GPU-ONLY PIPELINE (autotune_)
    // =========================================================================

    /// @brief Use GPU-only autotuning pipeline.
    /// When true, all format preprocessing happens on-device; the tuning phase
    /// produces final device data that compile() adopts directly (no re-upload).
    /// When false, uses the legacy CPU-based preprocessing path.
    bool autotune_gpu_only = true;

    /// @brief M4RM batch size for the dense head (rows grouped per lookup).
    /// Valid values: 8, 32, 64, 128. Only affects the dense head segment.
    int autotune_m4rm_rows = 8;

    /// @brief Skip M4RM benchmarking and always use M4RM for the dense head.
    /// Safe because M4RM is nearly always optimal for very dense rows.
    bool autotune_skip_m4rm_benchmark = false;

    /// @brief Block growth strategy for atomic block partitioning.
    /// FIXED: current behaviour (8-row head, doubling to 1024, then 65536-row steps).
    /// EXPONENTIAL: 8-row head then doubling (8,16,32,...,65536) — ~19 blocks total
    /// instead of ~7500, dramatically reducing DP solver cost.
    enum class BlockGrowth { FIXED, EXPONENTIAL };
    BlockGrowth autotune_block_growth = BlockGrowth::EXPONENTIAL;

    /// @brief Initial block size for exponential growth (rows).
    int autotune_initial_block_size = 8;

    /// @brief Maximum block size (rows). Growth caps at this value.
    int autotune_max_block_size = 65536;

    /// @brief Per-format enable flags. Disabling a format skips its GPU
    /// conversion and benchmarking entirely, reducing tuning time.
    bool autotune_enable_tiled_coo = true;
    bool autotune_enable_delta16 = true;
    bool autotune_enable_pfor_be = true;
    bool autotune_enable_golomb = true;

    // =========================================================================
    // 4. STAGE 1: KRYLOV GENERATION (stage1_)
    // =========================================================================

    /// @brief Skip Stage 1.
    /// Useful if restarting from a checkpoint (S sequence file).
    bool stage1_skip = false;

    /// @brief Krylov Sequence Length (L).
    /// Target number of iterations. If 0, automatically calculated based on N, m, n.
    int stage1_seq_len = 0;

    /// @brief GPU Pipeline Batch Size.
    /// Number of vectors processed in the GPU pipeline before PCIe transfer.
    /// Tuning parameter for latency hiding vs memory usage.
    int stage1_gpu_batch_size = 64;

    /// @brief Prefer the faster SpMM operator in Stage 1.
    /// When true, Stage 1 dynamically selects between A and A^T based on
    /// autotune measurements, using the identity S_k = (Z^T · B_alt^k · X)^T.
    /// Default: true. Set false to force the solve_transposed-determined operator.
    bool stage1_prefer_faster_op = true;

    /// @brief Keep S sequence on GPU for direct Stage 2 handoff.
    /// Eliminates the GPU→CPU→GPU round-trip between stages.
    /// Default: true. Set false to fall back to host-based pipeline
    /// (e.g., if GPU memory is constrained).
    bool stage1_keep_S_on_device = true;

    /// @brief Save S sequence to disk after Stage 1 completes.
    /// Independent of stage1_keep_S_on_device — can save to disk AND keep on device.
    /// Requires stage1_S_disk_path to be non-empty.
    bool stage1_save_S_to_disk = false;

    /// @brief File path for saving S to disk.
    /// Used when stage1_save_S_to_disk is true.
    std::string stage1_S_disk_path = "";

    /// @brief Stage 1 Hashing.
    /// Computes hashes for X, Y, and S.
    bool stage1_compute_hashes = false;

    /// @brief Stage 1 Hash Validation.
    /// Compares computed hashes against stage1_hash_*.
    bool stage1_hash_validation = false;

    // --- Validation Values (Stage 1) ---
    uint64_t stage1_hash_X = 0;
    uint64_t stage1_hash_Y = 0;
    uint64_t stage1_hash_S = 0;

    /// @brief Load Checkpoints.
    /// Checks for existence of X, Y, S files and loads them instead of computing.
    bool stage1_load_checkpoints = true;

    /// @brief Save Checkpoints.
    /// Saves generated X, Y, S to disk.
    bool stage1_save_checkpoints = false;

    // --- File Suffixes (Stage 1) ---
    std::string stage1_suffix_X = "_X.bin";
    std::string stage1_suffix_Y = "_Y.bin";
    std::string stage1_suffix_S = "_S.bin";

    // =========================================================================
    // 5. STAGE 2: LINEAR GENERATOR / LINGEN (stage2_)
    // =========================================================================

    /// @brief Skip Stage 2.
    /// Useful if restarting from a Generator (Pi) checkpoint.
    bool stage2_skip = false;

    /// @brief Stage 2 Sequence Length.
    /// Length of S to consume. If 0, uses all available data or stage1_seq_len.
    int stage2_seq_len = 0;

    /// @brief Degree Bound Delta.
    /// Expected degree of generator. If 0, defaults to seq_len / 2.
    int stage2_delta = 0;

    /// @brief GPU Hybrid Mode.
    /// If true, uses experimental GPU offloading for the recursive step.
    bool stage2_gpu_mode = false;

    /// @brief Load S sequence from disk instead of receiving from Stage 1.
    /// Useful for resuming Stage 2 independently or debugging.
    /// Requires stage2_S_disk_path to be non-empty.
    bool stage2_load_S_from_disk = false;

    /// @brief File path for loading S from disk.
    /// Used when stage2_load_S_from_disk is true.
    std::string stage2_S_disk_path = "";

    /// @brief Annihilation Check (GPU).
    /// Verifies S(x) * Pi(x) = 0 using high-throughput GPU kernels.
    bool stage2_check_annihilation_gpu = false;

    /// @brief Annihilation Check (Legacy CPU).
    /// Verifies S(x) * Pi(x) = 0 using bit-exact CPU arithmetic. Very slow.
    bool stage2_check_annihilation_legacy = false;

    /// @brief Reference State Checks.
    /// Logs hashes of internal state (Pi, Gamma) periodically.
    bool stage2_reference_check = false;

    /// @brief Reference Check Interval.
    int stage2_reference_check_interval = 100;

    /// @brief Internal Oracle Verification.
    /// Step-by-step comparison against CPU oracle during GPU execution.
    bool stage2_internal_oracle_verification = false;

    /// @brief Post-Run Legacy Check.
    /// Runs the independent Legacy solver after the main solver for bit-exact comparison.
    bool stage2_post_run_legacy_check = false;

    /// @brief Stage 2 Hashing.
    /// Computes hash of the generator Pi.
    bool stage2_compute_hashes = false;

    /// @brief Stage 2 Hash Validation.
    /// Compares against stage2_hash_Pi.
    bool stage2_hash_validation = false;

    // --- Validation Values (Stage 2) ---
    uint64_t stage2_hash_S = 0;  // Expected input hash
    uint64_t stage2_hash_Pi = 0; // Expected output hash

    /// @brief Save Checkpoints.
    /// Saves the resulting polynomial Pi to disk.
    bool stage2_save_checkpoints = false;

    // --- File Suffixes (Stage 2) ---
    std::string stage2_suffix_S = "_S.bin";   // Input file suffix
    std::string stage2_suffix_Pi = "_Pi.bin"; // Output file suffix

    // =========================================================================
    // 6. STAGE 3: SOLUTION RECONSTRUCTION (stage3_)
    // =========================================================================

    /// @brief Skip Stage 3.
    bool stage3_skip = false;

    /// @brief Reconstruction Batch Mode.
    /// If true, uses parallelized reconstruction. If false, uses serial legacy mode.
    bool stage3_batch_mode = true;

    /// @brief Maximum Solutions to Find.
    /// -1 for all candidates (find basis).
    int stage3_max_solutions = -1;

    /// @brief Un-permute Solutions.
    /// If true, applies P^T to results. Typically matches autotune_tune_spmm.
    bool stage3_perform_unpermutation = true;

    /// @brief History Buffer Depth.
    /// Size of the backtracking circular buffer (Stage 3 memory usage vs safety).
    int stage3_history_depth = 64;

    /// @brief Annihilation Check Interval.
    /// Frequency of checking for zero-columns during stripping.
    int stage3_check_interval = 16;

    /// @brief Valuation Stripping Limit.
    /// 0 = internal heuristic. >0 = strict limit.
    int stage3_stripping_limit = 0;

    /// @brief Oracle Verification.
    /// Runs "Record (Legacy) -> Verify (Batch)" pipeline.
    bool stage3_oracle_verification = false;

    /// @brief Stage 3 Hashing.
    /// Computes hashes of found solutions.
    bool stage3_compute_hashes = false;

    /// @brief Stage 3 Hash Validation.
    /// Compares hash of the *first* solution against stage3_hash_first_solution.
    bool stage3_hash_validation = false;

    // --- Validation Values (Stage 3) ---
    uint64_t stage3_hash_first_solution = 0;

    /// @brief Save Solutions.
    /// Saves reconstructed kernel vectors to disk.
    bool stage3_save_solutions = false;

    // --- File Suffixes (Stage 3) ---
    std::string stage3_suffix_Y = "_Y.bin";
    std::string stage3_suffix_Pi = "_Pi.bin";
    std::string stage3_suffix_solutions = "_solutions.bin";

    // =========================================================================
    // 7. CUDA GRAPH ACCELERATION (graph_)
    // =========================================================================

    /// @brief Master switch: enable CUDA graph capture across all stages.
    /// Individual per-stage use_cuda_graph flags are propagated from this.
    /// Default: false (opt-in). Requires CUDA 10.1+ and ENABLE_CUDA_GRAPHS=ON at build time.
    bool graph_enable = false;

    /// @brief Minimum loop iteration count before attempting graph capture.
    /// Graph capture + instantiation costs ~100-500 us; skip for short loops.
    /// Default: 100.
    int graph_min_iterations = 100;
};
  
/**
 * @brief Configuration parameters for Auto-Tuning and Pre-Flight Checks.
 * 
 * Controls the initial benchmarking and optimization phase.
 */
struct BWAutoTuneConfig {
    // -------------------------------------------------------------------------
    // Optimization Targets
    // -------------------------------------------------------------------------

    /// @brief Enable SpMM Tuning.
    /// If true, benchmarks various kernel parameters (blocking, ordering) for
    /// the sparse matrix operations and selects the optimal configuration.
    bool tune_spmm = true;

    /// @brief Enable Polynomial Arithmetic Tuning.
    /// Reserved for future tuning of FFT/multiplication kernels in Stage 2.
    bool tune_poly = false;

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /// @brief Skip Tuning if Database Exists.
    /// If true, loads parameters from tuning_db_path if available, skipping benchmarks.
    bool skip_if_present = true;

    /// @brief Force SpMM Verification.
    /// If true, runs a correctness check (A * x) on the optimized SpMM kernels
    /// immediately after tuning. Essential for "Pre-Flight" safety.
    bool verify_spmm = true;

    // -------------------------------------------------------------------------
    // Input / Output
    // -------------------------------------------------------------------------

    /// @brief Path to the Tuning Database (CSV).
    std::string tuning_db_path = "bw_tuning.csv";

    // -------------------------------------------------------------------------
    // GPU-Only Pipeline
    // -------------------------------------------------------------------------

    /// @brief Use GPU-only autotuning pipeline.
    bool gpu_only = true;

    /// @brief M4RM batch size.
    int m4rm_rows = 8;

    /// @brief Skip M4RM benchmarking.
    bool skip_m4rm_benchmark = false;

    /// @brief Block growth strategy.
    BWSolverConfig::BlockGrowth block_growth = BWSolverConfig::BlockGrowth::EXPONENTIAL;

    /// @brief Initial block size for exponential growth.
    int initial_block_size = 8;

    /// @brief Maximum block size.
    int max_block_size = 65536;

    /// @brief Per-format enable flags.
    bool enable_tiled_coo = true;
    bool enable_delta16 = true;
    bool enable_pfor_be = true;
    bool enable_golomb = true;
};

/**
 * @brief Configuration parameters for Stage 1 (Krylov Sequence Generation).
 * 
 * This structure isolates all settings required to generate the sequence
 * S_k = X^T * B^(k+1) * Z. It controls problem dimensions, execution tuning,
 * validation modes, and I/O naming conventions.
 */
struct BWStage1Config {
    // -------------------------------------------------------------------------
    // Problem Definition
    // -------------------------------------------------------------------------
    
    /// @brief Optionally Skip Stage 1
    bool skip = false;
  
    /// @brief Matrix dimension N (padded). Must match the SpMM operator size.
    int nrows = 0; 

    /// @brief Block size 'm'. 
    /// Corresponds to the number of columns in the projection block X 
    /// and the number of rows in the sequence matrices S_k.
    int m_block = 64; 

    /// @brief Block size 'n'. 
    /// Corresponds to the number of columns in the starting block Z (or Y)
    /// and the number of columns in the sequence matrices S_k.
    int n_block = 64; 

    /// @brief Target length of the Krylov sequence (L).
    /// Typically L = N/m + N/n + margin.
    int seq_len = 0;

    /// @brief Solve Transposed Mode.
    /// If true, the operator B is A^T (finding left kernel).
    /// If false, the operator B is A (finding right kernel).
    bool solve_transposed = false;

    // -------------------------------------------------------------------------
    // Execution & Tuning
    // -------------------------------------------------------------------------

    /// @brief Pipeline Batch Size (Circular Buffer Depth).
    /// Determines how many vectors V are processed in the GPU pipeline before
    /// downloading the resulting scalar products S to the host.
    /// Higher values amortize PCIe latency but increase GPU memory usage.
    /// Default: 64.
    int gpu_batch_size = 64;

    /// @brief Prefer the faster operator based on autotune results.
    /// Propagated from BWSolverConfig::stage1_prefer_faster_op.
    bool prefer_faster_op = true;

    /// @brief Keep S on device for zero-copy Stage 2 handoff.
    bool keep_S_on_device = true;

    /// @brief Save S to disk after generation.
    bool save_S_to_disk = false;

    /// @brief Disk path for S save.
    std::string S_disk_path = "";

    /// @brief Random Seed.
    /// Used to initialize the PRNG for creating random blocks X and Y
    /// if they are not loaded from disk.
    int seed = 12345;

    // -------------------------------------------------------------------------
    // Validation & Integrity
    // -------------------------------------------------------------------------

    /// @brief Enable Hashing.
    /// If true, FNV1a hashes are computed for X, Y, and the sequence S
    /// for logging and integrity reporting.
    bool compute_hashes = false;

    /// @brief Enable Hash Validation.
    /// If true, the computed hashes are compared against the provided expected 
    /// values (hash_X, hash_Y, hash_S). Mismatches trigger warnings or errors.
    bool hash_validation = false;

    /// @brief Expected FNV1a hash for the projection block X.
    uint64_t hash_X = 0;

    /// @brief Expected FNV1a hash for the starting block Z (Y in code).
    uint64_t hash_Y = 0;

    /// @brief Expected FNV1a hash for the resulting sequence S.
    uint64_t hash_S = 0;

    // -------------------------------------------------------------------------
    // Input / Output
    // -------------------------------------------------------------------------

    /// @brief Checkpoint Directory / Prefix.
    /// Used to load/save X, Y, and S. If empty, I/O is disabled.
    std::string checkpoint_prefix = "";

    /// @brief Load Checkpoints Flag.
    /// If true, checks for existence of files and loads them instead of computing.
    bool load_checkpoints = true;

    /// @brief Save Checkpoints Flag.
    /// If true, saves generated data (X, Y, S) to disk.
    bool save_checkpoints = true;

    // -------------------------------------------------------------------------
    // File Naming Conventions
    // -------------------------------------------------------------------------

    /// @brief Filename suffix for the projection block X.
    /// Full path: checkpoint_prefix + suffix_X
    std::string suffix_X = "_X.bin";

    /// @brief Filename suffix for the starting block Z (Y in code).
    /// Full path: checkpoint_prefix + suffix_Y
    std::string suffix_Y = "_Y.bin";

    /// @brief Filename suffix for the sequence S.
    /// Full path: checkpoint_prefix + suffix_S
    std::string suffix_S = "_S.bin";

    // -------------------------------------------------------------------------
    // CUDA Graph Acceleration
    // -------------------------------------------------------------------------

    /// @brief Enable CUDA graph capture for this stage's hot loop.
    /// Default: false. Propagated from BWSolverConfig::graph_enable.
    bool use_cuda_graph = false;

    /// @brief Minimum iteration count for graph capture.
    /// Propagated from BWSolverConfig::graph_min_iterations.
    int graph_min_iterations = 100;
};

/**
 * @brief Configuration parameters for Stage 2 (Linear Generator / Lingen).
 * 
 * This structure isolates all settings required to solve the Block Berlekamp-Massey 
 * problem, finding the generator matrix polynomial \f$ \pi(x) \f$ such that 
 * \f$ S(x) \cdot \pi(x) \equiv 0 \pmod{x^L} \f$.
 */
struct BWStage2Config {
    // -------------------------------------------------------------------------
    // Problem Definition
    // -------------------------------------------------------------------------

    /// @brief Block size 'm'. Number of rows in sequence matrices S_k.
    int m_block = 64; 

    /// @brief Block size 'n'. Number of columns in sequence matrices S_k.
    int n_block = 64;

    /// @brief Length of the input sequence S (L).
    int seq_len = 0;

    // -------------------------------------------------------------------------
    // Algorithm Parameters
    // -------------------------------------------------------------------------

    /// @brief The degree bound \f$ \delta \f$ for the recursive solver.
    /// Typically \f$ \delta \approx L/2 \f$. 
    /// If set to 0, the solver will automatically derive it as seq_len / 2.
    int delta = 0;

    // -------------------------------------------------------------------------
    // Execution Mode
    // -------------------------------------------------------------------------

    /// @brief Enable GPU Hybrid Mode.
    /// If true, uses the experimental hybrid solver where scalar discrepancies 
    /// and basis updates are offloaded to the GPU, while pivot decisions remain on CPU.
    /// If false, uses the reference CPU-only implementation.
    bool gpu_mode = false;

    /// @brief Load S from disk instead of receiving from Stage 1.
    bool load_S_from_disk = false;

    /// @brief Disk path for loading S.
    std::string S_disk_path = "";

    // -------------------------------------------------------------------------
    // Verification & Integrity (Runtime)
    // -------------------------------------------------------------------------

    /// @brief Enable Optimized GPU Annihilation Check.
    /// Verifies \f$ S(x) \cdot \pi(x) \equiv 0 \f$ using high-throughput GPU kernels.
    /// Run at the end of the solve phase.
    bool check_annihilation_gpu = false;

    /// @brief Enable Legacy CPU Annihilation Check.
    /// Verifies the generator property using the slow, bit-exact reference implementation.
    /// Very expensive for large instances.
    bool check_annihilation_legacy = false;

    /// @brief Enable Reference State Checks.
    /// If true, the solver computes and logs hashes of the internal state (Pi, Gamma)
    /// periodically during iteration. Useful for regression testing.
    bool reference_check = false;

    /// @brief Interval for Reference Checks.
    /// Determines how often (in iterations 't') the state hash is logged.
    /// Default: 100.
    int reference_check_interval = 100;

    /// @brief Enable Internal Oracle Verification.
    /// If true and running in GPU Hybrid mode, the solver performs a strict step-by-step 
    /// comparison of intermediate results against an internal CPU oracle. 
    /// Aborts execution immediately upon divergence.
    bool internal_oracle_verification = false;

    // -------------------------------------------------------------------------
    // Verification (Post-Run)
    // -------------------------------------------------------------------------

    /// @brief Enable Post-Run Legacy Check.
    /// If true, runs the independent Legacy solver after the main solver finishes 
    /// and compares the final \f$ \pi(x) \f$ bit-for-bit.
    bool post_run_legacy_check = false;

    // -------------------------------------------------------------------------
    // Hashing & Output Validation
    // -------------------------------------------------------------------------

    /// @brief Enable Result Hashing.
    /// Computes FNV1a hash of the resulting \f$ \pi(x) \f$.
    bool compute_hashes = false;

    /// @brief Enable Golden Hash Validation.
    /// Compares computed hashes against `expected_hash_Pi`.
    bool hash_validation = false;

    /// @brief Expected FNV1a hash for the input sequence S.
    /// Used if loading S from disk or verifying memory transfer from Stage 1.
    uint64_t expected_hash_S = 0;

    /// @brief Expected FNV1a hash for the output generator \f$ \pi(x) \f$.
    uint64_t expected_hash_Pi = 0;

    // -------------------------------------------------------------------------
    // Input / Output
    // -------------------------------------------------------------------------

    /// @brief Checkpoint Directory / Prefix.
    std::string checkpoint_prefix = "";

    /// @brief Save Checkpoints Flag.
    /// If true, saves the resulting polynomial \f$ \pi(x) \f$ to disk.
    bool save_checkpoints = true;

    /// @brief Filename suffix for the input sequence S.
    /// Full path: checkpoint_prefix + suffix_S
    std::string suffix_S = "_S.bin";

    /// @brief Filename suffix for the output generator \f$ \pi(x) \f$.
    /// Full path: checkpoint_prefix + suffix_Pi
    std::string suffix_Pi = "_Pi.bin";

    // -------------------------------------------------------------------------
    // CUDA Graph Acceleration
    // -------------------------------------------------------------------------

    /// @brief Enable CUDA graph capture for this stage's hot loop.
    /// Default: false. Propagated from BWSolverConfig::graph_enable.
    bool use_cuda_graph = false;

    /// @brief Minimum iteration count for graph capture.
    /// Propagated from BWSolverConfig::graph_min_iterations.
    int graph_min_iterations = 100;
};

/**
 * @brief Configuration parameters for Stage 3 (Solution Reconstruction).
 * 
 * This structure isolates all settings required to reconstruct kernel vectors
 * from the generator matrix \f$ \pi(x) \f$. It controls the evaluation strategy,
 * GPU memory allocation for history, and rigorous validation modes.
 */
struct BWStage3Config {
    // -------------------------------------------------------------------------
    // Problem Definition
    // -------------------------------------------------------------------------

    /// @brief Matrix dimension N (padded).
    int nrows = 0; 

    /// @brief Block size 'm'. Number of rows in sequence matrices S_k.
    int m_block = 64; 

    /// @brief Block size 'n'. Number of columns in sequence matrices S_k.
    int n_block = 64;

    /// @brief Solve Transposed Mode.
    /// If true, reconstructs x such that A^T x = 0.
    bool solve_transposed = false;

    // -------------------------------------------------------------------------
    // Execution Strategy
    // -------------------------------------------------------------------------

    /// @brief Enable Batch Mode.
    /// If true, uses the optimized parallel reconstructor that processes 
    /// multiple candidate columns from \f$ \pi(x) \f$ simultaneously.
    /// If false, uses the serial legacy implementation.
    bool batch_mode = true;

    /// @brief Maximum Solutions to Reconstruct.
    /// The algorithm terminates after finding this many linearly independent vectors.
    /// Set to -1 for no limit (find all candidates).
    int max_solutions = -1;

    /// @brief Un-permute Solutions.
    /// If SpMM tuning used reordering, the raw results correspond to the permuted matrix.
    /// If true, the solver applies the inverse permutation P^T to results before saving.
    /// Typically set based on AutoTuneConfig::tune_spmm.
    bool perform_unpermutation = true;

    // -------------------------------------------------------------------------
    // Algorithm Tuning (Batch Mode Internal Constants)
    // -------------------------------------------------------------------------

    /// @brief History Buffer Depth.
    /// Number of steps to keep in the circular buffer for backtracking.
    /// A larger depth allows recovering solutions that "vanish" slowly but uses more GPU memory.
    /// Default: 64.
    int history_depth = 64;

    /// @brief Annihilation Check Interval.
    /// Frequency (in steps) of checking if candidate vectors have become zero.
    /// Must be significantly smaller than history_depth to ensure recovery.
    /// Default: 16.
    int check_interval = 16;

    /// @brief Valuation Stripping Limit.
    /// Maximum number of iterations for the 'w <- B*w' loop.
    /// If 0, the solver uses an internal heuristic (approx N/1024 + 128).
    /// Set to a positive value to enforce a strict upper bound.
    int stripping_step_limit = 0;

    // -------------------------------------------------------------------------
    // Verification & Integrity
    // -------------------------------------------------------------------------

    /// @brief Enable Oracle Verification.
    /// If true, enables the rigorous "Record and Verify" pipeline:
    /// 1. Runs the Legacy solver to record intermediate hashes (Oracle).
    /// 2. Runs the Optimized solver and asserts it matches the Oracle.
    bool oracle_verification = false;

    /// @brief Enable Hashing.
    /// Computes FNV1a hashes of the found solution vectors.
    bool compute_hashes = false;

    /// @brief Enable Hash Validation.
    /// Compares the hash of the *first* found solution against expected_hash_first_solution.
    bool hash_validation = false;

    /// @brief Expected FNV1a hash for the first found solution.
    /// Used for regression testing (Golden Run).
    uint64_t expected_hash_first_solution = 0;

    // -------------------------------------------------------------------------
    // Input / Output
    // -------------------------------------------------------------------------

    /// @brief Checkpoint Directory / Prefix.
    std::string checkpoint_prefix = "";

    /// @brief Save Solutions Flag.
    /// If true, saves the reconstructed kernel vectors to disk.
    bool save_solutions = false;

    /// @brief Filename suffix for input starting block Y (Z).
    std::string suffix_Y = "_Y.bin";

    /// @brief Filename suffix for input generator Pi.
    std::string suffix_Pi = "_Pi.bin";

    /// @brief Filename suffix for output solutions.
    /// Full path: checkpoint_prefix + suffix_solutions
    std::string suffix_solutions = "_solutions.bin";

    // -------------------------------------------------------------------------
    // CUDA Graph Acceleration
    // -------------------------------------------------------------------------

    /// @brief Enable CUDA graph capture for this stage's hot loop.
    /// Default: false. Propagated from BWSolverConfig::graph_enable.
    bool use_cuda_graph = false;

    /// @brief Minimum iteration count for graph capture.
    /// Propagated from BWSolverConfig::graph_min_iterations.
    int graph_min_iterations = 100;
};

class BlockWiedemannSolver {
public:
    explicit BlockWiedemannSolver(BWSolverConfig config, const HostMatrix& A);
    ~BlockWiedemannSolver();

    void Solve();

    void AutoTune(); 
    void RunStage1();
    void RunStage2();
    void RunStage3();

    const std::vector<std::vector<uint64_t>>& get_solutions() const { return solutions_; }

    /// @brief Returns a device-side view of the packed solution bit-matrix.
    /// Valid after Solve() completes. Returns zeroed view if no solutions exist.
    BWKernelSolutionView get_device_solutions() const {
        return {d_solutions_,
                static_cast<uint32_t>(num_device_solutions_),
                static_cast<uint32_t>((stage3_cfg_.nrows + 63) / 64),
                static_cast<uint32_t>(stage3_cfg_.nrows)};
    }

private:
    // Global configuration struct
    BWSolverConfig cfg_;
    // Localized configuration structs
    BWAutoTuneConfig autotune_cfg_;
    BWStage1Config stage1_cfg_;
    BWStage2Config stage2_cfg_;
    BWStage3Config stage3_cfg_;

    // Configuration Synchronization Methods
    void UpdateAutoTuneConfig();
    void UpdateStage1Config();
    void UpdateStage2Config();
    void UpdateStage3Config();
    void UpdateAllConfigs();  
  
    const HostMatrix& A_host_;
    io::BWIOSystem io_;

    std::vector<uint64_t> hX_, hY_, hS_;
    std::vector<uint64_t> hPi_;
    std::vector<int> hGamma_;
    std::vector<std::vector<uint64_t>> solutions_;

    cudaStream_t stream_ = nullptr;
    std::unique_ptr<::BlockWiedemannSpMM> spmm_;

    /// @brief Stored autotune report for Stage 1 operator selection.
    SpMMPerformanceReport autotune_report_ = {};

    /// @brief Device buffer for full S sequence (Stage 1 → Stage 2 zero-copy).
    uint64_t* d_S_sequence_ = nullptr;

    /// @brief Total allocation size of d_S_sequence_ in bytes.
    size_t d_S_sequence_bytes_ = 0;

    /// @brief Device-side packed solution matrix (num_solutions × words_per_vec uint64_t).
    uint64_t* d_solutions_ = nullptr;

    /// @brief Number of solutions stored on device.
    size_t num_device_solutions_ = 0;

    std::unique_ptr<BwOperator> make_operator() const;
};

// Pads a rectangular matrix to be square (D x D) where D = max(rows, cols).
// Block Wiedemann requires a square linear operator for iteration.
HostMatrix pad_to_square(const HostMatrix& in);

} // namespace lingen
