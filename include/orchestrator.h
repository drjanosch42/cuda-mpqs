// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <string>
#include <vector>
#include <deque>
#include <map>
#include <memory>
#include <cstdint>
#include <atomic>
#include <thread>

// Common and Data Structures
#include "uint512.cuh"
#include "hpc_logger.h"
#include "common.h"           // factoringData
#include "prime_algorithms.h"  // generateFactorBase, determineParams

// Pipeline Components
#include "device_sieving_controller.h"
#include "mpqs_soa.h"
#include "postprocessing.h"
#include "largeprime.h"
#include "matrix_constructor.h"
#include "bw_solver.h"
#include "mpqs_common.h"      // defines HostMatrix (transitively via cuda_spmm/include/common.h)
#include "sqrt_step.h"
#include "autotune.h"
#include "merge_tree.h"
#include "merge_filter.h"
#include "expanded_matrix.h"
#include "preprocess.h"        // PreprocessResultV2 (M9f)
#include <optional>

// Forward declarations — defined in src/cluster/
namespace mpqs::cluster {
    class DataTap;
    class AccumulatorQueue;
    class RelationAccumulator;
    class FinalBatchHandoff;
    class CPULargePrimeTable;
    class DirectChannel;
    class CommBackend;
    class WorkPool;
    class ChunkScheduler;
}

namespace mpqs {

/**
 * @brief Execution Mode for the Orchestrator.
 */
enum class ExecutionMode {
    FULL_PIPELINE,      ///< Tuning -> Sieve -> Matrix -> Solve -> Sqrt
    SIEVE_ONLY,         ///< Sieve -> Write Relations to Disk -> Exit
    LINALG_ONLY,        ///< Load Matrix (not impl) -> Solve -> Exit
    SQRT_ONLY,          ///< Load Kernel Vectors -> Find Factors
    PARAM_TEST,         ///< Run parameter test/exploration -> Exit
    AUTOTUNE_ONLY,      ///< Tuning -> Autotune -> Print results -> Exit
    MATRIX_ONLY         ///< Load v2 relations -> Matrix -> BW -> Sqrt
};

/// Execution topology for the pipeline.
enum class ClusterMode { SOLO, COORDINATOR, WORKER };

/// Matrix construction mode.
/// AUTO: select based on LP fraction vs. lp_preprocess_threshold.
/// LEGACY: always use projected (F+2)-column matrix (pre-combine LP via sqrt LP handler).
/// PREPROCESS: always use expanded (F+2+L)-column matrix with M2-M4 merge/filter pipeline.
enum class MatrixMode { AUTO, LEGACY, PREPROCESS };

/**
 * @brief Global Configuration for the MPQS Orchestrator.
 */
struct MPQSConfig {
    // Global
    mpqs::uint512 N;
    uint32_t device_id = 0;
    uint32_t node_id = 0;
    ExecutionMode mode = ExecutionMode::FULL_PIPELINE;
    bool disk_io = false;
    std::string work_dir = "./mpqs_work";

    // Tuning
    bool auto_tune_parameters = true;
    bool lp1_variation = false;    // overriden by auto_tune if auto_tune=true
    uint32_t fb_bound = 0;         // 0 = Auto-calculate "F"
    uint64_t lp1_bound = 0;        // 0 = Auto-calculate
    uint64_t lp1_max_witness_capacity = 0; // 0 = Auto-calculate
    uint32_t lp_interval = 1;      // 0 = auto (adaptive), N > 0 = process LP every N batches
                                       // Validated optimal for RSA-100: lp_interval=1, sieve_batch_size=32
    uint32_t target_relations = 0; // 0 = Auto-calculate (FB size + margin)
    double dedup_safety_factor = 1.05; ///< Oversample margin for dedup: collect (target * factor) relations.
                                       ///< Auto-set to 1.35 for inputs < 80 digits. CLI: --dedup_safety_factor
    double lp_matrix_threshold = 0.01;  ///< DEPRECATED: use lp_preprocess_threshold. Kept as alias.
    MatrixMode matrix_mode = MatrixMode::AUTO;  ///< Matrix construction mode. CLI: --matrix_mode legacy|preprocess
    int matrix_backend = 0;  ///< 0=CPU, 1=GPU, 2=AUTO. CLI: --matrix_backend cpu|gpu|auto
    double lp_preprocess_threshold = 0.55;      ///< Auto-detect threshold: LP fraction above this → PREPROCESS.
                                                 ///< CLI: --lp_preprocess_threshold
    double partial_subsample = 1.0;  ///< Fraction of partials/LP-combined to retain in matrix_only.
                                      ///< Preprocess: subsamples raw partials. Legacy: subsamples LP-combined.
                                      ///< Range [0.0, 1.0]. Default 1.0 (no subsampling). CLI: --partial_subsample
    double smooth_subsample = 1.0;   ///< Fraction of pure smooths (large_primes <= 1) to retain in matrix_only.
                                      ///< LP-combined relations are always retained (inverse of partial_subsample).
                                      ///< Range [0.0, 1.0]. Default 1.0 (no subsampling). CLI: --smooth_subsample
    double truncation_factor = 1.05; ///< Matrix truncation enable flag. > 0 = enabled, 0 = disabled.
                                      ///< Per M12-S1 the actual target is char-col-aware and excess-based:
                                      ///<   target_rows = n_cols + n_extra_cols + matrix_truncation_excess.
                                      ///< The factor is retained as an on/off switch for backward CLI compatibility.
                                      ///< CLI: --truncation_factor
    uint32_t matrix_truncation_excess = 200; ///< Excess rows above (n_cols + n_extra_cols) post-truncation.
                                              ///< Determines how overdetermined the post-augmentation matrix is.
                                              ///< CLI: --matrix_truncation_excess
    uint32_t compact_cycles = 5; ///< Maximum compact-merge cycles (GPU backend only).
                                  ///< 0 = single pass (no compaction, reverts to pre-M10 behavior).
                                  ///< CLI: --compact_cycles
    double matrix_gf2_floor_factor = 0.5;  ///< M12-S2 GF(2) column-diversity floor: stop
                                            ///< compact-merge cycles when surviving GF(2) cols fall
                                            ///< below max(min_floor, factor × initial_gf2_cols).
                                            ///< CLI: --matrix_gf2_floor_factor
    uint32_t matrix_gf2_min_floor = 8192;  ///< M12-S2 absolute lower bound on the GF(2) col floor.
                                            ///< Prevents triggering on tiny test matrices.
                                            ///< CLI: --matrix_gf2_min_floor
    uint32_t sieve_bound = 0;      // 0 = Auto-calculate "M"
    uint32_t sieve_hcube_dimension = 0;  // 0 = Auto-calculate
    uint32_t sieve_batch_size = 0;       // 0 = Auto-calculate
    uint32_t sieve_gms_num_blocks = 0;   // 0 = Auto-calculate
    uint32_t cuda_graph_unroll = 0;  ///< 0 = disabled. N > 0: capture N batches as CUDA graph.
                                     ///< Must be even (double-buffer constraint). Recommended: 2 or 4.
    double probe_timeout = 120.0;  ///< Hard timeout (seconds) for TruncatedSieveRun(). CLI: --probe_timeout
    bool estimate_only = false;    ///< Run truncated sieve in current topology, print estimate, exit.

    // Truncated sieve parameters
    uint64_t sieve_max_relations = 0;     ///< 0 = disabled. Stop sieve after N relations.
    uint64_t sieve_max_batches = 0;       ///< 0 = disabled. Stop sieve after N batch iterations.
    bool sieve_truncate_continue = false; ///< If true, continue pipeline after truncated sieve.

    // Buffer sizing overrides (0 = auto)
    uint64_t accum_buffer_size = 0;        // 0 = auto: max(4096, batch_size * 2048)
    uint64_t partial_buffer_size = 0;      // 0 = auto (1 x accum)
    uint64_t persistent_buffer_size = 0;   // 0 = auto (target * 2 + accum)
    uint64_t lp1_max_combined_output = 0;  // 0 = auto (32768)
    uint32_t lp1_hash_bits = 0;            // 0 = auto-derived

    // Pin tracking: records which params were explicitly set via CLI.
    // Key = config field name, value = true if user-provided.
    std::map<std::string, bool> pinned_params;

    bool isPinned(const std::string& name) const {
        auto it = pinned_params.find(name);
        return it != pinned_params.end() && it->second;
    }

    // Linear Algebra
    uint32_t bw_m = 256; // vector block width m
    uint32_t bw_n = 256; // vector block width n

    // Component Configs
    postprocessing::PostProcConfig pp_config;
    lp::LargePrimeConfig lp_config;
    lingen::BWSolverConfig bw_config;

    // Param test/selection (from batch branch)
    uint32_t params[8] = {};
    bool useParams = false;

    // Autotune
    bool autotune_enabled = false;
    mpqs::autotune::AutotuneConfig autotune_config;
    std::vector<uint32_t> autotune_stages_run;   // Populated after autotune
    double autotune_confidence = 0.0;            // Populated after autotune

#ifdef SIEVING_DEBUG_FLAG
    uint32_t meta_snapshot_step = 0;
    uint32_t sas_snapshot_step = 0;
    uint32_t meta_P = 0;
    uint32_t meta_O = 0;
    bool meta_snapshot_enabled = false;
    bool sas_snapshot_enabled = false;
    bool meta_P_enabled = false;
    bool meta_O_enabled = false;
#endif

    // Probe mode (suppresses constructor/destructor log noise)
    bool silent = false;

    // Sqrt
    bool sqrt_legacy = false;  ///< If true, use CPU Perform() loop (debug/benchmark). Default: GPU batched.
    bool sqrt_diagnostic = false;  ///< If true, log additional sqrt diagnostics (solution diversity, d_valid counts).

    // --- Cluster Mode ---
    ClusterMode cluster_mode = ClusterMode::SOLO;
    cluster::DataTap* data_tap = nullptr;  ///< Set before SieveStage(); nullptr = solo mode

    // Coordinator-specific
    uint16_t listen_port = 9100;              ///< TCP port to listen on
    uint32_t expected_workers = 0;            ///< 0 = M1 (no remote workers)

    // Worker-specific
    std::string coordinator_host;             ///< Hostname/IP of coordinator
    uint16_t coordinator_port = 9100;         ///< Coordinator TCP port

    // Shared cluster config
    std::string transport = "tcp";            ///< "tcp" (default). Future: "mpi", "gpi2"
    uint32_t cluster_init_timeout = 300;      ///< Seconds: worker retry window + coordinator accept timeout
    std::string cluster_node_weights;         ///< Comma-separated per-node weights (overrides SM×clock)
    double cluster_headroom = 10.0;           ///< Per-node headroom percent (0 = exact, default 10)

    // Worker chunk tracking (set from WORK_ASSIGN / CHUNK_ASSIGN)
    uint64_t poly_range_start = 0;   ///< First a-index for this node (0 = natural start)
    uint64_t poly_range_count = 0;   ///< Number of a-values in this node's contiguous range
    mpqs::sieve::AFactorsSnapshot received_snapshot;  ///< From WORK_ASSIGN (workers only)

    // Sieving config is handled dynamically by DeviceSievingController based on N/M,
    // but we can pass overrides here if extended.
};

/**
 * @brief High-Performance MPQS Orchestrator.
 * 
 * Coordinates the integer factorization pipeline:
 * 1. Tuning: Selection of Factor Base and Sieve Interval.
 * 2. Sieving: GPU generation of smooth relations.
 * 3. Matrix: Construction of sparse system over GF(2).
 * 4. LinAlg: Block Wiedemann solver.
 * 5. Sqrt: Derivation of factors from kernel vectors.
 */
class MPQSOrchestrator {
public:
    explicit MPQSOrchestrator(const MPQSConfig& config);
    ~MPQSOrchestrator();

    void Run();

    /**
     * @brief Returns the non-trivial factors found.
     */
    std::vector<mpqs::uint512> GetFactors() const;

    // --- Pipeline Stage (public for probe orchestrator pattern) ---
    void TuningStage();

    // --- Sieve Progress Tracking (public for TruncatedSieveResult) ---

    /// Option F ETA: linear fit on instantaneous rates, integrate to quadratic projection.
    /// Rate model r(t) = r₀ + α·t fitted via OLS on consecutive-sample rates.
    /// Integration constant C fitted via least-squares: C = mean(y_i - r₀·t_i - (α/2)·t_i²).
    /// ETA = positive root of (α/2)·ΔT² + (r₀+α·t_now)·ΔT + (y_model_now - target) = 0.
    struct SieveProgressTracker {
        struct Sample { double t; double y; };
        std::deque<Sample> history;
        static constexpr size_t WINDOW_K = 24;
        double current_eta_sec = 0.0;
        uint64_t last_logged_count = 0;

        // Rate model state
        double smoothed_rate = 0.0;          ///< EMA-smoothed instantaneous rate (rel/s)
        double rate_acceleration = 0.0;      ///< Fitted α from rate OLS (rel/s²), constrained ≥ 0
        double integration_constant = 0.0;   ///< Fitted C from least-squares integration

        // EMA output smoothing
        double ema_eta = 0.0;
        static constexpr double RATE_EMA_ALPHA = 0.3;
        static constexpr double ETA_EMA_ALPHA = 0.4;

        /// Record a (time, relation_count) sample and recompute ETA.
        void addSample(double elapsed_sec, uint64_t relation_count, uint64_t target);

        /// True once enough data points exist for ETA display.
        bool hasETA() const { return history.size() >= 3; }
    };

    /**
     * @brief Linear projection of LP witness hash table fill at estimated run completion.
     *
     * Collects (time, witness_count) samples and fits a linear model w(t) = alpha*t + beta
     * using least-squares over a sliding window. Uses the current ETA from SieveProgressTracker
     * to extrapolate the witness count at run end.
     *
     * Linear model is appropriate because witness accumulation is roughly constant-rate:
     * unlike total relations (which accelerate as LP matches contribute), new witnesses
     * represent unique large primes encountered for the first time.
     */
    class LPFillProjector {
    public:
        struct Projection {
            double   projected_fill_pct;     ///< Projected fill % at estimated run end
            uint64_t projected_witnesses;    ///< Projected witness count at estimated run end
            uint64_t recommended_capacity;   ///< Next-power-of-2 of (projected * 1.25)
            bool     overflow_likely;        ///< projected_fill_pct > 95%
            bool     oversized;              ///< projected_fill_pct < 20%
        };

        /**
         * @brief Record a (time, witness_count) sample.
         * @param elapsed_sec  Seconds since sieve loop start.
         * @param witness_count  Current total unique witnesses.
         * @param capacity  Witness hash table capacity (used for fill%).
         */
        void addSample(double elapsed_sec, uint64_t witness_count, uint64_t capacity);

        /**
         * @brief Extrapolate witness count at total_estimated_sec.
         * @param total_estimated_sec  Total projected run time (elapsed + ETA).
         * @return Projection struct. If fewer than 2 samples, returns zero-initialized with
         *         overflow_likely=false, oversized=false.
         */
        Projection project(double total_estimated_sec) const;

        /// True once enough data points exist for a meaningful projection.
        bool ready() const { return history_.size() >= 3; }

    private:
        struct Sample { double t; uint64_t w; };
        std::deque<Sample> history_;
        uint64_t last_capacity_ = 0;
        static constexpr size_t WINDOW = 8;  ///< Sliding window size for LP fill projection
    };

    /// Result of a truncated sieve probe run.
    /// Contains all telemetry needed for runtime estimation.
    struct TruncatedSieveResult {
        SieveProgressTracker progress_tracker;
        postprocessing::BufferFillHistory buffer_fill;
        postprocessing::LPFillHistory lp_fill;
        LPFillProjector lp_projector;
        double elapsed_sec = 0.0;
        uint64_t relations_found = 0;
        uint64_t target_relations = 0;
        uint64_t steps_executed = 0;
        bool eta_reliable = false;
        bool converged_early = false;
    };

    /// Run a truncated legacy sieve to `frac` of target_relations.
    /// Requires TuningStage() to have been called first.
    /// Does NOT perform dedup, download, or disk I/O.
    /// @param frac  Hard ceiling: stop at frac * target_relations
    /// @param eta_convergence_threshold  Relative spread for early exit (default 5%)
    /// @param min_eta_samples  Minimum ETA samples before convergence can trigger
    /// @param min_progress_frac  Minimum progress before convergence can trigger
    TruncatedSieveResult TruncatedSieveRun(
        double frac = 0.12,
        double eta_convergence_threshold = 0.05,
        uint32_t min_eta_samples = 3,
        double min_progress_frac = 0.03
    );

    /// Factor base size after TuningStage(). Used by runtime estimator for matrix/linalg cost models.
    uint32_t getFactorBaseSize() const { return f_data_.size; }

    /// Full factoringData ref after TuningStage(). Used by preflight checks.
    const mpqs::sieve::factoringData& getFactoringData() const { return f_data_; }

private:
    // --- Pipeline Stages ---
    void SieveStage();
    void MatrixStage();
    void LinearAlgebraStage();
    void SquareRootStage();
    bool shouldAutoApply() const;

    // --- Sieve Init Helpers (shared by SieveStage and TruncatedSieveRun) ---
    /// Configure and return a PostProcConfig from current config_ and f_data_.
    /// @param accum_override  If > 0, overrides the default accumulation buffer size.
    postprocessing::PostProcConfig initPostProcessorConfig(uint64_t accum_override = 0);
    /// Configure and return a LargePrimeConfig + init largeprime_ from current config_.
    void initLargePrimes();
    /// Emit fire-once near-full and overflow warnings for all buffers.
    void logBufferWarnings();

    /// Thread A main loop (coordinator only). CPU-only — no CUDA context.
    void networkLoop();
    /// Helper: buffer raw partials for expanded-matrix path. Thread A only.
    void bufferClusterPartials(const structures::HostRelationBatch& partials);

    /**
     * @brief Comprehensive telemetry summary for the sieve stage.
     * Populated at end of sieve loop, displayed via logSieveStageSummary().
     */
    struct SieveStageSummary {
        // Timing
        double total_elapsed_ms  = 0.0;
        uint64_t total_batches   = 0;

        // Relations
        uint64_t sieved_full_relations   = 0;  ///< Direct smooth relations from trial division
        uint64_t lp_combined_relations   = 0;  ///< Full relations from LP matching
        uint64_t total_full_relations    = 0;  ///< After dedup (final count in persistent batch)
        uint64_t pre_dedup_relations     = 0;  ///< Before dedup
        uint64_t target_relations        = 0;

        // 1-Partials (LP) — zero if LP inactive
        uint64_t total_1partials_found   = 0;  ///< Total 1-partials encountered
        uint64_t unique_witnesses_stored = 0;  ///< Unique LPs in hash table at end
        uint64_t lp_witness_capacity     = 0;  ///< Max witness capacity
        double   witness_fill_pct        = 0.0;///< = stored / capacity x 100

        // Buffer peaks (from BufferFillHistory max fields)
        uint64_t accum_peak       = 0;
        uint64_t accum_capacity   = 0;
        uint64_t partial_peak     = 0;
        uint64_t partial_capacity = 0;
        uint64_t persistent_peak  = 0;
        uint64_t persistent_capacity = 0;

        // LP overflow counts (from SLPPinnedStats extended fields)
        uint64_t slab_overflows    = 0;
        uint64_t witness_overflows = 0;
        uint64_t output_overflows  = 0;

        // Throughput
        double relations_per_sec     = 0.0;
        double lp_witnesses_per_sec  = 0.0;
        double lp_matches_per_sec    = 0.0;

        // LP active flag
        bool lp_active = false;

        // Projection (Feature 5) — populated only if LP active and projector has data
        bool     has_projection          = false;
        double   projected_fill_pct      = 0.0;
        uint64_t projected_witnesses     = 0;
        uint64_t recommended_capacity    = 0;
    };

    /// Unified progress log line, called by both batch and legacy loops.
    /// @param tracker      Rate-model ETA state (mutated: sample added, ETA recomputed)
    /// @param rel_count    Current total relation count (full + LP)
    /// @param target       Target relation count
    /// @param elapsed_sec  Wall-clock seconds since sieve loop start
    /// @param lp_active    Whether LP variant is enabled
    /// @param witnesses    Total LP witnesses (ignored if !lp_active)
    /// @param lp_full_rels Cumulative LP full relations (ignored if !lp_active)
    void logSieveProgress(SieveProgressTracker& tracker,
                          uint64_t rel_count, uint64_t target,
                          double elapsed_sec,
                          bool lp_active, uint64_t witnesses,
                          uint64_t lp_full_rels);

    /// Prints comprehensive sieve stage telemetry at LOG_INFO.
    void logSieveStageSummary(const SieveStageSummary& s);

    // --- State ---
    MPQSConfig config_;
    mpqs::sieve::factoringData f_data_;
    std::vector<mpqs::uint512> result_factors_;
    bool is_jetson_ = false;  ///< SM 8.7 or unified-memory < 12 GB detected at runtime
    cluster::DataTap* data_tap_ = nullptr;  ///< Copied from config_ at SieveStage() entry

    // --- Cluster components (coordinator only, null in solo/worker) ---
    std::unique_ptr<cluster::AccumulatorQueue> cluster_queue_;
    std::unique_ptr<cluster::RelationAccumulator> cluster_accumulator_;
    std::unique_ptr<cluster::FinalBatchHandoff> cluster_handoff_;
    std::unique_ptr<cluster::CPULargePrimeTable> cluster_cpu_lp_;
    std::unique_ptr<cluster::DirectChannel> cluster_channel_;
    std::thread cluster_thread_a_;
    std::unique_ptr<cluster::CommBackend> comm_backend_;  ///< TCP/MPI backend (coordinator + worker)
    std::unique_ptr<cluster::WorkPool> cluster_work_pool_;
    std::unique_ptr<cluster::ChunkScheduler> cluster_scheduler_;
    uint64_t cluster_prev_pers_count_ = 0;  ///< Persistent-batch extraction watermark (survives SieveStage re-entry)
    std::atomic<bool> external_stop_flag_{false};  ///< Set by Thread A; atomic for ARM/Jetson correctness.
    mpqs::sieve::AFactorsSnapshot a_factors_snapshot_;  ///< Saved after init_a_factors for cluster serialization
    bool sieve_truncated_ = false;  ///< Set when sieve exits early via truncation limits

    // Cluster sieve telemetry — populated by Thread A at networkLoop() exit.
    // Read by main thread after join() for logging or diagnostics.
    double   cluster_sieve_elapsed_sec_ = 0.0;
    double   cluster_final_throughput_  = 0.0;
    uint64_t cluster_final_total_rels_  = 0;

    // Post-sieve telemetry (set at end of SieveStage, read in Run())
    uint64_t summary_total_relations_    = 0;  ///< Post-dedup relation count
    uint64_t summary_lp_combined_        = 0;  ///< LP-combined relations (0 if LP inactive)
    bool     summary_lp_active_          = false;

    // Post-matrix telemetry (set at end of MatrixStage, read in Run())
    uint64_t summary_matrix_rows_        = 0;
    uint64_t summary_matrix_cols_        = 0;
    uint64_t summary_matrix_nnz_         = 0;

    // --- Components ---
    std::unique_ptr<mpqs::sieve::DeviceSievingController> siever_;
    std::unique_ptr<mpqs::postprocessing::DevicePostProcessingController> postprocessor_;
    std::unique_ptr<mpqs::lp::LargePrimeVariant> largeprime_;
    std::unique_ptr<mpqs::matrix::MatrixConstructor> matrix_constructor_;
    std::unique_ptr<lingen::BlockWiedemannSolver> linalg_solver_;
    std::unique_ptr<mpqs::sqrt::SquareRootRefinement> sqrt_solver_;

    // --- Buffer telemetry state (Features 3-4) ---
    // Near-full warning flags (fire once)
    bool warned_accum_near_full_      = false;
    bool warned_partial_near_full_    = false;
    bool warned_persistent_near_full_ = false;
    bool warned_witness_near_full_    = false;

    // Overflow delta tracking
    uint64_t last_reported_slab_overflows_    = 0;
    uint64_t last_reported_witness_overflows_ = 0;
    uint64_t last_reported_output_overflows_  = 0;

    // Overflow "warn once" flags — first occurrence is WARNING, subsequent are DEBUG_1
    bool warned_slab_overflow_    = false;
    bool warned_witness_overflow_ = false;
    bool warned_output_overflow_  = false;

    // Contention detection (batch mode)
    uint32_t accum_contention_streak_ = 0;
    bool warned_accum_contention_     = false;

    // Last snapshot (updated by M2 polling)
    postprocessing::BufferFillSnapshot last_buffer_snapshot_{};

    // History trackers (populated by M2/M3)
    postprocessing::BufferFillHistory buffer_fill_history_;
    postprocessing::LPFillHistory     lp_fill_history_;

    // Batch counter for summary
    uint64_t total_batches_processed_ = 0;

    // --- Fill projection & adaptive interval (Feature 5) ---
    LPFillProjector lp_projector_;

    // Adaptive LP interval
    uint32_t adaptive_lp_batch_interval_ = 10;  ///< Initial value matches hardcoded default
    bool     interval_calibrated_        = false;
    bool     interval_recalibrated_      = false;
    static constexpr uint32_t N_LP_INTERVALS = 32;  ///< Target number of LP processing points

    // --- Intermediate Data ---
    // Host-side SoA Container for downloaded relations
    mpqs::structures::HostRelationBatch host_relations_soa_;
    // Raw partials (LP > 1), for expanded matrix construction
    mpqs::structures::HostRelationBatch host_partials_soa_;
    /// Raw partials accumulated on coordinator Thread A for expanded-matrix path.
    /// Written by Thread A only; read by Thread B after join().
    mpqs::structures::HostRelationBatch cluster_raw_partials_;
    // Raw smooths only (LP <= 1), for expanded matrix path.
    // LP-combined relations (LP > 1) are excluded because the expanded matrix
    // builder cannot assign LP columns for them (they enter via the smooths arg).
    mpqs::structures::HostRelationBatch raw_smooths_soa_;
    // Host matrix containing relations as rows
    HostMatrix matrix_A_;
    // Packed left solution vectors for matrix_A_
    std::vector<std::vector<uint64_t>> kernel_solutions_;

    // --- Preprocessing pipeline state (M5) ---
    /// Merge tree from M3/M4 preprocessing. Used for kernel vector expansion.
    matrix::MergeTree merge_tree_;
    /// Row map: reduced_row → merge tree node index.
    std::vector<uint32_t> preproc_row_map_;
    /// True if the expanded-matrix path was used (vs projected-matrix path).
    bool used_expanded_matrix_ = false;
    /// True if the V2 packed pipeline was used (M9f).
    bool used_packed_pipeline_ = false;
    /// V2 preprocessing result from packed GPU pipeline (M9f).
    /// Contains merged 1-partial data for direct sqrt consumption — no merge tree.
    std::optional<matrix::PreprocessResultV2> preproc_v2_result_;
    /// Precomputed LP Y-contributions per solution (Montgomery domain: value * R mod N).
    /// Computed from raw partials' LP values after kernel vector expansion.
    /// Applied via ApplyLPCorrection (GPU) or mont.mul in Perform (CPU).
    /// MUST NOT be passed through mont.transform() again -- already in Montgomery form.
    std::vector<mpqs::uint512> precomputed_lp_y_;
};

} // namespace mpqs
