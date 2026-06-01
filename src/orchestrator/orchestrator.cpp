// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "orchestrator.h"
#include "mpqs_structures.h"
#include "relation_io.h"
#include "autotune.h"
#include "auto_apply.h"
#include "autotune_history.h"
#include "data_tap.h"
#include "version.h"
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "accumulator.h"
#include "cpu_lp.h"
#include "direct_channel.h"
#include "comm_backend.h"
#include "cluster_common.h"
#include "serialization.h"
#include "async_network_data_tap.h"
#include "work_pool.h"
#include "chunk_scheduler.h"
#include "cost_models.h"
#include "character_columns.h"
#include "gpu_char_cols.cuh"
#include "matrix_truncation.h"
#include "preprocess.h"
#include "gpu_packed_expanded.cuh"     // M9f: gpuBuildPackedMatrix
#include "gpu_product_char_packed.cuh" // M9f: gpuProductCharCols_packed


/// Parse a comma-separated list of positive node weights (e.g. "175840,74880").
/// Values ≤ 0 are clamped to 0.01 to prevent division by zero.
static std::vector<double> parseNodeWeights(const std::string& s) {
    std::vector<double> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            double v = std::stod(token);
            result.push_back(v > 0.0 ? v : 0.01);
        } catch (...) {
            result.push_back(0.01);
        }
    }
    return result;
}

// Serialization moved to src/common/relation_io.{h,cpp}

namespace mpqs {

// -----------------------------------------------------------------------------
// Constructor & Destructor
// -----------------------------------------------------------------------------

MPQSOrchestrator::MPQSOrchestrator(const MPQSConfig& config)
    : config_(config)
{
    // Initialize Logger
    HPCLogger::Get().SetRank(config_.node_id);
    
    CUDA_CHECK_FATAL(cudaSetDevice(config_.device_id));

    if (!config_.silent) {
        LOG_SET_MODULE("Orchestrator");
        LOG_SET_STAGE(LOG_STAGE_ORCHESTRATOR_INITIALIZATION, "Init");
        LOG(LOG_INFO) << "=== MPQS Orchestrator Initialized ===";
        LOG(LOG_INFO) << "Target N: " << config_.N.to_string();
        LOG(LOG_INFO) << "Device:   GPU " << config_.device_id;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    if (!config_.silent) {
        LOG(LOG_STATS) << "--- Device Hardware Specs ---";
        LOG(LOG_STATS) << " Device Name: " << prop.name;
        LOG(LOG_STATS) << " Compute Capability: " << prop.major << "." << prop.minor;
        LOG(LOG_STATS) << " SMs: " << prop.multiProcessorCount << ", Global Mem: " << prop.totalGlobalMem / (1024 * 1024) << " MB";
    }
    
    // Jetson / integrated-GPU platform detection (done here so TuningStage()
    // has is_jetson_ available even when called directly without Run()).
    // Use prop.integrated (true only for integrated GPUs like Jetson Orin) plus
    // the SM 8.7 guard as a belt-and-suspenders check.  The old VRAM < 12 GB
    // threshold was unreliable: discrete GPUs (e.g. RTX 2080 Super MaxQ, 8 GB)
    // have unified addressing enabled but are NOT integrated.
    is_jetson_ = (prop.integrated != 0) || (prop.major == 8 && prop.minor == 7);
    if (is_jetson_ && !config_.silent) {
        LOG(LOG_STATS) << "Jetson/integrated-GPU platform detected (SM "
                      << prop.major << "." << prop.minor
                      << ", " << (prop.totalGlobalMem >> 20) << " MB)";
    }

    if(config_.disk_io) {
        // Ensure work directory exists
        std::filesystem::create_directories(config_.work_dir);
    }

    // Basic Data Init
    f_data_.N = config_.N;
}

MPQSOrchestrator::~MPQSOrchestrator() {
    // Explicitly clear remaining buffers
    
    if (!config_.silent)
        LOG(LOG_INFO) << "=== MPQS Orchestrator Shutting Down ===";
}

// -----------------------------------------------------------------------------
// Public Interface
// -----------------------------------------------------------------------------

std::vector<mpqs::uint512> MPQSOrchestrator::GetFactors() const {
    return result_factors_;
}

namespace {
    /// Queries cudaMemGetInfo and logs free/total GPU memory.
    void logGpuMemory(const char* label, int urgency = LOG_DEBUG_1) {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            LOG(urgency) << "[Memory] " << label << ": "
                         << (free_bytes >> 20) << " MB free / "
                         << (total_bytes >> 20) << " MB total";
        }
    }

    /// Format a comma-separated thousands number.
    std::string fmtNum(uint64_t v) {
        std::string raw = std::to_string(v);
        std::string result;
        int count = 0;
        for (int i = (int)raw.size() - 1; i >= 0; --i) {
            if (count > 0 && count % 3 == 0) result = "," + result;
            result = raw[i] + result;
            count++;
        }
        return result;
    }

    /// Format a uint512 factor for summary display: truncate to 10...3 if > 20 digits.
    std::string fmtFactor(const mpqs::uint512& f) {
        std::string s = f.to_string();
        if (s.size() > 20)
            return s.substr(0, 10) + "..." + s.substr(s.size() - 3);
        return s;
    }
} // anonymous namespace

void MPQSOrchestrator::Run() {
    // THREAD CONTEXT FIX: Ensure the thread executing Run() is bound to the correct device
    cudaSetDevice(config_.device_id);
    LOG_SET_MODULE("Orchestrator");

    LOG(LOG_INFO) << "cuda-mpqs " << CUDAMPQS_VERSION
                  << " (lingen " << CUDAMPQS_LINALG_VERSION << ")";

#ifdef SIEVING_DEBUG_FLAG
    LOG(LOG_INFO) << "RUNNING IN DEBUG MODE";
#endif

    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    // Per-stage timing (seconds)
    double time_tuning_sec   = 0.0, time_autotune_sec = 0.0, time_sieve_sec  = 0.0;
    double time_matrix_sec   = 0.0, time_linalg_sec   = 0.0, time_sqrt_sec   = 0.0;

    try {
        // 1. Tuning (Parameter Selection)
        // We need Tuning even for LINALG_ONLY to have the Factor Base for matrix construction
        if (config_.mode == ExecutionMode::FULL_PIPELINE ||
            config_.mode == ExecutionMode::SIEVE_ONLY ||
            config_.mode == ExecutionMode::LINALG_ONLY ||
            config_.mode == ExecutionMode::PARAM_TEST ||
            config_.mode == ExecutionMode::AUTOTUNE_ONLY)
        {
            auto t_tuning = clock::now();
            TuningStage();

            // --- Auto-Apply: history-based parameter selection (zero GPU probes) ---
            if (shouldAutoApply()) {
                mpqs::autotune::AutoApplyController aac(config_, f_data_);
                auto aa_result = aac.apply();
                bool fb_changed = aa_result.applied && aa_result.fb_bound > 0
                    && aa_result.fb_bound != f_data_.F;
                bool M_changed = aa_result.applied && aa_result.sieve_bound > 0
                    && aa_result.sieve_bound != f_data_.M;

                if (fb_changed) {
                    config_.fb_bound = aa_result.fb_bound;
                    f_data_.F = aa_result.fb_bound;
                    generateFactorBase(&f_data_);
                }
                if (M_changed) {
                    f_data_.M = aa_result.sieve_bound;
                    // Recompute a_target = sqrt(2N) / M
                    mpqs::uint512 twoN = config_.N;
                    twoN.lshift(1);
                    mpqs::uint512 sqrt2N = twoN.sqrt();
                    sqrt2N.div_uint32_inplace(f_data_.M);
                    f_data_.a_target = sqrt2N;
                }
                if (fb_changed || M_changed) {
                    init_a_factors(&f_data_);
                    config_.target_relations = f_data_.size + (f_data_.size / 20) + 64;
                }
            }
            time_tuning_sec = std::chrono::duration<double>(clock::now() - t_tuning).count();

            // --- Autotune ---
            if (config_.mode == ExecutionMode::AUTOTUNE_ONLY ||
                (config_.mode == ExecutionMode::FULL_PIPELINE && config_.autotune_enabled))
            {
                auto t_autotune = clock::now();
                mpqs::autotune::AutotuneController atc(config_.autotune_config, config_, f_data_);
                atc.run();
                config_.autotune_stages_run = atc.stagesRun();
                config_.autotune_confidence = atc.finalConfidence();
                time_autotune_sec = std::chrono::duration<double>(clock::now() - t_autotune).count();
                if (config_.mode == ExecutionMode::AUTOTUNE_ONLY) {
                    LOG(LOG_INFO) << "AUTOTUNE_ONLY complete.";
                    return;
                }
            }

            // =================================================================
            // WORKER PATH: connect, receive params, sieve, flush, exit
            // =================================================================
            if (config_.cluster_mode == ClusterMode::WORKER) {
                LOG(LOG_INFO) << "Worker: connecting to coordinator "
                              << config_.coordinator_host << ":" << config_.coordinator_port;

                // 1. Create and initialize CommBackend (worker mode)
                comm_backend_ = cluster::createCommBackend(
                    config_.transport,
                    /*is_coordinator=*/false,
                    config_.coordinator_host,
                    config_.coordinator_port,
                    /*expected_workers=*/0,
                    config_.cluster_init_timeout * 1000);

                cluster::HelloPayload hello{};
                {
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, config_.device_id);
                    std::strncpy(hello.gpu_name, prop.name, sizeof(hello.gpu_name) - 1);
                    hello.sm_version = static_cast<uint16_t>(prop.major * 10 + prop.minor);
                    hello.num_sms = static_cast<uint16_t>(prop.multiProcessorCount);
                    hello.vram_mb = static_cast<uint32_t>(prop.totalGlobalMem / (1024 * 1024));
                    hello.capacity_estimate = 1.0f;
                    int clock_khz = 0;
                    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, config_.device_id);
                    hello.clock_mhz = static_cast<uint16_t>(clock_khz / 1000);
                }
                comm_backend_->setLocalInfo(hello);

                if (!comm_backend_->initialize()) {
                    LOG(LOG_ERROR_CRITICAL) << "Worker: failed to connect to coordinator";
                    return;
                }
                LOG(LOG_INFO) << "Worker: connected, assigned ID " << (int)comm_backend_->selfId();

                // 2. Receive WORK_ASSIGN (blocking, 60s timeout)
                cluster::RecvMessage wa_msg;
                if (!comm_backend_->recvBlocking(wa_msg, 60000) ||
                    wa_msg.type != cluster::MsgType::WORK_ASSIGN) {
                    LOG(LOG_ERROR_CRITICAL) << "Worker: failed to receive WORK_ASSIGN";
                    comm_backend_->finalize();
                    return;
                }

                // 3. Deserialize WORK_ASSIGN -> populate f_data_ + snapshot
                uint32_t recv_batch_size;
                uint64_t recv_threshold, recv_lp1_bound;
                uint64_t recv_poly_start, recv_poly_count, recv_target;
                mpqs::sieve::AFactorsSnapshot recv_snapshot;

                if (!cluster::deserializeWorkAssign(
                        wa_msg.payload.data(), wa_msg.payload.size(),
                        f_data_, recv_batch_size, recv_threshold,
                        recv_lp1_bound, recv_poly_start, recv_poly_count,
                        recv_target, &recv_snapshot)) {
                    LOG(LOG_ERROR_CRITICAL) << "Worker: WORK_ASSIGN deserialization failed";
                    comm_backend_->finalize();
                    return;
                }

                // Save coordinator's authoritative values
                uint32_t coord_M    = f_data_.M;
                uint32_t coord_F    = f_data_.F;
                uint32_t coord_size = f_data_.size;

                // 4. Derive remaining f_data_ fields
                mpqs::sieve::determineParams(&f_data_);

                // Restore coordinator's tuned values
                f_data_.M       = coord_M;
                f_data_.F       = coord_F;
                f_data_.size    = coord_size;

                // 5. Initialize a-factor polynomial generation
                mpqs::sieve::init_a_factors(&f_data_);

                LOG(LOG_INFO) << "Worker: N=" << f_data_.N.to_string().size() << "d, "
                              << "FB=" << f_data_.size << ", M=" << f_data_.M
                              << ", shc_dim=" << (int)f_data_.a_factors.size()
                              << ", batch_size=" << recv_batch_size
                              << ", target=" << recv_target
                              << ", poly_range=[" << recv_poly_start
                              << ", " << recv_poly_start + recv_poly_count << ")";

                // 6. Apply received parameters to config
                config_.target_relations = recv_target;
                config_.lp1_bound = recv_lp1_bound;
                if (recv_batch_size > 0 && !config_.isPinned("sieve_batch_size")) {
                    config_.sieve_batch_size = recv_batch_size;
                }
                // NOTE: useParams is intentionally NOT cleared here.
                // Jetson workers are launched with explicit --fb_bound/--sieve_bound matching
                // the coordinator's F/M (see jetson_cluster_launch.sh WORKER_SIEVE), so autotune
                // params are valid for the main sieve geometry. Clearing useParams would force
                // loadStandardConfig() — a different kernel config that requires cold JIT
                // compilation (~22 min on Jetson Orin Nano, vs <10s with warm autotune cache).

                // M3: Store poly range and snapshot in config for SieveStage
                config_.poly_range_start = recv_poly_start;
                config_.poly_range_count = recv_poly_count;
                config_.received_snapshot = recv_snapshot;

                uint32_t current_chunk_id = 0;

                // 7. Create AsyncNetworkDataTap and inject into config
                cluster::AsyncNetworkDataTap::Config tap_config;
                tap_config.coalesce_count = is_jetson_ ? 4 : 8;
                cluster::AsyncNetworkDataTap network_tap(*comm_backend_, tap_config);
                // Pre-allocate ring slots. pp_config is finalized inside SieveStage so we
                // use a conservative per-extraction bound here (4096 relations × 16 factors).
                network_tap.preallocateSlots(4096, 4096 * 16);
                config_.data_tap = &network_tap;

                // ========================================
                // M3: Worker re-sieve loop with chunk transitions
                // ========================================
                LOG(LOG_INFO) << "Worker: starting sieve loop (M3 chunk mode)";
                auto t_sieve_total = clock::now();
                uint32_t chunks_completed = 0;
                bool worker_done = false;
                // M4-S6: One-shot warmup before first SieveStage — triggers PTX→SASS JIT so
                // that graph capture on the actual sieve finds cached SASS. On Jetson cold
                // cache this takes ~20 min once; all subsequent launches reuse the cache.
                // data_tap is set to nullptr to discard warmup relations (not the assigned range).
                bool warmup_done = false;
                while (!worker_done) {
                    auto t_chunk = clock::now();

                    if (!warmup_done && config_.cuda_graph_unroll > 0) {
                        LOG(LOG_INFO) << "Worker: warmup batch (JIT compile, no graph capture)";
                        const uint32_t saved_unroll      = config_.cuda_graph_unroll;
                        const uint32_t saved_max_batches = config_.sieve_max_batches;
                        cluster::DataTap* saved_tap      = config_.data_tap;
                        config_.cuda_graph_unroll  = 0;
                        config_.sieve_max_batches  = 1;
                        config_.data_tap           = nullptr;
                        SieveStage();
                        config_.cuda_graph_unroll  = saved_unroll;
                        config_.sieve_max_batches  = saved_max_batches;
                        config_.data_tap           = saved_tap;
                        LOG(LOG_INFO) << "Worker: warmup done, proceeding with cuda_graph_unroll="
                                      << saved_unroll;
                    }
                    warmup_done = true;

                    // Reset extraction watermark for fresh postprocessor each chunk
                    cluster_prev_pers_count_ = 0;

                    // Set range on AsyncNetworkDataTap BEFORE SieveStage
                    network_tap.setRange(config_.poly_range_count, current_chunk_id,
                                         config_.sieve_batch_size > 0 ? config_.sieve_batch_size : 1);

                    // Run SieveStage — siever handles resetAndAdvanceTo internally
                    SieveStage();

                    double chunk_elapsed_sec = std::chrono::duration<double>(
                        clock::now() - t_chunk).count();

                    // Check if we received STOP during sieving
                    if (network_tap.receivedStop()) {
                        LOG(LOG_INFO) << "Worker: received STOP during sieve";
                        worker_done = true;
                        break;
                    }

                    // S8: Check if this was a CHUNK_RECALL (partial) vs STOP/range-exhaustion
                    bool was_recalled = network_tap.receivedRecall();
                    if (was_recalled) {
                        LOG(LOG_INFO) << "Worker: chunk " << network_tap.currentChunkId()
                                      << " recalled after " << network_tap.aValsConsumed() << " a-values";
                        network_tap.clearRecall();
                    }

                    // Range exhausted (or recalled) — send CHUNK_COMPLETE
                    cluster::ChunkCompletePayload cc{};
                    cc.chunk_id = current_chunk_id;
                    cc.a_values_consumed = network_tap.aValsConsumed();
                    cc.relations_found = network_tap.batchesSent();  // Approximation
                    cc.partials_found = 0;  // Not tracked separately in AsyncNetworkDataTap
                    cc.elapsed_ms = static_cast<uint32_t>(chunk_elapsed_sec * 1000.0);
                    comm_backend_->send(0, cluster::MsgType::CHUNK_COMPLETE,
                                        &cc, sizeof(cc));
                    ++chunks_completed;
                    LOG(LOG_INFO) << "Worker: CHUNK_COMPLETE id=" << current_chunk_id
                                  << " elapsed=" << std::fixed << std::setprecision(2)
                                  << chunk_elapsed_sec << "s";

                    // Wait for CHUNK_ASSIGN or STOP (blocking, 30s timeout)
                    cluster::RecvMessage resp;
                    if (!comm_backend_->recvBlocking(resp, 30000)) {
                        LOG(LOG_WARNING) << "Worker: timeout waiting for CHUNK_ASSIGN/STOP";
                        worker_done = true;
                        break;
                    }

                    if (resp.type == cluster::MsgType::STOP) {
                        LOG(LOG_INFO) << "Worker: received STOP while waiting for chunk";
                        worker_done = true;
                        break;
                    }

                    if (resp.type == cluster::MsgType::CHUNK_ASSIGN) {
                        if (resp.payload.size() >= sizeof(cluster::ChunkAssignPayload)) {
                            cluster::ChunkAssignPayload ca;
                            std::memcpy(&ca, resp.payload.data(), sizeof(ca));

                            // Update config for next SieveStage re-entry
                            config_.poly_range_start = ca.poly_range_start;
                            config_.poly_range_count = ca.poly_range_count;
                            current_chunk_id = ca.chunk_id;

                            LOG(LOG_INFO) << "Worker: received overflow chunk "
                                          << ca.chunk_id << " ["
                                          << ca.poly_range_start << ", "
                                          << ca.poly_range_start + ca.poly_range_count
                                          << ")";
                            // Loop back to SieveStage with new range
                        } else {
                            LOG(LOG_WARNING) << "Worker: malformed CHUNK_ASSIGN payload";
                            worker_done = true;
                            break;
                        }
                    } else {
                        LOG(LOG_WARNING) << "Worker: unexpected message type "
                                         << static_cast<int>(
                                                static_cast<uint8_t>(resp.type))
                                         << " while waiting for CHUNK_ASSIGN";
                        worker_done = true;
                        break;
                    }
                }

                double sieve_total_sec = std::chrono::duration<double>(
                    clock::now() - t_sieve_total).count();
                LOG(LOG_INFO) << "Worker: sieve complete in " << std::fixed
                              << std::setprecision(2) << sieve_total_sec << "s, "
                              << chunks_completed << " chunks, "
                              << network_tap.batchesSent() << " batches sent";
                if (network_tap.droppedBatches() > 0) {
                    LOG(LOG_WARNING) << "AsyncDataTap: " << network_tap.droppedBatches()
                                     << " batches dropped (ring overflow)";
                }

                // Join I/O thread BEFORE any post-loop comm_backend_ calls.
                // Prevents: (a) use-after-free when comm_backend_ is reset below,
                // (b) concurrent send race between I/O thread heartbeat/sends and
                // main thread FLUSH_ACK.
                network_tap.shutdown();

                // 9. Send FLUSH_ACK to coordinator
                comm_backend_->send(0, cluster::MsgType::FLUSH_ACK, nullptr, 0);

                // 10. Cleanup
                config_.data_tap = nullptr;
                comm_backend_->finalize();
                comm_backend_.reset();

                LOG(LOG_INFO) << "Worker: done, exiting";
                return;  // Worker does NOT continue to Matrix/LinAlg/Sqrt
            }

            if (config_.mode != ExecutionMode::LINALG_ONLY) {

                // M24: shared post-sieve relation persistence used by both coordinator
                // and solo paths. Returns true iff the caller should immediately return
                // from Run() (SIEVE_ONLY mode). The exit_tag is appended to the
                // "SIEVE_ONLY mode complete" log line to distinguish coordinator/solo.
                auto saveRelationsIfRequested = [&](const char* exit_tag) -> bool {
                    if (!(config_.disk_io || config_.mode == ExecutionMode::SIEVE_ONLY))
                        return false;
                    std::string rel_path = config_.work_dir + "/relations.soa";
                    LOG(LOG_INFO) << "Writing SoA relations to " << rel_path;
                    std::filesystem::create_directories(config_.work_dir);
                    if (mpqs::io::serialize_v1(rel_path, host_relations_soa_)) {
                        LOG(LOG_INFO) << "Wrote " << host_relations_soa_.num_relations
                                      << " relations (" << host_relations_soa_.num_factors
                                      << " factors) to disk.";
                    } else {
                        LOG(LOG_ERROR_CRITICAL) << "Failed to write SoA relations to " << rel_path;
                    }
                    // Also write v2 format with partials (if available)
                    if (host_partials_soa_.num_relations > 0) {
                        std::string v2_path = config_.work_dir + "/relations.v2";
                        mpqs::io::V2Metadata meta;
                        meta.N = config_.N;
                        meta.factor_base = f_data_.factorBase;
                        meta.lp_bound = config_.lp1_bound;
                        meta.sieve_bound = config_.sieve_bound;
                        if (mpqs::io::serialize_v2(v2_path, host_relations_soa_,
                                                   host_partials_soa_, meta)) {
                            LOG(LOG_INFO) << "Wrote v2 relations: "
                                          << host_relations_soa_.num_relations
                                          << " smooths + "
                                          << host_partials_soa_.num_relations
                                          << " partials.";
                        } else {
                            LOG(LOG_ERROR_CRITICAL) << "Failed to write v2 relations to " << v2_path;
                        }
                    }
                    if (config_.mode == ExecutionMode::SIEVE_ONLY) {
                        LOG(LOG_INFO) << "SIEVE_ONLY mode complete" << exit_tag << ". Exiting.";
                        return true;
                    }
                    return false;
                };

                if (config_.cluster_mode == ClusterMode::COORDINATOR) {
                    // =============================================================
                    // COORDINATOR M3 PATH: dual sieve + work distribution
                    // =============================================================
                    external_stop_flag_ = false;

                    // --- Save AFactorsSnapshot for workers ---
                    // TuningStage() already called init_a_factors(), so f_data_
                    // has the canonical a-factor state. Capture directly — no
                    // need for a temporary siever (saveSnapshot just copies fields).
                    a_factors_snapshot_.a_factors = f_data_.a_factors;
                    a_factors_snapshot_.lowerHalfStart = f_data_.lowerHalfStart;
                    a_factors_snapshot_.upperHalfStart = f_data_.upperHalfStart;
                    a_factors_snapshot_.shc_dim = static_cast<uint32_t>(
                        f_data_.a_factors.size());

                    const uint32_t H = 1u << static_cast<uint32_t>(
                        f_data_.a_factors.size());
                    const uint32_t B = std::max(1u, config_.sieve_batch_size);
                    const uint32_t G = std::max(1u, config_.cuda_graph_unroll);
                    const uint32_t Q = (config_.sieve_batch_size > 0)
                        ? B * ((config_.cuda_graph_unroll > 0) ? G : 1)
                        : 1;

                    // --- Network setup (M2+: accept remote workers) ---
                    bool has_remote_workers = (config_.expected_workers > 0);
                    if (has_remote_workers) {
                        LOG(LOG_INFO) << "Coordinator: initializing TCP backend, expecting "
                                      << config_.expected_workers << " worker(s) on port "
                                      << config_.listen_port;

                        comm_backend_ = cluster::createCommBackend(
                            config_.transport,
                            /*is_coordinator=*/true,
                            /*host=*/"",
                            config_.listen_port,
                            config_.expected_workers,
                            config_.cluster_init_timeout * 1000);

                        if (!comm_backend_->initialize()) {
                            throw std::runtime_error("Coordinator: CommBackend init failed");
                        }
                        LOG(LOG_INFO) << "Coordinator: " << comm_backend_->peerCount()
                                      << " worker(s) connected";

                        // --- SM*clock weights for proportional range assignment ---
                        const uint32_t num_nodes = 1 + comm_backend_->peerCount();
                        std::vector<double> node_weights;
                        std::vector<uint32_t> sm_counts;
                        bool using_manual_weights = false;

                        if (!config_.cluster_node_weights.empty()) {
                            node_weights = parseNodeWeights(config_.cluster_node_weights);
                            if (node_weights.size() != num_nodes) {
                                LOG(LOG_WARNING) << "Coordinator: --cluster_node_weights count ("
                                    << node_weights.size() << ") != node count (" << num_nodes
                                    << ") - falling back to SM*clock";
                                node_weights.clear();
                            } else {
                                using_manual_weights = true;
                                LOG(LOG_INFO) << "Coordinator: using manual node weights";
                            }
                        }

                        if (node_weights.empty()) {
                            cudaDeviceProp coord_prop;
                            cudaGetDeviceProperties(&coord_prop, config_.device_id);
                            sm_counts.push_back(
                                static_cast<uint32_t>(coord_prop.multiProcessorCount));
                            int coord_clock_khz = 0;
                            cudaDeviceGetAttribute(&coord_clock_khz, cudaDevAttrClockRate,
                                                    config_.device_id);
                            node_weights.push_back(
                                static_cast<double>(coord_prop.multiProcessorCount)
                                * (coord_clock_khz / 1000.0));
                            for (uint32_t w = 1; w <= comm_backend_->peerCount(); ++w) {
                                auto info = comm_backend_->peerInfo(static_cast<uint8_t>(w));
                                sm_counts.push_back(info.num_sms);
                                if (info.clock_mhz == 0)
                                    LOG(LOG_WARNING) << "Coordinator: worker " << w
                                        << " sent clock_mhz=0 - assuming 1000 MHz";
                                node_weights.push_back(
                                    static_cast<double>(info.num_sms)
                                    * static_cast<double>(
                                        info.clock_mhz > 0 ? info.clock_mhz : 1000));
                            }
                        } else {
                            cudaDeviceProp coord_prop;
                            cudaGetDeviceProperties(&coord_prop, config_.device_id);
                            sm_counts.push_back(
                                static_cast<uint32_t>(coord_prop.multiProcessorCount));
                            for (uint32_t w = 1; w <= comm_backend_->peerCount(); ++w) {
                                auto info = comm_backend_->peerInfo(static_cast<uint8_t>(w));
                                sm_counts.push_back(info.num_sms);
                            }
                        }

                        for (size_t i = 0; i < node_weights.size(); ++i) {
                            LOG(LOG_INFO) << "Coordinator: node " << i
                                          << " weight=" << node_weights[i]
                                          << (using_manual_weights ? " (manual)" : " (SM*clock)");
                        }

                        // Compute weight-proportional ranges with headroom
                        uint64_t dummy_start = 0, dummy_size = 1;
                        auto dummy_pool = std::make_unique<cluster::WorkPool>(
                            dummy_start, dummy_size);
                        auto tmp_scheduler = std::make_unique<cluster::ChunkScheduler>(
                            *dummy_pool, num_nodes, /*total_a=*/0, H, Q);
                        auto ranges = tmp_scheduler->computeContiguousRanges(
                            node_weights, config_.target_relations, H, Q,
                            config_.cluster_headroom);
                        tmp_scheduler.reset();
                        dummy_pool.reset();

                        // Create real WorkPool for overflow a-indices
                        uint64_t overflow_start = 0;
                        for (auto& r : ranges) overflow_start += r.count;
                        uint64_t overflow_size = overflow_start;
                        cluster_work_pool_ = std::make_unique<cluster::WorkPool>(
                            overflow_start, overflow_size);
                        cluster_scheduler_ = std::make_unique<cluster::ChunkScheduler>(
                            *cluster_work_pool_, num_nodes,
                            overflow_start + overflow_size, H, Q);

                        // Log ranges
                        for (size_t i = 0; i < ranges.size(); ++i) {
                            LOG(LOG_INFO) << "Coordinator: node " << i
                                          << " range=[" << ranges[i].start << ", "
                                          << ranges[i].start + ranges[i].count
                                          << ") (" << ranges[i].count / H << " windows, "
                                          << (i < sm_counts.size() ? sm_counts[i] : 0u)
                                          << " SMs)";
                        }
                        LOG(LOG_INFO) << "Coordinator: overflow pool starts at "
                                      << overflow_start << " (" << overflow_size / H
                                      << " windows)";

                        // Send WORK_ASSIGN to each remote worker with actual ranges
                        uint64_t threshold_override = 0;
                        if (config_.lp1_bound > 0) {
                            threshold_override = std::min(config_.lp1_bound,
                                static_cast<uint64_t>(config_.fb_bound));
                        }
                        for (uint32_t w = 1; w <= comm_backend_->peerCount(); ++w) {
                            auto& range = ranges[w];
                            auto [buf, len] = cluster::serializeWorkAssign(
                                f_data_, config_.sieve_batch_size, threshold_override,
                                config_.lp1_bound,
                                /*poly_range_start=*/range.start,
                                /*poly_range_count=*/range.count,
                                config_.target_relations,
                                &a_factors_snapshot_);
                            comm_backend_->send(static_cast<uint8_t>(w),
                                cluster::MsgType::WORK_ASSIGN, buf.data(),
                                static_cast<uint32_t>(len));
                            LOG(LOG_INFO) << "Coordinator: WORK_ASSIGN to worker " << w
                                          << " range=[" << range.start << ", "
                                          << range.start + range.count << ")";
                        }
                    }

                    // --- Always create DirectChannel for local sieve (M3) ---
                    cluster_channel_ = std::make_unique<cluster::DirectChannel>();
                    config_.data_tap = cluster_channel_.get();

                    // --- Create cluster infrastructure ---
                    cluster_queue_ = std::make_unique<cluster::AccumulatorQueue>();
                    cluster_accumulator_ = std::make_unique<cluster::RelationAccumulator>(
                        config_.target_relations, /*dedup_margin=*/config_.dedup_safety_factor);
                    cluster_handoff_ = std::make_unique<cluster::FinalBatchHandoff>();

                    if (config_.lp1_bound > 0) {
                        cluster_cpu_lp_ = std::make_unique<cluster::CPULargePrimeTable>(
                            config_.lp1_bound, f_data_);
                    }

                    // --- Launch Thread A ---
                    cluster_thread_a_ = std::thread([this]() { networkLoop(); });

                    auto join_thread_a = [this]() {
                        if (cluster_thread_a_.joinable()) {
                            if (cluster_channel_) cluster_channel_->signalStop();
                            cluster_thread_a_.join();
                        }
                    };

                    // --- Thread B: always run SieveStage (M3 unified) ---
                    try {
                        if (is_jetson_) logGpuMemory("Before sieve");
                        auto t_sieve = clock::now();
                        SieveStage();
                        time_sieve_sec = std::chrono::duration<double>(
                            clock::now() - t_sieve).count();
                        if (is_jetson_) logGpuMemory("After sieve");

                        // Await handoff from Thread A
                        host_relations_soa_ = cluster_handoff_->await();
                        LOG(LOG_INFO) << "Coordinator: received "
                                      << host_relations_soa_.num_relations
                                      << " deduplicated relations from Thread A";
                    } catch (...) {
                        join_thread_a();
                        throw;
                    }

                    // Free sieve GPU resources
                    siever_.reset();
                    postprocessor_.reset();
                    largeprime_.reset();
                    cudaGetLastError();

                    // Join Thread A
                    cluster_thread_a_.join();

                    // M6: Populate raw partials for expanded matrix path
                    if (cluster_raw_partials_.num_relations > 0) {
                        host_partials_soa_ = std::move(cluster_raw_partials_);
                        LOG(LOG_INFO) << "Coordinator: " << host_partials_soa_.num_relations
                                      << " raw partials available for expanded matrix.";
                    }

                    // Finalize network
                    if (comm_backend_) {
                        comm_backend_->finalize();
                        comm_backend_.reset();
                    }

                    // Reset cluster resources
                    cluster_queue_.reset();
                    cluster_cpu_lp_.reset();
                    cluster_channel_.reset();
                    cluster_accumulator_.reset();
                    cluster_handoff_.reset();
                    cluster_work_pool_.reset();
                    cluster_scheduler_.reset();

                    // --- Cluster estimate_only: print estimate and return ---
                    if (config_.estimate_only) {
                        uint32_t fb_size = f_data_.size;
                        uint64_t target  = config_.target_relations > 0
                                         ? config_.target_relations : fb_size + 100;
                        double throughput = cluster_final_throughput_;
                        double sieve_est = (throughput > 0.0)
                                         ? static_cast<double>(target) / throughput : 0.0;
                        double matrix_est = autotune::estimateMatrixTime(fb_size);
                        double linalg_est = autotune::estimateLinalgTime(fb_size);
                        double total_est  = sieve_est + matrix_est + linalg_est;
                        uint32_t num_nodes = 1 + config_.expected_workers;

                        LOG(LOG_INFO) << "[Estimate] --- Cluster Runtime Estimate ---";
                        LOG(LOG_INFO) << "[Estimate] Nodes:          " << num_nodes;
                        LOG(LOG_INFO) << "[Estimate] Parameters: F=" << config_.fb_bound
                                      << " M=" << config_.sieve_bound
                                      << " L=" << config_.lp1_bound;
                        LOG(LOG_INFO) << "[Estimate] FB size:        " << fb_size;
                        LOG(LOG_INFO) << "[Estimate] Sieve time:     "
                                      << std::fixed << std::setprecision(1)
                                      << sieve_est << " s";
                        LOG(LOG_INFO) << "[Estimate] Matrix time:    "
                                      << std::fixed << std::setprecision(1)
                                      << matrix_est << " s";
                        LOG(LOG_INFO) << "[Estimate] LinAlg time:    "
                                      << std::fixed << std::setprecision(1)
                                      << linalg_est << " s";
                        LOG(LOG_INFO) << "[Estimate] Total estimate: "
                                      << std::fixed << std::setprecision(1)
                                      << total_est << " s";
                        LOG(LOG_INFO) << "[Estimate] Throughput:     "
                                      << std::fixed << std::setprecision(1)
                                      << throughput << " rels/s";
                        LOG(LOG_INFO) << "[Estimate] Probe duration: "
                                      << std::fixed << std::setprecision(1)
                                      << cluster_sieve_elapsed_sec_ << " s"
                                      << " (" << cluster_final_total_rels_ << " rels)";
                        return;
                    }

                    // Write relations to disk if requested (coordinator path)
                    if (saveRelationsIfRequested(" (coordinator)")) return;

                    // Falls through to Matrix -> LinAlg -> Sqrt

                } else {
                    // =============================================================
                    // SOLO / WORKER PATH (original code)
                    // =============================================================
                    if (is_jetson_) logGpuMemory("Before sieve");
                    auto t_sieve = clock::now();
                    SieveStage();
                    time_sieve_sec = std::chrono::duration<double>(clock::now() - t_sieve).count();
                    if (is_jetson_) logGpuMemory("After sieve");

                    // Write relations to disk if requested
                    if (saveRelationsIfRequested("")) return;
                }
            }
        }

        // Read relations from disk for LINALG_ONLY
        if (config_.mode == ExecutionMode::LINALG_ONLY) {
            std::string rel_path = config_.work_dir + "/relations.soa";
            LOG(LOG_INFO) << "LINALG_ONLY: Loading SoA relations from " << rel_path;
            if (!mpqs::io::deserialize_v1(rel_path, host_relations_soa_)) {
                LOG(LOG_ERROR_CRITICAL) << "Failed to load SoA relations from " << rel_path;
                return;
            }
            LOG(LOG_INFO) << "Loaded " << host_relations_soa_.num_relations
                          << " relations (" << host_relations_soa_.num_factors << " factors) from disk.";
        }

        // Load v2 relations for MATRIX_ONLY
        if (config_.mode == ExecutionMode::MATRIX_ONLY) {
            std::string rel_path = config_.work_dir + "/relations.v2";
            LOG(LOG_INFO) << "MATRIX_ONLY: Loading v2 relations from " << rel_path;
            mpqs::io::V2Metadata meta;
            int ver = mpqs::io::detect_and_deserialize(
                rel_path, host_relations_soa_,
                host_relations_soa_, host_partials_soa_, meta);
            if (ver == 0) {
                LOG(LOG_ERROR_CRITICAL) << "Failed to load relations from " << rel_path;
                return;
            }
            if (ver == 1) {
                LOG(LOG_INFO) << "Loaded v1 file (no partials). LP columns unavailable.";
            } else {
                LOG(LOG_INFO) << "Loaded v2: " << host_relations_soa_.num_relations << " smooths, "
                              << host_partials_soa_.num_relations << " partials.";
                // Populate factor base from metadata
                f_data_.factorBase = meta.factor_base;
                f_data_.size = meta.factor_base.size();
                config_.N = meta.N;
                config_.lp1_bound = meta.lp_bound;
            }
        }

        // Subsample partials/LP-combined for LP fraction experiments (matrix_only mode).
        if (config_.mode == ExecutionMode::MATRIX_ONLY && config_.partial_subsample < 1.0) {
            std::mt19937 rng(42);  // deterministic seed for reproducibility
            std::bernoulli_distribution coin(config_.partial_subsample);

            // Preprocess target: explicit PREPROCESS mode, or AUTO with partials present.
            const bool preprocess_target =
                (config_.matrix_mode == MatrixMode::PREPROCESS) ||
                (config_.matrix_mode == MatrixMode::AUTO &&
                 host_partials_soa_.num_relations > 0);

            if (preprocess_target) {
                // Subsample raw partials; pure smooths are unchanged.
                const size_t n_before = host_partials_soa_.num_relations;
                structures::HostRelationBatch sub;
                sub.factor_offsets.push_back(0);
                uint64_t fcursor = 0;
                for (size_t i = 0; i < n_before; ++i) {
                    if (!coin(rng)) continue;
                    sub.sqrt_Q.push_back(host_partials_soa_.sqrt_Q[i]);
                    sub.signs.push_back(host_partials_soa_.signs[i]);
                    sub.val_2_exps.push_back(host_partials_soa_.val_2_exps[i]);
                    sub.large_primes.push_back(host_partials_soa_.large_primes[i]);
                    const uint64_t fb = host_partials_soa_.factor_offsets[i];
                    const uint64_t fe = host_partials_soa_.factor_offsets[i + 1];
                    for (uint64_t j = fb; j < fe; ++j) {
                        sub.factor_indices.push_back(host_partials_soa_.factor_indices[j]);
                        sub.factor_counts.push_back(host_partials_soa_.factor_counts[j]);
                    }
                    fcursor += (fe - fb);
                    sub.factor_offsets.push_back(fcursor);
                }
                sub.num_relations = sub.sqrt_Q.size();
                sub.num_factors = fcursor;
                LOG(LOG_INFO) << "Subsampled raw partials: " << n_before << " -> "
                              << sub.num_relations << " (" << std::fixed << std::setprecision(1)
                              << (100.0 * sub.num_relations / std::max(n_before, size_t(1)))
                              << "% retained, seed=42).";
                host_partials_soa_ = std::move(sub);
            } else {
                // Legacy mode: subsample only LP-combined entries; pure smooths always retained.
                const size_t n_before = host_relations_soa_.num_relations;
                structures::HostRelationBatch sub;
                sub.factor_offsets.push_back(0);
                uint64_t fcursor = 0, lp_kept = 0, lp_total = 0;
                for (size_t i = 0; i < n_before; ++i) {
                    const bool is_lp = host_relations_soa_.large_primes[i] > 1;
                    if (is_lp) lp_total++;
                    if (is_lp && !coin(rng)) continue;  // subsample LP-combined only
                    sub.sqrt_Q.push_back(host_relations_soa_.sqrt_Q[i]);
                    sub.signs.push_back(host_relations_soa_.signs[i]);
                    sub.val_2_exps.push_back(host_relations_soa_.val_2_exps[i]);
                    sub.large_primes.push_back(host_relations_soa_.large_primes[i]);
                    const uint64_t fb = host_relations_soa_.factor_offsets[i];
                    const uint64_t fe = host_relations_soa_.factor_offsets[i + 1];
                    for (uint64_t j = fb; j < fe; ++j) {
                        sub.factor_indices.push_back(host_relations_soa_.factor_indices[j]);
                        sub.factor_counts.push_back(host_relations_soa_.factor_counts[j]);
                    }
                    fcursor += (fe - fb);
                    sub.factor_offsets.push_back(fcursor);
                    if (is_lp) lp_kept++;
                }
                sub.num_relations = sub.sqrt_Q.size();
                sub.num_factors = fcursor;
                LOG(LOG_INFO) << "Subsampled LP-combined: " << lp_total << " -> "
                              << lp_kept << " (" << std::fixed << std::setprecision(1)
                              << (100.0 * lp_kept / std::max(lp_total, uint64_t(1)))
                              << "% retained). Total smooths: " << n_before << " -> "
                              << sub.num_relations << " (seed=42).";
                host_relations_soa_ = std::move(sub);
            }
        }

        // Subsample pure smooths for LP fraction experiments (matrix_only mode).
        // Inverse of partial_subsample: LP-combined always kept, pure smooths thinned.
        if (config_.mode == ExecutionMode::MATRIX_ONLY && config_.smooth_subsample < 1.0) {
            std::mt19937 rng(43);  // seed 43 (distinct from partial_subsample seed 42)
            std::bernoulli_distribution coin(config_.smooth_subsample);

            const size_t n_before = host_relations_soa_.num_relations;
            structures::HostRelationBatch sub;
            sub.factor_offsets.push_back(0);
            uint64_t fcursor = 0, smooth_before = 0, smooth_kept = 0, lp_count = 0;
            for (size_t i = 0; i < n_before; ++i) {
                const bool is_lp = host_relations_soa_.large_primes[i] > 1;
                if (is_lp) {
                    ++lp_count;
                } else {
                    ++smooth_before;
                    if (!coin(rng)) continue;  // subsample pure smooths
                    ++smooth_kept;
                }
                sub.sqrt_Q.push_back(host_relations_soa_.sqrt_Q[i]);
                sub.signs.push_back(host_relations_soa_.signs[i]);
                sub.val_2_exps.push_back(host_relations_soa_.val_2_exps[i]);
                sub.large_primes.push_back(host_relations_soa_.large_primes[i]);
                const uint64_t fb = host_relations_soa_.factor_offsets[i];
                const uint64_t fe = host_relations_soa_.factor_offsets[i + 1];
                for (uint64_t j = fb; j < fe; ++j) {
                    sub.factor_indices.push_back(host_relations_soa_.factor_indices[j]);
                    sub.factor_counts.push_back(host_relations_soa_.factor_counts[j]);
                }
                fcursor += (fe - fb);
                sub.factor_offsets.push_back(fcursor);
            }
            sub.num_relations = sub.sqrt_Q.size();
            sub.num_factors = fcursor;
            LOG(LOG_INFO) << "Subsampled pure smooths: " << smooth_before << " -> "
                          << smooth_kept << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * smooth_kept / std::max(smooth_before, size_t(1)))
                          << "% retained). LP-combined: " << lp_count << " unchanged."
                          << " Total: " << n_before << " -> " << sub.num_relations
                          << " (seed=43).";
            host_relations_soa_ = std::move(sub);
        }

        // Truncated sieve without --sieve_truncate_continue: skip downstream stages
        const bool skip_downstream = sieve_truncated_ && !config_.sieve_truncate_continue;
        if (skip_downstream) {
            LOG(LOG_INFO) << "Truncated sieve — skipping matrix/BW/sqrt.";
        }

        // 2. Relation validation + Matrix Construction
        if (!skip_downstream &&
            (config_.mode == ExecutionMode::FULL_PIPELINE || config_.mode == ExecutionMode::LINALG_ONLY
             || config_.mode == ExecutionMode::MATRIX_ONLY)) {
            // Count sanity check — skip for preprocess mode (singleton removal
            // reduces effective column count, so underdetermined is expected).
            if (config_.matrix_mode != MatrixMode::PREPROCESS &&
                host_relations_soa_.num_relations < f_data_.factorBase.size()) {
                LOG(LOG_ERROR_CRITICAL) << "Insufficient relations: "
                    << host_relations_soa_.num_relations << " < FB size " << f_data_.factorBase.size();
                return;
            }

#ifdef MPQS_VALIDATE_RELATIONS
            // Full relation validation — expensive GPU sync; only enabled when
            // MPQS_VALIDATE_RELATIONS is defined at compile time.
            LOG(LOG_INFO) << "Validating " << host_relations_soa_.num_relations
                          << " relations before matrix construction...";
            structures::RelationBatch::validate_host_batch(
                host_relations_soa_, f_data_.factorBase, config_.N);
#endif

            {
                if (is_jetson_) logGpuMemory("Before matrix");
                auto t_matrix = clock::now();

                // Reliable LP fraction: count LP-combined relations in final batch.
                // host_relations_soa_ contains smooth + LP-combined (large_primes[i] > 1).
                // This is consistent across solo and cluster modes and does not depend
                // on the raw witness buffer size.
                uint64_t lp_combined_count = 0;
                for (size_t i = 0; i < host_relations_soa_.num_relations; ++i) {
                    if (host_relations_soa_.large_primes[i] > 1) ++lp_combined_count;
                }
                double lp_fraction = (host_relations_soa_.num_relations > 0)
                    ? static_cast<double>(lp_combined_count)
                      / static_cast<double>(host_relations_soa_.num_relations)
                    : 0.0;

                // Determine matrix mode from CLI override or auto-detection.
                bool force_preprocess = (config_.matrix_mode == MatrixMode::PREPROCESS);
                bool force_legacy     = (config_.matrix_mode == MatrixMode::LEGACY);

                if (force_preprocess && host_partials_soa_.num_relations == 0) {
                    LOG(LOG_WARNING) << "matrix_mode=preprocess requested but no raw partials "
                                     << "available. Falling back to legacy.";
                    force_preprocess = false;
                    force_legacy     = true;
                }

                if (force_legacy) {
                    used_expanded_matrix_ = false;
                } else if (force_preprocess) {
                    used_expanded_matrix_ = true;
                } else {
                    // AUTO: expand when LP fraction exceeds threshold and partials exist.
                    used_expanded_matrix_ = (host_partials_soa_.num_relations > 0 &&
                                             (lp_fraction > config_.lp_preprocess_threshold ||
                                              config_.mode == ExecutionMode::MATRIX_ONLY));
                }

                LOG(LOG_INFO) << "LP fraction: " << std::fixed << std::setprecision(1)
                              << (lp_fraction * 100.0) << "% ("
                              << lp_combined_count << "/"
                              << host_relations_soa_.num_relations << " LP-combined). "
                              << "Matrix mode: "
                              << (used_expanded_matrix_ ? "PREPROCESS" : "LEGACY")
                              << (config_.matrix_mode == MatrixMode::AUTO
                                      ? " (auto)" : " (forced)") << ".";

                if (used_expanded_matrix_) {

                    // Filter unmatched LP witnesses: keep only partials whose LP
                    // value appears >= 2 times (matched pairs).  Unmatched witnesses
                    // produce singleton LP columns removed by the singleton pass,
                    // but building them wastes O(n_partial) host memory (OOM at 500K+).
                    {
                        const size_t n_part = host_partials_soa_.num_relations;
                        std::unordered_map<unsigned __int128, uint32_t, Hash128> lp_freq;
                        lp_freq.reserve(n_part);
                        for (size_t i = 0; i < n_part; ++i)
                            lp_freq[host_partials_soa_.large_primes[i]]++;

                        structures::HostRelationBatch filtered;
                        uint64_t factor_cursor = 0;
                        filtered.factor_offsets.push_back(0);
                        for (size_t i = 0; i < n_part; ++i) {
                            if (lp_freq[host_partials_soa_.large_primes[i]] < 2) continue;
                            filtered.sqrt_Q.push_back(host_partials_soa_.sqrt_Q[i]);
                            filtered.signs.push_back(host_partials_soa_.signs[i]);
                            filtered.val_2_exps.push_back(host_partials_soa_.val_2_exps[i]);
                            filtered.large_primes.push_back(host_partials_soa_.large_primes[i]);
                            uint64_t f_begin = host_partials_soa_.factor_offsets[i];
                            uint64_t f_end   = host_partials_soa_.factor_offsets[i + 1];
                            for (uint64_t j = f_begin; j < f_end; ++j) {
                                filtered.factor_indices.push_back(
                                    host_partials_soa_.factor_indices[j]);
                                filtered.factor_counts.push_back(
                                    host_partials_soa_.factor_counts[j]);
                            }
                            factor_cursor += (f_end - f_begin);
                            filtered.factor_offsets.push_back(factor_cursor);
                        }
                        filtered.num_relations = filtered.sqrt_Q.size();
                        filtered.num_factors   = factor_cursor;

                        LOG(LOG_INFO) << "Filtered " << n_part << " partials to "
                                      << filtered.num_relations << " matched witnesses ("
                                      << std::fixed << std::setprecision(1)
                                      << (n_part > 0
                                          ? (1.0 - double(filtered.num_relations) / n_part) * 100.0
                                          : 0.0)
                                      << "% reduction).";
                        host_partials_soa_ = std::move(filtered);
                    }

                    // Conditional LP-combined inclusion based on cluster mode.
                    // Cluster mode: M6 buffers BOTH halves of each matched pair in
                    // host_partials_soa_ → including LP-combined creates GF(2) redundancy
                    // (C_ab = P_a + P_b) yielding trivial kernel vectors. Exclude them.
                    // Solo GPU LP: only the non-matching half is stored in witness buffer →
                    // no systematic redundancy → include LP-combined as smooth-like rows.
                    if (config_.cluster_mode != ClusterMode::SOLO &&
                        host_partials_soa_.num_relations > 0) {
                        // Cluster mode with both partial halves — exclude LP-combined
                        raw_smooths_soa_ = structures::HostRelationBatch{};
                        raw_smooths_soa_.factor_offsets.push_back(0);  // CSR sentinel
                        uint64_t excluded = 0;
                        for (size_t i = 0; i < host_relations_soa_.num_relations; ++i) {
                            if (host_relations_soa_.large_primes[i] > 1) {
                                ++excluded;
                                continue;
                            }
                            raw_smooths_soa_.sqrt_Q.push_back(host_relations_soa_.sqrt_Q[i]);
                            raw_smooths_soa_.signs.push_back(host_relations_soa_.signs[i]);
                            raw_smooths_soa_.val_2_exps.push_back(host_relations_soa_.val_2_exps[i]);
                            raw_smooths_soa_.large_primes.push_back(host_relations_soa_.large_primes[i]);
                            uint64_t fstart = host_relations_soa_.factor_offsets[i];
                            uint64_t fend   = host_relations_soa_.factor_offsets[i + 1];
                            for (uint64_t j = fstart; j < fend; ++j) {
                                raw_smooths_soa_.factor_indices.push_back(host_relations_soa_.factor_indices[j]);
                                raw_smooths_soa_.factor_counts.push_back(host_relations_soa_.factor_counts[j]);
                            }
                            raw_smooths_soa_.num_factors += (fend - fstart);
                            raw_smooths_soa_.factor_offsets.push_back(raw_smooths_soa_.num_factors);
                            raw_smooths_soa_.num_relations++;
                        }
                        LOG(LOG_INFO) << "Excluded " << excluded << " LP-combined relations from expanded matrix "
                                      << "(partial rows present, avoiding GF(2) redundancy). "
                                      << raw_smooths_soa_.num_relations << " raw smooths retained.";
                    } else {
                        // Solo mode or no partial rows — include LP-combined as smooth-like rows
                        raw_smooths_soa_ = host_relations_soa_;
                        uint64_t lp_combined_count = 0;
                        for (size_t i = 0; i < host_relations_soa_.num_relations; ++i) {
                            if (host_relations_soa_.large_primes[i] > 1) ++lp_combined_count;
                        }
                        if (lp_combined_count > 0) {
                            LOG(LOG_INFO) << "Including " << lp_combined_count
                                          << " LP-combined relations in expanded matrix as smooth-like rows "
                                          << "(no partial rows, no GF(2) redundancy).";
                        }
                    }
                }
                if (used_expanded_matrix_) {

                    // M9f: Packed GPU pipeline (solo mode, GPU backend)
                    // Replaces ExpandedMatrixBuilder + preprocessMatrix + product char cols
                    // with a single gpuPreprocessMatrix_packed() call chaining M9a→M9b→M9c→M9e→M9f.
                    // Supports both sieve path (device-resident data) and MATRIX_ONLY (host→device upload).
                    const bool use_packed_pipeline =
                        config_.cluster_mode == ClusterMode::SOLO &&
                        (config_.matrix_backend == 1 /*GPU*/ || config_.matrix_backend == 2 /*AUTO*/) &&
                        (postprocessor_ != nullptr || config_.mode == ExecutionMode::MATRIX_ONLY);

                    if (use_packed_pipeline) {
                        LOG(LOG_INFO) << "M9v2: Using packed GPU preprocessing pipeline.";

                        // Smooth data source: device-resident from sieve, or uploaded from disk.
                        structures::RelationBatch smooth_dev_batch;  // RAII; unused if sieve path
                        structures::RelationBatchView smooth_view{};
                        uint64_t n_smooth = 0;

                        if (postprocessor_ != nullptr) {
                            // Sieve path: smooths already on device
                            smooth_view = postprocessor_->getPersistentBatch()->get_view();
                            n_smooth = postprocessor_->getPersistentCount();
                        } else {
                            // MATRIX_ONLY: upload host smooths to device
                            LOG(LOG_INFO) << "MATRIX_ONLY: Uploading " << raw_smooths_soa_.num_relations
                                          << " smooths + " << host_partials_soa_.num_relations
                                          << " partials to device for M9v2 pipeline.";
                            smooth_dev_batch.initiate(config_.device_id);
                            smooth_dev_batch.uploadFromHost(raw_smooths_soa_);
                            smooth_view = smooth_dev_batch.get_view();
                            n_smooth = raw_smooths_soa_.num_relations;
                        }

                        // Upload filtered partials to device for packed pipeline.
                        // The partials were already filtered on host (LP freq >= 2).
                        structures::RelationBatch partial_dev_batch;
                        partial_dev_batch.initiate(config_.device_id);
                        partial_dev_batch.uploadFromHost(host_partials_soa_);
                        auto partial_view = partial_dev_batch.get_view();
                        uint64_t n_partial = host_partials_soa_.num_relations;

                        auto t_preproc = clock::now();
                        auto v2 = matrix::gpuPreprocessMatrix_packed(
                            smooth_view, n_smooth,
                            partial_view, n_partial,
                            static_cast<uint32_t>(f_data_.factorBase.size()),
                            config_.N, f_data_.factorBase,
                            10,  // k_max
                            200, // max_weight
                            config_.truncation_factor,
                            config_.compact_cycles,
                            config_.matrix_truncation_excess,
                            config_.matrix_gf2_floor_factor,
                            config_.matrix_gf2_min_floor);
                        double preproc_sec = std::chrono::duration<double>(
                            clock::now() - t_preproc).count();

                        LOG(LOG_INFO) << "M9v2: " << v2.singletons_removed
                                      << " singletons (" << v2.singleton_iterations << " iters), "
                                      << v2.w2_merges << " w2, " << v2.hw_merges << " hw → "
                                      << v2.reduced.n_rows << " x " << v2.reduced.n_cols
                                      << " (GF2 NNZ=" << v2.gf2_nnz << ") ["
                                      << std::fixed << std::setprecision(2) << preproc_sec << "s]";

                        // Persist telemetry
                        summary_matrix_rows_ = v2.reduced.n_rows;
                        summary_matrix_cols_ = v2.reduced.n_cols;
                        summary_matrix_nnz_  = v2.reduced.row_offsets.empty() ? 0
                                               : v2.reduced.row_offsets[v2.reduced.n_rows];

                        // Convert to HostMatrix for BW
                        matrix_A_ = matrix::ConvertFromCSR(v2.reduced);

                        int excess = static_cast<int>(matrix_A_.n_rows) - static_cast<int>(matrix_A_.n_cols);
                        if (excess < 0) {
                            LOG(LOG_ERROR_MAJOR) << "Underdetermined System! Deficit: " << -excess;
                        } else {
                            LOG(LOG_STATS) << "System Overdetermined. Excess: " << excess;
                        }

                        // Store V2 result for sqrt integration
                        used_packed_pipeline_ = true;
                        preproc_v2_result_ = std::move(v2);

                    } else {
                        // CPU preprocessing path (cluster mode or CPU backend)

                        // M1: Build expanded matrix (F+2+L columns)
                        matrix::ExpandedMatrixBuilder builder;
                        auto expanded = builder.build(raw_smooths_soa_, host_partials_soa_,
                                                      static_cast<uint32_t>(f_data_.factorBase.size()));
                        LOG(LOG_INFO) << "Expanded matrix: " << expanded.matrix.n_rows << " x "
                                      << expanded.matrix.n_cols << " ("
                                      << expanded.num_lp_cols << " LP cols).";

                        // M2–M4: Preprocessing pipeline (singleton removal → merges)
                        auto t_preproc = clock::now();
                        auto preproc = matrix::preprocessMatrix(
                            expanded.matrix,
                            static_cast<matrix::MatrixBackend>(config_.matrix_backend));
                        double preproc_sec = std::chrono::duration<double>(
                            clock::now() - t_preproc).count();
                        const char* preproc_backend_name =
                            (config_.matrix_backend == 1) ? "gpu" :
                            (config_.matrix_backend == 2) ? "auto" : "cpu";
                        LOG(LOG_INFO) << "Preprocessing: " << preproc.singletons_removed
                                      << " singletons (" << preproc.singleton_iterations << " iters), "
                                      << preproc.w2_merges << " w2 merges, "
                                      << preproc.hw_merges << " hw merges → "
                                      << preproc.reduced.n_rows << " x " << preproc.reduced.n_cols
                                      << "  [" << std::fixed << std::setprecision(2)
                                      << preproc_sec << "s, backend: " << preproc_backend_name << "]";

                        // M8: Matrix truncation (M12-S1: char-col-aware target).
                        // Char cols (32) are appended *after* this call (see below);
                        // pass n_extra_cols=32 so the target accounts for them.
                        if (config_.truncation_factor > 0.0) {
                            constexpr uint32_t kCharColCount = 32;
                            auto trunc = matrix::truncateMatrix(
                                preproc.reduced, preproc.row_map,
                                config_.truncation_factor,
                                /*n_extra_cols=*/kCharColCount,
                                /*k_excess=*/config_.matrix_truncation_excess);
                            if (trunc.rows_removed > 0) {
                                preproc.reduced = std::move(trunc.truncated);
                                preproc.row_map = std::move(trunc.row_map);
                                LOG(LOG_INFO) << "M8: Truncated " << trunc.rows_removed << " rows, "
                                              << trunc.cols_removed << " empty columns removed.";
                            }
                        }

                        // Store merge tree and row map for kernel vector expansion
                        merge_tree_ = std::move(preproc.merge_tree);
                        preproc_row_map_ = std::move(preproc.row_map);

                        // M7: Product character columns via merge-tree sqrt_Q products
                        {
                            auto product_chars = matrix::computeProductCharacterColumns(
                                preproc_row_map_, merge_tree_,
                                raw_smooths_soa_, host_partials_soa_,
                                raw_smooths_soa_.num_relations,
                                config_.N, f_data_.factorBase);
                            matrix::AppendCharacterColumns(
                                preproc.reduced, product_chars,
                                preproc.reduced.n_rows);
                            LOG(LOG_INFO) << "M7: Appended " << product_chars.k
                                          << " product character columns to reduced matrix ("
                                          << preproc.reduced.n_rows << " x "
                                          << preproc.reduced.n_cols << ").";
                        }

                        // Persist matrix telemetry
                        summary_matrix_rows_ = preproc.reduced.n_rows;
                        summary_matrix_cols_ = preproc.reduced.n_cols;
                        summary_matrix_nnz_  = preproc.reduced.row_offsets.empty() ? 0
                                               : preproc.reduced.row_offsets[preproc.reduced.n_rows];

                        // Convert reduced CSR to HostMatrix for BW
                        matrix_A_ = matrix::ConvertFromCSR(preproc.reduced);

                        int excess = static_cast<int>(matrix_A_.n_rows) - static_cast<int>(matrix_A_.n_cols);
                        if (excess < 0) {
                            LOG(LOG_ERROR_MAJOR) << "Underdetermined System! Deficit: " << -excess;
                        } else {
                            LOG(LOG_STATS) << "System Overdetermined. Excess: " << excess;
                        }
                    }
                } else {
                    // Existing projected-matrix path (no LP columns)
                    MatrixStage();
                }

                time_matrix_sec = std::chrono::duration<double>(clock::now() - t_matrix).count();
                if (is_jetson_) logGpuMemory("After matrix");
            }
        }

        // 3. Linear Algebra
        if (!skip_downstream &&
            (config_.mode == ExecutionMode::FULL_PIPELINE || config_.mode == ExecutionMode::LINALG_ONLY
             || config_.mode == ExecutionMode::MATRIX_ONLY)) {
            if (matrix_A_.n_rows > 0) {
                if (is_jetson_) logGpuMemory("Before linalg");
                auto t_linalg = clock::now();
                LinearAlgebraStage();
                time_linalg_sec = std::chrono::duration<double>(clock::now() - t_linalg).count();
                if (is_jetson_) logGpuMemory("After linalg");
            } else {
                LOG(LOG_ERROR_CRITICAL) << "Matrix A is empty. Cannot proceed to Linear Algebra.";
            }
        }

        // 3b. Kernel vector expansion (expanded-matrix path)
        //     Map BW kernel vectors from reduced-matrix row indices back to
        //     original relation indices via the merge tree, then build a
        //     concatenated HostRelationBatch (smooths + partials) for sqrt.
        if (!skip_downstream && used_expanded_matrix_ && !kernel_solutions_.empty()) {

          if (used_packed_pipeline_ && preproc_v2_result_.has_value()) {
            // ============================================================
            // V2 packed pipeline: build synthetic batch from merged 1-partials
            // No merge tree expansion — merged rows carry sqrt_Q and full exponents.
            // ============================================================
            LOG(LOG_INFO) << "M9f: Building sqrt batch from " << kernel_solutions_.size()
                          << " kernel vectors (packed pipeline, no merge tree)...";
            using clock_kv = std::chrono::high_resolution_clock;
            auto t_expand = clock_kv::now();

            const auto& v2 = *preproc_v2_result_;
            const uint32_t n_merged = static_cast<uint32_t>(v2.merged_sqrt_Q.size());
            const uint32_t v2_fb_size = v2.fb_size;

            // Reconstruct LP column → value mapping (M9a assigns in sorted order)
            std::vector<unsigned __int128> sorted_unique_lps;
            {
                std::set<unsigned __int128> lp_set;
                for (size_t i = 0; i < host_partials_soa_.num_relations; i++)
                    if (host_partials_soa_.large_primes[i] > 1)
                        lp_set.insert(host_partials_soa_.large_primes[i]);
                // LP-combined smooth rows also contribute LP values to M9a's column assignment
                for (size_t i = 0; i < host_relations_soa_.num_relations; i++)
                    if (host_relations_soa_.large_primes[i] > 1)
                        lp_set.insert(host_relations_soa_.large_primes[i]);
                sorted_unique_lps.assign(lp_set.begin(), lp_set.end());
            }

            // Build synthetic HostRelationBatch from merged 1-partial data.
            // Factor entries: only FB columns (expanded col in [2, fb_size+2)), remapped to FB index.
            // LP column entries are handled separately via LP correction below.
            structures::HostRelationBatch combined;
            combined.num_relations = n_merged;
            combined.sqrt_Q.resize(n_merged);
            combined.signs.resize(n_merged);
            combined.val_2_exps.resize(n_merged);
            combined.large_primes.resize(n_merged, 1);  // Disable sqrt's built-in LP handler
            combined.factor_offsets.reserve(n_merged + 1);
            combined.factor_offsets.push_back(0);

            for (uint32_t i = 0; i < n_merged; i++) {
                combined.sqrt_Q[i]     = v2.merged_sqrt_Q[i];
                combined.signs[i]      = v2.merged_signs[i];
                combined.val_2_exps[i] = v2.merged_val_2_exps[i];

                uint32_t f_begin = v2.merged_factor_offsets[i];
                uint32_t f_end   = v2.merged_factor_offsets[i + 1];
                for (uint32_t j = f_begin; j < f_end; j++) {
                    uint32_t post_sing_col = v2.merged_factor_indices[j];
                    uint8_t  exp           = v2.merged_factor_exponents[j];

                    // Map post-singleton col → expanded col
                    uint32_t expanded_col = (post_sing_col < v2.singleton_col_map.size())
                                          ? v2.singleton_col_map[post_sing_col]
                                          : post_sing_col;

                    // FB columns: expanded_col in [2, fb_size + 2)
                    if (expanded_col >= 2 && expanded_col < v2_fb_size + 2) {
                        combined.factor_indices.push_back(expanded_col - 2);
                        combined.factor_counts.push_back(exp);
                    }
                    // LP and sign/prime-2 columns: handled below or by metadata
                }
                combined.factor_offsets.push_back(
                    static_cast<uint64_t>(combined.factor_indices.size()));
            }
            combined.num_factors = combined.factor_indices.size();

            // BW kernel solutions are already over the GF(2) matrix rows (= merged rows).
            // No re-packing needed — the synthetic batch has the same indexing.

            // Precompute LP Y-corrections per solution in Montgomery domain.
            // For each solution, iterate selected merged rows' LP column entries,
            // look up LP value via singleton_col_map + sorted LP assignment, accumulate LP^(e/2).
            {
                mpqs::math::Montgomery mont(config_.N);
                std::vector<mpqs::uint512> precomputed_lp_y(kernel_solutions_.size());

                for (size_t sol = 0; sol < kernel_solutions_.size(); ++sol) {
                    const auto& kv = kernel_solutions_[sol];
                    // Count LP exponent contributions from selected merged rows
                    std::unordered_map<unsigned __int128, uint32_t, Hash128> lp_exp_sum;

                    // Iterate the bit-vector directly
                    for (uint32_t i = 0; i < n_merged; i++) {
                        uint32_t word = i / 64;
                        uint32_t bit  = i % 64;
                        if (word >= kv.size() || !((kv[word] >> bit) & 1)) continue;

                        uint32_t f_begin = v2.merged_factor_offsets[i];
                        uint32_t f_end   = v2.merged_factor_offsets[i + 1];
                        for (uint32_t j = f_begin; j < f_end; j++) {
                            uint32_t post_sing_col = v2.merged_factor_indices[j];
                            uint8_t  exp           = v2.merged_factor_exponents[j];

                            uint32_t expanded_col = (post_sing_col < v2.singleton_col_map.size())
                                                  ? v2.singleton_col_map[post_sing_col]
                                                  : post_sing_col;

                            // LP column: expanded_col >= fb_size + 2
                            if (expanded_col >= v2_fb_size + 2) {
                                uint32_t lp_offset = expanded_col - v2_fb_size - 2;
                                if (lp_offset < sorted_unique_lps.size()) {
                                    lp_exp_sum[sorted_unique_lps[lp_offset]] += exp;
                                }
                            }
                        }
                    }

                    // Compute ∏ LP^(e/2) in Montgomery domain
                    mpqs::uint512 product_mont = mont.transform(mpqs::uint512((uint32_t)1));
                    for (auto& [lp_val, total_exp] : lp_exp_sum) {
                        if (total_exp == 0) continue;
                        if (total_exp & 1) {
                            LOG(LOG_ERROR_CRITICAL) << "Odd LP exponent sum " << total_exp
                                << " in V2 solution " << sol << " — GF(2) invariant violated!";
                        }
                        uint32_t half = total_exp / 2;
                        if (half > 0) {
                            mpqs::uint512 base_mont = mont.transform(mpqs::uint512(lp_val));
                            mpqs::uint512 term = mont.pow(base_mont, mpqs::uint512(half));
                            product_mont = mont.mul(product_mont, term);
                        }
                    }
                    precomputed_lp_y[sol] = product_mont;
                }
                precomputed_lp_y_ = std::move(precomputed_lp_y);
            }

            // Replace orchestrator state for sqrt
            host_relations_soa_ = std::move(combined);
            // kernel_solutions_ stay as-is (already over GF(2) = merged rows)

            double expand_sec = std::chrono::duration<double>(
                clock_kv::now() - t_expand).count();
            LOG(LOG_INFO) << "M9f: V2 sqrt batch built in "
                          << FormatDuration(expand_sec * 1000.0)
                          << " (" << n_merged << " merged rows).";

          } else {
            // ============================================================
            // CPU preprocessing path: merge tree expansion (existing logic)
            // ============================================================
            LOG(LOG_INFO) << "Expanding " << kernel_solutions_.size()
                          << " kernel vectors via merge tree...";
            using clock_kv = std::chrono::high_resolution_clock;
            auto t_expand = clock_kv::now();

            const size_t ns = raw_smooths_soa_.num_relations;
            const size_t np = host_partials_soa_.num_relations;

            // Build concatenated batch: smooths [0, ns) then partials [ns, ns+np)
            structures::HostRelationBatch combined;
            combined.num_relations = ns + np;

            // Algebra vectors — concatenate smooth then partial
            auto concat = [](auto& dst, const auto& a, const auto& b) {
                dst.reserve(a.size() + b.size());
                dst.insert(dst.end(), a.begin(), a.end());
                dst.insert(dst.end(), b.begin(), b.end());
            };
            concat(combined.sqrt_Q,       raw_smooths_soa_.sqrt_Q,       host_partials_soa_.sqrt_Q);
            concat(combined.signs,        raw_smooths_soa_.signs,        host_partials_soa_.signs);
            concat(combined.val_2_exps,   raw_smooths_soa_.val_2_exps,   host_partials_soa_.val_2_exps);
            concat(combined.large_primes, raw_smooths_soa_.large_primes, host_partials_soa_.large_primes);

            // CSR factor data — shift partial offsets by smooth factor count
            const uint64_t smooth_nf = raw_smooths_soa_.num_factors;
            combined.factor_offsets = raw_smooths_soa_.factor_offsets;  // includes [0..ns]
            // Append partial offsets [1..np] shifted by smooth_nf
            for (size_t i = 1; i < host_partials_soa_.factor_offsets.size(); ++i) {
                combined.factor_offsets.push_back(
                    host_partials_soa_.factor_offsets[i] + smooth_nf);
            }
            concat(combined.factor_indices, raw_smooths_soa_.factor_indices,
                                            host_partials_soa_.factor_indices);
            concat(combined.factor_counts,  raw_smooths_soa_.factor_counts,
                                            host_partials_soa_.factor_counts);
            combined.num_factors = raw_smooths_soa_.num_factors
                                 + host_partials_soa_.num_factors;

            // Expand each kernel vector and repack as bits over the combined batch
            const uint32_t num_reduced_rows = static_cast<uint32_t>(preproc_row_map_.size());
            std::vector<std::vector<uint64_t>> expanded_solutions;
            expanded_solutions.reserve(kernel_solutions_.size());
            size_t total_selected = 0;

            for (const auto& kv : kernel_solutions_) {
                auto orig_indices = matrix::expandKernelVector(
                    kv, num_reduced_rows, preproc_row_map_, merge_tree_);

                // Repack as bit-vector over combined batch
                const size_t n_words = (combined.num_relations + 63) / 64;
                std::vector<uint64_t> bits(n_words, 0);
                for (uint32_t idx : orig_indices) {
                    if (idx < combined.num_relations) {
                        bits[idx / 64] |= (1ULL << (idx % 64));
                    }
                }
                total_selected += orig_indices.size();
                expanded_solutions.push_back(std::move(bits));
            }

#ifdef MPQS_DEBUG
            // Verify GF(2) column-sum invariant for the first expanded solution.
            if (!expanded_solutions.empty()) {
                const auto& bits = expanded_solutions[0];
                const size_t nr = combined.num_relations;
                std::unordered_map<uint32_t, uint32_t> col_sums;
                for (size_t idx = 0; idx < nr; ++idx) {
                    if (!(bits[idx / 64] & (1ULL << (idx % 64)))) continue;
                    if (combined.signs[idx]) col_sums[0]++;
                    if (combined.val_2_exps[idx] & 1) col_sums[1]++;
                    uint64_t fb = combined.factor_offsets[idx];
                    uint64_t fe = combined.factor_offsets[idx + 1];
                    for (uint64_t j = fb; j < fe; ++j) {
                        if (combined.factor_counts[j] & 1) {
                            col_sums[combined.factor_indices[j] + 2]++;
                        }
                    }
                }
                bool gf2_ok = true;
                for (auto& [col, sum] : col_sums) {
                    if (sum & 1) {
                        LOG(LOG_ERROR_CRITICAL) << "GF(2) invariant violated: column " << col
                                                << " has odd sum " << sum << " in expanded solution 0.";
                        gf2_ok = false;
                    }
                }
                if (gf2_ok) {
                    LOG(LOG_INFO) << "GF(2) column-sum invariant verified for expanded solution 0.";
                }
            }
#endif

            double avg_rels = expanded_solutions.empty() ? 0.0
                : static_cast<double>(total_selected) / expanded_solutions.size();
            LOG(LOG_INFO) << "Expanded " << expanded_solutions.size()
                          << " kernel vectors over " << combined.num_relations
                          << " relations (avg " << std::fixed << std::setprecision(1)
                          << avg_rels << " relations/vector).";

            // Change 5 (--sqrt_diagnostic): log solution diversity statistics
            if (config_.sqrt_diagnostic) {
                std::unordered_set<size_t> seen_hashes;
                for (const auto& sol : expanded_solutions) {
                    size_t h = sol.size();
                    for (uint64_t w : sol) {
                        h ^= std::hash<uint64_t>{}(w) + 0x9e3779b97f4a7c15ULL
                             + (h << 6) + (h >> 2);
                    }
                    seen_hashes.insert(h);
                }
                LOG(LOG_INFO) << "[sqrt_diagnostic] Solution diversity: "
                              << seen_hashes.size() << " distinct (by hash) / "
                              << expanded_solutions.size() << " total | avg "
                              << std::fixed << std::setprecision(1)
                              << avg_rels << " relations/solution.";
            }

            // Precompute LP Y-contributions per solution in Montgomery domain.
            {
                mpqs::math::Montgomery mont(config_.N);
                std::vector<mpqs::uint512> precomputed_lp_y(expanded_solutions.size());

                for (size_t sol = 0; sol < expanded_solutions.size(); ++sol) {
                    std::unordered_map<unsigned __int128, uint32_t, Hash128> lp_counts;
                    const auto& bits = expanded_solutions[sol];
                    for (size_t idx = 0; idx < combined.num_relations; ++idx) {
                        if (!(bits[idx / 64] & (1ULL << (idx % 64)))) continue;
                        unsigned __int128 lp = combined.large_primes[idx];
                        if (lp > 1) lp_counts[lp] += (idx < ns) ? 2 : 1;
                    }

                    mpqs::uint512 product_mont = mont.transform(mpqs::uint512((uint32_t)1));
                    for (auto& [lp_val, count] : lp_counts) {
                        if (count == 0) continue;
                        if (count & 1) {
                            LOG(LOG_ERROR_CRITICAL) << "Odd LP count " << count
                                << " in solution " << sol << " — GF(2) invariant violated!";
                        }
                        uint32_t half = count / 2;
                        if (half > 0) {
                            mpqs::uint512 base_mont = mont.transform(mpqs::uint512(lp_val));
                            mpqs::uint512 term = mont.pow(base_mont, mpqs::uint512(half));
                            product_mont = mont.mul(product_mont, term);
                        }
                    }
                    precomputed_lp_y[sol] = product_mont;
                }
                precomputed_lp_y_ = std::move(precomputed_lp_y);
            }

            // Disable sqrt's built-in LP handling — precomputed values handle it
            for (size_t i = 0; i < combined.num_relations; ++i) {
                combined.large_primes[i] = 1;
            }

            // Replace orchestrator state for sqrt
            host_relations_soa_ = std::move(combined);
            kernel_solutions_ = std::move(expanded_solutions);

            double expand_sec = std::chrono::duration<double>(
                clock_kv::now() - t_expand).count();
            LOG(LOG_INFO) << "Kernel vector expansion completed in "
                          << FormatDuration(expand_sec * 1000.0);
          }
        }

        // 4. Square Root
        if (config_.mode == ExecutionMode::SQRT_ONLY) {
            LOG(LOG_ERROR_CRITICAL) << "SQRT_ONLY mode is not yet implemented: "
                                    << "no disk loader for kernel_solutions_ or factor base. "
                                    << "Use LINALG_ONLY or FULL_PIPELINE instead.";
            return;
        }
        if (!skip_downstream &&
            (config_.mode == ExecutionMode::FULL_PIPELINE || config_.mode == ExecutionMode::LINALG_ONLY
             || config_.mode == ExecutionMode::MATRIX_ONLY)) {
            if (is_jetson_) logGpuMemory("Before sqrt");
            auto t_sqrt = clock::now();
            SquareRootStage();
            time_sqrt_sec = std::chrono::duration<double>(clock::now() - t_sqrt).count();
            if (is_jetson_) logGpuMemory("After sqrt");
        }

    } catch (const std::exception& e) {
        LOG(LOG_ERROR_CRITICAL) << "Pipeline Aborted: "
				<< e.what();
    }

    auto t_end = clock::now();
    double time_total_sec = std::chrono::duration<double>(t_end - t_start).count();
    double duration = time_total_sec * 1000.0;
    LOG(LOG_INFO) << "Pipeline Finished in "
		  << FormatDuration(duration);

    // Pipeline timing summary (spec §8.2)
    logGpuMemory("Pipeline end", LOG_STATS);

    auto timeLine = [](const char* label, double sec, const std::string& note = "") {
        std::ostringstream oss;
        oss << "  " << std::left << std::setw(17) << (std::string(label) + ":")
            << std::right << std::fixed << std::setprecision(2) << std::setw(8) << sec << " s";
        if (!note.empty()) oss << "    " << note;
        return oss.str();
    };

    const std::string sep64(64, '=');
    const std::string ssep64(64, '-');

    LOG(LOG_INFO) << "";
    LOG(LOG_INFO) << sep64;
    LOG(LOG_INFO) << "  Pipeline Timing Summary";
    LOG(LOG_INFO) << sep64;
    LOG(LOG_INFO) << timeLine("Tuning", time_tuning_sec);
    if (time_autotune_sec > 0)
        LOG(LOG_INFO) << timeLine("Autotune", time_autotune_sec);

    // Sieve line with relation count + LP annotation
    {
        std::string sieve_note;
        if (summary_total_relations_ > 0) {
            sieve_note = "(" + fmtNum(summary_total_relations_) + " relations";
            if (summary_lp_active_ && summary_lp_combined_ > 0)
                sieve_note += "; LP: " + fmtNum(summary_lp_combined_);
            sieve_note += ")";
        }
        LOG(LOG_INFO) << timeLine("Sieve", time_sieve_sec, sieve_note);
    }

    // Matrix line with dims + density annotation
    {
        std::string mat_note;
        if (summary_matrix_rows_ > 0) {
            double density = summary_matrix_nnz_ > 0
                ? 100.0 * summary_matrix_nnz_ / ((double)summary_matrix_rows_ * summary_matrix_cols_)
                : 0.0;
            std::ostringstream mn;
            mn << "(" << fmtNum(summary_matrix_rows_) << " x " << fmtNum(summary_matrix_cols_)
               << "; " << std::fixed << std::setprecision(3) << density << "% density)";
            mat_note = mn.str();
        }
        LOG(LOG_INFO) << timeLine("Matrix", time_matrix_sec, mat_note);
    }

    LOG(LOG_INFO) << timeLine("Linear Algebra", time_linalg_sec);
    LOG(LOG_INFO) << timeLine("Sqrt", time_sqrt_sec);
    LOG(LOG_INFO) << ssep64;
    double time_core_sec = time_sieve_sec + time_linalg_sec + time_sqrt_sec;
    LOG(LOG_INFO) << timeLine("Core (S+LA+Sq)", time_core_sec);
    LOG(LOG_INFO) << timeLine("Total", time_total_sec);

    // Factors line
    if (!result_factors_.empty()) {
        std::string factor_str;
        for (size_t i = 0; i < result_factors_.size(); i++) {
            if (i > 0) factor_str += " x ";
            factor_str += fmtFactor(result_factors_[i]);
        }
        LOG(LOG_INFO) << "  Factors:         " << factor_str;
    } else {
        LOG(LOG_INFO) << "  Factors:         (none found)";
    }
    LOG(LOG_INFO) << sep64;

    // Standalone result lines (for --mute filtering)
    if (!result_factors_.empty()) {
        LOG(LOG_RESULT) << "Factors found:";
        for (const auto& f : result_factors_) {
            LOG(LOG_RESULT) << " " << f.to_string();
        }
    } else {
        LOG(LOG_WARNING) << "No factors found.";
    }
}

// -----------------------------------------------------------------------------
// Stage 1: Tuning
// -----------------------------------------------------------------------------

void MPQSOrchestrator::TuningStage() {
    LOG_SET_MODULE("Tuning");
    LOG_SET_STAGE(LOG_STAGE_PARAM_TUNING, "Tuning");
    LOG(LOG_INFO) << "Parameter Tuning & Pre-computation";

    // --- Graceful rejection for too-small N ---
    int bits = config_.N.msb();
    if (bits < 80) {
        throw std::runtime_error(
            "N is too small (" + std::to_string(bits) + " bits, ~"
            + std::to_string(config_.N.to_string().size()) + " digits). "
            "Minimum supported: ~24 digits (~80 bits).");
    }

    f_data_.F = config_.fb_bound;
    f_data_.M = config_.sieve_bound;

    // --- Small-N parameter pre-selection (orchestrator-side) ---
    // The sieve submodule's determineParams() produces degenerate F/M for
    // small N. Pre-set F and M when both are auto-calculated (not user-pinned).
    // determineParams() skips F/M assignment when they are already nonzero.
    if (f_data_.F == 0 && f_data_.M == 0) {
        if (bits < 100) {
            f_data_.F = 25000;    f_data_.M = 16384;    // FB~1100, tiny N
        } else if (bits < 120) {
            f_data_.F = 30000;    f_data_.M = 32768;    // FB~1400
        } else if (bits < 140) {
            f_data_.F = 45000;    f_data_.M = 32768;    // FB~2000
        } else if (bits < 160) {
            f_data_.F = 65000;    f_data_.M = 65536;    // FB~3000
        } else if (bits < 180) {
            f_data_.F = 120000;   f_data_.M = 131072;   // FB~5500
        } else if (bits < 200) {
            f_data_.F = 200000;   f_data_.M = 262144;   // FB~9000, ~60-digit
        }
        // bits >= 200: leave F=0, M=0 for determineParams
        if (f_data_.F > 0) {
            // Pin small-N table values so auto-apply treats them as user-set
            config_.fb_bound = f_data_.F;
            config_.sieve_bound = f_data_.M;
            config_.pinned_params["fb_bound"] = true;
            config_.pinned_params["sieve_bound"] = true;
            LOG(LOG_STATS) << "Small-N pre-selection (bits=" << bits
                          << "): F=" << f_data_.F << " M=" << f_data_.M;
        }
    }

    // --- Dedup safety factor: auto-raise for small inputs (high dedup rate) ---
    if (!config_.isPinned("dedup_safety_factor") && bits < 265) {
        config_.dedup_safety_factor = 1.35;
    }

    if (config_.auto_tune_parameters) {
        determineParams(&f_data_);
        LOG(LOG_INFO) << "Selected F="
		      << f_data_.F << ", M="
		      << f_data_.M;
    }

    LOG(LOG_INFO) << "Generating Factor Base...";
    generateFactorBase(&f_data_);
    LOG(LOG_INFO) << "Factor Base Size: "
		  << f_data_.size;

    LOG(LOG_INFO) << "Initializing Polynomial Selection...";
    init_a_factors(&f_data_);

    // --- Validate and fix a_factors for small N ---
    // init_a_factors() hardcodes starting index 150 and slides upward.
    // For small N, a_target is small → primes at index 150+ overshoot, and the
    // sliding window can run past the end of the factor base producing garbage
    // indices.  Additionally, advance_a_factors() has underflow bugs when
    // shc_dim (= a_factors.size()) ≤ 3.  Detect both conditions and override
    // with a greedy selection from lower FB indices.
    {
        const uint32_t fb_size = f_data_.factorBase.size();
        bool a_factors_valid = !f_data_.a_factors.empty();
        for (auto idx : f_data_.a_factors) {
            if (idx >= fb_size) { a_factors_valid = false; break; }
        }
        uint32_t shc_dim = static_cast<uint32_t>(f_data_.a_factors.size());

        if (!a_factors_valid || shc_dim < 4) {
            LOG(LOG_WARNING) << "Small-N: init_a_factors produced "
                             << (a_factors_valid ? "valid" : "INVALID")
                             << " a_factors (shc_dim=" << shc_dim
                             << ", fb_size=" << fb_size
                             << "). Overriding with orchestrator-selected a_factors.";

            // Greedy selection: pick primes from index start_idx upward whose
            // product approaches a_target.  Use step-2 indices (matching the
            // submodule's convention) so that advance_a_factors' sliding window
            // has gaps to move into later.
            //
            // Constraints:
            //   - indices ≥ 32  (preserve small primes for sieving)
            //   - shc_dim ∈ [4, 10]
            //   - product ≈ a_target  (within ~2× is acceptable)
            uint32_t start_idx = (fb_size > 50) ? 50 : (fb_size > 32 ? 32 : 10);
            // Ensure start_idx is even (submodule uses step-2)
            if (start_idx % 2 != 0) start_idx--;

            const mpqs::uint512& target = f_data_.a_target;
            mpqs::uint512 product((uint32_t)1);
            std::vector<uint32_t> new_factors;

            for (uint32_t i = start_idx; i < fb_size && new_factors.size() < 10; i += 2) {
                uint32_t p = f_data_.factorBase[i];
                mpqs::uint512 test = product;
                test.mult_uint32(p);

                if (test < target || new_factors.size() < 4) {
                    // Still below target or haven't reached minimum shc_dim
                    product = test;
                    new_factors.push_back(i);

                    // Stop if we've reached target and have enough factors
                    if (product >= target && new_factors.size() >= 4) break;
                } else {
                    // Adding this prime would overshoot and we already have ≥ 4
                    break;
                }
            }

            // Fallback: if FB is extremely small, fill from index 2 upward
            if (new_factors.size() < 4) {
                new_factors.clear();
                product = mpqs::uint512((uint32_t)1);
                for (uint32_t i = 2; i < fb_size && new_factors.size() < 6; i++) {
                    product.mult_uint32(f_data_.factorBase[i]);
                    new_factors.push_back(i);
                }
            }

            f_data_.a_factors = std::move(new_factors);
            shc_dim = static_cast<uint32_t>(f_data_.a_factors.size());
            f_data_.max_a_index = 1u << shc_dim;
            f_data_.current_a_index = 0;

            // Set sliding-window pivots (mirroring init_a_factors convention)
            uint32_t hi = f_data_.a_factors.back();
            if (shc_dim % 2 == 0) {
                f_data_.lowerHalfStart = hi;
                f_data_.upperHalfStart = hi + 2;
            } else {
                f_data_.lowerHalfStart = hi;
                f_data_.upperHalfStart = hi + 4;
            }
            // Clamp pivots to FB bounds
            if (f_data_.upperHalfStart >= fb_size)
                f_data_.upperHalfStart = fb_size - 1;

            // Recompute a and B_values from the new factor set
            recalc_a(&f_data_);

            // Log result
            std::string idx_str;
            for (size_t i = 0; i < f_data_.a_factors.size(); i++) {
                if (i) idx_str += ",";
                idx_str += std::to_string(f_data_.a_factors[i]);
            }
            LOG(LOG_STATS) << "Overridden a_factors: shc_dim=" << shc_dim
                          << " a=" << f_data_.a.to_string()
                          << " indices=[" << idx_str << "]";
        }
    }

    // --- Jetson: ensure shc_dim >= 5 to avoid polynomial exhaustion at 40d ---
    // shc_dim=4 produces only 2^(4-1)=8 polynomials per 'a'; deduplication
    // exhausts the pool before collecting enough relations.  Forcing shc_dim=5
    // gives 16 polynomials per 'a', resolving dedup exhaustion on small inputs.
    if (is_jetson_) {
        uint32_t shc_dim = static_cast<uint32_t>(f_data_.a_factors.size());
        if (shc_dim > 0 && shc_dim < 5) {
            const uint32_t fb_size = f_data_.factorBase.size();
            uint32_t next_idx = f_data_.a_factors.back() + 2;
            if (next_idx < fb_size) {
                LOG(LOG_STATS) << "Jetson: raising shc_dim from " << shc_dim
                              << " to 5 (polynomial exhaustion guard, adding FB index "
                              << next_idx << ")";
                f_data_.a_factors.push_back(next_idx);
                shc_dim = 5;
                f_data_.max_a_index     = 1u << shc_dim;
                f_data_.current_a_index = 0;
                uint32_t hi = f_data_.a_factors.back();
                f_data_.lowerHalfStart  = hi;
                f_data_.upperHalfStart  = (shc_dim % 2 == 0) ? hi + 2 : hi + 4;
                if (f_data_.upperHalfStart >= fb_size)
                    f_data_.upperHalfStart = fb_size - 1;
                recalc_a(&f_data_);
            }
        }
    }

    if (config_.target_relations == 0) {
        // Heuristic: FB Size + 5% margin + 64 (for Block Wiedemann blocking)
        config_.target_relations = f_data_.size + (f_data_.size / 20) + 64;
    }
    LOG(LOG_INFO) << "Target Relations: "
		  << config_.target_relations;
    LOG(LOG_STATS) << "Dedup safety factor: " << config_.dedup_safety_factor
                  << " (effective target: "
                  << static_cast<uint64_t>(std::ceil(config_.target_relations * config_.dedup_safety_factor))
                  << ")";

    // --- Jetson parameter defaults (runtime-guarded, respects CLI overrides) ---
    if (is_jetson_) {
        if (!config_.isPinned("sieve_batch_size"))
            config_.sieve_batch_size = 8;       // 8 vs 32 (fewer SMs)

        if (!config_.isPinned("sieve_gms_num_blocks"))
            config_.sieve_gms_num_blocks = 16;  // 16 vs 64 (8 SMs)

        // accum_buffer_size: no Jetson hardcode needed — the formula
        // max(4096, batch_size * 2048) in initPostProcessorConfig handles it
        // (bs=8 → 16K, vs old 131K hardcode).

        if (!config_.isPinned("lp1_bound") && config_.lp1_bound > 500000000ULL)
            config_.lp1_bound = 500000000ULL;    // Cap at 500M for 1 MB L2

        if (!config_.isPinned("lp1_max_witness_capacity"))
            config_.lp1_max_witness_capacity = 4194304;  // 4M vs 16M

        if (!config_.isPinned("lp1_hash_bits"))
            config_.lp1_hash_bits = 18;          // 256K rows vs 1M

        LOG(LOG_STATS) << "Jetson defaults applied: batch="
                      << config_.sieve_batch_size
                      << " gms=" << config_.sieve_gms_num_blocks
                      << " accum=" << config_.accum_buffer_size
                      << " LP=" << config_.lp1_bound
                      << " witness=" << config_.lp1_max_witness_capacity;
    }
}

// -----------------------------------------------------------------------------
// Sieve Progress Tracking — shared by batch and legacy loops
// -----------------------------------------------------------------------------

void MPQSOrchestrator::SieveProgressTracker::addSample(
    double elapsed_sec, uint64_t relation_count, uint64_t target)
{
    double y = static_cast<double>(relation_count);
    history.push_back({elapsed_sec, y});
    if (history.size() > WINDOW_K) history.pop_front();
    if (history.size() < 2) return;

    // --- 1. Instantaneous rate from last two samples ---
    const auto& prev = history[history.size() - 2];
    double dt = elapsed_sec - prev.t;
    double dy = y - prev.y;
    if (dt < 1e-6) return;
    double inst_rate = dy / dt;

    // --- 2. EMA-smoothed current rate (fallback for < 4 samples) ---
    if (smoothed_rate <= 0.0) smoothed_rate = inst_rate;
    else smoothed_rate = RATE_EMA_ALPHA * inst_rate + (1.0 - RATE_EMA_ALPHA) * smoothed_rate;

    // --- 3. Fit rate model r(t) = r0 + alpha*t via OLS on consecutive-sample rates ---
    double r0 = smoothed_rate;
    double alpha = 0.0;
    double mean_rate = smoothed_rate;

    if (history.size() >= 4) {
        double st = 0, sr = 0, str = 0, st2 = 0;
        size_t n = 0;
        for (size_t i = 1; i < history.size(); ++i) {
            double dti = history[i].t - history[i-1].t;
            if (dti < 1e-6) continue;
            double ri = (history[i].y - history[i-1].y) / dti;
            double ti = (history[i].t + history[i-1].t) * 0.5;  // midpoint time
            st += ti; sr += ri; str += ti * ri; st2 += ti * ti;
            ++n;
        }
        if (n >= 3) {
            double dn = static_cast<double>(n);
            mean_rate = sr / dn;
            double denom = dn * st2 - st * st;
            if (std::abs(denom) > 1e-12) {
                alpha = (dn * str - st * sr) / denom;
                r0 = (sr - alpha * st) / dn;  // OLS intercept
            } else {
                r0 = mean_rate;
            }
        }
    }

    // --- 4. Constrain α ≥ 0 (LP acceleration is non-negative) ---
    if (alpha < 0.0) {
        alpha = 0.0;
        r0 = mean_rate;
    }
    if (r0 <= 0.0) {
        r0 = smoothed_rate;
        alpha = 0.0;
    }
    rate_acceleration = alpha;

    // --- 5. Integration constant C via least-squares ---
    //     y(t) = C + r0*t + (alpha/2)*t^2
    //     C = mean(y_i - r0*t_i - (alpha/2)*t_i^2)
    double sum_residual = 0.0;
    for (const auto& pt : history) {
        sum_residual += pt.y - r0 * pt.t - 0.5 * alpha * pt.t * pt.t;
    }
    double C = sum_residual / static_cast<double>(history.size());
    integration_constant = C;

    // --- 6. Solve for ETA ---
    //     y_model(t_now + dT) = target
    //     => (alpha/2)*dT^2 + r_now*dT - remaining = 0
    //     where r_now = r0 + alpha*t_now, remaining = target - y_model_now
    double t_now = elapsed_sec;
    double tgt = static_cast<double>(target);
    double y_model_now = C + r0 * t_now + 0.5 * alpha * t_now * t_now;
    double remaining = tgt - y_model_now;
    if (remaining <= 0.0) { current_eta_sec = 0.0; return; }

    double r_now = r0 + alpha * t_now;
    double raw_eta;
    if (alpha > 1e-9) {
        // Quadratic: dT = (-r_now + sqrt(r_now^2 + 2*alpha*remaining)) / alpha
        double disc = r_now * r_now + 2.0 * alpha * remaining;
        raw_eta = (-r_now + std::sqrt(disc)) / alpha;
    } else {
        // Linear fallback: dT = remaining / r_now
        raw_eta = (r_now > 0.0) ? remaining / r_now : 0.0;
    }
    if (raw_eta <= 0.0) return;

    // --- 7. EMA-smooth the ETA output ---
    if (ema_eta <= 0.0) ema_eta = raw_eta;
    else ema_eta = ETA_EMA_ALPHA * raw_eta + (1.0 - ETA_EMA_ALPHA) * ema_eta;
    current_eta_sec = ema_eta;
}

namespace {
    double pct(uint64_t fill, uint64_t cap) {
        return cap > 0 ? 100.0 * fill / cap : 0.0;
    }
    std::string fmtSize(uint64_t v) {
        if (v >= 1'000'000) return std::to_string(v / 1'000'000) + "M";
        if (v >= 1'000)     return std::to_string(v / 1'000) + "K";
        return std::to_string(v);
    }
}

void MPQSOrchestrator::LPFillProjector::addSample(double elapsed_sec, uint64_t witness_count, uint64_t capacity) {
    history_.push_back({elapsed_sec, witness_count});
    last_capacity_ = capacity;
    while (history_.size() > WINDOW) history_.pop_front();
}

MPQSOrchestrator::LPFillProjector::Projection MPQSOrchestrator::LPFillProjector::project(double total_estimated_sec) const {
    Projection p{};
    if (history_.size() < 2 || last_capacity_ == 0) return p;

    // Linear least-squares: w(t) = alpha * t + beta
    // alpha = (N*sum(t*w) - sum(t)*sum(w)) / (N*sum(t^2) - sum(t)^2)
    size_t N = history_.size();
    double sum_t = 0, sum_w = 0, sum_tw = 0, sum_t2 = 0;
    for (const auto& s : history_) {
        double w = static_cast<double>(s.w);
        sum_t  += s.t;
        sum_w  += w;
        sum_tw += s.t * w;
        sum_t2 += s.t * s.t;
    }

    double denom = N * sum_t2 - sum_t * sum_t;
    if (std::abs(denom) < 1e-12) {
        // Degenerate (all samples at same time) — use latest value
        p.projected_witnesses = history_.back().w;
    } else {
        double alpha = (N * sum_tw - sum_t * sum_w) / denom;
        double beta  = (sum_w - alpha * sum_t) / N;

        double projected = alpha * total_estimated_sec + beta;
        p.projected_witnesses = projected > 0 ? static_cast<uint64_t>(projected) : 0;
    }

    p.projected_fill_pct = 100.0 * p.projected_witnesses / last_capacity_;
    p.overflow_likely = (p.projected_fill_pct > 95.0);
    p.oversized       = (p.projected_fill_pct < 20.0);

    // Recommended capacity: next power-of-2 of (projected * 1.25), clamped to [16, 2^28]
    uint64_t target = static_cast<uint64_t>(p.projected_witnesses * 1.25);
    if (target < 16) target = 16;
    uint64_t pow2 = 1;
    while (pow2 < target && pow2 < (1ULL << 28)) pow2 <<= 1;
    p.recommended_capacity = pow2;

    return p;
}

void MPQSOrchestrator::logSieveProgress(
    SieveProgressTracker& tracker,
    uint64_t rel_count, uint64_t target, double elapsed_sec,
    bool lp_active, uint64_t witnesses, uint64_t lp_full_rels)
{
    // Record sample and update ETA
    tracker.addSample(elapsed_sec, rel_count, target);

    // LP throughput line (only when LP is active and has data)
    LOG_SET_SUBMODULE("LP_Pipeline");
    if (lp_active && witnesses > 0) {
        double wit_per_sec  = (elapsed_sec > 0) ? (witnesses / elapsed_sec) : 0.0;
        double full_per_sec = (elapsed_sec > 0) ? (lp_full_rels / elapsed_sec) : 0.0;
        LOG(LOG_DEBUG_1) << "Thruput: " << std::fixed << std::setprecision(2)
                         << wit_per_sec << " witnesses/s | "
                         << full_per_sec << " full rels/s";
    }

    // Progress line
    LOG_SET_SUBMODULE("BatchLoop");
    double progress_pct = (target > 0) ? (100.0 * rel_count / target) : 0.0;
    double rel_per_sec = (elapsed_sec > 0) ? (static_cast<double>(rel_count) / elapsed_sec) : 0.0;
    LOG(LOG_STATS) << "Progress: " << rel_count << " / " << target
                  << " (" << std::fixed << std::setprecision(1) << progress_pct << "%) "
                  << "| Overall Thruput: " << std::setprecision(1) << rel_per_sec << " rel/s";

    // ETA line (suppressed until >= 3 data points, and not shown once target reached)
    if (tracker.hasETA() && rel_count < target) {
        LOG(LOG_STATS) << "ETA: " << std::fixed << std::setprecision(1)
                      << tracker.current_eta_sec << "s | Total est. time for stage 2: "
                      << FormatDuration((tracker.current_eta_sec + elapsed_sec) * 1000.0);
    }

    // --- Buffer fill snapshot (--verbose) ---
    LOG_SET_SUBMODULE("Buffers");
    {
        const auto& snap = last_buffer_snapshot_;
        LOG(LOG_STATS) << "  Accum: "
            << std::fixed << std::setprecision(0) << pct(snap.accum_fill, snap.accum_capacity)
            << "% (" << fmtSize(snap.accum_fill) << "/" << fmtSize(snap.accum_capacity)
            << ") | Partial: "
            << pct(snap.partial_fill, snap.partial_capacity)
            << "% (" << fmtSize(snap.partial_fill) << "/" << fmtSize(snap.partial_capacity)
            << ") | Persist: "
            << pct(snap.persistent_fill, snap.persistent_capacity)
            << "% (" << fmtSize(snap.persistent_fill) << "/" << fmtSize(snap.persistent_capacity)
            << ")";

        if (lp_active && largeprime_) {
            const auto* lp_stats = largeprime_->getTelemetry();
            if (lp_stats && lp_stats->total_witnesses > 0) {
                LOG(LOG_STATS) << "  Witnesses: "
                    << std::fixed << std::setprecision(0)
                    << pct(lp_stats->total_witnesses, lp_fill_history_.witness_capacity)
                    << "% (" << fmtSize(lp_stats->total_witnesses) << "/" << fmtSize(lp_fill_history_.witness_capacity)
                    << ") | Empty buckets: " << lp_stats->empty_hash_buckets
                    << " | Full buckets: " << lp_stats->full_hash_buckets;
            }
        }
    }

    // --- Historical buffer stats (--debug) ---
    {
        const auto& bh = buffer_fill_history_;
        const auto& snap = last_buffer_snapshot_;
        LOG(LOG_DEBUG_2) << "Accum: curr=" << std::fixed << std::setprecision(0)
                         << pct(snap.accum_fill, snap.accum_capacity)
                         << "% avg=" << pct((uint64_t)bh.accum_avg(), snap.accum_capacity)
                         << "% max=" << pct(bh.accum_max, snap.accum_capacity) << "%";
        LOG(LOG_DEBUG_2) << "Partial: curr="
                         << pct(snap.partial_fill, snap.partial_capacity)
                         << "% avg=" << pct((uint64_t)bh.partial_avg(), snap.partial_capacity)
                         << "% max=" << pct(bh.partial_max, snap.partial_capacity) << "%";
        if (lp_active) {
            const auto& lph = lp_fill_history_;
            LOG(LOG_DEBUG_2) << "Witnesses: avg="
                             << pct((uint64_t)lph.witness_avg(), lph.witness_capacity)
                             << "% max=" << pct(lph.witness_max, lph.witness_capacity)
                             << "% | Overflows: slab=" << lph.total_slab_overflows
                             << " witness=" << lph.total_witness_overflows
                             << " output=" << lph.total_output_overflows;
        }
    }

    // --- LP fill projection ---
    LOG_SET_SUBMODULE("LP_Projection");
    if (lp_active && lp_projector_.ready() && tracker.hasETA()
        && rel_count < target && rel_count >= target * 0.05) {
        double total_est_sec = tracker.current_eta_sec + elapsed_sec;
        auto proj = lp_projector_.project(total_est_sec);

        if (proj.overflow_likely) {
            // Always visible — potential data loss
            LOG(LOG_WARNING) << "LP witness table projected to reach "
                             << std::fixed << std::setprecision(1) << proj.projected_fill_pct
                             << "% by end of run (" << fmtSize(proj.projected_witnesses)
                             << " / " << fmtSize(lp_fill_history_.witness_capacity) << ")";
            LOG(LOG_WARNING) << "          Recommended for next run: --lp1_max_witnesses "
                             << fmtSize(proj.recommended_capacity);
        } else if (proj.oversized) {
            // INFO level — not a problem, just advisory
            LOG(LOG_STATS) << "LP witness table projected fill: "
                          << std::fixed << std::setprecision(1) << proj.projected_fill_pct
                          << "% — table may be oversized";
            LOG(LOG_STATS) << "          Recommended for next run: --lp1_max_witnesses "
                          << fmtSize(proj.recommended_capacity);
        } else {
            // Normal — verbose only
            LOG(LOG_DEBUG_1) << "Witness fill at ETA: "
                             << std::fixed << std::setprecision(1) << proj.projected_fill_pct
                             << "% (" << fmtSize(proj.projected_witnesses)
                             << " / " << fmtSize(lp_fill_history_.witness_capacity) << ")";
        }
    }

    LOG_SET_SUBMODULE("");
    tracker.last_logged_count = rel_count;
}

void MPQSOrchestrator::logSieveStageSummary(const SieveStageSummary& s) {
    const char* sep = "================================================================";

    LOG(LOG_STATS) << sep;
    LOG(LOG_STATS) << "  Sieve Stage Summary";
    LOG(LOG_STATS) << sep;

    double elapsed_sec = s.total_elapsed_ms / 1000.0;
    LOG(LOG_STATS) << "  Duration:           " << FormatDuration(s.total_elapsed_ms)
                  << "  (" << std::fixed << std::setprecision(1) << elapsed_sec << "s)";
    LOG(LOG_STATS) << "  Batches processed:  " << fmtNum(s.total_batches);

    // RELATIONS
    LOG(LOG_STATS) << "  RELATIONS";
    LOG(LOG_STATS) << "    Sieved full:      " << fmtNum(s.sieved_full_relations);
    if (s.lp_active) {
        LOG(LOG_STATS) << "    LP combined:      " << fmtNum(s.lp_combined_relations);
    }
    double target_pct = s.target_relations > 0
        ? 100.0 * s.total_full_relations / s.target_relations : 0.0;
    LOG(LOG_STATS) << "    Total (deduped):  " << fmtNum(s.total_full_relations)
                  << " / " << fmtNum(s.target_relations)
                  << " (" << std::fixed << std::setprecision(1) << target_pct << "%)";
    if (s.pre_dedup_relations > s.total_full_relations) {
        LOG(LOG_STATS) << "    Duplicates:       "
                      << fmtNum(s.pre_dedup_relations - s.total_full_relations);
    }

    // 1-PARTIALS (Large Prime) — only if LP active
    if (s.lp_active) {
        LOG(LOG_STATS) << "  1-PARTIALS (Large Prime)";
        LOG(LOG_STATS) << "    Unique witnesses:  " << fmtNum(s.unique_witnesses_stored)
                      << " / " << fmtNum(s.lp_witness_capacity)
                      << " (" << std::fixed << std::setprecision(1)
                      << s.witness_fill_pct << "%)";
        if (s.unique_witnesses_stored > 0) {
            double match_rate = 100.0 * s.lp_combined_relations / s.unique_witnesses_stored;
            LOG(LOG_STATS) << "    Match rate:        " << std::fixed << std::setprecision(1)
                          << match_rate << "%";
        }
    }

    // THROUGHPUT
    LOG(LOG_STATS) << "  THROUGHPUT";
    LOG(LOG_STATS) << "    Relations/s:      " << fmtNum(static_cast<uint64_t>(s.relations_per_sec));
    if (s.lp_active) {
        LOG(LOG_STATS) << "    LP witnesses/s:   " << fmtNum(static_cast<uint64_t>(s.lp_witnesses_per_sec));
        LOG(LOG_STATS) << "    LP matches/s:     " << fmtNum(static_cast<uint64_t>(s.lp_matches_per_sec));
    }

    // BUFFER PEAKS
    LOG(LOG_STATS) << "  BUFFER PEAKS";
    auto bufLine = [&](const char* label, uint64_t peak, uint64_t cap) {
        double p = cap > 0 ? 100.0 * peak / cap : 0.0;
        LOG(LOG_STATS) << "    " << label << std::fixed << std::setprecision(0) << p
                      << "% (" << fmtNum(peak) << " / " << fmtNum(cap) << ")";
    };
    bufLine("Accum:            ", s.accum_peak, s.accum_capacity);
    if (s.lp_active) {
        bufLine("Partial:          ", s.partial_peak, s.partial_capacity);
    }
    bufLine("Persistent:       ", s.persistent_peak, s.persistent_capacity);
    if (s.lp_active) {
        bufLine("LP witnesses:     ", s.unique_witnesses_stored, s.lp_witness_capacity);
    }

    // OVERFLOW EVENTS — only if any overflow occurred
    if (s.lp_active && (s.slab_overflows + s.witness_overflows + s.output_overflows) > 0) {
        LOG(LOG_STATS) << "  OVERFLOW EVENTS";
        if (s.slab_overflows > 0)
            LOG(LOG_STATS) << "    [!] Slab row overflows: " << fmtNum(s.slab_overflows);
        if (s.witness_overflows > 0)
            LOG(LOG_STATS) << "    [!] Witness overflows:  " << fmtNum(s.witness_overflows);
        if (s.output_overflows > 0)
            LOG(LOG_STATS) << "    [!] Output overflows:   " << fmtNum(s.output_overflows);
    }

    // PROJECTIONS — only if LP active and projector had data
    if (s.has_projection) {
        LOG(LOG_STATS) << "  PROJECTIONS";
        LOG(LOG_STATS) << "    LP witnesses at completion: "
                      << fmtNum(s.projected_witnesses) << " / " << fmtNum(s.lp_witness_capacity)
                      << " (" << std::fixed << std::setprecision(1) << s.projected_fill_pct << "%)";
        LOG(LOG_STATS) << "    Recommended capacity: --lp1_max_witnesses "
                      << fmtSize(s.recommended_capacity);
    }

    // RECOMMENDATIONS
    if (s.lp_active && s.witness_fill_pct > 85.0 && !s.has_projection) {
        LOG(LOG_STATS) << "  -> Recommended: --lp1_max_witnesses "
                      << fmtSize(s.lp_witness_capacity * 2);
    }

    LOG(LOG_STATS) << sep;
}

// -----------------------------------------------------------------------------
// Sieve Init Helpers (shared by SieveStage and TruncatedSieveRun)
// -----------------------------------------------------------------------------

postprocessing::PostProcConfig MPQSOrchestrator::initPostProcessorConfig(uint64_t accum_override) {
    postprocessing::PostProcConfig pp_conf;

    // Accum buffer: scale with batch_size.  Observed peaks are ~800 candidates/batch
    // on RTX (bs=32) and ~70/batch on Jetson (bs=8).  bs*2048 gives ~2× headroom
    // over the worst observed peak (25K at bs=32), targeting 30-50% fill.
    uint64_t auto_accum = std::max(uint64_t{4096}, (uint64_t)config_.sieve_batch_size * 2048);
    uint64_t accumulate_buffer_size = config_.accum_buffer_size > 0
        ? config_.accum_buffer_size
        : (accum_override > 0 ? accum_override : auto_accum);
    pp_conf.accumulate_buffer_size = static_cast<uint32_t>(accumulate_buffer_size);
    pp_conf.accumulate_batch_purge_threshold = (80 * pp_conf.accumulate_buffer_size) / 100;
    pp_conf.shc_dim = (uint32_t)f_data_.a_factors.size();
    pp_conf.device_id = config_.device_id;

    if (!config_.target_relations)
        config_.target_relations = f_data_.size + 128;

    // Scale accumulate buffer for small N to prevent massive over-collection.
    // Default 524K buffer lets the sieve overshoot by one full batch — catastrophic
    // when target_relations is O(1K) (e.g., 325K relations for target 1564 at 30d).
    if (config_.accum_buffer_size == 0 && accum_override == 0
        && config_.target_relations < 16384) {
        uint32_t buf = std::max(4096u, config_.target_relations * 4);
        // Round up to next power of 2.
        uint32_t p = 1;
        while (p < buf) p <<= 1;
        accumulate_buffer_size = p;
        pp_conf.accumulate_buffer_size = static_cast<uint32_t>(p);
        pp_conf.accumulate_batch_purge_threshold = (80 * p) / 100;
        LOG(LOG_STATS) << "Small-N buffer sizing: accumulate_buffer_size="
                      << p << " (target_relations=" << config_.target_relations << ")";
    }

    // Persistent buffer: target × 2 + accum.  The 2× margin covers LP overshoot
    // (~1.2%), dedup headroom (~5%), and last-batch overshoot.  Observed peak fill
    // is 44-49% across RSA-100/110.  Old formula was target×4+accum (LP) or
    // target+accum (no LP), wasting 77-84% of the allocation.
    if (config_.persistent_buffer_size > 0) {
        pp_conf.persistent_device_buffer_size = static_cast<uint32_t>(config_.persistent_buffer_size);
    } else {
        pp_conf.persistent_device_buffer_size = config_.target_relations * 2
            + pp_conf.accumulate_buffer_size;
    }

    pp_conf.lp1_bound = config_.lp1_bound;

    if (config_.lp1_bound > 0) {
        uint32_t partial_from_config = config_.partial_buffer_size > 0
            ? static_cast<uint32_t>(config_.partial_buffer_size)
            : pp_conf.accumulate_buffer_size;
        if (partial_from_config < pp_conf.accumulate_buffer_size) {
            LOG(LOG_WARNING) << "partial_buffer_size=" << partial_from_config
                             << " < accum_buffer_size=" << pp_conf.accumulate_buffer_size
                             << " — clamping to " << pp_conf.accumulate_buffer_size;
            partial_from_config = pp_conf.accumulate_buffer_size;
        }
        pp_conf.partial_buffer_size = partial_from_config;
    }

    return pp_conf;
}

void MPQSOrchestrator::initLargePrimes() {
    if (config_.lp1_bound == 0) {
        LOG(LOG_DEBUG_2) << "Init skipped.";
        return;
    }

    LOG(LOG_DEBUG_1) << "Initializing with lp1 bound " << config_.lp1_bound;
    lp::LargePrimeConfig lp_conf;
    lp_conf.device_id = config_.device_id;

    lp_conf.max_combined_output = config_.lp1_max_combined_output > 0
        ? static_cast<uint32_t>(config_.lp1_max_combined_output)
        : (1u << 15);

    if (!config_.lp1_max_witness_capacity) {
        uint64_t base = (uint64_t)f_data_.size * 4;

        // Scale with LP-to-FB ratio: wider LP range produces more unique witnesses
        if (config_.lp1_bound > 0 && f_data_.F > 0) {
            double lp_ratio = static_cast<double>(config_.lp1_bound) / f_data_.F;
            double lp_mult = std::min(4.0, std::max(1.0,
                std::log2(std::max(1.0, lp_ratio / 10.0))));
            base = static_cast<uint64_t>(base * lp_mult);
        }

        uint64_t scaled = std::max(base, uint64_t{1} << 18);
        int log_2 = 64 - __builtin_clzll(scaled - 1);
        log_2 = std::min(log_2, 24);
        lp_conf.max_witness_capacity = 1ULL << log_2;
    } else {
        int log_2 = 63 - __builtin_clzll(config_.lp1_max_witness_capacity);
        uint64_t value = 1ULL << log_2;
        if (value < config_.lp1_max_witness_capacity) log_2++;
        if (log_2 < 4) log_2 = 4;
        lp_conf.max_witness_capacity = 1ULL << log_2;
    }

    if (config_.lp1_hash_bits > 0) {
        lp_conf.hash_bits = config_.lp1_hash_bits;
    } else {
        int log_cap = 63 - __builtin_clzll(lp_conf.max_witness_capacity);
        lp_conf.hash_bits = (log_cap > 4) ? (log_cap - 4) : 4;
    }

    LOG(LOG_DEBUG_1) << "LP config resolved:"
                     << " witnesses=" << lp_conf.max_witness_capacity
                     << " hash_bits=" << lp_conf.hash_bits
                     << " combined_output=" << lp_conf.max_combined_output;

    lp_conf.lp1_bound = config_.lp1_bound;

    // For small factor bases, purge witnesses after first match to prevent
    // one witness generating multiple combined relations (degenerate kernel vectors).
    lp_conf.purge_after_match = (f_data_.size < 20000);

    config_.lp_config = lp_conf;
    largeprime_ = std::make_unique<mpqs::lp::LargePrimeVariant>(postprocessor_->getCudaStream());
    largeprime_->initiate(config_.lp_config, config_.N);
}

void MPQSOrchestrator::logBufferWarnings() {
    // === Near-full warnings (fire once per buffer) ===
    if (!warned_accum_near_full_) {
        double accum_pct = pct(last_buffer_snapshot_.accum_fill, last_buffer_snapshot_.accum_capacity);
        if (accum_pct >= 90.0) {
            LOG(LOG_WARNING) << "Accumulation buffer at " << std::fixed << std::setprecision(1)
                             << accum_pct << "% (" << last_buffer_snapshot_.accum_fill
                             << "/" << last_buffer_snapshot_.accum_capacity
                             << "). Consider --accum_buf_size " << (last_buffer_snapshot_.accum_capacity * 2);
            warned_accum_near_full_ = true;
        }
    }
    if (!warned_partial_near_full_) {
        double partial_pct = pct(last_buffer_snapshot_.partial_fill, last_buffer_snapshot_.partial_capacity);
        if (partial_pct >= 80.0) {
            LOG(LOG_WARNING) << "Partial staging buffer at " << std::fixed << std::setprecision(1)
                             << partial_pct << "% (" << last_buffer_snapshot_.partial_fill
                             << "/" << last_buffer_snapshot_.partial_capacity
                             << "). Consider --partial_buf_size " << (last_buffer_snapshot_.partial_capacity * 2);
            warned_partial_near_full_ = true;
        }
    }
    if (!warned_persistent_near_full_) {
        double persist_pct = pct(last_buffer_snapshot_.persistent_fill, last_buffer_snapshot_.persistent_capacity);
        if (persist_pct >= 95.0) {
            LOG(LOG_WARNING) << "Persistent buffer at " << std::fixed << std::setprecision(1)
                             << persist_pct << "% (" << last_buffer_snapshot_.persistent_fill
                             << "/" << last_buffer_snapshot_.persistent_capacity
                             << "). Relation collection may stall.";
            warned_persistent_near_full_ = true;
        }
    }
    if (!warned_witness_near_full_ && config_.lp1_bound > 0 && largeprime_) {
        const auto* lp_stats = largeprime_->getTelemetry();
        if (lp_stats) {
            double wit_pct = pct(lp_stats->total_witnesses, lp_fill_history_.witness_capacity);
            if (wit_pct >= 85.0) {
                LOG(LOG_WARNING) << "LP witness buffer at " << std::fixed << std::setprecision(1)
                                 << wit_pct << "% (" << lp_stats->total_witnesses
                                 << "/" << lp_fill_history_.witness_capacity
                                 << "). Consider --lp1_max_witnesses " << (lp_fill_history_.witness_capacity * 2);
                warned_witness_near_full_ = true;
            }
        }
    }

    // === Overflow detection warnings (LP, fire on every new delta) ===
    if (config_.lp1_bound > 0 && largeprime_) {
        const auto* lp_stats = largeprime_->getTelemetry();
        if (lp_stats) {
            if (lp_stats->slab_overflow_count > last_reported_slab_overflows_) {
                uint64_t delta = lp_stats->slab_overflow_count - last_reported_slab_overflows_;
                if (!warned_slab_overflow_) {
                    LOG(LOG_WARNING) << "LP hash table: " << delta
                                     << " slab row overflows (cumulative: " << lp_stats->slab_overflow_count
                                     << ") — large primes silently lost. Consider --lp1_hash_bits "
                                     << (config_.lp_config.hash_bits + 2)
                                     << " or --lp1_max_witnesses " << fmtSize(config_.lp_config.max_witness_capacity * 2);
                    warned_slab_overflow_ = true;
                } else {
                    LOG(LOG_DEBUG_1) << "LP hash table: " << delta
                                     << " slab row overflows (cumulative: " << lp_stats->slab_overflow_count << ")";
                }
                last_reported_slab_overflows_ = lp_stats->slab_overflow_count;
            }
            if (lp_stats->witness_overflow_count > last_reported_witness_overflows_) {
                uint64_t delta = lp_stats->witness_overflow_count - last_reported_witness_overflows_;
                if (!warned_witness_overflow_) {
                    LOG(LOG_WARNING) << "LP witness buffer: " << delta
                                     << " overflows (cumulative: " << lp_stats->witness_overflow_count
                                     << ") — relations silently dropped. Increase --lp1_max_witnesses";
                    warned_witness_overflow_ = true;
                } else {
                    LOG(LOG_DEBUG_1) << "LP witness buffer: " << delta
                                     << " overflows (cumulative: " << lp_stats->witness_overflow_count << ")";
                }
                last_reported_witness_overflows_ = lp_stats->witness_overflow_count;
            }
            if (lp_stats->output_overflow_count > last_reported_output_overflows_) {
                uint64_t delta = lp_stats->output_overflow_count - last_reported_output_overflows_;
                if (!warned_output_overflow_) {
                    LOG(LOG_WARNING) << "LP combined output: " << delta
                                     << " overflows (cumulative: " << lp_stats->output_overflow_count
                                     << ") — matches lost. Increase --lp1_combined_buf";
                    warned_output_overflow_ = true;
                } else {
                    LOG(LOG_DEBUG_1) << "LP combined output: " << delta
                                     << " overflows (cumulative: " << lp_stats->output_overflow_count << ")";
                }
                last_reported_output_overflows_ = lp_stats->output_overflow_count;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Stage 2: Sieve
// -----------------------------------------------------------------------------

void MPQSOrchestrator::SieveStage() {
#ifdef SIEVING_DEBUG_FLAG
    JSON_IO j_io;
#endif
    LOG_SET_MODULE("Sieve");
    LOG_SET_STAGE(LOG_STAGE_SIEVE, "Sieve");
    LOG(LOG_INFO) << "Sieve";
    using clock = std::chrono::high_resolution_clock;
    auto t_sieve_start = clock::now();

    // --- Cluster mode setup ---
    data_tap_ = config_.data_tap;  // nullptr in solo mode — all DataTap code is no-op
    const bool cluster_mode = (config_.cluster_mode != ClusterMode::SOLO);

    // =========================================================================
    // 1. Initialize Siever
    // =========================================================================

    // TODO(jetson): dev_factorBase and dev_rootN are allocated inside the sieve
    // submodule (DeviceSievingController::initiate/loadData). Converting to
    // cudaMallocManaged saves ~56 MB on Jetson. Deferred to submodule update.
    LOG(LOG_DEBUG_1) << "Initializing on GPU device " << config_.device_id;
    // Use a non-default stream for the siever. This enables:
    // - CUDA graph capture (requires non-default stream)
    // - True async overlap between sieve and postprocessor streams
    cudaStream_t sieve_stream;
    CUDA_CHECK(cudaStreamCreate(&sieve_stream));
    siever_ = std::make_unique<mpqs::sieve::DeviceSievingController>(config_.device_id, sieve_stream);
    siever_->initiate(f_data_);
    // M3: wire external_stop for sub-batch stop latency (cluster mode)
    if (config_.cluster_mode != ClusterMode::SOLO && siever_) {
        siever_->setExternalStop(&external_stop_flag_);
    }
    if (config_.lp1_bound > 0)
        siever_->setThresholdOverride(config_.lp1_bound);

    // NOTE: setSievingBatchSize must be called AFTER loadStandardConfig/loadPartialCustomConfig
    // because those functions reset init_conf.batch_size = 0. Moved below config loading.

    // PARAM_TEST: exhaustive grid search, then return to Run()
    if(config_.mode == ExecutionMode::PARAM_TEST) {
        LOG(LOG_INFO) << "Testing parameter combinations...";
        siever_->loadStandardConfig();
        siever_->loadData();
        siever_->updateState();
        auto result = siever_->runParamTest(f_data_);
        LOG(LOG_INFO) << "Param test complete. Best timing: "
                      << result.best_timing_us << " us ("
                      << result.configs_tested << " configs tested)";
        return;
    }

    // Belt-and-suspenders sasGridDim floor (preflight auto-corrector is the primary fix).
    // Compute minimum from sieve geometry: work_units / maxRelationsPerBlock, rounded up to pow2.
    // Skip in cluster WORKER mode: workers don't run the GPU LP hash table, so the candidate
    // buffer pressure model differs; raising sasGridDim would violate num_sievingBlockBatches > 0
    // when autotune params were tuned without LP (blocksPerCycle < new sasGridDim).
    // COORDINATOR skips (CPU LP only — GPU LP hash table not active on coordinator).
    if (config_.lp1_bound > 0 && config_.useParams &&
        config_.cluster_mode != ClusterMode::COORDINATOR) {
        constexpr uint32_t maxRelationsPerBlock = 64;
        uint32_t work_units = config_.params[0] * config_.params[1]; // subCubeSize * numIntervals
        uint32_t min_sas = (work_units + maxRelationsPerBlock - 1) / maxRelationsPerBlock;
        // Round up to next power of 2
        if (min_sas > 0) { min_sas--; min_sas |= min_sas >> 1; min_sas |= min_sas >> 2;
            min_sas |= min_sas >> 4; min_sas |= min_sas >> 8; min_sas |= min_sas >> 16; min_sas++; }
        if (config_.params[6] < min_sas) {
            LOG(LOG_WARNING) << "LP active: raising sasGridDim from "
                             << config_.params[6] << " to " << min_sas << " (candidate buffer safety)";
            config_.params[6] = min_sas;
        }
    }

    // Safety cap matching __launch_bounds__(1024) on sieveAndScanKernel (mirrors preflight).
    constexpr uint32_t LEGACY_SAS_MAX_THREADS = 1024;
    if (config_.sieve_batch_size == 0 && config_.params[7] > LEGACY_SAS_MAX_THREADS) {
        LOG(LOG_WARNING) << "Capping sasBlockDim from "
                         << config_.params[7] << " to " << LEGACY_SAS_MAX_THREADS
                         << " (__launch_bounds__ safety net for legacy kernel)";
        config_.params[7] = LEGACY_SAS_MAX_THREADS;
    }

    // Configuration loading — adaptive for small-N, custom params, or standard
    uint32_t shc_dim = static_cast<uint32_t>(f_data_.a_factors.size());
    uint32_t max_polys = (1u << shc_dim) / 2;

    if (!config_.useParams && max_polys < 64) {
        // --- Small-N adaptive config ---
        // loadStandardConfig() sets metaB from GPU SM count, which exceeds
        // max_polys for small N, causing num_polyBlocksPerThreadBlock = 0.
        // Compute a safe 8-parameter config instead.
        //
        // The 8-parameter tuple for loadPartialCustomConfig:
        //   (totalPolys, totalIntervals, polyBlockSize, blocksPerCycle,
        //    metaB, metaT, sasB, sasT)

        // Round max_polys down to power of 2
        uint32_t totalPolys = 1;
        while (totalPolys * 2 <= max_polys) totalPolys *= 2;

        // Compute sievingBlockSize (replicates submodule logic from loadStandardConfig)
        uint32_t sievingBlockSize = 1;
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, config_.device_id);
            uint32_t target = (3 * static_cast<uint32_t>(prop.sharedMemPerBlock)) / 4;
            while (sievingBlockSize * 2 <= target) sievingBlockSize *= 2;
        }

        // totalIntervals = (2*M) / sievingBlockSize, floored to power-of-2, min 1
        uint32_t totalIntervals = (2 * f_data_.M) / sievingBlockSize;
        if (totalIntervals < 1) totalIntervals = 1;
        { uint32_t p = 1; while (p * 2 <= totalIntervals) p *= 2; totalIntervals = p; }

        uint32_t metaB = totalPolys;       // Key fix: metaB <= totalPolys
        uint32_t polyBlockSize = 1;        // Safe: minimizes constraint
        uint32_t blocksPerCycle = totalIntervals;  // Process all intervals at once

        LOG(LOG_STATS) << "Small-N adaptive config: shc_dim=" << shc_dim
                      << " totalPolys=" << totalPolys << " totalIntervals=" << totalIntervals
                      << " metaB=" << metaB;

        uint32_t sasB = std::min(256u, totalPolys);  // sasB must be ≤ totalPolys
        siever_->loadPartialCustomConfig(
            totalPolys, totalIntervals, polyBlockSize, blocksPerCycle,
            metaB, 256, sasB, 256);
    } else if (config_.useParams) {
        LOG(LOG_DEBUG_2) << "Loading Custom Config";

        // Post-autotune safety clamp: metaB (params[4]) and sasB (params[6])
        // must not exceed totalPolys. Autotune_stage1 may set these to 32+ even
        // when totalPolys is small (e.g., 2-16 for 60d composites), causing
        // totalPolys / gridDim = 0 iterations per block → zero candidates → hang.
        uint32_t max_polys_check = (1u << shc_dim) / 2;
        if (max_polys_check > 0) {
            bool clamped = false;
            if (config_.params[4] > max_polys_check) {
                LOG(LOG_WARNING) << "Clamping metaGridDim from "
                                 << config_.params[4] << " to " << max_polys_check
                                 << " (totalPolys safety for small N)";
                config_.params[4] = max_polys_check;
                clamped = true;
            }
            if (config_.params[6] > max_polys_check) {
                LOG(LOG_WARNING) << "Clamping sasGridDim from "
                                 << config_.params[6] << " to " << max_polys_check
                                 << " (totalPolys safety for small N)";
                config_.params[6] = max_polys_check;
                clamped = true;
            }
            if (clamped) {
                LOG(LOG_STATS) << "Reloading config with clamped grid dims";
            }
        }

        siever_->loadPartialCustomConfig(config_.params[0],
                                         config_.params[1],
                                         config_.params[2],
                                         config_.params[3],
                                         config_.params[4],
                                         config_.params[5],
                                         config_.params[6],
                                         config_.params[7]);
    } else {
        LOG(LOG_DEBUG_2) << "Loading Standard Config";
        siever_->loadStandardConfig();
    }

    // Set batch size AFTER config loading (loadStandardConfig resets init_conf.batch_size = 0)
    if(config_.sieve_batch_size > 0) {
        LOG(LOG_DEBUG_1) << "Batch Sieving with batch size " << config_.sieve_batch_size << " activated.";
        siever_->setSievingBatchSize(config_.sieve_batch_size);
    }

    siever_->printConfigs();
    if (!(siever_->validateConfigs())) {
        LOG(LOG_ERROR_CRITICAL) << "Configuration invalid!";
        throw std::runtime_error("Sieve config validation failed");
    }
    siever_->loadData();
    siever_->updateState();

    // Preflight guard: reject infeasible kernel configs before sieve loop
    if (config_.useParams) {
        auto pf = mpqs::autotune::preflightKernelLaunch(config_,
            static_cast<uint32_t>(f_data_.a_factors.size()), f_data_.M);
        if (!pf.feasible) {
            LOG(LOG_ERROR_CRITICAL) << "Kernel launch preflight FAILED: " << pf.reason;
            throw std::runtime_error("SieveStage preflight: " + pf.reason);
        }
        LOG(LOG_DEBUG_1) << "Kernel launch preflight passed";
    }

    if(config_.sieve_batch_size) {
        siever_->allocateBatchBuffers();
    }

    // =========================================================================
    // 2. Initialize Post-Processor
    // =========================================================================

    {
        LOG(LOG_DEBUG_1) << "Initializing.";
        auto pp_conf = initPostProcessorConfig();

        LOG(LOG_DEBUG_1) << "Buffer sizes resolved:"
                         << " accum=" << pp_conf.accumulate_buffer_size
                         << " persistent=" << pp_conf.persistent_device_buffer_size
                         << " partial=" << pp_conf.partial_buffer_size;

        config_.pp_config = pp_conf;
    }

    postprocessor_ = std::make_unique<mpqs::postprocessing::DevicePostProcessingController>();
    postprocessor_->initiate(f_data_, siever_->getDevicePointers(), config_.pp_config);

    // =========================================================================
    // 3. Initialize Large Prime Variant (shared: serves both batch and legacy paths)
    // =========================================================================

    if (!cluster_mode) {
        initLargePrimes();
    }

    // --- Buffer Size Validation ---
    {
        auto& pp = config_.pp_config;
        if (pp.accumulate_buffer_size < 1024) {
            LOG(LOG_WARNING) << "accum_buffer_size=" << pp.accumulate_buffer_size
                             << " is very small (min recommended: 1024)";
        }
        if (pp.lp1_bound > 0 && pp.partial_buffer_size < pp.accumulate_buffer_size) {
            LOG(LOG_WARNING) << "Auto-correcting partial_buffer_size from "
                             << pp.partial_buffer_size << " to " << pp.accumulate_buffer_size;
            pp.partial_buffer_size = pp.accumulate_buffer_size;
        }
        if (pp.persistent_device_buffer_size > 0 &&
            pp.persistent_device_buffer_size < config_.target_relations) {
            uint32_t corrected = config_.target_relations + pp.accumulate_buffer_size;
            LOG(LOG_WARNING) << "persistent_buffer_size=" << pp.persistent_device_buffer_size
                             << " < target_relations=" << config_.target_relations
                             << " — clamping to " << corrected;
            pp.persistent_device_buffer_size = corrected;
        }
        if (config_.lp1_bound > 0) {
            auto& lp = config_.lp_config;
            if (lp.hash_bits < 10 || lp.hash_bits > 28) {
                LOG(LOG_WARNING) << "lp1_hash_bits=" << lp.hash_bits
                                 << " outside recommended range [10, 28]";
            }
        }
    }

    logGpuMemory("After sieve/LP init");

    // =========================================================================
    // 3b. M3 Cluster: Jump to assigned poly range (workers only)
    // =========================================================================
    if (config_.poly_range_start > 0 && siever_) {
        if (config_.cluster_mode == ClusterMode::WORKER) {
            siever_->saveSnapshot();
            siever_->resetAndAdvanceTo(config_.poly_range_start);
            LOG(LOG_INFO) << "Siever: jumped to a_index=" << config_.poly_range_start;
        }
    }

    // =========================================================================
    // 4. Prepare Buffers & Telemetry State
    // =========================================================================

    host_relations_soa_.clear();
    auto t_sieve_loop_start = clock::now();
    uint32_t current_step = 0;

    LOG_SET_STAGE(LOG_STAGE_SIEVE_SIEVING, "Sieve");
    LOG(LOG_INFO) << "Init complete. Target: " << config_.target_relations;

    // =========================================================================
    // 5. Sieve Loop — Three-Path Dispatch
    // =========================================================================

    // === Truncation limit check (early sieve termination) ===
    auto truncation_limit_reached = [&](uint64_t rel_count, uint64_t batch_count) -> bool {
        if (config_.sieve_max_relations > 0 && rel_count >= config_.sieve_max_relations) return true;
        if (config_.sieve_max_batches > 0 && batch_count >= config_.sieve_max_batches) return true;
        return false;
    };

    // === Cluster extraction state (only when DataTap is active) ===
    cudaStream_t extract_stream = nullptr;
    cudaEvent_t  batch_done_event = nullptr;
    cudaEvent_t  extract_done_event[2] = {nullptr, nullptr};
    cudaEvent_t  partial_reset_event = nullptr;
    uint64_t&    prev_pers_count = cluster_prev_pers_count_;
    int          active_staging = 0;
    bool         extract_pending = false;
    structures::HostRelationBatch staging_full[2], staging_part[2];

    if (data_tap_) {
        CUDA_CHECK(cudaStreamCreate(&extract_stream));
        CUDA_CHECK(cudaEventCreateWithFlags(&batch_done_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&extract_done_event[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&extract_done_event[1], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&partial_reset_event, cudaEventDisableTiming));
        // Pre-allocate staging buffers
        uint64_t max_rels = config_.pp_config.accumulate_buffer_size;
        uint64_t max_factors = max_rels * 32;  // conservative average factors per relation
        for (int i = 0; i < 2; i++) {
            staging_full[i].resize(max_rels, max_factors);
            staging_part[i].resize(config_.pp_config.partial_buffer_size,
                                   config_.pp_config.partial_buffer_size * 32);
        }
    }

    if (config_.sieve_batch_size > 0) {
        // =================================================================
        // PATH 1: Batch Sieve Loop (Zero-Sync GPU Pipeline)
        // =================================================================
        LOG(LOG_INFO) << "Entering Non-Blocking Batch Sieving State Machine (BatchSize=" << config_.sieve_batch_size << ")";

        volatile uint32_t* pinned_counter = postprocessor_->h_pinned_accumulation_counter;

        // DAG Initialization: Enforce write-safety on the initial buffer
        auto* init_buffer = postprocessor_->getActiveAccumulationBuffer();
        cudaStreamWaitEvent(siever_->getCudaStream(), init_buffer->safe_to_write_event, 0);

        int processed_batches = 0;
        uint64_t global_lp_full_relations = 0;
        SieveProgressTracker progress_tracker;

        // --- Buffer fill history tracking (member variables) ---
        uint64_t last_fill_gen = 0;

        // --- LP fill history tracking (member variables) ---
        if (config_.lp1_bound > 0 && largeprime_)
            lp_fill_history_.witness_capacity = largeprime_->getWitnessCapacityRels();
        uint64_t last_lp_gen = 0;

        // LP batch interval: process partials periodically (every N batches).
        // Uses adaptive_lp_batch_interval_ (member, initially 10, calibrated at 5%/20% progress).

        bool lp_async_pending = false;  // True if an async LP batch is in flight

        // --- Adaptive convergence: configure prediction & device-side cap ---
        // Use dedup_safety_factor for oversieve margin (matches cluster mode accumulator).
        // dedup_safety_factor=1.05 → 5% margin (default), 1.70 → 70% oversieve, etc.
        uint32_t dedup_margin = std::max(256u,
            (uint32_t)(config_.target_relations * (config_.dedup_safety_factor - 1.0)));
        uint32_t relation_cap = config_.target_relations + dedup_margin;
        postprocessor_->getPersistentBatch()->setTargetCap(relation_cap);
        LOG(LOG_DEBUG_1) << "Relation cap: " << relation_cap
                        << " (target=" << config_.target_relations
                        << ", margin=" << dedup_margin << ")";

        const bool has_lp_stats = (largeprime_ != nullptr);
        postprocessor_->setPredictionParams(
            config_.target_relations,
            has_lp_stats ? largeprime_->getTelemetry() : nullptr);
        LOG(LOG_DEBUG_1) << "Params set: target=" << config_.target_relations
                         << ", lp_stats=" << (has_lp_stats ? "yes" : "no");

        // === CUDA Graph staging (if enabled) ===
        LOG_SET_SUBMODULE("Graph");
        const uint32_t graph_N = config_.cuda_graph_unroll;
        const char* launch_blocking_env = std::getenv("CUDA_LAUNCH_BLOCKING");
        const bool graph_blocked = launch_blocking_env && std::string(launch_blocking_env) == "1";
        bool use_graph = (graph_N > 0) && !graph_blocked;

        // Guard: legacy sieve mode (defensive — already inside batch path)
        if (use_graph && config_.sieve_batch_size == 0) {
            LOG(LOG_WARNING) << "--cuda_graph_unroll requires batch sieve mode. Disabling graph.";
            use_graph = false;
        }
        // Guard: truncated sieve with fewer batches than graph unroll
        if (use_graph && config_.sieve_max_batches > 0 && config_.sieve_max_batches < graph_N) {
            LOG(LOG_INFO) << "sieve_max_batches (" << config_.sieve_max_batches
                          << ") < graph_unroll (" << graph_N << ") — disabling graph.";
            use_graph = false;
        }
        // Guard: small target — graph capture overhead exceeds benefit
        if (use_graph && config_.target_relations < graph_N * 50) {
            LOG(LOG_DEBUG_1) << "Target too small (" << config_.target_relations
                             << " relations) — skipping graph capture";
            use_graph = false;
        }
        // Jetson: warn about experimental support
        if (use_graph && is_jetson_) {
            LOG(LOG_WARNING) << "CUDA graph on Jetson — experimental. "
                             << "Disable with --cuda_graph_unroll 0 if issues arise.";
        }
        // LP support: always-launch model (S2). LP kernels at every batch position.

        // Staging arrays for graph path
        std::vector<uint32_t*> h_staged_indices(graph_N, nullptr);
        std::vector<uint32_t*> d_staged_indices(graph_N, nullptr);
        std::vector<mpqs::uint512*> d_staged_a(graph_N, nullptr);
        std::vector<mpqs::uint512*> d_staged_B(graph_N, nullptr);

        const uint32_t shc_dim = static_cast<uint32_t>(f_data_.a_factors.size());
        const uint32_t batch_size = config_.sieve_batch_size;
        const size_t indices_stride = batch_size * shc_dim;
        const size_t indices_bytes  = indices_stride * sizeof(uint32_t);
        const size_t a_bytes        = batch_size * sizeof(mpqs::uint512);
        const size_t B_bytes        = batch_size * shc_dim * sizeof(mpqs::uint512);

        if (use_graph) {
            for (uint32_t i = 0; i < graph_N; i++) {
                cudaMallocHost(&h_staged_indices[i], indices_bytes);
                cudaMalloc(&d_staged_indices[i], indices_bytes);
                cudaMalloc(&d_staged_a[i], a_bytes);
                cudaMalloc(&d_staged_B[i], B_bytes);
            }
            LOG(LOG_STATS) << "Allocated " << graph_N << " staging slots ("
                          << (graph_N * (indices_bytes + a_bytes + B_bytes)) / 1024 << " KB device)";
        } else if (graph_N > 0 && graph_blocked) {
            LOG(LOG_STATS) << "Disabled: CUDA_LAUNCH_BLOCKING=1";
        }

        cudaGraph_t cuda_graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        bool graph_ran = false;
        uint64_t graph_replay_count = 0;

        if (use_graph) {
            // =============================================================
            // GRAPH-BASED SIEVE LOOP (single-stream capture + between-replay LP)
            //
            // Captures ONLY sieve kernels on the sieve stream (single-stream
            // graph). Postproc and LP run between graph replays on their own
            // streams, avoiding multi-stream graph capture issues (D2H memcpy
            // nodes in captured postproc/LP corrupt forked stream state).
            //
            // LP uses always-launch model: runs once per graph replay
            // (every graph_N batches), processing all accumulated partials.
            // =============================================================

            // Save original device pointers (restored after graph loop to
            // prevent double-free in siever destructor/clearSievingBuffers)
            auto orig_dp = siever_->getDevicePointers();

            mpqs::sieve::factoringData& sieve_fdata = siever_->getFactoringDataRef();

            // Use a fixed accumulation buffer for all graph batches
            auto* graph_buffer = postprocessor_->getActiveAccumulationBuffer();
            siever_->setPostProcessingLinks(graph_buffer);

            // --- Pre-compute initial N batches ---
            for (uint32_t i = 0; i < graph_N; i++) {
                auto next_indices = mpqs::sieve::prepareNextBatchIndices(&sieve_fdata, batch_size);
                memcpy(h_staged_indices[i], next_indices.data(), indices_bytes);
                cudaMemcpyAsync(d_staged_indices[i], h_staged_indices[i],
                                indices_bytes, cudaMemcpyHostToDevice, siever_->getCudaStream());
                siever_->prepareSievingBatchFromStaged(
                    d_staged_indices[i], d_staged_a[i], d_staged_B[i]);
            }
            cudaStreamSynchronize(siever_->getCudaStream());

            // --- Graph Capture: N sieve batches (single stream, no postproc/LP) ---
            cudaStreamBeginCapture(siever_->getCudaStream(), cudaStreamCaptureModeThreadLocal);

            for (uint32_t i = 0; i < graph_N; i++) {
                siever_->setJobArrays(d_staged_a[i], d_staged_B[i], d_staged_indices[i]);
                siever_->runSievingBatch(batch_size, 0);
            }

            cudaError_t end_err = cudaStreamEndCapture(siever_->getCudaStream(), &cuda_graph);
            if (end_err != cudaSuccess) {
                LOG(LOG_ERROR_CRITICAL) << "cudaStreamEndCapture failed: "
                                        << cudaGetErrorString(end_err)
                                        << " — falling back to standard loop";
                cuda_graph = nullptr;
            }

            if (cuda_graph) {
                cudaError_t graph_err = cudaGraphInstantiate(&graph_exec, cuda_graph, 0);
                cudaGraphDestroy(cuda_graph);
                cuda_graph = nullptr;

                if (graph_err != cudaSuccess) {
                    LOG(LOG_ERROR_CRITICAL) << "cudaGraphInstantiate failed: "
                                            << cudaGetErrorString(graph_err)
                                            << " — falling back to standard loop";
                    graph_exec = nullptr;
                } else {
                    LOG(LOG_DEBUG_1) << "Captured " << graph_N << "-batch graph"
                                    << (largeprime_ ? " (with LP between replays)" : "");
                    LOG(LOG_DEBUG_1) << "Configuration: unroll=" << graph_N
                                    << ", batch_size=" << batch_size
                                    << ", LP=" << (largeprime_ ? "between-replay" : "off");
                }
            }

            if (graph_exec) {
                graph_ran = true;
                bool lp_graph_pending = false;  // LP async state for graph path

                // --- Graph Replay Loop ---
                while (*postprocessor_->h_pinned_persistent_count < config_.target_relations &&
                       !(postprocessor_->h_prediction_result &&
                         postprocessor_->h_prediction_result->should_terminate) &&
                       !truncation_limit_reached(*postprocessor_->h_pinned_persistent_count, processed_batches) &&
                       !(data_tap_ && data_tap_->shouldStop())) {

                    // Adaptive fallback: exit graph mode when < 2 replays of work remain
                    // (keeps graph running to ~98.5% vs old 90%, saving ~10s at RSA-100 scale)
                    {
                        uint32_t current_rels = *postprocessor_->h_pinned_persistent_count;
                        uint32_t remaining = (config_.target_relations > current_rels)
                            ? (config_.target_relations - current_rels) : 0;
                        uint32_t graph_yield = (graph_replay_count > 0)
                            ? (current_rels / static_cast<uint32_t>(graph_replay_count))
                            : (graph_N * 600u);  // bootstrap estimate
                        if (remaining < 2 * graph_yield) {
                            LOG(LOG_DEBUG_1) << "Remaining " << remaining
                                             << " < 2×yield " << graph_yield
                                             << " — switching to single-batch";
                            break;
                        }
                    }

                    // Wait for partial batch reset if extraction in flight (cross-stream dependency)
                    if (data_tap_ && extract_pending) {
                        cudaStreamWaitEvent(postprocessor_->getCudaStream(), partial_reset_event, 0);
                    }

                    // Wait for postproc to finish (counter reset) before sieve starts
                    cudaStreamWaitEvent(siever_->getCudaStream(),
                                        graph_buffer->safe_to_write_event, 0);

                    // Pre-compute next N batches (CPU + H2D + polyGen)
                    for (uint32_t i = 0; i < graph_N; i++) {
                        auto next_indices = mpqs::sieve::prepareNextBatchIndices(&sieve_fdata, batch_size);
                        memcpy(h_staged_indices[i], next_indices.data(), indices_bytes);
                        cudaMemcpyAsync(d_staged_indices[i], h_staged_indices[i],
                                        indices_bytes, cudaMemcpyHostToDevice, siever_->getCudaStream());
                        siever_->prepareSievingBatchFromStaged(
                            d_staged_indices[i], d_staged_a[i], d_staged_B[i]);
                    }

                    // Launch graph (N sieve batches, candidates accumulate in graph_buffer)
                    cudaError_t launch_err = cudaGraphLaunch(graph_exec, siever_->getCudaStream());
                    if (launch_err != cudaSuccess) {
                        LOG(LOG_ERROR_CRITICAL) << "cudaGraphLaunch failed: "
                                                << cudaGetErrorString(launch_err);
                        break;
                    }

                    // Signal postproc that sieve data is ready
                    cudaEventRecord(graph_buffer->safe_to_read_event, siever_->getCudaStream());

                    // Process accumulated candidates (runs on postproc stream)
                    postprocessor_->updatePredictionSteps(current_step);
                    postprocessor_->processBatchBufferedCandidates();

                    // === LP PIPELINE (always-launch, between replays) ===
                    if (largeprime_) {
                        // Drain previous async LP if pending
                        if (lp_graph_pending) {
                            cudaStreamWaitEvent(postprocessor_->getCudaStream(),
                                                largeprime_->getDoneEvent(), 0);
                            largeprime_->launchDeviceAppend(
                                postprocessor_->getPersistentBatch(),
                                postprocessor_->getCudaStream());

                            const auto* lp_stats = largeprime_->getTelemetry();
                            if (lp_stats)
                                global_lp_full_relations += lp_stats->last_batch_full_relations;

                            postprocessor_->resyncPersistentDualCounter();
                            lp_graph_pending = false;
                        }

                        // Launch new async LP on accumulated partials
                        cudaEventRecord(largeprime_->getPartialsReadyEvent(),
                                        postprocessor_->getCudaStream());
                        largeprime_->processAndCommitAsync(
                            postprocessor_->getPartialBatch(),
                            postprocessor_->getPersistentBatch());
                        lp_graph_pending = true;

                        // Gate partial reset on LP completion (not just count snapshot —
                        // LP kernels still read partial data after snapshot)
                        cudaStreamWaitEvent(postprocessor_->getCudaStream(),
                                            largeprime_->getDoneEvent(), 0);
                        postprocessor_->resetPartialBatch();
                    }

                    // Toggle back — graph always writes to the same buffer
                    postprocessor_->toggleActiveBuffer();

                    // Wait for postproc to finish before next graph launch
                    cudaStreamSynchronize(postprocessor_->getCudaStream());

                    // === DATA EXTRACTION — graph path (cluster mode only) ===
                    if (data_tap_) {
                        // Deliver previous extraction if done
                        if (extract_pending) {
                            int prev_idx = active_staging ^ 1;
                            if (cudaEventQuery(extract_done_event[prev_idx]) == cudaSuccess) {
                                data_tap_->onBatchComplete(staging_full[prev_idx],
                                                           staging_part[prev_idx],
                                                           processed_batches - static_cast<int>(graph_N));
                                extract_pending = false;
                            }
                        }

                        // Counters are exact (just synced postproc_stream) — no additional sync needed
                        uint64_t curr_pers = *postprocessor_->h_pinned_persistent_count;
                        uint64_t delta = curr_pers - prev_pers_count;

                        if (delta > 0) {
                            postprocessor_->getPersistentBatch()->moveRangeToHost(
                                staging_full[active_staging], prev_pers_count, delta, extract_stream);
                        }
                        uint64_t curr_partial = postprocessor_->h_pinned_partial_count
                            ? *postprocessor_->h_pinned_partial_count : 0;
                        if (curr_partial > 0) {
                            postprocessor_->getPartialBatch()->moveToHost(
                                staging_part[active_staging], extract_stream);
                        }
                        if (delta > 0 || curr_partial > 0) {
                            if (curr_partial > 0) {
                                postprocessor_->resetPartialBatch(extract_stream);
                                cudaEventRecord(partial_reset_event, extract_stream);
                            }
                            cudaEventRecord(extract_done_event[active_staging], extract_stream);

                            prev_pers_count = curr_pers;
                            active_staging ^= 1;
                            extract_pending = true;
                        }
                    }

                    processed_batches += graph_N;
                    total_batches_processed_ += graph_N;
                    current_step += graph_N * batch_size;
                    graph_replay_count++;

                    // === Telemetry (pinned counters, no GPU sync) ===
                    uint32_t current_telemetry = *postprocessor_->h_pinned_accumulation_counter;
                    uint32_t current_rel_telemetry = *postprocessor_->h_pinned_persistent_count;

                    // LP telemetry
                    if (largeprime_) {
                        const auto* lp_stats = largeprime_->getTelemetry();
                        if (lp_stats && lp_stats->total_iterations > last_lp_gen) {
                            if (lp_stats->total_witnesses > lp_fill_history_.witness_max)
                                lp_fill_history_.witness_max = lp_stats->total_witnesses;
                            lp_fill_history_.witness_sum += lp_stats->total_witnesses;
                            lp_fill_history_.total_slab_overflows    = lp_stats->slab_overflow_count;
                            lp_fill_history_.total_witness_overflows = lp_stats->witness_overflow_count;
                            lp_fill_history_.total_output_overflows  = lp_stats->output_overflow_count;
                            ++lp_fill_history_.sample_count;
                            last_lp_gen = lp_stats->total_iterations;
                        }
                    }

                    // Poll buffer fill snapshot
                    auto* fill_snap = postprocessor_->getBufferFillSnapshot();
                    if (fill_snap && fill_snap->generation > last_fill_gen) {
                        last_buffer_snapshot_ = *fill_snap;
                        buffer_fill_history_.update(*fill_snap);
                        last_fill_gen = fill_snap->generation;
                    }

                    logBufferWarnings();

                    // Contention detection
                    if (pct(last_buffer_snapshot_.accum_fill, last_buffer_snapshot_.accum_capacity) >= 95.0) {
                        accum_contention_streak_++;
                    } else {
                        accum_contention_streak_ = 0;
                    }
                    if (!warned_accum_contention_ && accum_contention_streak_ >= 3) {
                        LOG(LOG_WARNING) << "Accumulation buffer contention detected — sieve throughput "
                                         << "may be degraded. Consider --accum_buf_size "
                                         << (last_buffer_snapshot_.accum_capacity * 2);
                        warned_accum_contention_ = true;
                    }

                    if (processed_batches % 10 == 0 || current_rel_telemetry >= config_.target_relations) {
                        LOG(LOG_DEBUG_1) << "Batch " << processed_batches
                                         << " yielded " << current_telemetry << " candidates. "
                                         << "Total Full Relations: " << current_rel_telemetry;

                        auto t_now = clock::now();
                        double elapsed_sec = std::chrono::duration<double>(t_now - t_sieve_loop_start).count();

                        // LP witness count from stale pinned stats
                        uint64_t witnesses = 0;
                        if (config_.lp1_bound > 0 && largeprime_) {
                            const auto* lp_stats = largeprime_->getTelemetry();
                            if (lp_stats) witnesses = lp_stats->total_witnesses;
                        }

                        logSieveProgress(progress_tracker,
                                         current_rel_telemetry, config_.target_relations,
                                         elapsed_sec,
                                         config_.lp1_bound > 0, witnesses,
                                         global_lp_full_relations);
                    }
                }

                // Graph replay summary
                LOG(LOG_DEBUG_1) << "Completed " << graph_replay_count << " graph replays ("
                                << (graph_replay_count * graph_N) << " batches via graph)";

                // Post-graph LP drain: last LP is still in flight
                if (lp_graph_pending) {
                    cudaStreamWaitEvent(postprocessor_->getCudaStream(),
                                        largeprime_->getDoneEvent(), 0);
                    largeprime_->launchDeviceAppend(
                        postprocessor_->getPersistentBatch(),
                        postprocessor_->getCudaStream());
                    const auto* lp_stats = largeprime_->getTelemetry();
                    if (lp_stats)
                        global_lp_full_relations += lp_stats->last_batch_full_relations;
                    postprocessor_->resyncPersistentDualCounter();
                    lp_graph_pending = false;
                }

                // Cleanup graph
                cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
            }

            // Restore original device pointers to prevent double-free
            // (setJobArrays redirected them to staging buffers during capture)
            siever_->setJobArrays(
                orig_dp.dev_job_a_array,
                orig_dp.dev_job_B_flat,
                orig_dp.dev_job_factor_indices
            );
        }

        if (!graph_ran || *postprocessor_->h_pinned_persistent_count < config_.target_relations) {
        // =============================================================
        // STANDARD BATCH LOOP (primary dispatch, or graph fallback for final ~10%)
        // =============================================================

        while (*postprocessor_->h_pinned_persistent_count < config_.target_relations &&
               !(postprocessor_->h_prediction_result &&
                 postprocessor_->h_prediction_result->should_terminate) &&
               !truncation_limit_reached(*postprocessor_->h_pinned_persistent_count, processed_batches) &&
               !(data_tap_ && data_tap_->shouldStop())) {

            // Wait for partial batch reset if extraction in flight (cross-stream dependency)
            if (data_tap_ && extract_pending) {
                cudaStreamWaitEvent(postprocessor_->getCudaStream(), partial_reset_event, 0);
            }

            auto* active_buffer = postprocessor_->getActiveAccumulationBuffer();

            // 1. Tell Sieve Stream to wait for PostProcessor to finish with this buffer
            cudaStreamWaitEvent(siever_->getCudaStream(), active_buffer->safe_to_write_event, 0);

            // 2. Queue Sieve Batch
            siever_->setPostProcessingLinks(active_buffer);
            siever_->prepareSievingBatch();
            siever_->runSievingBatch(config_.sieve_batch_size, 0);

            // 3. Tell Proc Stream that Sieving is done
            cudaEventRecord(active_buffer->safe_to_read_event, siever_->getCudaStream());

            // 4. Queue Post Processing
            postprocessor_->updatePredictionSteps(current_step);
            postprocessor_->processBatchBufferedCandidates();

            // Record batch completion for extraction pipeline (cluster mode)
            if (data_tap_) {
                cudaEventRecord(batch_done_event, postprocessor_->getCudaStream());
            }

            // === PERIODIC LP PROCESSING (ASYNC — Stage 3) ===
            // LP runs asynchronously on lp_stream while the next sieve batch executes.
            // Deferred append happens before the next LP invocation.
            uint32_t effective_lp_interval = (config_.lp_interval > 0)
                ? config_.lp_interval
                : adaptive_lp_batch_interval_;
            if (config_.lp1_bound > 0 && largeprime_ &&
                processed_batches > 0 && (static_cast<uint32_t>(processed_batches) % effective_lp_interval == 0))
            {
                // --- Drain previous async LP if pending (Stage 4: GPU-side append) ---
                if (lp_async_pending) {
                    // Wait for LP pipeline to finish on postproc stream
                    cudaStreamWaitEvent(postprocessor_->getCudaStream(), largeprime_->getDoneEvent(), 0);

                    // GPU-side append: LP output → persistent batch (no CPU reads, no sync)
                    largeprime_->launchDeviceAppend(
                        postprocessor_->getPersistentBatch(),
                        postprocessor_->getCudaStream()
                    );

                    // Telemetry (stale pinned stats — no GPU sync)
                    const mpqs::lp::SLPPinnedStats* lp_stats = largeprime_->getTelemetry();
                    if (lp_stats) {
                        global_lp_full_relations += lp_stats->last_batch_full_relations;
                    }

                    // Resync postprocessor dual counter after LP modified persistent batch
                    postprocessor_->resyncPersistentDualCounter();

                    lp_async_pending = false;
                }

                // --- Launch new async LP ---
                cudaEventRecord(largeprime_->getPartialsReadyEvent(), postprocessor_->getCudaStream());

#ifdef LP_DEBUG
                {
                    cudaStreamSynchronize(postprocessor_->getCudaStream());
                    uint64_t partial_count = postprocessor_->getPartialBatch()->getCount(0);
                    LOG(LOG_STATS) << "Pre-LP call at batch " << processed_batches
                                  << " | d_partial_batch count = " << partial_count;
                }
#endif

                largeprime_->processAndCommitAsync(
                    postprocessor_->getPartialBatch(),
                    postprocessor_->getPersistentBatch()
                );
                lp_async_pending = true;

                // Gate partial batch reset on LP completion: LP must finish reading
                // all partial data before postproc can reset and overwrite the batch.
                // (count_snapshot_event is too early — LP kernels still read partials after it.)
                cudaStreamWaitEvent(postprocessor_->getCudaStream(),
                                    largeprime_->getDoneEvent(), 0);
                postprocessor_->resetPartialBatch();

                // Re-establish DAG invariant
                auto* next_buf = postprocessor_->getActiveAccumulationBuffer();
                cudaStreamWaitEvent(siever_->getCudaStream(),
                                    next_buf->safe_to_write_event, 0);
            }

            // === DATA EXTRACTION (cluster mode only) ===
            if (data_tap_) {
                // 1. Deliver previous extraction if D2H is done
                if (extract_pending) {
                    int prev_idx = active_staging ^ 1;
                    if (cudaEventQuery(extract_done_event[prev_idx]) == cudaSuccess) {
                        data_tap_->onBatchComplete(staging_full[prev_idx],
                                                   staging_part[prev_idx],
                                                   processed_batches - 1);
                        extract_pending = false;
                    }
                }

                // 2. Start new extraction: wait for postproc on extract_stream
                cudaStreamWaitEvent(extract_stream, batch_done_event, 0);
                cudaStreamSynchronize(extract_stream);  // ~10us: read exact counters

                uint64_t curr_pers = *postprocessor_->h_pinned_persistent_count;
                uint64_t delta = curr_pers - prev_pers_count;

                if (delta > 0) {
                    postprocessor_->getPersistentBatch()->moveRangeToHost(
                        staging_full[active_staging], prev_pers_count, delta, extract_stream);
                }
                uint64_t curr_partial = postprocessor_->h_pinned_partial_count
                    ? *postprocessor_->h_pinned_partial_count : 0;
                if (curr_partial > 0) {
                    postprocessor_->getPartialBatch()->moveToHost(
                        staging_part[active_staging], extract_stream);
                }
                if (delta > 0 || curr_partial > 0) {
                    if (curr_partial > 0) {
                        // Reset on proc_stream (not extract_stream) so the reset is naturally
                        // ordered before the next batch's accumulation on the same stream,
                        // avoiding a cross-stream race that was silently dropping the reset.
                        postprocessor_->resetPartialBatch();
                        cudaEventRecord(partial_reset_event, postprocessor_->getCudaStream());
                    }
                    cudaEventRecord(extract_done_event[active_staging], extract_stream);

                    prev_pers_count = curr_pers;
                    active_staging ^= 1;
                    extract_pending = true;
                }
            }

            // === FLIGHT TELEMETRY (stale pinned counters — no GPU sync) ===
            uint32_t current_telemetry = *postprocessor_->h_pinned_accumulation_counter;
            uint32_t current_rel_telemetry = *postprocessor_->h_pinned_persistent_count;

            // Poll buffer fill snapshot (lock-free via generation ticket)
            auto* fill_snap = postprocessor_->getBufferFillSnapshot();
            if (fill_snap && fill_snap->generation > last_fill_gen) {
                last_buffer_snapshot_ = *fill_snap;
                buffer_fill_history_.update(*fill_snap);
                last_fill_gen = fill_snap->generation;
            }

            // Poll LP fill history (lock-free via generation ticket)
            if (config_.lp1_bound > 0 && largeprime_) {
                const auto* lp_stats = largeprime_->getTelemetry();
                if (lp_stats && lp_stats->total_iterations > last_lp_gen) {
                    if (lp_stats->total_witnesses > lp_fill_history_.witness_max)
                        lp_fill_history_.witness_max = lp_stats->total_witnesses;
                    lp_fill_history_.witness_sum += lp_stats->total_witnesses;
                    lp_fill_history_.total_slab_overflows    = lp_stats->slab_overflow_count;
                    lp_fill_history_.total_witness_overflows = lp_stats->witness_overflow_count;
                    lp_fill_history_.total_output_overflows  = lp_stats->output_overflow_count;
                    ++lp_fill_history_.sample_count;
                    last_lp_gen = lp_stats->total_iterations;
                }
            }

            logBufferWarnings();

            // === Contention detection (batch mode only) ===
            if (pct(last_buffer_snapshot_.accum_fill, last_buffer_snapshot_.accum_capacity) >= 95.0) {
                accum_contention_streak_++;
            } else {
                accum_contention_streak_ = 0;
            }
            if (!warned_accum_contention_ && accum_contention_streak_ >= 3) {
                LOG(LOG_WARNING) << "Accumulation buffer contention detected — sieve throughput "
                                 << "may be degraded. Consider --accum_buf_size "
                                 << (last_buffer_snapshot_.accum_capacity * 2);
                warned_accum_contention_ = true;
            }

            if (processed_batches > 0 && (processed_batches % 10 == 0 || current_rel_telemetry >= config_.target_relations)) {
                LOG(LOG_DEBUG_1) << "Batch " << processed_batches
                                 << " yielded " << current_telemetry << " candidates. "
                                 << "Total Full Relations: " << current_rel_telemetry;

                auto t_now = clock::now();
                double elapsed_sec = std::chrono::duration<double>(t_now - t_sieve_loop_start).count();

                // LP witness count from stale pinned stats (no GPU query)
                uint64_t witnesses = 0;
                if (config_.lp1_bound > 0 && largeprime_) {
                    const auto* lp_stats = largeprime_->getTelemetry();
                    if (lp_stats) witnesses = lp_stats->total_witnesses;
                }

                // Feed LP fill projector (Feature 5)
                if (config_.lp1_bound > 0 && largeprime_ && witnesses > 0) {
                    lp_projector_.addSample(elapsed_sec, witnesses, largeprime_->getWitnessCapacityRels());
                }

                logSieveProgress(progress_tracker,
                                 current_rel_telemetry, config_.target_relations,
                                 elapsed_sec,
                                 config_.lp1_bound > 0, witnesses,
                                 global_lp_full_relations);

                // --- Adaptive LP interval calibration ---
                if (config_.lp1_bound > 0 && largeprime_ && !interval_calibrated_) {
                    double progress = static_cast<double>(current_rel_telemetry) / config_.target_relations;
                    if (progress >= 0.05 && progress_tracker.hasETA()) {
                        double total_est_sec = progress_tracker.current_eta_sec + elapsed_sec;
                        double interval_sec  = total_est_sec / N_LP_INTERVALS;
                        double batches_per_sec = (elapsed_sec > 0) ? (processed_batches / elapsed_sec) : 1.0;
                        uint32_t new_interval = static_cast<uint32_t>(interval_sec * batches_per_sec);
                        adaptive_lp_batch_interval_ = std::clamp(new_interval, 1u, 100u);
                        interval_calibrated_ = true;

                        LOG(LOG_DEBUG_1) << "Adaptive interval: every " << adaptive_lp_batch_interval_
                                         << " batches (~" << std::fixed << std::setprecision(1)
                                         << interval_sec << "s)";
                    }
                }

                // Recalibrate once at ~20% progress for more accurate ETA
                if (config_.lp1_bound > 0 && largeprime_ && interval_calibrated_ && !interval_recalibrated_) {
                    double progress = static_cast<double>(current_rel_telemetry) / config_.target_relations;
                    if (progress >= 0.20 && progress_tracker.hasETA()) {
                        double total_est_sec = progress_tracker.current_eta_sec + elapsed_sec;
                        double interval_sec  = total_est_sec / N_LP_INTERVALS;
                        double batches_per_sec = (elapsed_sec > 0) ? (processed_batches / elapsed_sec) : 1.0;
                        uint32_t new_interval = static_cast<uint32_t>(interval_sec * batches_per_sec);
                        adaptive_lp_batch_interval_ = std::clamp(new_interval, 1u, 100u);
                        interval_recalibrated_ = true;

                        LOG(LOG_DEBUG_1) << "Adaptive interval recalibrated: every " << adaptive_lp_batch_interval_
                                         << " batches";
                    }
                }
            }

            processed_batches++;
            total_batches_processed_++;
            current_step += config_.sieve_batch_size;
        }
        } // end standard batch loop

        // Graph+fallback session summary
        if (graph_ran && graph_replay_count > 0) {
            uint64_t graph_batches = graph_replay_count * graph_N;
            uint64_t fallback = static_cast<uint64_t>(processed_batches) - graph_batches;
            LOG(LOG_DEBUG_1) << "Session: " << graph_batches << " batches graphed, "
                            << fallback << " single-dispatch";
        }

        // === CUDA Graph staging buffer cleanup ===
        if (use_graph) {
            for (uint32_t i = 0; i < graph_N; i++) {
                if (h_staged_indices[i]) cudaFreeHost(h_staged_indices[i]);
                if (d_staged_indices[i]) cudaFree(d_staged_indices[i]);
                if (d_staged_a[i]) cudaFree(d_staged_a[i]);
                if (d_staged_B[i]) cudaFree(d_staged_B[i]);
            }
        }

        // === DRAIN PENDING ASYNC LP (post-loop, Stage 4: GPU-side append) ===
        if (lp_async_pending) {
            cudaStreamWaitEvent(postprocessor_->getCudaStream(), largeprime_->getDoneEvent(), 0);
            largeprime_->launchDeviceAppend(
                postprocessor_->getPersistentBatch(),
                postprocessor_->getCudaStream()
            );
            const mpqs::lp::SLPPinnedStats* lp_stats = largeprime_->getTelemetry();
            if (lp_stats) {
                global_lp_full_relations += lp_stats->last_batch_full_relations;
            }
            postprocessor_->resyncPersistentDualCounter();
            lp_async_pending = false;
        }

        // === PIPELINE FLUSH & SYNC ===
        // Log early-stop info if prediction triggered termination
        if (postprocessor_->h_prediction_result &&
            postprocessor_->h_prediction_result->should_terminate) {
            const auto* pr = postprocessor_->h_prediction_result;
            LOG(LOG_INFO) << "Early stop triggered: effective_R="
                          << pr->effective_R
                          << " yield_rate=" << std::fixed << std::setprecision(4) << pr->yield_rate
                          << " lp_match_rate=" << std::fixed << std::setprecision(4) << pr->lp_match_rate;
        }
        LOG(LOG_DEBUG_1) << "Target reached.";
        cudaStreamSynchronize(siever_->getCudaStream());
        cudaStreamSynchronize(postprocessor_->getCudaStream());
        uint32_t stragglers = *pinned_counter;
        if (stragglers > 0) {
            postprocessor_->processBatchBufferedCandidates();
            cudaStreamSynchronize(postprocessor_->getCudaStream());
        }

        // === OVERSHOOT & PREDICTION TELEMETRY ===
        {
            uint32_t post_flush_count = *postprocessor_->h_pinned_persistent_count;
            int32_t overshoot = static_cast<int32_t>(post_flush_count) - static_cast<int32_t>(config_.target_relations);
            LOG(LOG_DEBUG_1) << "Post-flush relations: " << post_flush_count
                            << " | overshoot: " << overshoot
                            << " (cap=" << relation_cap << ")";

            if (postprocessor_->h_prediction_result) {
                const auto* pr = postprocessor_->h_prediction_result;
                LOG(LOG_DEBUG_1) << "Final state: effective_R=" << pr->effective_R
                                << " yield_rate=" << std::fixed << std::setprecision(4) << pr->yield_rate
                                << " lp_match_rate=" << std::fixed << std::setprecision(4) << pr->lp_match_rate
                                << " should_terminate=" << pr->should_terminate;
            }
        }

        // === FINAL LP PROCESSING: flush remaining partials ===
        if (config_.lp1_bound > 0 && largeprime_) {
            cudaStreamSynchronize(postprocessor_->getCudaStream());
            uint32_t lp_flush_hint = postprocessor_->h_pinned_partial_count
                ? *postprocessor_->h_pinned_partial_count : 0;
            largeprime_->processAndCommit(
                postprocessor_->getPartialBatch(),
                postprocessor_->getPersistentBatch(),
                lp_flush_hint
            );

            const mpqs::lp::SLPPinnedStats* lp_stats = largeprime_->getTelemetry();
            if (lp_stats) {
                global_lp_full_relations += lp_stats->last_batch_full_relations;

                LOG(LOG_DEBUG_1) << "Batch LP processing complete.";
                LOG(LOG_DEBUG_1) << " Witnesses stored: "
                                << lp_stats->total_witnesses << " / "
                                << largeprime_->getWitnessCapacityRels();
                LOG(LOG_DEBUG_1) << " Cumulative LP full relations: "
                                << global_lp_full_relations;
                LOG(LOG_DEBUG_1) << " Hash table: "
                                << lp_stats->empty_hash_buckets << " empty, "
                                << lp_stats->full_hash_buckets << " full buckets";
            }
        }

    } else {
        // =================================================================
        // PATH 2/3: Legacy Sieve Loop (with or without Large Primes)
        // The LP variant is activated when config_.lp1_bound > 0.
        // =================================================================

        // Telemetry state
        uint64_t last_observed_iter = 0;
        uint64_t global_lp_full_relations = 0;
        SieveProgressTracker progress_tracker;

        // --- Buffer fill history tracking (member variables) ---
        uint64_t last_fill_gen = 0;

        // --- LP fill history tracking (member variables) ---
        if (config_.lp1_bound > 0 && largeprime_)
            lp_fill_history_.witness_capacity = largeprime_->getWitnessCapacityRels();

        while (true) {
            if (data_tap_ && data_tap_->shouldStop()) break;
            siever_->updateState();

            // A. Dispatch Sieve Step (Asynchronous)
#ifdef SIEVING_DEBUG_FLAG
            if ((config_.meta_snapshot_enabled && current_step == config_.meta_snapshot_step) ||
                (config_.sas_snapshot_enabled && current_step == config_.sas_snapshot_step)) {
                siever_->sieveFullCubeSnapshot(
                    config_.meta_snapshot_enabled,
                    config_.meta_P_enabled,
                    config_.meta_P,
                    config_.meta_O_enabled,
                    config_.meta_O,
                    config_.sas_snapshot_enabled,
                    current_step,
                    j_io
                );
            } else {
                siever_->sieveFullCube();
            }
#else
            siever_->sieveFullCube();
#endif

            // B. Accumulate Candidates (Asynchronous)
            bool buffer_full = postprocessor_->accumulate(
                 siever_->getRawCandidates(),
                 siever_->getRawCandidateBufferSize(),
                 siever_->getFactoringData().a,
                 siever_->getDeviceA_Factors(),
                 (uint32_t)siever_->getFactoringData().a_factors.size(),
                 -((int32_t)siever_->getFactoringData().M),
                 siever_->getCudaStream()
            );

            current_step++;

            // C. Process Buffers when threshold is met
            if (buffer_full) {
                postprocessor_->processBufferedCandidates();
                postprocessor_->consolidateToPersistent();

                // === DATA EXTRACTION — legacy path (cluster mode only) ===
                if (data_tap_) {
                    // Sync to get exact counts (legacy path has no separate event mechanism)
                    cudaStreamSynchronize(postprocessor_->getCudaStream());

                    uint64_t curr_pers = postprocessor_->getPersistentCount();
                    uint64_t delta = curr_pers - prev_pers_count;

                    if (extract_pending) {
                        int prev_idx = active_staging ^ 1;
                        cudaEventSynchronize(extract_done_event[prev_idx]);
                        data_tap_->onBatchComplete(staging_full[prev_idx],
                                                   staging_part[prev_idx],
                                                   total_batches_processed_);
                        extract_pending = false;
                    }

                    if (delta > 0) {
                        postprocessor_->getPersistentBatch()->moveRangeToHost(
                            staging_full[active_staging], prev_pers_count, delta, extract_stream);
                    }
                    uint64_t curr_partial = postprocessor_->h_pinned_partial_count
                        ? *postprocessor_->h_pinned_partial_count : 0;
                    if (curr_partial > 0) {
                        postprocessor_->getPartialBatch()->moveToHost(
                            staging_part[active_staging], extract_stream);
                    }
                    if (delta > 0 || curr_partial > 0) {
                        if (curr_partial > 0) {
                            postprocessor_->resetPartialBatch();  // proc_stream — same fix as batch path
                        }
                        cudaEventRecord(extract_done_event[active_staging], extract_stream);

                        prev_pers_count = curr_pers;
                        active_staging ^= 1;
                        extract_pending = true;
                    }
                }

                if (config_.lp1_bound > 0 && largeprime_) {
                    // Signal LP that partials are ready (event on postproc stream)
                    cudaEventRecord(largeprime_->getPartialsReadyEvent(), postprocessor_->getCudaStream());

                    // Dispatch 2-Stage SLP Pipeline
                    largeprime_->processAndCommit(
                        postprocessor_->getPartialBatch(),
                        postprocessor_->getPersistentBatch()
                    );

                    // Ensure LP is done before resetting the partial batch
                    cudaStreamWaitEvent(postprocessor_->getCudaStream(), largeprime_->getDoneEvent(), 0);

                    // Reset partial buffer so final flush doesn't reprocess stale partials
                    postprocessor_->resetPartialBatch();

                    total_batches_processed_++;

                    // LP-path: progress sampling deferred to async LP telemetry block (post-LP)
                    uint64_t total_count = postprocessor_->getPersistentCount();
                    if (total_count >= config_.target_relations ||
                        truncation_limit_reached(total_count, total_batches_processed_)) break;
                } else {
                    total_batches_processed_++;

                    // Fallback monitoring for disabled LP Variant
                    uint64_t total_count = postprocessor_->getPersistentCount();

                    if ((total_count - progress_tracker.last_logged_count) > 200 || total_count >= config_.target_relations) {
                        auto t_now = clock::now();
                        double elapsed_sec = std::chrono::duration<double>(t_now - t_sieve_loop_start).count();

                        logSieveProgress(progress_tracker,
                                         total_count, config_.target_relations,
                                         elapsed_sec,
                                         false, 0, 0);
                    }
                    if (total_count >= config_.target_relations) break;
                }
            }

            // D. Asynchronous Telemetry Polling (Zero GPU Stalling)
            if (config_.lp1_bound > 0 && largeprime_) {
                const mpqs::lp::SLPPinnedStats* stats = largeprime_->getTelemetry();

                // Check generation ticket: If GPU has completed a new update
                if (stats && stats->total_iterations > last_observed_iter) {
                    last_observed_iter = stats->total_iterations;

                    // Update LP fill history
                    if (stats->total_witnesses > lp_fill_history_.witness_max)
                        lp_fill_history_.witness_max = stats->total_witnesses;
                    lp_fill_history_.witness_sum += stats->total_witnesses;
                    lp_fill_history_.total_slab_overflows    = stats->slab_overflow_count;
                    lp_fill_history_.total_witness_overflows = stats->witness_overflow_count;
                    lp_fill_history_.total_output_overflows  = stats->output_overflow_count;
                    ++lp_fill_history_.sample_count;

                    auto t_now = clock::now();
                    double elapsed_sec = std::chrono::duration<double>(t_now - t_sieve_loop_start).count();

                    // Feed LP fill projector (Feature 5)
                    lp_projector_.addSample(elapsed_sec, stats->total_witnesses, largeprime_->getWitnessCapacityRels());

                    global_lp_full_relations += stats->last_batch_full_relations;

                    uint64_t total_count = postprocessor_->getPersistentCount();

                    // --- LOG_DEBUG_2: Deep Pipeline Telemetry ---
                    LOG(LOG_DEBUG_2) << "Async Telemetry Update | Iteration: " << last_observed_iter;
                    LOG(LOG_DEBUG_2) << "Hash Table Dynamics: "
                                     << stats->empty_hash_buckets << " Empty Buckets, "
                                     << stats->full_hash_buckets << " Full Buckets";
                    LOG(LOG_DEBUG_2) << "Payload Slab Utilization: Witnesses = "
                                     << stats->total_witnesses << " / " << largeprime_->getWitnessCapacityRels();

                    // --- LOG_DEBUG_1: Batch-level yield ---
                    LOG(LOG_DEBUG_1) << "Batch Yield: +" << stats->last_batch_full_relations
                                     << " Full Relations, +" << stats->last_batch_new_witnesses << " New Witnesses.";

                    // --- LOG_INFO: User-facing Progress (shared function) ---
                    if ((total_count - progress_tracker.last_logged_count) > 1000 || total_count >= config_.target_relations) {
                        logSieveProgress(progress_tracker,
                                         total_count, config_.target_relations,
                                         elapsed_sec,
                                         true, stats->total_witnesses,
                                         global_lp_full_relations);
                    }

                    // Mathematical Guarantee: Pipeline fulfilled target
                    if (total_count >= config_.target_relations ||
                        truncation_limit_reached(total_count, total_batches_processed_)) break;
                }
            }

            // Poll buffer fill snapshot (lock-free via generation ticket)
            {
                auto* fill_snap = postprocessor_->getBufferFillSnapshot();
                if (fill_snap && fill_snap->generation > last_fill_gen) {
                    last_buffer_snapshot_ = *fill_snap;
                    buffer_fill_history_.update(*fill_snap);
                    last_fill_gen = fill_snap->generation;
                }
            }

            logBufferWarnings();

            siever_->advance_a(1);
        } // end while
    }

    // =========================================================================
    // 6. Cleanup & Final Flush
    // =========================================================================

    LOG(LOG_INFO) << "Loop finished. Flushing pipelines...";

    // Only flush for legacy path (batch path already synced and flushed)
    if (config_.sieve_batch_size == 0) {
        postprocessor_->flush();

        if (config_.lp1_bound > 0 && largeprime_) {
            largeprime_->processAndCommit(
                postprocessor_->getPartialBatch(),
                postprocessor_->getPersistentBatch()
            );
        }
    }

    // === Truncation summary ===
    {
        uint64_t rels_now = postprocessor_->getPersistentCount();
        sieve_truncated_ = truncation_limit_reached(rels_now, total_batches_processed_);
        if (sieve_truncated_) {
            // Drain LP for accurate counts
            if (config_.lp1_bound > 0 && largeprime_)
                cudaDeviceSynchronize();

            auto t_now = clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_sieve_loop_start).count();
            rels_now = postprocessor_->getPersistentCount();
            double throughput = (elapsed > 0) ? rels_now / elapsed : 0;
            double eta_full = (throughput > 0)
                ? (config_.target_relations - rels_now) / throughput : 0;

            LOG(LOG_STATS) << "=== Sieve Probe Summary ===";
            LOG(LOG_STATS) << "Elapsed:          " << std::fixed << std::setprecision(1)
                          << elapsed << " s (" << total_batches_processed_ << " batches)";
            LOG(LOG_STATS) << "Relations:        " << rels_now
                          << " / " << config_.target_relations << " target ("
                          << std::fixed << std::setprecision(1)
                          << (config_.target_relations > 0 ? 100.0 * rels_now / config_.target_relations : 0)
                          << "%)";
            LOG(LOG_STATS) << "Throughput:       " << std::fixed << std::setprecision(1)
                          << throughput << " rels/s";
            LOG(LOG_STATS) << "ETA (full sieve): " << std::fixed << std::setprecision(1)
                          << (elapsed + eta_full) << " s";

            if (config_.lp1_bound > 0 && largeprime_) {
                const auto* lp_stats = largeprime_->getTelemetry();
                if (lp_stats) {
                    LOG(LOG_STATS) << "LP witnesses:     " << lp_stats->total_witnesses
                                  << " / " << largeprime_->getWitnessCapacityRels();
                }
            }

            if (!config_.sieve_truncate_continue) {
                LOG(LOG_INFO) << "Exiting (use --sieve_truncate_continue to proceed to matrix/BW/sqrt).";
                return;
            }
            LOG(LOG_INFO) << "--sieve_truncate_continue: proceeding to matrix/BW/sqrt.";
        }
    }

    uint64_t pre_dedup_count = 0;
    uint64_t post_dedup_count = 0;

    if (!cluster_mode) {
    cudaSetDevice(config_.device_id);

    // Ensure LP append operations complete before dedup reads persistent batch
    if (config_.lp1_bound > 0 && largeprime_) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    pre_dedup_count = postprocessor_->getPersistentBatch()->getCount(
        postprocessor_->getCudaStream());
    cudaStreamSynchronize(postprocessor_->getCudaStream());

    postprocessor_->deduplicatePersistentBatch();

    post_dedup_count = postprocessor_->getPersistentBatch()->getCount(
        postprocessor_->getCudaStream());
    cudaStreamSynchronize(postprocessor_->getCudaStream());

    LOG(LOG_INFO) << "Dedup: " << pre_dedup_count << " -> " << post_dedup_count
                  << " (" << (pre_dedup_count - post_dedup_count) << " duplicates removed)";

    // =========================================================================
    // 6b. Post-dedup deficit re-sieve
    // =========================================================================
    // When shc_dim inflation (Jetson polynomial exhaustion guard) or other
    // factors cause high duplicate rates, post-dedup count may fall below
    // FB size.  Re-sieve to collect the deficit instead of aborting.
    {
        const uint64_t fb_size = f_data_.factorBase.size();
        int resieve_attempts = 0;
        constexpr int kMaxResieveAttempts = 3;

        while (post_dedup_count < fb_size + 64 && resieve_attempts < kMaxResieveAttempts) {
            ++resieve_attempts;
            double dedup_rate = (pre_dedup_count > 0)
                ? static_cast<double>(pre_dedup_count - post_dedup_count) / pre_dedup_count
                : 0.0;
            uint64_t deficit = fb_size + 128 - post_dedup_count;
            // Over-collect: deficit / (1 - dedup_rate) + margin
            uint64_t extra = static_cast<uint64_t>(
                deficit / std::max(0.3, 1.0 - dedup_rate)) + 512;

            LOG(LOG_INFO) << "Post-dedup deficit: " << deficit
                          << " relations short (dedup=" << std::fixed << std::setprecision(1)
                          << (dedup_rate * 100.0) << "%). Re-sieve " << resieve_attempts
                          << "/" << kMaxResieveAttempts << " for ~" << extra << " more.";

            // Dedup replaces d_persistent_batch with a tight-fit batch (cap == count).
            // Expand it to hold the deficit + margin before the continuation loop.
            uint64_t resieve_cap_rels = post_dedup_count + extra + 1024;
            uint64_t est_factors_per_rel = 30;  // conservative average
            uint64_t resieve_cap_factors = resieve_cap_rels * est_factors_per_rel;
            postprocessor_->getPersistentBatch()->resize(
                static_cast<size_t>(resieve_cap_rels),
                static_cast<size_t>(resieve_cap_factors));

            // Resync pinned counter + dual counter after dedup replaced the batch
            postprocessor_->resyncPersistentDualCounter();
            cudaStreamSynchronize(postprocessor_->getCudaStream());

            uint64_t resieve_target = post_dedup_count + extra;

            postprocessor_->getPersistentBatch()->setTargetCap(
                static_cast<uint32_t>(std::min(resieve_target + 512,
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))));

            // Synchronous continuation loop (sync-heavy but short — only covers deficit)
            while (*postprocessor_->h_pinned_persistent_count < resieve_target) {
                siever_->advance_a(1);

                if (config_.sieve_batch_size > 0) {
                    auto* buf = postprocessor_->getActiveAccumulationBuffer();
                    cudaStreamSynchronize(siever_->getCudaStream());

                    siever_->setPostProcessingLinks(buf);
                    siever_->prepareSievingBatch();
                    siever_->runSievingBatch(config_.sieve_batch_size, 0);
                    cudaEventRecord(buf->safe_to_read_event, siever_->getCudaStream());

                    postprocessor_->processBatchBufferedCandidates();
                    cudaStreamSynchronize(postprocessor_->getCudaStream());
                } else {
                    siever_->updateState();
                    siever_->sieveFullCube();

                    postprocessor_->accumulate(
                        siever_->getRawCandidates(),
                        siever_->getRawCandidateBufferSize(),
                        siever_->getFactoringData().a,
                        siever_->getDeviceA_Factors(),
                        static_cast<uint32_t>(siever_->getFactoringData().a_factors.size()),
                        -static_cast<int32_t>(siever_->getFactoringData().M),
                        siever_->getCudaStream());
                    postprocessor_->processBufferedCandidates();
                    postprocessor_->consolidateToPersistent();
                    cudaStreamSynchronize(postprocessor_->getCudaStream());
                }

                // LP processing if active (synchronous — deficit is small)
                if (config_.lp1_bound > 0 && largeprime_) {
                    largeprime_->processAndCommit(
                        postprocessor_->getPartialBatch(),
                        postprocessor_->getPersistentBatch()
                    );
                    cudaStreamWaitEvent(postprocessor_->getCudaStream(),
                                        largeprime_->getDoneEvent(), 0);
                    postprocessor_->resetPartialBatch();
                }
            }

            // Re-dedup
            if (config_.lp1_bound > 0 && largeprime_)
                cudaDeviceSynchronize();

            pre_dedup_count = postprocessor_->getPersistentBatch()->getCount(
                postprocessor_->getCudaStream());
            cudaStreamSynchronize(postprocessor_->getCudaStream());

            postprocessor_->deduplicatePersistentBatch();

            post_dedup_count = postprocessor_->getPersistentBatch()->getCount(
                postprocessor_->getCudaStream());
            cudaStreamSynchronize(postprocessor_->getCudaStream());

            LOG(LOG_INFO) << "Re-dedup: " << pre_dedup_count << " -> "
                          << post_dedup_count << " (" << (pre_dedup_count - post_dedup_count)
                          << " duplicates removed)";
        }

        if (post_dedup_count < fb_size + 64) {
            LOG(LOG_ERROR_CRITICAL) << "Still insufficient after "
                                    << kMaxResieveAttempts << " re-sieve attempts: "
                                    << post_dedup_count << " < " << (fb_size + 64);
        }
    }

    LOG(LOG_DEBUG_1) << "Downloading persistent relations to Host...";

    mpqs::structures::RelationBatch* d_persistent = postprocessor_->getPersistentBatch();
    if (d_persistent) {
        cudaStream_t pp_stream = postprocessor_->getCudaStream();
        d_persistent->moveToHost(host_relations_soa_, pp_stream);
        cudaStreamSynchronize(pp_stream);

        // Extract raw partial witnesses for expanded matrix path
        if (config_.lp1_bound > 0 && largeprime_) {
            largeprime_->moveWitnessesToHost(host_partials_soa_, pp_stream);
            cudaStreamSynchronize(pp_stream);
            LOG(LOG_INFO) << "Downloaded " << host_partials_soa_.num_relations
                          << " LP witness partials for expanded matrix.";
        }
    } else {
        LOG(LOG_ERROR_CRITICAL) << "Persistent buffer missing!";
    }
    } else {
        // Cluster mode: drain final extraction and signal completion
        if (data_tap_ && extract_pending) {
            int prev_idx = active_staging ^ 1;
            cudaEventSynchronize(extract_done_event[prev_idx]);
            data_tap_->onBatchComplete(staging_full[prev_idx],
                                       staging_part[prev_idx],
                                       total_batches_processed_);
            extract_pending = false;
        }
        // Deliver any remaining data from the last batch
        if (data_tap_) {
            cudaStreamSynchronize(postprocessor_->getCudaStream());
            uint64_t curr_pers = postprocessor_->getPersistentCount();
            uint64_t delta = curr_pers - prev_pers_count;
            if (delta > 0) {
                postprocessor_->getPersistentBatch()->moveRangeToHost(
                    staging_full[active_staging], prev_pers_count, delta, extract_stream);

                // Bug 3 fix: extract any remaining partials before final delivery
                uint64_t final_partial = postprocessor_->h_pinned_partial_count
                    ? *postprocessor_->h_pinned_partial_count : 0;
                if (final_partial > 0) {
                    postprocessor_->getPartialBatch()->moveToHost(
                        staging_part[active_staging], extract_stream);
                } else {
                    staging_part[active_staging].num_relations = 0;  // No stale partials
                }

                cudaStreamSynchronize(extract_stream);

                if (final_partial > 0) {
                    postprocessor_->resetPartialBatch(extract_stream);
                }

                data_tap_->onBatchComplete(staging_full[active_staging],
                                           staging_part[active_staging],
                                           total_batches_processed_ + 1);
            }
        }

        // Signal Thread A that the sieve loop has exited
        if (cluster_queue_) {
            cluster_queue_->signalSieveDone();
        }
    }

    // === Extraction resource cleanup ===
    if (data_tap_) {
        if (extract_stream)        cudaStreamDestroy(extract_stream);
        if (batch_done_event)      cudaEventDestroy(batch_done_event);
        if (extract_done_event[0]) cudaEventDestroy(extract_done_event[0]);
        if (extract_done_event[1]) cudaEventDestroy(extract_done_event[1]);
        if (partial_reset_event)   cudaEventDestroy(partial_reset_event);
    }

    // =========================================================================
    // 7. Stats & Teardown
    // =========================================================================

    auto t_sieve_end = clock::now();
    double duration_ms = std::chrono::duration<double>(t_sieve_end - t_sieve_start).count() * 1000.0;

    // ---- Populate SieveStageSummary ----
    SieveStageSummary summary;
    summary.total_elapsed_ms = duration_ms;
    summary.target_relations = config_.target_relations;
    summary.total_full_relations = host_relations_soa_.num_relations;  // post-dedup
    summary.pre_dedup_relations = pre_dedup_count;
    summary.total_batches = total_batches_processed_;

    // Relation breakdown
    if (config_.lp1_bound > 0 && largeprime_) {
        summary.lp_active = true;
        const auto* lp_stats = largeprime_->getTelemetry();
        if (lp_stats) {
            summary.lp_combined_relations = lp_stats->total_full_relations;
            summary.unique_witnesses_stored = lp_stats->total_witnesses;
            summary.slab_overflows = lp_stats->slab_overflow_count;
            summary.witness_overflows = lp_stats->witness_overflow_count;
            summary.output_overflows = lp_stats->output_overflow_count;
        }
        summary.lp_witness_capacity = largeprime_->getWitnessCapacityRels();
        summary.witness_fill_pct = summary.lp_witness_capacity > 0
            ? 100.0 * summary.unique_witnesses_stored / summary.lp_witness_capacity : 0.0;
        summary.sieved_full_relations = summary.total_full_relations > summary.lp_combined_relations
            ? summary.total_full_relations - summary.lp_combined_relations : summary.total_full_relations;
    } else {
        summary.sieved_full_relations = summary.total_full_relations;
    }

    // Buffer peaks from history (M2)
    summary.accum_peak         = buffer_fill_history_.accum_max;
    summary.accum_capacity     = last_buffer_snapshot_.accum_capacity;
    summary.partial_peak       = buffer_fill_history_.partial_max;
    summary.partial_capacity   = last_buffer_snapshot_.partial_capacity;
    summary.persistent_peak    = buffer_fill_history_.persistent_max;
    summary.persistent_capacity = last_buffer_snapshot_.persistent_capacity;

    // Throughput
    double elapsed_sec = duration_ms / 1000.0;
    summary.relations_per_sec = elapsed_sec > 0
        ? summary.total_full_relations / elapsed_sec : 0.0;
    if (summary.lp_active && elapsed_sec > 0) {
        summary.lp_witnesses_per_sec = summary.unique_witnesses_stored / elapsed_sec;
        summary.lp_matches_per_sec   = summary.lp_combined_relations / elapsed_sec;
    }

    // Projection (Feature 5) — use actual elapsed as total run time (run is done)
    if (summary.lp_active && lp_projector_.ready()) {
        double total_est_sec = elapsed_sec;  // Actual elapsed (run is done)
        auto proj = lp_projector_.project(total_est_sec);
        summary.has_projection = true;
        summary.projected_fill_pct = proj.projected_fill_pct;
        summary.projected_witnesses = proj.projected_witnesses;
        summary.recommended_capacity = proj.recommended_capacity;
    }

    logSieveStageSummary(summary);

    // Persist summary data for pipeline timing display in Run()
    summary_total_relations_ = summary.total_full_relations;
    summary_lp_combined_     = summary.lp_combined_relations;
    summary_lp_active_       = summary.lp_active;

    // --- Post-sieve history update (M4) ---
    if (!config_.autotune_config.history_file.empty() && config_.autotune_config.save_history) {
        mpqs::autotune::HistoryStore store;
        store.load(config_.autotune_config.history_file);  // OK if file doesn't exist

        mpqs::autotune::HistoryEntry entry;
        // N identification
        entry.N_decimal     = config_.N.to_string();
        entry.N_hash_sha256 = mpqs::autotune::sha256_hex(entry.N_decimal);
        entry.digit_count   = static_cast<uint32_t>(entry.N_decimal.size());
        entry.bit_length    = static_cast<uint32_t>(config_.N.msb());

        // CRITICAL: Use f_data_.F and f_data_.M (not config_.fb_bound which may be 0)
        entry.optimal_params.fb_bound    = f_data_.F;
        entry.optimal_params.sieve_bound = f_data_.M;
        entry.optimal_params.lp1_bound   = config_.lp1_bound;
        // Copy kernel params from config (populated by TuningStage)
        std::memcpy(entry.optimal_params.kernel_params, config_.params,
                     sizeof(entry.optimal_params.kernel_params));

        // Buffer recommendations derived from actual run telemetry
        if (summary.lp_active) {
            // Witness: use actual capacity, inflate if overflows occurred
            entry.optimal_params.recommended_witness_capacity = summary.lp_witness_capacity;
            if (summary.witness_fill_pct > 85.0
                || summary.slab_overflows + summary.witness_overflows > 0) {
                int log2_w = 64 - __builtin_clzll(
                    static_cast<uint64_t>(summary.lp_witness_capacity * 1.5));
                log2_w = std::min(log2_w, 24);
                entry.optimal_params.recommended_witness_capacity = 1ULL << log2_w;
            }
            entry.optimal_params.recommended_partial_buffer = std::max({
                static_cast<uint64_t>(summary.partial_peak * 2),
                static_cast<uint64_t>(summary.accum_capacity),
                uint64_t{65536}});
        }
        if (summary.accum_peak > 0) {
            uint64_t rec_accum = static_cast<uint64_t>(summary.accum_peak * 1.5);
            rec_accum = std::max(rec_accum, uint64_t{65536});
            int log2_a = 64 - __builtin_clzll(rec_accum - 1);
            log2_a = std::min(log2_a, 20);
            entry.optimal_params.recommended_accum_buffer = 1ULL << log2_a;
        }

        // Measured performance from SieveStageSummary
        entry.measured_performance.sieve_time_sec      = summary.total_elapsed_ms / 1000.0;
        entry.measured_performance.total_time_sec      = summary.total_elapsed_ms / 1000.0; // Updated later if full pipeline
        entry.measured_performance.relations_per_sec   = summary.relations_per_sec;
        entry.measured_performance.total_relations     = summary.total_full_relations;
        entry.measured_performance.lp_witnesses        = summary.unique_witnesses_stored;
        entry.measured_performance.lp_combined_relations = summary.lp_combined_relations;

        // Buffer telemetry from SieveStageSummary
        entry.measured_performance.witness_peak      = summary.unique_witnesses_stored;
        entry.measured_performance.witness_capacity   = summary.lp_witness_capacity;
        entry.measured_performance.witness_fill_pct   = summary.witness_fill_pct;
        entry.measured_performance.overflow_events    = summary.slab_overflows
                                                      + summary.witness_overflows
                                                      + summary.output_overflows;
        entry.measured_performance.accum_peak        = summary.accum_peak;
        entry.measured_performance.partial_peak      = summary.partial_peak;
        entry.measured_performance.persistent_peak   = summary.persistent_peak;

        // GPU environment from CUDA runtime
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, config_.device_id);
        entry.environment.gpu_name = prop.name;
        entry.environment.gpu_compute_capability =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);
        int cuda_ver = 0;
        cudaRuntimeGetVersion(&cuda_ver);
        entry.environment.cuda_version =
            std::to_string(cuda_ver / 1000) + "." + std::to_string((cuda_ver % 1000) / 10);

        entry.timestamp = mpqs::autotune::iso8601_now();
        entry.autotune_stages_run = config_.autotune_stages_run;
        entry.confidence = config_.autotune_confidence > 0.0
                         ? config_.autotune_confidence
                         : 0.5;  // Default for non-autotuned runs

        store.upsert(entry);
        if (store.save(config_.autotune_config.history_file)) {
            LOG(LOG_DEBUG_1) << "Saved entry for "
                            << entry.digit_count << "-digit N ("
                            << entry.bit_length << " bits) to "
                          << config_.autotune_config.history_file;
        } else {
            LOG(LOG_WARNING) << "Failed to save history to "
                             << config_.autotune_config.history_file;
        }
    }

    // --- Post-sieve benign history update ---
    if (!config_.autotune_config.benign_history_file.empty()) {
        mpqs::autotune::BenignHistoryStore benign;
        benign.load(config_.autotune_config.benign_history_file);  // OK if missing

        // Compute digit/bit info independently (entry is out of scope)
        std::string N_str = config_.N.to_string();
        uint32_t digit_count = static_cast<uint32_t>(N_str.size());
        uint32_t bit_length  = static_cast<uint32_t>(config_.N.msb());

        mpqs::autotune::BenignHistoryEntry be;
        be.digit_count_lo = (digit_count >= 3) ? digit_count - 2 : 1;
        be.digit_count_hi = digit_count + 2;
        be.bit_length     = bit_length;
        be.fb_bound       = f_data_.F;       // Actual F used (not config_.fb_bound which may be 0)
        be.sieve_bound    = f_data_.M;       // Actual M used
        be.lp1_bound      = config_.lp1_bound;
        be.confidence     = std::min(
            config_.autotune_confidence > 0.0 ? config_.autotune_confidence : 0.5,
            0.5);  // Cap at 0.5 for benign

        // Buffer recommendations from actual run
        if (summary.lp_active) {
            // Use recommended capacity from projector if available, otherwise actual capacity
            if (summary.recommended_capacity > 0)
                be.recommended_witness_capacity = summary.recommended_capacity;
            else
                be.recommended_witness_capacity = summary.lp_witness_capacity;

            // If overflows occurred, inflate recommendation
            if (summary.slab_overflows + summary.witness_overflows + summary.output_overflows > 0) {
                be.recommended_witness_capacity = static_cast<uint64_t>(
                    be.recommended_witness_capacity * 1.5);
            }

            be.recommended_partial_buffer = std::max({
                static_cast<uint64_t>(summary.partial_peak * 2),
                static_cast<uint64_t>(summary.accum_capacity),
                uint64_t{65536}});  // Floor: max(2×peak, accum_cap, 64K)
        }

        benign.upsert(be);
        benign.save(config_.autotune_config.benign_history_file);
        LOG(LOG_DEBUG_1) << "Updated benign history with "
                         << digit_count << "-digit entry";
    }

    siever_->clearSievingBuffers();
    postprocessor_->clearBuffers();
    // [CRITICAL] Do NOT clear persistent buffer here! MatrixStage needs it.
    if (largeprime_) largeprime_->clearBuffers();
}

// -----------------------------------------------------------------------------
// TruncatedSieveRun: Probe sieve for runtime estimation (M3 Tile 1.A)
// -----------------------------------------------------------------------------

MPQSOrchestrator::TruncatedSieveResult MPQSOrchestrator::TruncatedSieveRun(
    double frac, double eta_convergence_threshold,
    uint32_t min_eta_samples, double min_progress_frac)
{
    // Clear any sticky CUDA errors from prior stages (e.g., Stage 1 kernel benchmarking)
    cudaDeviceSynchronize();
    cudaGetLastError();

    TruncatedSieveResult result;
    result.target_relations = config_.target_relations;

    // Preflight guard: graceful degradation for infeasible autotune probes
    if (config_.useParams) {
        auto pf = autotune::preflightKernelLaunch(config_,
            static_cast<uint32_t>(f_data_.a_factors.size()), f_data_.M);
        if (!pf.feasible) {
            LOG(LOG_WARNING) << "Preflight failed: " << pf.reason
                          << " -- returning zero relations";
            TruncatedSieveResult empty{};
            empty.relations_found = 0;
            return empty;
        }
    }

    using clock = std::chrono::high_resolution_clock;

    // =========================================================================
    // 1. Initialize Siever (replicate SieveStage init)
    // =========================================================================

    {
        cudaStream_t sieve_stream;
        CUDA_CHECK(cudaStreamCreate(&sieve_stream));
        siever_ = std::make_unique<mpqs::sieve::DeviceSievingController>(config_.device_id, sieve_stream);
    }
    siever_->initiate(f_data_);
    if (config_.lp1_bound > 0)
        siever_->setThresholdOverride(config_.lp1_bound);

    // Force legacy loop — no batch mode for probes

    if (config_.useParams) {
        siever_->loadPartialCustomConfig(config_.params[0],
                                         config_.params[1],
                                         config_.params[2],
                                         config_.params[3],
                                         config_.params[4],
                                         config_.params[5],
                                         config_.params[6],
                                         config_.params[7]);
    } else {
        // Mirror SieveStage small-N adaptive config: loadStandardConfig() sets
        // metaB from SM count which exceeds max_polys for small N, producing
        // num_polyBlocksPerThreadBlock=0 and failing validation.
        uint32_t shc_dim_probe = static_cast<uint32_t>(f_data_.a_factors.size());
        uint32_t max_polys_probe = (1u << shc_dim_probe) / 2;
        if (max_polys_probe < 64) {
            uint32_t totalPolys = 1;
            while (totalPolys * 2 <= max_polys_probe) totalPolys *= 2;

            uint32_t sievingBlockSize = 1;
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, config_.device_id);
                uint32_t target = (3u * static_cast<uint32_t>(prop.sharedMemPerBlock)) / 4u;
                while (sievingBlockSize * 2 <= target) sievingBlockSize *= 2;
            }

            uint32_t totalIntervals = (2u * f_data_.M) / sievingBlockSize;
            if (totalIntervals < 1) totalIntervals = 1;
            { uint32_t p = 1; while (p * 2 <= totalIntervals) p *= 2; totalIntervals = p; }

            uint32_t sasB = std::min(256u, totalPolys);
            siever_->loadPartialCustomConfig(
                totalPolys, totalIntervals, /*polyBlockSize=*/1, /*blocksPerCycle=*/totalIntervals,
                /*metaB=*/totalPolys, /*metaT=*/256, sasB, /*sasT=*/256);
        } else {
            siever_->loadStandardConfig();
        }
    }
    if (!(siever_->validateConfigs())) {
        throw std::runtime_error("Sieve config validation failed");
    }
    siever_->loadData();
    siever_->updateState();

    // =========================================================================
    // 2. Initialize Post-Processor (replicate SieveStage init)
    // =========================================================================

    {
        // Probes need frequent progress checks — use small buffer so buffer_full
        // fires every ~6500 raw candidates instead of ~420K (Root Cause #3 fix).
        constexpr uint64_t PROBE_ACCUM_BUFFER = 8192;
        config_.pp_config = initPostProcessorConfig(PROBE_ACCUM_BUFFER);
        result.target_relations = config_.target_relations;
    }

    postprocessor_ = std::make_unique<mpqs::postprocessing::DevicePostProcessingController>();
    postprocessor_->initiate(f_data_, siever_->getDevicePointers(), config_.pp_config);

    // =========================================================================
    // 3. Initialize Large Prime Variant (replicate SieveStage init)
    // =========================================================================

    initLargePrimes();

    // =========================================================================
    // 4. Prepare Telemetry State
    // =========================================================================

    // Reset member telemetry for this probe
    buffer_fill_history_ = {};
    lp_fill_history_ = {};
    lp_projector_ = LPFillProjector{};

    uint64_t ceiling = static_cast<uint64_t>(frac * result.target_relations);
    std::deque<double> eta_window;

    // L6: shared ETA convergence check used by the LP buffer-full, no-LP, and LP
    // telemetry sites below. Pushes the latest ETA sample, drops the oldest if the
    // window is full, and returns true when the (max-min)/mean spread of the
    // window falls below eta_convergence_threshold (and we have made at least
    // min_progress_frac progress against the target). On convergence, sets
    // result.converged_early so the caller can exit the probe loop.
    auto checkEtaConvergence = [&](uint64_t total_count) -> bool {
        if (!result.progress_tracker.hasETA()) return false;
        eta_window.push_back(result.progress_tracker.current_eta_sec);
        if (eta_window.size() > min_eta_samples)
            eta_window.pop_front();
        if (eta_window.size() < min_eta_samples) return false;
        if (total_count < static_cast<uint64_t>(
                min_progress_frac * result.target_relations))
            return false;
        double mn = *std::min_element(eta_window.begin(), eta_window.end());
        double mx = *std::max_element(eta_window.begin(), eta_window.end());
        double mean = 0.0;
        for (double v : eta_window) mean += v;
        mean /= eta_window.size();
        if (mean > 0.0 && (mx - mn) / mean < eta_convergence_threshold) {
            result.converged_early = true;
            return true;
        }
        return false;
    };

    uint64_t last_observed_iter = 0;
    uint64_t global_lp_full_relations = 0;
    uint64_t last_fill_gen = 0;

    if (config_.lp1_bound > 0 && largeprime_)
        lp_fill_history_.witness_capacity = largeprime_->getWitnessCapacityRels();

    auto t0 = clock::now();

    // =========================================================================
    // 5. Legacy Loop with Dual Termination
    //    (replicate PATH 2/3 loop body, modified termination)
    // =========================================================================

    while (true) {
        siever_->updateState();

        // A. Dispatch Sieve Step
        siever_->sieveFullCube();

        // B. Accumulate Candidates
        bool buffer_full = postprocessor_->accumulate(
             siever_->getRawCandidates(),
             siever_->getRawCandidateBufferSize(),
             siever_->getFactoringData().a,
             siever_->getDeviceA_Factors(),
             (uint32_t)siever_->getFactoringData().a_factors.size(),
             -((int32_t)siever_->getFactoringData().M),
             siever_->getCudaStream()
        );

        result.steps_executed++;

        // C. Process Buffers when threshold is met
        if (buffer_full) {
            postprocessor_->processBufferedCandidates();
            postprocessor_->consolidateToPersistent();

            if (config_.lp1_bound > 0 && largeprime_) {
                largeprime_->processAndCommit(
                    postprocessor_->getPartialBatch(),
                    postprocessor_->getPersistentBatch()
                );
                postprocessor_->resetPartialBatch();

                // LP-path: progress sampling deferred to async LP telemetry block (post-LP)
                uint64_t total_count = postprocessor_->getPersistentCount();

                result.relations_found = total_count;
                if (total_count >= ceiling) break;

                // ETA convergence check (LP buffer-full path)
                if (checkEtaConvergence(total_count)) break;
            } else {
                // No-LP path: check progress on buffer-full events
                uint64_t total_count = postprocessor_->getPersistentCount();
                auto t_now = clock::now();
                double elapsed_sec = std::chrono::duration<double>(t_now - t0).count();

                if ((total_count - result.progress_tracker.last_logged_count) > 200
                    || total_count >= ceiling)
                {
                    logSieveProgress(result.progress_tracker,
                                     total_count, config_.target_relations,
                                     elapsed_sec, false, 0, 0);
                }

                result.relations_found = total_count;
                if (total_count >= ceiling) break;

                // ETA convergence check (no-LP)
                if (checkEtaConvergence(total_count)) break;
            }
        }

        // D. Asynchronous LP Telemetry Polling
        if (config_.lp1_bound > 0 && largeprime_) {
            const mpqs::lp::SLPPinnedStats* stats = largeprime_->getTelemetry();

            if (stats && stats->total_iterations > last_observed_iter) {
                last_observed_iter = stats->total_iterations;

                // Update LP fill history
                if (stats->total_witnesses > lp_fill_history_.witness_max)
                    lp_fill_history_.witness_max = stats->total_witnesses;
                lp_fill_history_.witness_sum += stats->total_witnesses;
                lp_fill_history_.total_slab_overflows    = stats->slab_overflow_count;
                lp_fill_history_.total_witness_overflows = stats->witness_overflow_count;
                lp_fill_history_.total_output_overflows  = stats->output_overflow_count;
                ++lp_fill_history_.sample_count;

                auto t_now = clock::now();
                double elapsed_sec = std::chrono::duration<double>(t_now - t0).count();

                // Feed LP fill projector
                lp_projector_.addSample(elapsed_sec, stats->total_witnesses,
                                        largeprime_->getWitnessCapacityRels());

                global_lp_full_relations += stats->last_batch_full_relations;

                uint64_t total_count = postprocessor_->getPersistentCount();

                // Progress logging (calls addSample internally)
                if ((total_count - result.progress_tracker.last_logged_count) > 1000
                    || total_count >= ceiling)
                {
                    logSieveProgress(result.progress_tracker,
                                     total_count, config_.target_relations,
                                     elapsed_sec, true, stats->total_witnesses,
                                     global_lp_full_relations);
                }

                result.relations_found = total_count;
                if (total_count >= ceiling) break;

                // ETA convergence check (LP path)
                if (checkEtaConvergence(total_count)) break;
            }
        }

        // Poll buffer fill snapshot
        {
            auto* fill_snap = postprocessor_->getBufferFillSnapshot();
            if (fill_snap && fill_snap->generation > last_fill_gen) {
                buffer_fill_history_.update(*fill_snap);
                last_fill_gen = fill_snap->generation;
            }
        }

        siever_->advance_a(1);

        // Hard timeout safety net — prevents infinite loops regardless of
        // buffer_full status (Root Cause #3 fix, belt-and-suspenders).
        {
            double elapsed = std::chrono::duration<double>(
                clock::now() - t0).count();
            if (elapsed > config_.probe_timeout) {
                LOG(LOG_WARNING) << "Hard timeout after "
                                 << elapsed << "s — exiting probe loop";
                break;
            }
        }
    } // end while

    // =========================================================================
    // 6. Finalize
    // =========================================================================

    auto t1 = clock::now();
    result.elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    result.eta_reliable = result.progress_tracker.hasETA()
                        && result.progress_tracker.history.size() >= 6;

    // Copy telemetry from member state
    result.buffer_fill = buffer_fill_history_;
    result.lp_fill = lp_fill_history_;
    result.lp_projector = lp_projector_;

    // =========================================================================
    // 7. Cleanup GPU resources — reset unique_ptrs so destructors run exactly once
    // =========================================================================
    cudaDeviceSynchronize();
    if (largeprime_) largeprime_.reset();
    postprocessor_.reset();
    siever_.reset();
    cudaGetLastError();

    return result;
}

// -----------------------------------------------------------------------------
// Stage 3: Matrix
// -----------------------------------------------------------------------------

void MPQSOrchestrator::MatrixStage() {
    LOG_SET_MODULE("Orchestrator");
    LOG_SET_STAGE(LOG_STAGE_MATRIX_PREPROCESSING, "Matrix");
    LOG(LOG_INFO) << "Matrix Construction";
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    // 1. Validation: ensure relations are available
    if (!postprocessor_ && host_relations_soa_.num_relations == 0) {
        LOG(LOG_ERROR_CRITICAL) << "No relations found. SieveStage must run first or relations must be loaded from disk.";
        return;
    }

    // 2. Construct CSR Matrix (High Performance)
    LOG(LOG_INFO) << "Constructing Matrix from SoA...";
    matrix_constructor_ = std::make_unique<mpqs::matrix::MatrixConstructor>();
    mpqs::matrix::HostMatrixCSR csr_matrix;
    csr_matrix.n_cols = f_data_.size + 2;

    if (postprocessor_) {
        // Normal path: construct from device SoA persistent batch
        auto batch_view = postprocessor_->getPersistentBatch()->get_view();
        size_t count = postprocessor_->getPersistentCount();

        LOG(LOG_STATS) << "Processing " << count << " relations from Device SoA.";
        matrix_constructor_->constructFromSoA(
            batch_view,
            count,
            csr_matrix
        );
    } else {
        // LINALG_ONLY fallback: upload host_relations_soa_ to GPU
        LOG(LOG_INFO) << "LINALG_ONLY: Uploading " << host_relations_soa_.num_relations
                      << " host relations to GPU for matrix construction.";

        mpqs::structures::RelationBatch temp_batch;
        temp_batch.initiate(config_.device_id);
        temp_batch.uploadFromHost(host_relations_soa_);

        auto batch_view = temp_batch.get_view();
        size_t count = host_relations_soa_.num_relations;

        LOG(LOG_STATS) << "Processing " << count << " relations from uploaded SoA.";
        matrix_constructor_->constructFromSoA(
            batch_view,
            count,
            csr_matrix
        );
        // temp_batch destructor frees GPU memory
    }

    // Append quadratic character columns (k=32 auxiliary primes).
    // Character columns encode Legendre symbols (Q_i | q_j) and constrain
    // the BW null space to exclude trivial-sigma vectors.
    // M8c/M9d: dispatch to GPU backend when matrix_backend == GPU (1) or AUTO (2).
    // M9d: when the persistent device batch is alive (full pipeline), read sqrt_Q
    // directly from device memory — eliminates the ~20 MB D→H→D round-trip.
    // LINALG_ONLY fallback: no persistent batch exists; use existing host-upload path.
    {
        matrix::CharacterColumnComputer cc;
        cc.selectAuxPrimes(config_.N, f_data_.factorBase);

        matrix::CharacterColumns chars;
        if ((config_.matrix_backend == 1 /*GPU*/ || config_.matrix_backend == 2 /*AUTO*/) &&
            postprocessor_) {
            // M9d: device-resident path — zero sqrt_Q upload
            auto batch_view = postprocessor_->getPersistentBatch()->get_view();
            chars = matrix::gpuComputeCharacterColumns_device(
                batch_view.sqrt_Q,
                static_cast<uint32_t>(postprocessor_->getPersistentCount()),
                cc.auxPrimes(), cc.nModQ());
        } else if (config_.matrix_backend == 1 /*GPU*/ || config_.matrix_backend == 2 /*AUTO*/) {
            // GPU backend but no persistent batch (LINALG_ONLY) — use host-upload path
            chars = matrix::gpuComputeCharacterColumns(
                host_relations_soa_, cc.auxPrimes(), cc.nModQ());
        } else {
            // CPU backend
            chars = cc.compute(host_relations_soa_);
        }

        matrix::AppendCharacterColumns(csr_matrix, chars,
                                       static_cast<uint32_t>(csr_matrix.n_rows));
        LOG(LOG_INFO) << "Appended " << chars.k << " character columns. "
                      << "Matrix: " << csr_matrix.n_rows << " x " << csr_matrix.n_cols
                      << " (+" << chars.k << " char cols).";
    }

#ifdef MPQS_DEBUG
    // 3. Validate CSR Integrity — O(NNZ) structural check; debug-only.
    LOG(LOG_DEBUG_1) << "Validating CSR Matrix Structure...";
    {
        bool valid = mpqs::matrix::ValidateHostMatrixCSR(csr_matrix);
        if (!valid) {
            LOG(LOG_ERROR_CRITICAL) << "Matrix Validation FAILED. Aborting.";
            throw std::runtime_error("Constructed matrix is corrupt");
        } else {
            LOG(LOG_INFO) << "Matrix Integrity Verified.";
        }
    }
#endif

    // Persist matrix telemetry for pipeline timing display in Run()
    summary_matrix_rows_ = csr_matrix.n_rows;
    summary_matrix_cols_ = csr_matrix.n_cols;
    summary_matrix_nnz_  = csr_matrix.row_offsets.empty() ? 0
                           : csr_matrix.row_offsets[csr_matrix.n_rows];

    // 4. Convert to Legacy HostMatrix for Solver (Legacy Compat)
    //    MPQSOrchestrator::matrix_A_ is of type HostMatrix (vector<vector>)
    LOG(LOG_DEBUG_1) << "Converting CSR to Solver Format (Legacy)...";
    matrix_A_ = mpqs::matrix::ConvertFromCSR(csr_matrix);

    // 5. Verify Dimensions (Solver Readiness)
    int excess = (int)matrix_A_.n_rows - (int)matrix_A_.n_cols;
    if (excess < 0) {
        LOG(LOG_ERROR_MAJOR) << "Underdetermined System! Deficit: " << -excess;
    } else {
        LOG(LOG_STATS) << "System Overdetermined. Excess: " << excess;
    }

    // 6. Cleanup, free device relation buffers
    if (postprocessor_) postprocessor_->clearPersistentBuffer();

    auto t_end = clock::now();
    double duration = std::chrono::duration<double>(t_end - t_start).count();

    LOG(LOG_INFO) << "Construction Completed in " << FormatDuration(duration * 1000.0);
}

// -----------------------------------------------------------------------------
// Stage 4: Linear Algebra
// -----------------------------------------------------------------------------

void MPQSOrchestrator::LinearAlgebraStage() {
    LOG_SET_MODULE("Orchestrator");
    LOG_SET_STAGE(LOG_STAGE_BW_INITIALIZATION, "LinAlg");
    LOG(LOG_INFO) << "Linear Algebra (Block Wiedemann)";
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    lingen::BWSolverConfig& bw_conf = config_.bw_config;
    bw_conf.stage2_gpu_mode = true;  // Default to GPU mode for High Performance
    bw_conf.solve_transposed = true; // (Left Kernel of A)
    bw_conf.nrows = matrix_A_.n_rows;
    bw_conf.device_id = config_.device_id;
    bw_conf.checkpoint_prefix = config_.work_dir + "/bw";
    bw_conf.m_block = (int)config_.bw_m;
    bw_conf.n_block = (int)config_.bw_n;
    bw_conf.block_size_pinned = config_.isPinned("bw_m") || config_.isPinned("bw_n");
    bw_conf.stage1_gpu_batch_size = 8;

    // Adaptive BW block sizes — only when not user-overridden
    uint32_t mat_dim = std::max(matrix_A_.n_rows, matrix_A_.n_cols);
    if (!config_.isPinned("bw_m") && !config_.isPinned("bw_n")) {
        if (mat_dim < 4000) {
            bw_conf.m_block = 64;
            bw_conf.n_block = 64;
        } else if (mat_dim < 16000) {
            bw_conf.m_block = 128;
            bw_conf.n_block = 128;
        }
        // else: keep 256 (default for large matrices)
    }

    // Skip SpMM autotuning for small matrices — overhead dominates computation
    if (mat_dim < 100000) {
        bw_conf.autotune_tune_spmm = false;
    }

    // 5. Prepare Solver Input (Pad to Square)
    LOG_INCREMENT_STAGE(10);
    HostMatrix A_square = lingen::pad_to_square(matrix_A_);    
    
    // Instantiate and Solve
    linalg_solver_ = std::make_unique<lingen::BlockWiedemannSolver>(bw_conf, A_square);
    logGpuMemory("Before BW solve");
    {
        LOG_SCOPED_MODULE("LinAlg");
        linalg_solver_->Solve();
    }

    const auto& solutions = linalg_solver_->get_solutions();
    if (solutions.empty()) {
        LOG(LOG_WARNING) << "No kernel vectors found.";
        return;
    }

    kernel_solutions_ = solutions;
    LOG(LOG_INFO) << "Kernel Solutions Found: "
		  << kernel_solutions_.size();

    auto sol_view = linalg_solver_->get_device_solutions();
    LOG(LOG_DEBUG_1) << "Device-side solutions: " << sol_view.num_solutions
                     << " vectors (" << sol_view.words_per_vec << " words each) at "
                     << (void*)sol_view.d_data;

    auto t_end = clock::now();
    double duration = std::chrono::duration<double>(t_end - t_start).count();
    LOG(LOG_INFO) << "Block Wiedemann execution completed in "
		  << FormatDuration(duration * 1000.0);
}

// -----------------------------------------------------------------------------
// Stage 5: Square Root
// -----------------------------------------------------------------------------

void MPQSOrchestrator::SquareRootStage() {
    LOG_SET_MODULE("Orchestrator");
    LOG_SET_STAGE(LOG_STAGE_SQRT, "Sqrt");
    LOG(LOG_INFO) << "Square Root";
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    if (kernel_solutions_.empty()) {
        LOG(LOG_WARNING) << "No solutions from kernel computation. Terminating.";
	return;
    }

    // 1. Initialize Solver
    sqrt_solver_ = std::make_unique<mpqs::sqrt::SquareRootRefinement>(config_.N);

    bool success = false;

    // When using expanded-matrix path, BW's device solutions reference the
    // reduced matrix.  We've expanded kernel_solutions_ to reference original
    // relations, so we must upload the expanded solutions to the GPU and build
    // a fresh BWKernelSolutionView.
    uint64_t* d_expanded_sol_data = nullptr;
    lingen::BWKernelSolutionView expanded_sol_view{};
    if (used_expanded_matrix_ && !kernel_solutions_.empty()) {
        const uint32_t n_sol = static_cast<uint32_t>(kernel_solutions_.size());
        const uint32_t n_rows = static_cast<uint32_t>(host_relations_soa_.num_relations);
        const uint32_t wpv = (n_rows + 63) / 64;

        // Pack into contiguous row-major buffer
        std::vector<uint64_t> packed(static_cast<size_t>(n_sol) * wpv, 0);
        for (uint32_t j = 0; j < n_sol; ++j) {
            const auto& kv = kernel_solutions_[j];
            size_t copy_words = std::min(static_cast<size_t>(wpv), kv.size());
            std::memcpy(&packed[static_cast<size_t>(j) * wpv], kv.data(),
                        copy_words * sizeof(uint64_t));
        }

        size_t buf_bytes = packed.size() * sizeof(uint64_t);
        cudaMalloc(&d_expanded_sol_data, buf_bytes);
        cudaMemcpy(d_expanded_sol_data, packed.data(), buf_bytes, cudaMemcpyHostToDevice);

        expanded_sol_view.d_data = d_expanded_sol_data;
        expanded_sol_view.num_solutions = n_sol;
        expanded_sol_view.words_per_vec = wpv;
        expanded_sol_view.num_rows = n_rows;
    }

    if (config_.sqrt_legacy) {
        // =====================================================================
        // CPU legacy path (--sqrt_legacy)
        // =====================================================================
        LOG(LOG_INFO) << "Using CPU legacy path (--sqrt_legacy).";
        LOG(LOG_INFO) << "CPU path: trying " << kernel_solutions_.size() << " kernel vectors.";

        for (size_t i = 0; (i < kernel_solutions_.size()) && !success; ++i) {
            LOG(LOG_DEBUG_1) << "Processing Solution Vector "
                             << (i+1) << " of "
                             << kernel_solutions_.size();
            const mpqs::uint512* lp_corr_ptr =
                (used_expanded_matrix_ && i < precomputed_lp_y_.size())
                ? &precomputed_lp_y_[i] : nullptr;
            std::pair<mpqs::uint512, mpqs::uint512> factors = sqrt_solver_->Perform(
                kernel_solutions_[i],
                host_relations_soa_,
                f_data_.factorBase,
                lp_corr_ptr
            );

            if (factors.first.is_zero() || factors.second.is_zero()) {
                LOG(LOG_ERROR_CRITICAL) << "SANITY CHECK FAILED (X²≢Y²) for solution " << i;
            } else if (factors.first.is_one() || factors.first == config_.N) {
                LOG(LOG_INFO) << "trivial GCD (X≡±Y mod N) for solution " << i;
            } else {
                LOG(LOG_INFO) << "Found Factors:";
                LOG(LOG_INFO) << " F1 = " << factors.first.to_string();
                LOG(LOG_INFO) << " F2 = " << factors.second.to_string();

                mpqs::uint512 product = factors.first;
                product.mult(factors.second);

                if (product != config_.N) {
                    LOG(LOG_ERROR_CRITICAL) << "FAILURE: F1 * F2 != N";
                    LOG(LOG_ERROR_CRITICAL) << "Computed Product: "
                                            << product.to_string();
                    success = false;
                } else {
                    LOG(LOG_DEBUG_1) << "Check: Product matches N.";
                    result_factors_.insert(result_factors_.end(),
                                           { std::move(factors.first),
                                             std::move(factors.second) }
                    );
                    success = true;
                }
            }
        }
    } else {
        // =====================================================================
        // GPU batched path (default)
        // =====================================================================
        LOG(LOG_INFO) << "Using GPU batched path.";

        auto ms = [](clock::time_point a, clock::time_point b) -> double {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        // 2. Obtain device-side solution view
        //    When expanded-matrix path was used, use our uploaded expanded solutions.
        auto sol_view = (d_expanded_sol_data != nullptr)
                        ? expanded_sol_view
                        : linalg_solver_->get_device_solutions();
        const int n_solutions = static_cast<int>(sol_view.num_solutions);
        LOG(LOG_STATS) << "Processing " << n_solutions << " solutions in batch.";

        // Pre-allocate device buffers with complete sizing to prevent
        // reallocation between ComputeX and ComputeY (which would destroy X results).
        sqrt_solver_->allocateDeviceBuffers(
            static_cast<uint32_t>(host_relations_soa_.num_relations),
            static_cast<uint32_t>(host_relations_soa_.num_factors),
            static_cast<uint32_t>(f_data_.factorBase.size()),
            n_solutions);

        // 3. BatchedComputeX — compute X[n] on GPU (results stay on device)
        auto t_x = clock::now();
        sqrt_solver_->ComputeXBatchedGPU(sol_view, host_relations_soa_);
        auto t_y = clock::now();
        LOG(LOG_DEBUG_1) << "BatchedComputeX: "
                         << std::fixed << std::setprecision(2) << ms(t_x, t_y) << " ms";

        // 4. BatchedComputeY — compute Y[n] on GPU (results stay on device)
        sqrt_solver_->ComputeYBatchedGPU(sol_view, host_relations_soa_, f_data_.factorBase);
        auto t_gcd = clock::now();
        LOG(LOG_DEBUG_1) << "BatchedComputeY: "
                         << std::fixed << std::setprecision(2) << ms(t_y, t_gcd) << " ms";

        // Apply precomputed LP Y-correction (expanded-matrix path)
        LOG(LOG_INFO) << "LP correction gate: used_expanded_matrix_=" << used_expanded_matrix_
                      << " precomputed_lp_y_.size()=" << precomputed_lp_y_.size();
        if (used_expanded_matrix_ && !precomputed_lp_y_.empty()) {
            mpqs::uint512* d_lp_correction = nullptr;
            size_t corr_bytes = precomputed_lp_y_.size() * sizeof(mpqs::uint512);
            cudaMalloc(&d_lp_correction, corr_bytes);
            cudaMemcpy(d_lp_correction, precomputed_lp_y_.data(), corr_bytes,
                       cudaMemcpyHostToDevice);
            sqrt_solver_->ApplyLPCorrection(d_lp_correction,
                                            static_cast<uint32_t>(precomputed_lp_y_.size()));
            cudaFree(d_lp_correction);
            LOG(LOG_DEBUG_1) << "Applied precomputed LP Y-correction for "
                             << precomputed_lp_y_.size() << " solutions.";
        }

        // Change 2 (always-on): surface solutions invalidated by HalveExponents.
        // d_valid[j] == 0 means HalveExponents found an odd accumulated exponent for
        // solution j — Y[j] is zeroed and the congruence X²≡Y² cannot hold.
        // Cost: n_solutions bytes D2H (~126 B at RSA-100 scale, unmeasurable).
        {
            std::vector<uint8_t> h_valid(n_solutions);
            cudaMemcpy(h_valid.data(), sqrt_solver_->getDeviceValid(),
                       n_solutions * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            int invalid_count = 0;
            for (int j = 0; j < n_solutions; ++j) {
                if (!h_valid[j]) {
                    ++invalid_count;
                    LOG(LOG_WARNING) << "Solution " << j
                                     << " invalidated by HalveExponents (odd exponent).";
                }
            }
            if (invalid_count > 0) {
                LOG(LOG_WARNING) << invalid_count << "/" << n_solutions
                                 << " solutions invalidated by HalveExponents"
                                 << " (odd accumulated exponent — GF(2) invariant violated?).";
            }
        }

        // Change 3 (always-on): unconditional X²≡Y² verification.
        // Downloads X and Y to host, computes modpow for each solution.
        // Cost: 2 × n_solutions × 64 B D2H + n_solutions modpow calls (~0.1 ms).
        {
            std::vector<mpqs::uint512> dbg_X(n_solutions), dbg_Y(n_solutions);
            cudaMemcpy(dbg_X.data(), sqrt_solver_->getDeviceX(),
                       n_solutions * sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg_Y.data(), sqrt_solver_->getDeviceY(),
                       n_solutions * sizeof(mpqs::uint512), cudaMemcpyDeviceToHost);
            int fail_count = 0;
            for (int j = 0; j < n_solutions; ++j) {
                mpqs::uint512 X2 = mpqs::math::modpow(dbg_X[j], 2, config_.N);
                mpqs::uint512 Y2 = mpqs::math::modpow(dbg_Y[j], 2, config_.N);
                if (X2 != Y2) {
                    ++fail_count;
                    LOG(LOG_ERROR_CRITICAL) << "X^2 != Y^2 mod N for solution " << j;
                }
            }
            if (fail_count == 0) {
                LOG(LOG_STATS) << "X²≡Y² (mod N) verified for all "
                               << n_solutions << " solutions.";
            }
        }

        // 6. BatchedGCD — use device pointers directly, no re-upload
        auto t_gcd_start = clock::now();
        auto factors = sqrt_solver_->BatchedGCD(
            sqrt_solver_->getDeviceX(),
            sqrt_solver_->getDeviceY(),
            n_solutions);
        auto t_done = clock::now();
        LOG(LOG_DEBUG_1) << "BatchedGCD+D2H:  "
                         << std::fixed << std::setprecision(2) << ms(t_gcd_start, t_done) << " ms";

        // No cudaFree needed — X/Y live in pool buffers

        // 8. Check GPU result
        if (!factors.first.is_zero() && !factors.second.is_zero()
            && !factors.first.is_one() && factors.first != config_.N) {
            LOG(LOG_INFO) << "Found Factors:";
            LOG(LOG_INFO) << " F1 = " << factors.first.to_string();
            LOG(LOG_INFO) << " F2 = " << factors.second.to_string();

            mpqs::uint512 product = factors.first;
            product.mult(factors.second);

            if (product != config_.N) {
                LOG(LOG_ERROR_CRITICAL) << "FAILURE: F1 * F2 != N";
                LOG(LOG_ERROR_CRITICAL) << "Computed Product: "
                                        << product.to_string();
            } else {
                LOG(LOG_DEBUG_1) << "Check: Product matches N.";
                result_factors_.insert(result_factors_.end(),
                                       { std::move(factors.first),
                                         std::move(factors.second) }
                );
                success = true;
            }
        }

        // 9. If GPU found no nontrivial factor, fall back to CPU loop
        if (!success) {
            LOG(LOG_WARNING) << "No nontrivial factor from GPU path. Falling back to CPU.";
            LOG(LOG_INFO) << "CPU fallback: trying " << kernel_solutions_.size() << " kernel vectors.";
            for (size_t i = 0; (i < kernel_solutions_.size()) && !success; ++i) {
                LOG(LOG_DEBUG_1) << "Processing Solution Vector "
                                 << (i+1) << " of "
                                 << kernel_solutions_.size();
                const mpqs::uint512* lp_corr_ptr =
                    (used_expanded_matrix_ && i < precomputed_lp_y_.size())
                    ? &precomputed_lp_y_[i] : nullptr;
                auto cpu_factors = sqrt_solver_->Perform(
                    kernel_solutions_[i],
                    host_relations_soa_,
                    f_data_.factorBase,
                    lp_corr_ptr
                );

                if (cpu_factors.first.is_zero() || cpu_factors.second.is_zero()) {
                    LOG(LOG_ERROR_CRITICAL) << "SANITY CHECK FAILED (X²≢Y²) for solution " << i;
                } else if (cpu_factors.first.is_one() || cpu_factors.first == config_.N) {
                    LOG(LOG_INFO) << "trivial GCD (X≡±Y mod N) for solution " << i;
                } else {
                    LOG(LOG_INFO) << "Found Factors:";
                    LOG(LOG_INFO) << " F1 = " << cpu_factors.first.to_string();
                    LOG(LOG_INFO) << " F2 = " << cpu_factors.second.to_string();

                    mpqs::uint512 product = cpu_factors.first;
                    product.mult(cpu_factors.second);

                    if (product != config_.N) {
                        LOG(LOG_ERROR_CRITICAL) << "FAILURE: F1 * F2 != N";
                        success = false;
                    } else {
                        LOG(LOG_DEBUG_1) << "Check: Product matches N.";
                        result_factors_.insert(result_factors_.end(),
                                               { std::move(cpu_factors.first),
                                                 std::move(cpu_factors.second) }
                        );
                        success = true;
                    }
                }
            }
        }
    }

    // Free expanded solution device buffer if allocated
    if (d_expanded_sol_data) {
        cudaFree(d_expanded_sol_data);
        d_expanded_sol_data = nullptr;
    }

    auto t_end = clock::now();
    double duration = std::chrono::duration<double>(t_end - t_start).count();
    LOG(LOG_INFO) << "Sqrt step completed in "
		  << FormatDuration(duration * 1000.0);
}

bool MPQSOrchestrator::shouldAutoApply() const {
    // Disabled if user explicitly said --autotune_no_history
    if (!config_.autotune_config.load_history) return false;
    // Only for modes that do sieving (or autotune-only, for warm start)
    return config_.mode == ExecutionMode::FULL_PIPELINE
        || config_.mode == ExecutionMode::SIEVE_ONLY
        || config_.mode == ExecutionMode::PARAM_TEST
        || config_.mode == ExecutionMode::AUTOTUNE_ONLY;
}

// =============================================================================
// bufferClusterPartials() — CSR-aware batch copy for expanded-matrix path
// =============================================================================

void MPQSOrchestrator::bufferClusterPartials(
    const structures::HostRelationBatch& partials)
{
    if (partials.num_relations == 0) return;
    for (size_t i = 0; i < partials.num_relations; ++i) {
        cluster_raw_partials_.sqrt_Q.push_back(partials.sqrt_Q[i]);
        cluster_raw_partials_.signs.push_back(partials.signs[i]);
        cluster_raw_partials_.val_2_exps.push_back(partials.val_2_exps[i]);
        cluster_raw_partials_.large_primes.push_back(partials.large_primes[i]);

        uint64_t fstart = partials.factor_offsets[i];
        uint64_t fend   = partials.factor_offsets[i + 1];
        for (uint64_t j = fstart; j < fend; ++j) {
            cluster_raw_partials_.factor_indices.push_back(partials.factor_indices[j]);
            cluster_raw_partials_.factor_counts.push_back(partials.factor_counts[j]);
        }
        cluster_raw_partials_.num_factors += (fend - fstart);
        cluster_raw_partials_.factor_offsets.push_back(cluster_raw_partials_.num_factors);
        cluster_raw_partials_.num_relations++;
    }
}

// =============================================================================
// networkLoop() — Thread A main loop (coordinator only, CPU-only)
// =============================================================================

void MPQSOrchestrator::networkLoop() {
    LOG(LOG_INFO) << "[Thread A] Network loop started"
                  << (comm_backend_ ? " (TCP active)" : " (local only)");

    // M6: Initialize raw partials buffer for expanded matrix path
    cluster_raw_partials_ = structures::HostRelationBatch{};
    cluster_raw_partials_.factor_offsets.push_back(0);  // CSR sentinel

    uint32_t workers_flushed = 0;
    uint32_t total_workers = comm_backend_ ? comm_backend_->peerCount() : 0;
    bool all_workers_flushed = (total_workers == 0);  // M1: no workers to flush

    // S8: Per-worker last recall timestamp for anti-thrashing (60s cooldown).
    // Accessed only from Thread A — no synchronization needed.
    std::unordered_map<uint8_t, std::chrono::steady_clock::time_point> last_recall_time_;

    // --- M3: Per-worker tracking for heartbeat timeout + chunk management ---
    struct WorkerTracker {
        std::chrono::steady_clock::time_point last_heartbeat;
        uint32_t current_chunk_id = UINT32_MAX;  // UINT32_MAX = no chunk in flight
        uint64_t total_rels = 0;
        bool alive = true;
    };
    std::unordered_map<uint8_t, WorkerTracker> worker_trackers;
    if (comm_backend_) {
        auto now = std::chrono::steady_clock::now();
        for (uint32_t w = 1; w <= total_workers; ++w) {
            WorkerTracker wt;
            wt.last_heartbeat = now;
            worker_trackers[static_cast<uint8_t>(w)] = wt;
        }
    }
    auto last_timeout_check = std::chrono::steady_clock::now();

    // --- SM3: Per-node contribution telemetry (coordinator only, in-memory) ---
    struct NodeTelemetry {
        uint8_t     node_id = 0;
        std::string gpu_name;
        uint64_t    full_relations = 0;
        uint64_t    partial_relations = 0;
        uint64_t    lp_combined = 0;        ///< CPU LP matches (coordinator only)
        double      first_relation_time = 0.0;  ///< Seconds since sieve_start
        double      last_relation_time = 0.0;
        uint32_t    chunks_completed = 0;
        uint64_t    a_values_consumed = 0;
        uint64_t    assigned_range = 0;     ///< A-values assigned (informational)
    };
    auto sieve_start = std::chrono::steady_clock::now();
    std::vector<NodeTelemetry> node_telemetry(1 + total_workers);
    {
        cudaDeviceProp coord_prop{};
        cudaGetDeviceProperties(&coord_prop, 0);
        node_telemetry[0].node_id = 0;
        node_telemetry[0].gpu_name = std::string(coord_prop.name);
    }
    for (uint32_t w = 1; w <= total_workers; ++w) {
        auto info = comm_backend_->peerInfo(static_cast<uint8_t>(w));
        node_telemetry[w].node_id = static_cast<uint8_t>(w);
        node_telemetry[w].gpu_name = std::string(info.gpu_name);
    }

    // --- Aggregate progress tracking (Feature 2) ---
    SieveProgressTracker cluster_progress_tracker;
    uint64_t last_sample_rels = 0;
    auto     last_log_time    = sieve_start;

    /// Sample current progress, update ETA model, and emit a throttled (~5s) status line.
    auto sampleProgress = [&]() {
        auto     now     = std::chrono::steady_clock::now();
        double   elapsed = std::chrono::duration<double>(now - sieve_start).count();
        uint64_t total   = cluster_accumulator_->totalRelations();
        uint64_t target  = cluster_accumulator_->effectiveTarget();
        if (total != last_sample_rels) {
            cluster_progress_tracker.addSample(elapsed, total, target);
            last_sample_rels = total;
        }
        if (now - last_log_time < std::chrono::seconds(5)) return;
        last_log_time = now;

        double pct  = (target > 0) ? 100.0 * static_cast<double>(total) / target : 0.0;
        double rate = cluster_progress_tracker.smoothed_rate;
        LOG(LOG_STATS) << "[Cluster] Progress: " << total << " / " << target
                       << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                       << " | " << std::fixed << std::setprecision(1) << rate << " rel/s";
        if (cluster_progress_tracker.hasETA()) {
            double eta = cluster_progress_tracker.current_eta_sec;
            double est = elapsed + eta;
            int h = static_cast<int>(est) / 3600;
            int m = (static_cast<int>(est) % 3600) / 60;
            int s = static_cast<int>(est) % 60;
            LOG(LOG_STATS) << "[Cluster] ETA: " << std::fixed << std::setprecision(1)
                           << eta << "s | Total est.: "
                           << std::setfill('0') << std::setw(2) << h << ":"
                           << std::setw(2) << m << ":" << std::setw(2) << s
                           << std::setfill(' ');
        }
        if (cluster_cpu_lp_) {
            uint64_t wit  = cluster_cpu_lp_->witnesses();
            uint64_t comb = cluster_cpu_lp_->totalCombines();
            uint64_t ins  = cluster_cpu_lp_->totalInserts();
            double wit_rate  = (elapsed > 0.0) ? static_cast<double>(wit)  / elapsed : 0.0;
            double comb_rate = (elapsed > 0.0) ? static_cast<double>(comb) / elapsed : 0.0;
            double yield     = (ins  > 0)      ? 100.0 * static_cast<double>(comb) / ins : 0.0;
            LOG(LOG_STATS) << "[Cluster] LP: " << wit << " witnesses ("
                           << std::fixed << std::setprecision(1) << wit_rate << "/s)"
                           << " | " << comb << " combines ("
                           << std::fixed << std::setprecision(1) << comb_rate << "/s)"
                           << " | yield " << std::fixed << std::setprecision(1) << yield << "%";
        }
    };

    // S8: Straggler detection lambda — runs after each CHUNK_COMPLETE.
    // Sends CHUNK_RECALL to a worker whose estimated remaining time exceeds 2×
    // the fastest worker's estimated remaining time.
    // Anti-thrashing: 30s minimum execution before first recall, 60s cooldown per worker.
    auto checkStragglers = [&]() {
        if (!cluster_work_pool_ || !cluster_scheduler_ || !comm_backend_) return;

        auto now_s = std::chrono::steady_clock::now();

        // Find fastest calibrated worker throughput (in rels/s, as proxy for speed).
        double fastest_rels_per_sec = 0.0;
        for (uint32_t wid = 1; wid <= total_workers; ++wid) {
            const auto& tp = cluster_scheduler_->throughput(static_cast<uint8_t>(wid));
            if (tp.chunks_completed > 0) {
                fastest_rels_per_sec = std::max(fastest_rels_per_sec, tp.rels_per_sec);
            }
        }
        if (fastest_rels_per_sec <= 0.0) return;

        // Check each in-flight chunk for straggler condition.
        for (const auto& cow : cluster_work_pool_->inFlightChunks()) {
            uint8_t wid = cow.worker_id;
            const auto& tp = cluster_scheduler_->throughput(wid);
            // EMA must have at least one completed chunk before recall.
            if (tp.a_vals_per_sec <= 0.0 || tp.chunks_completed < 1) continue;

            double elapsed = std::chrono::duration<double>(
                now_s - cow.checkout_time).count();

            // Anti-thrashing: minimum 30s execution before first recall.
            if (elapsed < 30.0) continue;

            // Anti-thrashing: 60s cooldown between recalls per worker.
            auto last_it = last_recall_time_.find(wid);
            if (last_it != last_recall_time_.end()) {
                double since_last = std::chrono::duration<double>(
                    now_s - last_it->second).count();
                if (since_last < 60.0) continue;
            }

            // Straggler criterion: worker estimated remaining > 2× fastest estimated remaining.
            double remaining_a = static_cast<double>(cow.unit.count);
            double straggler_time = remaining_a / tp.a_vals_per_sec;
            double fastest_time   = remaining_a / fastest_rels_per_sec;

            if (straggler_time > 2.0 * fastest_time) {
                LOG(LOG_INFO) << "[Thread A] Straggler detected — worker " << (int)wid
                              << " estimated " << std::fixed << std::setprecision(1)
                              << straggler_time << "s remaining"
                              << " (fastest: " << fastest_time << "s). Sending CHUNK_RECALL.";

                cluster::ChunkRecallPayload cr{};
                cr.chunk_id = cow.chunk_id;
                comm_backend_->send(wid, cluster::MsgType::CHUNK_RECALL, &cr, sizeof(cr));
                last_recall_time_[wid] = now_s;
            }
        }
    };

    while (true) {
        // 1. Poll DirectChannel (M1/M3: local sieve via Thread B)
        if (cluster_channel_) {
            cluster::DirectChannel::Payload payload;
            while (cluster_channel_->tryPop(payload)) {
                if (payload.full.num_relations > 0) {
                    cluster_accumulator_->addRelations(payload.full, /*source_id=*/0);
                    // Telemetry: node 0 full relations
                    double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - sieve_start).count();
                    auto& t0 = node_telemetry[0];
                    t0.full_relations += payload.full.num_relations;
                    if (t0.first_relation_time == 0.0) t0.first_relation_time = elapsed;
                    t0.last_relation_time = elapsed;
                }
                if (cluster_cpu_lp_ && payload.partials.num_relations > 0) {
                    bufferClusterPartials(payload.partials);  // M6
                    // Telemetry: node 0 partials + LP-combined
                    node_telemetry[0].partial_relations += payload.partials.num_relations;
                    uint64_t lp_before = cluster_accumulator_->relationsFrom(255);
                    cluster_cpu_lp_->insertAndMatch(payload.partials, *cluster_accumulator_);
                    node_telemetry[0].lp_combined +=
                        cluster_accumulator_->relationsFrom(255) - lp_before;
                }
                sampleProgress();
            }
        }

        // 2. Poll CommBackend (M2+: remote workers via TCP)
        if (comm_backend_) {
            cluster::RecvMessage msg;
            auto now = std::chrono::steady_clock::now();
            while (comm_backend_->recv(msg)) {
                if (!msg.valid) continue;

                switch (msg.type) {
                case cluster::MsgType::INCREMENTAL_BATCH: {
                    structures::HostRelationBatch full_batch, partial_batch;
                    if (cluster::deserializeIncrementalBatch(
                            msg.payload.data(), msg.payload.size(),
                            full_batch, partial_batch)) {
                        if (full_batch.num_relations > 0) {
                            cluster_accumulator_->addRelations(
                                full_batch, /*source_id=*/msg.sender_id);
                        }
                        if (cluster_cpu_lp_ && partial_batch.num_relations > 0) {
                            bufferClusterPartials(partial_batch);  // M6
                            uint64_t lp_before = cluster_accumulator_->relationsFrom(255);
                            cluster_cpu_lp_->insertAndMatch(
                                partial_batch, *cluster_accumulator_);
                            node_telemetry[0].lp_combined +=
                                cluster_accumulator_->relationsFrom(255) - lp_before;
                        }
                        // Telemetry: per-sender accumulation
                        if (msg.sender_id < node_telemetry.size()) {
                            double elapsed = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - sieve_start).count();
                            auto& t = node_telemetry[msg.sender_id];
                            t.full_relations += full_batch.num_relations;
                            t.partial_relations += partial_batch.num_relations;
                            if (t.first_relation_time == 0.0 && full_batch.num_relations > 0)
                                t.first_relation_time = elapsed;
                            if (full_batch.num_relations > 0 || partial_batch.num_relations > 0)
                                t.last_relation_time = elapsed;
                        }
                        sampleProgress();
                    } else {
                        LOG(LOG_WARNING) << "[Thread A] Failed to deserialize INCREMENTAL_BATCH from worker "
                                         << (int)msg.sender_id;
                    }
                    break;
                }

                // --- M3: CHUNK_COMPLETE handler ---
                case cluster::MsgType::CHUNK_COMPLETE: {
                    if (msg.payload.size() >= sizeof(cluster::ChunkCompletePayload)) {
                        cluster::ChunkCompletePayload cc;
                        std::memcpy(&cc, msg.payload.data(), sizeof(cc));

                        // Record completion in scheduler (EMA throughput update)
                        if (cluster_scheduler_) {
                            cluster_scheduler_->recordCompletion(
                                msg.sender_id, cc.relations_found,
                                cc.a_values_consumed,
                                cc.elapsed_ms / 1000.0);
                        }
                        // Mark chunk done in WorkPool.
                        // reclaimPartial handles both full and partial completions:
                        // if a_values_consumed < assigned, the remainder is returned
                        // to WorkPool::returned_ for redistribution.
                        if (cluster_work_pool_) {
                            cluster_work_pool_->reclaimPartial(cc.chunk_id,
                                                               cc.a_values_consumed);
                        }

                        auto& wt = worker_trackers[msg.sender_id];
                        wt.total_rels += cc.relations_found;
                        wt.current_chunk_id = UINT32_MAX;

                        // Telemetry: chunks and a-values consumed per worker
                        if (msg.sender_id < node_telemetry.size()) {
                            auto& t = node_telemetry[msg.sender_id];
                            t.chunks_completed += 1;
                            t.a_values_consumed += cc.a_values_consumed;
                        }

                        LOG(LOG_INFO) << "[Thread A] CHUNK_COMPLETE from worker "
                                      << (int)msg.sender_id
                                      << ": chunk=" << cc.chunk_id
                                      << " rels=" << cc.relations_found
                                      << " partials=" << cc.partials_found
                                      << " time=" << cc.elapsed_ms << "ms"
                                      << " a_vals=" << cc.a_values_consumed;

                        // Assign next overflow chunk (if work remains and target not reached)
                        if (!cluster_accumulator_->targetReached() &&
                            cluster_work_pool_ && !cluster_work_pool_->exhausted()) {
                            uint64_t next_size = cluster_scheduler_->nextChunkSize(msg.sender_id);
                            auto checkout = cluster_work_pool_->checkoutWork(
                                next_size, msg.sender_id);
                            if (checkout) {
                                cluster::ChunkAssignPayload ca{};
                                ca.chunk_id = checkout->chunk_id;
                                ca.poly_range_start = checkout->unit.start;
                                ca.poly_range_count = checkout->unit.count;
                                ca.flags = cluster::ChunkAssignPayload::FLAG_OVERFLOW;
                                comm_backend_->send(msg.sender_id,
                                    cluster::MsgType::CHUNK_ASSIGN,
                                    &ca, sizeof(ca));
                                wt.current_chunk_id = checkout->chunk_id;
                                LOG(LOG_INFO) << "[Thread A] Assigned overflow chunk "
                                              << checkout->chunk_id << " to worker "
                                              << (int)msg.sender_id
                                              << " [" << ca.poly_range_start << ", "
                                              << ca.poly_range_start + ca.poly_range_count << ")";
                            } else {
                                LOG(LOG_INFO) << "[Thread A] No overflow work for worker "
                                              << (int)msg.sender_id << " (pool exhausted)";
                            }
                        } else {
                            LOG(LOG_DEBUG_1) << "[Thread A] No more work for worker "
                                             << (int)msg.sender_id
                                             << " (target reached or pool exhausted)";
                        }

                        // S8: Check for stragglers after each CHUNK_COMPLETE.
                        checkStragglers();
                    }
                    break;
                }

                case cluster::MsgType::HEARTBEAT: {
                    if (msg.payload.size() >= sizeof(cluster::HeartbeatPayload)) {
                        cluster::HeartbeatPayload hb;
                        std::memcpy(&hb, msg.payload.data(), sizeof(hb));
                        // M3: record timestamp for timeout detection
                        worker_trackers[msg.sender_id].last_heartbeat = now;
                        LOG(LOG_DEBUG_1) << "[Thread A] Heartbeat from worker "
                                         << (int)hb.worker_id
                                         << ": batches=" << hb.batches_completed;
                    }
                    break;
                }
                case cluster::MsgType::FLUSH_ACK: {
                    ++workers_flushed;
                    LOG(LOG_INFO) << "[Thread A] FLUSH_ACK from worker "
                                  << (int)msg.sender_id
                                  << " (" << workers_flushed << "/" << total_workers << ")";
                    if (workers_flushed >= total_workers) {
                        all_workers_flushed = true;
                    }
                    break;
                }
                default:
                    LOG(LOG_DEBUG_1) << "[Thread A] Ignoring message type "
                                     << static_cast<int>(static_cast<uint8_t>(msg.type))
                                     << " from " << (int)msg.sender_id;
                    break;
                }
            }

            // --- M3: Heartbeat timeout check (every 5 seconds) ---
            if (now - last_timeout_check > std::chrono::seconds(5)) {
                last_timeout_check = now;
                for (auto& [wid, wt] : worker_trackers) {
                    if (!wt.alive) continue;
                    auto since_heartbeat = std::chrono::duration_cast<
                        std::chrono::seconds>(now - wt.last_heartbeat).count();
                    if (since_heartbeat > static_cast<long>(
                            cluster::kFlushTimeoutMs / 1000)) {
                        LOG(LOG_WARNING) << "[Thread A] Worker " << (int)wid
                                         << " heartbeat timeout (" << since_heartbeat
                                         << "s since last heartbeat)";
                        wt.alive = false;

                        // Reclaim in-flight work
                        if (cluster_work_pool_ && wt.current_chunk_id != UINT32_MAX) {
                            cluster_work_pool_->reclaimWork(wid);
                            LOG(LOG_INFO) << "[Thread A] Reclaimed in-flight chunk "
                                          << wt.current_chunk_id << " from worker "
                                          << (int)wid;
                            wt.current_chunk_id = UINT32_MAX;
                        }
                        if (comm_backend_) {
                            comm_backend_->disconnectPeer(wid);
                        }
                        // Count as flushed (won't get FLUSH_ACK from dead worker)
                        ++workers_flushed;
                        if (workers_flushed >= total_workers) {
                            all_workers_flushed = true;
                        }
                    }
                }
            }
        }

        // 3a. Estimate-only: time-limited probe — trigger STOP after probe_timeout
        if (config_.estimate_only) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - sieve_start).count();
            if (elapsed > config_.probe_timeout) {
                LOG(LOG_INFO) << "[Thread A] Estimate probe done after "
                              << std::fixed << std::setprecision(1) << elapsed
                              << "s (timeout=" << config_.probe_timeout << "s)";

                // Reuse exact targetReached() STOP mechanism
                external_stop_flag_ = true;
                if (cluster_channel_) {
                    cluster_channel_->signalStop();
                }
                if (comm_backend_) {
                    cluster::StopPayload stop{};
                    stop.reason = 0;
                    comm_backend_->broadcast(cluster::MsgType::STOP, &stop, sizeof(stop));
                    LOG(LOG_INFO) << "[Thread A] STOP broadcast sent to "
                                  << total_workers << " worker(s) (estimate probe)";
                }

                // Wait for worker flush (same pattern as targetReached)
                if (comm_backend_ && !all_workers_flushed) {
                    auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(cluster::kFlushTimeoutMs);
                    while (!all_workers_flushed &&
                           std::chrono::steady_clock::now() < deadline) {
                        cluster::RecvMessage msg;
                        while (comm_backend_->recv(msg)) {
                            if (!msg.valid) continue;
                            if (msg.type == cluster::MsgType::INCREMENTAL_BATCH) {
                                structures::HostRelationBatch fb, pb;
                                if (cluster::deserializeIncrementalBatch(
                                        msg.payload.data(), msg.payload.size(), fb, pb)) {
                                    if (fb.num_relations > 0)
                                        cluster_accumulator_->addRelations(fb, msg.sender_id);
                                    if (cluster_cpu_lp_ && pb.num_relations > 0) {
                                        bufferClusterPartials(pb);  // M6
                                        cluster_cpu_lp_->insertAndMatch(pb, *cluster_accumulator_);
                                    }
                                }
                            } else if (msg.type == cluster::MsgType::FLUSH_ACK) {
                                ++workers_flushed;
                                if (workers_flushed >= total_workers)
                                    all_workers_flushed = true;
                            }
                        }
                        if (!all_workers_flushed)
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }

                // Deliver whatever we have (no extractFinal — estimate doesn't need relations)
                auto final_batch = cluster_accumulator_->extractFinal();
                cluster_handoff_->deliver(std::move(final_batch));
                break;
            }
        }

        // 3b. Check if target reached
        if (cluster_accumulator_->targetReached()) {
            LOG(LOG_INFO) << "[Thread A] Target reached: "
                          << cluster_accumulator_->totalRelations()
                          << " / " << cluster_accumulator_->effectiveTarget();

            // Signal all sources to stop
            external_stop_flag_ = true;  // Sub-batch stop (checked by runSievingBatch)
            if (cluster_channel_) {
                cluster_channel_->signalStop();
            }
            if (comm_backend_) {
                cluster::StopPayload stop{};
                stop.reason = 0;  // target_reached
                comm_backend_->broadcast(cluster::MsgType::STOP, &stop, sizeof(stop));
                LOG(LOG_INFO) << "[Thread A] STOP broadcast sent to "
                              << total_workers << " worker(s)";
            }

            // Wait for all workers to flush (with timeout)
            if (comm_backend_ && !all_workers_flushed) {
                auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(cluster::kFlushTimeoutMs);
                while (!all_workers_flushed &&
                       std::chrono::steady_clock::now() < deadline) {
                    cluster::RecvMessage msg;
                    while (comm_backend_->recv(msg)) {
                        if (!msg.valid) continue;
                        if (msg.type == cluster::MsgType::INCREMENTAL_BATCH) {
                            structures::HostRelationBatch fb, pb;
                            if (cluster::deserializeIncrementalBatch(
                                    msg.payload.data(), msg.payload.size(), fb, pb)) {
                                if (fb.num_relations > 0)
                                    cluster_accumulator_->addRelations(fb, msg.sender_id);
                                if (cluster_cpu_lp_ && pb.num_relations > 0) {
                                    bufferClusterPartials(pb);  // M6
                                    uint64_t lp_before = cluster_accumulator_->relationsFrom(255);
                                    cluster_cpu_lp_->insertAndMatch(pb, *cluster_accumulator_);
                                    node_telemetry[0].lp_combined +=
                                        cluster_accumulator_->relationsFrom(255) - lp_before;
                                }
                                // Telemetry: flush-drain contributions
                                if (msg.sender_id < node_telemetry.size()) {
                                    double elapsed = std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - sieve_start).count();
                                    auto& t = node_telemetry[msg.sender_id];
                                    t.full_relations += fb.num_relations;
                                    t.partial_relations += pb.num_relations;
                                    if (t.first_relation_time == 0.0 && fb.num_relations > 0)
                                        t.first_relation_time = elapsed;
                                    if (fb.num_relations > 0 || pb.num_relations > 0)
                                        t.last_relation_time = elapsed;
                                }
                            }
                        } else if (msg.type == cluster::MsgType::FLUSH_ACK) {
                            ++workers_flushed;
                            LOG(LOG_INFO) << "[Thread A] FLUSH_ACK from worker "
                                          << (int)msg.sender_id << " (post-STOP)";
                            if (workers_flushed >= total_workers) {
                                all_workers_flushed = true;
                            }
                        }
                        // Late CHUNK_COMPLETEs after STOP: ignore (work is done)
                    }
                    if (!all_workers_flushed) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                if (!all_workers_flushed) {
                    LOG(LOG_WARNING) << "[Thread A] Flush timeout: "
                                     << workers_flushed << "/" << total_workers << " flushed";
                }
            }

            // Deliver final batch
            auto final_batch = cluster_accumulator_->extractFinal();
            LOG(LOG_INFO) << "[Thread A] Delivering " << final_batch.num_relations
                          << " relations to Thread B";
            cluster_handoff_->deliver(std::move(final_batch));
            break;
        }

        // 4. Check if all sources exhausted without reaching target
        bool local_done = !cluster_channel_ || cluster_queue_->isSieveDone();
        if (local_done && all_workers_flushed) {
            LOG(LOG_WARNING) << "[Thread A] All sources done before target: "
                             << cluster_accumulator_->totalRelations()
                             << " / " << cluster_accumulator_->effectiveTarget();

            if (cluster_channel_) {
                cluster::DirectChannel::Payload p;
                while (cluster_channel_->tryPop(p)) {
                    if (p.full.num_relations > 0) {
                        cluster_accumulator_->addRelations(p.full, 0);
                        // Telemetry: node 0 tail drain
                        double elapsed = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - sieve_start).count();
                        auto& t0 = node_telemetry[0];
                        t0.full_relations += p.full.num_relations;
                        if (t0.first_relation_time == 0.0) t0.first_relation_time = elapsed;
                        t0.last_relation_time = elapsed;
                    }
                    if (cluster_cpu_lp_ && p.partials.num_relations > 0) {
                        bufferClusterPartials(p.partials);  // M6
                        node_telemetry[0].partial_relations += p.partials.num_relations;
                        uint64_t lp_before = cluster_accumulator_->relationsFrom(255);
                        cluster_cpu_lp_->insertAndMatch(p.partials, *cluster_accumulator_);
                        node_telemetry[0].lp_combined +=
                            cluster_accumulator_->relationsFrom(255) - lp_before;
                    }
                }
            }

            auto final_batch = cluster_accumulator_->extractFinal();
            cluster_handoff_->deliver(std::move(final_batch));
            break;
        }

        // 5. Brief sleep to avoid busy-wait
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    LOG(LOG_INFO) << "[Thread A] Network loop exiting"
                  << " (accumulated " << cluster_accumulator_->totalRelations()
                  << " relations)";

    // Log per-source breakdown
    if (comm_backend_) {
        LOG(LOG_INFO) << "[Thread A] Relations by source: coord="
                      << cluster_accumulator_->relationsFrom(0);
        for (auto& [wid, wt] : worker_trackers) {
            LOG(LOG_INFO) << "[Thread A]   worker " << (int)wid
                          << ": " << cluster_accumulator_->relationsFrom(wid)
                          << " rels" << (wt.alive ? "" : " (DEAD)");
        }
    }

    // Log CPU LP statistics
    if (cluster_cpu_lp_) {
        LOG(LOG_INFO) << "[Thread A] CPU LP: inserts=" << cluster_cpu_lp_->totalInserts()
                      << " matches=" << cluster_cpu_lp_->totalMatches()
                      << " combines=" << cluster_cpu_lp_->totalCombines()
                      << " witnesses=" << cluster_cpu_lp_->witnesses();
    }

    // --- SM3: Cluster sieve telemetry table (coordinator only) ---
    if (config_.cluster_mode == ClusterMode::COORDINATOR && !node_telemetry.empty()) {
        LOG(LOG_INFO) << "=== Cluster Sieve Telemetry ===";
        LOG(LOG_INFO) << "Node | GPU              | Full    | Partial | LP-comb "
                      << "| Duration | Full/s  | Part/s  | Chunks | A-vals";

        uint64_t total_full = 0, total_partial = 0, total_lp = 0;
        uint32_t total_chunks = 0;
        uint64_t total_a = 0;

        for (auto& t : node_telemetry) {
            double duration = t.last_relation_time - t.first_relation_time;
            if (duration < 0.01) duration = 0.01;  // avoid div-by-zero
            double full_rate    = static_cast<double>(t.full_relations)    / duration;
            double partial_rate = static_cast<double>(t.partial_relations) / duration;

            LOG(LOG_INFO) << std::setw(4) << static_cast<int>(t.node_id) << " | "
                          << std::setw(16) << std::left
                          << t.gpu_name.substr(0, 16) << std::right
                          << " | " << std::setw(7) << t.full_relations
                          << " | " << std::setw(7) << t.partial_relations
                          << " | " << std::setw(7) << t.lp_combined
                          << " | " << std::setw(7) << std::fixed << std::setprecision(1)
                          << duration << "s"
                          << " | " << std::setw(7) << std::fixed << std::setprecision(1)
                          << full_rate
                          << " | " << std::setw(7) << std::fixed << std::setprecision(1)
                          << partial_rate
                          << " | " << std::setw(6) << t.chunks_completed
                          << " | " << std::setw(8) << t.a_values_consumed;

            total_full    += t.full_relations;
            total_partial += t.partial_relations;
            total_lp      += t.lp_combined;
            total_chunks  += t.chunks_completed;
            total_a       += t.a_values_consumed;
        }

        LOG(LOG_INFO) << "Total"
                      << std::string(19, ' ')
                      << "| " << std::setw(7) << total_full
                      << " | " << std::setw(7) << total_partial
                      << " | " << std::setw(7) << total_lp
                      << " |          |         |         "
                      << "| " << std::setw(6) << total_chunks
                      << " | " << std::setw(8) << total_a;
    }

    // --- Aggregate end-of-run summary ---
    double   elapsed_total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sieve_start).count();
    uint64_t final_rels   = cluster_accumulator_->totalRelations();
    double   throughput   = (elapsed_total > 0.0)
                          ? static_cast<double>(final_rels) / elapsed_total : 0.0;
    LOG(LOG_INFO) << "[Thread A] Aggregate: " << final_rels << " rels in "
                  << std::fixed << std::setprecision(1) << elapsed_total << "s"
                  << " (" << std::fixed << std::setprecision(1) << throughput << " rel/s)";
    if (cluster_cpu_lp_) {
        uint64_t comb = cluster_cpu_lp_->totalCombines();
        uint64_t ins  = cluster_cpu_lp_->totalInserts();
        double   yld  = (ins > 0) ? 100.0 * static_cast<double>(comb) / ins : 0.0;
        LOG(LOG_INFO) << "[Thread A] LP yield: " << std::fixed << std::setprecision(1)
                      << yld << "% (" << comb << " combines / " << ins << " inserts)";
    }

    // Persist telemetry to member variables — read by main thread after join().
    cluster_sieve_elapsed_sec_ = elapsed_total;
    cluster_final_throughput_  = throughput;
    cluster_final_total_rels_  = final_rels;
}

} // namespace mpqs
