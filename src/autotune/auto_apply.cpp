// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "auto_apply.h"
#include "memory_estimator.h"
#include "orchestrator.h"   // MPQSConfig full definition
#include "hpc_logger.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace mpqs::autotune {

AutoApplyController::AutoApplyController(MPQSConfig& config,
                                         const mpqs::sieve::factoringData& f_data)
    : config_(config)
    , f_data_(f_data)
{
    N_bits_   = f_data_.N.msb() + 1;
    N_digits_ = static_cast<uint32_t>(f_data_.N.to_string().size());
    N_hash_   = sha256_hex(f_data_.N.to_string());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    gpu_name_ = prop.name;
}

void AutoApplyController::loadHistories() {
    // Per-GPU history
    if (!config_.autotune_config.history_file.empty())
        history_.load(config_.autotune_config.history_file);

    // Benign history
    if (!config_.autotune_config.benign_history_file.empty()) {
        benign_.load(config_.autotune_config.benign_history_file);
        if (benign_.size() == 0) {
            benign_.loadDefaults();
            benign_.save(config_.autotune_config.benign_history_file);
        }
    }
}

void AutoApplyController::mergeSieveParams() {
    // Cascade: exact match → K-nearest → benign → current config (TuningStage heuristic)

    // Tier 1: Exact history match
    const HistoryEntry* exact = history_.findExact(N_hash_);
    if (exact) {
        if (!config_.isPinned("fb_bound") && exact->optimal_params.fb_bound > 0) {
            result_.fb_bound = exact->optimal_params.fb_bound;
            config_.fb_bound = result_.fb_bound;
            result_.sources.push_back({"fb_bound", "history",
                "exact match, conf=" + std::to_string(exact->confidence).substr(0, 4)});
        }
        if (!config_.isPinned("sieve_bound") && exact->optimal_params.sieve_bound > 0) {
            result_.sieve_bound = exact->optimal_params.sieve_bound;
            config_.sieve_bound = result_.sieve_bound;
            result_.sources.push_back({"sieve_bound", "history",
                "exact match, conf=" + std::to_string(exact->confidence).substr(0, 4)});
        }
        if (!config_.isPinned("lp1_bound")) {
            result_.lp1_bound = exact->optimal_params.lp1_bound;
            config_.lp1_bound = result_.lp1_bound;
            result_.sources.push_back({"lp1_bound", "history",
                "exact match"});
        }
        // Kernel params: apply if same GPU
        if (exact->environment.gpu_name == gpu_name_) {
            bool has_kp = false;
            for (int i = 0; i < 8; ++i)
                if (exact->optimal_params.kernel_params[i] > 0) { has_kp = true; break; }
            if (has_kp && !config_.isPinned("params")) {
                for (int i = 0; i < 8; ++i)
                    config_.params[i] = exact->optimal_params.kernel_params[i];
                config_.useParams = true;
                result_.kernel_params_applied = true;
            }
        }

        result_.confidence = exact->confidence;
        result_.applied = true;

        LOG(LOG_INFO) << "[AutoApply] Exact history match for "
                      << exact->digit_count << "-digit N (conf="
                      << std::to_string(exact->confidence).substr(0, 4)
                      << "): F=" << result_.fb_bound
                      << " M=" << result_.sieve_bound
                      << " L=" << result_.lp1_bound;
        return;
    }

    // Tier 2: K-nearest neighbors (interpolation by bit_length)
    auto neighbors = history_.findKNearest(N_bits_, 3);

    // Distance gate: skip KNN if nearest neighbor is >50 bits (~15 digits) away
    // to prevent wild extrapolation from distant history entries
    if (!neighbors.empty()) {
        uint32_t min_dist = UINT32_MAX;
        for (const auto* n : neighbors) {
            uint32_t d = (N_bits_ > n->bit_length)
                       ? (N_bits_ - n->bit_length) : (n->bit_length - N_bits_);
            min_dist = std::min(min_dist, d);
        }
        if (min_dist > 50) {
            LOG(LOG_DEBUG_1) << "[AutoApply] KNN skipped: nearest neighbor "
                             << min_dist << " bits away (threshold: 50)";
            neighbors.clear();
        }
    }

    if (!neighbors.empty()) {
        double total_weight = 0.0;
        double w_fb = 0.0, w_M = 0.0, w_lp = 0.0;
        double max_knn_conf = 0.0;
        for (const auto* n : neighbors) {
            double dist = std::abs(static_cast<double>(N_bits_) - n->bit_length);
            double w = 1.0 / (1.0 + dist);
            w_fb += w * n->optimal_params.fb_bound;
            w_M  += w * n->optimal_params.sieve_bound;
            w_lp += w * n->optimal_params.lp1_bound;
            total_weight += w;
            max_knn_conf = std::max(max_knn_conf, n->confidence);
        }

        // Confidence gate: prefer validated benign entry over low-confidence KNN
        const auto* benign_check = benign_.findByDigits(N_digits_);
        if (benign_check && benign_check->confidence > max_knn_conf) {
            LOG(LOG_INFO) << "[AutoApply] Benign override: conf="
                          << std::to_string(benign_check->confidence).substr(0, 4)
                          << " > KNN source conf="
                          << std::to_string(max_knn_conf).substr(0, 4)
                          << ", falling through to benign tier";
            // Don't apply KNN — fall through to Tier 3
        } else {
            if (total_weight > 0 && !config_.isPinned("fb_bound")) {
                result_.fb_bound = static_cast<uint32_t>(w_fb / total_weight);
                config_.fb_bound = result_.fb_bound;
                result_.sources.push_back({"fb_bound", "history",
                    "K-nearest (" + std::to_string(neighbors.size()) + " neighbors)"});
            }
            if (total_weight > 0 && !config_.isPinned("sieve_bound")) {
                result_.sieve_bound = static_cast<uint32_t>(w_M / total_weight);
                config_.sieve_bound = result_.sieve_bound;
                result_.sources.push_back({"sieve_bound", "history",
                    "K-nearest"});
            }
            if (total_weight > 0 && !config_.isPinned("lp1_bound")) {
                result_.lp1_bound = static_cast<uint64_t>(w_lp / total_weight);
                config_.lp1_bound = result_.lp1_bound;
                result_.sources.push_back({"lp1_bound", "history",
                    "K-nearest"});
            }

            result_.confidence = 0.6;  // Lower than exact match
            result_.applied = true;

            LOG(LOG_INFO) << "[AutoApply] K-nearest match (" << neighbors.size()
                          << " neighbors): F=" << result_.fb_bound
                          << " M=" << result_.sieve_bound
                          << " L=" << result_.lp1_bound;
            return;
        }
    }

    // Tier 3: Benign history
    const auto* benign_entry = benign_.findByDigits(N_digits_);
    if (benign_entry) {
        if (!config_.isPinned("fb_bound") && benign_entry->fb_bound > 0) {
            result_.fb_bound = benign_entry->fb_bound;
            config_.fb_bound = result_.fb_bound;
            result_.sources.push_back({"fb_bound", "benign",
                "cross-GPU baseline"});
        }
        if (!config_.isPinned("sieve_bound") && benign_entry->sieve_bound > 0) {
            result_.sieve_bound = benign_entry->sieve_bound;
            config_.sieve_bound = result_.sieve_bound;
            result_.sources.push_back({"sieve_bound", "benign",
                "cross-GPU baseline"});
        }
        if (!config_.isPinned("lp1_bound")) {
            result_.lp1_bound = benign_entry->lp1_bound;
            config_.lp1_bound = result_.lp1_bound;
            result_.sources.push_back({"lp1_bound", "benign",
                "cross-GPU baseline"});
        }

        result_.confidence = benign_entry->confidence;
        result_.applied = true;

        LOG(LOG_INFO) << "[AutoApply] Benign history match: F="
                      << result_.fb_bound << " M=" << result_.sieve_bound
                      << " L=" << result_.lp1_bound;
        return;
    }

    // Tier 4: No history data — leave TuningStage heuristic defaults
    LOG(LOG_DEBUG_1) << "[AutoApply] No matching history — using TuningStage defaults";
}

void AutoApplyController::mergeBufferParams() {
    // Cascade: pinned CLI → history telemetry → benign → defaults

    // Tier 1: History-based buffer sizing
    const HistoryEntry* exact = history_.findExact(N_hash_);
    if (exact) {
        const auto& mp = exact->measured_performance;
        const auto& op = exact->optimal_params;

        // Witness capacity
        if (!config_.isPinned("lp1_max_witness_capacity")) {
            if (op.recommended_witness_capacity > 0) {
                result_.lp1_max_witness_capacity = op.recommended_witness_capacity;
                config_.lp1_max_witness_capacity = result_.lp1_max_witness_capacity;
                result_.sources.push_back({"lp1_max_witness_capacity", "history",
                    "stored rec=" + std::to_string(op.recommended_witness_capacity)});
            } else if (mp.witness_capacity > 0) {
                // Derive from measured capacity + peak fill
                double safety = (mp.overflow_events > 0) ? 1.5 : 1.3;
                uint64_t rec = static_cast<uint64_t>(mp.witness_capacity * safety);
                int log2_w = (rec > 1) ? 64 - __builtin_clzll(rec - 1) : 0;
                log2_w = std::min(log2_w, 24);
                result_.lp1_max_witness_capacity = 1ULL << log2_w;
                config_.lp1_max_witness_capacity = result_.lp1_max_witness_capacity;
                result_.sources.push_back({"lp1_max_witness_capacity", "history",
                    "derived from peak, safety=" + std::to_string(safety).substr(0, 3)});
            }
        }

        // Partial buffer
        if (!config_.isPinned("partial_buffer_size") && op.recommended_partial_buffer > 0) {
            result_.partial_buffer_size = op.recommended_partial_buffer;
            if (result_.partial_buffer_size < kMinPartialBufferSize) {
                LOG(LOG_DEBUG_1) << "[AutoApply] Partial buffer " << result_.partial_buffer_size
                                 << " below minimum, clamping to " << kMinPartialBufferSize;
                result_.partial_buffer_size = kMinPartialBufferSize;
            }
            config_.partial_buffer_size = result_.partial_buffer_size;
            result_.sources.push_back({"partial_buffer_size", "history",
                "stored rec=" + std::to_string(op.recommended_partial_buffer)});
        }

        // Accum buffer
        if (!config_.isPinned("accum_buffer_size") && op.recommended_accum_buffer > 0) {
            result_.accum_buffer_size = op.recommended_accum_buffer;
            config_.accum_buffer_size = result_.accum_buffer_size;
            result_.sources.push_back({"accum_buffer_size", "history",
                "stored rec=" + std::to_string(op.recommended_accum_buffer)});
        }
    }

    // Tier 2: Benign history buffer recs (fallback if per-GPU had no buffer data)
    const auto* benign_entry = benign_.findByDigits(N_digits_);
    if (benign_entry) {
        if (!config_.isPinned("lp1_max_witness_capacity")
            && result_.lp1_max_witness_capacity == 0
            && benign_entry->recommended_witness_capacity > 0) {
            result_.lp1_max_witness_capacity = benign_entry->recommended_witness_capacity;
            config_.lp1_max_witness_capacity = result_.lp1_max_witness_capacity;
            result_.sources.push_back({"lp1_max_witness_capacity", "benign",
                "cross-GPU: cap=" + std::to_string(benign_entry->recommended_witness_capacity)});
        }
        if (!config_.isPinned("partial_buffer_size")
            && result_.partial_buffer_size == 0
            && benign_entry->recommended_partial_buffer > 0) {
            result_.partial_buffer_size = benign_entry->recommended_partial_buffer;
            if (result_.partial_buffer_size < kMinPartialBufferSize) {
                LOG(LOG_DEBUG_1) << "[AutoApply] Partial buffer " << result_.partial_buffer_size
                                 << " below minimum, clamping to " << kMinPartialBufferSize;
                result_.partial_buffer_size = kMinPartialBufferSize;
            }
            config_.partial_buffer_size = result_.partial_buffer_size;
            result_.sources.push_back({"partial_buffer_size", "benign",
                "cross-GPU: rec=" + std::to_string(benign_entry->recommended_partial_buffer)});
        }
    }

    // Ensure partial buffer is at least 8x accum (all modes, not just batch)
    if (!config_.isPinned("partial_buffer_size")
        && result_.partial_buffer_size > 0) {
        uint64_t accum = result_.accum_buffer_size > 0
            ? result_.accum_buffer_size : uint64_t{524288};
        uint64_t batch_floor = 8ULL * accum;
        if (result_.partial_buffer_size < batch_floor) {
            result_.partial_buffer_size = batch_floor;
            config_.partial_buffer_size = result_.partial_buffer_size;
        }
    }

    // Record effective values in result (for memory estimate and printing)
    if (result_.accum_buffer_size == 0)
        result_.accum_buffer_size = config_.accum_buffer_size;
    if (result_.partial_buffer_size == 0)
        result_.partial_buffer_size = config_.partial_buffer_size;
    if (result_.lp1_max_witness_capacity == 0)
        result_.lp1_max_witness_capacity = config_.lp1_max_witness_capacity;
    result_.lp1_max_combined_output = config_.lp1_max_combined_output;
    result_.lp1_hash_bits = config_.lp1_hash_bits;
    result_.persistent_buffer_size = config_.persistent_buffer_size;
}

void AutoApplyController::computeMemoryEstimate() {
    // Memory cost constants are shared with AutotuneController::printBufferRecommendations
    // — see memory_estimator.h for definitions and audit references.
    using memory_costs::DENSE_CANDIDATE_BYTES;
    using memory_costs::SOA_PER_REL_BYTES;
    using memory_costs::WITNESS_PAYLOAD_BYTES;
    using memory_costs::WITNESS_DIR_BYTES;
    using memory_costs::LP_PIPELINE_BYTES;

    uint64_t A = result_.accum_buffer_size > 0 ? result_.accum_buffer_size : 524288ULL;
    uint64_t P = result_.partial_buffer_size > 0 ? result_.partial_buffer_size : 8 * A;
    uint64_t R = result_.persistent_buffer_size > 0
               ? result_.persistent_buffer_size
               : config_.target_relations + A;
    uint64_t W = result_.lp1_max_witness_capacity > 0
               ? result_.lp1_max_witness_capacity : (1ULL << 20);
    uint64_t O = result_.lp1_max_combined_output > 0
               ? result_.lp1_max_combined_output : 32768ULL;
    uint32_t B = result_.lp1_hash_bits;
    if (B == 0 && W > 0) {
        int log2_w = 63 - __builtin_clzll(W > 0 ? W : 1);
        B = (log2_w > 4) ? (log2_w - 4) : 4;
    }

    bool lp_active = config_.lp1_bound > 0;

    size_t mem = 0;
    mem += 2 * A * DENSE_CANDIDATE_BYTES;   // Accum double-buffer (DenseCandidate)
    mem += A * SOA_PER_REL_BYTES;            // Full batch SoA (transient)
    mem += R * SOA_PER_REL_BYTES;            // Persistent SoA
    if (lp_active) {
        mem += P * SOA_PER_REL_BYTES;        // Partial SoA
        mem += (1ULL << B) * WITNESS_DIR_BYTES; // Witness hash directory
        mem += W * WITNESS_PAYLOAD_BYTES;    // Witness payloads
        mem += W * SOA_PER_REL_BYTES;        // Witness SoA
        mem += O * SOA_PER_REL_BYTES;        // Combined output SoA
        mem += P * LP_PIPELINE_BYTES;        // LP pipeline overhead
    }
    // Sieve internals approximation
    mem += f_data_.size * 12 + 2 * config_.sieve_bound + 500000;

    result_.estimated_gpu_memory_mb = mem / (1024 * 1024);

    // Warn if > 80% of device free memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (result_.estimated_gpu_memory_mb * 1024ULL * 1024ULL > static_cast<size_t>(0.8 * free_mem)) {
        LOG(LOG_WARNING) << "[AutoApply] Estimated GPU memory ("
                         << result_.estimated_gpu_memory_mb
                         << " MB) exceeds 80% of free device memory ("
                         << free_mem / (1024 * 1024) << " MB). "
                         << "Consider reducing --lp1_max_witnesses or --partial_buf_size.";
    }
}

void AutoApplyController::printRecommendations() {
    if (!result_.applied) return;

    std::string src = result_.sources.empty() ? "default" : result_.sources[0].source;
    LOG(LOG_DEBUG_1) << "[AutoApply] Applied params from " << src
                     << " (conf=" << std::to_string(result_.confidence).substr(0, 4)
                     << "): F=" << config_.fb_bound
                     << " M=" << config_.sieve_bound
                     << " L=" << config_.lp1_bound
                     << " mem=~" << result_.estimated_gpu_memory_mb << "MB";
}

AutoApplyResult AutoApplyController::apply() {
    loadHistories();

    if (history_.size() == 0 && benign_.size() == 0) {
        LOG(LOG_DEBUG_1) << "[AutoApply] No history data — skipping auto-apply";
        return result_;
    }

    mergeSieveParams();
    mergeBufferParams();
    computeMemoryEstimate();
    printRecommendations();

    return result_;
}

} // namespace mpqs::autotune
