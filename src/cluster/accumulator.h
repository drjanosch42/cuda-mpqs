// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file accumulator.h
/// @brief Thread-safe AccumulatorQueue, single-thread RelationAccumulator with incremental
///        dedup, and blocking FinalBatchHandoff for the refactored coordinator architecture.
///
/// Data flow:
///   Thread B (GPU sieve)  --pushRelations/pushPartials-->  AccumulatorQueue
///   Network thread         --pushRemoteRelations/Partials->  AccumulatorQueue
///                                                                |
///                                                   drain() <---+
///                                                     |
///                                                     v
///                                           RelationAccumulator  <-- CPU LP adds matches
///                                                     |
///                                           extractFinal()
///                                                     v
///                                           FinalBatchHandoff
///                                                     |
///                                      Thread B awaits <- deliver()

#include "mpqs_soa.h"

#include <mutex>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <optional>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>
#include <cmath>

namespace mpqs::cluster {

// ============================================================================
// AccumulatorQueue — thread-safe multi-producer, single-consumer queue
// ============================================================================

/// Thread-safe multi-producer queue for funneling relations and partials
/// from local sieve (Thread B) and network (Thread A) into the accumulator.
class AccumulatorQueue {
public:
    /// Producer: enqueue local sieve full relations
    void pushRelations(mpqs::structures::HostRelationBatch&& batch) {
        std::lock_guard<std::mutex> lock(mu_);
        local_relations_.push_back(std::move(batch));
    }

    /// Producer: enqueue local sieve 1-partials
    void pushPartials(mpqs::structures::HostRelationBatch&& batch) {
        std::lock_guard<std::mutex> lock(mu_);
        local_partials_.push_back(std::move(batch));
    }

    /// Producer: enqueue remote worker full relations
    void pushRemoteRelations(mpqs::structures::HostRelationBatch&& batch, uint8_t worker_id) {
        std::lock_guard<std::mutex> lock(mu_);
        remote_relations_.emplace_back(worker_id, std::move(batch));
    }

    /// Producer: enqueue remote worker 1-partials
    void pushRemotePartials(mpqs::structures::HostRelationBatch&& batch, uint8_t worker_id) {
        std::lock_guard<std::mutex> lock(mu_);
        remote_partials_.emplace_back(worker_id, std::move(batch));
    }

    /// Result of draining the queue.
    struct DrainResult {
        std::vector<mpqs::structures::HostRelationBatch> local_relations;
        std::vector<mpqs::structures::HostRelationBatch> local_partials;
        std::vector<std::pair<uint8_t, mpqs::structures::HostRelationBatch>> remote_relations;
        std::vector<std::pair<uint8_t, mpqs::structures::HostRelationBatch>> remote_partials;

        bool empty() const {
            return local_relations.empty() && local_partials.empty() &&
                   remote_relations.empty() && remote_partials.empty();
        }
    };

    /// Consumer: drain all queued items (non-blocking, returns empty if nothing queued).
    DrainResult drain() {
        std::lock_guard<std::mutex> lock(mu_);
        DrainResult result;
        result.local_relations.swap(local_relations_);
        result.local_partials.swap(local_partials_);
        result.remote_relations.swap(remote_relations_);
        result.remote_partials.swap(remote_partials_);
        return result;
    }

    /// Signal that Thread B (GPU sieve) is done producing.
    void signalSieveDone() { sieve_done_.store(true, std::memory_order_release); }
    bool isSieveDone() const { return sieve_done_.load(std::memory_order_acquire); }

private:
    std::mutex mu_;
    std::vector<mpqs::structures::HostRelationBatch> local_relations_;
    std::vector<mpqs::structures::HostRelationBatch> local_partials_;
    std::vector<std::pair<uint8_t, mpqs::structures::HostRelationBatch>> remote_relations_;
    std::vector<std::pair<uint8_t, mpqs::structures::HostRelationBatch>> remote_partials_;
    std::atomic<bool> sieve_done_{false};
};

// ============================================================================
// RelationAccumulator — single-thread dedup + counting (owned by Thread A)
// ============================================================================

/// Centralized relation accumulator with incremental dedup.
/// NOT thread-safe — designed to be owned by a single thread (Thread A).
class RelationAccumulator {
public:
    explicit RelationAccumulator(uint64_t target_relations, double dedup_margin = 1.05)
        : target_(target_relations)
        , effective_target_(static_cast<uint64_t>(std::ceil(target_relations * dedup_margin)))
    {
        // Initialize CSR sentinel for the accumulated batch.
        accumulated_.factor_offsets.push_back(0);
    }

    /// Ingest relations from a source. Deduplicates against all prior relations.
    /// source_id: 0 = local coordinator, 1-254 = workers, 255 = LP combines
    void addRelations(const mpqs::structures::HostRelationBatch& batch, uint8_t source_id) {
        for (size_t i = 0; i < batch.num_relations; i++) {
            uint64_t hash = computeRelationHash(batch, i);
            if (seen_.insert(hash).second) {
                appendSingleRelation(accumulated_, batch, i);
                per_source_[source_id]++;
            }
        }
    }

    /// Ingest LP-combined full relations (same as addRelations with source_id=255).
    void addLPRelations(const mpqs::structures::HostRelationBatch& batch) {
        addRelations(batch, 255);
    }

    /// Total unique relations accumulated so far.
    uint64_t totalRelations() const { return accumulated_.num_relations; }

    /// Whether effective target has been reached.
    bool targetReached() const { return accumulated_.num_relations >= effective_target_; }

    /// Per-source breakdown for logging.
    uint64_t relationsFrom(uint8_t source_id) const {
        auto it = per_source_.find(source_id);
        return (it != per_source_.end()) ? it->second : 0;
    }

    /// Extract final accumulated batch. Moves data out — accumulator is empty after this.
    mpqs::structures::HostRelationBatch extractFinal() {
        return std::move(accumulated_);
    }

    uint64_t effectiveTarget() const { return effective_target_; }
    uint64_t rawTarget() const { return target_; }

private:
    /// Compute 64-bit hash matching deduplicateHostBatch in mpqs_soa.cu:
    ///   hash = (len << 48) | (exp_xor << 32) | body_xor
    /// where body_xor includes factor_indices * MAGIC, sign, and val_2_exp.
    static uint64_t computeRelationHash(const mpqs::structures::HostRelationBatch& batch, size_t i) {
        constexpr uint32_t MAGIC = 0x9e3779b9;

        uint64_t start = batch.factor_offsets[i];
        uint64_t end   = batch.factor_offsets[i + 1];
        uint16_t len   = static_cast<uint16_t>(end - start);

        uint16_t exp_xor  = 0;
        uint32_t body_xor = 0;
        for (uint64_t k = start; k < end; k++) {
            exp_xor  ^= static_cast<uint16_t>(batch.factor_counts[k]);
            body_xor ^= (batch.factor_indices[k] * MAGIC);
        }

        // Encoding (canonical, see audit Appendix A): batch.signs[i] is uint8_t
        // with {1 = positive Q, 0xFF = negative Q}. Use the M11c encoding-agnostic
        // "negative iff != 1" extraction so this CPU dedup hash matches the GPU
        // path in postprocessing.cu:594. M12-S5 fixed the GPU side; M12-S5b
        // sweeps this parallel CPU site (cluster coordinator dedup) to prevent
        // GPU-worker / CPU-coordinator hash divergence.
        int32_t sign_val  = (batch.signs[i] != 1u) ? -1 : 1;
        int32_t shift_val = sign_val * (1 << (batch.val_2_exps[i] & 0x1F));
        body_xor ^= static_cast<uint32_t>(shift_val);

        return (static_cast<uint64_t>(len) << 48)
             | (static_cast<uint64_t>(exp_xor) << 32)
             | static_cast<uint64_t>(body_xor);
    }

    /// Append a single relation from src at index idx to dest, maintaining CSR validity.
    static void appendSingleRelation(mpqs::structures::HostRelationBatch& dest,
                                     const mpqs::structures::HostRelationBatch& src,
                                     size_t idx) {
        dest.sqrt_Q.push_back(src.sqrt_Q[idx]);
        dest.signs.push_back(src.signs[idx]);
        dest.val_2_exps.push_back(src.val_2_exps[idx]);
        dest.large_primes.push_back(src.large_primes[idx]);

        // CSR factor append
        uint64_t fstart = src.factor_offsets[idx];
        uint64_t fend   = src.factor_offsets[idx + 1];

        for (uint64_t f = fstart; f < fend; f++) {
            dest.factor_indices.push_back(src.factor_indices[f]);
            dest.factor_counts.push_back(src.factor_counts[f]);
        }
        dest.num_factors += (fend - fstart);
        dest.factor_offsets.push_back(dest.num_factors);

        dest.num_relations++;
    }

    uint64_t target_;
    uint64_t effective_target_;
    mpqs::structures::HostRelationBatch accumulated_;
    std::unordered_set<uint64_t> seen_;                 ///< 64-bit relation hashes for dedup
    std::unordered_map<uint8_t, uint64_t> per_source_;  ///< Per-source relation count
};

// ============================================================================
// FinalBatchHandoff — blocking producer-consumer handoff via condition variable
// ============================================================================

/// Blocking handoff of the final accumulated batch from Thread A to Thread B.
class FinalBatchHandoff {
public:
    /// Thread A calls this to deliver the final batch.
    void deliver(mpqs::structures::HostRelationBatch&& batch) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            batch_ = std::move(batch);
        }
        cv_.notify_one();
    }

    /// Thread B blocks here until the final batch is delivered.
    mpqs::structures::HostRelationBatch await() {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return batch_.has_value(); });
        return std::move(*batch_);
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::optional<mpqs::structures::HostRelationBatch> batch_;
};

} // namespace mpqs::cluster
