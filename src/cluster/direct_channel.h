// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include "data_tap.h"
#include "mpqs_soa.h"

#include <atomic>
#include <mutex>
#include <vector>

namespace mpqs::cluster {

/// SPSC channel for coordinator Thread B -> Thread A communication.
/// Implements DataTap (producer side, called by sieve loop on Thread B).
/// Thread A consumes via tryPop() and signals stop via signalStop().
///
/// Bounded circular buffer with mutex guard. Low contention: Thread B produces
/// at ~30ms intervals, Thread A polls at ~10ms. True lock-free ring deferred to M2.
///
/// Spec reference: cluster_v2_spec.md Section 4.3
class DirectChannel : public DataTap {
public:
    /// Payload delivered per extraction cycle.
    struct Payload {
        structures::HostRelationBatch full;
        structures::HostRelationBatch partials;
        uint64_t batch_idx;
    };

    /// @param capacity  Max buffered payloads before backpressure (drop-oldest).
    explicit DirectChannel(size_t capacity = 64);

    // --- DataTap interface (Thread B / producer) ---

    /// Enqueue an incremental batch. Copies full_relations and partials into a Payload.
    /// If the queue is full, drops the oldest entry (backpressure).
    /// Cost: ~50us for ~500 KB copy at RSA-100 scale.
    /// @param a_values_advanced True a-values consumed since the previous call
    ///        (0 → use batch_a_values_; graph replay passes batch_size × graph_N).
    void onBatchComplete(
        const structures::HostRelationBatch& full_relations,
        const structures::HostRelationBatch& partials,
        uint64_t batch_index,
        uint64_t a_values_advanced = 0) override;

    /// Returns true when Thread A has signaled stop OR the assigned a-range is
    /// exhausted (cluster coordinator only — solo leaves range_a_limit_ == 0).
    bool shouldStop() const override;

    // --- Range management (coordinator local sieve, mirrors AsyncNetworkDataTap) ---

    /// Bound the coordinator's own (node-0) local sieve to its assigned a-range.
    /// Solo mode never calls this, so range_a_limit_ stays 0 and shouldStop()
    /// keeps its original stop_-only semantics. Without this bound the coordinator
    /// (poly_range_start == 0, so the resetAndAdvanceTo guard is skipped) sieves
    /// unbounded from a_index 0 and overruns into worker a-ranges, generating
    /// byte-identical duplicate partials.
    /// @param count        Assigned a-range width (node-0's ranges[0].count).
    /// @param batch_a_vals Default a-values per onBatchComplete call (= sieve_batch_size).
    void setRange(uint64_t count, uint32_t batch_a_vals);

    /// a-values consumed so far against the assigned range (telemetry/tests).
    uint64_t aValsConsumed() const;

    /// True iff Thread A has EXPLICITLY signaled stop (target reached or
    /// teardown). Distinct from shouldStop(), which ALSO returns true on
    /// per-chunk a-range exhaustion. The coordinator's overflow self-assign loop
    /// needs to tell "this chunk's a-range is done — fetch the next chunk" (a
    /// range bound, NOT terminal) apart from "the whole sieve is done — stop
    /// looping" (this flag). Solo / worker code never calls it.
    bool stopped() const { return stop_.load(std::memory_order_acquire); }

    // --- Consumer API (Thread A) ---

    /// Non-blocking pop. Returns true and fills `out` if a payload was available.
    bool tryPop(Payload& out);

    /// Signal Thread B to stop sieving. Called by Thread A when target reached.
    void signalStop();

    /// Returns true if the queue has no pending data.
    bool empty() const;

private:
    mutable std::mutex mu_;
    std::vector<Payload> buffer_;
    size_t head_     = 0;  ///< Consumer reads from head
    size_t tail_     = 0;  ///< Producer writes to tail
    size_t count_    = 0;  ///< Current number of buffered payloads
    size_t capacity_;      ///< Max buffered payloads

    std::atomic<bool> stop_{false};

    // Range tracking (Thread B / sieve thread writes via onBatchComplete; shouldStop reads).
    // range_a_limit_ == 0 disables the range check (solo / unbounded default).
    std::atomic<uint64_t> range_a_consumed_{0};
    uint64_t range_a_limit_   = 0;
    uint64_t batch_a_values_  = 1;
};

// ---------------------------------------------------------------------------
// Inline implementation (header-only for M1 simplicity)
// ---------------------------------------------------------------------------

inline DirectChannel::DirectChannel(size_t capacity)
    : buffer_(capacity), capacity_(capacity) {}

inline void DirectChannel::onBatchComplete(
    const structures::HostRelationBatch& full_relations,
    const structures::HostRelationBatch& partials,
    uint64_t batch_index,
    uint64_t a_values_advanced)
{
    {
        std::lock_guard<std::mutex> lock(mu_);

        // Backpressure: if full, drop oldest (advance head)
        if (count_ == capacity_) {
            head_ = (head_ + 1) % capacity_;
            --count_;
        }

        // Write to tail slot
        Payload& slot = buffer_[tail_];
        slot.full      = full_relations;   // vector copy
        slot.partials  = partials;         // vector copy
        slot.batch_idx = batch_index;

        tail_ = (tail_ + 1) % capacity_;
        ++count_;
    }

    // Track a-values consumed for the coordinator a-range guard. Single-batch
    // loops pass 0 → batch_a_values_; the graph replay loop passes the true
    // count (batch_size × graph_N). No-op for solo (range_a_limit_ == 0).
    const uint64_t a_advanced =
        (a_values_advanced > 0) ? a_values_advanced : batch_a_values_;
    range_a_consumed_.fetch_add(a_advanced, std::memory_order_relaxed);
}

inline bool DirectChannel::shouldStop() const {
    if (stop_.load(std::memory_order_acquire)) return true;
    if (range_a_limit_ > 0 &&
        range_a_consumed_.load(std::memory_order_relaxed) >= range_a_limit_)
        return true;
    return false;
}

inline void DirectChannel::setRange(uint64_t count, uint32_t batch_a_vals) {
    range_a_limit_ = count;
    range_a_consumed_.store(0, std::memory_order_relaxed);
    batch_a_values_ = (batch_a_vals > 0) ? batch_a_vals : 1;
}

inline uint64_t DirectChannel::aValsConsumed() const {
    return range_a_consumed_.load(std::memory_order_relaxed);
}

inline bool DirectChannel::tryPop(Payload& out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (count_ == 0) return false;

    out = std::move(buffer_[head_]);
    head_ = (head_ + 1) % capacity_;
    --count_;
    return true;
}

inline void DirectChannel::signalStop() {
    stop_.store(true, std::memory_order_release);
}

inline bool DirectChannel::empty() const {
    std::lock_guard<std::mutex> lock(mu_);
    return count_ == 0;
}

} // namespace mpqs::cluster
