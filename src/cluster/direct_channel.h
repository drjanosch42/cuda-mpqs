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
    void onBatchComplete(
        const structures::HostRelationBatch& full_relations,
        const structures::HostRelationBatch& partials,
        uint64_t batch_index) override;

    /// Returns true when Thread A has signaled stop.
    bool shouldStop() const override;

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
};

// ---------------------------------------------------------------------------
// Inline implementation (header-only for M1 simplicity)
// ---------------------------------------------------------------------------

inline DirectChannel::DirectChannel(size_t capacity)
    : buffer_(capacity), capacity_(capacity) {}

inline void DirectChannel::onBatchComplete(
    const structures::HostRelationBatch& full_relations,
    const structures::HostRelationBatch& partials,
    uint64_t batch_index)
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

inline bool DirectChannel::shouldStop() const {
    return stop_.load(std::memory_order_acquire);
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
