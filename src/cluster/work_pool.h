// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file work_pool.h
/// @brief Thread-safe polynomial work-unit pool with tracked-checkout for fault tolerance.
///
/// S5 rewrite: replaces linear allocator with a checked-out/completed/reclaimed model.
/// Each checkout is assigned a unique chunk_id and tracked in in_flight_.
/// Reclaimed work (from dead workers) is served before advancing the linear cursor.

#include <cassert>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>
#include <chrono>
#include <algorithm>

namespace mpqs::cluster {

/// A contiguous range of a-value indices.
struct WorkUnit {
    uint64_t start;  ///< First a-value index
    uint64_t count;  ///< Number of a-values in this unit
};

/// Thread-safe pool of polynomial work units with tracked checkout.
class WorkPool {
public:
    /// Record of a checked-out chunk (in-flight until completed or reclaimed).
    struct CheckedOutWork {
        uint32_t chunk_id;
        WorkUnit unit;
        uint8_t  worker_id;
        std::chrono::steady_clock::time_point checkout_time;
    };

    /// @param a_start   First a-value index in the pool
    /// @param total_a   Total number of a-values
    /// @param unit_size Default work unit size for legacy requestWork() (default: 64)
    WorkPool(uint64_t a_start, uint64_t total_a, uint32_t unit_size = 64);

    // --- S2 legacy interface (backward compat, wraps checkoutWork) ---

    /// Request a default-sized work unit. Returns nullopt if exhausted.
    std::optional<WorkUnit> requestWork();

    /// Request a specific number of a-values. May return fewer if near exhaustion.
    std::optional<WorkUnit> requestWork(uint64_t requested_count);

    /// Remaining a-values in the pool (linear + returned, excludes in-flight).
    uint64_t remaining() const;

    /// True if no unassigned a-values remain (linear exhausted AND returned_ empty).
    bool exhausted() const;

    // --- S5: tracked checkout ---

    /// Checkout a chunk. Assigns unique chunk_id. Records in in_flight_.
    /// Serves returned_ (reclaimed) work before advancing the linear cursor.
    std::optional<CheckedOutWork> checkoutWork(uint64_t count, uint8_t worker_id);

    /// Mark chunk as completed. Removes from in_flight_.
    void completeChunk(uint32_t chunk_id);

    /// Return a single checked-out chunk to the assignable (returned_) pool
    /// without consuming any of it — used when a CHUNK_ASSIGN send fails so the
    /// chunk can be re-dispatched immediately. Removes the chunk from in_flight_
    /// and pushes its full WorkUnit onto returned_. No-op if not in flight.
    /// Returns the number of a-values returned (0 if not found). Thread-safe.
    uint64_t returnChunk(uint32_t chunk_id);

    /// Reclaim all in-flight chunks for a dead worker. Returns count of a-values
    /// returned to returned_ queue (available for redistribution via checkoutWork).
    uint64_t reclaimWork(uint8_t worker_id);

    /// Reclaim the unconsumed portion of a recalled chunk.
    /// Splits chunk at consumed_count: removes from in_flight_, returns
    /// [start+consumed, start+count) to returned_ queue.
    /// If consumed_count >= chunk.count: equivalent to completeChunk (nothing returned).
    /// Thread-safe.
    void reclaimPartial(uint32_t chunk_id, uint64_t consumed_count);

    /// Read-only snapshot of in-flight chunks (for straggler detection).
    /// Returns a copy to avoid holding the lock. Thread-safe.
    std::vector<CheckedOutWork> inFlightChunks() const;

    /// Total a-values currently in-flight (checked out, not yet completed).
    uint64_t inFlight() const;

    /// Total a-values in-flight for a specific worker.
    uint64_t inFlightFor(uint8_t worker_id) const;

    /// Total a-values remaining OR in-flight (useful for "is all work done?" check).
    uint64_t remainingOrInFlight() const;

    // --- S6: cursor accessors for coordinator checkpoint ---

    /// Current linear cursor position (next unassigned a-value index).
    uint64_t nextCursor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_;
    }

    /// End cursor (one past last a-value index).
    uint64_t endCursor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return end_;
    }

    /// Completed contiguous-prefix cursor (S3 coordinator checkpoint, B2).
    ///
    /// Returns the largest cursor C such that EVERY a-index in [a_start, C) has been
    /// fully sieved (checked out AND completed). Computed as
    ///   min( next_, min start over in_flight_ ∪ returned_ ).
    /// Rationale: chunks are handed out in increasing order from next_; anything below
    /// every in-flight/returned start AND below next_ has been completed and erased from
    /// in_flight_. An in-flight or returned chunk is, by definition, NOT yet completed —
    /// so its start bounds the completed prefix from above.
    ///
    /// This is deliberately NOT nextCursor(): nextCursor() == next_ silently drops the
    /// in_flight_ ∪ returned_ chunks (which live only in coordinator RAM and are lost on
    /// a kill). Resuming at next_ would SKIP those a-values entirely, and dedup cannot
    /// recover a-values that are never sieved → under-collection + premature overflow
    /// drain, re-introducing the a-value pool exhaustion fixed in 4d20d7b. Resume must
    /// restore the completed prefix. Thread-safe.
    uint64_t completedPrefixCursor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t c = next_;
        for (const auto& cw : in_flight_) c = std::min(c, cw.unit.start);
        for (const auto& wu : returned_)  c = std::min(c, wu.start);
        return c;
    }

    /// Restore pool cursor from coordinator checkpoint. Only valid at startup
    /// before any requestWork/checkoutWork calls. Asserts clean state.
    void setCursor(uint64_t cursor) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(in_flight_.empty() && returned_.empty());
        assert(cursor <= end_);
        next_ = cursor;
    }

private:
    mutable std::mutex mutex_;
    uint64_t next_;          ///< Next unassigned a-value index (linear cursor)
    uint64_t end_;           ///< One past last a-value index
    uint32_t unit_size_;     ///< Default work unit size

    uint32_t next_chunk_id_ = 0;
    std::vector<CheckedOutWork> in_flight_;
    std::vector<WorkUnit>       returned_;  ///< Reclaimed work, served before next_
};

} // namespace mpqs::cluster
