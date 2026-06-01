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
