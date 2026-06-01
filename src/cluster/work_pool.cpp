// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file work_pool.cpp
/// @brief WorkPool implementation — tracked-checkout model (S5).

#include "work_pool.h"

namespace mpqs::cluster {

WorkPool::WorkPool(uint64_t a_start, uint64_t total_a, uint32_t unit_size)
    : next_(a_start), end_(a_start + total_a), unit_size_(unit_size) {}

// ============================================================================
// S2 legacy interface (wraps checkoutWork with worker_id=0)
// ============================================================================

std::optional<WorkUnit> WorkPool::requestWork() {
    auto cw = checkoutWork(unit_size_, /*worker_id=*/0);
    if (!cw) return std::nullopt;
    return cw->unit;
}

std::optional<WorkUnit> WorkPool::requestWork(uint64_t requested_count) {
    auto cw = checkoutWork(requested_count, /*worker_id=*/0);
    if (!cw) return std::nullopt;
    return cw->unit;
}

// ============================================================================
// S5: tracked checkout
// ============================================================================

std::optional<WorkPool::CheckedOutWork> WorkPool::checkoutWork(uint64_t count, uint8_t worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Prefer reclaimed (returned_) work over linear cursor
    if (!returned_.empty()) {
        WorkUnit wu = returned_.back();
        returned_.pop_back();

        // If the returned unit is larger than requested, split it
        if (wu.count > count) {
            WorkUnit remainder{wu.start + count, wu.count - count};
            returned_.push_back(remainder);
            wu.count = count;
        }

        CheckedOutWork cw{};
        cw.chunk_id = next_chunk_id_++;
        cw.unit = wu;
        cw.worker_id = worker_id;
        cw.checkout_time = std::chrono::steady_clock::now();
        in_flight_.push_back(cw);
        return cw;
    }

    // Linear cursor
    if (next_ >= end_) return std::nullopt;
    uint64_t actual = std::min(count, end_ - next_);
    WorkUnit wu{next_, actual};
    next_ += actual;

    CheckedOutWork cw{};
    cw.chunk_id = next_chunk_id_++;
    cw.unit = wu;
    cw.worker_id = worker_id;
    cw.checkout_time = std::chrono::steady_clock::now();
    in_flight_.push_back(cw);
    return cw;
}

void WorkPool::completeChunk(uint32_t chunk_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = in_flight_.begin(); it != in_flight_.end(); ++it) {
        if (it->chunk_id == chunk_id) {
            in_flight_.erase(it);
            return;
        }
    }
}

uint64_t WorkPool::reclaimWork(uint8_t worker_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t reclaimed = 0;
    auto it = in_flight_.begin();
    while (it != in_flight_.end()) {
        if (it->worker_id == worker_id) {
            returned_.push_back(it->unit);
            reclaimed += it->unit.count;
            it = in_flight_.erase(it);
        } else {
            ++it;
        }
    }
    return reclaimed;
}

void WorkPool::reclaimPartial(uint32_t chunk_id, uint64_t consumed_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(in_flight_.begin(), in_flight_.end(),
        [chunk_id](const CheckedOutWork& c) { return c.chunk_id == chunk_id; });

    if (it == in_flight_.end()) return;  // Already completed or not found

    const WorkUnit& unit = it->unit;
    uint64_t actual_consumed = std::min(consumed_count, unit.count);
    uint64_t remainder = unit.count - actual_consumed;

    if (remainder > 0) {
        returned_.push_back(WorkUnit{unit.start + actual_consumed, remainder});
    }
    in_flight_.erase(it);
}

std::vector<WorkPool::CheckedOutWork> WorkPool::inFlightChunks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_flight_;
}

uint64_t WorkPool::remaining() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t linear = (next_ < end_) ? (end_ - next_) : 0;
    uint64_t ret = 0;
    for (auto& wu : returned_) ret += wu.count;
    return linear + ret;
}

bool WorkPool::exhausted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_ >= end_ && returned_.empty();
}

uint64_t WorkPool::inFlight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (auto& cw : in_flight_) total += cw.unit.count;
    return total;
}

uint64_t WorkPool::inFlightFor(uint8_t worker_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (auto& cw : in_flight_) {
        if (cw.worker_id == worker_id) total += cw.unit.count;
    }
    return total;
}

uint64_t WorkPool::remainingOrInFlight() const {
    // Both methods acquire mutex independently — acceptable for approximate queries.
    return remaining() + inFlight();
}

} // namespace mpqs::cluster
