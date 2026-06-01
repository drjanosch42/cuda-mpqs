// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file spsc_ring.h
/// @brief Lock-free Single-Producer Single-Consumer ring buffer.
///
/// Lamport queue with power-of-2 capacity. Cache-line-aligned head/tail
/// atomics to avoid false sharing. Used by AsyncNetworkDataTap to decouple
/// the sieve thread (producer) from the I/O thread (consumer).
///
/// No CUDA dependencies — pure C++20.

#include "mpqs_soa.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mpqs::cluster {

/// Payload stored in each ring slot for the async DataTap pipeline.
struct TapSlot {
    structures::HostRelationBatch full;
    structures::HostRelationBatch partials;
    uint64_t batch_index = 0;
    uint64_t a_values_this_batch = 0;
};

/// Lock-free SPSC ring buffer (Lamport queue).
/// @tparam T Element type (must be default-constructible).
/// @tparam N Capacity (must be power of 2).
template <typename T, size_t N>
class SPSCRing {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

public:
    SPSCRing() : slots_(N) {}

    // Non-copyable, non-movable (contains atomics)
    SPSCRing(const SPSCRing&) = delete;
    SPSCRing& operator=(const SPSCRing&) = delete;

    /// Producer: try to push an element. Returns false if ring is full.
    /// Move semantics: element is moved into the slot.
    bool tryPush(T&& item) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t next = (t + 1) & mask_;
        if (next == head_.load(std::memory_order_acquire)) return false;  // full
        slots_[t] = std::move(item);
        tail_.store(next, std::memory_order_release);
        return true;
    }

    /// Consumer: try to pop an element. Returns false if ring is empty.
    bool tryPop(T& out) {
        const size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire)) return false;  // empty
        out = std::move(slots_[h]);
        head_.store((h + 1) & mask_, std::memory_order_release);
        return true;
    }

    /// Consumer: pop up to max_n elements into out. Returns count popped.
    size_t popBatch(std::vector<T>& out, size_t max_n) {
        size_t count = 0;
        const size_t t = tail_.load(std::memory_order_acquire);
        size_t h = head_.load(std::memory_order_relaxed);
        while (count < max_n && h != t) {
            out.push_back(std::move(slots_[h]));
            h = (h + 1) & mask_;
            ++count;
        }
        if (count > 0) head_.store(h, std::memory_order_release);
        return count;
    }

    /// Approximate size (safe to call from either thread). Not exact due to
    /// relaxed loads, but sufficient for diagnostics and coalescing decisions.
    size_t approxSize() const {
        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_relaxed);
        return (t - h) & mask_;
    }

    /// Direct mutable access to slot storage (for pre-allocation).
    /// ONLY call before any push/pop — not thread-safe during operation.
    T& slot(size_t i) { return slots_[i]; }

    static constexpr size_t capacity() { return N; }

private:
    // Padding ensures head_ and tail_ are on separate cache lines
    static constexpr size_t kCacheLine = 64;

    alignas(kCacheLine) std::atomic<size_t> head_{0};  // Consumer writes, producer reads
    alignas(kCacheLine) std::atomic<size_t> tail_{0};  // Producer writes, consumer reads

    std::vector<T> slots_;

    static constexpr size_t mask_ = N - 1;
};

} // namespace mpqs::cluster
