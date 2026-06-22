// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file async_network_data_tap.h
/// @brief Async DataTap: SPSC ring + I/O thread replaces synchronous NetworkDataTap.
///
/// onBatchComplete() completes in <50us (copy into ring slot, no serialization).
/// Background I/O thread drains ring, coalesces batches via mergeRelationBatches(),
/// serializes once, and sends once per coalesce group. Heartbeat consolidated into
/// the I/O thread — single background thread total.
///
/// Spec reference: cluster_v2_spec.md Section 4.4, M4-S3

#include "data_tap.h"
#include "comm_backend.h"
#include "spsc_ring.h"
#include "cluster_common.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

namespace mpqs::cluster {

/// Configuration for AsyncNetworkDataTap coalescing behavior.
/// Hoisted to namespace scope (GCC 11 cannot use nested aggregates with
/// default member initializers as default arguments before class completion).
struct AsyncNetworkDataTapConfig {
    uint32_t coalesce_count = 8;           ///< Batches to coalesce per send (8 RTX, 4 Jetson)
    uint32_t max_coalesce_delay_ms = 500;  ///< Force-send after this delay even if < coalesce_count
};

class AsyncNetworkDataTap : public DataTap {
public:
    using Config = AsyncNetworkDataTapConfig;

    /// @param backend  Initialized CommBackend in worker mode.
    /// @param config   Coalescing and ring parameters.
    explicit AsyncNetworkDataTap(CommBackend& backend,
                                  Config config = Config{});
    ~AsyncNetworkDataTap();

    // Non-copyable, non-movable (owns thread + backend reference)
    AsyncNetworkDataTap(const AsyncNetworkDataTap&) = delete;
    AsyncNetworkDataTap& operator=(const AsyncNetworkDataTap&) = delete;

    /// Pre-allocate HostRelationBatch vectors in all ring slots.
    /// Call after pp_config is known (before SieveStage).
    /// @param max_rels    Max relations per extraction (from pp_config).
    /// @param max_factors Max factors per extraction.
    void preallocateSlots(size_t max_rels, size_t max_factors);

    /// Join the I/O thread and log any dropped batches.
    /// Safe to call multiple times — subsequent calls are no-ops.
    /// Must be called before any post-sieve comm_backend_ sends to avoid
    /// concurrent send races and use-after-free when comm_backend_ is reset.
    void shutdown();

    // --- DataTap interface ---

    /// Copy batch data into ring slot and push (<50us, no serialization).
    /// @param a_values_advanced 0 → use batch_a_values_; else the true per-call
    ///        a-value count (graph replay passes sieve_batch_size × graph_N).
    void onBatchComplete(
        const structures::HostRelationBatch& full_relations,
        const structures::HostRelationBatch& partials,
        uint64_t batch_index,
        uint64_t a_values_advanced = 0) override;

    /// Check stop_ atomic + range exhaustion. (<10ns, no network I/O)
    bool shouldStop() const override;

    // --- Chunk/range management (same API as NetworkDataTap) ---

    void setRange(uint64_t count, uint32_t chunk_id, uint32_t batch_a_vals);
    uint32_t currentChunkId() const;
    uint64_t aValsConsumed() const;
    bool receivedStop() const;
    uint64_t batchesSent() const;
    uint64_t droppedBatches() const;
    bool receivedRecall() const;
    void clearRecall();

    /// Take a CHUNK_ASSIGN frame captured by the I/O thread (mid-sieve overflow
    /// assignment), if one is pending. The worker socket has a SINGLE reader —
    /// the I/O thread — so a CHUNK_ASSIGN delivered while the worker is in its
    /// chunk-wait loop must be routed through here rather than via a second,
    /// racing recv() on the shared socket (which previously swallowed the frame).
    /// @return true and fills @p out if a CHUNK_ASSIGN was pending (and clears it);
    ///         false otherwise. Thread-safe.
    bool tryTakeChunkAssign(ChunkAssignPayload& out);

private:
    CommBackend& backend_;
    Config config_;

    // SPSC ring: sieve thread pushes, I/O thread pops.
    SPSCRing<TapSlot, 32> ring_;

    // I/O thread
    std::thread io_thread_;
    std::atomic<bool> shutdown_{false};  ///< Set by destructor to signal I/O thread exit

    // Shared atomics (sieve thread writes, I/O thread reads for heartbeat)
    std::atomic<uint64_t> batches_sent_{0};
    std::atomic<uint64_t> dropped_batches_{0};

    // Stop signal (I/O thread writes on STOP recv, sieve thread reads)
    mutable std::atomic<bool> stop_{false};
    mutable std::atomic<bool> recall_{false};
    mutable uint32_t recalled_chunk_id_{0};

    // CHUNK_ASSIGN capture: the I/O thread is the sole socket reader. A
    // CHUNK_ASSIGN that arrives mid-sieve (overflow re-dispatch) is stored here
    // and handed to the main thread via tryTakeChunkAssign(). Guarded by a mutex
    // (frames are infrequent; a full payload copy is too large for a lock-free
    // atomic). has_pending_assign_ true ⇒ pending_assign_ holds an unconsumed frame.
    mutable std::mutex assign_mutex_;
    ChunkAssignPayload  pending_assign_{};
    bool                has_pending_assign_ = false;

    // Range tracking (sieve thread writes, shouldStop reads)
    uint64_t range_a_limit_ = 0;
    std::atomic<uint64_t> range_a_consumed_{0};
    uint32_t current_chunk_id_ = 0;
    uint32_t batch_a_values_ = 1;

    /// I/O thread main loop: drain ring, coalesce, serialize, send, poll STOP, heartbeat.
    void ioThreadMain();

    /// Send heartbeat to coordinator (called only from I/O thread).
    void sendHeartbeat();
};

} // namespace mpqs::cluster
