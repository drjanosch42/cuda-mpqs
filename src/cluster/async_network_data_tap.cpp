// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file async_network_data_tap.cpp
/// @brief AsyncNetworkDataTap: SPSC ring + I/O thread for zero-sync batch extraction.

#include "async_network_data_tap.h"
#include "serialization.h"
#include "hpc_logger.h"

#include <algorithm>
#include <vector>

namespace mpqs::cluster {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

AsyncNetworkDataTap::AsyncNetworkDataTap(CommBackend& backend, Config config)
    : backend_(backend), config_(config)
{
    // Immediate heartbeat so coordinator's timeout timer starts fresh
    // before CUDA graph capture + first replay completes.
    sendHeartbeat();

    // Launch I/O thread (handles sending, heartbeat, STOP polling)
    io_thread_ = std::thread(&AsyncNetworkDataTap::ioThreadMain, this);
}

void AsyncNetworkDataTap::shutdown() {
    // exchange returns the old value; only the first caller proceeds.
    if (!shutdown_.exchange(true, std::memory_order_acq_rel)) {
        if (io_thread_.joinable()) io_thread_.join();
        const uint64_t dropped = dropped_batches_.load(std::memory_order_relaxed);
        if (dropped > 0) {
            LOG(LOG_WARNING) << "AsyncNetworkDataTap: " << dropped
                             << " batches dropped (ring full)";
        }
    }
}

AsyncNetworkDataTap::~AsyncNetworkDataTap() {
    shutdown();  // no-op if already called by orchestrator
}

// ---------------------------------------------------------------------------
// Pre-allocation
// ---------------------------------------------------------------------------

void AsyncNetworkDataTap::preallocateSlots(size_t max_rels, size_t max_factors) {
    for (size_t i = 0; i < SPSCRing<TapSlot, 32>::capacity(); ++i) {
        ring_.slot(i).full.resize(max_rels, max_factors);
        ring_.slot(i).partials.resize(max_rels, max_factors);
    }
}

// ---------------------------------------------------------------------------
// DataTap interface — sieve thread hot path
// ---------------------------------------------------------------------------

void AsyncNetworkDataTap::onBatchComplete(
    const structures::HostRelationBatch& full_relations,
    const structures::HostRelationBatch& partials,
    uint64_t batch_index)
{
    if (stop_.load(std::memory_order_relaxed)) return;

    // Option A: copy into a local TapSlot, then move into the ring.
    // Vector copy triggers allocation + memcpy (~50us for RSA-100 batch sizes).
    // Still 24x faster than the 1.2ms synchronous serialize+send.
    TapSlot slot;
    slot.full = full_relations;
    slot.partials = partials;
    slot.batch_index = batch_index;
    slot.a_values_this_batch = batch_a_values_;

    if (!ring_.tryPush(std::move(slot))) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
    }

    batches_sent_.fetch_add(1, std::memory_order_relaxed);
    range_a_consumed_.fetch_add(batch_a_values_, std::memory_order_relaxed);
}

bool AsyncNetworkDataTap::shouldStop() const {
    if (stop_.load(std::memory_order_acquire)) return true;
    if (range_a_limit_ > 0 &&
        range_a_consumed_.load(std::memory_order_relaxed) >= range_a_limit_)
        return true;
    return false;
}

// ---------------------------------------------------------------------------
// Chunk/range management
// ---------------------------------------------------------------------------

void AsyncNetworkDataTap::setRange(uint64_t count, uint32_t chunk_id,
                                    uint32_t batch_a_vals) {
    range_a_limit_ = count;
    range_a_consumed_.store(0, std::memory_order_relaxed);
    current_chunk_id_ = chunk_id;
    batch_a_values_ = batch_a_vals;
}

uint32_t AsyncNetworkDataTap::currentChunkId() const { return current_chunk_id_; }
uint64_t AsyncNetworkDataTap::aValsConsumed() const {
    return range_a_consumed_.load(std::memory_order_relaxed);
}
bool AsyncNetworkDataTap::receivedStop() const {
    return stop_.load(std::memory_order_acquire);
}
uint64_t AsyncNetworkDataTap::batchesSent() const {
    return batches_sent_.load(std::memory_order_relaxed);
}
uint64_t AsyncNetworkDataTap::droppedBatches() const {
    return dropped_batches_.load(std::memory_order_relaxed);
}
bool AsyncNetworkDataTap::receivedRecall() const {
    return recall_.load(std::memory_order_acquire);
}
void AsyncNetworkDataTap::clearRecall() {
    recall_.store(false, std::memory_order_release);
    recalled_chunk_id_ = 0;
}

// ---------------------------------------------------------------------------
// I/O thread
// ---------------------------------------------------------------------------

void AsyncNetworkDataTap::ioThreadMain() {
    using clock = std::chrono::steady_clock;
    auto last_send_time = clock::now();
    auto last_heartbeat_time = clock::now();
    const auto coalesce_delay = std::chrono::milliseconds(config_.max_coalesce_delay_ms);
    const auto heartbeat_interval = std::chrono::milliseconds(kHeartbeatIntervalMs);

    std::vector<TapSlot> items;
    items.reserve(config_.coalesce_count);

    while (true) {
        // Shutdown: drain remaining items, then exit
        if (shutdown_.load(std::memory_order_acquire) && ring_.approxSize() == 0)
            break;

        // 1. Drain ring (up to coalesce_count items)
        items.clear();
        ring_.popBatch(items, config_.coalesce_count);

        // If fewer than coalesce_count and delay not exceeded, try to wait
        // (but only if we're not shutting down and ring had something)
        if (items.empty() && !shutdown_.load(std::memory_order_relaxed)) {
            // Check coalesce delay: if items are trickling in, force-send
            if (ring_.approxSize() > 0 &&
                (clock::now() - last_send_time) >= coalesce_delay) {
                ring_.popBatch(items, config_.coalesce_count);
            }
        }

        // 2. Coalesce and send
        if (!items.empty()) {
            std::vector<const structures::HostRelationBatch*> full_ptrs;
            std::vector<uint64_t> full_counts;
            std::vector<const structures::HostRelationBatch*> partial_ptrs;
            std::vector<uint64_t> partial_counts;

            full_ptrs.reserve(items.size());
            full_counts.reserve(items.size());
            partial_ptrs.reserve(items.size());
            partial_counts.reserve(items.size());

            for (const auto& slot : items) {
                full_ptrs.push_back(&slot.full);
                full_counts.push_back(slot.full.num_relations);
                partial_ptrs.push_back(&slot.partials);
                partial_counts.push_back(slot.partials.num_relations);
            }

            auto merged_full = mergeRelationBatches(full_ptrs, full_counts);
            auto merged_partial = mergeRelationBatches(partial_ptrs, partial_counts);

            auto [buf, len] = serializeIncrementalBatch(
                merged_full, merged_full.num_relations,
                merged_partial, merged_partial.num_relations);

            if (len > 0) {
                backend_.send(0, MsgType::INCREMENTAL_BATCH,
                              buf.data(), static_cast<uint32_t>(len));
            }

            last_send_time = clock::now();
        }

        // 3. Poll for STOP/ERROR from coordinator
        RecvMessage msg;
        if (backend_.recv(msg) && msg.valid) {
            if (msg.type == MsgType::STOP || msg.type == MsgType::ERROR) {
                stop_.store(true, std::memory_order_release);
            } else if (msg.type == MsgType::CHUNK_RECALL) {
                ChunkRecallPayload cr{};
                if (msg.payload.size() >= sizeof(cr))
                    memcpy(&cr, msg.payload.data(), sizeof(cr));
                if (cr.chunk_id == current_chunk_id_) {
                    recalled_chunk_id_ = cr.chunk_id;
                    recall_.store(true, std::memory_order_release);
                }
            }
        }

        // 4. Heartbeat timer
        if ((clock::now() - last_heartbeat_time) >= heartbeat_interval) {
            sendHeartbeat();
            last_heartbeat_time = clock::now();
        }

        // 5. Brief sleep if nothing to do (avoid busy-spin)
        if (items.empty() && !shutdown_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    // Final heartbeat before exit
    sendHeartbeat();
}

void AsyncNetworkDataTap::sendHeartbeat() {
    HeartbeatPayload hb{};
    hb.worker_id = backend_.selfId();
    hb.batches_completed = batches_sent_.load(std::memory_order_relaxed);
    hb.relations_found = 0;  // Not tracked per-worker in M3/M4
    backend_.send(0, MsgType::HEARTBEAT, &hb, sizeof(hb));
}

} // namespace mpqs::cluster
