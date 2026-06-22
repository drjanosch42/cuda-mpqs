// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file chunk_scheduler.cpp
/// @brief ChunkScheduler implementation — EMA throughput tracking + adaptive chunk sizing.
///
/// v2: contiguous range computation replaces computeInitialChunks(WorkerState).
///     Quantum rounding and hypercube alignment applied to all chunk sizes.

#include "chunk_scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>

namespace mpqs::cluster {

// ============================================================================
// WorkerThroughput
// ============================================================================

void WorkerThroughput::update(uint64_t rels, uint64_t a_vals, double elapsed_s, double alpha) {
    if (elapsed_s <= 0.0) return;

    double srps = static_cast<double>(rels)   / elapsed_s;
    double saps = static_cast<double>(a_vals) / elapsed_s;

    if (chunks_completed == 0) {
        rels_per_sec   = srps;
        a_vals_per_sec = saps;
    } else {
        rels_per_sec   = alpha * srps + (1.0 - alpha) * rels_per_sec;
        a_vals_per_sec = alpha * saps + (1.0 - alpha) * a_vals_per_sec;
    }
    chunks_completed++;
    total_rels   += rels;
    total_a_vals += a_vals;
}

// ============================================================================
// ChunkScheduler
// ============================================================================

ChunkScheduler::ChunkScheduler(WorkPool& pool, uint32_t num_workers, uint64_t total_a,
                                 uint32_t hypercube_size, uint32_t quantum)
    : pool_(pool), num_workers_(num_workers), total_a_(total_a),
      H_(hypercube_size), Q_(quantum) {}

// ============================================================================
// Contiguous range computation (replaces v1 computeInitialChunks)
// ============================================================================

std::vector<ChunkScheduler::ContiguousRange> ChunkScheduler::computeContiguousRanges(
        const std::vector<uint32_t>& sm_counts,
        uint32_t target_rels,
        uint32_t H,
        uint32_t Q,
        double headroom_pct) const {

    const uint32_t num_nodes = static_cast<uint32_t>(sm_counts.size());
    if (num_nodes == 0 || H == 0) return {};

    // W_est = max(4 * num_nodes, ceil(target_rels / (5 * H)) * 2)
    uint64_t w_est = std::max(
        static_cast<uint64_t>(4) * num_nodes,
        ((static_cast<uint64_t>(target_rels) + 5 * H - 1) / (5 * H)) * 2);

    // Proportional split by SM count
    uint32_t s_total = 0;
    for (auto s : sm_counts) s_total += s;
    if (s_total == 0) s_total = num_nodes;  // fallback: equal split

    const uint64_t max_count = 2 * w_est * static_cast<uint64_t>(H);

    std::vector<ContiguousRange> ranges;
    uint64_t cursor = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        uint64_t windows = (w_est * sm_counts[i] + s_total - 1) / s_total;
        if (windows == 0) windows = 1;  // every node gets at least 1 window
        uint64_t count = windows * H;

        // Apply headroom: extend each node's range by headroom_pct%
        if (headroom_pct > 0.0) {
            uint64_t extended = static_cast<uint64_t>(
                std::ceil(static_cast<double>(count) * (1.0 + headroom_pct / 100.0)));
            count = ((extended + H - 1) / H) * H;  // re-align to H
            if (Q > 1) count = ((count + Q - 1) / Q) * Q;
            if (count > max_count) {
                count = max_count;
                // ensure alignment still holds after cap
                count = (count / H) * H;
                if (Q > 1) count = (count / Q) * Q;
            }
        } else {
            // Quantum alignment on top of HC alignment
            if (Q > 1) {
                count = ((count + Q - 1) / Q) * Q;
            }
        }

        // DEBUG repro harness (DEFAULT-INERT): shrink initial ranges to force
        // overflow turnover. Cap is a multiple of H, preserving HC alignment.
        count = applyDebugWindowCap(count);

        ContiguousRange r;
        r.node_id = static_cast<uint8_t>(i);
        r.start = cursor;
        r.count = count;
        ranges.push_back(r);
        cursor += count;
    }
    return ranges;
}

// ============================================================================
// Throughput-proportional contiguous range computation (M4-S7)
// ============================================================================

std::vector<ChunkScheduler::ContiguousRange> ChunkScheduler::computeContiguousRanges(
        const std::vector<double>& throughput_weights,
        uint32_t target_rels,
        uint32_t H,
        uint32_t Q,
        double headroom_pct) const {

    const uint32_t num_nodes = static_cast<uint32_t>(throughput_weights.size());
    if (num_nodes == 0 || H == 0) return {};

    double t_total = 0.0;
    for (auto t : throughput_weights) t_total += t;
    if (t_total <= 0.0) {
        // Fallback: equal split (degenerate case — all probes returned 0)
        std::vector<uint32_t> equal(num_nodes, 1);
        return computeContiguousRanges(equal, target_rels, H, Q, headroom_pct);
    }

    // W_est = max(4 * num_nodes, ceil(target_rels / (5 * H)) * 2)
    uint64_t w_est = std::max(
        static_cast<uint64_t>(4) * num_nodes,
        ((static_cast<uint64_t>(target_rels) + 5 * H - 1) / (5 * H)) * 2);

    const uint64_t max_count = 2 * w_est * static_cast<uint64_t>(H);

    std::vector<ContiguousRange> ranges;
    uint64_t cursor = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        double fraction = throughput_weights[i] / t_total;
        uint64_t windows = std::max(uint64_t{1},
            static_cast<uint64_t>(std::ceil(w_est * fraction)));
        uint64_t count = windows * H;

        // Apply headroom: extend each node's range by headroom_pct%
        if (headroom_pct > 0.0) {
            uint64_t extended = static_cast<uint64_t>(
                std::ceil(static_cast<double>(count) * (1.0 + headroom_pct / 100.0)));
            count = ((extended + H - 1) / H) * H;  // re-align to H
            if (Q > 1) count = ((count + Q - 1) / Q) * Q;
            if (count > max_count) {
                count = max_count;
                count = (count / H) * H;
                if (Q > 1) count = (count / Q) * Q;
            }
        } else {
            if (Q > 1) count = ((count + Q - 1) / Q) * Q;
        }

        // DEBUG repro harness (DEFAULT-INERT): shrink initial ranges to force
        // overflow turnover. Cap is a multiple of H, preserving HC alignment.
        count = applyDebugWindowCap(count);

        ContiguousRange r;
        r.node_id = static_cast<uint8_t>(i);
        r.start   = cursor;
        r.count   = count;
        ranges.push_back(r);
        cursor += count;
    }
    return ranges;
}

// ============================================================================
// Alignment helpers
// ============================================================================

uint64_t ChunkScheduler::roundToQuantum(uint64_t count) const {
    if (Q_ <= 1) return count;
    return ((count + Q_ - 1) / Q_) * Q_;
}

uint64_t ChunkScheduler::alignToHypercube(uint64_t count) const {
    if (H_ == 0) return count;
    return ((count + H_ - 1) / H_) * H_;
}

uint64_t ChunkScheduler::applyDebugWindowCap(uint64_t count) const {
    // Parse MPQS_DEBUG_MAX_CHUNK_WINDOWS exactly once (0 = disabled).
    static const uint64_t cap_windows = [] {
        const char* env = std::getenv("MPQS_DEBUG_MAX_CHUNK_WINDOWS");
        if (!env || !*env) return uint64_t{0};
        try {
            const long long w = std::stoll(std::string(env));
            return (w >= 1) ? static_cast<uint64_t>(w) : uint64_t{0};
        } catch (...) {
            return uint64_t{0};
        }
    }();
    if (cap_windows == 0) return count;  // default-inert
    const uint64_t window = (H_ > 0) ? H_ : 1;
    const uint64_t cap = cap_windows * window;
    return std::min(count, cap);
}

uint64_t ChunkScheduler::minChunk() const {
    return std::max({static_cast<uint64_t>(H_), static_cast<uint64_t>(4) * Q_, kMinChunkFallback});
}

// ============================================================================
// Chunk sizing
// ============================================================================

uint64_t ChunkScheduler::baseChunk() const {
    uint64_t divisor = 10 * (static_cast<uint64_t>(num_workers_) + 1);
    uint64_t base = (divisor > 0) ? total_a_ / divisor : total_a_;
    return std::max(minChunk(), std::min(base, maxChunk()));
}

uint64_t ChunkScheduler::maxChunk() const {
    uint64_t divisor = 2 * (static_cast<uint64_t>(num_workers_) + 1);
    return std::max(minChunk(), (divisor > 0) ? total_a_ / divisor : total_a_);
}

void ChunkScheduler::recordCompletion(uint8_t worker_id, uint64_t rels,
                                       uint64_t a_vals, double elapsed_s) {
    std::lock_guard<std::mutex> lock(mu_);
    throughput_[worker_id].update(rels, a_vals, elapsed_s, kEmaAlpha);
}

uint64_t ChunkScheduler::nextChunkSize(uint8_t worker_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = throughput_.find(worker_id);
    if (it == throughput_.end() || it->second.chunks_completed < kCalibrationChunks)
        return applyDebugWindowCap(baseChunk());

    // Compute mean a_vals_per_sec across all workers with >= kCalibrationChunks
    double total_aps = 0.0;
    uint32_t count = 0;
    for (auto& [id, t] : throughput_) {
        if (t.chunks_completed >= kCalibrationChunks) {
            total_aps += t.a_vals_per_sec;
            count++;
        }
    }
    double mean_aps = (count > 0) ? total_aps / count : 1.0;
    double ratio = (mean_aps > 0) ? (it->second.a_vals_per_sec / mean_aps) : 1.0;

    // Confidence ramp: min(1.0, chunks_completed / 4.0)
    double confidence = std::min(1.0, static_cast<double>(it->second.chunks_completed) / 4.0);
    uint64_t size = static_cast<uint64_t>(baseChunk() * ratio * confidence);

    // Prefer hypercube alignment, then quantum rounding
    uint64_t aligned = alignToHypercube(size);
    aligned = roundToQuantum(aligned);
    // DEBUG repro harness (DEFAULT-INERT): cap overflow chunk size to force many
    // CHUNK_COMPLETE -> CHUNK_ASSIGN cycles. Applied after the [min,max] clamp so
    // the cap is the binding upper bound when armed.
    return applyDebugWindowCap(std::clamp(aligned, minChunk(), maxChunk()));
}

const WorkerThroughput& ChunkScheduler::throughput(uint8_t worker_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    static const WorkerThroughput empty{};
    auto it = throughput_.find(worker_id);
    return (it != throughput_.end()) ? it->second : empty;
}

} // namespace mpqs::cluster
