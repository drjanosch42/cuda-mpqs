// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file chunk_scheduler.h
/// @brief Per-worker EMA throughput tracker and adaptive chunk-size scheduler.
///
/// S5: Maintains per-worker throughput models via exponential moving average
/// and computes next chunk sizes proportional to observed throughput.
/// Two initial balance modes: SM_COUNT (default), MANUAL.
///
/// v2 extensions: contiguous range computation, quantum-aware rounding,
/// hypercube alignment. Replaces computeInitialChunks(WorkerState) with
/// computeContiguousRanges(sm_counts).

#include "work_pool.h"
#include "cluster_common.h"
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mpqs::cluster {

/// Initial balance strategy for first chunk assignment.
enum class InitBalanceMode {
    SM_COUNT,         ///< Default: proportional to worker SM counts
    MANUAL,           ///< Use capacity_estimate from HelloPayload (--cluster_capacity)
};

/// Per-worker throughput statistics (EMA-based).
struct WorkerThroughput {
    double rels_per_sec   = 0.0;  ///< EMA relation throughput
    double a_vals_per_sec = 0.0;  ///< EMA a-value consumption rate
    uint32_t chunks_completed = 0;
    uint64_t total_rels   = 0;
    uint64_t total_a_vals = 0;

    /// Update with completed chunk measurement.
    void update(uint64_t rels, uint64_t a_vals, double elapsed_s, double alpha);
};

/// Adaptive chunk scheduler with contiguous range computation and quantum alignment.
/// Maintains per-worker throughput models and computes next chunk sizes.
class ChunkScheduler {
public:
    /// @param pool         Work pool for overflow chunks
    /// @param num_workers  Total worker count (coordinator exclusive; coordinator is node 0)
    /// @param total_a      Total a-values in pool (for max_chunk computation)
    /// @param hypercube_size  H = 2^shc_dim (0 = no HC alignment)
    /// @param quantum      Q = batch_size * max(graph_unroll, 1) (1 = no quantum constraint)
    ChunkScheduler(WorkPool& pool, uint32_t num_workers, uint64_t total_a,
                   uint32_t hypercube_size = 0, uint32_t quantum = 1);

    /// Contiguous range for one node (coordinator = node 0).
    struct ContiguousRange {
        uint8_t  node_id;
        uint64_t start;       ///< Global a-index (multiple of H)
        uint64_t count;       ///< Number of a-values (multiple of H)
    };

    /// Compute contiguous range assignments at setup time.
    /// Formula: W_est = max(4*num_nodes, ceil(target_rels/(5*H))*2)
    ///          R_i   = ceil(W_est * S_i / S_total) * H
    /// @param sm_counts     SM counts per node [coord, w1, w2, ...]
    /// @param target_rels   Target relation count
    /// @param H             Hypercube size (2^shc_dim)
    /// @param Q             Quantum (batch_size * max(graph_unroll, 1))
    /// @return One range per node. Overflow pool starts at sum of all range counts.
    std::vector<ContiguousRange> computeContiguousRanges(
        const std::vector<uint32_t>& sm_counts,
        uint32_t target_rels,
        uint32_t H,
        uint32_t Q,
        double headroom_pct = 0.0) const;

    /// Compute throughput-proportional contiguous ranges (M4-S7 / SM2).
    /// Formula: W_est = max(4*num_nodes, ceil(target_rels/(5*H))*2)
    ///          R_i   = ceil(W_est * T_i / T_total) * H
    /// @param throughput_weights  Relative throughput per node [coord, w1, w2, ...]
    ///                            Units: SM×clock product, rels/s from probe, or capacity estimate.
    /// @param target_rels   Target relation count
    /// @param H             Hypercube size (2^shc_dim)
    /// @param Q             Quantum (batch_size * max(graph_unroll, 1))
    /// @param headroom_pct  Extend each node's range by this percent (default 0 = exact)
    /// @return One range per node. Overflow pool starts at sum of all range counts.
    std::vector<ContiguousRange> computeContiguousRanges(
        const std::vector<double>& throughput_weights,
        uint32_t target_rels,
        uint32_t H,
        uint32_t Q,
        double headroom_pct = 0.0) const;

    /// Record chunk completion and update throughput model.
    void recordCompletion(uint8_t worker_id, uint64_t rels, uint64_t a_vals, double elapsed_s);

    /// Compute next chunk size for a worker based on observed throughput.
    /// Returns baseChunk() before calibration is complete (< kCalibrationChunks).
    /// Result is hypercube-aligned and quantum-rounded.
    uint64_t nextChunkSize(uint8_t worker_id) const;

    /// Get per-worker throughput stats (const access).
    const WorkerThroughput& throughput(uint8_t worker_id) const;

    /// Minimum chunk size: max(H_, 4*Q_, kMinChunkFallback).
    uint64_t minChunk() const;

    void setInitialMode(InitBalanceMode mode) { init_mode_ = mode; }
    InitBalanceMode initialMode() const { return init_mode_; }

private:
    WorkPool& pool_;
    uint32_t  num_workers_;
    uint64_t  total_a_;
    uint32_t  H_ = 0;   ///< Hypercube size (2^shc_dim), 0 = no HC alignment
    uint32_t  Q_ = 1;   ///< Quantum (batch_size * max(graph_unroll, 1))
    InitBalanceMode init_mode_ = InitBalanceMode::SM_COUNT;

    mutable std::mutex mu_;  ///< Guards throughput_ map (Thread A + Thread B access)
    mutable std::unordered_map<uint8_t, WorkerThroughput> throughput_;

    static constexpr uint64_t kMinChunkFallback  = 16;  ///< Fallback when H_=0 and Q_=1
    static constexpr uint64_t kCalibrationChunks = 2;
    static constexpr double   kEmaAlpha          = 0.3;

    uint64_t baseChunk() const;   ///< total_a / (10 * (num_workers+1)), clamped [minChunk, maxChunk]
    uint64_t maxChunk()  const;   ///< total_a / (2 * (num_workers+1)), at least minChunk

    /// Round count up to nearest multiple of quantum Q_.
    uint64_t roundToQuantum(uint64_t count) const;

    /// Round count up to nearest multiple of hypercube size H_.
    uint64_t alignToHypercube(uint64_t count) const;

    /// DEBUG repro harness (DEFAULT-INERT): when the environment variable
    /// MPQS_DEBUG_MAX_CHUNK_WINDOWS=<W> is set (W>=1), cap any computed chunk
    /// `count` to W hypercube windows (W * max(H_,1) a-values). This shrinks
    /// both the initial contiguous ranges and the overflow chunks so that a
    /// short run produces many CHUNK_COMPLETE -> CHUNK_ASSIGN turnover cycles,
    /// exercising the overflow-assignment path. Returns the (possibly) capped
    /// count; identity when the var is unset/invalid (parsed once per process).
    uint64_t applyDebugWindowCap(uint64_t count) const;
};

} // namespace mpqs::cluster
