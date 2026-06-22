// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include "mpqs_soa.h"
#include <cstdint>

namespace mpqs::cluster {

/// Abstract callback interface injected into SieveStage() for cluster data extraction.
/// Default (solo mode): null pointer — all callbacks are no-ops.
/// Spec reference: cluster_v2_spec.md Section 3.3
class DataTap {
public:
    virtual ~DataTap() = default;

    /// Called after each postprocessor batch with incremental relation snapshots.
    /// MUST be non-blocking (< 50us). Enqueues to lock-free queue or send buffer.
    /// @param full_relations    New full relations from this extraction cycle.
    /// @param partials          New 1-partial relations (reset after each extraction).
    /// @param batch_index       Monotonic batch counter.
    /// @param a_values_advanced Number of polynomial a-values consumed since the
    ///        previous call. 0 means "use the per-batch default registered via
    ///        setRange" — single-batch loops pass 0; the CUDA-graph replay loop
    ///        passes the true count (= sieve_batch_size × cuda_graph_unroll), since
    ///        one replay advances graph_N batches but invokes this callback once.
    ///        Without this the per-worker a-range guard under-counts by the graph
    ///        unroll factor and nodes overrun into each other's a-ranges, producing
    ///        byte-identical duplicate partials (cluster duplicate-partial bug).
    virtual void onBatchComplete(
        const structures::HostRelationBatch& full_relations,
        const structures::HostRelationBatch& partials,
        uint64_t batch_index,
        uint64_t a_values_advanced = 0) = 0;

    /// Polled in sieve loop condition. Returns true when coordinator signals STOP.
    virtual bool shouldStop() const = 0;
};

} // namespace mpqs::cluster
