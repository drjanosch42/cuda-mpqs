// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// CUDA-graph capture scope: how much of the per-batch pipeline
// (sieve -> post-processing -> GPU large-prime matching) is captured into the
// --cuda_graph_unroll graph.
//
// The resolver below is a pure function of (CLI request, run context). It maps
//     (cli, has_lp, is_cluster, lp_diag_active, graph_N)  ->  a concrete decision,
// and is deliberately dependency-free (<cstdint> only) so it can be unit-tested
// without linking the orchestrator.
//
// Invariants enforced here (and asserted by tools/sqrt_failure/test_graph_capture_scope.cu):
//   (I1) CLUSTER NEVER CAPTURES LP. In cluster mode large primes are matched on the
//        coordinator CPU (src/cluster/cpu_lp.cu) and largeprime_ is not even
//        instantiated (orchestrator.cpp: `if (!cluster_mode) initLargePrimes();`).
//        An explicit `full` request is downgraded to `postproc` and WARNED about,
//        never silently honoured.
//   (I2) MPQS_LP_DIAG=1 FORCES SIEVE-ONLY. The LP diagnostic block
//        (src/largeprimes/largeprime.cu) calls cudaMalloc / cudaStreamSynchronize /
//        synchronous cudaMemcpy, which return CUDA errors 900 / 906 inside a stream
//        capture and invalidate the whole graph.
//   (I3) graph_N NEVER DOWNGRADES THE SCOPE. graph_N == 1 is a documented probe value:
//        the capture body runs on the host exactly once (buffer assignment is baked per
//        graph node, it cannot drift between replays) and consecutive cudaGraphLaunch
//        calls into one stream are fully serialized. graph_N is an input solely so the
//        no-intra-replay-overlap case can be reported (`single_buffer`).
//   (I4) keep_host_sync = is_cluster || !capture_lp. Whenever LP is NOT in the graph but
//        still runs (between replays, on the post-processing stream), the graph does not
//        order against that stream, so the per-replay host synchronization must stay.

#pragma once

#include <cstdint>

namespace mpqs {

/// Requested / achieved capture scope, ordered by increasing coverage.
enum class GraphCaptureScope : uint8_t {
    AUTO = 0,   ///< internal default: FULL in solo, POSTPROC in cluster. NOT accepted on the CLI.
    SIEVE,      ///< capture sieve kernels only — exactly the v1.0.5 behaviour; the rollback path
    POSTPROC,   ///< capture sieve + per-batch post-processing
    FULL        ///< + GPU large-prime matching (SOLO only)
};

/// Outcome of scope resolution. Consumed by the capture rewrite; every field is derived,
/// none is a free parameter.
struct GraphCaptureScopeResult {
    GraphCaptureScope achieved;      ///< what will actually be captured
    bool capture_postproc;           ///< achieved >= POSTPROC
    bool capture_lp;                 ///< achieved == FULL (never true when is_cluster)
    bool keep_host_sync;             ///< keep the per-replay host sync (cluster, or scope < FULL)
    bool single_buffer;              ///< graph_N == 1: no intra-replay overlap (informational)
    bool downgraded_cluster;         ///< an explicit FULL was downgraded because of cluster mode
    bool downgraded_lp_diag;         ///< downgraded to SIEVE because MPQS_LP_DIAG=1
    bool between_replay_lp_retained; ///< has_lp && !capture_lp: the non-captured LP path stays live
};

/**
 * @brief Resolve the effective CUDA-graph capture scope.
 *
 * @param cli             requested scope; AUTO = context default (FULL solo / POSTPROC cluster)
 * @param has_lp          a GPU large-prime matcher exists (SOLO only, by construction)
 * @param is_cluster      cluster_mode != SOLO (belt: || data_tap_ != nullptr)
 * @param lp_diag_active  MPQS_LP_DIAG=1 — uncapturable diagnostics are live
 * @param graph_N         --cuda_graph_unroll; reported, never a downgrade trigger (I3)
 */
inline GraphCaptureScopeResult resolveGraphCaptureScope(GraphCaptureScope cli,
                                                        bool has_lp,
                                                        bool is_cluster,
                                                        bool lp_diag_active,
                                                        uint32_t graph_N) {
    GraphCaptureScopeResult r{};
    r.single_buffer = (graph_N == 1);

    if (lp_diag_active) {
        // (I2) nothing capturable while the LP diagnostics run.
        r.achieved            = GraphCaptureScope::SIEVE;
        r.downgraded_lp_diag  = true;
    } else if (cli == GraphCaptureScope::AUTO) {
        // Context default — never a warning.
        r.achieved = is_cluster ? GraphCaptureScope::POSTPROC : GraphCaptureScope::FULL;
    } else if (cli == GraphCaptureScope::FULL && is_cluster) {
        // (I1) explicit request that cannot be honoured — downgrade loudly.
        r.achieved           = GraphCaptureScope::POSTPROC;
        r.downgraded_cluster = true;
    } else {
        r.achieved = cli;
    }

    r.capture_postproc = (r.achieved == GraphCaptureScope::POSTPROC ||
                          r.achieved == GraphCaptureScope::FULL);
    r.capture_lp       = (r.achieved == GraphCaptureScope::FULL) && has_lp && !is_cluster;
    r.keep_host_sync   = is_cluster || !r.capture_lp;             // (I4)
    r.between_replay_lp_retained = has_lp && !r.capture_lp;
    return r;
}

/// Renders a scope for log output. Lives here so every call site prints the same spellings.
inline const char* graphCaptureScopeName(GraphCaptureScope s) {
    switch (s) {
        case GraphCaptureScope::SIEVE:    return "sieve";
        case GraphCaptureScope::POSTPROC: return "postproc";
        case GraphCaptureScope::FULL:     return "full";
        default:                          return "auto";
    }
}

} // namespace mpqs
