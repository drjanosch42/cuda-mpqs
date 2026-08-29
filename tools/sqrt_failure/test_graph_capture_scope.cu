// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Regression test for resolveGraphCaptureScope() — the --cuda_graph_capture decision.
//
// The resolver maps (cli request, has_lp, is_cluster, lp_diag_active, graph_N) to what
// the CUDA-graph capture actually covers. Two of its invariants are correctness-critical
// rather than performance-critical and are therefore asserted here rather than described
// in prose:
//
//   (a) CLUSTER NEVER CAPTURES LP — in cluster mode large primes are matched on the
//       coordinator CPU and no GPU matcher exists; capturing one would be a correctness
//       break. Asserted for every CLI value, including the belt-and-braces case
//       has_lp == true, and an explicit `full` must be downgraded *loudly*.
//   (b) MPQS_LP_DIAG=1 FORCES SIEVE-ONLY — the LP diagnostic block issues cudaMalloc /
//       cudaStreamSynchronize / synchronous cudaMemcpy, which fail inside a stream
//       capture (CUDA 900 / 906) and invalidate the whole graph.
//
// Further asserted: the solo/cluster AUTO defaults, that graph_N never downgrades the
// scope (graph_N == 1 is a documented probe value) while still reporting single_buffer,
// that a live-but-uncaptured LP path is flagged (between_replay_lp_retained) and that
// keep_host_sync holds whenever the graph does not order against the LP work.
//
// Pure host logic over plain scalars (the resolver is a free inline function with no
// dependency beyond <cstdint>); compiled as CUDA only for uniformity with the sibling
// host tests. No kernel launches, no GPU required.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "graph_capture_scope.h"

#include <cstdint>
#include <cstdio>

using namespace mpqs;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    }                                                                    \
} while (0)

static const GraphCaptureScope kAllCli[4] = {
    GraphCaptureScope::AUTO,
    GraphCaptureScope::SIEVE,
    GraphCaptureScope::POSTPROC,
    GraphCaptureScope::FULL
};

int main() {
    const uint32_t N4 = 4;  // representative unroll factor

    // ------------------------------------------------------------------
    // (1) Cluster never captures LP — for every CLI value, even with has_lp.
    // ------------------------------------------------------------------
    for (GraphCaptureScope cli : kAllCli) {
        auto r = resolveGraphCaptureScope(cli, /*has_lp=*/true, /*is_cluster=*/true,
                                          /*lp_diag_active=*/false, N4);
        CHECK(r.capture_lp == false, "cluster: capture_lp must be false (has_lp=true)");
        CHECK(r.achieved != GraphCaptureScope::FULL || r.capture_lp == false,
              "cluster: FULL scope must never imply in-graph LP");

        auto r0 = resolveGraphCaptureScope(cli, /*has_lp=*/false, /*is_cluster=*/true,
                                           /*lp_diag_active=*/false, N4);
        CHECK(r0.capture_lp == false, "cluster: capture_lp must be false (has_lp=false)");
    }

    // ------------------------------------------------------------------
    // (2) Explicit FULL in cluster downgrades to POSTPROC, loudly.
    // ------------------------------------------------------------------
    {
        auto r = resolveGraphCaptureScope(GraphCaptureScope::FULL, true, /*is_cluster=*/true,
                                          false, N4);
        CHECK(r.achieved == GraphCaptureScope::POSTPROC, "cluster FULL -> POSTPROC");
        CHECK(r.downgraded_cluster == true, "cluster FULL sets downgraded_cluster");
        CHECK(r.capture_postproc == true, "cluster FULL still captures postproc");
        CHECK(r.downgraded_lp_diag == false, "cluster FULL does not set the lp_diag flag");
    }

    // ------------------------------------------------------------------
    // (3) MPQS_LP_DIAG=1 forces SIEVE for every CLI value and both topologies.
    // ------------------------------------------------------------------
    for (GraphCaptureScope cli : kAllCli) {
        for (int cluster = 0; cluster <= 1; ++cluster) {
            auto r = resolveGraphCaptureScope(cli, /*has_lp=*/true, cluster != 0,
                                              /*lp_diag_active=*/true, N4);
            CHECK(r.achieved == GraphCaptureScope::SIEVE, "lp_diag -> SIEVE");
            CHECK(r.downgraded_lp_diag == true, "lp_diag sets downgraded_lp_diag");
            CHECK(r.capture_postproc == false, "lp_diag: no postproc capture");
            CHECK(r.capture_lp == false, "lp_diag: no LP capture");
        }
    }

    // ------------------------------------------------------------------
    // (4) Solo default (AUTO) = FULL with in-graph LP, no warnings.
    // ------------------------------------------------------------------
    {
        auto r = resolveGraphCaptureScope(GraphCaptureScope::AUTO, /*has_lp=*/true,
                                          /*is_cluster=*/false, false, N4);
        CHECK(r.achieved == GraphCaptureScope::FULL, "solo AUTO -> FULL");
        CHECK(r.capture_lp == true, "solo AUTO captures LP");
        CHECK(r.capture_postproc == true, "solo AUTO captures postproc");
        CHECK(r.downgraded_cluster == false, "solo AUTO is not a downgrade");
        CHECK(r.downgraded_lp_diag == false, "solo AUTO is not an lp_diag downgrade");
        CHECK(r.keep_host_sync == false, "solo AUTO with in-graph LP drops the host sync");
        CHECK(r.between_replay_lp_retained == false, "solo AUTO has no between-replay LP");

        // FULL scope without an LP matcher: scope stays FULL, nothing to capture.
        auto r_nolp = resolveGraphCaptureScope(GraphCaptureScope::AUTO, /*has_lp=*/false,
                                               /*is_cluster=*/false, false, N4);
        CHECK(r_nolp.achieved == GraphCaptureScope::FULL, "solo AUTO stays FULL without LP");
        CHECK(r_nolp.capture_lp == false, "no LP matcher -> capture_lp false");
        CHECK(r_nolp.between_replay_lp_retained == false, "no LP matcher -> nothing retained");
    }

    // ------------------------------------------------------------------
    // (5) Cluster default (AUTO) = POSTPROC, no warning flags.
    // ------------------------------------------------------------------
    {
        auto r = resolveGraphCaptureScope(GraphCaptureScope::AUTO, /*has_lp=*/false,
                                          /*is_cluster=*/true, false, N4);
        CHECK(r.achieved == GraphCaptureScope::POSTPROC, "cluster AUTO -> POSTPROC");
        CHECK(r.capture_postproc == true, "cluster AUTO captures postproc");
        CHECK(r.capture_lp == false, "cluster AUTO never captures LP");
        CHECK(r.downgraded_cluster == false, "cluster AUTO is not a downgrade");
        CHECK(r.downgraded_lp_diag == false, "cluster AUTO is not an lp_diag downgrade");
        CHECK(r.keep_host_sync == true, "cluster keeps the host sync");
    }

    // ------------------------------------------------------------------
    // (6) graph_N never downgrades; single_buffer == (graph_N == 1).
    // ------------------------------------------------------------------
    {
        const uint32_t unrolls[5] = {1, 2, 4, 8, 16};
        for (uint32_t n : unrolls) {
            for (GraphCaptureScope cli : kAllCli) {
                for (int cluster = 0; cluster <= 1; ++cluster) {
                    auto ref = resolveGraphCaptureScope(cli, true, cluster != 0, false, N4);
                    auto r   = resolveGraphCaptureScope(cli, true, cluster != 0, false, n);
                    CHECK(r.achieved == ref.achieved, "graph_N must not change the scope");
                    CHECK(r.capture_lp == ref.capture_lp, "graph_N must not change capture_lp");
                    CHECK(r.single_buffer == (n == 1), "single_buffer == (graph_N == 1)");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // (7) A live but uncaptured LP path is flagged for the between-replay drain.
    // ------------------------------------------------------------------
    {
        // Solo, scope explicitly reduced to postproc: the GPU matcher still runs,
        // just outside the graph.
        auto r = resolveGraphCaptureScope(GraphCaptureScope::POSTPROC, /*has_lp=*/true,
                                          /*is_cluster=*/false, false, N4);
        CHECK(r.capture_lp == false, "solo POSTPROC does not capture LP");
        CHECK(r.between_replay_lp_retained == true, "uncaptured LP must stay live");
        CHECK(r.keep_host_sync == true, "uncaptured LP forces keep_host_sync");

        auto rs = resolveGraphCaptureScope(GraphCaptureScope::SIEVE, /*has_lp=*/true,
                                           /*is_cluster=*/false, false, N4);
        CHECK(rs.between_replay_lp_retained == true, "SIEVE scope retains the LP path");
        CHECK(rs.capture_postproc == false, "SIEVE scope captures no postproc");
    }

    // ------------------------------------------------------------------
    // (8) keep_host_sync holds whenever is_cluster or !capture_lp — exhaustively.
    // ------------------------------------------------------------------
    for (GraphCaptureScope cli : kAllCli) {
        for (int has_lp = 0; has_lp <= 1; ++has_lp) {
            for (int cluster = 0; cluster <= 1; ++cluster) {
                for (int diag = 0; diag <= 1; ++diag) {
                    auto r = resolveGraphCaptureScope(cli, has_lp != 0, cluster != 0,
                                                      diag != 0, N4);
                    CHECK(r.keep_host_sync == (cluster != 0 || !r.capture_lp),
                          "keep_host_sync == is_cluster || !capture_lp");
                    CHECK(r.capture_postproc == (r.achieved == GraphCaptureScope::POSTPROC ||
                                                 r.achieved == GraphCaptureScope::FULL),
                          "capture_postproc == (achieved >= POSTPROC)");
                    CHECK(r.between_replay_lp_retained == (has_lp != 0 && !r.capture_lp),
                          "between_replay_lp_retained == has_lp && !capture_lp");
                }
            }
        }
    }

    // Log spellings are contract.
    CHECK(graphCaptureScopeName(GraphCaptureScope::SIEVE)[0] == 's', "name(SIEVE) = sieve");
    CHECK(graphCaptureScopeName(GraphCaptureScope::POSTPROC)[0] == 'p', "name(POSTPROC) = postproc");
    CHECK(graphCaptureScopeName(GraphCaptureScope::FULL)[0] == 'f', "name(FULL) = full");
    CHECK(graphCaptureScopeName(GraphCaptureScope::AUTO)[0] == 'a', "name(AUTO) = auto");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
