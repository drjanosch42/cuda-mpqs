// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// =============================================================================
// test_cluster_resume — focused unit tests for the S4 cluster-coordinator RESUME path
// =============================================================================
//
// Three independently-testable pieces of the S4 design, none of which need a live
// cluster (the live 2-node kill+resubmit smoke is a separate maintainer package):
//
//   (A) computeResumeTrim()       — per-node initial-range trim [orig_start+hw, orig_count-hw),
//                                    incl. the count==0 -> re-sieve-last-hypercube hardening and
//                                    the eff_hw + count == orig_count invariant (M1, option (a)).
//   (B) clusterResumeTopologyOk() — the N2 guard: REJECT a resume whose checkpoint node_count
//                                    differs from the current run, or whose completed-prefix
//                                    cursor falls outside the current overflow pool.
//   (C) re-inject ordering        — addRelations(smooths) rebuilds the RelationAccumulator
//                                    seen_ dedup set BEFORE the loaded partials' combines are
//                                    added, so a re-emitted leg-1 combine (loaded x loaded,
//                                    already persisted) is dropped while a genuine NEW combine
//                                    is kept (m-dedup-order). insertAndMatch() emits its combines
//                                    via addLPRelations() -> addRelations(255), so this pins the
//                                    exact dedup mechanism the cluster re-feed relies on.
//
// CPU-only host test (compiled as .cu only to share the build toolchain). Links mpqs_common
// for uint512 / HostRelationBatch / the shared computeRelationHash; the helpers under test are
// header-inline (sieve_checkpoint.h) and RelationAccumulator is header-only (accumulator.h).
// =============================================================================

#include "sieve_checkpoint.h"   // mpqs::ckpt::computeResumeTrim / clusterResumeTopologyOk / ResumeTrim
#include "accumulator.h"        // mpqs::cluster::RelationAccumulator
#include "mpqs_soa.h"

#include <cstdio>
#include <cstdint>
#include <vector>

using mpqs::ckpt::ResumeTrim;
using mpqs::ckpt::computeResumeTrim;
using mpqs::ckpt::clusterResumeTopologyOk;
using mpqs::cluster::RelationAccumulator;
using mpqs::structures::HostRelationBatch;

namespace {
int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); ++g_failures; } } while (0)

/// Append one relation to a HostRelationBatch (mirrors RelationAccumulator::appendSingleRelation
/// so the resulting batch hashes/dedups exactly like a real pooled relation). `factors` is a list
/// of (factor_index, exponent) pairs. The batch must already carry its CSR sentinel (offsets={0}).
void appendRel(HostRelationBatch& b, uint8_t sign, int32_t v2,
               const std::vector<std::pair<uint32_t, uint8_t>>& factors,
               unsigned __int128 large_prime = 0, uint32_t char_bit = 0) {
    if (b.factor_offsets.empty()) b.factor_offsets.push_back(0);
    b.sqrt_Q.push_back(mpqs::uint512{});   // value irrelevant to the dedup hash
    b.signs.push_back(sign);
    b.val_2_exps.push_back(v2);
    b.large_primes.push_back(large_prime);
    b.char_bits.push_back(char_bit);
    for (auto& [idx, cnt] : factors) {
        b.factor_indices.push_back(idx);
        b.factor_counts.push_back(cnt);
    }
    b.num_factors += factors.size();
    b.factor_offsets.push_back(b.num_factors);
    b.num_relations++;
}
} // namespace

// -----------------------------------------------------------------------------
// (A) computeResumeTrim — range trimming + boundary hardening (M1, option (a))
// -----------------------------------------------------------------------------
static void test_resume_trim() {
    const uint64_t H = 256;  // a representative hypercube size (2^8)

    // (A1) Normal mid-range trim: re-issue [orig_start + hw, orig_count - hw).
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/1000, /*orig_count=*/4096, /*hw=*/1024, H);
        CHECK(t.start  == 1000 + 1024, "A1: trimmed start == orig_start + hw");
        CHECK(t.count  == 4096 - 1024, "A1: trimmed count == orig_count - hw");
        CHECK(t.eff_hw == 1024,        "A1: eff_hw == hw");
        CHECK(t.eff_hw + t.count == 4096, "A1: eff_hw + count == orig_count");
    }

    // (A2) hw == 0 (no progress): the whole initial range is re-issued unchanged.
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/0, /*orig_count=*/4096, /*hw=*/0, H);
        CHECK(t.start == 0 && t.count == 4096 && t.eff_hw == 0, "A2: hw=0 -> full range, identity");
    }

    // (A3) hw == orig_count (initial range fully done): a naive trim would give count==0, which
    //      both DataTaps read as UNBOUNDED. The guard re-sieves the LAST hypercube instead, and
    //      eff_hw + count must still equal orig_count so a re-checkpoint records the absolute hw.
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/2000, /*orig_count=*/4096, /*hw=*/4096, H);
        CHECK(t.count == H,                  "A3: count==0 guard -> re-sieve last H");
        CHECK(t.start == 2000 + 4096 - H,    "A3: start == end - H");
        CHECK(t.eff_hw == 4096 - H,          "A3: eff_hw == orig_count - H");
        CHECK(t.eff_hw + t.count == 4096,    "A3: eff_hw + count == orig_count (re-checkpoint exact)");
    }

    // (A4) hw clamped: hw > orig_count (impossible but defensive) clamps to orig_count, then the
    //      A3 guard fires.
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/0, /*orig_count=*/4096, /*hw=*/9999, H);
        CHECK(t.count == H && t.eff_hw == 4096 - H, "A4: hw>orig_count clamps, then last-H guard");
    }

    // (A5) Idempotency: feeding eff_hw back in (as the WORK_ASSIGN loop does) reproduces the same
    //      {start, count} — no second trigger of the count==0 branch.
    {
        ResumeTrim t1 = computeResumeTrim(0, 4096, 4096, H);            // -> eff_hw = 4096 - H
        ResumeTrim t2 = computeResumeTrim(0, 4096, t1.eff_hw, H);
        CHECK(t1.start == t2.start && t1.count == t2.count && t1.eff_hw == t2.eff_hw,
              "A5: computeResumeTrim is idempotent under eff_hw");
    }

    // (A6) orig_count == 0 (a zero-weight node with no initial range): pass through unchanged.
    //      count stays 0 (that node already went straight to overflow on a fresh run); the guard
    //      does NOT fire (orig_count < H), so there is no spurious last-H re-sieve.
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/5000, /*orig_count=*/0, /*hw=*/0, H);
        CHECK(t.start == 5000 && t.count == 0 && t.eff_hw == 0, "A6: orig_count==0 passthrough");
    }

    // (A7) orig_count == H exactly (smallest non-empty range), fully done: re-sieve the whole
    //      single hypercube (eff_hw=0, count=H), never count==0.
    {
        ResumeTrim t = computeResumeTrim(/*orig_start=*/100, /*orig_count=*/H, /*hw=*/H, H);
        CHECK(t.start == 100 && t.count == H && t.eff_hw == 0, "A7: orig_count==H done -> whole H");
    }
}

// -----------------------------------------------------------------------------
// (B) clusterResumeTopologyOk — the N2 topology/geometry guard
// -----------------------------------------------------------------------------
static void test_topology_guard() {
    const uint64_t ov_start = 1000, ov_end = 5000;

    // (B1) Matching node count + prefix inside the pool -> OK.
    CHECK(clusterResumeTopologyOk(/*loaded=*/4, /*current=*/4, /*prefix=*/2500, ov_start, ov_end),
          "B1: matching topology + in-bounds prefix accepted");

    // (B2) Node-count mismatch (the headline N2 case) -> REJECT, so the per-node high-water
    //      array is never index->range mis-mapped onto a different topology.
    CHECK(!clusterResumeTopologyOk(/*loaded=*/8, /*current=*/4, 2500, ov_start, ov_end),
          "B2: node_count mismatch rejected (more loaded nodes)");
    CHECK(!clusterResumeTopologyOk(/*loaded=*/2, /*current=*/4, 2500, ov_start, ov_end),
          "B2b: node_count mismatch rejected (fewer loaded nodes)");

    // (B3) Prefix below overflow_start (a shifted overflow geometry) -> REJECT.
    CHECK(!clusterResumeTopologyOk(4, 4, /*prefix=*/999, ov_start, ov_end),
          "B3: prefix below overflow_start rejected");

    // (B4) Prefix above overflow_end (would trip setCursor's cursor<=end_ assert) -> REJECT.
    CHECK(!clusterResumeTopologyOk(4, 4, /*prefix=*/5001, ov_start, ov_end),
          "B4: prefix above overflow_end rejected");

    // (B5) Prefix exactly at the pool boundaries -> OK (a run that never reached overflow sits at
    //      overflow_start; a fully-drained pool sits at overflow_end).
    CHECK(clusterResumeTopologyOk(4, 4, ov_start, ov_start, ov_end), "B5a: prefix==overflow_start OK");
    CHECK(clusterResumeTopologyOk(4, 4, ov_end,   ov_start, ov_end), "B5b: prefix==overflow_end OK");
}

// -----------------------------------------------------------------------------
// (C) Re-inject ordering — smooths rebuild seen_ BEFORE partial combines (m-dedup-order)
// -----------------------------------------------------------------------------
static void test_reinject_ordering() {
    // R: a "loaded smooth" that is ALSO derivable as a leg-1 combine (loaded x loaded) on resume.
    // The cluster re-feed re-emits exactly such combines via insertAndMatch -> addLPRelations.
    HostRelationBatch loaded_smooths;
    appendRel(loaded_smooths, /*sign=*/1, /*v2=*/0, {{3, 1}, {7, 1}, {11, 2}});

    // C_dup: the re-emitted leg-1 combine — SAME factorization as R (so SAME dedup hash).
    HostRelationBatch reemitted_dup;
    appendRel(reemitted_dup, 1, 0, {{3, 1}, {7, 1}, {11, 2}});

    // C_new: a GENUINE cross-checkpoint combine (loaded partial x NEW partial) — different
    // factorization, so a different hash; this one MUST be kept.
    HostRelationBatch genuine_new;
    appendRel(genuine_new, 1, 0, {{3, 1}, {13, 1}, {17, 1}});

    // (C1) Canonical order: smooths via addRelations FIRST -> seen_ holds R's hash; the re-emitted
    //      duplicate combine is then dropped; the genuine new combine is appended.
    {
        RelationAccumulator acc(/*target=*/100);
        acc.addRelations(loaded_smooths, /*source_id=*/0);
        CHECK(acc.totalRelations() == 1, "C1: loaded smooth seeded accumulator");
        acc.addLPRelations(reemitted_dup);         // == insertAndMatch's re-emitted leg-1 combine
        CHECK(acc.totalRelations() == 1, "C1: re-emitted leg-1 combine deduped (seen_ already had it)");
        acc.addLPRelations(genuine_new);           // == a real cross-checkpoint combine
        CHECK(acc.totalRelations() == 2, "C1: genuine new cross-checkpoint combine kept");
    }

    // (C2) Demonstrate the invariant the ORDER protects: the dedup is a set union, so the FINAL
    //      count is the same even if combines arrive first — BUT only because addRelations(smooths)
    //      eventually dedups them. Re-feeding the same complete partial set can never inflate the
    //      pool past the persisted smooths (no phantom relations), which is the resume-correctness
    //      guarantee. (The canonical smooths-first order additionally avoids transiently inflating
    //      accumulated_ before Thread A starts.)
    {
        RelationAccumulator acc(/*target=*/100);
        acc.addLPRelations(reemitted_dup);         // combine arrives first (seen_ empty)
        CHECK(acc.totalRelations() == 1, "C2: combine added when seen_ empty");
        acc.addRelations(loaded_smooths, 0);       // the persisted smooth dedups against it
        CHECK(acc.totalRelations() == 1, "C2: union converges — no double-count either order");
    }
}

int main() {
    test_resume_trim();
    test_topology_guard();
    test_reinject_ordering();

    if (g_failures == 0) {
        std::printf("test_cluster_resume: ALL CHECKS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "test_cluster_resume: %d CHECK(s) FAILED\n", g_failures);
    return 1;
}
