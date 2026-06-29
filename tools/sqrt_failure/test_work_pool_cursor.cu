// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// =============================================================================
// test_work_pool_cursor — focused unit test for WorkPool::completedPrefixCursor() (S3, B2)
// =============================================================================
//
// completedPrefixCursor() must return the COMPLETED contiguous prefix =
//   min( next_, min start over in_flight_ ∪ returned_ )
// and NOT nextCursor() (== next_), which silently drops in_flight_/returned_ chunks —
// resuming at next_ would skip a-values that were checked out but never completed, and
// dedup cannot recover a-values that were never sieved (re-introduces the 4d20d7b pool
// exhaustion). This test constructs a pool with known next_/in_flight_/returned_ states
// and asserts the min, including the case where it diverges from nextCursor().
//
// CPU-only host test (compiled as .cu only so it shares the build toolchain). work_pool.cpp
// is compiled directly into the target — no CUDA, no cluster-library link.
// =============================================================================

#include "work_pool.h"

#include <cstdio>
#include <cstdint>
#include <optional>

using mpqs::cluster::WorkPool;

namespace {
int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); ++g_failures; } } while (0)
} // namespace

int main() {
    // ---------------------------------------------------------------------
    // (1) Fresh pool: prefix == next_ == a_start.
    // ---------------------------------------------------------------------
    {
        WorkPool pool(/*a_start=*/1000, /*total_a=*/1000);  // [1000, 2000)
        CHECK(pool.completedPrefixCursor() == 1000, "fresh pool: prefix == a_start");
        CHECK(pool.nextCursor() == 1000, "fresh pool: next_ == a_start");
    }

    // ---------------------------------------------------------------------
    // (2) In-flight chunks bound the prefix below next_; completing the
    //     lowest-start chunk advances the prefix to the next in-flight start.
    // ---------------------------------------------------------------------
    {
        WorkPool pool(/*a_start=*/0, /*total_a=*/1000);  // [0, 1000)
        auto a = pool.checkoutWork(100, /*worker=*/1);   // A = [0, 100),   next_=100
        auto b = pool.checkoutWork(100, /*worker=*/2);   // B = [100, 200), next_=200
        CHECK(a && b, "checkouts succeeded");
        CHECK(pool.nextCursor() == 200, "next_ advanced to 200");
        // Both in flight ⇒ nothing completed ⇒ prefix pinned at the lowest start (0).
        CHECK(pool.completedPrefixCursor() == 0, "two in-flight: prefix pinned at 0");

        pool.completeChunk(a->chunk_id);                 // A done; B still in flight
        // [0,100) completed; B=[100,200) bounds the prefix.
        CHECK(pool.completedPrefixCursor() == 100, "after completing A: prefix == 100");
        CHECK(pool.nextCursor() == 200, "next_ unchanged at 200");

        pool.completeChunk(b->chunk_id);                 // B done; in_flight empty
        CHECK(pool.completedPrefixCursor() == 200, "all complete: prefix == next_ (200)");
    }

    // ---------------------------------------------------------------------
    // (3) returned_ chunks (reclaimed / never completed) bound the prefix —
    //     the B2 case where completedPrefixCursor() diverges from nextCursor().
    // ---------------------------------------------------------------------
    {
        WorkPool pool(/*a_start=*/0, /*total_a=*/1000);  // [0, 1000)
        auto a = pool.checkoutWork(200, 1);              // A = [0, 200),   next_=200
        auto b = pool.checkoutWork(200, 2);              // B = [200, 400), next_=400
        auto c = pool.checkoutWork(200, 1);              // C = [400, 600), next_=600
        CHECK(a && b && c, "three checkouts succeeded");
        CHECK(pool.nextCursor() == 600, "next_ at 600");

        pool.completeChunk(a->chunk_id);                 // A completed
        // in_flight = {B[200,400), C[400,600)} ⇒ prefix pinned at 200.
        CHECK(pool.completedPrefixCursor() == 200, "A done, B/C in flight: prefix 200");

        uint64_t ret = pool.returnChunk(b->chunk_id);    // B → returned_ (never completed)
        CHECK(ret == 200, "returnChunk returned 200 a-vals");
        pool.completeChunk(c->chunk_id);                 // C completed; in_flight empty
        // returned_ = {B[200,400)} ⇒ a-values [200,400) were NEVER sieved.
        // completedPrefixCursor must stop at 200 even though next_=600 and C is done.
        CHECK(pool.completedPrefixCursor() == 200,
              "returned B bounds prefix at 200 (B2: NOT next_)");
        CHECK(pool.nextCursor() == 600, "nextCursor() == 600 (would WRONGLY skip [200,600))");
        CHECK(pool.completedPrefixCursor() != pool.nextCursor(),
              "B2: completedPrefixCursor() diverges from nextCursor()");
    }

    // ---------------------------------------------------------------------
    // (4) reclaimPartial: the un-consumed remainder returns to returned_ and
    //     bounds the prefix at the recall point (start + consumed).
    // ---------------------------------------------------------------------
    {
        WorkPool pool(/*a_start=*/0, /*total_a=*/1000);
        auto a = pool.checkoutWork(300, 1);              // A = [0, 300), next_=300
        CHECK(a.has_value(), "checkout A");
        pool.reclaimPartial(a->chunk_id, /*consumed=*/120);
        // [0,120) consumed; remainder [120,300) → returned_ ⇒ prefix bounded at 120.
        CHECK(pool.completedPrefixCursor() == 120, "reclaimPartial: prefix at recall point 120");
    }

    if (g_failures == 0) {
        std::printf("work_pool_cursor: ALL CHECKS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "work_pool_cursor: %d CHECK(s) FAILED\n", g_failures);
    return 1;
}
