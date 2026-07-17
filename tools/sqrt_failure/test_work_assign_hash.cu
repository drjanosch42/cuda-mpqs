// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Regression test for the v2 WORK_ASSIGN wire format (FB hash instead of FB arrays).
//
// At high F the former [factorBase][rootN] blob (fb_size*8 B) exceeded the 64 MiB
// TCP frame cap (RSA-155 F=500M -> ~105 MB), killing every worker. WORK_ASSIGN now
// ships a 64-bit FNV-1a hash of the coordinator's FB (computeFactorBaseHash);
// workers regenerate the FB from the shipped (N, F) via generateFactorBase
// (deterministic) and verify the hash before sieving.
//
//   (a) ROUND-TRIP — serializeWorkAssign -> deserializeWorkAssign preserves
//       N / fb_size / M / F / batch_size / threshold / lp1_bound / poly range /
//       target / AFactorsSnapshot; fb_hash_out == computeFactorBaseHash(sender);
//       the receiver's factorBase/rootN are NOT populated from the wire; the
//       payload is 132 + snapshot bytes, INDEPENDENT of fb_size.
//   (b) REGEN-EQUIVALENCE — generateFactorBase(N, F) run twice into independent
//       factoringData yields the identical hash (worker regen == coordinator FB).
//   (c) MISMATCH — any divergence (a factorBase prime, a rootN root, a truncated
//       FB, a different F) changes the hash, i.e. the worker fail-loud path fires.
//
// CPU-only host test (no kernel launches); compiled as CUDA only so the
// __host__ __device__ math headers compile.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "serialization.h"        // serializeWorkAssign / deserializeWorkAssign / computeFactorBaseHash
#include "common.h"               // factoringData, AFactorsSnapshot
#include "prime_algorithms.h"     // generateFactorBase
#include "uint512.cuh"

#include <cstdint>
#include <cstdio>
#include <vector>

using mpqs::uint512;
using mpqs::sieve::factoringData;
using mpqs::sieve::AFactorsSnapshot;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    }                                                                    \
} while (0)

// Build a factoringData with a real generated FB for the given (N, F).
static factoringData makeFData(const uint512& N, uint32_t F, uint32_t M) {
    factoringData fd{};
    fd.N = N;
    fd.F = F;
    fd.M = M;
    mpqs::sieve::generateFactorBase(&fd);
    return fd;
}

int main() {
    // Odd semiprime (10007 * 10009 = 100160063) — small enough for a fast FB,
    // structurally a real MPQS input (odd, squarefree).
    const uint512 N("100160063");
    const uint32_t F = 2000, M = 65536;

    // ------------------------------------------------------------------
    // (a) WORK_ASSIGN round-trip
    // ------------------------------------------------------------------
    std::printf("[a] WORK_ASSIGN round-trip\n");
    factoringData tx = makeFData(N, F, M);
    CHECK(tx.size > 10, "generated FB is non-trivial");
    // A few a_factors so shc_dim is nonzero on the wire.
    tx.a_factors = {2, 4, 6, 8};

    AFactorsSnapshot snap;
    snap.a_factors = {2, 4, 6, 8};
    snap.lowerHalfStart = 1;
    snap.upperHalfStart = 3;
    snap.shc_dim = 4;

    const uint64_t tx_hash = mpqs::cluster::computeFactorBaseHash(tx);

    auto [buf, len] = mpqs::cluster::serializeWorkAssign(
        tx, /*sieve_batch_size=*/16, /*threshold_override=*/12345,
        /*lp1_bound=*/1000000007ULL, /*poly_range_start=*/77,
        /*poly_range_count=*/4096, /*target_relations=*/99999, &snap);

    const size_t snapshot_size = 4 + snap.shc_dim * 4 + 8;
    CHECK(len == 132 + snapshot_size, "payload == 132 + snapshot bytes");

    factoringData rx{};
    uint32_t rx_bs = 0;
    uint64_t rx_thr = 0, rx_lp1 = 0, rx_start = 0, rx_count = 0, rx_target = 0;
    uint64_t rx_hash = 0;
    AFactorsSnapshot rx_snap;
    CHECK(mpqs::cluster::deserializeWorkAssign(buf.data(), len, rx, rx_bs, rx_thr,
                                               rx_lp1, rx_start, rx_count,
                                               rx_target, rx_hash, &rx_snap),
          "deserializeWorkAssign succeeds");

    CHECK(rx.N == tx.N, "N round-trips");
    CHECK(rx.M == tx.M, "M round-trips");
    CHECK(rx.F == tx.F, "F round-trips");
    CHECK(rx.size == tx.size, "fb_size round-trips");
    CHECK(rx_bs == 16, "sieve_batch_size round-trips");
    CHECK(rx_thr == 12345, "threshold_override round-trips");
    CHECK(rx_lp1 == 1000000007ULL, "lp1_bound round-trips");
    CHECK(rx_start == 77 && rx_count == 4096, "poly range round-trips");
    CHECK(rx_target == 99999, "target_relations round-trips");
    CHECK(rx_hash == tx_hash, "fb_hash_out == computeFactorBaseHash(sender)");
    CHECK(rx.factorBase.empty() && rx.rootN.empty(),
          "FB arrays are NOT shipped (receiver must regenerate)");
    CHECK(rx_snap.shc_dim == 4 && rx_snap.a_factors == snap.a_factors &&
          rx_snap.lowerHalfStart == 1 && rx_snap.upperHalfStart == 3,
          "AFactorsSnapshot round-trips");

    // Payload size must be independent of F: a much larger FB, same wire bytes.
    factoringData tx_bigF = makeFData(N, /*F=*/50000, M);
    tx_bigF.a_factors = tx.a_factors;
    CHECK(tx_bigF.size > 2 * tx.size, "F=50000 FB is much larger");
    auto [buf2, len2] = mpqs::cluster::serializeWorkAssign(
        tx_bigF, 16, 12345, 1000000007ULL, 77, 4096, 99999, &snap);
    CHECK(len2 == len, "payload length independent of fb_size");

    // Min-length guard: a truncated buffer (131 B) must be rejected.
    CHECK(!mpqs::cluster::deserializeWorkAssign(buf.data(), 131, rx, rx_bs, rx_thr,
                                                rx_lp1, rx_start, rx_count,
                                                rx_target, rx_hash, nullptr),
          "buffers shorter than 132 B are rejected");

    // ------------------------------------------------------------------
    // (b) Regen-equivalence: worker-side regeneration reproduces the hash
    // ------------------------------------------------------------------
    std::printf("[b] regen-equivalence\n");
    factoringData regen = makeFData(N, F, M);   // fresh, independent regen
    CHECK(mpqs::cluster::computeFactorBaseHash(regen) == tx_hash,
          "generateFactorBase(N, F) is deterministic (hash reproduces)");
    CHECK(regen.size == tx.size, "regenerated fb_size matches");

    // ------------------------------------------------------------------
    // (c) Mismatch detection: any FB divergence flips the hash
    // ------------------------------------------------------------------
    std::printf("[c] mismatch detection\n");
    {   // A single corrupted prime value.
        factoringData mut = makeFData(N, F, M);
        mut.factorBase[mut.size / 2] ^= 1u;
        CHECK(mpqs::cluster::computeFactorBaseHash(mut) != tx_hash,
              "one flipped factorBase entry changes the hash");
    }
    {   // A single corrupted root.
        factoringData mut = makeFData(N, F, M);
        mut.rootN[mut.size / 3] += 1u;
        CHECK(mpqs::cluster::computeFactorBaseHash(mut) != tx_hash,
              "one flipped rootN entry changes the hash");
    }
    {   // Truncated FB (fb_size is mixed into the hash).
        factoringData mut = makeFData(N, F, M);
        mut.factorBase.pop_back();
        mut.rootN.pop_back();
        CHECK(mpqs::cluster::computeFactorBaseHash(mut) != tx_hash,
              "truncated FB changes the hash");
    }
    {   // A different F (the divergent-launch-flag scenario the verify guards).
        factoringData mut = makeFData(N, /*F=*/2500, M);
        CHECK(mpqs::cluster::computeFactorBaseHash(mut) != tx_hash,
              "different F yields a different hash");
    }

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
