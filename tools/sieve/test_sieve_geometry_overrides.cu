// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Committed regression test for the v1.0.6 narrow-batch sieve geometry overrides
// (--sieve_block_size / --sieve_big_prime_start) and for the narrow-batch interval-coverage
// invariant they make reachable.
//
// It exercises the REAL production helpers from src/sieve/sieve_memory_model.h --
// narrowBatchCoverageOk(), admissibleSieveBlockSize(), admissibleBigPrimeStart() -- which are
// the single source of truth for these rules: validateConfigs() calls the first, and the CLI
// (tests/cuda-mpqs.cpp) calls the other two. Nothing is restated here, so the test cannot drift
// from the shipped predicate.
//
// Why the coverage invariant matters (device_sieving_controller.cpp):
// DeviceSievingController::runSievingBatch() -- the production batch entry -- never advances
// ds_params.startIndex, and the batch GATHER loop runs num_sievingBlocksPerSieveCall blocks from
// that single origin. So gs_conf.num_sievingBlockBatches is a FICTION on the narrow batch path,
// and the loader's pre-existing [C1] guard -- which multiplies coverage BY that count -- PASSES
// a batch config that silently sieves only the half-interval [-M, 0). That is a ~50 % yield loss
// with no error, and lowering SB via --sieve_block_size at a fixed numIntervals is exactly how a
// probe would reach it.
//
// Pure host arithmetic -- no sieve kernels, no device allocation, no CUDA device needed
// (compiled as .cu only so the __host__ __device__ math headers pulled in by
// sieve_memory_model.h compile). Deterministic on any machine.
//
// Asserts:
//   (A) Every shipped narrow-batch --params tuple, at every supported device-derived SB,
//       satisfies the coverage invariant -- i.e. the new guard is a NO-OP on production.
//   (B) The halved-SB failure case (numIntervals 8, SB 32768, M 262144) is REJECTED and its
//       fix (numIntervals 16) is ACCEPTED.
//   (C) Over-coverage (SB == M, the autotune M-sweep regime) is ACCEPTED -- the invariant is
//       >= 2M, never == 2M.
//   (D) The admissible-set predicates accept the intended values and reject 0-adjacent,
//       non-power-of-two, and <= midPrimeStartIndex inputs.
//   (E) No uint32 overflow: a large-M config that would wrap a 32-bit product must not be
//       reported as covered.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "sieve_memory_model.h"

#include <cstdint>
#include <cstdio>

using mpqs::sieve::narrowBatchCoverageOk;
using mpqs::sieve::admissibleSieveBlockSize;
using mpqs::sieve::admissibleBigPrimeStart;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do {                                            \
    ++g_checks;                                                          \
    if (!(cond)) { ++g_failures;                                         \
        std::printf("  FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
    } else {                                                             \
        std::printf("  ok:   %s\n", (msg));                              \
    }                                                                    \
} while (0)

int main()
{
    std::printf("=== v1.0.6 sieve geometry overrides + narrow-batch coverage invariant ===\n");

    // -------------------------------------------------------------------------------------
    // (A) The guard is a NO-OP on every shipped narrow-batch geometry.
    //
    // On the loadStandardConfig path num_sievingBlocksPerSieveCall is DERIVED as 2M/SB, so
    // coverage is exactly 2M by construction. On the --params path the shipped tuples pair
    // numIntervals with the device's budget-derived SB so the product is 2M. Both are checked
    // at the three production M values and the three device SB rungs actually observed
    // (A100/H100 65536, RTX 5070 Ti 65536, and the Turing-class 32768 the loader names).
    // -------------------------------------------------------------------------------------
    std::printf("[A] shipped narrow-batch geometries\n");
    struct Shipped { uint32_t intervals, SB, M; const char* what; };
    const Shipped shipped[] = {
        // --params tuples of record (field 2 = numIntervals), at their measured device SB.
        {  8,  65536, 262144, "RTX 5070 Ti RSA-100 record 560,8,... SB=65536 M=262144" },
        {  8,  65536, 262144, "A100 RSA-100 exact-wave 864,8,... SB=65536 M=262144" },
        {  8,  65536, 262144, "A100 pow2 1024,8,... SB=65536 M=262144" },
        // v1.0.6 halved-SB arms: SB halved AND numIntervals doubled, product preserved.
        { 16,  32768, 262144, "halved-SB 864,16,... SB=32768 M=262144" },
        // loadStandardConfig-style derivations: intervals = 2M/SB, exact by construction.
        {  4, 131072, 262144, "derived intervals=2M/SB at SB=131072" },
        { 16,  32768, 262144, "derived intervals=2M/SB at SB=32768" },
        { 32,  32768, 524288, "derived intervals=2M/SB at M=524288" },
        {  8,  65536, 262144, "derived intervals=2M/SB at SB=65536" },
    };
    for (const auto& t : shipped)
        CHECK(narrowBatchCoverageOk(t.intervals, t.SB, t.M), t.what);

    // -------------------------------------------------------------------------------------
    // (B) The half-sieve case and its fix. This is the single most likely way a first halved-SB
    //     run silently produces wrong results: halve SB with --sieve_block_size but leave
    //     --params field 2 at its production value.
    // -------------------------------------------------------------------------------------
    std::printf("[B] the half-sieve case and its fix\n");
    CHECK(!narrowBatchCoverageOk(8, 32768, 262144),
          "REJECT intervals=8, SB=32768, M=262144 (covers M, not 2M -> sieves only [-M,0))");
    CHECK(narrowBatchCoverageOk(16, 32768, 262144),
          "ACCEPT intervals=16, SB=32768, M=262144 (the fix: numIntervals doubled with SB)");
    CHECK(!narrowBatchCoverageOk(4, 32768, 262144),
          "REJECT intervals=4, SB=32768 (quarter coverage)");
    CHECK(!narrowBatchCoverageOk(8, 32768, 262145),
          "REJECT one short of exact coverage (boundary is 2M, strictly)");
    CHECK(narrowBatchCoverageOk(2, 131072, 131072),
          "ACCEPT exact coverage 2*131072 = 2M at M=131072");
    // The Turing-class case the loader's own comment names: already half-sieving today,
    // silently. From v1.0.6 it is rejected LOUDLY -- a deliberate behaviour change.
    CHECK(!narrowBatchCoverageOk(8, 32768, 262144),
          "REJECT the loader-named Turing intervals=8/SB=32768 at M=262144");

    // -------------------------------------------------------------------------------------
    // (C) Over-coverage stays legal. The full autotune M-sweep drives SB = min(M, ...) == M
    //     with the tuple's interval count held fixed, giving intervals*SB > 2M. That is the
    //     correct conservative behaviour and the loader already blesses it, so the invariant
    //     is >= 2M and never == 2M.
    // -------------------------------------------------------------------------------------
    std::printf("[C] over-coverage is legal\n");
    CHECK(narrowBatchCoverageOk(8, 262144, 262144), "ACCEPT SB == M with intervals=8 (4x over)");
    CHECK(narrowBatchCoverageOk(2, 262144, 262144), "ACCEPT SB == M with intervals=2 (exact 2M)");
    CHECK(narrowBatchCoverageOk(64, 8192, 262144),  "ACCEPT small SB with intervals raised to match");

    // -------------------------------------------------------------------------------------
    // (D) Admissible sets. Never a silent floor or clamp -- these predicates only ever say
    //     yes or no, and the CLI turns a no into exit(1) naming the rule.
    // -------------------------------------------------------------------------------------
    std::printf("[D] admissible sets\n");
    CHECK(admissibleSieveBlockSize(0),      "SB override: 0 (OFF) is admissible");
    CHECK(admissibleSieveBlockSize(32768),  "SB override: 32768 admissible (the halved-SB value)");
    CHECK(admissibleSieveBlockSize(65536),  "SB override: 65536 admissible (production)");
    CHECK(admissibleSieveBlockSize(131072), "SB override: 131072 admissible (deferred Tier 3)");
    CHECK(admissibleSieveBlockSize(256),    "SB override: 256 admissible (lower bound, inclusive)");
    CHECK(!admissibleSieveBlockSize(128),   "SB override: 128 REJECTED (below the 256 floor)");
    CHECK(!admissibleSieveBlockSize(1),     "SB override: 1 REJECTED (0-adjacent)");
    CHECK(!admissibleSieveBlockSize(48000), "SB override: 48000 REJECTED (not a power of two)");
    CHECK(!admissibleSieveBlockSize(65535), "SB override: 65535 REJECTED (not a power of two)");
    CHECK(!admissibleSieveBlockSize(98304), "SB override: 98304 REJECTED (3*2^15, not a power of two)");

    CHECK(admissibleBigPrimeStart(0),    "bPSI override: 0 (OFF) is admissible");
    CHECK(admissibleBigPrimeStart(1024), "bPSI override: 1024 admissible (the halved-SB value)");
    CHECK(admissibleBigPrimeStart(2048), "bPSI override: 2048 admissible (the doubled-bPSI value)");
    CHECK(admissibleBigPrimeStart(33),   "bPSI override: 33 admissible (just above midPrimeStart)");
    CHECK(admissibleBigPrimeStart(1000), "bPSI override: 1000 admissible (pow2 NOT required)");
    CHECK(!admissibleBigPrimeStart(32),  "bPSI override: 32 REJECTED (== midPrimeStartIndex)");
    CHECK(!admissibleBigPrimeStart(16),  "bPSI override: 16 REJECTED (mid-prime range would invert)");
    CHECK(!admissibleBigPrimeStart(1),   "bPSI override: 1 REJECTED");

    // -------------------------------------------------------------------------------------
    // (E) No uint32 overflow. intervals*SB exceeds 2^32 for large but representable M, and a
    //     32-bit product would wrap to a small value and report a FALSE pass.
    // -------------------------------------------------------------------------------------
    std::printf("[E] 64-bit coverage arithmetic\n");
    CHECK(narrowBatchCoverageOk(65536u, 65536u, 2147483647u),
          "ACCEPT 65536*65536 = 2^32 >= 2*(2^31-1) (would wrap to 0 in uint32)");
    CHECK(!narrowBatchCoverageOk(32768u, 65536u, 2147483647u),
          "REJECT 32768*65536 = 2^31 < 2*(2^31-1) (no false pass from wrapping)");
    CHECK(!narrowBatchCoverageOk(0u, 65536u, 262144u),
          "REJECT intervals = 0 (zero coverage)");
    CHECK(!narrowBatchCoverageOk(8u, 0u, 262144u),
          "REJECT SB = 0 (zero coverage)");

    std::printf("=== %d checks, %d failures ===\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
