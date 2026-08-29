// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Committed regression test for the WIDE (uint16) num_polys apply-path clamp.
//
// It exercises the REAL production helper clampWideNumPolys() from
// src/sieve/sieve_memory_model.h — the single source of truth the custom apply
// path (device_sieving_controller.cpp::loadPartialCustomConfig, the
// autotune / pinned-tuple / AutoApply route) now runs to cap
// num_polysPerSieveCall at kWideNumPolysCap (512) on the wide path. Before the
// clamp that path set num_polysPerSieveCall directly from the tuned tuple, so an
// autotune winner could re-inflate polys and re-introduce the ~8-16x bucket-write
// traffic the 27f810e wide-geometry fix removed.
//
// Pure host arithmetic — no sieve kernels, no device allocation, deterministic on
// any machine (it does not even need a CUDA device; it is compiled as .cu only so
// the __host__ __device__ math headers pulled in by sieve_memory_model.h compile).
//
// Asserts:
//   (a) WIDE: the cap fires — a poly-tuple that would yield 1024 (or the raw
//       32768 seed) is clamped to exactly 512; values already <= 512 pass through.
//   (b) NARROW: the cap is a NO-OP — every input value passes through byte-for-byte
//       (the narrow uint8 path, which carries all validated <=RSA-140 records, is
//       provably unaffected).
//   (c) The num_subCubes compensation invariant: total polys sieved per hypercube
//       (num_subCubes * num_polys) is preserved across the clamp, so the clamp
//       changes ONLY per-launch bucket traffic, never the amount of work.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "sieve_memory_model.h"   // clampWideNumPolys, kWideNumPolysCap

#include <cstdint>
#include <cstdio>

using mpqs::sieve::clampWideNumPolys;
using mpqs::sieve::kWideNumPolysCap;

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

// Mirror of loadPartialCustomConfig's num_subCubes derivation (line ~1108) so we
// can assert the clamp preserves total polys sieved. min(32768, 2^(shc_dim-1)) is
// the per-hypercube poly budget; num_subCubes = that / num_polysPerSieveCall.
static uint32_t numSubCubes(uint32_t shc_dim, uint32_t num_polys) {
    uint32_t budget = 32768u;
    uint32_t half = (shc_dim >= 1) ? (1u << (shc_dim - 1)) : 0u;
    if (half < budget) budget = half;
    return (num_polys == 0) ? 0 : budget / num_polys;
}

int main() {
    std::printf("=== wide num_polys apply-path clamp regression test ===\n");

    // Constant sanity: the cap the custom path and loadStandardConfig both enforce.
    CHECK(kWideNumPolysCap == 512u, "kWideNumPolysCap == 512");

    // -------------------------------------------------------------------------
    // (a) WIDE path: the cap fires.
    // -------------------------------------------------------------------------
    CHECK(clampWideNumPolys(1024u, /*wide=*/true) == 512u,
          "WIDE: 1024 -> 512 (cap fires — the H-A re-inflation hole is closed)");
    CHECK(clampWideNumPolys(2048u, true) == 512u, "WIDE: 2048 -> 512");
    CHECK(clampWideNumPolys(32768u, true) == 512u,
          "WIDE: 32768 (raw seed) -> 512");
    CHECK(clampWideNumPolys(513u, true) == 512u, "WIDE: 513 -> 512");
    CHECK(clampWideNumPolys(512u, true) == 512u, "WIDE: 512 -> 512 (idempotent at cap)");
    CHECK(clampWideNumPolys(256u, true) == 256u, "WIDE: 256 -> 256 (below cap, unchanged)");
    CHECK(clampWideNumPolys(1u, true) == 1u, "WIDE: 1 -> 1 (below cap, unchanged)");

    // -------------------------------------------------------------------------
    // (b) NARROW path: the cap is a NO-OP — byte-for-byte passthrough. This is the
    //     load-bearing guarantee that the narrow (uint8) autotune win is untouched.
    // -------------------------------------------------------------------------
    const uint32_t narrow_inputs[] = {1u, 256u, 512u, 513u, 1024u, 2048u, 8192u, 32768u};
    bool narrow_ok = true;
    for (uint32_t v : narrow_inputs)
        if (clampWideNumPolys(v, /*wide=*/false) != v) narrow_ok = false;
    CHECK(narrow_ok, "NARROW: every input passes through unchanged (clamp is a no-op)");

    // -------------------------------------------------------------------------
    // (c) num_subCubes compensation: total polys sieved per hypercube is preserved.
    //     Pick shc_dim=14 => per-hypercube budget min(32768, 2^13)=8192.
    //       unclamped 1024 -> 8 subcubes  (8*1024   = 8192)
    //       clamped   512  -> 16 subcubes (16*512   = 8192)
    // -------------------------------------------------------------------------
    {
        const uint32_t shc_dim = 14;              // budget = min(32768, 8192) = 8192
        const uint32_t raw     = 1024u;
        const uint32_t clamped = clampWideNumPolys(raw, /*wide=*/true);
        const uint32_t total_raw     = numSubCubes(shc_dim, raw)     * raw;
        const uint32_t total_clamped = numSubCubes(shc_dim, clamped) * clamped;
        std::printf("  shc_dim=%u: raw np=%u (subCubes=%u), clamped np=%u (subCubes=%u)\n",
                    shc_dim, raw, numSubCubes(shc_dim, raw),
                    clamped, numSubCubes(shc_dim, clamped));
        CHECK(clamped == 512u, "compensation: raw 1024 clamps to 512");
        CHECK(total_raw == total_clamped,
              "compensation: total polys/hypercube preserved (only bucket traffic drops)");
    }

    std::printf("=== %d/%d checks passed (%d failures) ===\n",
                g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
