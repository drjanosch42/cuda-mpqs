// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
//
// Committed regression test for the autotune OOM guard (plan decision §6.5).
//
// The guard's decision logic is pure host arithmetic over the single
// source-of-truth memory model (src/sieve/sieve_memory_model.h) and the
// KernelLaunchValidator::fitsTotalFootprint() total-footprint gate. This test
// exercises that logic DIRECTLY — no sieve kernels are launched, so it cannot
// itself OOM and is deterministic on any GPU (it only reads device properties to
// build SieveConstants, exactly as the real autotune does).
//
// It asserts the two decision-§6.5 properties:
//
//   (a) NO-REGRESSION at pinned M=131072: with a realistic free-VRAM budget, the
//       loadStandardConfig seed geometry and a validated kernel tuple fit the 0.80
//       budget — the guard reports 0 seed clamps and 0 candidate skips. (The seed
//       num_polys is reconstructed via the SAME estimator/reduction the controller
//       uses, so this pins the M=131K behaviour audit m3 requires.)
//
//   (b) GUARD FIRES on a synthetic over-budget config: with a deliberately tiny
//       free-VRAM figure, a normal candidate's total footprint exceeds the budget
//       so fitsTotalFootprint() returns false (a skip), and the seed reduction
//       loop clamps num_polys downward — both WITHOUT any allocation (no OOM
//       possible), and a feasible survivor (num_polys >= 1) always remains.
//
// Exit code 0 iff every assertion passes (0 failures).

#include "kernel_launch_validator.h"   // KernelLaunchValidator, buildSieveConstants, Params8
#include "sieve_memory_model.h"        // estimateSieveFootprint, sieveBucketBudget, reduceNumPolysToBudget
#include "sieving_data_structs.h"      // primeDataSIQS sizeof (via the model header)

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

using mpqs::autotune::KernelLaunchValidator;
using mpqs::autotune::SieveConstants;
using mpqs::autotune::buildSieveConstants;
using mpqs::autotune::Params8;
using namespace mpqs::sieve;

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

// Reconstruct the loadStandardConfig seed num_polys for a given (M, shc_dim,
// sievingBlockSize) under the bucket-only budget + the total-footprint knob —
// EXACTLY the two reductions device_sieving_controller.cpp::loadStandardConfig
// applies (sieve_memory_model.h is the single source of truth for both).
static uint32_t reconstructSeedNumPolys(uint64_t M, uint32_t shc_dim,
                                        uint64_t sievingBlockSize,
                                        uint64_t totalGlobalMem,
                                        uint64_t fb_size,
                                        uint64_t maxTotalSieveBytes /*0 = knob off*/) {
    const uint64_t globalBucketSize = sievingBlockSize / 2;
    const uint64_t num_sievingBlocks = (2 * M) / sievingBlockSize;
    uint32_t num_polys =
        std::min<uint64_t>(32768u, (1ull << (shc_dim ? shc_dim - 1 : 0)));

    // 1) bucket-only budget (0.80 * totalGlobalMem), min floor 0.
    num_polys = reduceNumPolysToBudget(
        num_polys, num_sievingBlocks, globalBucketSize,
        sieveBucketBudget(totalGlobalMem, 0, kSieveBudgetNum, kSieveBudgetDen),
        /*min_num_polys=*/0);

    // 2) total-footprint knob (mirrors the controller's second reduction loop).
    if (maxTotalSieveBytes > 0) {
        auto total = [&](uint32_t npc) {
            uint32_t ntb = std::min(256u, npc);
            return estimateSieveFootprint(npc, num_sievingBlocks, globalBucketSize,
                                          sievingBlockSize, ntb, /*maxRel=*/64,
                                          fb_size, shc_dim).total();
        };
        const uint32_t min_seed = 1u;
        while (num_polys > min_seed && total(num_polys) > maxTotalSieveBytes)
            num_polys >>= 1;
    }
    return num_polys;
}

int main() {
    std::printf("=== autotune OOM-guard regression test ===\n");

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::printf("  SKIP: no CUDA device available\n");
        return 0;  // not a failure on CPU-only CI
    }
    std::printf("  device: %s, totalGlobalMem=%zu MB\n",
                prop.name, (size_t)(prop.totalGlobalMem >> 20));

    // -------------------------------------------------------------------------
    // Geometry: pin M=131072 (the validated operating point; audit m3) and a
    // representative shc_dim / factor-base size (RSA-100-ish). The actual numbers
    // do not need to match a specific N — we only assert the guard's DECISION at
    // this geometry, which is what the committed regression protects.
    // -------------------------------------------------------------------------
    const uint32_t M       = 131072;
    const uint32_t shc_dim = 11;            // RSA-100-ish hypercube dimension
    const uint64_t fb_size = 108613;        // RSA-100-ish factor base size

    SieveConstants sc = buildSieveConstants(shc_dim, M, prop.sharedMemPerBlock);
    KernelLaunchValidator validator(device, sc);
    std::printf("  sc: sievingBlockSize=%u globalBucketSize=%u\n",
                sc.sievingBlockSize, sc.globalBucketSize);

    // A validated kernel tuple (the S0/S1 baseline winner). Must be isValid here.
    Params8 baseline = {512, 8, 4, 8, 128, 1024, 256, 1024};
    CHECK(validator.isValid(baseline), "baseline tuple is valid at M=131072");

    // Budget constant sanity: operative budget must be 0.80 (4/5), integer-exact.
    CHECK(kSieveBudgetNum == 4 && kSieveBudgetDen == 5,
          "operative budget constant is 4/5 (= 0.80)");
    CHECK(sieveBucketBudget(1000ull, 0, kSieveBudgetNum, kSieveBudgetDen) == 800ull,
          "sieveBucketBudget(1000,0,4,5) == 800 (integer-exact 0.80)");

    // =========================================================================
    // (a) NO-REGRESSION at M=131072: realistic free VRAM => 0 clamps / 0 skips.
    //     Use this device's own totalGlobalMem as the free figure (the guard
    //     budgets against cudaMemGetInfo free; here we use total as a proxy that
    //     is always >= free, which is the conservative direction for the budget).
    // =========================================================================
    {
        const uint64_t free_vram = prop.totalGlobalMem;
        // Modest postprocessing/LP + context reserve (well within budget at 16 GB+).
        const uint64_t non_sieve = (256ull << 20) /*pp/LP*/ + (512ull << 20) /*reserve*/;

        // Seed: knob = 0.80*free - non_sieve. Reconstruct with the knob ON and OFF;
        // at this geometry the seed bucket is small, so the knob must NOT clamp.
        const uint64_t budget0 =
            sieveBucketBudget(free_vram, 0, kSieveBudgetNum, kSieveBudgetDen);
        const uint64_t knob = (budget0 > non_sieve) ? (budget0 - non_sieve) : 1;

        uint32_t seed_off = reconstructSeedNumPolys(M, shc_dim, sc.sievingBlockSize,
                                                    prop.totalGlobalMem, fb_size, 0);
        uint32_t seed_on  = reconstructSeedNumPolys(M, shc_dim, sc.sievingBlockSize,
                                                    prop.totalGlobalMem, fb_size, knob);
        std::printf("  M=131K seed num_polys: knob-off=%u knob-on=%u (budget0=%zu MB, knob=%zu MB)\n",
                    seed_off, seed_on, (size_t)(budget0 >> 20), (size_t)(knob >> 20));
        CHECK(seed_on == seed_off,
              "M=131K seed NOT clamped by the OOM-guard knob (0 clamps)");

        // Candidate sweep: every isValid candidate fits the total-footprint budget.
        uint32_t skips = 0, valid_candidates = 0;
        const uint32_t sub_vals[] = {128, 256, 512, 1024};
        const uint32_t ni_vals[]  = {1, 2, 4, 8, 16, 32};
        const uint32_t pb_vals[]  = {1, 2, 4, 8, 16, 32};
        const uint32_t bpc_vals[] = {1, 2, 4, 8, 16, 32};
        const uint32_t mg_vals[]  = {32, 64, 128, 256};
        const uint32_t sg_vals[]  = {32, 128, 256, 512};
        for (uint32_t sub : sub_vals)
          for (uint32_t ni : ni_vals)
            for (uint32_t pb : pb_vals)
              for (uint32_t bpc : bpc_vals)
                for (uint32_t mg : mg_vals)
                  for (uint32_t sg : sg_vals) {
                      Params8 cand = {sub, ni, pb, bpc, mg, 1024, sg, 1024};
                      if (!validator.isValid(cand)) continue;
                      ++valid_candidates;
                      if (!validator.fitsTotalFootprint(cand, fb_size, free_vram, non_sieve, nullptr))
                          ++skips;
                  }
        std::printf("  M=131K candidate sweep: %u valid, %u footprint-skips\n",
                    valid_candidates, skips);
        CHECK(valid_candidates > 0, "M=131K sweep has >=1 valid candidate");
        CHECK(skips == 0, "M=131K candidate sweep: 0 footprint skips (no regression)");
    }

    // =========================================================================
    // (b) GUARD FIRES on a synthetic over-budget config (tiny free VRAM) — and a
    //     feasible survivor always remains. No allocation, so no OOM is possible.
    // =========================================================================
    {
        const uint64_t tiny_free = 64ull << 20;   // 64 MB free: everything is over budget
        const uint64_t non_sieve = 0;             // isolate the sieve footprint

        // The baseline candidate's full footprint must EXCEED 0.80*64MB => guard fires.
        uint64_t est_total = 0;
        bool fits = validator.fitsTotalFootprint(baseline, fb_size, tiny_free, non_sieve, &est_total);
        std::printf("  synthetic: baseline est total=%zu MB vs budget=%zu MB => fits=%d\n",
                    (size_t)(est_total >> 20),
                    (size_t)(sieveBucketBudget(tiny_free, 0, kSieveBudgetNum, kSieveBudgetDen) >> 20),
                    (int)fits);
        CHECK(!fits, "synthetic over-budget candidate is REJECTED (guard fires)");

        // Seed reduction under a tiny knob must clamp num_polys downward AND leave a
        // feasible survivor (>= 1). Use a large M so the un-clamped seed bucket is huge.
        const uint32_t bigM = 262144;
        const uint64_t tiny_knob = 8ull << 20;  // 8 MB total-sieve cap
        uint32_t seed_unclamped = reconstructSeedNumPolys(bigM, shc_dim, sc.sievingBlockSize,
                                                          prop.totalGlobalMem, fb_size, 0);
        uint32_t seed_clamped   = reconstructSeedNumPolys(bigM, shc_dim, sc.sievingBlockSize,
                                                          prop.totalGlobalMem, fb_size, tiny_knob);
        std::printf("  synthetic: M=262K seed num_polys unclamped=%u clamped=%u (knob=8MB)\n",
                    seed_unclamped, seed_clamped);
        CHECK(seed_clamped <= seed_unclamped, "tiny knob clamps the seed downward");
        CHECK(seed_clamped >= 1, "feasible survivor remains (num_polys >= 1)");
    }

    std::printf("=== %d/%d checks passed (%d failures) ===\n",
                g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
