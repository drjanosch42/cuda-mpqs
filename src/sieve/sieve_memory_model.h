// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

// =============================================================================
// Single source-of-truth sieve device-memory model.
//
// Historically the sieve bucket-memory formula
//   num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize * 8
// was hand-rolled in five places that had already drifted (loadStandardConfig's
// reduction loop, validateConfigs' LEQ_CHECK, KernelLaunchValidator::checkGlobalMem,
// its diagnose() message, and printBufferRecommendations). This header is the ONE
// place those formulas live; every site routes through the functions below.
//
// The estimator mirrors EXACTLY the ten cudaMalloc calls in
//   src/sieve/kernel.cu:588-597  (loadSievingData)
//   src/sieve/kernel.cu:604-622  (loadSievingDataParamTest re-allocates a subset)
// on the CALLER'S realized geometry — each call site passes ITS OWN
// globalBucketSize / sievingBlockSize / num_threadBlocks, preserving the existing
// geometry differences between loadStandardConfig, loadPartialCustomConfig and
// buildSieveConstants. "Mirror the cudaMalloc line" means identical arithmetic on
// the caller's values, NOT unifying everyone to one canonical sievingBlockSize.
//
// Pure host-side functions: no cudaMalloc, no device state, no logging. The
// __host__ annotation lets the file be included from .cu translation units; it is
// header-only so it adds no link edges (it lives in src/sieve so the sieve
// controller can call it WITHOUT depending on src/autotune — autotune depends on
// sieve, not vice-versa).
// =============================================================================

#include <cstdint>
#include "sieving_data_structs.h"   // primeDataSIQS, candidateRelation, uint512 (via uint512.cuh)

#ifndef __host__
#define __host__
#endif

namespace mpqs {
namespace sieve {

// -----------------------------------------------------------------------------
// Operative sieve-config memory budget, as an EXACT integer fraction num/den.
//
// The budget is `(vram * num) / den`, truncating integer arithmetic — see
// sieveBucketBudget() for the load-bearing reason this must NOT use `double`.
//
//   S1: 3/4 = 0.75 everywhere — byte-identical to the pre-refactor hand-rolled
//       `3*totalGlobalMem/4` at every site.
//   S2 (this stage): flipped to 4/5 = 0.80 in this ONE place. Every in-tree
//       sizing site (loadStandardConfig loop, validateConfigs LEQ_CHECK,
//       checkGlobalMem, its diagnose() message, printBufferRecommendations) AND
//       the autotune OOM guard read this single constant, so the autotune and
//       the production validator share one budget — the autotune can never pick
//       a config the production in-tree checks then reject. 0.80 (vs the old
//       0.75) reclaims ~5% of VRAM for the bucket while still leaving headroom
//       for the factor base / primeData / postprocessing-LP buffers that share
//       the same device; the total-footprint OOM guard (autotune side) budgets
//       those explicitly against this same fraction.
// -----------------------------------------------------------------------------
inline constexpr uint32_t kSieveBudgetNum = 4;
inline constexpr uint32_t kSieveBudgetDen = 5;

// -----------------------------------------------------------------------------
// Per-config device-memory footprint of one DeviceSievingController instance,
// in bytes, split so callers can budget the (dominant, M-linear) bucket against
// the rest of the footprint.
//
// Grouping of the ten kernel.cu:588-597 cudaMalloc terms:
//   bucket_bytes     = (6) dev_globalBucketEntries + (7) dev_globalBucketCounts
//   persistent_bytes = (1) dev_factorBase + (2) dev_rootN + (3) dev_a_factors
//                    + (4) dev_B_values + (5) dev_primeData
//   scratch_bytes    = (8) dev_blockRelationCounts + (9) dev_candidateRelations
//                    + (10) dev_indexToCandidate
// -----------------------------------------------------------------------------
struct SieveDeviceFootprint {
    uint64_t bucket_bytes;        ///< dev_globalBucketEntries + dev_globalBucketCounts (M-linear, dominant)
    uint64_t persistent_bytes;    ///< factorBase + rootN + a_factors + B_values + primeData
    uint64_t scratch_bytes;       ///< blockRelationCounts + candidateRelations + indexToCandidate
    uint64_t total() const { return bucket_bytes + persistent_bytes + scratch_bytes; }
};

// -----------------------------------------------------------------------------
// Closed-form footprint from the caller's REALIZED geometry. Mirrors EXACTLY the
// ten cudaMalloc calls in kernel.cu:588-597 — one estimator term per line:
//
//   (1)  dev_factorBase          = fb_size            * sizeof(uint32_t)
//   (2)  dev_rootN               = fb_size            * sizeof(uint32_t)   (rootN.size()==factorBase.size())
//   (3)  dev_a_factors           = shc_dim            * sizeof(uint32_t)
//   (4)  dev_B_values            = shc_dim            * sizeof(uint512)
//   (5)  dev_primeData           = fb_size            * sizeof(primeDataSIQS)
//   (5b) dev_primeBValues        = fb_size * shc_dim  * sizeof(uint32_t)   (B-values decoupled from primeDataSIQS)
//   (6)  dev_globalBucketEntries = num_polys * num_sievingBlocks * globalBucketSize * sizeof(uint64_t)
//   (7)  dev_globalBucketCounts  = num_polys * num_sievingBlocks                    * sizeof(uint32_t)
//   (8)  dev_blockRelationCounts = num_threadBlocks                                 * sizeof(uint32_t)
//   (9)  dev_candidateRelations  = maxRelationsPerBlock * num_threadBlocks * sizeof(candidateRelation)
//   (10) dev_indexToCandidate    = sievingBlockSize    * num_threadBlocks * sizeof(uint32_t)
//
// All products promote to uint64_t before multiplying (the bucket term in
// kernel.cu:593 casts the first factor to (long long) for the same reason).
// -----------------------------------------------------------------------------
inline __host__ SieveDeviceFootprint estimateSieveFootprint(
    uint64_t num_polysPerSieveCall,
    uint64_t num_sievingBlocksPerSieveCall,
    uint64_t globalBucketSize,
    uint64_t sievingBlockSize,
    uint32_t num_threadBlocks,         // ss_conf.num_threadBlocks (scratch grid)
    uint32_t maxRelationsPerBlock,     // gs_conf.maxRelationsPerBlock (64)
    uint64_t fb_size,
    uint32_t shc_dim)
{
    SieveDeviceFootprint fp;

    // (6) + (7): the dominant, M-linear bucket terms.
    fp.bucket_bytes =
          num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize * sizeof(uint64_t)
        + num_polysPerSieveCall * num_sievingBlocksPerSieveCall * sizeof(uint32_t);

    // (1)..(5b): persistent buffers (survive the candidate-reload path).
    fp.persistent_bytes =
          fb_size * sizeof(uint32_t)                       // (1) dev_factorBase
        + fb_size * sizeof(uint32_t)                       // (2) dev_rootN
        + (uint64_t)shc_dim * sizeof(uint32_t)             // (3) dev_a_factors
        + (uint64_t)shc_dim * sizeof(mpqs::uint512)        // (4) dev_B_values
        + fb_size * sizeof(primeDataSIQS)                  // (5) dev_primeData
        + fb_size * (uint64_t)shc_dim * sizeof(uint32_t);  // (5b) dev_primeBValues (B-values decoupled from struct)

    // (8)..(10): per-call scratch buffers (re-allocated by loadSievingDataParamTest).
    fp.scratch_bytes =
          (uint64_t)num_threadBlocks * sizeof(uint32_t)                                  // (8)  dev_blockRelationCounts
        + (uint64_t)maxRelationsPerBlock * num_threadBlocks * sizeof(candidateRelation)  // (9)  dev_candidateRelations
        + sievingBlockSize * num_threadBlocks * sizeof(uint32_t);                        // (10) dev_indexToCandidate

    return fp;
}

// -----------------------------------------------------------------------------
// The single dev_globalBucketEntries term (kernel.cu:593), entries only:
//   num_polys * num_sievingBlocks * globalBucketSize * sizeof(uint64_t)
//
// This is the EXACT term the four legacy "bucket fits the budget" checks/loops
// tested (loadStandardConfig loop, validateConfigs LEQ_CHECK, checkGlobalMem, and
// its diagnose() message) — none of them included the small dev_globalBucketCounts
// term. Those sites route through THIS helper to stay byte-identical; the COMPLETE
// bucket footprint (entries + counts) lives in SieveDeviceFootprint::bucket_bytes
// and is consumed only by the S2 total-footprint guard.
// -----------------------------------------------------------------------------
inline __host__ uint64_t bucketEntriesBytes(uint64_t num_polysPerSieveCall,
                                            uint64_t num_sievingBlocksPerSieveCall,
                                            uint64_t globalBucketSize)
{
    return num_polysPerSieveCall * num_sievingBlocksPerSieveCall * globalBucketSize * sizeof(uint64_t);
}

// -----------------------------------------------------------------------------
// Integer-exact byte budget for the sieve BUCKET, given the rest of the footprint.
//
//   budget = (free_or_total_vram * num) / den  -  non_bucket_bytes   (clamped >= 0)
//
// INTEGER-VS-DOUBLE HAZARD (load-bearing). Every legacy site computed the budget
// as the integer expression `3*totalGlobalMem/4` (truncating). A `double`
// `0.80*total` (or a uint64_t-vs-double comparison) can round differently at some
// VRAM sizes and flip the reduction loop's LAST halving -> a different seed
// num_polys. So this stays in exact integer arithmetic and the comparisons that
// consume it are same-type uint64_t. The operative budget is now num=4, den=5
// (= 0.80, S2); the general path `(v*num)/den` is integer-exact for it.
//
// The num==3&&den==4 special-case below is now dormant (S1 used 3/4) but is kept:
// it documents that the legacy call reduced bit-for-bit to `3*v/4`, and any future
// caller passing 3/4 still gets that exact value.
//
// Overflow note: free_or_total_vram * num for VRAM up to a few hundred GB and num
// a small constant cannot overflow uint64_t (a few hundred GB * 5 << 2^64).
// -----------------------------------------------------------------------------
inline __host__ uint64_t sieveBucketBudget(uint64_t free_or_total_vram,
                                           uint64_t non_bucket_bytes,
                                           uint32_t num,
                                           uint32_t den)
{
    // Bit-for-bit identical to the historical hand-rolled `3*v/4`.
    uint64_t gross = (num == 3 && den == 4)
                   ? (3 * free_or_total_vram / 4)
                   : (free_or_total_vram * num) / den;
    return (gross > non_bucket_bytes) ? (gross - non_bucket_bytes) : 0;
}

// -----------------------------------------------------------------------------
// Largest num_polys (power-of-two-reduced from `start`) whose bucket fits
// `budget_bytes`. This is the shared "halve num_polys until the bucket fits"
// loop extracted from loadStandardConfig (device_sieving_controller.cpp:827) and
// re-derived in printBufferRecommendations.
//
// Mirrors the original loop EXACTLY: the bucket term is
//   start * num_sievingBlocks * globalBucketSize * sizeof(uint64_t)
// (i.e. ONLY dev_globalBucketEntries — the same single term the legacy loops
// tested), compared `>` against budget_bytes with same-type uint64_t arithmetic.
//
// NOTE the two original loops differ only in their floor:
//   - loadStandardConfig (:827)        halves with no lower bound (can reach 0)
//   - printBufferRecommendations (:904) halves only while num_polys > 1
// This helper takes `min_num_polys` so each caller preserves its own floor
// byte-for-byte. loadStandardConfig passes 0; printBufferRecommendations passes 1.
// -----------------------------------------------------------------------------
inline __host__ uint32_t reduceNumPolysToBudget(uint32_t start_num_polys,
                                                uint64_t num_sievingBlocks,
                                                uint64_t globalBucketSize,
                                                uint64_t budget_bytes,
                                                uint32_t min_num_polys)
{
    uint32_t num_polys = start_num_polys;
    while (num_polys > min_num_polys &&
           (uint64_t)num_polys * num_sievingBlocks * globalBucketSize * sizeof(uint64_t) > budget_bytes) {
        num_polys >>= 1;
    }
    return num_polys;
}

} // namespace sieve
} // namespace mpqs
