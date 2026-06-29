// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "kernel_launch_validator.h"
#include "orchestrator.h"
#include "sieve_memory_model.h"   // single source-of-truth sieve memory model
#include <bit>       // std::countl_zero (C++20)
#include <algorithm> // std::min
#include <cstring>   // memset
#include <string>

namespace mpqs::autotune {

// ---------------------------------------------------------------------------
// Construction: query device properties once
// ---------------------------------------------------------------------------

KernelLaunchValidator::KernelLaunchValidator(int device_id, const SieveConstants& sc)
    : sc_(sc)
{
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, device_id);

    dev_.totalGlobalMem           = prop.totalGlobalMem;
    dev_.maxSharedMemPerBlock     = prop.sharedMemPerBlockOptin - 1024;
    dev_.sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
    dev_.maxThreadsPerBlock       = prop.maxThreadsPerBlock;
    dev_.maxThreadsPerMP          = prop.maxThreadsPerMultiProcessor;
    dev_.multiProcessorCount      = prop.multiProcessorCount;
    dev_.maxGridDimX              = prop.maxGridSize[0];
    dev_.maxBlockDimX             = prop.maxThreadsDim[0];
}

// ---------------------------------------------------------------------------
// isValid — ordered from cheapest to most expensive
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::isValid(const Params8& p) const {
    return checkPow2(p)
        && checkArithmeticConstraints(p)
        && checkSharedMem(p)
        && checkGlobalMem(p)
        && checkDeviceLimits(p)
        && checkNonZeroDerived(p);
}

// ---------------------------------------------------------------------------
// diagnose — re-run checks individually, return first failure reason
// ---------------------------------------------------------------------------

std::string KernelLaunchValidator::diagnose(const Params8& p) const {
    if (!checkPow2(p)) {
        for (int i = 0; i < 8; ++i) {
            uint32_t x = p[i];
            if (x == 0 || (x & (x - 1)) != 0)
                return "param[" + std::to_string(i) + "]=" + std::to_string(x) + " is not a power of 2";
        }
    }
    if (!checkArithmeticConstraints(p)) {
        uint32_t subCubeSize  = p[P_SUB_CUBE_SIZE];
        uint32_t numIntervals = p[P_NUM_INTERVALS];
        uint32_t polyBlockSize = p[P_POLY_BLOCK_SIZE];
        uint32_t blocksPerCyc = p[P_BLOCKS_PER_CYC];
        uint32_t metaGridDim  = p[P_META_GRID_DIM];
        if (subCubeSize > (1u << (sc_.shc_dim - 1)))
            return "subCubeSize=" + std::to_string(subCubeSize)
                 + " > 2^(shc_dim-1)=" + std::to_string(1u << (sc_.shc_dim - 1));
        if (blocksPerCyc > numIntervals)
            return "blocksPerCycle=" + std::to_string(blocksPerCyc)
                 + " > numIntervals=" + std::to_string(numIntervals);
        if (metaGridDim * polyBlockSize > subCubeSize)
            return "metaGridDim*polyBlockSize=" + std::to_string(metaGridDim * polyBlockSize)
                 + " > subCubeSize=" + std::to_string(subCubeSize);
        if (p[P_SAS_GRID_DIM] > subCubeSize)
            return "sasGridDim=" + std::to_string(p[P_SAS_GRID_DIM])
                 + " > subCubeSize=" + std::to_string(subCubeSize)
                 + " (polysPerSieveCall/gridDim.x would be 0)";
        uint32_t divisor = metaGridDim * polyBlockSize;
        uint32_t num_polyBlocksPerTB = subCubeSize / divisor;
        if (num_polyBlocksPerTB == 0 || (num_polyBlocksPerTB & (num_polyBlocksPerTB - 1)) != 0)
            return "num_polyBlocksPerTB=" + std::to_string(num_polyBlocksPerTB) + " is not a power of 2";
        return "arithmetic decomposition mismatch";
    }
    if (!checkSharedMem(p)) {
        size_t sharedMem_meta = static_cast<size_t>(p[P_BLOCKS_PER_CYC] * p[P_POLY_BLOCK_SIZE]) * sizeof(int);
        size_t sharedMem_sas  = static_cast<size_t>(sc_.sievingBlockSize) * sizeof(uint8_t)
                              + 3u * sc_.bigPrimeStartIndex * sizeof(int);
        if (sharedMem_meta > dev_.maxSharedMemPerBlock)
            return "shared memory overflow (meta): " + std::to_string(sharedMem_meta)
                 + " > " + std::to_string(dev_.maxSharedMemPerBlock);
        return "shared memory overflow (sas): " + std::to_string(sharedMem_sas)
             + " > " + std::to_string(dev_.maxSharedMemPerBlock);
    }
    if (!checkGlobalMem(p)) {
        // Routed through the single source-of-truth memory model (sieve_memory_model.h);
        // byte-identical message to the prior hand-rolled formula.
        uint64_t mem = mpqs::sieve::bucketEntriesBytes(
            p[P_SUB_CUBE_SIZE], p[P_NUM_INTERVALS], sc_.globalBucketSize);
        uint64_t budget = mpqs::sieve::sieveBucketBudget(
            dev_.totalGlobalMem, 0,
            mpqs::sieve::kSieveBudgetNum, mpqs::sieve::kSieveBudgetDen);
        return "global memory overflow: " + std::to_string(mem)
             + " > " + std::to_string(budget);
    }
    if (!checkDeviceLimits(p)) {
        if (p[P_META_BLOCK_DIM] > dev_.maxThreadsPerBlock)
            return "metaBlockDim=" + std::to_string(p[P_META_BLOCK_DIM])
                 + " > maxThreadsPerBlock=" + std::to_string(dev_.maxThreadsPerBlock);
        if (p[P_SAS_BLOCK_DIM] > dev_.maxThreadsPerBlock)
            return "sasBlockDim=" + std::to_string(p[P_SAS_BLOCK_DIM])
                 + " > maxThreadsPerBlock=" + std::to_string(dev_.maxThreadsPerBlock);
        return "grid dimension exceeds device limit";
    }
    if (!checkNonZeroDerived(p)) {
        uint32_t batches = (2 * sc_.M) / (sc_.sievingBlockSize * p[P_NUM_INTERVALS]);
        if (batches == 0)
            return "num_sievingBlockBatches=0 (M=" + std::to_string(sc_.M)
                 + ", sievingBlockSize=" + std::to_string(sc_.sievingBlockSize)
                 + ", numIntervals=" + std::to_string(p[P_NUM_INTERVALS]) + ")";
        uint32_t maxSC = std::min(32768u, (1u << sc_.shc_dim) / 2);
        uint32_t nsc = maxSC / p[P_SUB_CUBE_SIZE];
        if (nsc == 0)
            return "num_subCubes=0 (maxSubCube=" + std::to_string(maxSC)
                 + ", subCubeSize=" + std::to_string(p[P_SUB_CUBE_SIZE]) + ")";
        return "zero-valued derived quantity";
    }
    return "";  // all checks pass
}

// ---------------------------------------------------------------------------
// Step 1: All 8 params must be powers of two
// Mirrors POW2_CHECK on lines 864-872 (the 8 user-facing params)
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkPow2(const Params8& p) const {
    for (uint32_t x : p) {
        if (x == 0 || (x & (x - 1)) != 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Step 2: Arithmetic constraints (mirrors validateConfigs lines 874-885)
//
// Derived quantities:
//   num_polyBlocksPerTB = subCubeSize / (metaGridDim * polyBlockSize)
//   num_metaSieveCycles = numIntervals / blocksPerCycle
//   num_activeBuckets   = blocksPerCycle * polyBlockSize  (definitional)
//
// Checks:
//   LEQ:   subCubeSize <= (1 << (shc_dim - 1))                     [line 884]
//   LEQ:   blocksPerCycle <= numIntervals                           [line 885]
//   LEQ:   metaGridDim <= subCubeSize                               [line 882]
//   LEQ:   metaGridDim * polyBlockSize <= subCubeSize               [line 883]
//   POW2:  num_polyBlocksPerTB must be power-of-two                 [line 866]
//   EQUAL: num_polyBlocksPerTB * polyBlockSize * metaGridDim == subCubeSize [line 875]
//   EQUAL: num_metaSieveCycles * blocksPerCycle == numIntervals     [line 879]
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkArithmeticConstraints(const Params8& p) const {
    uint32_t subCubeSize  = p[P_SUB_CUBE_SIZE];
    uint32_t numIntervals = p[P_NUM_INTERVALS];
    uint32_t polyBlockSize = p[P_POLY_BLOCK_SIZE];
    uint32_t blocksPerCyc = p[P_BLOCKS_PER_CYC];
    uint32_t metaGridDim  = p[P_META_GRID_DIM];

    // LEQ checks
    if (subCubeSize > (1u << (sc_.shc_dim - 1))) return false;
    if (blocksPerCyc > numIntervals) return false;
    if (metaGridDim > subCubeSize) return false;
    if (metaGridDim * polyBlockSize > subCubeSize) return false;
    // sasGridDim must not exceed subCubeSize (polysPerSieveCall / gridDim.x must be > 0)
    if (p[P_SAS_GRID_DIM] > subCubeSize) return false;

    // Derived: num_polyBlocksPerThreadBlock = subCubeSize / (metaGridDim * polyBlockSize)
    uint32_t divisor = metaGridDim * polyBlockSize;
    uint32_t num_polyBlocksPerTB = subCubeSize / divisor;

    // Must be power-of-two (mirrors POW2_CHECK at line 866)
    if (num_polyBlocksPerTB == 0 || (num_polyBlocksPerTB & (num_polyBlocksPerTB - 1)) != 0)
        return false;

    // EQUAL: num_polyBlocksPerTB * polyBlockSize * metaGridDim == subCubeSize (line 875)
    if (num_polyBlocksPerTB * polyBlockSize * metaGridDim != subCubeSize) return false;

    // Derived: num_metaSieveCycles = numIntervals / blocksPerCycle
    uint32_t num_metaSieveCycles = numIntervals / blocksPerCyc;

    // EQUAL: num_metaSieveCycles * blocksPerCycle == numIntervals (line 879)
    if (num_metaSieveCycles * blocksPerCyc != numIntervals) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Step 3: Shared memory constraints (mirrors lines 889-890)
//
//   sharedMem_meta = blocksPerCycle * polyBlockSize * sizeof(int)
//   sharedMem_sas  = sievingBlockSize * sizeof(uint8_t) + 3 * bigPrimeStartIndex * sizeof(int)
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkSharedMem(const Params8& p) const {
    size_t sharedMem_meta = static_cast<size_t>(p[P_BLOCKS_PER_CYC] * p[P_POLY_BLOCK_SIZE])
                          * sizeof(int);
    size_t sharedMem_sas  = static_cast<size_t>(sc_.sievingBlockSize) * sizeof(uint8_t)
                          + 3u * sc_.bigPrimeStartIndex * sizeof(int);
    return sharedMem_meta <= dev_.maxSharedMemPerBlock
        && sharedMem_sas  <= dev_.maxSharedMemPerBlock;
}

// ---------------------------------------------------------------------------
// Step 4: Global memory constraint (mirrors line 888)
//
//   subCubeSize * numIntervals * globalBucketSize * sizeof(uint64_t) <= 3/4 * totalGlobalMem
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkGlobalMem(const Params8& p) const {
    // Routed through the single source-of-truth memory model (sieve_memory_model.h).
    // bucketEntriesBytes() == the prior hand-rolled dev_globalBucketEntries term on
    // this candidate's geometry (subCubeSize=p[0] polys, numIntervals=p[1] blocks); the
    // budget sieveBucketBudget(totalGlobalMem,0,4,5) == (4*totalGlobalMem)/5 == 0.80*VRAM
    // (S2; the operative kSieveBudget flipped from 3/4 to 4/5). This is the bucket-only
    // gate; the autotune total-footprint OOM guard adds persistent+scratch+pp/LP+reserve.
    uint64_t mem = mpqs::sieve::bucketEntriesBytes(
        p[P_SUB_CUBE_SIZE], p[P_NUM_INTERVALS], sc_.globalBucketSize);
    return mem <= mpqs::sieve::sieveBucketBudget(
        dev_.totalGlobalMem, 0,
        mpqs::sieve::kSieveBudgetNum, mpqs::sieve::kSieveBudgetDen);
}

// ---------------------------------------------------------------------------
// Total-footprint OOM guard (S2). Additive to checkGlobalMem; never looser.
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::fitsTotalFootprint(const Params8& p,
                                               uint64_t fb_size,
                                               uint64_t free_vram,
                                               uint64_t non_sieve_bytes,
                                               uint64_t* est_total_out) const {
    // Candidate geometry mirrors loadPartialCustomConfig: num_polys=p[0],
    // num_sievingBlocks=p[1], num_threadBlocks=p[6] (sasGridDim), maxRel=64.
    constexpr uint32_t kMaxRelationsPerBlock = 64;
    mpqs::sieve::SieveDeviceFootprint fp = mpqs::sieve::estimateSieveFootprint(
        p[P_SUB_CUBE_SIZE], p[P_NUM_INTERVALS],
        sc_.globalBucketSize, sc_.sievingBlockSize,
        p[P_SAS_GRID_DIM], kMaxRelationsPerBlock,
        fb_size, sc_.shc_dim);
    uint64_t total = fp.total() + non_sieve_bytes;
    if (est_total_out) *est_total_out = total;
    // Operative budget: kSieveBudget fraction (0.80) of free VRAM, integer-exact.
    uint64_t budget = mpqs::sieve::sieveBucketBudget(
        free_vram, 0, mpqs::sieve::kSieveBudgetNum, mpqs::sieve::kSieveBudgetDen);
    return total <= budget;
}

// ---------------------------------------------------------------------------
// Step 5: Hardware launch dimension limits
//
// Not explicitly in validateConfigs() but essential safety — the CUDA runtime
// would reject launches exceeding these limits.
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkDeviceLimits(const Params8& p) const {
    return p[P_META_BLOCK_DIM] <= dev_.maxThreadsPerBlock
        && p[P_SAS_BLOCK_DIM]  <= dev_.maxThreadsPerBlock
        && p[P_META_GRID_DIM]  <= static_cast<uint32_t>(dev_.maxGridDimX)
        && p[P_SAS_GRID_DIM]   <= static_cast<uint32_t>(dev_.maxGridDimX);
}

// ---------------------------------------------------------------------------
// Step 6: Non-zero derived quantities (mirrors lines 892-895)
//
//   num_sievingBlockBatches = (2*M) / (sievingBlockSize * numIntervals) > 0
//   num_subCubes = min(32768, (1 << shc_dim) / 2) / subCubeSize > 0
//   num_polyBlocksPerTB = subCubeSize / (metaGridDim * polyBlockSize) > 0
//   num_metaSieveCycles = numIntervals / blocksPerCycle > 0
// ---------------------------------------------------------------------------

bool KernelLaunchValidator::checkNonZeroDerived(const Params8& p) const {
    // Mirror loadPartialCustomConfig()'s num_sievingBlockBatches clamp: the sieve repairs a
    // floored-to-0 batch count to 1 (one batch over-covers [-M,M) when sievingBlockSize*
    // numIntervals > 2*M), so a config that would otherwise yield 0 batches is in fact runnable.
    // Without this mirror the preflight would reject those configs, which (a) stalls the full-
    // autotune M-sweep — every small-M probe is skipped with "preflight failed —
    // num_sievingBlockBatches=0" in estimateRuntime() — and (b) throws at the real SieveStage
    // preflight. For every M where 2*M >= sievingBlockSize*numIntervals (all M used by
    // --autotune_stage1 / no-autotune), the raw quotient is already >= 1 and std::max is the
    // identity, so those paths are unchanged.
    uint32_t num_sievingBlockBatches = std::max(1u,
        (2 * sc_.M) / (sc_.sievingBlockSize * p[P_NUM_INTERVALS]));
    uint32_t maxSubCube = std::min(32768u, (1u << sc_.shc_dim) / 2);
    uint32_t num_subCubes = maxSubCube / p[P_SUB_CUBE_SIZE];
    uint32_t num_polyBlocksPerTB = p[P_SUB_CUBE_SIZE] / (p[P_META_GRID_DIM] * p[P_POLY_BLOCK_SIZE]);
    uint32_t num_metaSieveCycles = p[P_NUM_INTERVALS] / p[P_BLOCKS_PER_CYC];

    return num_sievingBlockBatches > 0
        && num_subCubes > 0
        && num_polyBlocksPerTB > 0
        && num_metaSieveCycles > 0;
}

// ---------------------------------------------------------------------------
// Enumerate all valid 8-tuples from predefined value arrays
// ---------------------------------------------------------------------------

std::vector<Params8> enumerateValidConfigs(const KernelLaunchValidator& v) {
    std::vector<Params8> out;
    out.reserve(2048);

    for (uint32_t sub : {128u, 256u, 512u, 1024u})
      for (uint32_t ni : {1u, 2u, 4u, 8u, 16u, 32u})
        for (uint32_t pb : {1u, 2u, 4u, 8u, 16u, 32u})
          for (uint32_t bpc : {1u, 2u, 4u, 8u, 16u, 32u})
            for (uint32_t mG : {32u, 64u, 128u, 256u})
              for (uint32_t mB : {256u, 512u, 1024u})
                for (uint32_t sG : {32u, 128u, 256u, 512u})
                  for (uint32_t sB : {256u, 512u, 1024u}) {
                      Params8 p = {sub, ni, pb, bpc, mG, mB, sG, sB};
                      if (v.isValid(p)) out.push_back(p);
                  }

    return out;
}

// ---------------------------------------------------------------------------
// Build SieveConstants from factoringData dimensions + device properties
// ---------------------------------------------------------------------------

SieveConstants buildSieveConstants(uint32_t shc_dim, uint32_t M,
                                   size_t maxSharedMemPerBlock) {
    // Largest power of 2 <= x
    auto pow2leq = [](uint32_t x) -> uint32_t {
        return (x < 1) ? 0 : 1u << (31 - std::countl_zero(x));
    };

    SieveConstants sc;
    sc.shc_dim            = shc_dim;
    sc.M                  = M;
    sc.sievingBlockSize   = std::min(M, pow2leq(3 * static_cast<uint32_t>(maxSharedMemPerBlock) / 4));
    sc.globalBucketSize   = sc.sievingBlockSize / 2;
    sc.bigPrimeStartIndex = sc.sievingBlockSize / 32;
    return sc;
}

// ---------------------------------------------------------------------------
// Preflight check: build validator from raw dimensions + device, return result
// ---------------------------------------------------------------------------

PreflightResult preflightKernelLaunch(
    const Params8& params,
    uint32_t shc_dim,
    uint32_t M,
    int device_id) {

    // 1. Query device properties and build SieveConstants
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    cudaGetDeviceProperties(&prop, device_id);

    SieveConstants sc = buildSieveConstants(shc_dim, M, prop.sharedMemPerBlock);

    // 2. Validate
    KernelLaunchValidator validator(device_id, sc);
    if (validator.isValid(params)) {
        return {true, ""};
    }

    // 3. Diagnose failure
    return {false, validator.diagnose(params)};
}

PreflightResult preflightKernelLaunch(
    mpqs::MPQSConfig& config,
    uint32_t shc_dim,
    uint32_t M) {

    if (!config.useParams) {
        return {true, ""};  // Standard config path; no custom params to validate
    }

    // LP-aware sasGridDim auto-correction: when LP is active, ensure sasGridDim
    // is large enough that per-block candidate buffers (maxRelationsPerBlock=64)
    // don't overflow. Compute minimum from sieve geometry.
    if (config.lp1_bound > 0) {
        // total_polys = subCubeSize, total_sieve_blocks = numIntervals
        // Each sas thread block handles (total_polys * total_sieve_blocks / sasGridDim) work.
        // maxRelationsPerBlock = 64 (hardcoded in kernel).
        constexpr uint32_t maxRelationsPerBlock = 64;
        uint32_t total_polys        = config.params[P_SUB_CUBE_SIZE];
        uint32_t total_sieve_blocks = config.params[P_NUM_INTERVALS];
        uint32_t work_units = total_polys * total_sieve_blocks;
        uint32_t min_sas = (work_units + maxRelationsPerBlock - 1) / maxRelationsPerBlock;

        // Round up to next power of 2 (params must be powers of 2)
        if (min_sas > 0) {
            min_sas--;
            min_sas |= min_sas >> 1;
            min_sas |= min_sas >> 2;
            min_sas |= min_sas >> 4;
            min_sas |= min_sas >> 8;
            min_sas |= min_sas >> 16;
            min_sas++;
        }

        if (config.params[P_SAS_GRID_DIM] < min_sas) {
            LOG(LOG_WARNING) << "[Preflight] LP active: auto-correcting sasGridDim from "
                             << config.params[P_SAS_GRID_DIM] << " to " << min_sas
                             << " (candidate buffer overflow prevention)";
            config.params[P_SAS_GRID_DIM] = min_sas;
        }
    }

    // sasGridDim must not exceed subCubeSize (polysPerSieveCall / gridDim.x must be > 0)
    if (config.params[P_SAS_GRID_DIM] > config.params[P_SUB_CUBE_SIZE]) {
        LOG(LOG_WARNING) << "[Preflight] sasGridDim=" << config.params[P_SAS_GRID_DIM]
                         << " > subCubeSize=" << config.params[P_SUB_CUBE_SIZE]
                         << " → zero-work sieve iterations; clamping sasGridDim to subCubeSize";
        config.params[P_SAS_GRID_DIM] = config.params[P_SUB_CUBE_SIZE];
    }

    // Safety cap matching __launch_bounds__(1024) on sieveAndScanKernel.
    // With __launch_bounds__, the compiler guarantees ≤64 regs/thread (spilling if needed),
    // so 1024 threads is now safe. This cap is a belt-and-suspenders defense.
    constexpr uint32_t LEGACY_SAS_MAX_THREADS = 1024;
    if (config.sieve_batch_size == 0 && config.params[P_SAS_BLOCK_DIM] > LEGACY_SAS_MAX_THREADS) {
        LOG(LOG_WARNING) << "[Preflight] Legacy sieveAndScanKernel: capping sasBlockDim from "
                         << config.params[P_SAS_BLOCK_DIM] << " to " << LEGACY_SAS_MAX_THREADS
                         << " (__launch_bounds__ safety net)";
        config.params[P_SAS_BLOCK_DIM] = LEGACY_SAS_MAX_THREADS;
    }

    Params8 params;
    for (int i = 0; i < 8; ++i)
        params[i] = config.params[i];

    return preflightKernelLaunch(params, shc_dim, M, config.device_id);
}

} // namespace mpqs::autotune
