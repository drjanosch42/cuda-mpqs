// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include <string>
#include <vector>

// Forward-declare to avoid circular include (orchestrator.h -> autotune.h)
namespace mpqs { struct MPQSConfig; }

namespace mpqs::autotune {

/// Hardware limits queried once at construction from cudaDeviceProp.
struct DeviceLimits {
    size_t   totalGlobalMem;
    size_t   maxSharedMemPerBlock;        ///< cudaDeviceProp::sharedMemPerBlock
    size_t   sharedMemPerMultiprocessor;  ///< cudaDeviceProp::sharedMemPerMultiprocessor
    uint32_t maxThreadsPerBlock;          ///< cudaDeviceProp::maxThreadsPerBlock
    uint32_t maxThreadsPerMP;             ///< cudaDeviceProp::maxThreadsPerMultiProcessor
    uint32_t multiProcessorCount;
    int      maxGridDimX;                 ///< cudaDeviceProp::maxGridSize[0]
    int      maxBlockDimX;                ///< cudaDeviceProp::maxBlockDim[0]
};

/// Pre-computed sieve constants, fixed for a given (N, device) pair.
struct SieveConstants {
    uint32_t shc_dim;            ///< Hypercube dimension (f_data.a_factors.size())
    uint32_t M;                  ///< Sieve interval radius (f_data.M)
    uint32_t sievingBlockSize;   ///< pow2leq(3 * maxSharedMem / 4), typically 32768
    uint32_t globalBucketSize;   ///< sievingBlockSize / 2
    uint32_t bigPrimeStartIndex; ///< Cutoff index for small/large primes, hardcoded 1024
};

/// Ordered 8-parameter tuple for kernel launch configuration.
using Params8 = std::array<uint32_t, 8>;

/// Parameter index mapping (matches runParamTest / loadPartialCustomConfig order).
enum ParamIndex : uint32_t {
    P_SUB_CUBE_SIZE   = 0,  ///< subCubeSize     = gs_conf.num_polysPerSieveCall
    P_NUM_INTERVALS   = 1,  ///< numIntervals    = gs_conf.num_sievingBlocksPerSieveCall
    P_POLY_BLOCK_SIZE = 2,  ///< polyBlockSize   = gms_conf.polyBlockSize
    P_BLOCKS_PER_CYC  = 3,  ///< blocksPerCycle  = gms_conf.num_activeBlocksPerCycle
    P_META_GRID_DIM   = 4,  ///< metaGridDim     = gms_conf.num_threadBlocks
    P_META_BLOCK_DIM  = 5,  ///< metaBlockDim    = gms_conf.num_threadsPerBlock
    P_SAS_GRID_DIM    = 6,  ///< sasGridDim      = ss_conf.num_threadBlocks
    P_SAS_BLOCK_DIM   = 7   ///< sasBlockDim     = ss_conf.num_threadsPerBlock
};

/// Result of a kernel launch preflight check.
struct PreflightResult {
    bool feasible = false;
    std::string reason;   ///< Empty if feasible; diagnostic message if not
};

/// Standalone kernel launch parameter validator.
///
/// Mirrors all constraint checks from DeviceSievingController::validateConfigs()
/// using only the 8-parameter tuple and hardware properties. Pure arithmetic —
/// no GPU kernel launches, no CUDA stream operations. Only the constructor
/// calls cudaGetDeviceProperties().
class KernelLaunchValidator {
public:
    /// Queries cudaGetDeviceProperties for device_id, stores sc.
    KernelLaunchValidator(int device_id, const SieveConstants& sc);

    /// Returns true iff the 8-tuple passes ALL feasibility checks.
    /// Silent on failure — designed for bulk filtering (called thousands of times).
    bool isValid(const Params8& p) const;

    /// Re-runs each check individually and returns the first failing check's
    /// name and the parameter values that caused it. Returns empty string if valid.
    /// Designed for diagnostics, not bulk filtering.
    std::string diagnose(const Params8& p) const;

    /// Autotune OOM guard (S2): true iff the candidate's COMPLETE device footprint
    /// (sieve bucket + persistent + scratch via mpqs::sieve::estimateSieveFootprint,
    /// PLUS the caller-supplied non_sieve_bytes = postprocessing/LP + context reserve)
    /// fits the operative budget (kSieveBudget fraction, 0.80) of free_vram.
    ///
    /// This is ADDITIVE to (never looser than) the bucket-only checkGlobalMem/isValid
    /// gate: it can reject a candidate the bucket-only check accepts, never the reverse.
    /// The candidate geometry mirrors loadPartialCustomConfig (num_polys=p[0],
    /// num_sievingBlocks=p[1], num_threadBlocks=p[6]=sasGridDim, maxRelationsPerBlock=64)
    /// on this validator's SieveConstants (globalBucketSize/sievingBlockSize/shc_dim).
    ///
    /// @param p             candidate 8-tuple
    /// @param fb_size       factor base size (validator does not carry it)
    /// @param free_vram     cudaMemGetInfo free bytes (amortized once per Stage-1)
    /// @param non_sieve_bytes  postprocessing/LP footprint + CUDA-context reserve
    /// @param est_total_out optional: receives the candidate's full footprint estimate
    bool fitsTotalFootprint(const Params8& p,
                            uint64_t fb_size,
                            uint64_t free_vram,
                            uint64_t non_sieve_bytes,
                            uint64_t* est_total_out = nullptr) const;

    const DeviceLimits& getDeviceLimits() const { return dev_; }
    const SieveConstants& getSieveConstants() const { return sc_; }

private:
    DeviceLimits dev_;
    SieveConstants sc_;

    bool checkPow2(const Params8& p) const;
    bool checkArithmeticConstraints(const Params8& p) const;
    bool checkSharedMem(const Params8& p) const;
    bool checkGlobalMem(const Params8& p) const;
    bool checkDeviceLimits(const Params8& p) const;
    bool checkNonZeroDerived(const Params8& p) const;
};

/// Enumerate all valid parameter combinations by iterating over predefined
/// value arrays and filtering through the validator.
/// Completes in <1 ms (~50k pure-arithmetic iterations).
std::vector<Params8> enumerateValidConfigs(const KernelLaunchValidator& v);

/// Preflight check from raw Params8 + factoringData dimensions + device.
/// Builds SieveConstants, creates validator, returns structured result.
PreflightResult preflightKernelLaunch(
    const Params8& params,
    uint32_t shc_dim,
    uint32_t M,
    int device_id);

/// Convenience overload: extracts Params8 from MPQSConfig::params[8].
/// Short-circuits with {true, ""} when config.useParams == false.
/// When LP is active and sasGridDim is too low, auto-corrects config.params[6]
/// to the minimum safe value (next power of 2) and logs a warning.
PreflightResult preflightKernelLaunch(
    mpqs::MPQSConfig& config,
    uint32_t shc_dim,
    uint32_t M);

/// Build SieveConstants from factoringData dimensions + device properties.
///   shc_dim            = f_data.a_factors.size()
///   M                  = f_data.M
///   sievingBlockSize   = pow2leq(3 * maxSharedMemPerBlock / 4)
///   globalBucketSize   = sievingBlockSize / 2
///   bigPrimeStartIndex = 1024
SieveConstants buildSieveConstants(uint32_t shc_dim, uint32_t M,
                                   size_t maxSharedMemPerBlock);

} // namespace mpqs::autotune
