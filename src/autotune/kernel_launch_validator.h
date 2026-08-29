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
    uint32_t sievingBlockSize;   ///< narrow: pow2leq(3*maxShared/4); wide: pow2leq(3*maxShared*32/(4*76))
    uint32_t globalBucketSize;   ///< sievingBlockSize / 2
    uint32_t bigPrimeStartIndex; ///< sievingBlockSize / 32
    uint32_t accumulatorBytes = 1; ///< sieve-accumulator width in bytes: 1 = uint8 (narrow), 2 = uint16 (wide)
    /// v1.0.6: when true, {subCubeSize, metaGridDim, sasGridDim} and the derived
    /// num_polyBlocksPerThreadBlock are NOT required to be powers of two — the SM-aligned
    /// narrow BATCH geometry. Set ONLY for the narrow batch path (batch_size > 0 && !wide);
    /// the default false reproduces the legacy/wide rule set byte-for-byte for every caller.
    /// The normative admissible set is the predicate enforced by the validators below.
    bool allow_nonpow2_geometry = false;
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

    /// Explicit-limits constructor — identical checks, no CUDA call. Lets host unit tests
    /// exercise the admissible-set policy deterministically on any machine (v1.0.6).
    KernelLaunchValidator(const DeviceLimits& dev, const SieveConstants& sc)
        : dev_(dev), sc_(sc) {}

    /// Returns true iff the 8-tuple passes ALL feasibility checks.
    /// Silent on failure — designed for bulk filtering (called thousands of times).
    bool isValid(const Params8& p) const;

    /// Re-runs each check individually and returns the first failing check's
    /// name and the parameter values that caused it. Returns empty string if valid.
    /// Designed for diagnostics, not bulk filtering.
    std::string diagnose(const Params8& p) const;

    /// Autotune OOM guard: true iff the candidate's COMPLETE device footprint
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

    /// True iff parameter `idx` must be a power of two under this validator's regime.
    /// Always true unless sc_.allow_nonpow2_geometry (narrow batch), which frees exactly
    /// {P_SUB_CUBE_SIZE, P_META_GRID_DIM, P_SAS_GRID_DIM}. v1.0.6.
    bool pow2Required(uint32_t idx) const;

    bool checkPow2(const Params8& p) const;
    bool checkArithmeticConstraints(const Params8& p) const;
    bool checkSharedMem(const Params8& p) const;
    bool checkGlobalMem(const Params8& p) const;
    bool checkDeviceLimits(const Params8& p) const;
    bool checkNonZeroDerived(const Params8& p) const;
};

/// Smallest admissible sasGridDim >= min_sas for a given num_polysPerSieveCall `np`.
///
/// The batch GATHER kernel decomposes its polynomial index as `polyIdPrefix | gray(poly)`
/// with `polyIdPrefix = blockIdx.x * chunk` and `chunk = np / sasGridDim`. The OR is a valid
/// addition iff `chunk` is a power of two, and coverage is exact iff `sasGridDim` divides
/// `np` — i.e. the admissible grids are exactly `{ np / 2^j : 2^j | np }`. Returns the
/// smallest such value that is >= min_sas, or `np` (the largest admissible value, chunk 1)
/// when min_sas exceeds every admissible grid.
///
/// For a power-of-two `np` the admissible set IS the power-of-two ladder `{1,2,...,np}`, so
/// this reproduces the previous "round min_sas up to the next power of two, then clamp to np"
/// behaviour exactly. Pure arithmetic; `np == 0` returns 0.
uint32_t admissibleSasGridDim(uint32_t np, uint32_t min_sas);

/// Enumerate all valid parameter combinations by iterating over predefined
/// value arrays and filtering through the validator.
/// Completes in <1 ms (~50k pure-arithmetic iterations).
std::vector<Params8> enumerateValidConfigs(const KernelLaunchValidator& v);

/// Preflight check from raw Params8 + factoringData dimensions + device.
/// Builds SieveConstants, creates validator, returns structured result.
/// use_wide selects the uint16 (wide) sieve geometry (SB, accumulator byte width);
/// default false = the uint8 (narrow) behaviour, byte-identical for all callers.
PreflightResult preflightKernelLaunch(
    const Params8& params,
    uint32_t shc_dim,
    uint32_t M,
    int device_id,
    bool use_wide = false,
    bool use_u8sat = false,    // saturating-uint8 wide accumulator geometry
    bool allow_nonpow2_geometry = false);  // v1.0.6: SM-aligned narrow BATCH geometry

/// Convenience overload: extracts Params8 from MPQSConfig::params[8].
/// Short-circuits with {true, ""} when config.useParams == false.
/// When LP is active and sasGridDim is too low, auto-corrects config.params[6]
/// to the minimum safe value (next power of 2) and logs a warning.
PreflightResult preflightKernelLaunch(
    mpqs::MPQSConfig& config,
    uint32_t shc_dim,
    uint32_t M,
    bool allow_nonpow2_geometry = false);

/// Build SieveConstants from factoringData dimensions + device properties.
///   shc_dim            = f_data.a_factors.size()
///   M                  = f_data.M
///   sievingBlockSize   = min(M, pow2leq(3*maxSharedMemPerBlock/4))            [narrow, uint8]
///                        min(M, pow2leq(3*maxSharedMemPerBlock*32/(4*76)))    [wide,   uint16]
///   globalBucketSize   = sievingBlockSize / 2
///   bigPrimeStartIndex = sievingBlockSize / 32
///   accumulatorBytes   = use_wide ? (use_u8sat ? 1 : 2) : 1
/// use_wide mirrors DeviceSievingController::loadPartialCustomConfig's wide geometry so the
/// standalone validator/preflight rank the config as it will actually run on the wide kernel.
/// use_u8sat selects the saturating-uint8 wide accumulator (SB restored, den=44).
/// Defaults false = byte-identical to the narrow build for every existing caller.
SieveConstants buildSieveConstants(uint32_t shc_dim, uint32_t M,
                                   size_t maxSharedMemPerBlock,
                                   bool use_wide = false, bool use_u8sat = false,
                                   bool allow_nonpow2_geometry = false);

} // namespace mpqs::autotune
