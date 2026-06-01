// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sieving_data_structs.h"
#include "uint512.cuh"
#include <vector>

namespace mpqs {
namespace sieve {

// ============================================================================
// Kernel Declarations
// ============================================================================

__global__ void globalMetaSieveKernel(devicePointers dev_pointers, fixedSievingParams fs_params, dynamicSievingParams ds_params, generalSievingConfig gs_conf, globalMetaSieveConfig gms_conf);

__global__ void sieveAndScanKernel(devicePointers dev_pointers, fixedSievingParams fs_params, dynamicSievingParams ds_params, generalSievingConfig gs_conf, sieveAndScanConfig ss_conf);
__global__ void sieveAndScanBatchKernel(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    const mpqs::uint512* __restrict__ batch_a_array,
    const mpqs::uint512* __restrict__ batch_B_flat,
    uint32_t step_index,
    int32_t sieveIntervalStart, // Previously ds_params.startIndex (usually -M)
    uint32_t* __restrict__ dev_blockRelationCounts, // Output for compaction
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf); // We want gridDim.x blocks with size of blockDim.x

__global__ void initPrimeDataKernel(devicePointers dev_pointers, generalSievingConfig gs_conf, fixedSievingParams fs_params, dynamicSievingParams ds_params);

// Updated: a_factors is now uint32_t*
__global__ void markInactivePrimesKernel(int shc_dim, uint32_t* a_factors, primeDataSIQS* primeData);

// ============================================================================
// Host Helper Declarations
// ============================================================================

void loadSievingData(std::vector<uint32_t>& factorBase,
    std::vector<uint32_t>& rootN,
    std::vector<uint32_t>& a_factors,
    uint32_t shc_dim,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    devicePointers& dev_pointers);

void loadSievingDataParamTest(std::vector<uint32_t>& factorBase,
    std::vector<uint32_t>& rootN,
    std::vector<uint32_t>& a_factors,
    uint32_t shc_dim,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    devicePointers& dev_pointers);
// Updated: B_values vector uses mpqs::uint512
void updateSievingData(devicePointers& dev_pointers, fixedSievingParams fs_params, std::vector<uint32_t>& a_factors, std::vector<mpqs::uint512> B_values);

void warmup();
void getDeviceInfo(gpuInfo& gInfo, int device);

// ============================================================================
// Device Inline Functions (Math)
// ============================================================================

/**
 * @brief Modular Addition: (a + b) % modulus
 * Safe for unsigned inputs where a, b < modulus.
 * Replaces sign-bit logic with conditional subtraction.
 */
__device__ __forceinline__
uint32_t modAdd(uint32_t a, uint32_t b, uint32_t modulus) {
    uint32_t tmp = a + b;
    return (tmp >= modulus) ? (tmp - modulus) : tmp;
}

/**
 * @brief Modular Subtraction: (a - b) % modulus
 * Safe for unsigned inputs where a, b < modulus.
 * Prevents underflow by adding modulus if b > a.
 */
__device__ __forceinline__
uint32_t modSub(uint32_t a, uint32_t b, uint32_t modulus) {
    return (a >= b) ? (a - b) : (modulus - (b - a));
}

/**
 * @brief Shifted Modular Subtraction.
 * Returns a value in [1, modulus] representing (a - b).
 * Used for calculating positive offsets into sieving blocks.
 *
 * Logic:
 * If a > b: result is a - b.
 * If a <= b: result is a - b + modulus.
 */
__device__ __forceinline__
uint32_t modSub_shifted(uint32_t a, uint32_t b, uint32_t modulus) {
    return (a > b) ? (a - b) : (a + modulus - b);
}

/**
 * @brief Modular Sum with Signed Delta.
 * Computes (a + b) % modulus where 'b' can be negative.
 * Essential for Hypercube traversal where roots update by +/- B_val.
 *
 * @param a Current root (unsigned residue [0, m-1]).
 * @param b Delta (signed value, |b| < m).
 * @param modulus The prime p.
 */
__device__ __forceinline__
uint32_t modSum(uint32_t a, int32_t b, uint32_t modulus) {
    // Cast to signed 64-bit to safely handle the addition/subtraction
    int64_t tmp = (int64_t)a + b;

    // Normalize result to [0, modulus-1]
    // Since |b| < modulus, tmp is in (-modulus, 2*modulus).
    // We only need one conditional check for each bound.
    if (tmp < 0) tmp += modulus;
    if (tmp >= modulus) tmp -= modulus;

    return (uint32_t)tmp;
}

__device__ __forceinline__
int log2(uint32_t a) {
    return 31 - mpqs::clz32(a);
    // return 31 - __clz(a);
}

// Return smallest y >= bound with y ≡ x (mod p), assuming p > 0.
__device__ __forceinline__ int align_up_to_hit(int x, int bound, int p) {
    int64_t xx = (int64_t)x;
    int64_t bb = (int64_t)bound;
    int64_t pp = (int64_t)p;
    if (xx >= bb) return x;
    int64_t k = (bb - xx + pp - 1) / pp; // ceil div pos, p > 0 necessary
    return (int)(xx + k * pp);
}

// ============================================================================
// Polynomial & Root Helpers (Updated signatures)
// ============================================================================

/**
 * @brief Reconstructs roots for a specific Poly ID from the hypercube.
 * Result roots are in [0, p-1].
 */
__device__ __forceinline__
void rootsFromPolyId(uint32_t id, int shc_dim, primeDataSIQS& primeData, uint32_t& result1, uint32_t& result2);

/**
 * @brief Reconstructs coefficient 'b' from Poly ID.
 * Updated to use mpqs::uint512.
 */
__device__ __forceinline__
void bFromPolyId(uint32_t id, int shc_dim, mpqs::uint512* B_values, mpqs::uint512& result);

/**
 * @brief Updates roots when moving between adjacent Gray codes.
 * Uses modSum to handle signed updates (+/- B_val) on unsigned roots.
 */
__device__ __forceinline__
void advanceRoots(uint32_t id1, uint32_t id2, primeDataSIQS& primeData, uint32_t& root1, uint32_t& root2);

/**
 * @brief Updates coefficient 'b' when moving between adjacent Gray codes.
 * Updated to use mpqs::uint512.
 */
__device__ __forceinline__
void advance_b(uint32_t id1, uint32_t id2, mpqs::uint512* B_values, mpqs::uint512& b);

__device__ __forceinline__
int atomicByteAdd(uint8_t* array, int index, uint8_t x);

/**
 * @brief Scans sieve block for candidates.
 * Updated: 'p_data' contains unsigned types, implementation must handle safe distance calc.
 */
__device__
int excludeNonRelations(
    uint8_t* __restrict__ blockEntries,
    int32_t* __restrict__ indexToCandidate,
    candidateRelation* __restrict__ candidates, // <--- Added
    const mpqs::uint512& b,                     // <--- Added
    uint32_t poly_id,                           // <--- Added
    uint32_t candidatesFound,
    int startOffset,
    int sievingBlockSize,
    int maxPerBlock, // <--- Added
    polyData& p_data
);

// ============================================================================
// Global Host Wrappers
// ============================================================================

void initPrimeData(
    devicePointers dev_pointers,
    initConfig init_conf,
    generalSievingConfig gs_conf,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    cudaStream_t stream
);

void globalMetaSieve(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    generalSievingConfig gs_conf,
    globalMetaSieveConfig gms_conf,
    cudaStream_t stream
);

void sieveAndScan(
    devicePointers dev_pointers,
    fixedSievingParams fs_params,
    dynamicSievingParams ds_params,
    generalSievingConfig gs_conf,
    sieveAndScanConfig ss_conf,
    cudaStream_t stream
);

// ----- BATCH SIEVING LAUNCHERS ----

// We want this launcher to be as lightweight as possible
void prepareSievingBatch(
    const devicePointers* dev_pointers_ptr,
    const std::vector<uint32_t>* next_factor_indices_ptr,
    const uint32_t shc_dim,
    const uint32_t batch_size,
    cudaStream_t stream,
    uint32_t* h_pinned_factor_indices = nullptr
);

/// Graph-capturable polynomial generation — no H2D copy, indices already on device.
void prepareSievingBatchFromStaged(
    const devicePointers* dev_pointers_ptr,
    const uint32_t* d_indices,
    mpqs::uint512* a_array_out,
    mpqs::uint512* B_flat_out,
    const uint32_t shc_dim,
    const uint32_t batch_size,
    cudaStream_t stream
);

// We want this launcher to be as lightweight as possible
void runSievingBatch(
    devicePointers* dev_pointers_ptr,
    initConfig* init_conf_ptr,
    generalSievingConfig* gs_conf_ptr,
    fixedSievingParams* fs_params_ptr,
    dynamicSievingParams* ds_params_ptr,
    globalMetaSieveConfig* gms_conf_ptr,
    sieveAndScanConfig* ss_conf_ptr,
    int num_steps,
    int start_batch_index,
    cudaStream_t stream
);

} // namespace sieve
} // namespace mpqs
