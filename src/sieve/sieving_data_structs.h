// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "uint512.cuh"

namespace mpqs {
namespace sieve {

// =============================================================================
// Primitive Structures (Factor Base & Polynomials)
// =============================================================================

/**
 * @brief Maximum hypercube dimension (number of factors in 'a').
 *
 * Must be >= fs_params.shc_dim for the target composite. RSA-140 requires
 * shc_dim=19; 32 provides headroom through RSA-155 and beyond.
 *
 * NOTE: The per-prime B_values are NO LONGER stored inside primeDataSIQS (which
 * the hot meta-sieve kernels stream once per factor-base prime each pass). They
 * live in a SEPARATE device array (devicePointers::dev_primeBValues), sized to
 * the actual run shc_dim (not MAX_SHC_DIM), column-major [k*fb_size + prime].
 * Hence MAX_SHC_DIM only bounds that separate array and per-thread caches; it no
 * longer inflates the hot-streamed struct, so it is bandwidth-neutral for the
 * sieve at every size (the bloat that regressed RSA-100 is eliminated).
 */
static constexpr int MAX_SHC_DIM = 32;

/**
 * @brief Per-Prime data structure for the SIQS Factor Base.
 *
 * Stores precomputed modular inverses and Gray code update constants
 * to facilitate fast polynomial switching on the GPU.
 */
struct primeDataSIQS {
    uint32_t p;             ///< The prime number p.
    uint32_t r;             ///< Root of N mod p: r^2 = N (mod p).
    uint32_t mod_inverse_a; ///< (a^-1) mod p. Used to update roots when 'a' changes.
    uint32_t inv_aN;        ///< (a^-1 * r) mod p.

    /**
     * @brief Inactive flag.
     * If != 0, this prime does not divide 'a' and participates in sieving.
     * If == 0, prime divides 'a' and is skipped for this batch.
     */
    uint32_t inactive;

    // NOTE: The per-prime Gray-code update values B_values[k] = (b_k * a^-1) mod p
    // were REMOVED from this struct and moved to a separate device array
    // (devicePointers::dev_primeBValues), column-major [k*fb_size + primeIndex],
    // sized to the actual run shc_dim. The hot meta-sieve kernels load this struct
    // per factor-base prime every pass; embedding B_values[MAX_SHC_DIM] here bloated
    // the per-prime stream stride (84->148 B at MAX_SHC_DIM=32) and made the sieve
    // bandwidth-bound on padding. Decoupling restores the struct to 20 B for ALL
    // sizes and reads only the live shc_dim B-values, coalesced. See rootsFromPolyId
    // / advanceRoots in kernel.cu.
};

// =============================================================================
// Output Structures (GPU -> Host / Post-Processing)
// =============================================================================

/**
 * @brief A lightweight candidate relation found by the GPU Siever.
 *
 * Instead of storing full 512-bit integers 'a' and 'b', we store the
 * polynomial index and the sieve offset. The Post-Processing kernel
 * reconstructs the actual values using the DeviceHypercubeContext.
 *
 * Relation: (ax + b)^2 - N = Q(x)
 */
struct candidateRelation {
    mpqs::uint512 b;       ///< We store 'b' explicitly to allow 'stateless' postprocessing
    uint32_t poly_id;      ///< Poly index containing Gray code index identifying the specific 'b' coefficient, for debugging purposes
    int32_t sieve_offset; ///< The 'x' offset in the sieve interval where a relation was found.
    uint32_t global_idx;   ///< Unique global index for debugging/tracking.
    uint32_t num_factors;  ///< Number of small factors found during trial division.
    uint32_t factors[32];  ///< Array of small prime indices/values dividing Q(x).
};

/**
 * @brief A compact representation of a relation candidate for GPU processing.
 *
 * @details
 * This structure sits in a dense array on the GPU. Unlike `candidateRelation`,
 * it is packed (no empty slots). It includes `hint_count` to facilitate
 * sorting via Thrust, which optimizes warp convergence during factorization.
 */
struct DenseCandidate {
    mpqs::uint512 a;
    mpqs::uint512 b;
    int32_t true_x;  // The actual integer x = startIndex + offset, where relation is Q(x)
    uint32_t num_factors;  // Number of total hint factors: factors of a + Siever output
    uint32_t factor_indices[48]; // The factors (odd primes) dividing a and found by the Siever
    // int8_t sign_of_Q;   // sign will be computed at a later stage by evaluation Q(x)
    // uint32_t val_2_exp; // exponent of 2 will be computed at a lager stage, by evaluating Q(x)
    uint32_t _padding[2];  // Add 8 bytes for total of 336 bytes (divisible by 16) for alignment.

    /**
     * @brief Comparator for Thrust sorting.
     * Sorts descending by hint count.
     * Motivation: Candidates with many hints reduce faster, keeping heavy-work threads together.
     * Lexicographic ordering:
     * We try to maximize memory coalescence and simplify comparison (doublette finding)
     */
    __host__ __device__ bool operator<(const DenseCandidate& other) const {
        // In the case of equality we compare the largest known factor.
        // This allows for efficient duplicate detection of relations during preprocessing.
        // For different relations, it is possible, but unlikely that the largest sieved prime divisors agree.
        // Therefore we may branch in this case and run a full comparison in case of agreement.
        return (num_factors > other.num_factors); // ||
        /*
	  (
	   (num_factors == other.num_factors) &&
	   (
	    (factor_indices[0] < other.factor_indices[0]) ||
	    (
	    (factor_indices[0] == other.factor_indices[0]) &&
	    (factor_indices[num_factors-1] > other.factor_indices[num_factors-1])
	    )
	   )
	  );
        */
    }
};

/**
 * @brief Fully reconstructed relation (Host-side or Intermediate).
 */
struct relation {
    uint32_t a_factors[16]; ///< Indices of primes composing 'a'.
    mpqs::uint512 axb;      ///< The value (ax + b).
    uint32_t factors[64];   ///< Full list of factors found.
    uint32_t num_factors;
};

// =============================================================================
// Device Contexts (Memory Layout)
// =============================================================================

/**
 * @brief Global immutable constants residing in GPU Global Memory.
 * These do not change throughout the entire factorization of N.
 */
struct DeviceConstants {
    mpqs::uint512* dev_N;          ///< Pointer to N (512-bit).
    uint32_t* dev_factorBase;      ///< Array of all primes in the factor base.
    uint32_t* dev_rootN;           ///< Array of modular square roots of N.
    uint32_t num_primes;           ///< Total size of factor base.
};

/**
 * @brief Context for the current Hypercube (Batch of Polynomials).
 *
 * A "Batch" is defined by a fixed coefficient 'a'.
 * The coefficient 'b' varies via Gray code enumeration.
 * This struct allows the GPU to reconstruct Q(x) without passing heavy data.
 */
struct DeviceHypercubeContext {
    mpqs::uint512 current_a;        ///< The fixed 'a' coefficient for this batch.
    uint32_t* dev_a_factors;        ///< Indices of primes that divide 'a'.
    mpqs::uint512* dev_b_components;///< Precomputed terms to construct 'b' via Gray code.
    primeDataSIQS* dev_primeData;   ///< Per-prime data tailored for this 'a' (roots, inverses).

    // Bucket Sorting / Sieving buffers
    uint64_t* dev_globalBucketEntries;
    uint32_t* dev_globalBucketCounts;
    uint32_t* dev_indexToCandidate;
    candidateRelation* dev_candidateRelations;
};

// Legacy pointer struct (should be phased out at some point,
// preference for DeviceConstants/DeviceHypercubeContext)
struct devicePointers {
    uint32_t* dev_a_factors = nullptr; // obsolete for batch sieving
    mpqs::uint512* dev_B_values = nullptr; // obsolete for batch sieving
    uint32_t* dev_factorBase = nullptr;
    primeDataSIQS* dev_primeData = nullptr;
    // Per-prime Gray-code B-update values, decoupled from primeDataSIQS to keep the
    // hot-streamed struct small. Column-major: dev_primeBValues[k*fb_size + prime]
    // for k in [0, shc_dim). Size: fb_size * shc_dim * sizeof(uint32_t).
    uint32_t* dev_primeBValues = nullptr;
    uint32_t* dev_rootN = nullptr;
    uint64_t* dev_globalBucketEntries = nullptr;
    uint32_t* dev_globalBucketCounts = nullptr;
    int32_t* dev_indexToCandidate = nullptr;
    candidateRelation* dev_candidateRelations = nullptr;
    // --- Batch Job Data (The "Blueprint") ---
    // Stores the 'a' coefficient for each step in the batch.
    // Size: batch_size * sizeof(mpqs::uint512)
    mpqs::uint512* dev_job_a_array = nullptr;
    // Stores the flattened B values for each step.
    // Size: batch_size * shc_dim * sizeof(mpqs::uint512)
    mpqs::uint512* dev_job_B_flat = nullptr;
    // Stores the indices into FactorBase for primes composing 'a' for each step.
    // Size: batch_size * shc_dim * sizeof(uint32_t)
    uint32_t* dev_job_factor_indices = nullptr;
    // --- Inter-Kernel Communication Buffers ---
    // Stores the number of relations found by each sieve block in the current step.
    // Enables the compacting kernel to skip empty blocks.
    // Size: num_sievingBlocksPerSieveCall * sizeof(uint32_t)
    uint32_t* dev_blockRelationCounts = nullptr;
    // --- Post-Processing Integration Pointers ---
    // These allow the Siever to write directly into the Post-Processor's input buffer.
    // (These will be copied from the PostProcessor instance during setup).
    void* dev_pp_accumulation_buffer = nullptr; // Pointer to DenseCandidate buffer
    uint32_t* dev_pp_counter = nullptr;         // Pointer to atomic fill counter
    uint32_t pp_max_capacity = 0;               // Value, not pointer (config)
};

// =============================================================================
// Configuration Structures
// =============================================================================

struct sievingInfo {
    uint32_t num_threadsPerBlock;
    uint32_t num_threadBlocks;
    uint32_t batch_size;             ///< Parameter for batch sieving

    uint32_t num_sievingCrates;      ///< Number of crates sieved in one kernel launch.
    uint32_t num_sievingBlocks;      ///< Number of sievingBlocks being sieved at once.
    uint32_t sievingBlockSize;
    uint32_t log2_sievingBlockSize;

    uint32_t polyBlockSize;
    uint32_t log2_polyBlockSize;
    uint32_t num_polyBlocksPerThreadBlock;
    uint32_t num_polyBlocks;
    uint32_t log2_num_polyBlocks;

    // Info for dividing shared memory
    size_t sharedMemoryPerSievingBlock;
    uint32_t num_sharedMemoryBuckets; ///< given by num_sievingBlocks * polyBlockSize.
    uint32_t bucketSize;
    uint32_t globalBucketSize;
    uint32_t log2_bucketSize;
    uint32_t num_iterationsPerStep;   ///< Sync threads after this many iterations.
    uint32_t purgeThreshold;          ///< Threshold to trigger bucket emptying.
    uint32_t bigPrimeStartIndex;

    uint32_t num_threadBlocks2;
    uint32_t maxRelationsPerBlock;    ///< Overestimate to avoid dynamic allocation.
};

struct initConfig {
    uint32_t num_threadsPerBlock;
    uint32_t num_threadBlocks;
    uint32_t batch_size = 0;          ///< Parameter for batch sieving
};

// Constant across all sieving calls
struct generalSievingConfig {
    uint32_t sievingBlockSize;
    uint32_t log2_sievingBlockSize;

    uint32_t num_polysPerSieveCall;         ///< Unique polys sieved across all blocks per call.
    uint32_t num_subCubes;
    uint32_t num_sievingBlocksPerSieveCall; ///< Unique sievingBlocks sieved across all blocks per call.
    uint32_t num_sievingBlockBatches;       ///< For fixed polys, how many calls would one need to sieve over [-M,M)
    uint32_t batch_size;                    ///< Parameter for batch sieving

    uint32_t globalBucketSize;
    uint32_t bigPrimeStartIndex;
    uint32_t midPrimeStartIndex;

    uint32_t maxRelationsPerBlock;
};

struct globalMetaSieveConfig {
    uint32_t num_threadsPerBlock;
    uint32_t num_threadBlocks = 0;
    uint32_t batch_size;                    ///< Parameter for batch sieving

    uint32_t maxActiveBucketsTotal;
    uint32_t num_activeBucketsPerThreadBlock;

    uint32_t polyBlockSize;
    uint32_t log2_polyBlockSize;
    uint32_t num_polyBlocksPerThreadBlock;
    uint32_t log2_num_polyBlocksPerThreadBlock;

    uint32_t num_activeBlocksPerCycle;
    uint32_t num_metaSieveCycles;

    size_t sharedMemReq;
};

struct sieveAndScanConfig {
    uint32_t num_threadsPerBlock;
    uint32_t num_threadBlocks;
    uint32_t batch_size;                    ///< Parameter for batch sieving

    size_t sharedMemReq;
};

struct processRelationsConfig {
    uint32_t num_threadsPerBlock;
    uint32_t num_threadBlocks;
};

struct polyData {
    uint32_t approxPolyRoot; ///< Approximate positive root of the sieved polynomials.
    uint32_t log2_a;         ///< Log2 of coefficient 'a'.
    uint32_t threshold;      ///< Logarithmic threshold for identifying smooth numbers.
};

struct fixedSievingParams {
    uint32_t fb_size;        ///< Factor base size.
    uint32_t shc_dim;        ///< Dimension of the hypercube (number of factors in 'a').
    uint32_t M;              ///< Sieve interval radius.
    uint32_t approxPolyRoot;
    uint32_t threshold;
};

struct dynamicSievingParams {
    mpqs::uint512 a;         ///< The coefficient 'a' (512-bit).
    uint32_t log2_a;         ///< Log2 of 'a'.
    int32_t startIndex;      ///< Starting index for sieving.
    uint32_t subCube;        ///< Sub-cube index for work distribution.
    uint32_t newCube;        ///< Flag for sieving with a new cube
};

struct gpuInfo {
    size_t totalGlobalMem;
    size_t sharedMemPerMultiprocessor;
    size_t maxSharedMemPerBlock;
    uint32_t multiProcessorCount;
    uint32_t maxThreadsPerMultiProcessor;
};

} // namespace sieve

namespace postprocessing {
// -----------------------------------------------------------------------------
// Memory Architecture primitives
// -----------------------------------------------------------------------------

// We formalize a Double Buffer Object
struct DoubleBuffer {
    mpqs::sieve::DenseCandidate* d_data; // The candidate payload
    uint32_t* d_counter;                 // VRAM atomic counter (Fast GPU atomics)
    uint32_t capacity;                   // Maximum number of candidates before overflow

    /**
     * @brief Cross-stream Synchronization Events.
     * Let Q_s be the Sieve execution queue and Q_p be the PostProcessing queue.
     * These locks preserve the DAG execution invariants without stalling the CPU.
     */
    cudaEvent_t safe_to_write_event;     // Lock: Factorization is done, Siever can overwrite
    cudaEvent_t safe_to_read_event;      // Lock: Accumulation is done, Factorizer can read
};

} // namespace postprocessing

} // namespace mpqs
