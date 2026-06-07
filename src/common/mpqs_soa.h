// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <memory>
#include "uint512.cuh"
#include "hpc_logger.h" 

// Forward declaration of view
namespace mpqs {
namespace structures {

struct RelationBatchView; // Defined below

/**
 * @brief Host-side mirror for final download.
 */
struct HostRelationBatch {
    // Algebra
    std::vector<mpqs::uint512> sqrt_Q;
    std::vector<uint8_t> signs;
    std::vector<int32_t> val_2_exps;
    std::vector<unsigned __int128> large_primes;
    /// Branch-fixed character vector (Stage 4): r-bit (r=32) per-relation char vector
    /// computed at relation birth from the SIGNED (ax+b). One bit per branch aux prime.
    /// Always 0 (a defined value) under --char_mode norm; only meaningful under branch.
    std::vector<uint32_t> char_bits;

    // CSR Factors
    std::vector<uint64_t> factor_offsets; 
    std::vector<uint32_t> factor_indices; 
    std::vector<uint8_t>  factor_counts;  

    // Counters
    size_t num_relations = 0;
    size_t num_factors = 0;

    void resize(size_t n_rels, size_t n_factors);
    void clear();
};

/**
 * @brief Lightweight View for Kernels.
 * Passed by value to CUDA kernels.
 */
struct RelationBatchView {
    // Algebra
    mpqs::uint512* sqrt_Q;
    uint8_t* signs;
    int32_t* val_2_exps;
    unsigned __int128* large_primes;
    uint32_t* char_bits;  ///< Branch-fixed per-relation char vector (Stage 4); 0 in norm mode.
    
    // CSR Factors
    uint64_t* factor_offsets;    // [Size + 1]
    uint32_t* factor_indices;    // [Factor_Capacity]
    uint8_t*  factor_counts;     // [Factor_Capacity]

    // Global Atomic Counters (Device Pointers)
    uint64_t* global_count;      // Current number of relations
    uint64_t* global_factor_idx; // Current number of factors

    // Bounds for Safety
    uint32_t max_relations;
    uint64_t max_factors;

    // Adaptive convergence cap (0 = unlimited)
    uint32_t target_cap = 0;
};

/**
 * @brief Raw Pointer SoA Container.
 * Manages memory lifecycle and resizing logic.
 */
class RelationBatch {
public:
    RelationBatch() = default;
    ~RelationBatch();

    // Prevent copying to prevent double-free
    RelationBatch(const RelationBatch&) = delete;
    RelationBatch& operator=(const RelationBatch&) = delete;

    // Move operations deleted: raw CUDA pointers require explicit lifecycle management
    RelationBatch(RelationBatch&&) = delete;
    RelationBatch& operator=(RelationBatch&&) = delete;

    /**
     * @brief Initialization.
     * Sets device context and allocates pinned memory for counters.
     */
    void initiate(int device_id);

/**
     * @brief Resizes the internal buffers to the specified capacity.
     * * @note NON-DESTRUCTIVE: Existing data (relations and factors) is PRESERVED 
     * up to the current count. If the new capacity is smaller than the current 
     * count, data is truncated.
     * * @param num_rels_needed Minimum capacity for relations.
     * @param num_factors_needed Minimum capacity for factors (CSR storage).
     */
    void resize(size_t num_rels_needed, size_t num_factors_needed);

    /**
     * @brief Resets the relation and factor counters to zero.
     * Does not deallocate memory. Effectively empties the batch.
     * * @param stream Stream to execute the memset on.
     */
    void clear(cudaStream_t stream = 0);

    /**
     * @brief Appends relations from another batch to this batch.
     * * @note SAFE APPEND: If the other batch is larger than the remaining capacity
     * of this batch, this method will append as many relations as possible (fitting
     * both relation slots and factor slots) and DISCARD the rest.
     * * @param other Source batch.
     * @param other_count_rels Number of relations in the source batch to consider.
     * @param stream CUDA stream for async operations.
     */
    void append(
        RelationBatch& other, 
        uint64_t other_count_rels, 
        cudaStream_t stream
    );

    /**
     * @brief Resets the atomic counters to 0.
     * Does NOT zero out data arrays (performance optimization).
     */
    void reset_counters(cudaStream_t stream);

    /**
     * @brief Download contents to Host (Device to Host).
     */
    void moveToHost(HostRelationBatch& dest, cudaStream_t stream);

    /**
     * @brief Download a sub-range [offset, offset+count) to host.
     * Two-phase: flat arrays first (async), then sync to read CSR offsets,
     * then CSR factors (async). Re-bases factor_offsets to start at 0.
     *
     * @param dest    Host-side output batch (resized internally).
     * @param offset  Starting relation index in the device batch.
     * @param count   Number of relations to download.
     * @param stream  CUDA stream for async D2H copies.
     * @note One cudaStreamSynchronize between phases for CSR offset readback.
     */
    void moveRangeToHost(HostRelationBatch& dest, uint64_t offset, uint64_t count,
                         cudaStream_t stream);

    /**
     * @brief Get the current number of relations stored (Async copy from device).
     */
    uint64_t getCount(cudaStream_t stream) const;
  
    /**
     * @brief Get the current number of factors stored (Async copy from device).
     */
    uint64_t getFactorCount(cudaStream_t stream) const;

    /**
     * @brief Async Stats: Request copy from Device -> Host Pinned
     */
    void requestStats(cudaStream_t stream);
    
    /**
     * @brief Updates relations and factors counters in pinned host
     * memory asynchronously. Launches a tiny kernel. Does not stall CPU.
     */
    void updateStats(cudaStream_t stream);
  
    /**
     * @brief Read the stats requested previously (non-blocking if request was processed)
     * Returns pair { relations, factors }
     */
    std::pair<uint64_t, uint64_t> readStats() const;

    /**
     * @brief Returns a View struct suitable for passing to CUDA kernels.
     */
    RelationBatchView get_view();

    /**
     * @brief Sync only atomic counters (16 bytes) from device to host pinned memory.
     * Used by managed-memory moveToHost() path to avoid full D→H array copies.
     */
    void syncCounters(cudaStream_t stream);

    /**
     * @brief Sets the adaptive convergence relation cap.
     * @param cap Target relation count (0 = unlimited).
     */
    void setTargetCap(uint32_t cap) { target_cap_ = cap; }

    /// @brief Returns true if this batch uses cudaMallocManaged (unified memory).
    bool useManagedMemory() const { return use_managed_memory_; }

    // -- Accessors for Thrust or Raw Operations --
    unsigned __int128* getLargePrimesData() const { return d_large_primes; }
    uint64_t* getFactorOffsetsData() const { return d_factor_offsets; }
    
    // Returns the current allocation capacity (not the fill count)
    size_t getCapacityRels() const { return cap_rels; }
    size_t getCapacityFactors() const { return cap_factors; }

    /// @brief Returns device pointer to the global relation count atomic.
    uint64_t* getDeviceCountPtr() { return d_global_count_ptr; }

    /**
     * @brief Validation Debug Structure
     */
    struct ValidationInfo {
        uint32_t rel_idx;
        int error_type; // 1: Sign, 2: Product Mismatch, 3: Bad CSR
        mpqs::uint512 calculated_diff;
        mpqs::uint512 calculated_prod;
        uint8_t sign_stored;
        uint8_t sign_expected;
    };

    /**
     * @brief Validates the relations currently stored in this device batch.
     * Computes |(ax+b)^2 - N| and compares it with the product of factors.
     * 
     * @param N The modulus.
     * @param d_factor_base Device pointer to the factor base (uint32_t array).
     * @param fb_size Number of elements in the factor base.
     */
    void validate_relations(const mpqs::uint512& N, const uint32_t* d_factor_base, size_t fb_size);

    /**
     * @brief Static helper to validate a HostRelationBatch.
     * Transfers data to GPU temporarily, runs validation, logs results, cleans up.
     * 
     * @param host_batch The batch on CPU.
     * @param host_fb The factor base on CPU.
     * @param N The modulus.
     */
    static void validate_host_batch(
        const HostRelationBatch& host_batch,
        const std::vector<uint32_t>& host_fb,
        const mpqs::uint512& N
    );

    /**
     * @brief Upload a HostRelationBatch to this device batch.
     * Resizes internal buffers, copies all arrays H→D, and sets atomic counters.
     * Used by LINALG_ONLY mode to reconstruct device state from serialized host data.
     *
     * @param host_batch The host-side batch to upload.
     */
    void uploadFromHost(const HostRelationBatch& host_batch);

private:
    int device_id = -1;
    bool use_managed_memory_ = false;  ///< true when cudaMallocManaged used (Jetson / unified memory)
    bool use_mem_advise_ = false;      ///< true when concurrentManagedAccess supported (discrete GPU)

    // -- Adaptive convergence --
    uint32_t target_cap_ = 0; // Relation count ceiling (0 = unlimited)

    // -- Capacities --
    size_t cap_rels = 0;    // Allocated capacity for relations
    size_t cap_factors = 0; // Allocated capacity for factors

    // -- Device Pointers (Relations) --
    mpqs::uint512*     d_sqrt_Q = nullptr;
    uint8_t*           d_signs = nullptr;
    int32_t*           d_val_2_exps = nullptr;
    unsigned __int128* d_large_primes = nullptr;
    uint32_t*          d_char_bits = nullptr;  ///< Branch-fixed per-relation char vector (Stage 4).

    // -- Device Pointers (CSR) --
    uint64_t* d_factor_offsets = nullptr;
    uint32_t* d_factor_indices = nullptr;
    uint8_t*  d_factor_counts  = nullptr;

    // -- Counters --
    uint64_t* d_global_count_ptr = nullptr;  // Device-side atomic
    uint64_t* d_global_factor_ptr = nullptr; // Device-side atomic
    uint64_t* h_pinned_counters = nullptr;   // Host-side pinned [0]=rels, [1]=factors
    uint64_t* d_pinned_counters = nullptr;   // Device-mapped alias of h_pinned_counters (via cudaHostGetDevicePointer), [0]=rels, [1]=factors
};

} // namespace structures
} // namespace mpqs
