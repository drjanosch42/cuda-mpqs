// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <memory> // unique_ptr

#include "common.h"
#include "mpqs_structures.h"       // For Relation for temp usage
#include "mpqs_soa.h"
#include "sieving_data_structs.h"    // For candidateRelation, devicePointers, factoringData
#include "uint512.cuh"             // uint512 primitives
#include "math_utils.cuh"           // For Q(X) computation
#include "logger/hpc_logger.h"

// Forward declaration for LP stats (avoids pulling in largeprime.h)
namespace mpqs { namespace lp { struct SLPPinnedStats; } }

namespace mpqs {
namespace postprocessing {

// -----------------------------------------------------------------------------
// Prediction Result (mapped pinned memory — zero-sync host readback)
// -----------------------------------------------------------------------------

/**
 * @brief GPU-written prediction state for adaptive batch convergence.
 * Allocated as cudaHostAllocMapped so the host can poll without synchronization.
 */
struct PredictionResult {
    volatile uint32_t should_terminate;  ///< 1 = stop launching batches
    uint32_t          effective_R;       ///< R + predicted LP yield
    float             yield_rate;        ///< λ = R / total_steps
    float             lp_match_rate;     ///< μ = LP_full / (LP_full + W)
};

// -----------------------------------------------------------------------------
// Buffer Fill Telemetry (mapped pinned memory -- zero-sync host readback)
// -----------------------------------------------------------------------------

/**
 * @brief GPU-written buffer fill snapshot for all major pipeline buffers.
 * Allocated as cudaHostAllocMapped so the host can poll without synchronization.
 * Uses generation ticket for lock-free consistency (same pattern as SLPPinnedStats).
 */
struct BufferFillSnapshot {
    uint64_t accum_fill;          ///< Current accumulation buffer fill (candidates)
    uint64_t accum_capacity;      ///< Accumulation buffer capacity
    uint64_t partial_fill;        ///< Current partial buffer fill (LP staging)
    uint64_t partial_capacity;    ///< Partial buffer capacity
    uint64_t persistent_fill;     ///< Persistent relation store fill
    uint64_t persistent_capacity; ///< Persistent store capacity
    volatile uint64_t generation; ///< Monotonic ticket for lock-free reads
};

/**
 * @brief Host-side historical statistics accumulated from BufferFillSnapshot polls.
 * Updated by the orchestrator each time generation advances.
 */
struct BufferFillHistory {
    uint64_t accum_max = 0, accum_sum = 0;
    uint64_t partial_max = 0, partial_sum = 0;
    uint64_t persistent_max = 0;
    uint64_t sample_count = 0;

    void update(const BufferFillSnapshot& snap) {
        if (snap.accum_fill > accum_max)       accum_max = snap.accum_fill;
        if (snap.partial_fill > partial_max)   partial_max = snap.partial_fill;
        if (snap.persistent_fill > persistent_max) persistent_max = snap.persistent_fill;
        accum_sum += snap.accum_fill;
        partial_sum += snap.partial_fill;
        ++sample_count;
    }

    double accum_avg() const { return sample_count ? (double)accum_sum / sample_count : 0.0; }
    double partial_avg() const { return sample_count ? (double)partial_sum / sample_count : 0.0; }
};

/**
 * @brief Host-side historical statistics for LP buffer usage.
 * Updated by the orchestrator each time SLPPinnedStats generation advances.
 * Note: update() is not defined inline because SLPPinnedStats is only
 * forward-declared in this header. Call-site code must access stats fields directly.
 */
struct LPFillHistory {
    uint64_t witness_capacity = 0;   ///< Max witness capacity (set once at init)
    uint64_t witness_max = 0, witness_sum = 0;
    uint64_t total_slab_overflows = 0;
    uint64_t total_witness_overflows = 0;
    uint64_t total_output_overflows = 0;
    uint64_t sample_count = 0;

    double witness_avg() const { return sample_count ? (double)witness_sum / sample_count : 0.0; }
};

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/**
 * @brief Configuration parameters for the Post-Processing pipeline.
 */
struct PostProcConfig {
    /// @brief Number of candidates to accumulate before triggering factorization.
    /// Recommended: 65536 or 131072 to saturate GPU during sorting/scan.
    uint32_t accumulate_buffer_size;
    /// @brief Triggers batch factorization in postprocessing
    uint32_t accumulate_batch_purge_threshold; // Recommended: about 0.9 x accumulate_buffer_size

    uint32_t persistent_device_buffer_size = 0; // If non-zero, activates persistent device Relation buffer
    
    uint64_t lp1_bound = 0; // bound for single large prime variation. 0 = Disabled.
    /// @brief Size of the temporary buffer for 1-partial relations (if lp1_bound > 0).
    /// Defaults to accumulate_buffer_size if 0.
    uint32_t partial_buffer_size = 0;
    
    /// @brief Dimension of hypercube (number of prime factors in a).
    uint32_t shc_dim;

    // -- Branch-fixed character columns (Stage 4) --
    /// @brief When true, processCandidate computes the r-bit branch character vector
    /// from the SIGNED (ax+b) for every EMITTED relation (--char_mode branch). When
    /// false (default --char_mode norm) the computation is fully gated off — char_bits
    /// is written as a defined 0 with zero added hot-path arithmetic.
    bool char_branch = false;
    /// @brief Number of branch aux primes (r <= 32). Valid only when char_branch.
    uint32_t char_r = 0;
    /// @brief Device array of r branch aux primes q_s (64-bit, > lp1_bound).
    const uint64_t* d_char_q = nullptr;
    /// @brief Device array of r fixed Tonelli roots t_s (t_s^2 == N mod q_s), in [0,q_s).
    const uint64_t* d_char_t = nullptr;

   /// CUDA
   int device_id = 0;
};

// -----------------------------------------------------------------------------
// Controller Class
// -----------------------------------------------------------------------------

/**
 * @brief Manages the Device-side Post-Processing pipeline.
 * 
 * @details
 * Implements a "Saturate and Sort" strategy:
 * 1. Accumulates sparse candidates from Siever into a dense buffer.
 * 2. When full, sorts candidates by complexity (hint count).
 * 3. Launches a batched factorization kernel using L2-cached Factor Base.
 */
class DevicePostProcessingController {
public:
    DevicePostProcessingController();
    ~DevicePostProcessingController();

    /**
     * @brief Allocates device memory and configures the controller.
     * 
     * @param f_data Factoring data (needed for N).
     * @param common_ptrs Pointers to FactorBase, RootN, etc. (Must be valid on GPU).
     * @param cfg Configuration parameters.
     */
    void initiate(const mpqs::sieve::factoringData& f_data, 
                  const mpqs::sieve::devicePointers& common_ptrs, 
                  const PostProcConfig& cfg);

    /**
     * @brief Ingests raw/sparse output from the Siever.
     * 
     * Runs a compaction kernel on the provided stream to copy valid 
     * candidates into the internal accumulation buffer.
     * 
     * @param raw_input Pointer to Siever's candidateRelation buffer (Device).
     * @param raw_size Number of elements in raw_input.
     * @param current_a: mpqs::uint512 value of current a (from ax+b).
     * @param current_a_factors: vector<uint32_t> (indices for odd factor base divisors of a).
     * @param start_index: -M (lower sieving interval end)
     * @param stream The CUDA stream used by the Siever (ensures serialization).
     * @return true if the internal buffer is full and ready for processing.
     */
    bool accumulate(const mpqs::sieve::candidateRelation* raw_input,
		    uint32_t raw_size,
		    mpqs::uint512 current_a,
		    const uint32_t* dev_a_factors_ptr,
		    uint32_t num_a_factors,
		    int32_t start_index,
		    cudaStream_t stream);

    /**
     * @brief Triggers the processing pipeline (Sort -> Factorize).
     * Populates d_full_batch (Full Relations) and d_partial_batch (1-Partials).
     */
    void processBufferedCandidates();

    /**
     * @brief Moves valid Full Relations from the temporary processing batch
     * into the long-term persistent storage on Device.
     * Clears the temporary processing batch counters after move.
     */
    void consolidateToPersistent();  

    /**
     * @brief Forces processing of any remaining candidates in the buffer.
     */
    void flush();

    // --- Accessors ---

    /**
     * @brief Access to sieving stage critical internal states.
     */
    bool isProcessing() const { return processing_active; }
    uint32_t getAccumulatedCount(cudaStream_t sync_stream = 0);

    // Handshake for Siever
    mpqs::sieve::DenseCandidate* getAccumulationBufferPtr() { return d_accumulation_buffer; };
    uint32_t* getCounterPtr() { return d_current_accumulation_counter; };

    // Handshake for LargePrime
    mpqs::structures::RelationBatch* getPartialBatch() { return d_partial_batch.get(); }
    uint32_t getPartialCount();

    /// @brief Resets the partial batch counters and dual counter for the next LP cycle.
    /// Called after LP processes accumulated partials in the batch-sieve path.
    void resetPartialBatch();

    /// @brief Stream-parameterized variant for cluster extraction on a non-default stream.
    void resetPartialBatch(cudaStream_t stream);

    /// @brief Resyncs d_persistent_dual_counter with the actual persistent batch counts.
    /// Must be called after any external append to the persistent batch (e.g., LP).
    void resyncPersistentDualCounter();
  
    // Accessor for Persistent Storage (for LargePrime or Final Download)
    mpqs::structures::RelationBatch* getPersistentBatch() { return d_persistent_batch.get(); }  
    
    // Returns the current count of relations in the persistent buffer
    uint64_t getPersistentCount();

    void requestStats(cudaStream_t stream);
    
   /**
     * @brief Request stats for the CURRENT active witness buffer
     * Triggers a kernel launch which does not stall host.
     */
    void updateStats();
  
   /**
     * @brief Read stats from pinned memory.
     */
    std::pair<uint64_t, uint64_t> getPartialStats() const;
    std::pair<uint64_t, uint64_t> getPersistentStats() const;

    size_t getPartialCapRels() const { return d_partial_batch ? d_partial_batch->getCapacityRels() : 0; }
    size_t getPartialCapFactors() const { return d_partial_batch ? d_partial_batch->getCapacityFactors() : 0; }  

    /**
     * @brief Deduplicates the persistent relation batch in-place using a specialized 64-bit hash.
     * This prepares the data for Matrix Construction by removing redundant relations.
     */
    void deduplicatePersistentBatch();  

    /**
     * @brief Returns the CUDA stream of the post processor
     * for handshake with large prime variation processing
     */
    cudaStream_t getCudaStream() { return proc_stream; };

    // Frees internal buffers. Safe to be called repeatedly.
    void clearBuffers();
  
    // Frees persistent buffer. Safe to be called repeatedly
    // or if no persistent buffering is disabled.
    void clearPersistentBuffer();


    // debug csr structure
#ifdef DEBUG_SOA
    void debugDumpHead(int n);
#endif

    // -------------------------------------------------------------------------
    // Batch-Sieve Interface
    // -------------------------------------------------------------------------

    /**
     * @brief Batch-sieve factorization entry point.
     *
     * Reads the candidate count from the active DoubleBuffer's device counter
     * (zero CPU/GPU sync), launches the SoA factorization kernel writing to
     * d_persistent_batch via append_to_soa(), and toggles the active buffer.
     */
    void processBatchBufferedCandidates();

    /**
     * @brief Bipartite state transition f: idx -> idx ^ 1.
     * Used exclusively in the batch-sieving manifold to alternate subsets.
     */
    void toggleActiveBuffer() { active_accum_idx ^= 1; }

    /// @brief Returns a pointer to the currently active accumulation DoubleBuffer.
    DoubleBuffer* getActiveAccumulationBuffer() { return &buffers[active_accum_idx]; }

    /**
     * @brief Explicitly overrides the internal pipeline flush state.
     *
     * In the batch-sieving path, the accumulation buffer is populated via
     * device-to-device transactions (bypassing host-side accumulate()), so the
     * flush guard must be set manually before launching factorization.
     */
    void setFlushedState(bool state) { is_flushed = state; }

    /**
     * @brief Returns the CUDA event recorded after counter reset on proc_stream.
     * Used for multi-stream coordination between sieve and postprocessor.
     */
    cudaEvent_t getResetEvent() const { return reset_event; }

    // --- Volatile pinned counters for zero-copy batch-sieve polling ---
    // CPU reads these directly without cudaMemcpy.
    volatile uint32_t* h_pinned_accumulation_counter = nullptr;
    volatile uint32_t* h_pinned_persistent_count = nullptr;
    volatile uint32_t* h_pinned_partial_count = nullptr;  ///< LP partial batch count (zero-sync)

    // --- Adaptive convergence prediction (mapped pinned memory) ---
    PredictionResult* h_prediction_result = nullptr;

    // --- Prediction parameter setters ---
    void setPredictionParams(uint32_t target, const mpqs::lp::SLPPinnedStats* lp_stats_device_ptr) {
        prediction_target_ = target;
        d_lp_stats_ = lp_stats_device_ptr;
    }
    // total_steps is u64: the solo cursor `current_step` was widened u32→u64 for the
    // mid-sieve checkpoint (RSA-140 scale exceeds 2^32 a-values), and it flows in here.
    void updatePredictionSteps(uint64_t total_steps) { prediction_total_steps_ = total_steps; }

    /// @brief Lock-free accessor for buffer fill snapshot (mapped pinned memory).
    const BufferFillSnapshot* getBufferFillSnapshot() const { return h_buffer_fill_; }

private:
    PostProcConfig config;
    mpqs::uint512 N; // The number being factored

    // CUDA Stream for post-processing (possibly distinct from Sieving stream)
    cudaStream_t proc_stream;

    // Batch-sieve double buffer state
    DoubleBuffer buffers[2];
    int active_accum_idx = 0; // 0 or 1

    // Double Buffering logic
    uint32_t* d_counter_A = nullptr; // Counter for accumulation Buffer A
    uint32_t* d_counter_B = nullptr; // Counter for accumulation Buffer B
    mpqs::sieve::DenseCandidate* d_accumulation_buffer = nullptr; // Input buffer (filling up)
    mpqs::sieve::DenseCandidate* d_processing_buffer = nullptr;   // Work buffer (being factored)
    
    uint32_t accumulation_count;      // Items currently in accumulation buffer
    uint32_t* d_current_accumulation_counter = nullptr; // Atomic counter on device
    uint32_t* d_current_processing_counter = nullptr;   // Atomic counter on device

    // --- SoA BUFFERS ---
    // We use unique_ptr to manage lifetime, preventing header dependency issues 
    // if RelationBatch includes heavy Thrust headers.
    
    // 1. Output Buffer (replaces d_output_buffer)
    // Used for transfer to Host and transfer to Persistent buffer.
    std::unique_ptr<mpqs::structures::RelationBatch> d_full_batch;

    // 2. Partial Buffer (replaces d_partial_buffer)
    std::unique_ptr<mpqs::structures::RelationBatch> d_partial_batch;

    // 3. Persistent Buffer (replaces d_persistent_relations)
    std::unique_ptr<mpqs::structures::RelationBatch> d_persistent_batch;
  
    // Atomic dual counters for d_full_batch and d_partial batch
    // CAUTION: Only safe for total (factor) counts < 2^32 (during relation accumulation)
    // CONSEQUENCE: d_full_batch and d_partial_batch should not exceed
    // "2^32 / max_number_of_factors_per_relation"
    unsigned long long* d_full_dual_counter = nullptr;
    unsigned long long* d_partial_dual_counter = nullptr;
    unsigned long long* d_persistent_dual_counter = nullptr; // For batch-sieve direct-to-persistent writes

    // External references (Device Pointers to constant memory)
    mpqs::sieve::devicePointers dev_common;

    bool processing_active = false;
    bool is_flushed = true;
    cudaEvent_t reset_event; // to synchronize pointer swapping for CUDA streams 0 and 1
  
    // --- Prediction kernel state ---
    PredictionResult* d_prediction_result = nullptr;
    uint32_t prediction_target_ = 0;
    uint64_t prediction_total_steps_ = 0;
    const mpqs::lp::SLPPinnedStats* d_lp_stats_ = nullptr;

    // --- Buffer fill telemetry (mapped pinned memory) ---
    BufferFillSnapshot* h_buffer_fill_ = nullptr;
    BufferFillSnapshot* d_buffer_fill_ = nullptr;

    // Internal Helpers
    void swapBuffers();
    void resetAccumulation();
};

} // namespace postprocessing
} // namespace mpqs
