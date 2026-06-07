// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <memory>
#include <utility>

#include "mpqs_soa.h"
#include "uint512.cuh"
#include "logger/hpc_logger.h"

namespace mpqs {
namespace lp {

/**
 * @brief Lock-Free Pinned Statistics Structure for Host/Device monitoring.
 * 
 * Mathematically tracks the throughput and state of the Hash Table and Global Buffers.
 * 
 * To ensure safe asynchronous reads by the Host without stalling the GPU:
 * The GPU will update fields [0..9] first, execute a memory fence (__threadfence_system),
 * and finally update `total_iterations`. The host can observe `total_iterations` as a
 * monotonic "ticket" that guarantees the integrity of the preceding 10 fields.
 */
struct SLPPinnedStats {
    uint64_t new_partials_buffer_fill;   ///< Number of 1-partials in the last consumed batch
    uint64_t total_witnesses;            ///< Total unique single large primes currently stored
    uint64_t total_full_relations;       ///< Total number of full relations successfully produced
    uint64_t last_batch_full_relations;  ///< Number of full relations produced during the last batch
    uint64_t last_batch_new_witnesses;   ///< Number of new witnesses added during the last batch
    uint64_t empty_hash_buckets;         ///< Number of buckets with Count == 0
    uint64_t full_hash_buckets;          ///< Number of buckets with Count >= ROW_WIDTH_ELEMS

    // --- Overflow counters (cumulative, monotonically increasing) ---
    uint64_t slab_overflow_count;        ///< Silently dropped slab appends (row full)
    uint64_t witness_overflow_count;     ///< Silently dropped witness SoA reservations
    uint64_t output_overflow_count;      ///< Silently dropped combined output reservations

    // MUST BE LAST - Acts as the generation/validity ticket.
    volatile uint64_t total_iterations;  ///< Total number of times 1-partials batch was processed
};

/**
 * @brief Configuration for the 2-Stage Slab Hash Table SLP Pipeline.
 */
struct LargePrimeConfig {
    /// @brief Absolute maximum number of unique large primes to store.
    /// Determines the number of rows allocated in the Payload Slab.
    uint64_t max_witness_capacity = 16777216; // e.g., 16M
    
    /// @brief Capacity for the combined relations output buffer.
    uint32_t max_combined_output = 0;

    /// @brief Bit width for the Hash Table Directory size (B). 
    /// Total Buckets = 2^B. Default 20 = 1,048,576 Buckets.
    uint32_t hash_bits = 20;

    /// @brief Bound for Large Prime 1. If 0, LP variant is effectively disabled.
    uint64_t lp1_bound = 0;

    /// @brief When true, delete matched witness from slab after first match.
    /// Prevents one witness generating multiple combined relations (degenerate
    /// kernel vectors) at small factor-base sizes.
    bool purge_after_match = false;

    /// @brief When true, capture per-combined-relation constituent provenance
    /// (probe root u_p, witness root u_w, signs, val_2_exps, LP) into a separate
    /// device buffer BEFORE the slab purge erases the linkage. STRICTLY ADDITIVE
    /// and DIAGNOSTIC: when false, NO provenance buffer is allocated and NO extra
    /// kernel is launched, so the factorization path and its performance are
    /// byte-for-byte unchanged. Gated by CLI --dump_combine_provenance.
    bool capture_provenance = false;

    int device_id = 0;
};

/**
 * @brief One captured combined-relation provenance record.
 *
 * Records the two constituents of a combined relation u_c = u_p * u_w (mod N),
 * formed in global_combine_kernel from an input "probe" 1-partial (u_p) and a
 * stored "witness" (u_w) sharing the same large prime. Captured by a SEPARATE
 * diagnostic kernel that reads the identical inputs as the combine kernel, so it
 * is order-aligned with no semantic effect. The (lp, u_c) pair uniquely links
 * each record back to its row in the combined-output / relations.v2 store.
 *
 * Sign encoding mirrors the SoA convention: 1 = positive, anything else = negative.
 * POD / trivially copyable for direct binary serialization.
 */
struct CombineProvenanceEntry {
    mpqs::uint512 u_c;        ///< Combined root = u_p * u_w (mod N)  [must match output sqrt_Q]
    mpqs::uint512 u_p;        ///< Probe (input 1-partial) root sqrt_Q
    mpqs::uint512 u_w;        ///< Witness (stored) root sqrt_Q
    unsigned __int128 lp;     ///< Shared large prime (both constituents)
    uint32_t input_idx;       ///< Probe index within its sieve batch (intra-batch only)
    uint32_t witness_idx;     ///< Global witness SoA index (stable across the whole run)
    int32_t  v2_p;            ///< Probe val_2_exp
    int32_t  v2_w;            ///< Witness val_2_exp
    uint8_t  sign_p;          ///< Probe sign (1=positive)
    uint8_t  sign_w;          ///< Witness sign (1=positive)
    uint8_t  _pad[6];         ///< Explicit padding (keep struct size deterministic for I/O)
};

/**
 * @brief State Machine Flags for the 2-Stage SLP Pipeline.
 */
enum class SLPStatus : uint8_t {
    CONSUMED = 0,            ///< Default zero-initialized state: dropped, redundant, or already merged.
    NEEDS_ALLOCATION = 2,    ///< Hash bucket was empty. Needs to allocate a new Slab Row.
    NEEDS_PROBE = 3,         ///< Hash bucket has items. Needs to vectorized-probe the Slab.
    MATCH_FOUND = 4,         ///< Matched an existing witness. Ready to form Full Relation.
    NEEDS_APPEND = 5         ///< Hash collision but no LP match. Must append to the Slab Row.
};

/**
 * @brief Manages the O(M) Single Large Prime (SLP) variation using an Append-Only Slab Hash Table.
 * 
 * Mathematical Memory Model:
 * Let B = hash_bits. Let C_max = max_witness_capacity.
 * 1. Directory Array: S_dir = 2^B * 8 bytes.
 *    Mapping: h(p) = (p >> 1) mod 2^B
 * 2. Payload Slab: S_slab = C_max * ROW_WIDTH_ELEMS * 8 bytes.
 *    Payload Encoding: T(p) = p >> (B + 1).
 *    To fit T(p) into a 32-bit tag, p < 2^(32 + B + 1). For B=20, p < 2^53.
 */
class LargePrimeVariant {
public:
    LargePrimeVariant();
    LargePrimeVariant(const cudaStream_t stream);
    ~LargePrimeVariant();

    /**
     * @brief Allocates the Hash Table, SoA buffers, and atomic counters.
     */
    void initiate(const LargePrimeConfig& cfg, const mpqs::uint512& modulus);

    /**
     * @brief Frees all Device and Host memory allocated by this module.
     */
    void clearBuffers();

    // The logic execution pipeline (Milestones 2-5)
    void processAndCommit(
        mpqs::structures::RelationBatch* input_partials,
        mpqs::structures::RelationBatch* persistent_storage,
        uint32_t input_count_hint = 0  // Pinned counter from postprocessor (0 = fallback to getCount)
    );

    /// @brief Enqueues the full LP kernel chain on lp_stream and returns immediately.
    /// Does NOT block the CPU. Records lp_done_event_ at the end.
    /// The orchestrator is responsible for deferred append() after lp_done_event fires.
    /// Uses device-side count snapshot (no host readback of input count).
    /// @param input_partials Partial batch from postprocessor (device memory).
    /// @param persistent_storage Persistent batch to append LP output to (saved for reference).
    void processAndCommitAsync(
        mpqs::structures::RelationBatch* input_partials,
        mpqs::structures::RelationBatch* persistent_storage
    );

    /// @brief Non-blocking check: has the LP stream finished its last dispatch?
    /// Uses cudaEventQuery(lp_done_event_). Returns true if complete or no LP active.
    bool isComplete() const;

    // --- Accessors for non-blocking telemetry ---
    void requestStats(); 
    void updateStats(); 
    std::pair<uint64_t, uint64_t> getWitnessStats() const;
    size_t getWitnessCapacityRels() const;
    size_t getWitnessCapacityFactors() const;

    /**
     * @brief Lock-Free Host Accessor for Pipeline Telemetry.
     * Returns a pointer to the pinned memory region continuously updated by the GPU.
     */
    const SLPPinnedStats* getTelemetry() const { return h_pinned_stats; }

    /// @brief Returns the most recent LP combined relation count from pinned memory (no sync).
    uint64_t getCombinedCountPinned() const {
        return h_pinned_lp_combined_count ? h_pinned_lp_combined_count[0] : 0;
    }

    /// @brief Returns the LP-done event for orchestrator to wait on.
    cudaEvent_t getDoneEvent() const { return lp_done_event_; }

    /// @brief Returns the count-snapshot event: fires after d_lp_input_count_ is written.
    /// Orchestrator waits on this before resetting the partial batch.
    cudaEvent_t getCountSnapshotEvent() const { return count_snapshot_event_; }

    /// @brief Returns the partials-ready event for orchestrator to record on postproc stream.
    cudaEvent_t getPartialsReadyEvent() const { return partials_ready_event_; }

    /// @brief Returns the LP output batch (for deferred append by orchestrator).
    mpqs::structures::RelationBatch& getOutputBatch() { return *d_output_batch; }

    /// @brief Resets LP output batch counters for the next LP invocation (default lp_stream).
    void resetOutputBatch() {
        d_output_batch->reset_counters(lp_stream);
        cudaMemsetAsync(d_output_batch->get_view().factor_offsets, 0, sizeof(uint64_t), lp_stream);
    }

    /// @brief Resets LP output batch counters on a specific stream.
    void resetOutputBatch(cudaStream_t stream) {
        d_output_batch->reset_counters(stream);
        cudaMemsetAsync(d_output_batch->get_view().factor_offsets, 0, sizeof(uint64_t), stream);
    }

    /// @brief GPU-side append: copies LP output batch → persistent batch.
    /// Launches device_append_kernel on the given stream (typically postproc_stream).
    /// Also resets the output batch counters on the same stream.
    /// No CPU sync or host-side counter reads.
    void launchDeviceAppend(mpqs::structures::RelationBatch* persistent, cudaStream_t stream);

    /// @brief Download all stored witness partials to host SoA.
    /// Call after sieve loop completes, before matrix construction.
    void moveWitnessesToHost(mpqs::structures::HostRelationBatch& dest, cudaStream_t stream);

    /// @brief Download all captured combined-relation provenance records to host.
    /// No-op (clears dest) unless config.capture_provenance was set at initiate().
    /// Call after the sieve loop completes, before serialization. STRICTLY ADDITIVE.
    void moveProvenanceToHost(std::vector<CombineProvenanceEntry>& dest, cudaStream_t stream);

private:
    LargePrimeConfig config;
    cudaStream_t lp_stream;
    mpqs::uint512 N;

    // --- Pipeline Buffers ---

    /// @brief Capacity of the current pipeline buffers (in number of elements)
    size_t pipeline_capacity_ = 0;

    /// @brief Pre-allocated pipeline capacity (fixed at init, no runtime resize)
    size_t max_pipeline_capacity_ = 0;

    /// @brief Packed routing keys: [Status(4) | RowIdx(28) | Tag(32)]
    uint64_t* d_routing_keys = nullptr;

    /**
     * @brief Dynamically manages allocation for pipeline arrays ensuring minimal reallocation.
     * @param num_items Number of 1-partials coming in the current batch.
     */
    void resizePipelineBuffers(size_t num_items);

    // --- Pipeline Status & Dual Counters ---
    
    /// @brief Status flag for each element in the batch
    SLPStatus* d_status_flags = nullptr;

    /// @brief Dual Counter for lock-free consistent reservation in the Output Batch CSR arrays
    /// Packed as: [uint32_t relation_count | uint32_t factor_count]
    uint64_t* d_output_dual_counter = nullptr;

    /// @brief Dual Counter for lock-free consistent reservation in the Global Witness CSR arrays
    /// Packed as: [uint32_t relation_count | uint32_t factor_count]
    uint64_t* d_witness_dual_counter = nullptr;

    // --- Match Tracking Buffers ---
    
    /// @brief Target index array. When a thread successfully matches a witness in the 
    /// Hash Table, it records the Global Witness Index here.
    uint32_t* d_target_idx_array = nullptr;
  
    // --- Core 2-Stage Hash Table Structures (Device) ---
    
    /// @brief Directory Array. Size: 2^hash_bits * sizeof(uint64_t).
    /// Bit Layout: [Lock: 1 bit | Count: 15 bits | Row Index: 48 bits]
    uint64_t* d_directory = nullptr;

    /// @brief Compact Payload Array. Size: max_witness_capacity * ROW_WIDTH_ELEMS * sizeof(uint64_t).
    /// Bit Layout: [Tag: 32 bits | Global Witness Index: 32 bits]
    uint64_t* d_payload_slabs = nullptr;

    /// @brief Allocator monotonically issues Row Indexes to buckets when they transition from Count 0 to 1.
    uint32_t* d_global_row_allocator = nullptr;

    /// @brief Maximum slab rows allocated = min(max_witness_capacity, 2^hash_bits).
    uint32_t max_slab_rows_ = 0;

    // --- SoA Appending Buffers ---

    /// @brief Long-term storage for unique witnesses. Data is strictly appended, never moved.
    std::unique_ptr<mpqs::structures::RelationBatch> d_global_witnesses;

    /// @brief Stores combined Full Relations found in the current processing batch.
    std::unique_ptr<mpqs::structures::RelationBatch> d_output_batch;

    // --- Overflow tracking (device atomics, cumulative, never reset) ---
    uint64_t* d_slab_overflow_count = nullptr;
    uint64_t* d_witness_overflow_count = nullptr;
    uint64_t* d_output_overflow_count = nullptr;

    // --- Non-Blocking Telemetry ---
    SLPPinnedStats* h_pinned_stats = nullptr; ///< Host-visible mapped memory
    SLPPinnedStats* d_pinned_stats = nullptr; ///< Device pointer to the same mapped memory
    /// Tracks the witness count from the previous batch to compute deltas
    uint64_t last_witness_count_ = 0;
    /// Cumulative full relations produced across all LP batches
    uint64_t cumulative_full_relations_ = 0;

    // --- Pinned counters for zero-sync LP output readback (Stage 1) ---
    uint64_t* h_pinned_lp_combined_count = nullptr;  // Host-visible: [0]=rels, [1]=factors
    uint64_t* d_pinned_lp_combined_count = nullptr;   // Device alias of above

    // --- Pinned counter for device_append_kernel telemetry (Stage 4) ---
    uint64_t* h_pinned_appended_count = nullptr;  // Host-visible: relations appended
    uint64_t* d_pinned_appended_count = nullptr;   // Device alias of above

    // --- CUDA Events for async signaling (Stage 2/3) ---
    cudaEvent_t partials_ready_event_ = nullptr;  ///< Postproc → LP: partial batch is ready
    cudaEvent_t lp_done_event_ = nullptr;         ///< LP → Orchestrator: processing complete
    cudaEvent_t count_snapshot_event_ = nullptr;   ///< LP → Orchestrator: input count captured

    // --- Device-side input count for async path (Stage 3) ---
    uint64_t* d_lp_input_count_ = nullptr;  ///< 1-element device buffer holding snapshotted count

    /// @brief Saved pointer to persistent batch for deferred append (async path, Stage 3)
    mpqs::structures::RelationBatch* pending_persistent_ = nullptr;

    // --- Combine-provenance capture (DIAGNOSTIC, allocated only when enabled) ---
    /// @brief Run-lifetime device buffer of captured CombineProvenanceEntry records.
    /// Appended to (across all batches) by capture_provenance_kernel. nullptr unless
    /// config.capture_provenance. Never touched on the normal path.
    CombineProvenanceEntry* d_provenance_ = nullptr;
    /// @brief Device atomic append counter for d_provenance_ (number of records).
    uint64_t* d_provenance_count_ = nullptr;
    /// @brief Capacity (in records) of d_provenance_.
    uint64_t  provenance_capacity_ = 0;
    /// @brief Device overflow counter: combined relations dropped because buffer full.
    uint64_t* d_provenance_overflow_ = nullptr;
    /// @brief Launches capture_provenance_kernel on lp_stream (gated on capture_provenance).
    void captureProvenance(const mpqs::structures::RelationBatchView& input_view,
                           const mpqs::structures::RelationBatchView& global_witness_view,
                           uint32_t grid_size, uint32_t block_size);

#ifdef LP_DEBUG
    uint32_t lp_call_counter_ = 0;
#endif
};

} // namespace lp
} // namespace mpqs
