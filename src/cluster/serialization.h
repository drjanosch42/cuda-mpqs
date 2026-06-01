// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file serialization.h
/// @brief Binary serialization for HostRelationBatch and WORK_ASSIGN payloads.

#include "mpqs_soa.h"
#include "common.h"          // factoringData
#include "uint512.cuh"
#include <cstdint>
#include <vector>
#include <utility>

namespace mpqs::cluster {

/// Serialize first `count` relations from a HostRelationBatch to a contiguous buffer.
/// Wire layout: [num_relations:u32][num_factors:u32]
///   [sqrt_Q: N*64B][signs: N*1B][val_2_exps: N*4B]
///   [large_primes: N*16B][factor_offsets: (N+1)*8B]  <-- CSR sentinel included
///   [factor_indices: NNZ*4B][factor_counts: NNZ*1B]  <-- per-factor exponents
/// @return (buffer, byte count written).
std::pair<std::vector<uint8_t>, size_t>
serializeRelationBatch(const mpqs::structures::HostRelationBatch& batch, uint64_t count);

/// Deserialize buffer into a HostRelationBatch. Returns number of relations.
uint64_t deserializeRelationBatch(const uint8_t* data, size_t len,
                                   mpqs::structures::HostRelationBatch& out);

/// Alias: serialize partial relations (same wire format as full relations).
inline std::pair<std::vector<uint8_t>, size_t>
serializePartialBatch(const mpqs::structures::HostRelationBatch& batch, uint64_t count) {
    return serializeRelationBatch(batch, count);
}

/// Serialize WORK_ASSIGN: N + factor base + sieve params + poly range + AFactorsSnapshot (M3).
/// Wire layout: [N:64B][fb_size:u32][M:u32][F:u32]
///   [sieve_batch_size:u32][shc_dim:u8][pad:3B][threshold_override:u64]
///   [lp1_bound:u64][poly_range_start:u64][poly_range_count:u64][target_relations:u64]
///   [factorBase: fb_size*4B][rootN: fb_size*4B]
///   [snapshot_dim:u32][snapshot_a_factors: dim*4B]
///   [snapshot_lowerHalfStart:u32][snapshot_upperHalfStart:u32]  (when snapshot != nullptr)
/// @param snapshot         Optional AFactorsSnapshot (M3). Pass nullptr for M2-compatible output.
/// @return (buffer, byte count written).
std::pair<std::vector<uint8_t>, size_t>
serializeWorkAssign(const mpqs::sieve::factoringData& fdata,
                    uint32_t sieve_batch_size,
                    uint64_t threshold_override,
                    uint64_t lp1_bound,
                    uint64_t poly_range_start,
                    uint64_t poly_range_count,
                    uint64_t target_relations,
                    const mpqs::sieve::AFactorsSnapshot* snapshot = nullptr);

/// Deserialize WORK_ASSIGN payload. Populates fdata and output params.
/// @param snapshot_out         If non-null, populated from the M3 snapshot extension when present.
///                             Left unchanged if the message was sent by an M2 sender (no snapshot).
/// Backward compatible: returns true even if buffer contains no snapshot fields.
bool deserializeWorkAssign(const uint8_t* data, size_t len,
                           mpqs::sieve::factoringData& fdata,
                           uint32_t& sieve_batch_size,
                           uint64_t& threshold_override,
                           uint64_t& lp1_bound,
                           uint64_t& poly_range_start,
                           uint64_t& poly_range_count,
                           uint64_t& target_relations,
                           mpqs::sieve::AFactorsSnapshot* snapshot_out = nullptr);

/// Serialize an incremental batch (combined full + partial relations).
/// Wire layout: [full_data_len:u32][full_batch_data][partial_data_len:u32][partial_batch_data]
/// Each sub-batch uses the existing serializeRelationBatch binary format.
/// @return (buffer, total byte count)
std::pair<std::vector<uint8_t>, size_t>
serializeIncrementalBatch(const mpqs::structures::HostRelationBatch& full_batch,
                          uint64_t full_count,
                          const mpqs::structures::HostRelationBatch& partial_batch,
                          uint64_t partial_count);

/// Deserialize an INCREMENTAL_BATCH payload.
/// @param full_out   Receives full relations.
/// @param partial_out  Receives 1-partial relations.
/// @return true on success, false if buffer is malformed.
bool deserializeIncrementalBatch(const uint8_t* data, size_t len,
                                 mpqs::structures::HostRelationBatch& full_out,
                                 mpqs::structures::HostRelationBatch& partial_out);

/// Merge N HostRelationBatches into a single batch by appending SoA vectors.
/// CSR factor_offsets are fixed up with cumulative shift.
/// @param batches   Pointers to source batches.
/// @param counts    Per-batch relation counts (how many relations to take from each).
/// @return Merged batch with valid CSR structure.
structures::HostRelationBatch mergeRelationBatches(
    const std::vector<const structures::HostRelationBatch*>& batches,
    const std::vector<uint64_t>& counts);

} // namespace mpqs::cluster
