// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

/// @file cpu_lp.h
/// @brief CPU-side single large prime table for cluster mode.
/// Replaces GPU LargePrimeVariant on the coordinator to avoid blocking GPU sieve.

#include "mpqs_soa.h"
#include "montgomery.cuh"
#include "uint512.cuh"
#include "common.h"

#include <unordered_map>
#include <vector>
#include <cstdint>

namespace mpqs::cluster {

class RelationAccumulator;  // forward decl (from accumulator.h)

/// A stored 1-partial relation, extracted from HostRelationBatch.
struct PartialRelation {
    mpqs::uint512    sqrt_Q;
    uint8_t          sign;
    int32_t          val_2_exp;
    unsigned __int128 large_prime;
    std::vector<uint32_t> factor_indices;
    std::vector<uint8_t>  factor_counts;
    /// Stage 4: branch char vector of this raw partial (0 in norm mode). Carried so
    /// Stage 5 can XOR-combine it with the matched partner; Stage 4 only preserves it.
    uint32_t          char_bits = 0;
};

/// CPU hash table for single large prime matching.
/// NOT thread-safe — designed to be owned by Thread A exclusively.
class CPULargePrimeTable {
public:
    /// @param lp1_bound  Large prime bound (LP values < lp1_bound < 2^40)
    /// @param fdata      Factor base data (for N, used in Montgomery context)
    CPULargePrimeTable(uint64_t lp1_bound, const mpqs::sieve::factoringData& fdata);

    /// Process a batch of 1-partials: insert new ones, combine matches into full relations.
    /// Combined relations are added to the accumulator.
    void insertAndMatch(const mpqs::structures::HostRelationBatch& partials,
                        RelationAccumulator& accumulator);

    uint64_t witnesses() const { return table_.size(); }
    uint64_t totalInserts() const { return total_inserts_; }
    uint64_t totalMatches() const { return total_matches_; }
    uint64_t totalCombines() const { return total_combines_; }
    /// Number of matches dropped because the two partials were byte-identical
    /// (same sqrt_Q) — combining them would yield a perfect square (X==Y) and a
    /// trivial sqrt. Diagnostic for the cross-node duplicate-partial pathology.
    uint64_t totalDupDropped() const { return total_dup_dropped_; }

private:
    /// Combine two partials with matching LP into a full relation.
    bool combinePartials(const PartialRelation& stored,
                         const mpqs::structures::HostRelationBatch& batch, size_t idx,
                         mpqs::structures::HostRelationBatch& output);

    /// Extract a single partial from a HostRelationBatch at index idx.
    PartialRelation extractPartial(const mpqs::structures::HostRelationBatch& batch, size_t idx);

    /// Merge two sorted factor lists, summing exponents for shared indices.
    static void mergeFactors(const std::vector<uint32_t>& a_idx, const std::vector<uint8_t>& a_cnt,
                             const std::vector<uint32_t>& b_idx, const std::vector<uint8_t>& b_cnt,
                             std::vector<uint32_t>& out_idx, std::vector<uint8_t>& out_cnt);

    uint64_t lp1_bound_;
    mpqs::math::Montgomery mont_;

    /// Hash table: LP value (64-bit) -> stored partial.
    std::unordered_map<uint64_t, PartialRelation> table_;

    uint64_t total_inserts_  = 0;
    uint64_t total_matches_  = 0;
    uint64_t total_combines_ = 0;
    uint64_t total_dup_dropped_ = 0;
};

} // namespace mpqs::cluster
