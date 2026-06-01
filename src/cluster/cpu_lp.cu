// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file cpu_lp.cu
/// @brief CPU-side single large prime table implementation.
/// Uses .cu extension for Montgomery/uint512 __host__ __device__ compatibility.

#include "cpu_lp.h"
#include "accumulator.h"
#include "hpc_logger.h"

namespace mpqs::cluster {

CPULargePrimeTable::CPULargePrimeTable(uint64_t lp1_bound,
                                        const mpqs::sieve::factoringData& fdata)
    : lp1_bound_(lp1_bound)
    , mont_(fdata.N)
{
    table_.reserve(1 << 20);  // 1M initial buckets
}

void CPULargePrimeTable::insertAndMatch(
    const mpqs::structures::HostRelationBatch& partials,
    RelationAccumulator& accumulator)
{
    if (partials.num_relations == 0) return;

    mpqs::structures::HostRelationBatch combined_output;
    combined_output.factor_offsets.push_back(0);  // CSR sentinel

    for (size_t i = 0; i < partials.num_relations; i++) {
        uint64_t lp_key = static_cast<uint64_t>(partials.large_primes[i]);
        if (lp_key == 0 || lp_key == 1) continue;

        auto it = table_.find(lp_key);
        if (it != table_.end()) {
            // MATCH — combine the two partials into a full relation
            total_matches_++;
            if (combinePartials(it->second, partials, i, combined_output)) {
                total_combines_++;
            }
            table_.erase(it);  // Each LP used at most once
        } else {
            // NEW — store for future matching
            total_inserts_++;
            table_.emplace(lp_key, extractPartial(partials, i));
        }
    }

    if (combined_output.num_relations > 0) {
        accumulator.addLPRelations(combined_output);
    }
}

bool CPULargePrimeTable::combinePartials(
    const PartialRelation& stored,
    const mpqs::structures::HostRelationBatch& batch, size_t idx,
    mpqs::structures::HostRelationBatch& output)
{
    // sqrt_Q values are in standard form — transform, multiply, reduce.
    mpqs::uint512 a_mont = mont_.transform(stored.sqrt_Q);
    mpqs::uint512 b_mont = mont_.transform(batch.sqrt_Q[idx]);
    mpqs::uint512 product_mont = mont_.mul(a_mont, b_mont);
    mpqs::uint512 combined_sqrt_Q = mont_.reduce(product_mont);

    // Sign merge: encoding-agnostic (M11c pattern, commit 19ababd). sign=1 means
    // positive; anything else means negative. Avoids the fragile uint8 mod-256
    // multiplication that previously assumed the {1, 0xFF} encoding.
    bool neg_a = (stored.sign != 1u), neg_b = (batch.signs[idx] != 1u);
    uint8_t combined_sign = (neg_a ^ neg_b) ? static_cast<uint8_t>(0xFF) : static_cast<uint8_t>(1);
    int32_t combined_v2 = stored.val_2_exp + batch.val_2_exps[idx];

    // Merge factor lists
    uint64_t fstart = batch.factor_offsets[idx];
    uint64_t fend   = batch.factor_offsets[idx + 1];
    std::vector<uint32_t> b_indices(batch.factor_indices.begin() + fstart,
                                     batch.factor_indices.begin() + fend);
    std::vector<uint8_t>  b_counts(batch.factor_counts.begin() + fstart,
                                    batch.factor_counts.begin() + fend);

    std::vector<uint32_t> merged_indices;
    std::vector<uint8_t>  merged_counts;
    mergeFactors(stored.factor_indices, stored.factor_counts,
                 b_indices, b_counts,
                 merged_indices, merged_counts);

    // Append to output batch
    output.sqrt_Q.push_back(combined_sqrt_Q);
    output.signs.push_back(combined_sign);
    output.val_2_exps.push_back(combined_v2);
    output.large_primes.push_back(stored.large_prime);  // Store LP for validator

    for (auto idx_val : merged_indices) output.factor_indices.push_back(idx_val);
    for (auto cnt_val : merged_counts)  output.factor_counts.push_back(cnt_val);
    output.num_factors += merged_indices.size();
    output.factor_offsets.push_back(output.num_factors);
    output.num_relations++;

    return true;
}

void CPULargePrimeTable::mergeFactors(
    const std::vector<uint32_t>& a_idx, const std::vector<uint8_t>& a_cnt,
    const std::vector<uint32_t>& b_idx, const std::vector<uint8_t>& b_cnt,
    std::vector<uint32_t>& out_idx, std::vector<uint8_t>& out_cnt)
{
    size_t i = 0, j = 0;
    while (i < a_idx.size() && j < b_idx.size()) {
        if (a_idx[i] < b_idx[j]) {
            out_idx.push_back(a_idx[i]);
            out_cnt.push_back(a_cnt[i]);
            i++;
        } else if (a_idx[i] > b_idx[j]) {
            out_idx.push_back(b_idx[j]);
            out_cnt.push_back(b_cnt[j]);
            j++;
        } else {
            // Same factor index — sum exponents
            out_idx.push_back(a_idx[i]);
            out_cnt.push_back(a_cnt[i] + b_cnt[j]);
            i++; j++;
        }
    }
    while (i < a_idx.size()) { out_idx.push_back(a_idx[i]); out_cnt.push_back(a_cnt[i]); i++; }
    while (j < b_idx.size()) { out_idx.push_back(b_idx[j]); out_cnt.push_back(b_cnt[j]); j++; }
}

PartialRelation CPULargePrimeTable::extractPartial(
    const mpqs::structures::HostRelationBatch& batch, size_t idx)
{
    PartialRelation p;
    p.sqrt_Q     = batch.sqrt_Q[idx];
    p.sign       = batch.signs[idx];
    p.val_2_exp  = batch.val_2_exps[idx];
    p.large_prime = batch.large_primes[idx];

    uint64_t fstart = batch.factor_offsets[idx];
    uint64_t fend   = batch.factor_offsets[idx + 1];
    p.factor_indices.assign(batch.factor_indices.begin() + fstart,
                            batch.factor_indices.begin() + fend);
    p.factor_counts.assign(batch.factor_counts.begin() + fstart,
                           batch.factor_counts.begin() + fend);
    return p;
}

} // namespace mpqs::cluster
