// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.

/// @file serialization.cpp
/// @brief Binary serialization for HostRelationBatch and WORK_ASSIGN.

#include "serialization.h"
#include <cstring>
#include <stdexcept>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "Serialization assumes little-endian byte order");

namespace mpqs::cluster {

static_assert(sizeof(mpqs::uint512) == 64, "uint512 must be 64 bytes");

// =============================================================================
// Helpers
// =============================================================================

static void writeBytes(std::vector<uint8_t>& buf, size_t& offset,
                       const void* src, size_t len) {
    memcpy(buf.data() + offset, src, len);
    offset += len;
}

/// Bounds-checked reader for deserialization (defensive, per communication audit §2.5).
struct SafeReader {
    const uint8_t* data;
    size_t total;
    size_t pos = 0;
    bool ok = true;

    bool readBytes(void* dst, size_t n) {
        if (pos + n > total) { ok = false; return false; }
        std::memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }
};

// =============================================================================
// Relation Batch
// =============================================================================

std::pair<std::vector<uint8_t>, size_t>
serializeRelationBatch(const mpqs::structures::HostRelationBatch& batch, uint64_t count) {
    // HostRelationBatch CSR layout:
    //   factor_offsets: (n_rels + 1) entries — CSR row pointers with sentinel
    //   factor_indices: n_factors entries — column indices
    //   factor_counts:  n_factors entries — per-factor exponent counts
    //
    // Total factors for first `count` relations = factor_offsets[count] (sentinel)
    uint64_t num_factors = 0;
    if (count > 0 && batch.factor_offsets.size() > count) {
        num_factors = batch.factor_offsets[count];
    }

    // Total: 8 (header) + count*(64+1+4+16) + (count+1)*8 + num_factors*(4+1)
    size_t total = 8
        + count * (sizeof(mpqs::uint512) + 1 + 4 + 16)  // sqrt_Q, signs, val_2_exps, large_primes
        + (count + 1) * sizeof(uint64_t)                  // factor_offsets (CSR with sentinel)
        + num_factors * sizeof(uint32_t)                   // factor_indices
        + num_factors * sizeof(uint8_t);                   // factor_counts (per-factor exponents)

    std::vector<uint8_t> buf(total);
    size_t offset = 0;

    uint32_t n_rels = static_cast<uint32_t>(count);
    uint32_t n_facs = static_cast<uint32_t>(num_factors);
    writeBytes(buf, offset, &n_rels, 4);
    writeBytes(buf, offset, &n_facs, 4);

    // SoA arrays
    if (count > 0) {
        writeBytes(buf, offset, batch.sqrt_Q.data(), count * sizeof(mpqs::uint512));
        writeBytes(buf, offset, batch.signs.data(), count * sizeof(uint8_t));
        writeBytes(buf, offset, batch.val_2_exps.data(), count * sizeof(int32_t));
        writeBytes(buf, offset, batch.large_primes.data(), count * sizeof(unsigned __int128));
        // CSR: write (count + 1) factor_offsets including sentinel
        writeBytes(buf, offset, batch.factor_offsets.data(), (count + 1) * sizeof(uint64_t));
        if (num_factors > 0) {
            writeBytes(buf, offset, batch.factor_indices.data(), num_factors * sizeof(uint32_t));
            writeBytes(buf, offset, batch.factor_counts.data(), num_factors * sizeof(uint8_t));
        }
    }

    return {std::move(buf), offset};
}

uint64_t deserializeRelationBatch(const uint8_t* data, size_t len,
                                   mpqs::structures::HostRelationBatch& out) {
    if (len < 8) return 0;

    SafeReader r{data, len};
    uint32_t n_rels = 0, n_facs = 0;
    r.readBytes(&n_rels, 4);
    r.readBytes(&n_facs, 4);
    if (!r.ok) return 0;

    out.resize(n_rels, n_facs);

    if (n_rels > 0) {
        r.readBytes(out.sqrt_Q.data(), n_rels * sizeof(mpqs::uint512));
        r.readBytes(out.signs.data(), n_rels * sizeof(uint8_t));
        r.readBytes(out.val_2_exps.data(), n_rels * sizeof(int32_t));
        r.readBytes(out.large_primes.data(), n_rels * sizeof(unsigned __int128));
        r.readBytes(out.factor_offsets.data(), (n_rels + 1) * sizeof(uint64_t));
        if (n_facs > 0) {
            r.readBytes(out.factor_indices.data(), n_facs * sizeof(uint32_t));
            r.readBytes(out.factor_counts.data(), n_facs * sizeof(uint8_t));
        }
        if (!r.ok) return 0;
    }

    out.num_relations = n_rels;
    out.num_factors = n_facs;
    return n_rels;
}

// =============================================================================
// Work Assign
// =============================================================================

std::pair<std::vector<uint8_t>, size_t>
serializeWorkAssign(const mpqs::sieve::factoringData& fdata,
                    uint32_t sieve_batch_size,
                    uint64_t threshold_override,
                    uint64_t lp1_bound,
                    uint64_t poly_range_start,
                    uint64_t poly_range_count,
                    uint64_t target_relations,
                    const mpqs::sieve::AFactorsSnapshot* snapshot) {
    uint32_t fb_size = static_cast<uint32_t>(fdata.factorBase.size());

    // Fixed header: 64(N) + 4(fb_size) + 4(M) + 4(F) + 4(sieve_batch_size)
    //             + 1(shc_dim) + 3(pad) + 8(threshold) + 8(lp1_bound)
    //             + 8(start) + 8(count) + 8(target)
    //             = 124 bytes
    // Plus: fb_size*4 (factorBase) + fb_size*4 (rootN)
    // Plus (M3, optional): 4(dim) + dim*4(a_factors) + 4(lowerHalfStart) + 4(upperHalfStart)
    size_t snapshot_size = 0;
    if (snapshot) {
        snapshot_size = sizeof(uint32_t)                              // dim
                      + snapshot->shc_dim * sizeof(uint32_t)         // a_factors
                      + 2 * sizeof(uint32_t);                        // lowerHalfStart + upperHalfStart
    }
    size_t total = 124 + fb_size * 8 + snapshot_size;

    std::vector<uint8_t> buf(total);
    size_t offset = 0;

    writeBytes(buf, offset, &fdata.N, 64);
    writeBytes(buf, offset, &fb_size, 4);
    writeBytes(buf, offset, &fdata.M, 4);
    writeBytes(buf, offset, &fdata.F, 4);
    writeBytes(buf, offset, &sieve_batch_size, 4);

    uint8_t shc_dim = static_cast<uint8_t>(fdata.a_factors.size());
    writeBytes(buf, offset, &shc_dim, 1);
    uint8_t pad[3] = {0, 0, 0};
    writeBytes(buf, offset, pad, 3);

    writeBytes(buf, offset, &threshold_override, 8);
    writeBytes(buf, offset, &lp1_bound, 8);
    writeBytes(buf, offset, &poly_range_start, 8);
    writeBytes(buf, offset, &poly_range_count, 8);
    writeBytes(buf, offset, &target_relations, 8);

    if (fb_size > 0) {
        writeBytes(buf, offset, fdata.factorBase.data(), fb_size * sizeof(uint32_t));
        writeBytes(buf, offset, fdata.rootN.data(), fb_size * sizeof(uint32_t));
    }

    // AFactorsSnapshot extension (M3) — appended after all existing fields
    if (snapshot) {
        uint32_t dim = snapshot->shc_dim;
        writeBytes(buf, offset, &dim, sizeof(dim));
        if (dim > 0) {
            writeBytes(buf, offset, snapshot->a_factors.data(), dim * sizeof(uint32_t));
        }
        writeBytes(buf, offset, &snapshot->lowerHalfStart, sizeof(uint32_t));
        writeBytes(buf, offset, &snapshot->upperHalfStart, sizeof(uint32_t));
    }

    return {std::move(buf), offset};
}

bool deserializeWorkAssign(const uint8_t* data, size_t len,
                           mpqs::sieve::factoringData& fdata,
                           uint32_t& sieve_batch_size,
                           uint64_t& threshold_override,
                           uint64_t& lp1_bound,
                           uint64_t& poly_range_start,
                           uint64_t& poly_range_count,
                           uint64_t& target_relations,
                           mpqs::sieve::AFactorsSnapshot* snapshot_out) {
    if (len < 124) return false;

    SafeReader r{data, len};
    uint32_t fb_size = 0;

    r.readBytes(&fdata.N, 64);
    r.readBytes(&fb_size, 4);
    r.readBytes(&fdata.M, 4);
    r.readBytes(&fdata.F, 4);
    r.readBytes(&sieve_batch_size, 4);

    uint8_t shc_dim = 0;
    r.readBytes(&shc_dim, 1);
    r.pos += 3; // skip pad

    r.readBytes(&threshold_override, 8);
    r.readBytes(&lp1_bound, 8);
    r.readBytes(&poly_range_start, 8);
    r.readBytes(&poly_range_count, 8);
    r.readBytes(&target_relations, 8);

    if (!r.ok) return false;

    fdata.size = fb_size;
    fdata.factorBase.resize(fb_size);
    fdata.rootN.resize(fb_size);

    if (fb_size > 0) {
        r.readBytes(fdata.factorBase.data(), fb_size * sizeof(uint32_t));
        r.readBytes(fdata.rootN.data(), fb_size * sizeof(uint32_t));
        if (!r.ok) return false;
    }

    // AFactorsSnapshot extension (M3) — optional, backward compatible.
    // A short buffer (M2-era sender without snapshot) is silently accepted.
    if (snapshot_out && r.pos < r.total) {
        uint32_t dim = 0;
        if (!r.readBytes(&dim, sizeof(dim))) return true;
        snapshot_out->shc_dim = dim;
        if (dim > 0) {
            snapshot_out->a_factors.resize(dim);
            if (!r.readBytes(snapshot_out->a_factors.data(), dim * sizeof(uint32_t)))
                return true;  // short buffer — base fields already populated
        }
        if (!r.readBytes(&snapshot_out->lowerHalfStart, sizeof(uint32_t))) return true;
        if (!r.readBytes(&snapshot_out->upperHalfStart, sizeof(uint32_t))) return true;
    }

    return true;
}

// =============================================================================
// Incremental Batch (INCREMENTAL_BATCH = 0x22)
// =============================================================================

std::pair<std::vector<uint8_t>, size_t>
serializeIncrementalBatch(const structures::HostRelationBatch& full_batch,
                          uint64_t full_count,
                          const structures::HostRelationBatch& partial_batch,
                          uint64_t partial_count)
{
    // Serialize each sub-batch independently
    auto [full_buf, full_len]       = serializeRelationBatch(full_batch, full_count);
    auto [partial_buf, partial_len] = serializeRelationBatch(partial_batch, partial_count);

    // Combine: [full_len:u32][full_data][partial_len:u32][partial_data]
    size_t total = 4 + full_len + 4 + partial_len;
    std::vector<uint8_t> out(total);
    size_t offset = 0;

    uint32_t fl = static_cast<uint32_t>(full_len);
    writeBytes(out, offset, &fl, 4);
    writeBytes(out, offset, full_buf.data(), full_len);

    uint32_t pl = static_cast<uint32_t>(partial_len);
    writeBytes(out, offset, &pl, 4);
    writeBytes(out, offset, partial_buf.data(), partial_len);

    return {std::move(out), total};
}

bool deserializeIncrementalBatch(const uint8_t* data, size_t len,
                                 structures::HostRelationBatch& full_out,
                                 structures::HostRelationBatch& partial_out)
{
    if (len < 8) return false;  // minimum: two u32 length prefixes

    SafeReader r{data, len};

    uint32_t full_len;
    r.readBytes(&full_len, 4);
    if (!r.ok || r.pos + full_len > r.total) return false;

    deserializeRelationBatch(data + r.pos, full_len, full_out);
    r.pos += full_len;

    uint32_t partial_len;
    r.readBytes(&partial_len, 4);
    if (!r.ok || r.pos + partial_len > r.total) return false;

    deserializeRelationBatch(data + r.pos, partial_len, partial_out);
    return true;
}

// =============================================================================
// Batch Merge
// =============================================================================

structures::HostRelationBatch mergeRelationBatches(
    const std::vector<const structures::HostRelationBatch*>& batches,
    const std::vector<uint64_t>& counts)
{
    structures::HostRelationBatch out;

    // Phase 1: compute total sizes
    uint64_t total_rels = 0;
    uint64_t total_factors = 0;
    for (size_t i = 0; i < batches.size(); ++i) {
        uint64_t n = counts[i];
        if (n == 0) continue;
        total_rels += n;
        if (batches[i]->factor_offsets.size() > n) {
            total_factors += batches[i]->factor_offsets[n];
        }
    }

    if (total_rels == 0) return out;

    // Phase 2: pre-allocate output (factor_offsets sized to total_rels+1 by resize)
    out.resize(total_rels, total_factors);

    // Phase 3: append SoA arrays and fix up CSR offsets
    uint64_t rel_cursor = 0;
    uint64_t fac_cursor = 0;

    for (size_t i = 0; i < batches.size(); ++i) {
        uint64_t n = counts[i];
        if (n == 0) continue;

        const auto* b = batches[i];
        uint64_t n_facs = (b->factor_offsets.size() > n) ? b->factor_offsets[n] : 0;

        // Flat SoA arrays: trivially copyable POD, bulk memcpy
        std::memcpy(out.sqrt_Q.data()       + rel_cursor, b->sqrt_Q.data(),       n * sizeof(mpqs::uint512));
        std::memcpy(out.signs.data()        + rel_cursor, b->signs.data(),         n * sizeof(uint8_t));
        std::memcpy(out.val_2_exps.data()   + rel_cursor, b->val_2_exps.data(),    n * sizeof(int32_t));
        std::memcpy(out.large_primes.data() + rel_cursor, b->large_primes.data(),  n * sizeof(unsigned __int128));

        // CSR factor storage
        if (n_facs > 0) {
            std::memcpy(out.factor_indices.data() + fac_cursor,
                        b->factor_indices.data(), n_facs * sizeof(uint32_t));
            std::memcpy(out.factor_counts.data()  + fac_cursor,
                        b->factor_counts.data(),  n_facs * sizeof(uint8_t));
        }

        // CSR offsets: first batch is copied verbatim (includes its own 0-sentinel);
        // subsequent batches skip their redundant 0-entry and shift all offsets by fac_cursor.
        if (rel_cursor == 0) {
            // Copy offsets[0..n] — (n+1) entries including the sentinel
            std::memcpy(out.factor_offsets.data(),
                        b->factor_offsets.data(), (n + 1) * sizeof(uint64_t));
        } else {
            // Append offsets[1..n] shifted by fac_cursor into output positions [rel_cursor+1..rel_cursor+n]
            for (uint64_t j = 1; j <= n; ++j) {
                out.factor_offsets[rel_cursor + j] = b->factor_offsets[j] + fac_cursor;
            }
        }

        rel_cursor += n;
        fac_cursor += n_facs;
    }

    // Write final CSR sentinel (also overwrites first-batch's sentinel when there is only one batch)
    out.factor_offsets[total_rels] = total_factors;
    out.num_relations = total_rels;
    out.num_factors   = total_factors;

    return out;
}

} // namespace mpqs::cluster
