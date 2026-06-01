// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace mpqs::autotune {

/// Architecture-independent parameter record, keyed by digit range.
/// Does NOT contain kernel_params or gpu_name — these are GPU-specific.
struct BenignHistoryEntry {
    uint32_t digit_count_lo = 0;   // Digit range lower bound (inclusive)
    uint32_t digit_count_hi = 0;   // Digit range upper bound (inclusive)
    uint32_t bit_length     = 0;   // Representative bit-length (midpoint of range)
    uint32_t fb_bound       = 0;   // Recommended factor base bound (F)
    uint32_t sieve_bound    = 0;   // Recommended sieve interval half-width (M)
    uint64_t lp1_bound      = 0;   // Recommended LP bound (L)
    double   confidence     = 0.5; // Lower than local (typ. 0.3–0.5)

    // Buffer recommendations (initially zero, populated by tile 7.7/7.10)
    uint64_t recommended_witness_capacity = 0;  // 0 = use default
    uint64_t recommended_partial_buffer   = 0;  // 0 = use default
};

/// Persistent store of benign (cross-GPU) history entries.
/// Backed by a JSON file separate from the per-GPU autotune_history.json.
class BenignHistoryStore {
public:
    /// Load from JSON file. Returns false if file missing or parse error.
    bool load(const std::string& path);

    /// Save to JSON file (atomic: write to .tmp, rename).
    bool save(const std::string& path) const;

    /// Find the best-matching entry for the given digit count.
    /// Returns nullptr if no entry's [digit_count_lo, digit_count_hi] contains digit_count.
    const BenignHistoryEntry* findByDigits(uint32_t digit_count) const;

    /// Find the best-matching entry for the given bit length.
    /// Returns nullptr if no entry matches (matches by closest bit_length).
    const BenignHistoryEntry* findByBits(uint32_t bit_length) const;

    /// All entries.
    const std::vector<BenignHistoryEntry>& entries() const;

    /// Insert or update an entry for the given digit range.
    /// Key = [digit_count_lo, digit_count_hi]. Replaces if range overlaps.
    void upsert(const BenignHistoryEntry& entry);

    /// Number of entries.
    size_t size() const;

    /// Populate with hardcoded default records for ~80-digit and ~100-digit composites.
    void loadDefaults();

private:
    std::vector<BenignHistoryEntry> entries_;
};

} // namespace mpqs::autotune
