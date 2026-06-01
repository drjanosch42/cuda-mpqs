// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace mpqs::autotune {

/// Compute SHA-256 hex digest of a string. Self-contained, no external dependency.
/// FIPS 180-4 compliant.
std::string sha256_hex(const std::string& input);

/// Current UTC timestamp in ISO 8601 format (e.g., "2026-03-15T14:30:00Z").
std::string iso8601_now();

/// Single history record: optimal parameters + measured performance for one N.
struct HistoryEntry {
    std::string N_decimal;          // Full decimal representation of N
    std::string N_hash_sha256;      // SHA-256 hex digest of N_decimal (64-char hex)
    uint32_t digit_count = 0;       // Number of decimal digits
    uint32_t bit_length  = 0;       // N.bit_length() — primary index for interpolation

    struct OptimalParams {
        uint32_t fb_bound    = 0;
        uint32_t sieve_bound = 0;
        uint64_t lp1_bound   = 0;
        uint32_t kernel_params[8] = {};

        // Buffer recommendations (populated by autotune and post-sieve save)
        uint64_t recommended_witness_capacity = 0;
        uint64_t recommended_partial_buffer   = 0;
        uint64_t recommended_accum_buffer     = 0;
    } optimal_params;

    struct MeasuredPerformance {
        double   sieve_time_sec      = 0.0;
        double   total_time_sec      = 0.0;
        double   relations_per_sec   = 0.0;
        uint64_t total_relations     = 0;
        uint64_t lp_witnesses        = 0;
        uint64_t lp_combined_relations = 0;

        // Buffer telemetry (populated by post-sieve history save)
        uint64_t witness_peak       = 0;   // High-water mark of LP witness count
        uint64_t witness_capacity   = 0;   // Witness table capacity used
        double   witness_fill_pct   = 0.0; // Peak witness fill percentage
        uint64_t overflow_events    = 0;   // Total slab + witness + output overflows
        uint64_t accum_peak         = 0;   // Accumulation buffer high-water mark
        uint64_t partial_peak       = 0;   // Partial (LP staging) buffer peak
        uint64_t persistent_peak    = 0;   // Persistent buffer peak
    } measured_performance;

    struct Environment {
        std::string gpu_name;
        std::string gpu_compute_capability;  // "major.minor"
        std::string cuda_version;
    } environment;

    std::string timestamp;                    // ISO 8601 UTC
    std::vector<uint32_t> autotune_stages_run;
    double confidence = 0.0;                  // 0.0–1.0
};

/// Persistent store of autotune history entries, backed by a JSON file.
/// Entries are indexed by N_hash_sha256 and maintained sorted by bit_length.
class HistoryStore {
public:
    /// Load from JSON file. Returns false if file does not exist (store remains empty).
    /// On parse error: logs warning, returns false, store is empty.
    bool load(const std::string& path);

    /// Save to JSON file. Atomic: writes to path+".tmp", then renames.
    /// Creates parent directories if needed. Returns false on I/O error.
    bool save(const std::string& path) const;

    /// Insert or update an entry. Key = N_hash_sha256.
    /// Replace if new entry has higher confidence.
    /// If equal confidence, replace if new entry has lower total_time_sec.
    void upsert(const HistoryEntry& entry);

    /// Exact lookup by N hash. Returns nullptr if not found.
    const HistoryEntry* findExact(const std::string& N_hash) const;

    /// Find entries within [min_bits, max_bits], sorted by bit_length ascending.
    std::vector<const HistoryEntry*> findByBitRange(
        uint32_t min_bits, uint32_t max_bits) const;

    /// Find the k nearest entries by |bit_length - target|, sorted by distance.
    std::vector<const HistoryEntry*> findKNearest(
        uint32_t bit_length, size_t k) const;

    /// All entries, sorted by bit_length ascending.
    const std::vector<HistoryEntry>& entries() const;

    /// Number of entries.
    size_t size() const;

    /// True if the store was loaded from a valid file.
    bool loaded() const;

    /// Filter to entries matching the given GPU name.
    /// Non-matching entries are removed from the in-memory store (but remain in the file).
    /// Used for kernel param projection (sieve params are GPU-independent).
    void filterByGPU(const std::string& gpu_name);

private:
    std::vector<HistoryEntry> entries_;
    std::unordered_map<std::string, size_t> hash_index_;  // N_hash -> index in entries_
    bool loaded_ = false;

    /// Sort entries_ by bit_length ascending, rebuild hash_index_.
    void rebuildIndex();
};

} // namespace mpqs::autotune
