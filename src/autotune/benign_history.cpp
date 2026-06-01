// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "benign_history.h"
#include "json_reader.h"
#include <fstream>
#include <sstream>
#include <cstdio>      // std::rename
#include <iomanip>
#include <algorithm>
#include <cmath>       // std::abs
#include <stdexcept>
#include <cctype>

namespace mpqs::autotune {

// ---------------------------------------------------------------------------
// loadDefaults — hardcoded seed records for ~80-digit and ~100-digit composites
// ---------------------------------------------------------------------------

void BenignHistoryStore::loadDefaults() {
    entries_.clear();

    // Small-N entries (validated by small-N campaign sweep)
    auto push = [this](uint32_t lo, uint32_t hi, uint32_t bits,
                       uint32_t fb, uint32_t sieve, uint64_t lp, double conf) {
        BenignHistoryEntry e;
        e.digit_count_lo = lo;  e.digit_count_hi = hi;
        e.bit_length = bits;    e.fb_bound = fb;
        e.sieve_bound = sieve;  e.lp1_bound = lp;
        e.confidence = conf;
        entries_.push_back(e);
    };

    //              lo   hi  bits    F        M       L     conf
    push(           27,  31,   96,   25000,   16384,  0,    0.5);
    push(           37,  41,  129,  100000,  131072,  0,    0.5);
    push(           47,  51,  162,  120000,  131072,  0,    0.5);
    push(           57,  61,  195,  200000,  262144,  0,    0.5);
    push(           62,  66,  212,  500000,  262144,  0,    0.5);   // NEW: closes 62-66d gap
    push(           67,  71,  228,  300000,  262144,  0,    1.0);   // VALIDATED: L=0 sweep, real core 2.01s
    push(           72,  76,  245,  700000,   65536,  0,    1.0);   // VALIDATED: interpolated from 75d/80d L=0 data
    push(           77,  81,  261,  700000,   65536,  0,    1.0);   // VALIDATED: L=0 sweep, real core 5.44s
    push(           82,  86,  276, 1500000,  262144,  0,    1.0);   // VALIDATED: L=0 sweep, real core 23.9s
    push(           87,  95,  300, 3000000,  262144,  300000000, 1.0);   // VALIDATED: LP=300M safe, 11.4% speedup (36.75s)
    push(           95, 105,  332, 7000000,  262144,  1000000000000ULL, 1.0);   // VALIDATED: F=7M,L=1T, full pipeline 2m0.7s (RTX 5070 Ti, March 17 benchmark)
    entries_.back().recommended_witness_capacity = 4194304;  // 4M witnesses for RSA-100 scale

    push(          105, 115,  364, 9000000,  262144,  1000000000000ULL, 1.0);   // VALIDATED: F=9M,L=1T, full pipeline 1048s (RTX 5070 Ti, March 25 sweep)
    entries_.back().recommended_witness_capacity = 4194304;  // 4M witnesses; L>1T causes sqrt failures regardless of hash sizing

    // --- Jetson Orin Nano (SM 8.7) estimates ---
    // Extrapolated from desktop params with reduced LP bounds for 8 GB / 1 MB L2.
    // Sieve batch_size (8) and other Jetson-specific params handled by runtime detection (P1).
    // conf=0.5 ensures desktop entries (conf=1.0) take priority on RTX.
    push(           67,  71,  233,  300000,  262144,              0ULL, 0.5);   // Jetson 70d: no LP
    push(           77,  81,  266,  700000,   65536,              0ULL, 0.5);   // Jetson 80d: no LP
    push(           82,  86,  282, 1500000,  262144,              0ULL, 0.5);   // Jetson 85d: no LP
    push(           87,  95,  299, 3000000,  262144,      100000000ULL, 0.5);   // Jetson 90d: LP=100M
    push(           95, 105,  332, 7000000,  262144,      500000000ULL, 0.5);   // Jetson RSA-100: LP=500M
    entries_.back().recommended_witness_capacity = 4194304;  // 4M witnesses
}

// ---------------------------------------------------------------------------
// Lookup methods
// ---------------------------------------------------------------------------

// findByDigits performs a LINEAR SCAN and returns the FIRST entry whose
// [digit_count_lo, digit_count_hi] range contains `digit_count`.
//
// LOAD-BEARING ORDERING: desktop / default entries (confidence = 1.0) MUST
// appear before Jetson entries (confidence = 0.5) in `entries_` so that
// RTX-class hosts hit the high-confidence record first. The
// `loadDefaults()` seed list above is ordered accordingly, and any new
// architecture-specific entries must be appended AFTER the desktop block.
//
// Callers (auto_apply.cpp::mergeSieveParams / mergeBufferParams) assume
// this ordering and do not perform any subsequent confidence-based
// preference; that disambiguation is delegated to the seed-order
// invariant.
const BenignHistoryEntry* BenignHistoryStore::findByDigits(uint32_t digit_count) const {
    for (const auto& e : entries_) {
        if (digit_count >= e.digit_count_lo && digit_count <= e.digit_count_hi)
            return &e;
    }
    return nullptr;
}

const BenignHistoryEntry* BenignHistoryStore::findByBits(uint32_t bit_length) const {
    if (entries_.empty()) return nullptr;

    const BenignHistoryEntry* best = nullptr;
    uint32_t best_dist = UINT32_MAX;
    for (const auto& e : entries_) {
        uint32_t dist = (bit_length >= e.bit_length)
                            ? (bit_length - e.bit_length)
                            : (e.bit_length - bit_length);
        if (dist < best_dist) {
            best_dist = dist;
            best = &e;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Container methods
// ---------------------------------------------------------------------------

const std::vector<BenignHistoryEntry>& BenignHistoryStore::entries() const {
    return entries_;
}

size_t BenignHistoryStore::size() const {
    return entries_.size();
}

void BenignHistoryStore::upsert(const BenignHistoryEntry& entry) {
    // Linear scan for overlap: any existing entry whose digit range overlaps
    // [entry.digit_count_lo, entry.digit_count_hi]
    for (auto& e : entries_) {
        if (e.digit_count_lo <= entry.digit_count_hi &&
            entry.digit_count_lo <= e.digit_count_hi) {
            // Keep validated (higher-confidence) entries from being overwritten
            if (e.confidence > entry.confidence) return;
            e = entry;  // Replace
            return;
        }
    }
    entries_.push_back(entry);
}

// ---------------------------------------------------------------------------
// JSON serializer — pretty-print with 2-space indentation.
// Follows the same pattern as autotune_history.cpp::serializeHistory().
// ---------------------------------------------------------------------------

static std::string serializeBenignHistory(const std::vector<BenignHistoryEntry>& entries) {
    std::ostringstream os;
    os << std::setprecision(10);

    os << "{\n";
    os << "  \"version\": 1,\n";
    os << "  \"generator\": \"cuda-mpqs-benign\",\n";
    os << "  \"entries\": [";

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (i > 0) os << ",";
        os << "\n    {\n";
        os << "      \"digit_count_lo\": " << e.digit_count_lo << ",\n";
        os << "      \"digit_count_hi\": " << e.digit_count_hi << ",\n";
        os << "      \"bit_length\": " << e.bit_length << ",\n";
        os << "      \"fb_bound\": " << e.fb_bound << ",\n";
        os << "      \"sieve_bound\": " << e.sieve_bound << ",\n";
        os << "      \"lp1_bound\": " << e.lp1_bound << ",\n";
        os << "      \"confidence\": " << e.confidence << ",\n";
        os << "      \"recommended_witness_capacity\": " << e.recommended_witness_capacity << ",\n";
        os << "      \"recommended_partial_buffer\": " << e.recommended_partial_buffer << "\n";
        os << "    }";
    }

    if (!entries.empty()) os << "\n  ";
    os << "]\n";
    os << "}\n";

    return os.str();
}

bool BenignHistoryStore::save(const std::string& path) const {
    std::string json = serializeBenignHistory(entries_);
    std::string tmp_path = path + ".tmp";

    {
        std::ofstream ofs(tmp_path);
        if (!ofs.is_open()) return false;
        ofs << json;
        if (!ofs.good()) return false;
    }

    if (std::rename(tmp_path.c_str(), path.c_str()) != 0)
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// JSON parser — uses the shared JsonReader from json_reader.h.
// Forward-compatible: unknown keys are silently skipped.
// Missing numeric fields default to 0.
// ---------------------------------------------------------------------------

bool BenignHistoryStore::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    try {
        JsonReader reader{content, 0};
        reader.expect('{');
        reader.skipWS();

        entries_.clear();

        if (reader.peek() != '}') {
            while (true) {
                std::string key = reader.parseString();
                reader.expect(':');

                if (key == "entries") {
                    reader.expect('[');
                    reader.skipWS();
                    if (reader.peek() != ']') {
                        while (true) {
                            entries_.push_back(reader.parseBenignEntry());
                            reader.skipWS();
                            if (reader.peek() == ',') { ++reader.pos; reader.skipWS(); }
                            else break;
                        }
                    }
                    reader.expect(']');
                } else {
                    reader.skipValue();
                }

                reader.skipWS();
                if (reader.peek() == ',') { ++reader.pos; reader.skipWS(); }
                else break;
            }
        }
        reader.expect('}');
    } catch (const std::exception&) {
        entries_.clear();
        return false;
    }

    return true;
}

} // namespace mpqs::autotune
