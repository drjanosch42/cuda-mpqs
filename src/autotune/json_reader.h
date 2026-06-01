// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#pragma once

#include "autotune_history.h"
#include "benign_history.h"

#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mpqs::autotune {

/// Lightweight cursor-based recursive-descent JSON reader.
///
/// Shared by `HistoryStore::load` (per-GPU autotune history) and
/// `BenignHistoryStore::load` (architecture-independent benign history).
///
/// Forward-compatible: unknown keys are silently skipped via `skipValue()`.
/// Missing numeric fields default to 0 (handled by callers).
///
/// Error reporting: throws `std::runtime_error` with an "JSON: ..." prefix.
struct JsonReader {
    const std::string& data;
    size_t pos = 0;

    void skipWS() {
        while (pos < data.size() && (data[pos] == ' ' || data[pos] == '\t' ||
               data[pos] == '\n' || data[pos] == '\r'))
            ++pos;
    }

    char peek() {
        skipWS();
        if (pos >= data.size()) throw std::runtime_error("JSON: unexpected end of input");
        return data[pos];
    }

    void expect(char c) {
        skipWS();
        if (pos >= data.size() || data[pos] != c)
            throw std::runtime_error(std::string("JSON: expected '") + c +
                                     "', got '" + (pos < data.size() ? data[pos] : '?') + "'");
        ++pos;
    }

    std::string parseString() {
        skipWS();
        expect('"');
        std::string result;
        while (pos < data.size() && data[pos] != '"') {
            if (data[pos] == '\\') {
                ++pos;
                if (pos >= data.size()) throw std::runtime_error("JSON: unterminated escape");
                switch (data[pos]) {
                    case '"':  result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/':  result.push_back('/'); break;
                    case 'n':  result.push_back('\n'); break;
                    case 'r':  result.push_back('\r'); break;
                    case 't':  result.push_back('\t'); break;
                    default:   result.push_back(data[pos]); break;
                }
            } else {
                result.push_back(data[pos]);
            }
            ++pos;
        }
        if (pos >= data.size()) throw std::runtime_error("JSON: unterminated string");
        ++pos;  // consume closing '"'
        return result;
    }

    /// Parse a JSON number as double. Handles integers and floats.
    double parseDouble() {
        skipWS();
        size_t start = pos;
        if (pos < data.size() && data[pos] == '-') ++pos;
        while (pos < data.size() && std::isdigit(static_cast<unsigned char>(data[pos]))) ++pos;
        if (pos < data.size() && data[pos] == '.') {
            ++pos;
            while (pos < data.size() && std::isdigit(static_cast<unsigned char>(data[pos]))) ++pos;
        }
        if (pos < data.size() && (data[pos] == 'e' || data[pos] == 'E')) {
            ++pos;
            if (pos < data.size() && (data[pos] == '+' || data[pos] == '-')) ++pos;
            while (pos < data.size() && std::isdigit(static_cast<unsigned char>(data[pos]))) ++pos;
        }
        if (pos == start) throw std::runtime_error("JSON: expected number");
        return std::stod(data.substr(start, pos - start));
    }

    /// Parse a JSON integer as uint64_t.
    uint64_t parseUint64() {
        skipWS();
        uint64_t val = 0;
        if (pos >= data.size() || !std::isdigit(static_cast<unsigned char>(data[pos])))
            throw std::runtime_error("JSON: expected unsigned integer");
        while (pos < data.size() && std::isdigit(static_cast<unsigned char>(data[pos]))) {
            val = val * 10 + static_cast<uint64_t>(data[pos] - '0');
            ++pos;
        }
        // Skip fractional part if present (e.g., "60000.0" written by some serializers)
        if (pos < data.size() && data[pos] == '.') {
            ++pos;
            while (pos < data.size() && std::isdigit(static_cast<unsigned char>(data[pos]))) ++pos;
        }
        return val;
    }

    /// Skip any JSON value recursively (for unknown fields).
    void skipValue() {
        skipWS();
        if (pos >= data.size()) throw std::runtime_error("JSON: unexpected end of input");
        char c = data[pos];
        if (c == '"') {
            parseString();
        } else if (c == '{') {
            ++pos;
            skipWS();
            if (peek() != '}') {
                while (true) {
                    parseString();  // key
                    expect(':');
                    skipValue();    // value
                    skipWS();
                    if (peek() == ',') { ++pos; skipWS(); }
                    else break;
                }
            }
            expect('}');
        } else if (c == '[') {
            ++pos;
            skipWS();
            if (peek() != ']') {
                while (true) {
                    skipValue();
                    skipWS();
                    if (peek() == ',') { ++pos; skipWS(); }
                    else break;
                }
            }
            expect(']');
        } else if (c == 't' || c == 'f') {
            // true / false
            while (pos < data.size() && std::isalpha(static_cast<unsigned char>(data[pos]))) ++pos;
        } else if (c == 'n') {
            // null
            while (pos < data.size() && std::isalpha(static_cast<unsigned char>(data[pos]))) ++pos;
        } else {
            // number
            parseDouble();
        }
    }

    /// Parse an array of uint32_t (e.g., kernel_params, autotune_stages_run).
    std::vector<uint32_t> parseUint32Array() {
        std::vector<uint32_t> result;
        expect('[');
        skipWS();
        if (peek() != ']') {
            while (true) {
                result.push_back(static_cast<uint32_t>(parseUint64()));
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect(']');
        return result;
    }

    /// Parse HistoryEntry::OptimalParams from a JSON object.
    HistoryEntry::OptimalParams parseOptimalParams() {
        HistoryEntry::OptimalParams p{};
        expect('{');
        skipWS();
        if (peek() != '}') {
            while (true) {
                std::string key = parseString();
                expect(':');
                if (key == "fb_bound")         p.fb_bound = static_cast<uint32_t>(parseUint64());
                else if (key == "sieve_bound") p.sieve_bound = static_cast<uint32_t>(parseUint64());
                else if (key == "lp1_bound")   p.lp1_bound = parseUint64();
                else if (key == "kernel_params") {
                    auto kp = parseUint32Array();
                    for (size_t i = 0; i < 8 && i < kp.size(); ++i)
                        p.kernel_params[i] = kp[i];
                }
                else if (key == "recommended_witness_capacity") p.recommended_witness_capacity = parseUint64();
                else if (key == "recommended_partial_buffer")   p.recommended_partial_buffer = parseUint64();
                else if (key == "recommended_accum_buffer")     p.recommended_accum_buffer = parseUint64();
                else skipValue();
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect('}');
        return p;
    }

    /// Parse HistoryEntry::MeasuredPerformance from a JSON object.
    HistoryEntry::MeasuredPerformance parseMeasuredPerformance() {
        HistoryEntry::MeasuredPerformance m{};
        expect('{');
        skipWS();
        if (peek() != '}') {
            while (true) {
                std::string key = parseString();
                expect(':');
                if (key == "sieve_time_sec")             m.sieve_time_sec = parseDouble();
                else if (key == "total_time_sec")        m.total_time_sec = parseDouble();
                else if (key == "relations_per_sec")     m.relations_per_sec = parseDouble();
                else if (key == "total_relations")       m.total_relations = parseUint64();
                else if (key == "lp_witnesses")          m.lp_witnesses = parseUint64();
                else if (key == "lp_combined_relations") m.lp_combined_relations = parseUint64();
                else if (key == "witness_peak")       m.witness_peak = parseUint64();
                else if (key == "witness_capacity")   m.witness_capacity = parseUint64();
                else if (key == "witness_fill_pct")   m.witness_fill_pct = parseDouble();
                else if (key == "overflow_events")    m.overflow_events = parseUint64();
                else if (key == "accum_peak")         m.accum_peak = parseUint64();
                else if (key == "partial_peak")       m.partial_peak = parseUint64();
                else if (key == "persistent_peak")    m.persistent_peak = parseUint64();
                else skipValue();
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect('}');
        return m;
    }

    /// Parse HistoryEntry::Environment from a JSON object.
    HistoryEntry::Environment parseEnvironment() {
        HistoryEntry::Environment e;
        expect('{');
        skipWS();
        if (peek() != '}') {
            while (true) {
                std::string key = parseString();
                expect(':');
                if (key == "gpu_name")                    e.gpu_name = parseString();
                else if (key == "gpu_compute_capability") e.gpu_compute_capability = parseString();
                else if (key == "cuda_version")           e.cuda_version = parseString();
                else skipValue();
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect('}');
        return e;
    }

    /// Parse a single HistoryEntry from a JSON object (per-GPU autotune history schema).
    HistoryEntry parseEntry() {
        HistoryEntry entry;
        expect('{');
        skipWS();
        if (peek() != '}') {
            while (true) {
                std::string key = parseString();
                expect(':');
                if (key == "N_decimal")              entry.N_decimal = parseString();
                else if (key == "N_hash_sha256")     entry.N_hash_sha256 = parseString();
                else if (key == "digit_count")       entry.digit_count = static_cast<uint32_t>(parseUint64());
                else if (key == "bit_length")        entry.bit_length = static_cast<uint32_t>(parseUint64());
                else if (key == "optimal_params")    entry.optimal_params = parseOptimalParams();
                else if (key == "measured_performance") entry.measured_performance = parseMeasuredPerformance();
                else if (key == "environment")       entry.environment = parseEnvironment();
                else if (key == "timestamp")         entry.timestamp = parseString();
                else if (key == "autotune_stages_run") entry.autotune_stages_run = parseUint32Array();
                else if (key == "confidence")        entry.confidence = parseDouble();
                else skipValue();
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect('}');
        return entry;
    }

    /// Parse a single BenignHistoryEntry from a JSON object (cross-GPU benign schema).
    /// Distinct from parseEntry() because the schema is flat (no nested
    /// optimal_params/measured_performance/environment objects).
    BenignHistoryEntry parseBenignEntry() {
        BenignHistoryEntry entry;
        expect('{');
        skipWS();
        if (peek() != '}') {
            while (true) {
                std::string key = parseString();
                expect(':');
                if (key == "digit_count_lo")       entry.digit_count_lo = static_cast<uint32_t>(parseUint64());
                else if (key == "digit_count_hi")  entry.digit_count_hi = static_cast<uint32_t>(parseUint64());
                else if (key == "bit_length")      entry.bit_length = static_cast<uint32_t>(parseUint64());
                else if (key == "fb_bound")        entry.fb_bound = static_cast<uint32_t>(parseUint64());
                else if (key == "sieve_bound")     entry.sieve_bound = static_cast<uint32_t>(parseUint64());
                else if (key == "lp1_bound")       entry.lp1_bound = parseUint64();
                else if (key == "confidence")      entry.confidence = parseDouble();
                else if (key == "recommended_witness_capacity") entry.recommended_witness_capacity = parseUint64();
                else if (key == "recommended_partial_buffer")   entry.recommended_partial_buffer = parseUint64();
                else skipValue();
                skipWS();
                if (peek() == ',') { ++pos; skipWS(); }
                else break;
            }
        }
        expect('}');
        return entry;
    }
};

} // namespace mpqs::autotune
