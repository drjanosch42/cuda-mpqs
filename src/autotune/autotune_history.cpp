// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "autotune_history.h"
#include "json_reader.h"
#include "memory_estimator.h"
#include "hpc_logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mpqs::autotune {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4, self-contained)
// ---------------------------------------------------------------------------
namespace {

/// 64 round constants: first 32 bits of the fractional parts of the cube
/// roots of the first 64 primes (2..311).
static constexpr uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t Sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
inline uint32_t Sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
inline uint32_t sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
inline uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

/// Compute SHA-256 digest, returning 32 raw bytes.
void sha256_raw(const uint8_t* data, size_t len, uint8_t out[32]) {
    // Initial hash values: first 32 bits of the fractional parts of the
    // square roots of the first 8 primes (2..19).
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Pre-processing: padding
    // Message length in bits (64-bit big-endian)
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;

    // Padded message: original + 0x80 + zeros + 8-byte length
    // Total length must be multiple of 64 bytes
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> msg(padded_len, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    // Append length as big-endian 64-bit
    for (int i = 0; i < 8; ++i)
        msg[padded_len - 1 - i] = static_cast<uint8_t>(bit_len >> (i * 8));

    // Process each 512-bit (64-byte) block
    for (size_t offset = 0; offset < padded_len; offset += 64) {
        uint32_t W[64];
        // Prepare message schedule
        for (int t = 0; t < 16; ++t) {
            W[t] = (static_cast<uint32_t>(msg[offset + t * 4 + 0]) << 24) |
                   (static_cast<uint32_t>(msg[offset + t * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[offset + t * 4 + 2]) <<  8) |
                   (static_cast<uint32_t>(msg[offset + t * 4 + 3]));
        }
        for (int t = 16; t < 64; ++t)
            W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16];

        // Initialize working variables
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

        // 64 rounds of compression
        for (int t = 0; t < 64; ++t) {
            uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K256[t] + W[t];
            uint32_t T2 = Sigma0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }

        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    // Produce 32-byte digest (big-endian)
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>(H[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(H[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(H[i] >>  8);
        out[i * 4 + 3] = static_cast<uint8_t>(H[i]);
    }
}

} // anonymous namespace

std::string sha256_hex(const std::string& input) {
    uint8_t digest[32];
    sha256_raw(reinterpret_cast<const uint8_t*>(input.data()), input.size(), digest);
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (int i = 0; i < 32; ++i) {
        result.push_back(hex_chars[digest[i] >> 4]);
        result.push_back(hex_chars[digest[i] & 0x0F]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// ISO 8601 UTC timestamp
// ---------------------------------------------------------------------------

std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&time_t_now, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return buf;
}

// ---------------------------------------------------------------------------
// Minimal JSON parser — handles the fixed history schema only.
// Forward-compatible: skips unknown fields.
// JsonReader is now defined in json_reader.h, shared with benign_history.cpp.
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// JSON serializer — pretty-print with 2-space indentation.
// ---------------------------------------------------------------------------

/// Escape a string for JSON output.
std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

/// Serialize the entire history to a JSON string.
std::string serializeHistory(const std::vector<HistoryEntry>& entries) {
    std::ostringstream os;
    os << std::setprecision(10);

    os << "{\n";
    os << "  \"version\": 1,\n";
    os << "  \"generator\": \"cuda-mpqs-autotune\",\n";
    os << "  \"entries\": [";

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (i > 0) os << ",";
        os << "\n    {\n";
        os << "      \"N_decimal\": \"" << jsonEscape(e.N_decimal) << "\",\n";
        os << "      \"N_hash_sha256\": \"" << jsonEscape(e.N_hash_sha256) << "\",\n";
        os << "      \"digit_count\": " << e.digit_count << ",\n";
        os << "      \"bit_length\": " << e.bit_length << ",\n";

        // optimal_params
        os << "      \"optimal_params\": {\n";
        os << "        \"fb_bound\": " << e.optimal_params.fb_bound << ",\n";
        os << "        \"sieve_bound\": " << e.optimal_params.sieve_bound << ",\n";
        os << "        \"lp1_bound\": " << e.optimal_params.lp1_bound << ",\n";
        os << "        \"kernel_params\": [";
        for (int k = 0; k < 8; ++k) {
            if (k > 0) os << ", ";
            os << e.optimal_params.kernel_params[k];
        }
        os << "],\n";
        os << "        \"recommended_witness_capacity\": " << e.optimal_params.recommended_witness_capacity << ",\n";
        os << "        \"recommended_partial_buffer\": " << e.optimal_params.recommended_partial_buffer << ",\n";
        os << "        \"recommended_accum_buffer\": " << e.optimal_params.recommended_accum_buffer << "\n";
        os << "      },\n";

        // measured_performance
        os << "      \"measured_performance\": {\n";
        os << "        \"sieve_time_sec\": " << e.measured_performance.sieve_time_sec << ",\n";
        os << "        \"total_time_sec\": " << e.measured_performance.total_time_sec << ",\n";
        os << "        \"relations_per_sec\": " << e.measured_performance.relations_per_sec << ",\n";
        os << "        \"total_relations\": " << e.measured_performance.total_relations << ",\n";
        os << "        \"lp_witnesses\": " << e.measured_performance.lp_witnesses << ",\n";
        os << "        \"lp_combined_relations\": " << e.measured_performance.lp_combined_relations << ",\n";
        os << "        \"witness_peak\": " << e.measured_performance.witness_peak << ",\n";
        os << "        \"witness_capacity\": " << e.measured_performance.witness_capacity << ",\n";
        os << "        \"witness_fill_pct\": " << e.measured_performance.witness_fill_pct << ",\n";
        os << "        \"overflow_events\": " << e.measured_performance.overflow_events << ",\n";
        os << "        \"accum_peak\": " << e.measured_performance.accum_peak << ",\n";
        os << "        \"partial_peak\": " << e.measured_performance.partial_peak << ",\n";
        os << "        \"persistent_peak\": " << e.measured_performance.persistent_peak << "\n";
        os << "      },\n";

        // environment
        os << "      \"environment\": {\n";
        os << "        \"gpu_name\": \"" << jsonEscape(e.environment.gpu_name) << "\",\n";
        os << "        \"gpu_compute_capability\": \"" << jsonEscape(e.environment.gpu_compute_capability) << "\",\n";
        os << "        \"cuda_version\": \"" << jsonEscape(e.environment.cuda_version) << "\"\n";
        os << "      },\n";

        os << "      \"timestamp\": \"" << jsonEscape(e.timestamp) << "\",\n";

        // autotune_stages_run
        os << "      \"autotune_stages_run\": [";
        for (size_t s = 0; s < e.autotune_stages_run.size(); ++s) {
            if (s > 0) os << ", ";
            os << e.autotune_stages_run[s];
        }
        os << "],\n";

        os << "      \"confidence\": " << e.confidence << "\n";
        os << "    }";
    }

    if (!entries.empty()) os << "\n  ";
    os << "]\n";
    os << "}\n";

    return os.str();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// HistoryStore implementation
// ---------------------------------------------------------------------------

bool HistoryStore::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        // File does not exist — store remains empty, loaded_ = false.
        return false;
    }

    std::string contents((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
    ifs.close();

    try {
        JsonReader reader{contents, 0};
        reader.expect('{');

        uint64_t version = 0;
        reader.skipWS();
        while (reader.peek() != '}') {
            std::string key = reader.parseString();
            reader.expect(':');
            if (key == "version") {
                version = reader.parseUint64();
            } else if (key == "entries") {
                reader.expect('[');
                reader.skipWS();
                if (reader.peek() != ']') {
                    while (true) {
                        entries_.push_back(reader.parseEntry());
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
        reader.expect('}');

        if (version != 1) {
            LOG(LOG_WARNING) << "[HistoryStore] Warning: unsupported version " << version
                             << " in " << path << ". Ignoring contents.";
            entries_.clear();
            return false;
        }

        // --- Hard filters (F1, F2, F3): sanitize at load time ---
        size_t pre_filter = entries_.size();

        // F2: Partial buffer sanity floor (clamp, don't remove)
        for (auto& e : entries_) {
            if (e.optimal_params.recommended_partial_buffer > 0 &&
                e.optimal_params.recommended_partial_buffer < kMinPartialBufferSize) {
                e.optimal_params.recommended_partial_buffer = kMinPartialBufferSize;
            }
        }

        // F1 + F3: Remove degenerate kernel params and low-confidence entries
        std::erase_if(entries_, [](const HistoryEntry& e) {
            // F1: Degenerate kernel params (fallback from failed microbenchmark)
            if (e.optimal_params.kernel_params[0] == 128 &&
                e.optimal_params.kernel_params[1] == 4 &&
                e.optimal_params.kernel_params[2] == 4 &&
                e.optimal_params.kernel_params[3] == 4) {
                LOG(LOG_DEBUG_1) << "[History] Filtering degenerate kernel params entry for "
                                 << e.bit_length << "b";
                return true;
            }
            // F3: Low confidence entries are unreliable
            if (e.confidence < 0.3) {
                LOG(LOG_DEBUG_1) << "[History] Filtering low-confidence entry ("
                                 << e.confidence << ") for " << e.bit_length << "b";
                return true;
            }
            return false;
        });

        if (entries_.size() < pre_filter) {
            LOG(LOG_DEBUG_1) << "[History] Filtered " << (pre_filter - entries_.size())
                             << " of " << pre_filter << " entries at load time";
        }

        rebuildIndex();
        loaded_ = true;
        return true;

    } catch (const std::exception& e) {
        LOG(LOG_WARNING) << "[HistoryStore] Warning: JSON parse error in " << path
                         << ": " << e.what() << ". Store is empty.";
        entries_.clear();
        hash_index_.clear();
        loaded_ = false;
        return false;
    }
}

bool HistoryStore::save(const std::string& path) const {
    // Create parent directories if needed
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            LOG(LOG_ERROR) << "[HistoryStore] Error: cannot create directory "
                           << parent << ": " << ec.message();
            return false;
        }
    }

    // Atomic write: serialize → write to .tmp → rename
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream ofs(tmp_path);
        if (!ofs.is_open()) {
            LOG(LOG_ERROR) << "[HistoryStore] Error: cannot open " << tmp_path << " for writing.";
            return false;
        }
        ofs << serializeHistory(entries_);
        ofs.flush();
        if (!ofs.good()) {
            LOG(LOG_ERROR) << "[HistoryStore] Error: write failed to " << tmp_path << ".";
            return false;
        }
    }

    // Rename atomically (POSIX guarantee)
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        LOG(LOG_ERROR) << "[HistoryStore] Error: rename " << tmp_path << " → " << path
                       << " failed: " << std::strerror(errno);
        return false;
    }

    return true;
}

void HistoryStore::upsert(const HistoryEntry& entry) {
    auto it = hash_index_.find(entry.N_hash_sha256);
    if (it != hash_index_.end()) {
        // Entry exists — decide whether to replace
        auto& existing = entries_[it->second];
        bool replace = false;
        if (entry.confidence > existing.confidence) {
            replace = true;
        } else if (entry.confidence == existing.confidence &&
                   entry.measured_performance.total_time_sec <
                   existing.measured_performance.total_time_sec) {
            replace = true;
        }
        if (replace) {
            existing = entry;
            // Re-sort if bit_length changed (unlikely for same N, but defensive)
            rebuildIndex();
        }
    } else {
        // New entry — append and rebuild
        entries_.push_back(entry);
        rebuildIndex();
    }
}

const HistoryEntry* HistoryStore::findExact(const std::string& N_hash) const {
    auto it = hash_index_.find(N_hash);
    if (it == hash_index_.end()) return nullptr;
    return &entries_[it->second];
}

std::vector<const HistoryEntry*> HistoryStore::findByBitRange(
        uint32_t min_bits, uint32_t max_bits) const {
    std::vector<const HistoryEntry*> result;
    // entries_ is sorted by bit_length — could binary search, but linear is fine for <1000 entries
    for (const auto& e : entries_) {
        if (e.bit_length >= min_bits && e.bit_length <= max_bits)
            result.push_back(&e);
    }
    return result;
}

std::vector<const HistoryEntry*> HistoryStore::findKNearest(
        uint32_t bit_length, size_t k) const {
    if (entries_.empty() || k == 0) return {};

    struct DistEntry {
        uint32_t distance;
        const HistoryEntry* entry;
    };

    // Pass 1: Separate autotuned (strict) from heuristic-only (fallback) entries
    std::vector<DistEntry> ranked;
    std::vector<DistEntry> fallback;
    ranked.reserve(entries_.size());
    fallback.reserve(entries_.size());
    for (const auto& e : entries_) {
        uint32_t dist = (e.bit_length > bit_length)
                        ? (e.bit_length - bit_length)
                        : (bit_length - e.bit_length);
        // F5: prefer autotuned entries (non-empty stages_run or confidence > 0.5)
        if (!e.autotune_stages_run.empty() || e.confidence > 0.5) {
            ranked.push_back({dist, &e});
        } else {
            fallback.push_back({dist, &e});
        }
    }

    // Pass 2: Backfill from heuristic entries if fewer than K pass the quality filter
    if (ranked.size() < k && !fallback.empty()) {
        std::partial_sort(fallback.begin(),
                          fallback.begin() + static_cast<ptrdiff_t>(
                              std::min(k - ranked.size(), fallback.size())),
                          fallback.end(),
                          [](const DistEntry& a, const DistEntry& b) {
                              if (a.distance != b.distance) return a.distance < b.distance;
                              return a.entry->bit_length < b.entry->bit_length;
                          });
        size_t fill = std::min(k - ranked.size(), fallback.size());
        for (size_t i = 0; i < fill; ++i)
            ranked.push_back(fallback[i]);
    }

    // Select K nearest from combined pool
    size_t n = std::min(k, ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + static_cast<ptrdiff_t>(n), ranked.end(),
                      [](const DistEntry& a, const DistEntry& b) {
                          if (a.distance != b.distance) return a.distance < b.distance;
                          return a.entry->bit_length < b.entry->bit_length;
                      });

    std::vector<const HistoryEntry*> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i)
        result.push_back(ranked[i].entry);
    return result;
}

const std::vector<HistoryEntry>& HistoryStore::entries() const {
    return entries_;
}

size_t HistoryStore::size() const {
    return entries_.size();
}

bool HistoryStore::loaded() const {
    return loaded_;
}

void HistoryStore::filterByGPU(const std::string& gpu_name) {
    std::erase_if(entries_, [&gpu_name](const HistoryEntry& e) {
        return e.environment.gpu_name != gpu_name;
    });
    rebuildIndex();
}

void HistoryStore::rebuildIndex() {
    // Sort entries by bit_length ascending, stable for determinism
    std::stable_sort(entries_.begin(), entries_.end(),
                     [](const HistoryEntry& a, const HistoryEntry& b) {
                         return a.bit_length < b.bit_length;
                     });

    // Rebuild hash index
    hash_index_.clear();
    hash_index_.reserve(entries_.size());
    for (size_t i = 0; i < entries_.size(); ++i)
        hash_index_[entries_[i].N_hash_sha256] = i;
}

} // namespace mpqs::autotune
