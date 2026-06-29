// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "sieve_checkpoint.h"
#include "relation_hash.h"
#include "hpc_logger.h"

#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>
#include <filesystem>
#include <unordered_set>

#include <fcntl.h>
#include <unistd.h>

namespace mpqs::ckpt {

namespace {

// --- byte-buffer append helpers (little-endian, field-by-field — no struct dump) ---
template <typename T>
void put(std::vector<char>& buf, const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "put<T> requires trivially-copyable T");
    const char* p = reinterpret_cast<const char*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}
void put_bytes(std::vector<char>& buf, const void* src, size_t n) {
    const char* p = reinterpret_cast<const char*>(src);
    buf.insert(buf.end(), p, p + n);
}

// --- stream read helpers ---
template <typename T>
bool get(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(f);
}

/// Serialize the in-memory trailer to bytes (field order is the on-disk contract).
std::vector<char> serializeTrailer(const CheckpointTrailer& t) {
    std::vector<char> buf;
    put_bytes(buf, CKPT_TRAILER_MAGIC, 9);
    put<uint32_t>(buf, CKPT_SCHEMA_VERSION);
    put<uint64_t>(buf, t.global_a_index);
    put<uint64_t>(buf, t.target_relations);
    put<uint64_t>(buf, t.loaded_smooths_raw);
    put<uint64_t>(buf, t.loaded_smooths_dedup);
    put<uint64_t>(buf, t.loaded_partials);
    put<uint64_t>(buf, t.lp1_bound);
    put<uint32_t>(buf, t.sieve_bound);
    put_bytes(buf, t.N.limbs, 64);                  // 16 × uint32_t, little-endian
    put<uint8_t>(buf, t.cluster_section_present);
    put<uint64_t>(buf, t.elapsed_sieve_sec);
    return buf;
}

/// Append the variable-size cluster block (S3) to `buf` (field-by-field, little-endian).
/// Form: completed_prefix_cursor (u64), node_count (u32), initial_high_water[node_count] (u64).
void serializeClusterBlock(std::vector<char>& buf, const CheckpointClusterBlock& cb) {
    put<uint64_t>(buf, cb.completed_prefix_cursor);
    put<uint32_t>(buf, static_cast<uint32_t>(cb.initial_high_water.size()));
    for (uint64_t hw : cb.initial_high_water) put<uint64_t>(buf, hw);
}

/// Read the cluster block at the current stream position (immediately after the trailer).
/// `region_bytes_left` bounds node_count so a corrupt count cannot trigger a huge alloc.
bool readClusterBlock(std::ifstream& f, CheckpointClusterBlock& cb, uint64_t region_bytes_left) {
    if (!get(f, cb.completed_prefix_cursor)) return false;
    uint32_t node_count = 0;
    if (!get(f, node_count)) return false;
    // Each entry is 8 bytes; reject a count that cannot fit in the remaining region.
    if (static_cast<uint64_t>(node_count) * 8ull + 12ull > region_bytes_left) {
        LOG(LOG_WARNING) << "sieve_checkpoint: cluster block node_count=" << node_count
                         << " exceeds the trailer region — corrupt";
        return false;
    }
    cb.initial_high_water.resize(node_count);
    for (uint32_t i = 0; i < node_count; ++i) {
        if (!get(f, cb.initial_high_water[i])) return false;
    }
    return true;
}

/// Read the trailer at the current stream position. Returns false on a bad magic / short read.
bool readTrailer(std::ifstream& f, CheckpointTrailer& t) {
    char magic[9];
    f.read(magic, 9);
    if (!f || std::memcmp(magic, CKPT_TRAILER_MAGIC, 9) != 0) return false;
    uint32_t schema = 0;
    if (!get(f, schema)) return false;
    if (schema != CKPT_SCHEMA_VERSION) {
        LOG(LOG_WARNING) << "sieve_checkpoint: trailer schema " << schema
                         << " != " << CKPT_SCHEMA_VERSION;
        return false;
    }
    if (!get(f, t.global_a_index))       return false;
    if (!get(f, t.target_relations))     return false;
    if (!get(f, t.loaded_smooths_raw))   return false;
    if (!get(f, t.loaded_smooths_dedup)) return false;
    if (!get(f, t.loaded_partials))      return false;
    if (!get(f, t.lp1_bound))            return false;
    if (!get(f, t.sieve_bound))          return false;
    f.read(reinterpret_cast<char*>(t.N.limbs), 64);
    if (!f) return false;
    if (!get(f, t.cluster_section_present)) return false;
    if (!get(f, t.elapsed_sieve_sec))       return false;
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Host-side dedup of a scratch copy (B1: never touches live device state)
// ---------------------------------------------------------------------------

void dedupRelationsInPlace(mpqs::structures::HostRelationBatch& b) {
    if (b.num_relations == 0) return;

    mpqs::structures::HostRelationBatch out;
    out.factor_offsets.push_back(0);
    std::unordered_set<uint64_t> seen;
    seen.reserve(b.num_relations);

    const bool has_char = (b.char_bits.size() == b.num_relations);

    for (size_t i = 0; i < b.num_relations; ++i) {
        uint64_t h = mpqs::computeRelationHash(b, i);
        if (!seen.insert(h).second) continue;  // duplicate — drop

        out.sqrt_Q.push_back(b.sqrt_Q[i]);
        out.signs.push_back(b.signs[i]);
        out.val_2_exps.push_back(b.val_2_exps[i]);
        out.large_primes.push_back(b.large_primes[i]);
        out.char_bits.push_back(has_char ? b.char_bits[i] : 0u);

        uint64_t fs = b.factor_offsets[i];
        uint64_t fe = b.factor_offsets[i + 1];
        for (uint64_t f = fs; f < fe; ++f) {
            out.factor_indices.push_back(b.factor_indices[f]);
            out.factor_counts.push_back(b.factor_counts[f]);
        }
        out.num_factors += (fe - fs);
        out.factor_offsets.push_back(out.num_factors);
        out.num_relations++;
    }

    b = std::move(out);
}

// ---------------------------------------------------------------------------
// Atomic write
// ---------------------------------------------------------------------------

bool writeCheckpointAtomic(const std::string& ckpt_dir,
                           const mpqs::structures::HostRelationBatch& smooths,
                           const mpqs::structures::HostRelationBatch& partials,
                           const mpqs::io::V2Metadata& meta,
                           const CheckpointTrailer& trailer,
                           const CheckpointClusterBlock* cluster) {
    std::error_code ec;
    std::filesystem::create_directories(ckpt_dir, ec);  // serialize_v2 also does this

    const std::string live = ckpt_dir + "/sieve.ckpt";
    const std::string tmp  = ckpt_dir + "/sieve.ckpt.tmp";
    const std::string prev = ckpt_dir + "/sieve.ckpt.prev";

    // 0. Unlink any stale tmp from a prior crash — never append to a leftover (m-tmp).
    std::filesystem::remove(tmp, ec);

    // 1. serialize_v2 payload → tmp (reuses the tested serializer; tmp is intra-FS so the
    //    commit rename is truly atomic).
    if (!mpqs::io::serialize_v2(tmp, smooths, partials, meta)) {
        LOG(LOG_ERROR_CRITICAL) << "sieve_checkpoint: serialize_v2 failed for " << tmp;
        return false;
    }

    // Trailer begins where the v2 payload ends.
    uint64_t trailer_offset = std::filesystem::file_size(tmp, ec);
    if (ec) {
        LOG(LOG_ERROR_CRITICAL) << "sieve_checkpoint: file_size(" << tmp << ") failed";
        return false;
    }

    // 2. Build trailer (+ optional S3 cluster block) + fixed EOF footer, then append + fsync.
    std::vector<char> tail = serializeTrailer(trailer);
    // S3: append the variable-size cluster block between the trailer and the footer, iff the
    // trailer advertises it. The footer's trailer_len then spans trailer + cluster block, so
    // the reader can recover both by seeking. A solo file (cluster_section_present == 0, or a
    // null cluster ptr from S1) appends nothing → trailer_len == trailer bytes.
    if (trailer.cluster_section_present && cluster) {
        serializeClusterBlock(tail, *cluster);
    }
    const uint64_t trailer_len = tail.size();

    put_bytes(tail, CKPT_FOOTER_MAGIC, 9);
    put<uint64_t>(tail, trailer_offset);
    put<uint64_t>(tail, trailer_len);
    put<uint32_t>(tail, CKPT_SCHEMA_VERSION);

    {
        int fd = ::open(tmp.c_str(), O_WRONLY | O_APPEND);
        if (fd < 0) {
            LOG(LOG_ERROR_CRITICAL) << "sieve_checkpoint: open(" << tmp << ") for append failed";
            return false;
        }
        size_t off = 0;
        bool wr_ok = true;
        while (off < tail.size()) {
            ssize_t n = ::write(fd, tail.data() + off, tail.size() - off);
            if (n <= 0) { wr_ok = false; break; }
            off += static_cast<size_t>(n);
        }
        if (wr_ok && ::fsync(fd) != 0) wr_ok = false;
        ::close(fd);
        if (!wr_ok) {
            LOG(LOG_ERROR_CRITICAL) << "sieve_checkpoint: write/fsync of trailer+footer failed";
            return false;
        }
    }

    // 3. Commit: retain one prior generation, then a single atomic rename installs the new
    //    file. A kill before this leaves the prior sieve.ckpt intact + a stray .tmp.
    if (std::filesystem::exists(live)) {
        if (::rename(live.c_str(), prev.c_str()) != 0) {
            LOG(LOG_WARNING) << "sieve_checkpoint: rename(" << live << " -> " << prev
                             << ") failed (errno " << errno << "); continuing";
        }
    }
    if (::rename(tmp.c_str(), live.c_str()) != 0) {
        LOG(LOG_ERROR_CRITICAL) << "sieve_checkpoint: commit rename(" << tmp << " -> " << live
                                << ") failed (errno " << errno << ")";
        return false;
    }

    // 4. fsync the directory so the rename itself is durable across power/node loss.
    {
        int dfd = ::open(ckpt_dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

bool readCheckpoint(const std::string& path, CheckpointLoadResult& out) {
    out.ok = false;
    std::error_code ec;
    uint64_t fsz = std::filesystem::file_size(path, ec);
    if (ec || fsz < CKPT_FOOTER_SIZE) return false;

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // 1. Footer at (file_size - CKPT_FOOTER_SIZE). Missing magic ⇒ torn / not a checkpoint.
    f.seekg(static_cast<std::streamoff>(fsz - CKPT_FOOTER_SIZE), std::ios::beg);
    char fmagic[9];
    f.read(fmagic, 9);
    uint64_t trailer_offset = 0, trailer_len = 0;
    uint32_t fver = 0;
    if (!get(f, trailer_offset)) return false;
    if (!get(f, trailer_len))    return false;
    if (!get(f, fver))           return false;
    if (!f) return false;
    if (std::memcmp(fmagic, CKPT_FOOTER_MAGIC, 9) != 0) {
        LOG(LOG_WARNING) << "sieve_checkpoint: " << path << " has no EOF footer magic (torn/incomplete)";
        return false;
    }
    if (fver != CKPT_SCHEMA_VERSION) {
        LOG(LOG_WARNING) << "sieve_checkpoint: footer schema " << fver
                         << " != " << CKPT_SCHEMA_VERSION << " in " << path;
        return false;
    }
    // Sanity: trailer + cluster block + footer must exactly fill the file.
    if (trailer_offset + trailer_len + CKPT_FOOTER_SIZE != fsz) {
        LOG(LOG_WARNING) << "sieve_checkpoint: footer offsets inconsistent in " << path;
        return false;
    }

    // 2. Trailer at trailer_offset.
    f.seekg(static_cast<std::streamoff>(trailer_offset), std::ios::beg);
    if (!readTrailer(f, out.trailer)) {
        LOG(LOG_WARNING) << "sieve_checkpoint: trailer read failed in " << path;
        return false;
    }

    // 2b. Variable-size cluster block (S3) immediately after the trailer, iff advertised.
    //     The stream is positioned exactly at the trailer's end (readTrailer consumes only
    //     the trailer bytes). After reading, the position must land at trailer_offset +
    //     trailer_len (the trailer region exactly spans trailer + cluster block).
    if (out.trailer.cluster_section_present) {
        uint64_t consumed = static_cast<uint64_t>(f.tellg()) - trailer_offset;
        uint64_t region_left = (trailer_len > consumed) ? (trailer_len - consumed) : 0;
        if (!readClusterBlock(f, out.cluster, region_left)) {
            LOG(LOG_WARNING) << "sieve_checkpoint: cluster block read failed in " << path;
            return false;
        }
        uint64_t end_pos = static_cast<uint64_t>(f.tellg());
        if (end_pos != trailer_offset + trailer_len) {
            LOG(LOG_WARNING) << "sieve_checkpoint: cluster block size mismatch in " << path
                             << " (read to " << end_pos << ", expected "
                             << (trailer_offset + trailer_len) << ")";
            return false;
        }
    }

    // 3. Relation payload — deserialize_v2 reads from the start and ignores the trailing
    //    trailer/footer bytes (contractual; see relation_io.cpp invariant comment).
    if (!mpqs::io::deserialize_v2(path, out.smooths, out.partials, out.meta)) {
        LOG(LOG_WARNING) << "sieve_checkpoint: deserialize_v2 payload failed in " << path;
        return false;
    }

    out.ok = true;
    return true;
}

bool loadLatestCheckpoint(const std::string& dir, CheckpointLoadResult& out) {
    const std::string live = dir + "/sieve.ckpt";
    const std::string prev = dir + "/sieve.ckpt.prev";

    if (std::filesystem::exists(live) && readCheckpoint(live, out)) return true;

    if (std::filesystem::exists(prev)) {
        LOG(LOG_WARNING) << "sieve_checkpoint: " << live
                         << " missing/incomplete — falling back to " << prev;
        if (readCheckpoint(prev, out)) return true;
    }
    return false;
}

} // namespace mpqs::ckpt
