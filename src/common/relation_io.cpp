// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// This file is part of cuda-mpqs (GPU-accelerated SIQS/MPQS factorization).
// See LICENSE at the repository root for licensing terms, including the NVIDIA CUDA Toolkit exception.
#include "relation_io.h"
#include "hpc_logger.h"
#include <fstream>
#include <cstring>
#include <filesystem>

namespace {

constexpr char V1_MAGIC[8] = {'M','P','Q','S','_','S','O','A'};
constexpr char V2_MAGIC[8] = {'M','P','Q','S','_','V','2','\0'};

// v2 flags bitfield
constexpr uint32_t FLAG_HAS_PARTIALS = 0x1;
constexpr uint32_t FLAG_HAS_METADATA = 0x2;

template<typename T>
void write_vec(std::ofstream& f, const std::vector<T>& v) {
    uint64_t n = v.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n > 0) f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}

template<typename T>
void read_vec(std::ifstream& f, std::vector<T>& v) {
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    v.resize(n);
    if (n > 0) f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
}

/// Write all fields of a HostRelationBatch (excluding counters, which are written separately).
void write_batch(std::ofstream& f, const mpqs::structures::HostRelationBatch& b) {
    write_vec(f, b.sqrt_Q);
    write_vec(f, b.signs);
    write_vec(f, b.val_2_exps);
    write_vec(f, b.large_primes);
    write_vec(f, b.factor_offsets);
    write_vec(f, b.factor_indices);
    write_vec(f, b.factor_counts);
}

/// Read all fields of a HostRelationBatch (excluding counters, which are read separately).
void read_batch(std::ifstream& f, mpqs::structures::HostRelationBatch& b) {
    read_vec(f, b.sqrt_Q);
    read_vec(f, b.signs);
    read_vec(f, b.val_2_exps);
    read_vec(f, b.large_primes);
    read_vec(f, b.factor_offsets);
    read_vec(f, b.factor_indices);
    read_vec(f, b.factor_counts);
}

} // anonymous namespace

namespace mpqs::io {

// ---------------------------------------------------------------------------
// v1 serialization — exact replica of the former orchestrator anonymous-ns code
// ---------------------------------------------------------------------------

bool serialize_v1(const std::string& path,
                  const structures::HostRelationBatch& batch) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(V1_MAGIC, 8);
    uint64_t nr = batch.num_relations, nf = batch.num_factors;
    f.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
    f.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
    write_batch(f, batch);
    return f.good();
}

bool deserialize_v1(const std::string& path,
                    structures::HostRelationBatch& batch) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, V1_MAGIC, 8) != 0) return false;
    uint64_t nr, nf;
    f.read(reinterpret_cast<char*>(&nr), sizeof(nr));
    f.read(reinterpret_cast<char*>(&nf), sizeof(nf));
    batch.num_relations = nr;
    batch.num_factors = nf;
    read_batch(f, batch);
    return f.good();
}

// ---------------------------------------------------------------------------
// v2 serialization — two-batch layout with pipeline metadata
// ---------------------------------------------------------------------------

bool serialize_v2(const std::string& path,
                  const structures::HostRelationBatch& full_smooths,
                  const structures::HostRelationBatch& partials,
                  const V2Metadata& meta) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // Header
    f.write(V2_MAGIC, 8);
    uint32_t version = 2;
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    uint32_t flags = FLAG_HAS_METADATA;
    if (partials.num_relations > 0) flags |= FLAG_HAS_PARTIALS;
    f.write(reinterpret_cast<const char*>(&flags), sizeof(flags));

    // Metadata section
    // uint512: 16 x uint32_t, written directly from limbs[] (little-endian)
    f.write(reinterpret_cast<const char*>(meta.N.limbs), 64);
    uint32_t fb_size = static_cast<uint32_t>(meta.factor_base.size());
    f.write(reinterpret_cast<const char*>(&fb_size), sizeof(fb_size));
    if (fb_size > 0)
        f.write(reinterpret_cast<const char*>(meta.factor_base.data()), fb_size * sizeof(uint32_t));
    f.write(reinterpret_cast<const char*>(&meta.lp_bound), sizeof(meta.lp_bound));
    f.write(reinterpret_cast<const char*>(&meta.sieve_bound), sizeof(meta.sieve_bound));

    // Full smooth section
    uint64_t nr = full_smooths.num_relations, nf = full_smooths.num_factors;
    f.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
    f.write(reinterpret_cast<const char*>(&nf), sizeof(nf));
    write_batch(f, full_smooths);

    // Partial section (if present)
    if (flags & FLAG_HAS_PARTIALS) {
        uint64_t np = partials.num_relations, npf = partials.num_factors;
        f.write(reinterpret_cast<const char*>(&np), sizeof(np));
        f.write(reinterpret_cast<const char*>(&npf), sizeof(npf));
        write_batch(f, partials);
    }

    return f.good();
}

bool deserialize_v2(const std::string& path,
                    structures::HostRelationBatch& full_smooths,
                    structures::HostRelationBatch& partials,
                    V2Metadata& meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // Header
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, V2_MAGIC, 8) != 0) return false;
    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 2) {
        LOG(LOG_WARNING) << "relation_io: unsupported v2 version " << version;
        return false;
    }
    uint32_t flags;
    f.read(reinterpret_cast<char*>(&flags), sizeof(flags));

    // Warn about unknown flags but continue
    constexpr uint32_t KNOWN_FLAGS = FLAG_HAS_PARTIALS | FLAG_HAS_METADATA;
    if (flags & ~KNOWN_FLAGS) {
        LOG(LOG_WARNING) << "relation_io: unknown v2 flags 0x" << std::hex << (flags & ~KNOWN_FLAGS)
                         << std::dec << " — ignoring";
    }

    // Metadata section
    if (flags & FLAG_HAS_METADATA) {
        f.read(reinterpret_cast<char*>(meta.N.limbs), 64);
        uint32_t fb_size;
        f.read(reinterpret_cast<char*>(&fb_size), sizeof(fb_size));
        meta.factor_base.resize(fb_size);
        if (fb_size > 0)
            f.read(reinterpret_cast<char*>(meta.factor_base.data()), fb_size * sizeof(uint32_t));
        f.read(reinterpret_cast<char*>(&meta.lp_bound), sizeof(meta.lp_bound));
        f.read(reinterpret_cast<char*>(&meta.sieve_bound), sizeof(meta.sieve_bound));
    }

    // Full smooth section
    uint64_t nr, nf;
    f.read(reinterpret_cast<char*>(&nr), sizeof(nr));
    f.read(reinterpret_cast<char*>(&nf), sizeof(nf));
    full_smooths.num_relations = nr;
    full_smooths.num_factors = nf;
    read_batch(f, full_smooths);

    // Partial section
    if (flags & FLAG_HAS_PARTIALS) {
        uint64_t np, npf;
        f.read(reinterpret_cast<char*>(&np), sizeof(np));
        f.read(reinterpret_cast<char*>(&npf), sizeof(npf));
        partials.num_relations = np;
        partials.num_factors = npf;
        read_batch(f, partials);
    } else {
        partials.clear();
    }

    return f.good();
}

// ---------------------------------------------------------------------------
// Auto-detect + dispatch
// ---------------------------------------------------------------------------

int detect_and_deserialize(const std::string& path,
                           structures::HostRelationBatch& batch,
                           structures::HostRelationBatch& smooths,
                           structures::HostRelationBatch& partials,
                           V2Metadata& meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    char magic[8];
    f.read(magic, 8);
    if (!f) return 0;
    f.close();

    if (std::memcmp(magic, V2_MAGIC, 8) == 0) {
        return deserialize_v2(path, smooths, partials, meta) ? 2 : 0;
    }
    if (std::memcmp(magic, V1_MAGIC, 8) == 0) {
        return deserialize_v1(path, batch) ? 1 : 0;
    }

    LOG(LOG_ERROR_CRITICAL) << "relation_io: unrecognized file magic in " << path;
    return 0;
}

} // namespace mpqs::io
