// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.



// -----------------------------------------------------------------------------
// FNV-1a Hashing Helper (FNV-1a 8-bit) for Reference Checks
// -----------------------------------------------------------------------------
inline uint64_t fnv1a_hash_bytes(const void* data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint8_t* p = (const uint8_t*)data;
    for(size_t i=0; i<len; ++i) {
        hash ^= p[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// -----------------------------------------------------------------------------
// Hashing Helper (FNV-1a 64-bit) for Reference Checks
// -----------------------------------------------------------------------------
inline uint64_t fnv1a_hash_uint64(const std::vector<uint64_t>& data) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for(uint64_t val : data) {
        const uint8_t* bytes = (const uint8_t*)&val;
        for(int i = 0; i < 8; ++i) {
            hash ^= bytes[i];
            hash *= 0x100000001b3ULL;
        }
    }
    return hash;
}  

