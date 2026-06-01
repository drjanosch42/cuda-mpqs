// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>

/// Manages temporary and persistent GPU memory allocations for format conversion.
/// Each allocation is an independent cudaMalloc. Winner allocations survive after
/// losers are freed via free_temporaries(). ~25 allocations total, so cudaMalloc
/// overhead is negligible.
class FormatArena {
public:
    struct Allocation {
        void* ptr;
        size_t bytes;
        std::string tag;     // e.g., "tiledcoo_coords_block[0,64)"
        bool persistent;     // false = temporary (freed after benchmark phase)
    };

private:
    std::vector<Allocation> allocs_;

public:
    /// Allocate temporary GPU memory (freed by free_temporaries())
    void* alloc_temporary(size_t bytes, const std::string& tag);

    /// Allocate persistent GPU memory (kept until explicit free)
    void* alloc_persistent(size_t bytes, const std::string& tag);

    /// Promote a temporary allocation to persistent (winner format)
    void promote_to_persistent(void* ptr);

    /// Free all temporary allocations (after DP solver picks winners)
    void free_temporaries();

    /// Watermark-based scoped cleanup (P2-B)
    size_t get_watermark() const { return allocs_.size(); }
    void free_since(size_t watermark);

    /// Free everything (cleanup)
    void free_all();

    /// Stats
    size_t total_temporary_bytes() const;
    size_t total_persistent_bytes() const;
    size_t total_bytes() const;
};
