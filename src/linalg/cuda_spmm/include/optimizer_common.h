// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "kernels.h"
#include "m4rm_data.h"
#include <string>
#include <vector>
#include <tuple>

// Available Kernel Types
enum class KernelID {
    M4RM,
    Dense_Bitslice,
    Sparse_WarpCSR,
    Sparse_TiledCOO,
    Sparse_TiledCOO_Unrolled,
    Sparse_PForDelta,
    Sparse_PForDelta_BitExact,
    Sparse_Delta16,
    Sparse_Golomb,
    Sparse_Ellpack
};

// Configuration for a specific segment
struct KernelConfig {
    KernelID id;
    int vector_width_bits; 
    
    struct Params {
        int tiled_block_size = 256;
        float pfor_threshold = 0.90f;
        int m4rm_rows = 8; 
        int max_nnz_per_slice = 0; 
    } params;

    std::string name;

    bool operator==(const KernelConfig& other) const {
        if (id != other.id || vector_width_bits != other.vector_width_bits) return false;
        if (id == KernelID::Sparse_TiledCOO || id == KernelID::Sparse_TiledCOO_Unrolled) {
            return params.tiled_block_size == other.params.tiled_block_size;
        }
        if (id == KernelID::Sparse_PForDelta || id == KernelID::Sparse_PForDelta_BitExact) {
            return params.pfor_threshold == other.params.pfor_threshold;
        }
        if (id == KernelID::M4RM) {
            return params.m4rm_rows == other.params.m4rm_rows;
        }
        return true;
    }

    bool operator!=(const KernelConfig& other) const { return !(*this == other); }
};

struct SegmentRecipe {
    row_idx_t start_row;
    row_idx_t end_row;
    KernelConfig best_config;
    double measured_throughput_gnnz;

    // Live device data from GPU autotuning.
    // Mutable so compile() can transfer ownership (null out after adoption)
    // even when iterating over a const ExecutionPlan.  This prevents
    // double-free when verify_A()/verify_AT() re-calls compile() on the
    // same plan — has_device_data() returns false after the first adoption,
    // causing the second compile() to fall back to CPU preprocessing.
    mutable DeviceMatrix device_data = {};
    mutable M4RMContext  m4rm_data   = {};

    bool has_device_data() const {
        return device_data.d_sparse_tiled_coords != nullptr
            || device_data.d_delta_16_stream != nullptr
            || device_data.d_pfor_be_data != nullptr
            || device_data.d_stream_remainders != nullptr
            || device_data.warp_csr_row_ptr != nullptr
            || m4rm_data.d_pattern_stream != nullptr;
    }
};

struct ExecutionPlan {
    bool is_transposed;
    double estimated_total_time_ms;
    double estimated_throughput_gnnz;
    std::vector<SegmentRecipe> segments;
    std::vector<row_idx_t> density_perm;  // maps permuted→original row indices
};

// Execution Strategy Flags (3 bits — 8 combinations benchmarked by tune_execution_strategy)
struct LaunchConfig {
    bool spawn_dense_group = false;       // Parallelize dense+M4RM kernels via multi-stream
    bool spawn_sparse_group = false;      // Parallelize sparse kernels
    bool spawn_dense_sparse_merge = false;// Run Dense+M4RM and Sparse phases in parallel
};
