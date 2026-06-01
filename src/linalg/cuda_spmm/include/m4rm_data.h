// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "common.h"

// Forward declaration
struct HostMatrix;

/**
 * @brief Holds device pointers and metadata for the M4RM execution context.
 */
struct M4RMContext {
    // Stream of 8-bit patterns (one byte per relation/column).
    uint8_t* d_pattern_stream = nullptr;

    // Total number of relations (columns in the matrix).
    size_t num_relations = 0;
    
    // Fixed at 8 for this implementation.
    int num_dense_rows = 8;
};

/**
 * @brief Encodes the first `num_rows` of AT into M4RM streams.
 */
void prepare_m4rm_streams(const HostMatrixView& AT, int num_rows, M4RMContext& ctx);

/**
 * @brief Frees GPU resources associated with the context.
 */
void free_m4rm_context(M4RMContext& ctx);
