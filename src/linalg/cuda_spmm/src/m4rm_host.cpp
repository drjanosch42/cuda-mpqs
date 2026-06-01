// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "m4rm_data.h"
#include "common.h"
#include <vector>
#include <cstring>

#include "hpc_logger.h"

void prepare_m4rm_streams(const HostMatrixView& AT, int num_rows, M4RMContext& ctx) {
    if (num_rows != 8) {
        LOG(LOG_ERROR) << "[M4RM] Error: Encoder strictly supports 8 rows.";
        return;
    }

    ctx.num_relations = AT.n_cols;
    ctx.num_dense_rows = num_rows;

    // std::cout << "[M4RM] Encoding streams for " << ctx.num_relations << " relations..." << std::endl;

    // 1. Prepare Host Pattern Buffer
    // Pad to 128 bytes to prevent OOB reads in vectorized kernels
    size_t padded_size = (ctx.num_relations + 127) / 128 * 128;
    std::vector<uint8_t> h_patterns(padded_size, 0);
    
    // 2. Encode Patterns (Serial execution to ensure correctness of bitwise OR)
    // AT rows correspond to the bits in the pattern.
    for (int r = 0; r < num_rows; ++r) {
        if (r >= (int)AT.n_rows) break;

        const auto& row_indices = AT.get_row(r);
        uint8_t bit_mask = (1 << r);

        for (idx_t rel_idx : row_indices) {
            if (rel_idx < ctx.num_relations) {
                h_patterns[rel_idx] |= bit_mask;
            }
        }
    }

    // 3. Allocate Device Memory
    CUDA_CHECK(cudaMalloc((void**)&ctx.d_pattern_stream, padded_size * sizeof(uint8_t)));

    // 4. Transfer Data
    CUDA_CHECK(cudaMemcpy(ctx.d_pattern_stream, h_patterns.data(), padded_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // std::cout << "[M4RM] Setup Complete. " << ctx.num_relations << " patterns encoded." << std::endl;
}

void free_m4rm_context(M4RMContext& ctx) {
    if (ctx.d_pattern_stream) cudaFree(ctx.d_pattern_stream);
    ctx = {};
}
