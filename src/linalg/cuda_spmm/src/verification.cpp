// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "verification.h"
#include "hpc_logger.h"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstring>

void Verification::cpu_spmm_gf2(
    const HostMatrix& A, 
    const void* V, 
    void* C_ref,
    const std::vector<row_idx_t>& permutation,
    int bit_width
) {
    const uint32_t* V32 = (const uint32_t*)V;
    uint32_t* C32 = (uint32_t*)C_ref;
    
    int ints_per_row = bit_width / 32;
    size_t total_ints = A.n_rows * ints_per_row;
    
    // Zero out C_ref
    std::memset(C_ref, 0, total_ints * 4);

    #pragma omp parallel for schedule(dynamic, 1024)
    for(size_t i = 0; i < A.n_rows; ++i) {
        row_idx_t orig_idx = permutation[i];
        const auto& row_indices = A.rows[orig_idx];

        // Temp accumulator (stack allocated VLA or small vec)
        std::vector<uint32_t> acc(ints_per_row, 0);

        for(idx_t col : row_indices) {
            for(int w=0; w<ints_per_row; ++w) {
                acc[w] ^= V32[col * ints_per_row + w];
            }
        }

        for(int w=0; w<ints_per_row; ++w) {
            C32[i * ints_per_row + w] = acc[w];
        }
    }
}

bool Verification::check_results(const void* C_gpu, const void* C_ref, size_t n_rows, int bit_width) {
    const uint32_t* gpu = (const uint32_t*)C_gpu;
    const uint32_t* ref = (const uint32_t*)C_ref;
    
    int ints_per_row = bit_width / 32;
    size_t errors = 0;

    for(size_t i = 0; i < n_rows; ++i) {
        for(int w=0; w < ints_per_row; ++w) {
            size_t idx = i * ints_per_row + w;
            if(gpu[idx] != ref[idx]) {
                if(errors < 5) {
                    LOG(LOG_DEBUG_1) << "[Verification] Error at Row " << i << " Word " << w 
                              << " | GPU: " << std::hex << gpu[idx] 
                              << " | REF: " << ref[idx] << std::dec << std::endl;
                }
                errors++;
            }
        }
    }

    if(errors > 0) {
        LOG(LOG_DEBUG_1) << "[Verification] FAILED with " << errors << " 32-bit word errors." << std::endl;
        return false;
    }
    LOG(LOG_DEBUG_1) << "[Verification] PASSED." << std::endl;
    return true;
}
