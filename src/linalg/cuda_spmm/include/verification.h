// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include <vector>

class Verification {
public:
    /**
     * @brief Reference CPU implementation of SpMM over GF(2).
     * Computes C = A * V.
     * 
     * @param A The original host matrix (Coordinate List).
     * @param V The input vector block (flat array of uint64_t).
     * @param C_ref Output buffer for the result.
     * @param permutation The row permutation applied during preprocessing. 
     *                    Logic: C_ref[i] corresponds to A.rows[permutation[i]] * V.
     * @param bit_width 32/64/128/256/512 for width of V
     */
    static void cpu_spmm_gf2(
        const HostMatrix& A, 
        const void* V, 
        void* C_ref,
        const std::vector<row_idx_t>& permutation,
        int bit_width
    );

    static bool check_results(
        const void* C_gpu, 
        const void* C_ref,
        size_t n_rows,
        int bit_width
    );
};
