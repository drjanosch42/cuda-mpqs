// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include <vector>

class MatrixReordering {
public:
    /**
     * @brief Computes a column permutation using the Reverse Cuthill-McKee (RCM) algorithm
     * adapted for bipartite graphs.
     * 
     * @param matrix The sparse matrix structure.
     * @param dense_col_limit Columns below this limit are preserved (not reordered).
     * @return std::vector<idx_t> Permutation array 'sigma' where new_col = sigma[old_col_sparse_offset].
     */
    static std::vector<idx_t> compute_rcm_permutation(
        const HostMatrix& matrix, 
        idx_t dense_col_limit
    );

    /**
     * @brief Applies the permutation to the matrix structure.
     * Returns a new HostMatrix with columns remapped.
     */
    static HostMatrix apply_permutation(
        const HostMatrix& matrix,
        const std::vector<idx_t>& sigma,
        idx_t dense_col_limit
    );
};
