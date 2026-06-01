// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "reordering.h"
#include <queue>
#include <algorithm>
#include <numeric>
#include <omp.h>

#include "hpc_logger.h"

std::vector<idx_t> MatrixReordering::compute_rcm_permutation(
    const HostMatrix& matrix, 
    idx_t dense_col_limit
) {
    LOG(LOG_INFO) << "[RCM] Computing Permutation...";
    
    idx_t n_sparse_cols = matrix.n_cols - dense_col_limit;
    if (n_sparse_cols == 0) return {};

    size_t total_nodes = matrix.n_rows + n_sparse_cols;
    
    // Degrees and Adjacency
    std::vector<uint32_t> degrees(total_nodes, 0);
    std::vector<std::vector<idx_t>> adj(total_nodes); 

    // Build Bipartite Graph
    // Row nodes: 0 to n_rows-1
    // Col nodes: n_rows to total_nodes-1
    for(row_idx_t r = 0; r < matrix.n_rows; ++r) {
        for(idx_t c : matrix.rows[r]) {
            if (c >= dense_col_limit) {
                idx_t sparse_c_idx = c - dense_col_limit;
                size_t row_node = r;
                size_t col_node = matrix.n_rows + sparse_c_idx;
                
                adj[row_node].push_back((idx_t)col_node);
                adj[col_node].push_back((idx_t)row_node);
                degrees[row_node]++;
                degrees[col_node]++;
            }
        }
    }

    // Sort adjacency lists by degree
    #pragma omp parallel for schedule(dynamic, 1024)
    for(size_t i = 0; i < total_nodes; ++i) {
        std::sort(adj[i].begin(), adj[i].end(), [&](idx_t a, idx_t b) {
            return degrees[a] < degrees[b];
        });
    }

    // BFS Traversal
    std::vector<bool> visited(total_nodes, false);
    std::vector<idx_t> rcm_order;
    rcm_order.reserve(total_nodes);

    for (size_t i = 0; i < total_nodes; ++i) {
        if (visited[i]) continue;
        if (degrees[i] == 0) continue; // Skip isolated nodes (if any)
        
        std::queue<idx_t> Q;
        Q.push((idx_t)i);
        visited[i] = true;
        rcm_order.push_back((idx_t)i);

        while(!Q.empty()) {
            idx_t u = Q.front();
            Q.pop();

            for(idx_t v : adj[u]) {
                if(!visited[v]) {
                    visited[v] = true;
                    rcm_order.push_back(v);
                    Q.push(v);
                }
            }
        }
    }
    
    // Add isolated nodes at the end to ensure permutation completeness
    if (rcm_order.size() < total_nodes) {
        for(size_t i=0; i<total_nodes; ++i) {
             if(!visited[i]) rcm_order.push_back((idx_t)i);
        }
    }

    // Reverse Order
    std::reverse(rcm_order.begin(), rcm_order.end());

    // Extract Column Permutation
    // We map: old_sparse_idx -> new_sparse_idx
    std::vector<idx_t> sigma(n_sparse_cols);
    idx_t new_idx_counter = 0;

    for(idx_t node : rcm_order) {
        if (node >= matrix.n_rows) {
            idx_t old_sparse_idx = node - matrix.n_rows;
            sigma[old_sparse_idx] = new_idx_counter++;
        }
    }
    
    // Safety check: ensure all columns mapped
    if (new_idx_counter != n_sparse_cols) {
        LOG(LOG_WARNING) << "[RCM] Warning: Node count mismatch!";
    }

    LOG(LOG_INFO) << "[RCM] Permutation computed.";
    return sigma;
}

HostMatrix MatrixReordering::apply_permutation(
    const HostMatrix& matrix,
    const std::vector<idx_t>& sigma,
    idx_t dense_col_limit
) {
    if (sigma.empty()) return matrix;

    LOG(LOG_INFO) << "[RCM] Applying permutation to matrix...";
    HostMatrix new_mat = matrix; // Copy structure (rows vector)

    #pragma omp parallel for schedule(dynamic, 128)
    for(row_idx_t r = 0; r < new_mat.n_rows; ++r) {
        for(size_t i = 0; i < new_mat.rows[r].size(); ++i) {
            idx_t c = new_mat.rows[r][i];
            if (c >= dense_col_limit) {
                idx_t sparse_idx = c - dense_col_limit;
                if (sparse_idx < sigma.size()) {
                    new_mat.rows[r][i] = dense_col_limit + sigma[sparse_idx];
                }
            }
        }
        std::sort(new_mat.rows[r].begin(), new_mat.rows[r].end());
    }

    return new_mat;
}
