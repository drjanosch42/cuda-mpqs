// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include "matrix_constructor.h"  // HostMatrixCSR, ValidateHostMatrixCSR
#include "merge_tree.h"
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Result of singleton removal.
struct SingletonResult {
    HostMatrixCSR reduced;              ///< Compacted matrix (dense row/col indices)
    std::vector<uint32_t> row_map;      ///< reduced_row -> original_row
    std::vector<uint32_t> col_map;      ///< new_col -> old_col
    uint32_t iterations  = 0;           ///< Number of fixpoint iterations
    uint32_t rows_removed = 0;          ///< Total rows removed
    uint32_t cols_removed = 0;          ///< Total columns removed (weight 0 after fixpoint)
};

/// Result of weight-2 merge pass.
struct MergeResult {
    HostMatrixCSR reduced;                ///< Matrix after merges + compaction
    std::vector<uint32_t> row_map;        ///< reduced_row -> node index in merge tree
    MergeTree merge_tree;                 ///< Merge history for sqrt reconstruction
    uint32_t merges_performed = 0;        ///< Number of weight-2 columns eliminated
    uint32_t rows_removed = 0;            ///< Net rows removed (= merges_performed)
};

/// Iterative merge/filter pipeline for GF(2) sparse matrices.
/// M2: singleton removal. M3: weight-2 column merges with merge tree.
/// Extended in M4 (higher-weight merges).
class MergeFilterPipeline {
public:
    /// Remove all singleton columns and their containing rows iteratively.
    /// Converges when no weight-1 columns remain.  Compacts the result to dense
    /// row/column indices and validates the output CSR.
    /// @param input  Expanded (F+2+L)-column CSR matrix from ExpandedMatrixBuilder.
    /// @return SingletonResult with reduced matrix and index mappings.
    SingletonResult removeSingletons(const HostMatrixCSR& input);

    /// Merge all weight-2 columns by XOR-ing their two containing rows.
    /// Builds a merge tree recording the merge history for sqrt reconstruction.
    /// After merging, every surviving column has weight 0 or >= 3.
    /// @param input         Singleton-free matrix (from removeSingletons).
    /// @param input_row_map Row map from singleton removal (reduced -> original).
    /// @return MergeResult with reduced matrix, node-indexed row map, and merge tree.
    MergeResult mergeWeight2(const HostMatrixCSR& input,
                             const std::vector<uint32_t>& input_row_map);

    /// Merge columns of weight [3, k_max] using pivot-based XOR.
    /// Extends the merge tree from a prior mergeWeight2() call.
    /// For each eligible column, the lightest row is chosen as pivot and XOR-ed
    /// with every other row sharing that column.  The pivot is then removed.
    /// Fill-in control: merges producing rows heavier than
    /// max(max_weight, 2 * avg_initial_weight) are skipped.
    /// @param input       MergeResult from mergeWeight2().
    /// @param k_max       Maximum column weight to process (default 10).
    /// @param max_weight  Maximum allowed row weight after merge (default 200).
    /// @return Updated MergeResult with further-reduced matrix and extended merge tree.
    MergeResult mergeHigherWeight(const MergeResult& input,
                                  uint32_t k_max = 10,
                                  uint32_t max_weight = 200);
};

} // namespace matrix
} // namespace mpqs
