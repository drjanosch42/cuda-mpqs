// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Records the merge history of matrix row combinations over GF(2).
/// Leaves are indices [0, num_leaves). Internal node k has global index num_leaves + k.
/// Used by the sqrt stage (M5) to expand BW kernel vectors back to original relations.
struct MergeTree {
    struct Node {
        uint32_t left;   ///< Child index (leaf or internal)
        uint32_t right;  ///< Child index (leaf or internal)
    };

    uint32_t num_leaves = 0;                 ///< Number of original relations
    std::vector<Node> internal_nodes;        ///< internal_nodes[k] has global index num_leaves + k

    /// Expand a node index to the set of original relation indices (leaves).
    /// Uses iterative stack-based traversal to avoid recursion depth issues.
    /// @param node_idx  Global node index (leaf or internal).
    /// @param out       Output vector; cleared then populated with leaf indices.
    void expand(uint32_t node_idx, std::vector<uint32_t>& out) const;
};

/// Expand a BW kernel vector (packed bits over reduced matrix rows) to a sorted
/// list of original relation indices, handling XOR cancellation (even-frequency
/// elimination).  For each set bit i in packed_bits (i < num_reduced_rows),
/// row_map[i] is expanded through the merge tree to leaf indices.  Leaves
/// appearing an even number of times cancel (GF(2) semantics); only
/// odd-frequency leaves survive.
/// @param packed_bits       BW solution: bit i set ⟹ reduced row i selected.
/// @param num_reduced_rows  Number of rows in the reduced matrix.
/// @param row_map           reduced_row → merge tree node index.
/// @param tree              Merge tree from M3/M4.
/// @return Sorted vector of original relation indices with odd occurrence count.
std::vector<uint32_t> expandKernelVector(
    const std::vector<uint64_t>& packed_bits,
    uint32_t num_reduced_rows,
    const std::vector<uint32_t>& row_map,
    const MergeTree& tree);

} // namespace matrix
} // namespace mpqs
