// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "merge_tree.h"

#include <algorithm>
#include <cassert>
#include <stack>
#include <unordered_map>

namespace mpqs {
namespace matrix {

void MergeTree::expand(uint32_t node_idx, std::vector<uint32_t>& out) const {
    out.clear();
    std::stack<uint32_t> work;
    work.push(node_idx);

    while (!work.empty()) {
        uint32_t idx = work.top();
        work.pop();

        if (idx < num_leaves) {
            out.push_back(idx);
        } else {
            uint32_t internal_idx = idx - num_leaves;
            assert(internal_idx < internal_nodes.size());
            work.push(internal_nodes[internal_idx].left);
            work.push(internal_nodes[internal_idx].right);
        }
    }
}

std::vector<uint32_t> expandKernelVector(
    const std::vector<uint64_t>& packed_bits,
    uint32_t num_reduced_rows,
    const std::vector<uint32_t>& row_map,
    const MergeTree& tree)
{
    std::unordered_map<uint32_t, uint32_t> freq;
    std::vector<uint32_t> leaves;

    for (uint32_t i = 0; i < num_reduced_rows; ++i) {
        uint32_t word = i / 64;
        uint32_t bit  = i % 64;
        if (word >= packed_bits.size()) break;
        if (!(packed_bits[word] & (1ULL << bit))) continue;

        leaves.clear();
        tree.expand(row_map[i], leaves);

        for (uint32_t leaf : leaves) {
            freq[leaf]++;
        }
    }

    // GF(2): keep only odd-frequency leaves
    std::vector<uint32_t> result;
    result.reserve(freq.size());
    for (auto& [leaf, count] : freq) {
        if (count & 1) {
            result.push_back(leaf);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace matrix
} // namespace mpqs
