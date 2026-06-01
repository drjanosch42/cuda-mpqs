// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "merge_filter.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <set>

namespace mpqs {
namespace matrix {

/// Maximum singleton-removal iterations before forced termination.
static constexpr uint32_t kMaxSingletonIterations = 100;
/// Warn if convergence takes more than this many iterations.
static constexpr uint32_t kWarnIterationThreshold = 20;

SingletonResult MergeFilterPipeline::removeSingletons(const HostMatrixCSR& input) {
    LOG_SET_MODULE("MergeFilter");

    const uint32_t n_rows = input.n_rows;
    const uint32_t n_cols = input.n_cols;

    // --- Phase 1: compute initial column weights ---
    std::vector<uint32_t> col_weight(n_cols, 0);
    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t begin = input.row_offsets[r];
        const uint32_t end   = input.row_offsets[r + 1];
        for (uint32_t j = begin; j < end; ++j) {
            col_weight[input.col_indices[j]]++;
        }
    }

    // --- Phase 2: iterative singleton removal ---
    std::vector<bool> row_alive(n_rows, true);
    uint32_t total_removed = 0;
    uint32_t iteration = 0;

    while (iteration < kMaxSingletonIterations) {
        uint32_t removed_this_pass = 0;

        for (uint32_t r = 0; r < n_rows; ++r) {
            if (!row_alive[r]) continue;

            const uint32_t begin = input.row_offsets[r];
            const uint32_t end   = input.row_offsets[r + 1];

            // Check if this row contains any singleton column
            bool has_singleton = false;
            for (uint32_t j = begin; j < end; ++j) {
                if (col_weight[input.col_indices[j]] == 1) {
                    has_singleton = true;
                    break;
                }
            }

            if (has_singleton) {
                row_alive[r] = false;
                for (uint32_t j = begin; j < end; ++j) {
                    col_weight[input.col_indices[j]]--;
                }
                removed_this_pass++;
            }
        }

        total_removed += removed_this_pass;
        ++iteration;

        LOG(LOG_INFO) << "Singleton pass " << iteration
                      << ": removed " << removed_this_pass << " rows, "
                      << (n_rows - total_removed) << " remaining.";

        if (removed_this_pass == 0) break;
    }

    if (iteration >= kMaxSingletonIterations) {
        LOG(LOG_WARNING) << "Singleton removal did not converge in "
                         << kMaxSingletonIterations << " iterations.";
    } else if (iteration > kWarnIterationThreshold) {
        LOG(LOG_WARNING) << "Singleton removal took " << iteration
                         << " iterations (> " << kWarnIterationThreshold << ").";
    }

    // --- Phase 3: column compaction (old_col -> new_col) ---
    std::vector<uint32_t> old_to_new_col(n_cols, UINT32_MAX);
    std::vector<uint32_t> col_map;
    col_map.reserve(n_cols);  // upper bound
    uint32_t new_col_count = 0;
    for (uint32_t c = 0; c < n_cols; ++c) {
        if (col_weight[c] > 0) {
            old_to_new_col[c] = new_col_count++;
            col_map.push_back(c);
        }
    }

    // --- Phase 4: row compaction (alive rows -> new CSR) ---
    const uint32_t new_row_count = n_rows - total_removed;
    HostMatrixCSR reduced;
    reduced.n_rows = new_row_count;
    reduced.n_cols = new_col_count;
    reduced.row_offsets.reserve(new_row_count + 1);
    reduced.row_offsets.push_back(0);

    std::vector<uint32_t> row_map;
    row_map.reserve(new_row_count);

    for (uint32_t r = 0; r < n_rows; ++r) {
        if (!row_alive[r]) continue;

        row_map.push_back(r);
        const uint32_t begin = input.row_offsets[r];
        const uint32_t end   = input.row_offsets[r + 1];
        for (uint32_t j = begin; j < end; ++j) {
            uint32_t new_col = old_to_new_col[input.col_indices[j]];
            assert(new_col != UINT32_MAX && "alive row references dead column");
            reduced.col_indices.push_back(new_col);
        }
        reduced.row_offsets.push_back(
            static_cast<uint32_t>(reduced.col_indices.size()));
    }

    assert(reduced.n_rows == static_cast<uint32_t>(row_map.size()));

    const uint32_t cols_removed = n_cols - new_col_count;
    LOG(LOG_INFO) << "Singleton removal complete: "
                  << n_rows << " -> " << reduced.n_rows << " rows ("
                  << total_removed << " removed), "
                  << n_cols << " -> " << new_col_count << " cols ("
                  << cols_removed << " removed), "
                  << iteration << " iterations.";

    // Validate the compacted CSR
    ValidateHostMatrixCSR(reduced);

    SingletonResult result;
    result.reduced      = std::move(reduced);
    result.row_map      = std::move(row_map);
    result.col_map      = std::move(col_map);
    result.iterations   = iteration;
    result.rows_removed = total_removed;
    result.cols_removed = cols_removed;
    return result;
}

// ---------------------------------------------------------------------------
// M3: Weight-2 column merges
// ---------------------------------------------------------------------------

/// Symmetric difference of two sorted vectors (GF(2) row addition).
/// Elements present in both cancel; elements in only one survive.
/// O(|a| + |b|) via two-pointer merge.  Result is sorted.
static std::vector<uint32_t> xor_rows(
    const std::vector<uint32_t>& a,
    const std::vector<uint32_t>& b)
{
    std::vector<uint32_t> result;
    result.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            result.push_back(a[i++]);
        } else if (a[i] > b[j]) {
            result.push_back(b[j++]);
        } else {
            // Equal: cancel in GF(2)
            ++i; ++j;
        }
    }
    while (i < a.size()) result.push_back(a[i++]);
    while (j < b.size()) result.push_back(b[j++]);
    return result;
}

MergeResult MergeFilterPipeline::mergeWeight2(
    const HostMatrixCSR& input,
    const std::vector<uint32_t>& input_row_map)
{
    LOG_SET_MODULE("MergeFilter");

    const uint32_t n_rows = input.n_rows;
    const uint32_t n_cols = input.n_cols;

    // --- 1. Convert CSR to row-major for efficient XOR ---
    std::vector<std::vector<uint32_t>> rows(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) {
        uint32_t start = input.row_offsets[r];
        uint32_t end   = input.row_offsets[r + 1];
        rows[r].assign(input.col_indices.begin() + start,
                       input.col_indices.begin() + end);
    }

    // --- 2. Initialize merge tree ---
    // num_leaves = one past the maximum original row index referenced by input_row_map.
    MergeTree tree;
    tree.num_leaves = 0;
    if (!input_row_map.empty()) {
        tree.num_leaves = *std::max_element(input_row_map.begin(),
                                            input_row_map.end()) + 1;
    }

    // node_of[r] tracks the merge-tree node for alive row r.
    // Initially each row maps to its original relation index (a leaf).
    std::vector<uint32_t> node_of(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) {
        node_of[r] = input_row_map[r];
    }

    // --- 3. Column weights + inverted index (column -> rows) ---
    std::vector<uint32_t> col_weight(n_cols, 0);
    std::vector<std::vector<uint32_t>> col_rows(n_cols);
    std::vector<bool> row_alive(n_rows, true);

    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t c : rows[r]) {
            col_weight[c]++;
            col_rows[c].push_back(r);
        }
    }

    // --- 4. Seed the weight-2 column set ---
    std::set<uint32_t> w2_cols;
    for (uint32_t c = 0; c < n_cols; ++c) {
        if (col_weight[c] == 2) w2_cols.insert(c);
    }

    // --- 5. Process weight-2 columns ---
    uint32_t merges = 0;

    while (!w2_cols.empty()) {
        uint32_t c = *w2_cols.begin();
        w2_cols.erase(w2_cols.begin());

        // Weight may have changed since insertion; re-check.
        if (col_weight[c] != 2) continue;

        // Find the two alive rows containing column c.
        uint32_t r1 = UINT32_MAX, r2 = UINT32_MAX;
        for (uint32_t r : col_rows[c]) {
            if (!row_alive[r]) continue;
            if (r1 == UINT32_MAX) r1 = r;
            else { r2 = r; break; }
        }
        if (r1 == UINT32_MAX || r2 == UINT32_MAX) continue;

        // XOR the two rows (symmetric difference over GF(2)).
        auto new_row = xor_rows(rows[r1], rows[r2]);

        // Create merge-tree internal node.
        tree.internal_nodes.push_back({node_of[r1], node_of[r2]});
        uint32_t new_node = tree.num_leaves +
                            static_cast<uint32_t>(tree.internal_nodes.size()) - 1;

        // Decrement weights for the two old rows.
        for (uint32_t col : rows[r1]) {
            col_weight[col]--;
        }
        for (uint32_t col : rows[r2]) {
            col_weight[col]--;
        }

        // Clean stale col_rows entries to prevent phantom merges.
        // r2 is about to die — remove it from every column's inverted index.
        for (uint32_t col : rows[r2]) {
            auto& cr = col_rows[col];
            cr.erase(std::remove(cr.begin(), cr.end(), r2), cr.end());
        }
        // r1 survives but its column set changed (old → new_row).
        // Remove r1 from columns that cancelled in the XOR.
        for (uint32_t col : rows[r1]) {
            if (!std::binary_search(new_row.begin(), new_row.end(), col)) {
                auto& cr = col_rows[col];
                cr.erase(std::remove(cr.begin(), cr.end(), r1), cr.end());
            }
        }

        // Increment weights for the new row and register in col_rows.
        // Only add r1 to col_rows[col] for columns NEW to r1's row (i.e., came from
        // r2). Columns already in the old rows[r1] retain their existing col_rows
        // entry and must not get a duplicate, which would cause a spurious self-merge.
        for (uint32_t col : new_row) {
            col_weight[col]++;
            if (!std::binary_search(rows[r1].begin(), rows[r1].end(), col)) {
                col_rows[col].push_back(r1);
            }
        }

        // Check for newly created weight-2 columns among all affected columns.
        // The affected set is the union of old-r1, old-r2, and new_row columns.
        // Since we already decremented old and incremented new, just scan new_row
        // plus any columns that were in old rows but not in new_row.
        // Simpler: scan all three sets.  Duplicates in w2_cols are harmless (set).
        for (uint32_t col : rows[r1]) {
            if (col_weight[col] == 2) w2_cols.insert(col);
        }
        for (uint32_t col : rows[r2]) {
            if (col_weight[col] == 2) w2_cols.insert(col);
        }
        for (uint32_t col : new_row) {
            if (col_weight[col] == 2) w2_cols.insert(col);
        }

        // Replace r1 with the merged row, kill r2.
        rows[r1] = std::move(new_row);
        node_of[r1] = new_node;
        row_alive[r2] = false;
        rows[r2].clear();
        rows[r2].shrink_to_fit();

        ++merges;
    }

    LOG(LOG_INFO) << "Weight-2 merges: " << merges << " columns eliminated, "
                  << n_rows << " -> " << (n_rows - merges) << " rows.";

    // --- 6. Column compaction (remove weight-0 columns) ---
    std::vector<uint32_t> old_to_new_col(n_cols, UINT32_MAX);
    uint32_t new_col_count = 0;
    for (uint32_t c = 0; c < n_cols; ++c) {
        if (col_weight[c] > 0) {
            old_to_new_col[c] = new_col_count++;
        }
    }

    // --- 7. Row compaction + CSR rebuild ---
    HostMatrixCSR reduced;
    reduced.n_cols = new_col_count;
    std::vector<uint32_t> new_row_map;

    reduced.row_offsets.push_back(0);
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (!row_alive[r]) continue;
        new_row_map.push_back(node_of[r]);

        for (uint32_t col : rows[r]) {
            uint32_t nc = old_to_new_col[col];
            assert(nc != UINT32_MAX && "alive row references dead column");
            reduced.col_indices.push_back(nc);
        }
        // rows[r] is already sorted (xor_rows preserves order, col compaction
        // is monotone), so no re-sort needed.
        reduced.row_offsets.push_back(
            static_cast<uint32_t>(reduced.col_indices.size()));
    }
    reduced.n_rows = static_cast<uint32_t>(new_row_map.size());

    LOG(LOG_INFO) << "After weight-2 + compaction: " << reduced.n_rows << " rows x "
                  << reduced.n_cols << " cols, " << reduced.col_indices.size() << " NNZ.";

    ValidateHostMatrixCSR(reduced);

    MergeResult result;
    result.reduced          = std::move(reduced);
    result.row_map          = std::move(new_row_map);
    result.merge_tree       = std::move(tree);
    result.merges_performed = merges;
    result.rows_removed     = merges;
    return result;
}

// ---------------------------------------------------------------------------
// M4: Higher-weight column merges (weight 3..k_max)
// ---------------------------------------------------------------------------

MergeResult MergeFilterPipeline::mergeHigherWeight(
    const MergeResult& input,
    uint32_t k_max,
    uint32_t max_weight)
{
    LOG_SET_MODULE("MergeFilter");

    const uint32_t n_rows = input.reduced.n_rows;
    const uint32_t n_cols = input.reduced.n_cols;

    // --- 1. Convert CSR to row-major ---
    std::vector<std::vector<uint32_t>> rows(n_rows);
    for (uint32_t r = 0; r < n_rows; ++r) {
        uint32_t start = input.reduced.row_offsets[r];
        uint32_t end   = input.reduced.row_offsets[r + 1];
        rows[r].assign(input.reduced.col_indices.begin() + start,
                       input.reduced.col_indices.begin() + end);
    }

    // --- 2. Compute initial average row weight for dynamic max_weight ---
    uint64_t total_nnz = 0;
    for (uint32_t r = 0; r < n_rows; ++r) {
        total_nnz += rows[r].size();
    }
    double avg_initial = static_cast<double>(total_nnz) /
                         static_cast<double>(std::max(1u, n_rows));
    uint32_t effective_max = std::max(max_weight,
                                      static_cast<uint32_t>(2.0 * avg_initial));

    LOG(LOG_INFO) << "Higher-weight merges: k_max=" << k_max
                  << ", max_weight=" << effective_max
                  << " (base " << max_weight
                  << ", 2*avg=" << static_cast<uint32_t>(2.0 * avg_initial) << "), "
                  << n_rows << " rows x " << n_cols << " cols, "
                  << total_nnz << " NNZ, avg weight "
                  << std::fixed << std::setprecision(1) << avg_initial << ".";

    // --- 3. Inherit merge tree and node mapping ---
    MergeTree tree = input.merge_tree;
    std::vector<uint32_t> node_of = input.row_map;
    std::vector<bool> row_alive(n_rows, true);

    // --- 4. Column weights + inverted index ---
    std::vector<uint32_t> col_weight(n_cols, 0);
    std::vector<std::vector<uint32_t>> col_rows(n_cols);
    for (uint32_t r = 0; r < n_rows; ++r) {
        for (uint32_t c : rows[r]) {
            col_weight[c]++;
            col_rows[c].push_back(r);
        }
    }

    // --- 5. Process columns by ascending weight ---
    uint32_t cols_eliminated = 0;
    uint32_t rows_removed = 0;
    uint32_t merges_performed = 0;
    uint32_t merges_skipped = 0;
    uint32_t cols_skipped_fillin = 0;

    for (uint32_t w = 3; w <= k_max; ++w) {
        // Collect columns with current weight w.
        std::vector<uint32_t> target_cols;
        for (uint32_t c = 0; c < n_cols; ++c) {
            if (col_weight[c] == w) target_cols.push_back(c);
        }

        uint32_t w_cols_elim = 0;

        for (uint32_t c : target_cols) {
            // Re-check: weight may have changed from prior merges at same w.
            if (col_weight[c] == 0) continue;
            if (col_weight[c] > k_max) continue;

            // Gather alive rows containing column c.
            std::vector<uint32_t> containing;
            for (uint32_t r : col_rows[c]) {
                if (row_alive[r]) containing.push_back(r);
            }
            if (containing.size() < 2) continue;

            // Choose lightest row as pivot (Markowitz heuristic).
            uint32_t pivot_idx = 0;
            for (size_t i = 1; i < containing.size(); ++i) {
                if (rows[containing[i]].size() < rows[containing[pivot_idx]].size()) {
                    pivot_idx = static_cast<uint32_t>(i);
                }
            }
            uint32_t pivot = containing[pivot_idx];

            // XOR pivot with each non-pivot row.
            bool any_merge = false;
            bool all_skipped = true;
            for (size_t i = 0; i < containing.size(); ++i) {
                if (i == pivot_idx) continue;
                uint32_t r = containing[i];

                auto candidate = xor_rows(rows[pivot], rows[r]);
                if (candidate.size() > effective_max) {
                    merges_skipped++;
                    continue;
                }

                all_skipped = false;

                // Create merge tree node.
                tree.internal_nodes.push_back({node_of[pivot], node_of[r]});
                uint32_t new_node = tree.num_leaves +
                    static_cast<uint32_t>(tree.internal_nodes.size()) - 1;

                // Update column weights: remove old row r's contribution.
                for (uint32_t col : rows[r]) col_weight[col]--;

                // Clean stale col_rows: remove r from columns cancelled by XOR.
                for (uint32_t col : rows[r]) {
                    if (!std::binary_search(candidate.begin(), candidate.end(), col)) {
                        auto& cr = col_rows[col];
                        cr.erase(std::remove(cr.begin(), cr.end(), r), cr.end());
                    }
                }

                // Add new row's contribution.  Guard against duplicate col_rows
                // entries for columns already present in the old rows[r].
                for (uint32_t col : candidate) {
                    col_weight[col]++;
                    if (!std::binary_search(rows[r].begin(), rows[r].end(), col)) {
                        col_rows[col].push_back(r);
                    }
                }

                rows[r] = std::move(candidate);
                node_of[r] = new_node;
                any_merge = true;
                merges_performed++;
            }

            if (any_merge) {
                // Remove pivot row and clean its col_rows entries.
                row_alive[pivot] = false;
                for (uint32_t col : rows[pivot]) {
                    col_weight[col]--;
                    auto& cr = col_rows[col];
                    cr.erase(std::remove(cr.begin(), cr.end(), pivot), cr.end());
                }
                rows[pivot].clear();
                rows[pivot].shrink_to_fit();
                rows_removed++;
                cols_eliminated++;
                w_cols_elim++;
            } else if (all_skipped && containing.size() >= 2) {
                cols_skipped_fillin++;
            }
        }

        if (w_cols_elim > 0 || !target_cols.empty()) {
            LOG(LOG_DEBUG_1) << "Weight-" << w << " pass: "
                             << target_cols.size() << " candidates, "
                             << w_cols_elim << " eliminated.";
        }
    }

    LOG(LOG_INFO) << "Higher-weight merges: "
                  << cols_eliminated << " columns eliminated, "
                  << rows_removed << " rows removed, "
                  << merges_performed << " merges performed, "
                  << merges_skipped << " merges skipped (fill-in > " << effective_max << "), "
                  << cols_skipped_fillin << " columns skipped entirely.";

    // --- 6. Column compaction ---
    std::vector<uint32_t> old_to_new_col(n_cols, UINT32_MAX);
    uint32_t new_col_count = 0;
    for (uint32_t c = 0; c < n_cols; ++c) {
        if (col_weight[c] > 0) old_to_new_col[c] = new_col_count++;
    }

    // --- 7. Row compaction + CSR rebuild ---
    HostMatrixCSR reduced;
    reduced.n_cols = new_col_count;
    std::vector<uint32_t> new_row_map;

    reduced.row_offsets.push_back(0);
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (!row_alive[r]) continue;
        new_row_map.push_back(node_of[r]);

        for (uint32_t col : rows[r]) {
            uint32_t nc = old_to_new_col[col];
            assert(nc != UINT32_MAX && "alive row references dead column");
            reduced.col_indices.push_back(nc);
        }
        // rows[r] are sorted (xor_rows preserves order, col compaction is monotone).
        uint32_t row_start = reduced.row_offsets.back();
        std::sort(reduced.col_indices.begin() + row_start,
                  reduced.col_indices.end());
        reduced.row_offsets.push_back(
            static_cast<uint32_t>(reduced.col_indices.size()));
    }
    reduced.n_rows = static_cast<uint32_t>(new_row_map.size());

    // Compute final average row weight.
    double avg_final = reduced.col_indices.size() /
                       static_cast<double>(std::max(1u, reduced.n_rows));

    LOG(LOG_INFO) << "After higher-weight merges: " << reduced.n_rows << " rows x "
                  << reduced.n_cols << " cols, " << reduced.col_indices.size()
                  << " NNZ, avg weight " << std::fixed << std::setprecision(1)
                  << avg_final << ".";

    if (avg_final > 1.5 * avg_initial) {
        LOG(LOG_WARNING) << "Average row weight increased by "
                         << std::fixed << std::setprecision(0)
                         << ((avg_final / avg_initial - 1.0) * 100.0)
                         << "% (from " << std::setprecision(1) << avg_initial
                         << " to " << avg_final << "). Fill-in may be excessive.";
    }

    ValidateHostMatrixCSR(reduced);

    MergeResult result;
    result.reduced          = std::move(reduced);
    result.row_map          = std::move(new_row_map);
    result.merge_tree       = std::move(tree);
    result.merges_performed = cols_eliminated;
    result.rows_removed     = rows_removed;
    return result;
}

} // namespace matrix
} // namespace mpqs
