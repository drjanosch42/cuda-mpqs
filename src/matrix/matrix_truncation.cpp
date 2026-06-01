// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "matrix_truncation.h"
#include "hpc_logger.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace mpqs {
namespace matrix {

namespace {

/// Per-column admission count for greedy coverage selection.
/// A column is "covered" once at least one admitted row references it.
constexpr uint32_t kCoverageThreshold = 1;

/// Maximum 10%-step relaxation iterations before giving up on coverage.
/// In practice the coverage-greedy selector hits coverage in pass 1 and this
/// loop is never entered. The bound exists only to prevent an infinite loop
/// on a pathological input.
constexpr uint32_t kRelaxationMaxIters = 64;

} // namespace

TruncationResult truncateMatrix(
    const HostMatrixCSR& input,
    const std::vector<uint32_t>& input_row_map,
    double truncation_factor,
    uint32_t n_extra_cols,
    uint32_t k_excess)
{
    LOG_SET_MODULE("MergeFilter");

    const uint32_t n_rows = input.n_rows;
    const uint32_t n_cols = input.n_cols;

    // Disabled by sentinel.
    if (truncation_factor <= 0.0) {
        TruncationResult result;
        result.truncated = input;
        result.row_map = input_row_map;
        return result;
    }

    // Target = post-augmentation columns + small excess. Capped at n_rows so the
    // function is still a no-op when the matrix is already at or below target.
    const uint64_t target64 = static_cast<uint64_t>(n_cols)
                              + static_cast<uint64_t>(n_extra_cols)
                              + static_cast<uint64_t>(k_excess);
    const uint32_t target = static_cast<uint32_t>(
        std::min<uint64_t>(target64, static_cast<uint64_t>(n_rows)));

    LOG(LOG_INFO) << "Truncation: input " << n_rows << " rows x " << n_cols
                  << " cols, target " << target << " rows (n_cols=" << n_cols
                  << " + n_extra_cols=" << n_extra_cols
                  << " + k_excess=" << k_excess << ").";

    if (n_rows <= target) {
        LOG(LOG_INFO) << "Truncation: " << n_rows << " rows <= target "
                      << target << " — no truncation needed.";
        TruncationResult result;
        result.truncated = input;  // copy
        result.row_map = input_row_map;
        return result;
    }

    auto weight = [&](uint32_t r) -> uint32_t {
        return input.row_offsets[r + 1] - input.row_offsets[r];
    };

    // 1. Pre-compute per-column total coverage on the *full* input. A column
    //    that has zero rows referencing it in the input is structurally
    //    absent and is not a coverage requirement (it would be removed by
    //    column compaction at step 4 regardless).
    std::vector<uint32_t> input_col_count(n_cols, 0);
    for (uint32_t c : input.col_indices) {
        if (c < n_cols) ++input_col_count[c];
    }
    uint32_t cols_with_coverage = 0;
    for (uint32_t c = 0; c < n_cols; ++c)
        if (input_col_count[c] > 0) ++cols_with_coverage;

    // 2. Build row-index list, ordered by ascending weight (deterministic
    //    tie-break by row index). The order matters for two reasons:
    //      (a) the coverage-greedy pass walks rows in this order, admitting
    //          any row that contributes a previously-uncovered column. Lighter
    //          rows go first so that, when coverage is already satisfied, the
    //          fill phase naturally biases towards sparse rows for downstream
    //          BW cache locality.
    //      (b) determinism: identical inputs must produce identical truncated
    //          matrices across runs.
    std::vector<uint32_t> indices(n_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&](uint32_t a, uint32_t b) {
            uint32_t wa = weight(a), wb = weight(b);
            return (wa != wb) ? (wa < wb) : (a < b);
        });

    // 3. Coverage-greedy admit pass. `admitted_coverage[c]` counts rows in
    //    the admitted set that touch column c. Admit row r iff
    //    weight(r) > 0 AND r introduces a column whose admitted_coverage is
    //    below kCoverageThreshold.
    std::vector<uint8_t> admitted(n_rows, 0);
    std::vector<uint32_t> admitted_coverage(n_cols, 0);
    uint32_t admitted_count = 0;
    uint32_t covered_cols = 0;

    auto admit_row = [&](uint32_t r) {
        admitted[r] = 1;
        ++admitted_count;
        const uint32_t start = input.row_offsets[r];
        const uint32_t end   = input.row_offsets[r + 1];
        for (uint32_t j = start; j < end; ++j) {
            uint32_t c = input.col_indices[j];
            if (c < n_cols) {
                if (admitted_coverage[c] == 0) ++covered_cols;
                ++admitted_coverage[c];
            }
        }
    };

    for (uint32_t r : indices) {
        if (admitted_count >= target) break;
        const uint32_t start = input.row_offsets[r];
        const uint32_t end   = input.row_offsets[r + 1];
        bool useful = false;
        for (uint32_t j = start; j < end; ++j) {
            uint32_t c = input.col_indices[j];
            if (c < n_cols && admitted_coverage[c] < kCoverageThreshold) {
                useful = true;
                break;
            }
        }
        if (useful) admit_row(r);
    }

    // 4. Fill pass: top up to `target` rows with the next lightest unadmitted
    //    rows. Cache locality for BW prefers low NNZ; the coverage pass already
    //    captured the structurally essential rows so this is a pure budget fill.
    if (admitted_count < target) {
        for (uint32_t r : indices) {
            if (admitted_count >= target) break;
            if (!admitted[r]) admit_row(r);
        }
    }

    // 5. Relaxation fallback (safety net). If any column with input coverage
    //    is uncovered in the admitted set, grow `target` and re-admit. This
    //    code path should rarely fire now that step 3 explicitly guards
    //    coverage; we keep it because the column-compaction at step 8 would
    //    otherwise silently drop columns that the BW solver expects to see.
    bool relaxation_fired = false;
    if (covered_cols < cols_with_coverage) {
        relaxation_fired = true;
        LOG(LOG_WARNING) << "Truncation coverage fallback triggered: "
                         << covered_cols << " / " << cols_with_coverage
                         << " input-covered cols admitted at target=" << target
                         << " rows. Relaxing target.";

        uint32_t relaxed_target = target;
        for (uint32_t iter = 0; iter < kRelaxationMaxIters; ++iter) {
            if (covered_cols >= cols_with_coverage) break;
            if (relaxed_target >= n_rows) break;

            uint32_t new_target = static_cast<uint32_t>(
                std::min<uint64_t>(static_cast<uint64_t>(std::ceil(relaxed_target * 1.1)),
                                   static_cast<uint64_t>(n_rows)));
            if (new_target == relaxed_target) new_target = std::min(relaxed_target + 1, n_rows);
            relaxed_target = new_target;

            // Prefer rows that contribute uncovered columns first.
            for (uint32_t r : indices) {
                if (admitted[r]) continue;
                if (admitted_count >= relaxed_target) break;
                const uint32_t start = input.row_offsets[r];
                const uint32_t end   = input.row_offsets[r + 1];
                bool useful = false;
                for (uint32_t j = start; j < end; ++j) {
                    uint32_t c = input.col_indices[j];
                    if (c < n_cols && admitted_coverage[c] < kCoverageThreshold) {
                        useful = true;
                        break;
                    }
                }
                if (useful) admit_row(r);
            }
            // If still under-budget after coverage attempts, fill.
            if (admitted_count < relaxed_target) {
                for (uint32_t r : indices) {
                    if (admitted[r]) continue;
                    if (admitted_count >= relaxed_target) break;
                    admit_row(r);
                }
            }
        }

        if (covered_cols < cols_with_coverage) {
            LOG(LOG_WARNING) << "Truncation coverage fallback exhausted: "
                             << covered_cols << " / " << cols_with_coverage
                             << " input-covered cols still uncovered after relax to "
                             << relaxed_target << " rows. Skipping truncation.";
            TruncationResult result;
            result.truncated = input;
            result.row_map = input_row_map;
            return result;
        }
        LOG(LOG_INFO) << "Truncation coverage fallback resolved: "
                      << covered_cols << " / " << cols_with_coverage
                      << " input-covered cols at " << admitted_count << " rows.";
    }

    // 6. Materialise admitted index list, ordered by original row index for
    //    CSR contiguity. (BW + sqrt expect ascending row id.)
    std::vector<uint32_t> selected;
    selected.reserve(admitted_count);
    for (uint32_t r = 0; r < n_rows; ++r) {
        if (admitted[r]) selected.push_back(r);
    }
    const uint32_t effective_k = static_cast<uint32_t>(selected.size());

    // 7. Build truncated CSR and new row_map.
    HostMatrixCSR truncated;
    truncated.row_offsets.reserve(effective_k + 1);
    truncated.row_offsets.push_back(0);

    std::vector<uint32_t> new_row_map;
    new_row_map.reserve(effective_k);

    for (uint32_t r : selected) {
        new_row_map.push_back(input_row_map[r]);
        uint32_t start = input.row_offsets[r];
        uint32_t end   = input.row_offsets[r + 1];
        for (uint32_t j = start; j < end; ++j) {
            truncated.col_indices.push_back(input.col_indices[j]);
        }
        truncated.row_offsets.push_back(
            static_cast<uint32_t>(truncated.col_indices.size()));
    }
    truncated.n_rows = effective_k;

    // 8. Column compaction: remove columns that became empty after row selection.
    //    Coverage-greedy admit minimises this set, but rows pure-empty in the
    //    input or columns the heuristic could not touch (none, by construction)
    //    still need stripping so BW sees a tight column count.
    std::vector<uint32_t> col_weight(n_cols, 0);
    for (uint32_t c : truncated.col_indices) col_weight[c]++;

    std::vector<uint32_t> old_to_new(n_cols, UINT32_MAX);
    uint32_t new_col_count = 0;
    uint32_t zero_coverage_cols = 0;
    for (uint32_t c = 0; c < n_cols; ++c) {
        if (col_weight[c] > 0) {
            old_to_new[c] = new_col_count++;
        } else if (input_col_count[c] > 0) {
            ++zero_coverage_cols;
        }
    }

    for (auto& c : truncated.col_indices) {
        c = old_to_new[c];
    }
    truncated.n_cols = new_col_count;

    const uint32_t rows_removed = n_rows - effective_k;
    const uint32_t cols_removed = n_cols - new_col_count;

    LOG(LOG_INFO) << "Truncation: " << n_rows << " -> " << effective_k << " rows ("
                  << rows_removed << " removed), "
                  << n_cols << " -> " << new_col_count << " cols ("
                  << cols_removed << " empty cols removed). "
                  << "Zero-coverage input cols dropped: " << zero_coverage_cols
                  << ", relaxation fallback fired: " << (relaxation_fired ? "yes" : "no")
                  << ". n_extra_cols=" << n_extra_cols << " accounted for.";

    TruncationResult result;
    result.truncated   = std::move(truncated);
    result.row_map     = std::move(new_row_map);
    result.rows_removed = rows_removed;
    result.cols_removed = cols_removed;
    return result;
}

} // namespace matrix
} // namespace mpqs
