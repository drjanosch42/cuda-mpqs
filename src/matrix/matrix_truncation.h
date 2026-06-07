// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include "matrix_constructor.h"
#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Result of matrix truncation.
struct TruncationResult {
    HostMatrixCSR truncated;             ///< Truncated + column-compacted CSR
    std::vector<uint32_t> row_map;       ///< truncated_row -> merge tree node
    uint32_t rows_removed = 0;           ///< Number of rows dropped
    uint32_t cols_removed = 0;           ///< Columns removed by compaction
};

/// Truncate a preprocessed matrix to a row count just above the post-augmentation
/// column count. Coverage-aware row selection preserves LP-pair survival on
/// high-LP data: every column with at least one input row is required to be
/// covered by at least one selected row. Once coverage is satisfied, the
/// remaining row budget is filled with lightest-weight rows for cache locality.
///
/// Target row count (when `truncation_factor > 0`):
///     target_rows = max(input.n_cols + n_extra_cols + k_excess,
///                       <coverage minimum>)
///
/// where `n_extra_cols` is the number of columns about to be appended *after*
/// truncation (typically the character-column count, default 32 in this
/// codebase) and `k_excess` is the excess of rows over that final column count.
///
/// Backward-compatibility: when `truncation_factor <= 0`, no truncation is
/// performed and the input is returned unchanged. Otherwise the factor is used
/// only as an on/off switch — the actual target is excess-based, not
/// factor-based, because the factor-based target is char-col-blind and
/// produced a no-op or destructive truncation on high-LP data.
///
/// Selection priority (per row r):
///   1. Coverage-greedy admit pass: traverse rows in moderate-weight order
///      (deterministic by ascending Hamming weight, tie-break by row index).
///      Admit any row that newly covers at least one column whose admitted
///      coverage is currently below `kCoverageThreshold` (default 1).
///   2. Fill pass: if fewer than `target_rows` rows admitted, admit remaining
///      rows in ascending-weight order until `target_rows` is hit.
///   3. Relaxation fallback (rare safety net): if any column has zero coverage
///      after step 1+2, progressively grow target_rows by 10% until either
///      every column is covered or target_rows == n_rows. A warning is logged
///      whenever this fallback fires; the previous 50% column-survival
///      heuristic is retained in spirit but should rarely trigger now that
///      the primary selector is coverage-greedy.
///
/// @param input            Preprocessed CSR matrix (post-merges, post-singletons).
/// @param input_row_map    Row map from merge pipeline (row -> merge tree node).
/// @param truncation_factor  > 0 enables truncation, <= 0 disables it.
///                           Otherwise unused for target computation
///                           (target is `n_cols + n_extra_cols + k_excess`).
/// @param n_extra_cols     Number of columns to be appended *after* truncation
///                         (e.g. 32 for character columns). Default 0.
/// @param k_excess         Excess rows above `n_cols + n_extra_cols`.
///                         Default 200 (chosen so BW is barely overdetermined
///                         for the user's RSA-110 strategy; see audit App C).
/// @return TruncationResult with reduced matrix and updated row_map.
TruncationResult truncateMatrix(
    const HostMatrixCSR& input,
    const std::vector<uint32_t>& input_row_map,
    double truncation_factor = 1.05,
    uint32_t n_extra_cols    = 0,
    uint32_t k_excess        = 200);

} // namespace matrix
} // namespace mpqs
