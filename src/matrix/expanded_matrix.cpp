// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "expanded_matrix.h"
#include "matrix_utils.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <iomanip>

namespace mpqs {
namespace matrix {

/// Count columns for a single relation row (shared logic for smooth and partial).
/// @param lp_col  LP column index, or UINT32_MAX if none (full smooth).
static uint32_t countRowCols(
    const structures::HostRelationBatch& batch, size_t rel_idx,
    uint32_t lp_col)
{
    uint32_t count = 0;

    // Column 0: sign (matches GPU kernel convention: sign != 1 → column present)
    if (batch.signs[rel_idx] != 1) ++count;

    // Column 1: prime-2 parity
    if (batch.val_2_exps[rel_idx] & 1) ++count;

    // FB columns: odd-exponent factors
    uint64_t off_start = batch.factor_offsets[rel_idx];
    uint64_t off_end   = batch.factor_offsets[rel_idx + 1];
    for (uint64_t j = off_start; j < off_end; ++j) {
        if (batch.factor_counts[j] & 1) ++count;
    }

    // LP column (partials only)
    if (lp_col != UINT32_MAX) ++count;

    return count;
}

/// Fill column indices for a single relation row. Columns are written in sorted order
/// (sign < prime-2 < FB indices < LP column, since FB indices are naturally >= 2 and
/// LP columns start at fb_size+2 which is always > any FB column).
/// @return Number of columns written.
static uint32_t fillRowCols(
    const structures::HostRelationBatch& batch, size_t rel_idx,
    uint32_t lp_col,
    uint32_t* out)
{
    uint32_t pos = 0;

    // Column 0: sign
    if (batch.signs[rel_idx] != 1) {
        out[pos++] = 0;
    }

    // Column 1: prime-2 parity
    if (batch.val_2_exps[rel_idx] & 1) {
        out[pos++] = 1;
    }

    // FB columns: collect odd-exponent factor indices (+2 offset)
    uint64_t off_start = batch.factor_offsets[rel_idx];
    uint64_t off_end   = batch.factor_offsets[rel_idx + 1];
    uint32_t fb_start = pos;
    for (uint64_t j = off_start; j < off_end; ++j) {
        if (batch.factor_counts[j] & 1) {
            out[pos++] = batch.factor_indices[j] + 2;
        }
    }

    // Sort the FB columns (factor_indices may not be sorted)
    std::sort(out + fb_start, out + pos);

    // LP column (always larger than any FB column, so appending preserves sort order)
    if (lp_col != UINT32_MAX) {
        out[pos++] = lp_col;
    }

    return pos;
}

ExpandedMatrixResult ExpandedMatrixBuilder::build(
    const structures::HostRelationBatch& smooths,
    const structures::HostRelationBatch& partials,
    uint32_t fb_size)
{
    LOG_SET_MODULE("Matrix");

    // -------------------------------------------------------------------------
    // Phase 1: LP column assignment
    // -------------------------------------------------------------------------
    std::unordered_map<uint64_t, uint32_t> lp_col_map;
    lp_col_map.reserve(partials.num_relations);
    uint32_t next_lp_col = fb_size + 2;

    for (size_t i = 0; i < partials.num_relations; ++i) {
        uint64_t lp = static_cast<uint64_t>(partials.large_primes[i]);
        assert(lp > 1 && "Partial relation must have LP > 1");
        assert(lp < (1ULL << 48) && "LP value exceeds safe uint64_t range");
        auto [it, inserted] = lp_col_map.emplace(lp, next_lp_col);
        if (inserted) ++next_lp_col;
    }

    uint32_t num_lp_cols = next_lp_col - (fb_size + 2);
    LOG(LOG_INFO) << "Expanded matrix: " << num_lp_cols << " distinct LP columns assigned";

    // -------------------------------------------------------------------------
    // Phase 2: CSR construction (two-pass)
    // -------------------------------------------------------------------------
    uint32_t total_rows = static_cast<uint32_t>(smooths.num_relations + partials.num_relations);
    uint32_t total_cols = fb_size + 2 + num_lp_cols;

    // Pass 1: count NNZ per row and compute row offsets
    std::vector<uint32_t> row_offsets(total_rows + 1, 0);

    // Smooth rows
    for (size_t i = 0; i < smooths.num_relations; ++i) {
        row_offsets[i + 1] = countRowCols(smooths, i, UINT32_MAX);
    }

    // Partial rows
    for (size_t i = 0; i < partials.num_relations; ++i) {
        uint64_t lp = static_cast<uint64_t>(partials.large_primes[i]);
        uint32_t lp_col = lp_col_map[lp];
        uint32_t row = static_cast<uint32_t>(smooths.num_relations + i);
        row_offsets[row + 1] = countRowCols(partials, i, lp_col);
    }

    // Prefix sum
    for (uint32_t i = 1; i <= total_rows; ++i) {
        row_offsets[i] += row_offsets[i - 1];
    }

    uint64_t total_nnz = row_offsets[total_rows];

    // Pass 2: fill column indices
    std::vector<uint32_t> col_indices(total_nnz);

    // Smooth rows
    for (size_t i = 0; i < smooths.num_relations; ++i) {
        fillRowCols(smooths, i, UINT32_MAX, col_indices.data() + row_offsets[i]);
    }

    // Partial rows
    for (size_t i = 0; i < partials.num_relations; ++i) {
        uint64_t lp = static_cast<uint64_t>(partials.large_primes[i]);
        uint32_t lp_col = lp_col_map[lp];
        uint32_t row = static_cast<uint32_t>(smooths.num_relations + i);
        fillRowCols(partials, i, lp_col, col_indices.data() + row_offsets[row]);
    }

    // -------------------------------------------------------------------------
    // Phase 3: Assemble result and validate
    // -------------------------------------------------------------------------
    HostMatrixCSR csr;
    csr.row_offsets = std::move(row_offsets);
    csr.col_indices = std::move(col_indices);
    csr.n_rows = total_rows;
    csr.n_cols = total_cols;

    // Log matrix statistics
    double density = (total_rows > 0 && total_cols > 0)
        ? static_cast<double>(total_nnz) / (static_cast<double>(total_rows) * total_cols) * 100.0
        : 0.0;
    double avg_per_row = (total_rows > 0)
        ? static_cast<double>(total_nnz) / total_rows
        : 0.0;
    LOG(LOG_INFO) << "Expanded matrix: " << fmtNum(total_rows) << " rows x "
                  << fmtNum(total_cols) << " cols, " << fmtNum(total_nnz)
                  << " NNZ (" << std::fixed << std::setprecision(3) << density
                  << "% density, avg " << std::setprecision(1) << avg_per_row << " per row)";
    LOG(LOG_INFO) << "  Smooth rows: " << fmtNum(smooths.num_relations)
                  << ", Partial rows: " << fmtNum(partials.num_relations);

    // LP column weight histogram
    std::vector<uint32_t> lp_col_weights(num_lp_cols, 0);
    for (size_t i = 0; i < partials.num_relations; ++i) {
        uint64_t lp = static_cast<uint64_t>(partials.large_primes[i]);
        uint32_t col_idx = lp_col_map[lp] - (fb_size + 2);
        ++lp_col_weights[col_idx];
    }
    uint32_t w1 = 0, w2 = 0, w3plus = 0;
    for (uint32_t w : lp_col_weights) {
        if (w == 1) ++w1;
        else if (w == 2) ++w2;
        else if (w >= 3) ++w3plus;
    }
    LOG(LOG_INFO) << "  LP column weights: " << w1 << " singletons, "
                  << w2 << " pairs, " << w3plus << " weight-3+";

    // Validate CSR
    if (!ValidateHostMatrixCSR(csr)) {
        LOG(LOG_ERROR_CRITICAL) << "Expanded matrix CSR validation FAILED";
        return {};
    }

    ExpandedMatrixResult result;
    result.matrix = std::move(csr);
    result.lp_col_map = std::move(lp_col_map);
    result.num_lp_cols = num_lp_cols;
    return result;
}

} // namespace matrix
} // namespace mpqs
