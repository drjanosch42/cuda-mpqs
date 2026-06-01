// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include "matrix_constructor.h"  // HostMatrixCSR, ValidateHostMatrixCSR
#include "mpqs_soa.h"            // HostRelationBatch
#include <unordered_map>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Result of expanded matrix construction.
struct ExpandedMatrixResult {
    HostMatrixCSR matrix;
    std::unordered_map<uint64_t, uint32_t> lp_col_map;  ///< LP value -> column index
    uint32_t num_lp_cols = 0;                             ///< Number of distinct LP columns
};

/// Builds the (F+2+L)-column GF(2) matrix retaining LP columns.
///
/// Full smooth rows have sign + prime-2 + FB prime columns only.
/// Partial rows additionally have exactly one LP column set.
/// Column layout: [0]=sign, [1]=prime-2, [2..F+1]=FB primes, [F+2..F+1+L]=LP columns.
class ExpandedMatrixBuilder {
public:
    /// Build the expanded matrix from separate smooth and partial batches.
    /// @param smooths   Full smooth relations (large_primes[i] == 1).
    /// @param partials  Raw partial relations (large_primes[i] > 1).
    /// @param fb_size   Factor base size (number of FB primes, excluding sign/prime-2).
    /// @return ExpandedMatrixResult with CSR matrix and LP column mapping.
    ExpandedMatrixResult build(
        const structures::HostRelationBatch& smooths,
        const structures::HostRelationBatch& partials,
        uint32_t fb_size);
};

} // namespace matrix
} // namespace mpqs
