// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_packed_expanded.cuh
//
// GPU packed expanded matrix construction (M9a).
// Builds the (F+2+L)-column packed CSR directly on device from
// device-resident relation data. Each entry stores (col_index << 8) | exponent.

#pragma once

#include "device_packed_csr.cuh"
#include "mpqs_soa.h"   // RelationBatchView
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Result of GPU packed expanded matrix construction.
struct GpuPackedExpandedResult {
    DevicePackedCSR csr;       ///< Device-resident packed CSR with per-row metadata
    uint32_t        n_lp_cols; ///< Number of distinct LP columns
};

/// Build expanded (F+2+L)-column packed CSR on device from device-resident relation data.
/// Solo mode only. Each entry stores (col_index << 8) | exponent.
/// Per-row metadata (sqrt_Q, signs, val_2_exps) copied from relation batches.
///
/// LP column assignment is deterministic (sort-based ascending LP value order),
/// which is a column permutation relative to the CPU ExpandedMatrixBuilder's
/// insertion-order assignment.
///
/// @param smooth_view  Device pointers to persistent smooth relation batch.
/// @param n_smooth     Number of smooth relations.
/// @param partial_view Device pointers to LP witness (partial) relation batch.
/// @param n_partial    Number of partial relations.
/// @param fb_size      Factor base size (number of FB primes, excluding sign/prime-2).
/// @return GpuPackedExpandedResult with device-resident packed CSR and LP column count.
GpuPackedExpandedResult gpuBuildPackedMatrix(
    const structures::RelationBatchView& smooth_view,
    uint64_t n_smooth,
    const structures::RelationBatchView& partial_view,
    uint64_t n_partial,
    uint32_t fb_size);

} // namespace matrix
} // namespace mpqs
