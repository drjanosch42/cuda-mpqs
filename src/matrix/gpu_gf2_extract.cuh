// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_gf2_extract.cuh
//
// GF(2) extraction from packed merged CSR (M9f).
// Two-pass: count odd-exponent entries per row → prefix sum → write col_indices.
// Produces binary CSR identical to the result of binary XOR merges (audit v4 §10.5).

#pragma once

#include "device_packed_csr.cuh"   // PackedEntry, packed_col, packed_exp
#include "matrix_constructor.h"     // HostMatrixCSR
#include "gpu_batch_merge.cuh"      // DeviceMergeWorkspace, BatchMergeResult, ROW_DEAD, ROW_WS_BIT
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace mpqs {
namespace matrix {

/// Result of GF(2) extraction from packed merged CSR.
struct GF2ExtractionResult {
    HostMatrixCSR gf2_csr;              ///< Binary CSR for BW solver
    std::vector<uint32_t> row_map;      ///< gf2_row → merged_row logical index
    uint32_t gf2_nnz = 0;              ///< Total NNZ in GF(2) matrix
};

/// Extract GF(2) binary CSR from packed merged data.
///
/// Two-phase operation:
/// 1. Flatten: iterate h_row_ptr, for each alive row, read entries from
///    original CSR or workspace (via row_ptr indirection).
/// 2. Filter: keep entries with odd exponents, emit column indices only.
///
/// The resulting CSR is identical to the binary CSR that would result from
/// performing all merges as GF(2) XOR operations (proof in audit v4 §10.5).
///
/// @param csr           Original packed CSR (post-singleton, device-resident).
/// @param merge_result  Result from gpuBatchMerge() — workspace + h_row_ptr.
/// @return GF2ExtractionResult with binary CSR on host and row mapping.
GF2ExtractionResult gpuExtractGF2(
    const DevicePackedCSR& csr,
    const BatchMergeResult& merge_result);

} // namespace matrix
} // namespace mpqs
