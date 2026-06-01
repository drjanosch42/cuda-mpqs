// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_singleton_packed.cuh
//
// Packed GPU singleton removal: in-place iterative fixpoint over a device-resident
// packed CSR matrix. Adapts the binary gpu_singleton.cu pipeline for packed entries
// ((col_index << 8) | exponent) and adds per-row metadata compaction.
//
// The existing gpuRemoveSingletons() (binary, for cluster mode) is retained
// unmodified. This packed variant operates directly on DevicePackedCSR from M9a —
// no H→D upload needed.

#pragma once

#include "device_packed_csr.cuh"  // DevicePackedCSR, PackedEntry, packed_col(), packed_exp(), make_packed()

#include <vector>
#include <cstdint>

namespace mpqs {
namespace matrix {

/// Result of GPU packed singleton removal. Input DevicePackedCSR is consumed
/// (its device buffers are freed; callers must not access the input after this call).
struct PackedSingletonResult {
    DevicePackedCSR       reduced;       ///< Compacted packed CSR (device-resident)
    std::vector<uint32_t> row_map;       ///< reduced_row → original_row (host)
    std::vector<uint32_t> col_map;       ///< new_col → old_col (host)
    uint32_t  iterations   = 0;
    uint32_t  rows_removed = 0;
    uint32_t  cols_removed = 0;
};

/// In-place GPU singleton removal on a device-resident packed CSR matrix.
///
/// Operates directly on DevicePackedCSR (no H→D upload — M9a built the matrix
/// on device). Extracts column indices via packed_col(entry) = entry >> 8 and
/// remaps the column-index portion of each entry after compaction while preserving
/// the 8-bit exponent. Compacts per-row metadata (sqrt_Q, signs, val_2_exps) in
/// parallel with the entry compaction.
///
/// The singleton-free submatrix is uniquely determined by the input regardless of
/// GPU parallel removal order, matching the result of gpuRemoveSingletons() on the
/// equivalent binary CSR.
///
/// @param device_csr  Packed CSR from gpuBuildPackedMatrix() (M9a). Consumed on
///                    return: all 5 device arrays are freed and nulled.
/// @return PackedSingletonResult with reduced packed CSR (device-resident), host
///         row_map and col_map, and removal diagnostics.
PackedSingletonResult gpuRemoveSingletons_packed(DevicePackedCSR& device_csr);

} // namespace matrix
} // namespace mpqs
