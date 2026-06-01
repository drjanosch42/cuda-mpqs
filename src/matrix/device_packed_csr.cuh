// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/device_packed_csr.cuh
//
// Packed CSR types for device-resident matrix preprocessing (M9v2).
// Each CSR entry stores (col_index << 8) | exponent in a single uint32_t.
// Per-row metadata (sqrt_Q, signs, val_2_exps) enables merge-tree-free
// sqrt reconstruction.

#pragma once

#include "matrix_constructor.h"  // HostMatrixCSR
#include "uint512.cuh"           // uint512
#include <cuda_runtime.h>
#include <cstdint>

namespace mpqs {
namespace matrix {

// ============================================================================
// Packed entry format: 24-bit column index + 8-bit exponent
// ============================================================================

/// A packed CSR entry: 24-bit column index + 8-bit exponent in a single uint32_t.
/// Column range: [0, 2^24) = 16M — sufficient up to RSA-200.
/// Exponent range: [0, 255] — sufficient for all practical MPQS.
using PackedEntry = uint32_t;

/// Extract column index from packed entry.
__host__ __device__ __forceinline__
uint32_t packed_col(PackedEntry e) { return e >> 8; }

/// Extract exponent from packed entry.
__host__ __device__ __forceinline__
uint8_t packed_exp(PackedEntry e) { return static_cast<uint8_t>(e & 0xFF); }

/// Construct packed entry from column index and exponent.
__host__ __device__ __forceinline__
PackedEntry make_packed(uint32_t col, uint8_t exp) { return (col << 8) | exp; }

// ============================================================================
// DevicePackedView — non-owning kernel-passable POD view
// ============================================================================

/// Non-owning kernel-passable view of a packed CSR matrix + per-row metadata.
struct DevicePackedView {
    // CSR structure (packed entries)
    uint32_t*    d_row_offsets;   ///< [n_rows + 1]
    PackedEntry* d_entries;       ///< [nnz] — (col_index << 8) | exponent

    // Per-row metadata (SoA)
    uint512*     d_sqrt_Q;       ///< [n_rows] — Montgomery product of constituent sqrt_Q values
    uint8_t*     d_signs;        ///< [n_rows] — product of signs
    int32_t*     d_val_2_exps;   ///< [n_rows] — sum of val_2 exponents

    // Dimensions
    uint32_t     n_rows;
    uint32_t     n_cols;
    uint32_t     nnz;
};

// ============================================================================
// DevicePackedCSR — RAII owner of device packed CSR buffers
// ============================================================================

/// RAII owner of device packed CSR buffers. Non-copyable, non-movable.
/// Holds both CSR structure and per-row 1-partial metadata.
struct DevicePackedCSR {
    // CSR structure
    uint32_t*    d_row_offsets = nullptr;
    PackedEntry* d_entries     = nullptr;

    // Per-row metadata
    uint512*     d_sqrt_Q      = nullptr;
    uint8_t*     d_signs       = nullptr;
    int32_t*     d_val_2_exps  = nullptr;

    // Dimensions
    uint32_t     n_rows = 0;
    uint32_t     n_cols = 0;
    uint32_t     nnz    = 0;
    bool         use_managed = false;

    DevicePackedCSR() = default;
    ~DevicePackedCSR();

    // Non-copyable, movable
    DevicePackedCSR(const DevicePackedCSR&) = delete;
    DevicePackedCSR& operator=(const DevicePackedCSR&) = delete;
    DevicePackedCSR(DevicePackedCSR&&) noexcept;
    DevicePackedCSR& operator=(DevicePackedCSR&&) noexcept;

    /// Allocate device buffers for the packed CSR and per-row metadata.
    /// @param rows     Number of rows.
    /// @param cols     Number of columns (fb_size + 2 + n_lp_cols).
    /// @param nnz_cap  Total number of packed entries.
    /// @param jetson   If true, use cudaMallocManaged for all buffers.
    void alloc(uint32_t rows, uint32_t cols, uint32_t nnz_cap, bool jetson);

    /// Return a non-owning kernel-passable view.
    DevicePackedView view() const;

    /// Download CSR structure to host HostMatrixCSR.
    /// Extracts column indices from packed entries (entry >> 8) for backward
    /// compatibility with the BW solver. Callers needing exponents access
    /// packed entries directly via view().
    void downloadToHost(HostMatrixCSR& out, cudaStream_t stream = 0) const;
};

// ============================================================================
// Jetson detection
// ============================================================================

/// Detect Jetson platform. SM 8.7 or unified addressing with < 12 GB VRAM.
/// NEVER uses concurrentManagedAccess (Blackwell SM 12.0 also reports 0).
bool isJetsonDevice();

} // namespace matrix
} // namespace mpqs
