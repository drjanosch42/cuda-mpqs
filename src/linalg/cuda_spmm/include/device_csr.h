// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include <cuda_runtime.h>

struct DeviceCSRSlice;

/// Canonical CSR representation of a sparse matrix on the GPU.
/// Owns all device memory; call free() before destruction.
struct DeviceCSR {
    uint32_t* d_row_ptr      = nullptr;  // [n_rows + 1] — CSR row pointer
    uint32_t* d_col_ind      = nullptr;  // [nnz] — sorted column indices per row
    uint32_t  n_rows         = 0;
    uint32_t  n_cols         = 0;
    uint64_t  nnz            = 0;

    // Density metadata (computed on-device after upload)
    uint32_t* d_density_perm = nullptr;  // [n_rows] — row permutation by descending NNZ
    uint32_t  n_dense_rows   = 0;        // Rows exceeding density threshold (M4RM boundary)

    /// Free all device memory and reset to defaults
    void free();

    /// Create a non-owning slice for rows [start_row, start_row + row_count).
    /// Reads 2 uint32_t from d_row_ptr to compute slice NNZ.
    /// The returned slice is invalidated if this DeviceCSR is freed or its
    /// arrays are reallocated (e.g., by device_csr_permute_by_density).
    DeviceCSRSlice slice(uint32_t start_row, uint32_t row_count) const;

    /// Total device memory in bytes
    size_t device_bytes() const;
};

/// Non-owning view into a contiguous row range of a DeviceCSR.
/// Does not own memory — invalidated if the parent DeviceCSR is freed or reallocated.
struct DeviceCSRSlice {
    const uint32_t* d_row_ptr;   // Points into parent's d_row_ptr + start_row
    const uint32_t* d_col_ind;   // Points to parent's d_col_ind (shared)
    uint32_t start_row;          // Offset into parent CSR
    uint32_t n_rows;
    uint32_t n_cols;             // Same as parent
    uint64_t nnz;                // Computed from d_row_ptr[n_rows] - d_row_ptr[0]
};

/// Upload HostMatrix to device as canonical CSR.
/// Builds host-side CSR arrays, then performs 2× cudaMalloc + 2× cudaMemcpyAsync.
DeviceCSR upload_host_matrix_to_device_csr(const HostMatrix& mat, cudaStream_t stream = 0);

/// Compute density permutation and physically reorder CSR rows on-device.
/// After return, d_row_ptr/d_col_ind are in descending-NNZ order.
/// Sets csr.d_density_perm and csr.n_dense_rows.
/// @param dense_threshold  Minimum NNZ for a row to be considered "dense" (0 = n_cols/4)
void device_csr_permute_by_density(DeviceCSR& csr, uint32_t dense_threshold = 0, cudaStream_t stream = 0);

/// Build the transpose CSR on device (for A^T).
/// Allocates new d_row_ptr and d_col_ind; caller owns the returned DeviceCSR.
DeviceCSR device_csr_transpose(const DeviceCSR& csr, cudaStream_t stream = 0);
