// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "device_csr.h"
#include "kernels.h"
#include "m4rm_data.h"
#include "format_arena.h"
#include <cuda_runtime.h>

/// Convert CSR slice to TiledCOO format.
/// Populates out.d_sparse_tiled_coords, d_sparse_tiled_ptr, sparse_tiled_size,
/// tiled_row_block_size, n_rows, n_sparse_cols.
void gpu_convert_csr_to_tiledcoo(
    const DeviceCSRSlice& slice,
    uint32_t tile_block_size,         // typically 256
    FormatArena& arena,
    DeviceMatrix& out,
    cudaStream_t stream = 0
);

/// Convert CSR slice to Delta16 format.
/// Populates out.d_delta_16_stream, d_delta_16_offsets, n_rows, n_sparse_cols.
void gpu_convert_csr_to_delta16(
    const DeviceCSRSlice& slice,
    FormatArena& arena,
    DeviceMatrix& out,
    cudaStream_t stream = 0
);

/// Convert CSR slice to M4RM format.
/// Populates out.d_pattern_stream, num_relations, num_dense_rows.
void gpu_convert_csr_to_m4rm(
    const DeviceCSRSlice& slice,
    int num_m4rm_rows,                // typically 8
    FormatArena& arena,
    M4RMContext& out,
    cudaStream_t stream = 0
);

/// Convert CSR slice to PForDelta_BitExact format (GPU or CPU fallback).
/// When allow_cpu_fallback && host_mat != nullptr, falls back to
/// MatrixPreprocessor::preprocess() + SpMMKernels::allocate_and_copy().
/// CPU-fallback allocations are NOT tracked by FormatArena.
void gpu_convert_csr_to_pfor_be(
    const DeviceCSRSlice& slice,
    float exception_threshold,        // typically 0.90
    FormatArena& arena,
    DeviceMatrix& out,
    bool allow_cpu_fallback,
    const HostMatrix* host_mat,       // needed for CPU fallback (nullable)
    cudaStream_t stream = 0
);

/// Convert CSR slice to Golomb-Rice format (GPU or CPU fallback).
/// Same fallback pattern as PForDelta_BE.
void gpu_convert_csr_to_golomb(
    const DeviceCSRSlice& slice,
    FormatArena& arena,
    DeviceMatrix& out,
    bool allow_cpu_fallback,
    const HostMatrix* host_mat,
    cudaStream_t stream = 0
);
