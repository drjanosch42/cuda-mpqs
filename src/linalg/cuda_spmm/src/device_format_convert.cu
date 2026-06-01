// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "device_format_convert.h"
#include "format_arena.h"
#include "common.h"
#include "preprocessing.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <stdexcept>
#include <algorithm>

// =============================================================================
// FormatArena Implementation
// =============================================================================

void* FormatArena::alloc_temporary(size_t bytes, const std::string& tag) {
    if (bytes == 0) return nullptr;
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    allocs_.push_back({ptr, bytes, tag, /*persistent=*/false});
    return ptr;
}

void* FormatArena::alloc_persistent(size_t bytes, const std::string& tag) {
    if (bytes == 0) return nullptr;
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    allocs_.push_back({ptr, bytes, tag, /*persistent=*/true});
    return ptr;
}

void FormatArena::promote_to_persistent(void* ptr) {
    for (auto& a : allocs_) {
        if (a.ptr == ptr) { a.persistent = true; return; }
    }
}

void FormatArena::free_temporaries() {
    auto it = std::remove_if(allocs_.begin(), allocs_.end(), [](Allocation& a) {
        if (!a.persistent) { cudaFree(a.ptr); return true; }
        return false;
    });
    allocs_.erase(it, allocs_.end());
}

void FormatArena::free_since(size_t watermark) {
    for (size_t i = allocs_.size(); i > watermark; --i) {
        if (!allocs_[i - 1].persistent) {
            cudaFree(allocs_[i - 1].ptr);
        }
    }
    allocs_.resize(watermark);
}

void FormatArena::free_all() {
    for (auto& a : allocs_) cudaFree(a.ptr);
    allocs_.clear();
}

size_t FormatArena::total_temporary_bytes() const {
    size_t total = 0;
    for (const auto& a : allocs_)
        if (!a.persistent) total += a.bytes;
    return total;
}

size_t FormatArena::total_persistent_bytes() const {
    size_t total = 0;
    for (const auto& a : allocs_)
        if (a.persistent) total += a.bytes;
    return total;
}

size_t FormatArena::total_bytes() const {
    size_t total = 0;
    for (const auto& a : allocs_) total += a.bytes;
    return total;
}

// =============================================================================
// TiledCOO Kernels
// =============================================================================

/// Phase 1: Count NNZ per tile (one thread per row)
__global__ void k_tiledcoo_count(
    const uint32_t* d_row_ptr,
    uint32_t n_rows,
    uint32_t tile_size,
    uint32_t* d_tile_counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_rows) return;
    uint32_t tile_id = tid / tile_size;
    uint32_t nnz = d_row_ptr[tid + 1] - d_row_ptr[tid];
    atomicAdd(&d_tile_counts[tile_id], nnz);
}

/// Phase 3: Pack coordinates as (local_row << col_bits) | (col & col_mask)
/// col_bits is derived from tile_size: row_bits = ceil_log2(tile_size), col_bits = 32 - row_bits
__global__ void k_tiledcoo_pack(
    const uint32_t* d_row_ptr,
    const uint32_t* d_col_ind,
    uint32_t n_rows,
    uint32_t tile_size,
    uint32_t* d_coords,
    uint32_t* d_write_offsets,
    int col_bits                  // number of bits for column index
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t tile_id = row / tile_size;
    uint32_t local_row = row % tile_size;
    uint32_t col_start = d_row_ptr[row];
    uint32_t col_end   = d_row_ptr[row + 1];
    uint32_t col_mask = (1u << col_bits) - 1;

    for (uint32_t j = col_start; j < col_end; ++j) {
        uint32_t col = d_col_ind[j];
        uint32_t packed = (local_row << col_bits) | (col & col_mask);
        uint32_t pos = atomicAdd(&d_write_offsets[tile_id], 1);
        d_coords[pos] = packed;
    }
}

// =============================================================================
// Delta16 Kernels
// =============================================================================

/// Compute delta stream offsets from row_ptr: d_offsets[i] = d_row_ptr[i] - base.
/// Fast path for n_cols <= 65535 where NNZ == stream length (no escape pairs).
__global__ void k_rowptr_to_offsets(
    const uint32_t* d_row_ptr,
    uint32_t base,
    uint32_t n_entries,    // n_rows + 1
    uint32_t* d_offsets
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_entries) {
        d_offsets[i] = d_row_ptr[i] - base;
    }
}

/// Count escape-expanded delta stream entries per row.
/// Each NNZ produces 1 final delta + 2 entries per 0xFFFF overflow (escape pair).
/// Only needed when n_cols > 65535.
__global__ void k_delta16_escape_sizes(
    const uint32_t* d_row_ptr,
    const uint32_t* d_col_ind,
    uint32_t n_rows,
    uint32_t* d_expanded_sizes   // output: total uint16_t entries per row
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t col_start = d_row_ptr[row];
    uint32_t col_end   = d_row_ptr[row + 1];
    uint32_t count = 0;
    uint32_t prev_col = 0xFFFFFFFF;  // matches CPU: dense_col_limit - 1 = 0 - 1

    for (uint32_t j = col_start; j < col_end; ++j) {
        uint32_t col = d_col_ind[j];
        uint32_t delta = col - prev_col;
        // 1 final entry + 2 entries per escape pair
        count += 1 + 2 * ((delta - 1) / 0xFFFF);
        prev_col = col;
    }
    d_expanded_sizes[row] = count;
}

/// Encode column indices as delta stream with escape sequences.
/// Matches CPU reference (preprocessing.cpp): prev_col = dense_col_limit - 1.
/// With dense_col_limit = 0, prev_col = 0xFFFFFFFF (uint32_t wrap).
/// Gaps > 0xFFFF emit (0xFFFF, 0x0000) escape pairs before the final delta.
__global__ void k_delta16_encode(
    const uint32_t* d_row_ptr,
    const uint32_t* d_col_ind,
    uint32_t n_rows,
    const uint32_t* d_delta_offsets,
    uint16_t* d_delta_stream
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t col_start = d_row_ptr[row];
    uint32_t col_end   = d_row_ptr[row + 1];
    uint32_t out_pos   = d_delta_offsets[row];

    // prev_col = dense_col_limit - 1 = 0 - 1 = 0xFFFFFFFF for dense_col_limit = 0
    uint32_t prev_col = 0xFFFFFFFF;
    for (uint32_t j = col_start; j < col_end; ++j) {
        uint32_t col = d_col_ind[j];
        uint32_t delta = col - prev_col;
        while (delta > 0xFFFF) {
            d_delta_stream[out_pos++] = 0xFFFF;
            d_delta_stream[out_pos++] = 0x0000;
            delta -= 0xFFFF;
        }
        d_delta_stream[out_pos++] = static_cast<uint16_t>(delta);
        prev_col = col;
    }
}

// =============================================================================
// M4RM Kernels
// =============================================================================

/// Device binary search in sorted array
__device__ bool binary_search_device(
    const uint32_t* arr, uint32_t start, uint32_t end, uint32_t target
) {
    while (start < end) {
        uint32_t mid = start + (end - start) / 2;
        uint32_t val = arr[mid];
        if (val == target) return true;
        if (val < target) start = mid + 1;
        else end = mid;
    }
    return false;
}

/// Encode M4RM patterns: one thread per column, binary search each of num_rows rows
__global__ void k_m4rm_encode(
    const uint32_t* d_row_ptr,
    const uint32_t* d_col_ind,
    uint32_t n_rows,    // <= 8
    uint32_t n_cols,
    uint8_t* d_patterns
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;

    uint8_t pattern = 0;
    for (uint32_t r = 0; r < n_rows && r < 8; ++r) {
        uint32_t start = d_row_ptr[r];
        uint32_t end   = d_row_ptr[r + 1];
        if (binary_search_device(d_col_ind, start, end, col))
            pattern |= (1u << r);
    }
    d_patterns[col] = pattern;
}

// =============================================================================
// TiledCOO Converter
// =============================================================================

/// Compute col_bits for TiledCOO packing: row_bits = ceil_log2(tile_size), col_bits = 32 - row_bits
static int tiledcoo_col_bits(uint32_t tile_size) {
    int row_bits = 0;
    uint32_t tmp = tile_size - 1;
    while (tmp > 0) { row_bits++; tmp >>= 1; }
    return 32 - row_bits;
}

void gpu_convert_csr_to_tiledcoo(
    const DeviceCSRSlice& slice,
    uint32_t tile_block_size,
    FormatArena& arena,
    DeviceMatrix& out,
    cudaStream_t stream
) {
    if (slice.n_rows == 0) {
        out.n_rows = 0;
        return;
    }

    int col_bits = tiledcoo_col_bits(tile_block_size);
    uint32_t n_tiles = (slice.n_rows + tile_block_size - 1) / tile_block_size;

    // Phase 1: count NNZ per tile
    uint32_t* d_tile_counts = (uint32_t*)arena.alloc_temporary(
        n_tiles * sizeof(uint32_t), "tiledcoo_counts");
    CUDA_CHECK(cudaMemsetAsync(d_tile_counts, 0, n_tiles * sizeof(uint32_t), stream));
    k_tiledcoo_count<<<(slice.n_rows + 255) / 256, 256, 0, stream>>>(
        slice.d_row_ptr, slice.n_rows, tile_block_size, d_tile_counts);

    // Phase 2: inclusive scan → tile_ptrs[1..n_tiles], tile_ptrs[0] = 0
    uint32_t* d_tile_ptrs = (uint32_t*)arena.alloc_temporary(
        (n_tiles + 1) * sizeof(uint32_t), "tiledcoo_tile_ptrs");
    CUDA_CHECK(cudaMemsetAsync(d_tile_ptrs, 0, sizeof(uint32_t), stream));
    thrust::device_ptr<uint32_t> d_counts_ptr(d_tile_counts);
    thrust::device_ptr<uint32_t> d_ptrs_out(d_tile_ptrs + 1);
    thrust::inclusive_scan(thrust::cuda::par.on(stream),
        d_counts_ptr, d_counts_ptr + n_tiles, d_ptrs_out);

    // Read total NNZ
    uint32_t total_coords;
    CUDA_CHECK(cudaMemcpyAsync(&total_coords, d_tile_ptrs + n_tiles,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (total_coords == 0) {
        out.d_sparse_tiled_coords = nullptr;
        out.d_sparse_tiled_ptr    = d_tile_ptrs;
        out.sparse_tiled_size     = n_tiles;
        out.tiled_row_block_size  = tile_block_size;
        out.tiled_col_bits        = col_bits;
        out.n_rows                = slice.n_rows;
        out.n_sparse_cols         = slice.n_cols;
        return;
    }

    // Phase 3: pack coordinates
    uint32_t* d_coords = (uint32_t*)arena.alloc_temporary(
        total_coords * sizeof(uint32_t), "tiledcoo_coords");
    uint32_t* d_write_offsets = (uint32_t*)arena.alloc_temporary(
        n_tiles * sizeof(uint32_t), "tiledcoo_write_off");
    CUDA_CHECK(cudaMemcpyAsync(d_write_offsets, d_tile_ptrs,
        n_tiles * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    k_tiledcoo_pack<<<(slice.n_rows + 255) / 256, 256, 0, stream>>>(
        slice.d_row_ptr, slice.d_col_ind,
        slice.n_rows, tile_block_size,
        d_coords, d_write_offsets, col_bits);

    // Populate DeviceMatrix
    out.d_sparse_tiled_coords = d_coords;
    out.d_sparse_tiled_ptr    = d_tile_ptrs;
    out.sparse_tiled_size     = n_tiles;
    out.tiled_row_block_size  = tile_block_size;
    out.tiled_col_bits        = col_bits;
    out.n_rows                = slice.n_rows;
    out.n_sparse_cols         = slice.n_cols;
}

// =============================================================================
// Delta16 Converter
// =============================================================================

void gpu_convert_csr_to_delta16(
    const DeviceCSRSlice& slice,
    FormatArena& arena,
    DeviceMatrix& out,
    cudaStream_t stream
) {
    if (slice.n_rows == 0) {
        out.n_rows = 0;
        return;
    }

    uint32_t* d_delta_offsets;
    uint32_t total_deltas;

    if (slice.n_cols <= 65535) {
        // Fast path: no escape sequences possible, 1:1 NNZ-to-stream mapping.
        // Compute offsets by subtracting the base row_ptr value — O(n_rows+1).
        uint32_t row_ptr_base;
        CUDA_CHECK(cudaMemcpyAsync(&row_ptr_base, slice.d_row_ptr,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        uint32_t n_entries = slice.n_rows + 1;
        d_delta_offsets = (uint32_t*)arena.alloc_temporary(
            n_entries * sizeof(uint32_t), "delta16_offsets");
        k_rowptr_to_offsets<<<(n_entries + 255) / 256, 256, 0, stream>>>(
            slice.d_row_ptr, row_ptr_base, n_entries, d_delta_offsets);
        total_deltas = static_cast<uint32_t>(slice.nnz);
    } else {
        // Escape path: compute escape-expanded stream sizes per row
        uint32_t* d_expanded_sizes = (uint32_t*)arena.alloc_temporary(
            slice.n_rows * sizeof(uint32_t), "delta16_sizes");
        k_delta16_escape_sizes<<<(slice.n_rows + 255) / 256, 256, 0, stream>>>(
            slice.d_row_ptr, slice.d_col_ind, slice.n_rows, d_expanded_sizes);

        // Exclusive scan → per-row output offsets
        d_delta_offsets = (uint32_t*)arena.alloc_temporary(
            (slice.n_rows + 1) * sizeof(uint32_t), "delta16_offsets");
        CUDA_CHECK(cudaMemsetAsync(d_delta_offsets, 0, sizeof(uint32_t), stream));
        thrust::device_ptr<uint32_t> d_sizes_ptr(d_expanded_sizes);
        thrust::device_ptr<uint32_t> d_offsets_out(d_delta_offsets + 1);
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
            d_sizes_ptr, d_sizes_ptr + slice.n_rows, d_offsets_out);

        // Read total stream length
        CUDA_CHECK(cudaMemcpyAsync(&total_deltas, d_delta_offsets + slice.n_rows,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (total_deltas == 0) {
        out.d_delta_16_stream  = nullptr;
        out.d_delta_16_offsets = d_delta_offsets;
        out.n_rows             = slice.n_rows;
        out.n_sparse_cols      = slice.n_cols;
        return;
    }

    // Phase 3: Encode deltas with escape pairs
    uint16_t* d_delta_stream = (uint16_t*)arena.alloc_temporary(
        total_deltas * sizeof(uint16_t), "delta16_stream");
    k_delta16_encode<<<(slice.n_rows + 255) / 256, 256, 0, stream>>>(
        slice.d_row_ptr, slice.d_col_ind,
        slice.n_rows, d_delta_offsets, d_delta_stream);

    // Populate DeviceMatrix
    out.d_delta_16_stream  = d_delta_stream;
    out.d_delta_16_offsets = d_delta_offsets;
    out.n_rows             = slice.n_rows;
    out.n_sparse_cols      = slice.n_cols;
}

// =============================================================================
// M4RM Converter
// =============================================================================

void gpu_convert_csr_to_m4rm(
    const DeviceCSRSlice& slice,
    int num_m4rm_rows,
    FormatArena& arena,
    M4RMContext& out,
    cudaStream_t stream
) {
    if (slice.n_cols == 0 || slice.n_rows == 0) {
        out.d_pattern_stream = nullptr;
        out.num_relations = 0;
        out.num_dense_rows = num_m4rm_rows;
        return;
    }

    // Pad to 128 bytes to prevent OOB reads in vectorized M4RM kernels
    size_t padded_size = (slice.n_cols + 127) / 128 * 128;
    uint8_t* d_patterns = (uint8_t*)arena.alloc_temporary(
        padded_size * sizeof(uint8_t), "m4rm_patterns");
    CUDA_CHECK(cudaMemsetAsync(d_patterns, 0, padded_size * sizeof(uint8_t), stream));

    uint32_t rows_to_encode = std::min(static_cast<uint32_t>(num_m4rm_rows), slice.n_rows);
    k_m4rm_encode<<<(slice.n_cols + 255) / 256, 256, 0, stream>>>(
        slice.d_row_ptr, slice.d_col_ind,
        rows_to_encode, slice.n_cols, d_patterns);

    out.d_pattern_stream = d_patterns;
    out.num_relations    = slice.n_cols;
    out.num_dense_rows   = num_m4rm_rows;
}

// =============================================================================
// PForDelta_BE Converter — CPU Fallback
// =============================================================================

void gpu_convert_csr_to_pfor_be(
    const DeviceCSRSlice& slice,
    float exception_threshold,
    FormatArena& arena,
    DeviceMatrix& out,
    bool allow_cpu_fallback,
    const HostMatrix* host_mat,
    cudaStream_t stream
) {
    if (allow_cpu_fallback) {
        if (!host_mat) {
            throw std::runtime_error(
                "PForDelta_BE CPU fallback requested but host_mat is null — "
                "this indicates a pruning logic bug in the autotuner");
        }
        HostMatrixView view(*host_mat, slice.start_row, slice.start_row + slice.n_rows);
        CompressedMatrix cm = MatrixPreprocessor::preprocess(
            view, /*dense_col_limit=*/0, /*heavy_row_limit=*/0,
            /*use_ellpack=*/false, exception_threshold,
            /*tiled_bs=*/256, /*sort_rows=*/false);
        out = SpMMKernels::allocate_and_copy(cm);
        return;
    }

    throw std::runtime_error(
        "GPU PForDelta_BE not yet implemented — set allow_cpu_fallback=true");
}

// =============================================================================
// Golomb-Rice Converter — CPU Fallback
// =============================================================================

void gpu_convert_csr_to_golomb(
    const DeviceCSRSlice& slice,
    FormatArena& arena,
    DeviceMatrix& out,
    bool allow_cpu_fallback,
    const HostMatrix* host_mat,
    cudaStream_t stream
) {
    // NOTE: Golomb CPU fallback uses MatrixPreprocessor which selects format
    // based on row density. For very sparse rows, it may produce PForDelta data
    // instead of Golomb. This is a known limitation — Golomb is disabled by
    // default (Config::enable_golomb = false) due to this mismatch.
    if (allow_cpu_fallback) {
        if (!host_mat) {
            throw std::runtime_error(
                "Golomb CPU fallback requested but host_mat is null — "
                "this indicates a pruning logic bug in the autotuner");
        }
        HostMatrixView view(*host_mat, slice.start_row, slice.start_row + slice.n_rows);
        CompressedMatrix cm = MatrixPreprocessor::preprocess(
            view, /*dense_col_limit=*/0, /*heavy_row_limit=*/0,
            /*use_ellpack=*/false, /*pfor_threshold=*/0.90f,
            /*tiled_bs=*/256, /*sort_rows=*/false);
        out = SpMMKernels::allocate_and_copy(cm);
        return;
    }

    throw std::runtime_error(
        "GPU Golomb-Rice not yet implemented — set allow_cpu_fallback=true");
}
