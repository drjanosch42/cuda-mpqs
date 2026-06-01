// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "m4rm_data.h"
#include "preprocessing.h"

// M4RM Launcher (Updated to bits)
void launch_m4rm_full(
    const M4RMContext& ctx, 
    const void* d_V, 
    void* d_C, 
    int total_width_bits, 
    cudaStream_t stream = 0
);

struct DeviceMatrix {
    uint32_t* d_dense_values = nullptr;
    idx_t n_dense_cols;

    row_idx_t n_heavy_rows = 0;
    uint32_t* d_heavy_bitmaps = nullptr;
    uint32_t* d_heavy_csr_offsets = nullptr;
    uint32_t* d_heavy_csr_indices = nullptr;
    uint32_t* d_heavy_tiled_coords = nullptr;
    uint32_t* d_heavy_tiled_ptr = nullptr;
    size_t heavy_tiled_size = 0;

    uint32_t tiled_row_block_size = 256;
    int tiled_col_bits = 22;  // bits allocated for column index in TiledCOO packing

    uint32_t* d_stream_remainders = nullptr;
    uint32_t* d_stream_quotients = nullptr;
    uint32_t* d_warp_rice_params = nullptr; 
    uint32_t* d_row_offsets_rem = nullptr;
    uint32_t* d_row_offsets_quo = nullptr;

    uint32_t* d_sparse_tiled_coords = nullptr;
    uint32_t* d_sparse_tiled_ptr = nullptr;
    size_t sparse_tiled_size = 0;

    // CSR-direct fields (for WarpCSR — no format conversion needed)
    const uint32_t* warp_csr_row_ptr = nullptr;  // owned copy of CSR row pointers
    const uint32_t* warp_csr_col_ind = nullptr;  // owned copy of CSR column indices
    uint32_t  warp_csr_n_rows = 0;
    uint32_t  warp_csr_start_row = 0;  // global row offset for output indexing

    uint16_t* d_delta_16_stream = nullptr;
    uint32_t* d_delta_16_offsets = nullptr;

    uint8_t*  d_pfor_data = nullptr;
    uint16_t* d_pfor_meta = nullptr;
    uint32_t* d_pfor_block_byte_offsets = nullptr;
    uint32_t* d_pfor_row_block_starts = nullptr;

    uint32_t* d_pfor_be_data = nullptr;
    uint32_t* d_pfor_be_meta_bits = nullptr;
    uint32_t* d_pfor_be_meta_ex_start = nullptr;
    uint32_t* d_pfor_be_meta_data_start = nullptr;
    uint32_t* d_pfor_be_exceptions = nullptr;
    uint8_t*  d_pfor_be_exception_idx = nullptr;
    uint32_t* d_pfor_be_row_block_starts = nullptr;

    idx_t n_sparse_cols;
    row_idx_t n_rows;
};

struct SlicedDeviceMatrix { std::vector<DeviceMatrix> slices; };
struct HybridDeviceMatrix { M4RMContext m4rm_ctx; DeviceMatrix sparse_mat; };

class SpMMKernels {
public:
    static DeviceMatrix allocate_and_copy(const CompressedMatrix& host_data);
    static void free_matrix(DeviceMatrix& mat);
    
    // Execution - void* for flexibility and alignment safety
    static void run_spmm(const DeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream = 0);

    static SlicedDeviceMatrix allocate_sliced(const std::vector<CompressedMatrix>& host_slices);
    static void free_sliced(SlicedDeviceMatrix& mat);
    static void run_spmm_sliced(const SlicedDeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream = 0);

    static void run_hybrid(const HybridDeviceMatrix& mat, void* d_C, const void* d_V, SpMMConfig config, cudaStream_t stream = 0);

    static void flush_l2_cache();
    static void cleanup_l2_cache();  
};

// GPU Primitives
// Compute Density Permutation
uint32_t* gpu_compute_density_permutation(const uint32_t* d_row_offsets, size_t n_rows, bool ascending = false);

// Generic Vector Permutation (Gather)
// Supports 32, 64, 128, 256, 512 bit widths
void gpu_permute_vector(const void* d_src, void* d_dst, const uint32_t* d_map, size_t num_elements, int bit_width, cudaStream_t stream = 0);

// Sparse Matrix Transpose
void gpu_transpose_csr(size_t n_rows, size_t n_cols, size_t nnz, const uint32_t* d_csr_row_ptr, const uint32_t* d_csr_col_ind, uint32_t** d_out_row_ptr, uint32_t** d_out_col_ind);
