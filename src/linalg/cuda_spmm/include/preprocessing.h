// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "m4rm_data.h"
#include <vector>

#define ELLPACK_CHUNK_SIZE 128

struct CompressedMatrix {
    idx_t n_dense_cols;
    std::vector<uint32_t> dense_values; 

    // --- Heavy Rows ---
    row_idx_t n_heavy_rows;              
    std::vector<uint32_t> heavy_bitmaps;  
    std::vector<uint32_t> heavy_csr_offsets;
    std::vector<uint32_t> heavy_csr_indices;
    std::vector<uint32_t> heavy_tiled_coords;
    std::vector<uint32_t> heavy_tiled_ptr; 

    // --- Sparse Format A: Golomb-Rice ---
    bool use_ellpack; 
    std::vector<uint32_t> stream_remainders; 
    std::vector<uint32_t> stream_quotients;
    std::vector<uint32_t> warp_rice_params; 
    std::vector<uint32_t> row_rem_offsets;
    std::vector<uint32_t> row_quo_offsets;

    // --- Sparse Format B: Sliced Ellpack ---
    idx_t n_sparse_cols;
    std::vector<row_idx_t> permutation;
    std::vector<uint32_t> ellpack_stream;
    std::vector<uint32_t> ellpack_metadata;
    std::vector<uint32_t> ellpack_anchors;
    std::vector<uint32_t> ellpack_chunk_ptrs;
    std::vector<uint32_t> warp_meta_offsets;
    std::vector<uint8_t>  warp_num_chunks;

    // --- Sparse Format C: Tiled COO ---
    std::vector<uint32_t> sparse_tiled_coords;
    std::vector<uint32_t> sparse_tiled_ptr;
    
    // Store the block size used to generate this data
    uint32_t tiled_row_block_size = 256; 

    std::vector<uint16_t> delta_16_stream;
    std::vector<uint32_t> delta_16_offsets;
    std::vector<uint8_t> pfor_data;
    std::vector<uint16_t> pfor_meta;
    std::vector<uint32_t> pfor_block_byte_offsets;
    std::vector<uint32_t> pfor_row_block_starts;
    std::vector<uint32_t> pfor_be_data;
    std::vector<uint32_t> pfor_be_meta_bits;      
    std::vector<uint32_t> pfor_be_meta_ex_start;  
    std::vector<uint32_t> pfor_be_meta_data_start;
    std::vector<uint32_t> pfor_be_exceptions;     
    std::vector<uint8_t>  pfor_be_exception_idx;  
    std::vector<uint32_t> pfor_be_row_block_starts;

    size_t total_nnz;
};

class MatrixPreprocessor {
public:
    static CompressedMatrix preprocess(
        const HostMatrixView& raw_matrix, 
        idx_t dense_col_limit,
        row_idx_t heavy_row_limit,
        bool use_ellpack,
        float pfor_threshold = 0.90f,
        uint32_t tiled_row_block_size = 256,
        bool sort_rows = false // NEW: Default to false for transparency
    );

    static std::vector<HostMatrix> slice_matrix_vertical(
        const HostMatrix& matrix,
        idx_t dense_col_limit,
        uint32_t row_block_size,
        uint32_t max_nnz_per_block
    );

    static M4RMContext preprocess_m4rm(
        const HostMatrixView& AT, 
        int num_rows = 8
    );

private:
    static void generate_tiled_coo(
        const HostMatrixView& raw_matrix,
        const std::vector<row_idx_t>& permutation,
        row_idx_t start_row,
        row_idx_t end_row,
        idx_t dense_col_limit,
        std::vector<uint32_t>& out_coords,
        std::vector<uint32_t>& out_ptrs,
        uint32_t block_size 
    );
};
