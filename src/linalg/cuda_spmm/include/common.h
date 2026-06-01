// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

using idx_t = uint32_t;
using row_idx_t = uint32_t;

struct HostMatrix {
    row_idx_t n_rows;
    idx_t n_cols;
    std::vector<std::vector<idx_t>> rows; 
};

struct HostMatrixView {
    const HostMatrix* parent;
    row_idx_t start_row;
    row_idx_t n_rows;
    idx_t n_cols;

    HostMatrixView(const HostMatrix& mat) 
        : parent(&mat), start_row(0), n_rows(mat.n_rows), n_cols(mat.n_cols) {}

    HostMatrixView(const HostMatrix& mat, row_idx_t start, row_idx_t end)
        : parent(&mat), start_row(start), n_rows(end - start), n_cols(mat.n_cols) {}

    const std::vector<idx_t>& get_row(row_idx_t r) const {
        return parent->rows[start_row + r];
    }
};

enum class DenseKernelType {
    CUDA_CORES,    
    TENSOR_CORES   
};

struct SpMMConfig {
    // 1. Dense Columns
    bool enable_dense_bitslice = true;
    
    // 2. Heavy Rows (Legacy)
    bool enable_heavy_rows = false;      
    bool enable_heavy_bitmap = false;    
    bool enable_heavy_warp_csr = false;  
    bool enable_heavy_tiled_coo = false; 
    
    // 3. Sparse Rows
    bool enable_sparse = true;
    bool enable_sparse_tiled_coo = false;
    bool enable_sparse_tiled_coo_unrolled = false;
    bool enable_sparse_warp_csr = false;
    
    // 4. Optimization Passes
    bool enable_rcm = false; 

    // 5. Vertical Slicing
    bool enable_vertical_slicing = false;
    uint32_t max_nnz_per_slice_block = 4096;

    // 6. Compression Formats
    bool enable_sparse_delta_16 = false;  
    bool enable_sparse_pfor = false;       
    bool enable_sparse_pfor_bit_exact = false; 

    // 7. M4RM (Method of the Four Russians)
    bool enable_m4rm = false;
    int m4rm_rows = 8; // Default to 8

    int pfor_block_size = 128;             
    float pfor_exception_threshold = 0.90f;

    DenseKernelType dense_kernel_type = DenseKernelType::CUDA_CORES;
    int vector_width_bits = 128;

    uint32_t tiled_row_block_size = 256; 
};

// Helper for memory formatting
inline std::string format_bytes(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int s = 0;
    double count = (double)bytes;
    while (count >= 1024 && s < 4) {
        count /= 1024;
        s++;
    }
    std::stringstream ss;
    ss.precision(2);
    ss << std::fixed << count << " " << suffixes[s];
    return ss.str();
}


