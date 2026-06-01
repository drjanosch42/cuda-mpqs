// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// Import HostMatrix definition from the SpMM library
#include "common.h"

namespace lingen {

// Note: HostMatrix is defined in common.h as:
// struct HostMatrix {
//     row_idx_t n_rows;
//     idx_t n_cols;
//     std::vector<std::vector<idx_t>> rows; 
// };
// We use it directly.

/**
 * @brief A lightweight view into a polynomial of dense matrices.
 * 
 * Represents P(x) = sum(C_i * x^i).
 * Each coefficient C_i is an N x N binary matrix.
 * 
 * Memory Layout: Contiguous. [Mat_0][Mat_1]...[Mat_deg]
 * Matrix Size (bytes): N * (N / 8)
 * Matrix Stride (uint64s): N * (N / 64)
 */
template<int N>
struct PolyMatrixView {
    uint64_t* data;  // Pointer to the start of the buffer (Device memory)
    size_t length;   // Number of coefficients (Degree + 1)

    // Helper: Size of one N x N matrix in uint64_t words
    static constexpr size_t MAT_STRIDE_WORDS = (N * N) / 64;

    // Helper: Size of one N x N matrix in bytes
    static constexpr size_t MAT_SIZE_BYTES = (N * N) / 8;

    __host__ __device__ PolyMatrixView() : data(nullptr), length(0) {}
    __host__ __device__ PolyMatrixView(uint64_t* ptr, size_t len) : data(ptr), length(len) {}

    // Get pointer to the i-th coefficient matrix
    __host__ __device__ uint64_t* coeff(size_t i) const {
        return data + i * MAT_STRIDE_WORDS;
    }

    // Create a sub-view (slicing)
    __host__ __device__ PolyMatrixView split(size_t start, size_t len) const {
        // Bounds check could be added here, but omitted for device performance
        return PolyMatrixView(coeff(start), len);
    }
};

/**
 * @brief A mutable view of a row-major rectangular binary matrix.
 * 
 * Mathematical Object: A matrix \f$ M \in \mathbb{F}_2^{rows \times cols} \f$.
 * Data Layout:
 *  - Rows are contiguous in memory, aligned to 64-bit words.
 *  - Stride (in words) = ceil(cols / 64).
 *  - Bit (r, c) is at word `r * stride + c/64`, bit `c%64`.
 * 
 * Invariants:
 *  - data points to valid memory of size `rows * stride_words * 8` bytes.
 *  - operations are modulo 2 (XOR).
 */
struct PackedBitMatView {
    uint64_t* data;       ///< Pointer to the first word of the matrix
    int rows;             ///< Number of rows
    int cols;             ///< Number of columns
    size_t stride_words;  ///< Words per row = (cols + 63) / 64

    __host__ __device__ PackedBitMatView() 
        : data(nullptr), rows(0), cols(0), stride_words(0) {}

    __host__ __device__ PackedBitMatView(uint64_t* ptr, int r, int c)
        : data(ptr), rows(r), cols(c) {
        stride_words = (static_cast<size_t>(c) + 63) / 64;
    }

    /**
     * @brief Computes the address of the start of row r.
     * @param r Row index (0 <= r < rows)
     * @return Pointer to the first word of row r.
     */
    __host__ __device__ uint64_t* row_ptr(int r) const {
        return data + r * stride_words;
    }

    /**
     * @brief Reads the bit at (r, c) over GF(2).
     * @param r Row index.
     * @param c Column index.
     * @return The value of the bit (0 or 1).
     */
    __host__ __device__ bool get_bit(int r, int c) const {
        return (row_ptr(r)[c / 64] >> (c % 64)) & 1ULL;
    }

    /**
     * @brief Performs elementary row operation: Row_dst <- Row_dst + Row_src (GF(2) addition).
     * 
     * Computes component-wise XOR of the source row into the destination row.
     * 
     * @param r_dst Destination row index in *this.
     * @param src Source view.
     * @param r_src Source row index in src.
     * 
     * Precondition: this->cols == src.cols (strides must match roughly, or at least width).
     * In this implementation, we iterate up to `this->stride_words` and assume `src` has sufficient data.
     */
    __host__ __device__ void xor_row(int r_dst, const PackedBitMatView& src, int r_src) const {
        uint64_t* d = row_ptr(r_dst);
        const uint64_t* s = src.row_ptr(r_src);
        for (size_t i = 0; i < stride_words; ++i) {
            d[i] ^= s[i];
        }
    }
};

/**
 * @brief A mutable view into a polynomial with rectangular binary matrix coefficients.
 * 
 * Mathematical Object: P(x) = \sum_{k=0}^{deg} C_k x^k, where C_k \in \mathbb{F}_2^{rows \times cols}.
 * Memory Layout:
 *  - Coefficient matrices are stored contiguously: C_0, C_1, ..., C_{len-1}.
 *  - Total words = length * (rows * stride_words).
 */
struct PackedBitPolyView {
    uint64_t* data;           ///< Pointer to start of polynomial
    size_t length;            ///< Number of coefficients (Degree + 1)
    int rows;                 ///< Rows in each coefficient matrix
    int cols;                 ///< Cols in each coefficient matrix
    size_t mat_stride_words;  ///< Offset in words between C_k and C_{k+1}

    __host__ __device__ PackedBitPolyView() 
        : data(nullptr), length(0), rows(0), cols(0), mat_stride_words(0) {}

    __host__ __device__ PackedBitPolyView(uint64_t* ptr, size_t len, int r, int c)
        : data(ptr), length(len), rows(r), cols(c) {
        size_t row_stride = (static_cast<size_t>(c) + 63) / 64;
        mat_stride_words = row_stride * r;
    }

    /**
     * @brief Returns a view of the k-th coefficient matrix C_k.
     * @param k Coefficient index (0 <= k < length).
     * @return PackedBitMatView wrapping the k-th coefficient.
     */
    __host__ __device__ PackedBitMatView coeff(size_t k) const {
        return PackedBitMatView(data + k * mat_stride_words, rows, cols);
    }
};  

} // namespace lingen
