// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include <random>

class MatrixGenerator {
public:
    /**
     * @brief Constructor
     * @param seed Random seed for reproducibility
     */
    MatrixGenerator(uint64_t seed);

    /**
     * @brief Generates the factor base (primes)
     * @param n_cols Number of primes to generate
     */
    void generate_factor_base(idx_t n_cols);

    /**
     * @brief Generates the sparse binary matrix A
     * 
     * Logic: P(A_ij = 1) = 1 / (p_j + 1).
     * This creates a dense left side (small p) and sparse right side.
     * 
     * @param n_rows Number of relations to simulate
     * @param alpha Density multiplier (simulates polynomial selection optimization)
     * @param use_gpu If true, uses CUDA kernels for generation (fast). If false, uses OpenMP CPU (slow).
     * @return HostMatrix Structure containing the generated data
     */
    HostMatrix generate_matrix(row_idx_t n_rows, double alpha = 1.0, bool use_gpu = true);

    /**
     * @brief Creates a physical transpose of the matrix.
     * New Rows = Old Columns.
     * New Columns = Old Rows.
     */
    static HostMatrix transpose(const HostMatrix& A);

    /**
     * @brief Generates a random block vector for SpMM input.
     * 
     * @param n_elements The number of vector elements (e.g., n_cols of the matrix).
     * @param bit_width The width of each vector element in bits (e.g., 32, 64, ..., 512).
     * @param use_gpu If true, uses CUDA kernels for generation. If false, uses OpenMP CPU.
     * @return std::vector<uint8_t> A flat byte vector containing the random data.
     */
    std::vector<uint8_t> generate_random_vector(size_t n_elements, int bit_width, bool use_gpu = true);

    // Getters for inspection
    const std::vector<uint64_t>& get_primes() const { return primes_; }

private:
    uint64_t seed_;
    std::vector<uint64_t> primes_;
    
    // Internal helper for sieving
    void simple_sieve(size_t limit);

    // Internal implementations
    HostMatrix generate_matrix_gpu(row_idx_t n_rows, double alpha);
    HostMatrix generate_matrix_cpu(row_idx_t n_rows, double alpha);

    // Internal implementations for vector generation
    std::vector<uint8_t> generate_random_vector_gpu(size_t n_elements, int bit_width);
    std::vector<uint8_t> generate_random_vector_cpu(size_t n_elements, int bit_width);
};
