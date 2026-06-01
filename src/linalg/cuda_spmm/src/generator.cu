// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "generator.h"
#include "hpc_logger.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <omp.h>
#include <cstring>

// CUDA / Thrust Headers
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// =================================================================================
// Transpose (CPU)
// =================================================================================

HostMatrix MatrixGenerator::transpose(const HostMatrix& A) {
    HostMatrix AT;
    AT.n_rows = A.n_cols; 
    AT.n_cols = (idx_t)A.n_rows; 
    AT.rows.resize(AT.n_rows);

    LOG(LOG_DEBUG_1) << "[Generator] Transposing Matrix..." << std::endl;
    
    // 1. Count degrees for pre-allocation
    std::vector<size_t> counts(AT.n_rows, 0);
    for(const auto& r : A.rows) {
        for(idx_t c : r) {
            if(c < AT.n_rows) counts[c]++;
        }
    }
    
    // 2. Reserve
    #pragma omp parallel for
    for(size_t i=0; i<AT.n_rows; ++i) {
        AT.rows[i].reserve(counts[i]);
    }

    // 3. Fill (Serial to avoid mutex contention, usually fast enough)
    for(row_idx_t r = 0; r < A.n_rows; ++r) {
        for(idx_t c : A.rows[r]) {
            if(c < AT.n_rows) {
                AT.rows[c].push_back((idx_t)r);
            }
        }
    }

    // 4. Sort columns
    #pragma omp parallel for schedule(dynamic, 128)
    for(size_t i=0; i < AT.n_rows; ++i) {
        std::sort(AT.rows[i].begin(), AT.rows[i].end());
    }
    
    return AT;
}

// =================================================================================
// Factor Base Generation (CPU)
// =================================================================================

MatrixGenerator::MatrixGenerator(uint64_t seed) : seed_(seed) {}

void MatrixGenerator::simple_sieve(size_t limit) {
    if (limit < 2) return;
    std::vector<bool> is_prime(limit + 1, true);
    is_prime[0] = is_prime[1] = false;

    for (size_t p = 2; p * p <= limit; p++) {
        if (is_prime[p]) {
            for (size_t i = p * p; i <= limit; i += p)
                is_prime[i] = false;
        }
    }

    primes_.clear();
    for (size_t p = 2; p <= limit; p++) {
        if (is_prime[p]) primes_.push_back(p);
    }
}

void MatrixGenerator::generate_factor_base(idx_t n_cols) {
    size_t limit;
    if (n_cols < 10) limit = 30;
    else limit = (size_t)(n_cols * (std::log(n_cols) + 1.2)); 

    simple_sieve(limit);

    if (primes_.size() > n_cols) {
        primes_.resize(n_cols);
    } else if (primes_.size() < n_cols) {
        simple_sieve(limit * 2);
        if (primes_.size() > n_cols) primes_.resize(n_cols);
    }
    
    LOG(LOG_DEBUG_1) << "[Generator] Generated " << primes_.size() << " factor base primes. "
              << "Max prime: " << primes_.back() << std::endl;
}

// =================================================================================
// Matrix Generation Kernels (GPU)
// =================================================================================

// -----------------------------------------------------------------------------
// Robust Stateless PRNG (SplitMix64)
// -----------------------------------------------------------------------------
// This provides excellent avalanche properties, ensuring that adjacent indices
// produce uncorrelated 64-bit words. Essential for high-rank matrix generation.
__device__ __forceinline__ uint64_t splitmix64_stateless(uint64_t index, uint64_t seed) {
    uint64_t z = index + seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Stateless hash [0, 1)
__device__ __forceinline__ float hash_rng(uint64_t seed, uint32_t row, uint32_t col) {
    uint64_t h = seed + row;
    h ^= (uint64_t)col * 0x5bf03635b9076f63ULL;
    h ^= h >> 29;
    h *= 0x5bf03635b9076f63ULL;
    h ^= h >> 32;
    return (float)(h & 0xFFFFFFFF) * 2.3283064e-10f; 
}

__global__ void k_count_nnz(
    uint64_t seed,
    row_idx_t n_rows, 
    idx_t n_cols, 
    double alpha, 
    const uint64_t* primes, 
    uint32_t* row_counts
) {
    row_idx_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;

    uint32_t count = 0;
    for (idx_t c = 0; c < n_cols; ++c) {
        uint64_t p = primes[c];
        // Probability = alpha / (p + 1)
        float prob = (float)(alpha / (double)(p + 1));
        if (prob > 1.0f) prob = 1.0f;

        //if (hash_rng(seed, r, c) < prob) {
	uint64_t h = splitmix64_stateless((uint64_t)(r*n_cols + c), seed);
	if(((float)(h & 0xFFFFFFFF) * 2.3283064e-10f) < prob) {
            count++;
        }
    }
    row_counts[r] = count;
}

__global__ void k_fill_indices(
    uint64_t seed,
    row_idx_t n_rows, 
    idx_t n_cols, 
    double alpha, 
    const uint64_t* primes, 
    const uint32_t* row_ptrs, 
    uint32_t* col_indices
) {
    row_idx_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;

    uint32_t write_idx = row_ptrs[r];
    for (idx_t c = 0; c < n_cols; ++c) {
        uint64_t p = primes[c];
        float prob = (float)(alpha / (double)(p + 1));
        if (prob > 1.0f) prob = 1.0f;

        //if (hash_rng(seed, r, c) < prob) {
	uint64_t h = splitmix64_stateless((uint64_t)(r*n_cols + c), seed);
	if(((float)(h & 0xFFFFFFFF) * 2.3283064e-10f) < prob) {
            col_indices[write_idx++] = c;
        }
    }
}

// =================================================================================
// Matrix Generation Implementations
// =================================================================================

HostMatrix MatrixGenerator::generate_matrix_gpu(row_idx_t n_rows, double alpha) {
    LOG(LOG_DEBUG_1) << "[Generator-GPU] Generating " << n_rows << "x" << primes_.size() << " on GPU..." << std::endl;
    
    idx_t n_cols = (idx_t)primes_.size();
    
    // 1. Upload Primes
    uint64_t* d_primes;
    cudaMalloc(&d_primes, primes_.size() * sizeof(uint64_t));
    cudaMemcpy(d_primes, primes_.data(), primes_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 2. Count NNZ
    uint32_t* d_row_counts;
    cudaMalloc(&d_row_counts, n_rows * sizeof(uint32_t));

    int block_size = 128;
    int grid_size = (n_rows + block_size - 1) / block_size;
    k_count_nnz<<<grid_size, block_size>>>(seed_, n_rows, n_cols, alpha, d_primes, d_row_counts);

    // 3. Scan for Pointers
    uint32_t* d_row_ptrs;
    cudaMalloc(&d_row_ptrs, (n_rows + 1) * sizeof(uint32_t));
    thrust::device_ptr<uint32_t> t_counts(d_row_counts);
    thrust::device_ptr<uint32_t> t_ptrs(d_row_ptrs);
    thrust::exclusive_scan(t_counts, t_counts + n_rows, t_ptrs);
    
    // Get Total NNZ
    uint32_t last_start, last_count;
    cudaMemcpy(&last_start, d_row_ptrs + n_rows - 1, 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_count, d_row_counts + n_rows - 1, 4, cudaMemcpyDeviceToHost);
    size_t total_nnz = last_start + last_count;

    // 4. Fill Indices
    uint32_t* d_col_indices;
    cudaMalloc(&d_col_indices, total_nnz * sizeof(uint32_t));
    k_fill_indices<<<grid_size, block_size>>>(seed_, n_rows, n_cols, alpha, d_primes, d_row_ptrs, d_col_indices);
    cudaDeviceSynchronize();

    // 5. Download
    LOG(LOG_DEBUG_1) << "[Generator-GPU] Downloading " << total_nnz << " non-zeros..." << std::endl;
    HostMatrix mat;
    mat.n_rows = n_rows;
    mat.n_cols = n_cols;
    mat.rows.resize(n_rows);

    std::vector<uint32_t> h_row_ptrs(n_rows);
    std::vector<uint32_t> h_col_indices(total_nnz);
    
    cudaMemcpy(h_row_ptrs.data(), d_row_ptrs, n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_indices.data(), d_col_indices, total_nnz * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 6. Reconstruct Vector of Vectors
    #pragma omp parallel for
    for (row_idx_t i = 0; i < n_rows; ++i) {
        uint32_t start = h_row_ptrs[i];
        uint32_t end = (i == n_rows - 1) ? (uint32_t)total_nnz : h_row_ptrs[i+1];
        
        mat.rows[i].reserve(end - start);
        for (uint32_t k = start; k < end; ++k) {
            mat.rows[i].push_back(h_col_indices[k]);
        }
    }

    cudaFree(d_primes); cudaFree(d_row_counts);
    cudaFree(d_row_ptrs); cudaFree(d_col_indices);

    return mat;
}

HostMatrix MatrixGenerator::generate_matrix_cpu(row_idx_t n_rows, double alpha) {
    LOG(LOG_DEBUG_1) << "[Generator-CPU] Using CPU generation..." << std::endl;
    HostMatrix matrix;
    matrix.n_rows = n_rows;
    matrix.n_cols = (idx_t)primes_.size();
    matrix.rows.resize(n_rows);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(seed_ + tid);
        #pragma omp for schedule(dynamic, 128)
        for (row_idx_t i = 0; i < n_rows; ++i) {
            matrix.rows[i].reserve(64);
            for (idx_t j = 0; j < matrix.n_cols; ++j) {
                // p = alpha / (prime + 1)
                double p = alpha / (double)(primes_[j] + 1);
                if (p > 1.0) p = 1.0;
                
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                if (dist(rng) < p) {
                    matrix.rows[i].push_back(j);
                }
            }
        }
    }
    return matrix;
}

// Main Wrapper with Stats
HostMatrix MatrixGenerator::generate_matrix(row_idx_t n_rows, double alpha, bool use_gpu) {
    if (primes_.empty()) throw std::runtime_error("Factor base not generated. Call generate_factor_base first.");

    HostMatrix mat;
    if (use_gpu) {
        mat = generate_matrix_gpu(n_rows, alpha);
    } else {
        mat = generate_matrix_cpu(n_rows, alpha);
    }

    // Stats Gathering
    size_t total_nnz = 0;
    for(const auto& r : mat.rows) total_nnz += r.size();
    
    LOG(LOG_DEBUG_1) << "[Generator] Generation Complete." << std::endl;
    LOG(LOG_DEBUG_1) << "[Generator]  Total NNZ: " << total_nnz << std::endl;
    double density = (n_rows > 0 && mat.n_cols > 0) ? (double)total_nnz / ((double)n_rows * mat.n_cols) : 0.0;
    double weight = (n_rows > 0) ? (double)total_nnz / n_rows : 0.0;
    
    LOG(LOG_DEBUG_1) << "[Generator]  Avg Density: " << density << std::endl;
    LOG(LOG_DEBUG_1) << "[Generator]  Avg Weight/Row: " << weight << std::endl;

    return mat;
}

// =================================================================================
// Random Vector Generation
// =================================================================================

__global__ void k_fill_random_vector(uint64_t seed, uint64_t* output, size_t n_words) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t i = idx; i < n_words; i += stride) {
        output[i] = splitmix64_stateless(i, seed);
    }
}

std::vector<uint8_t> MatrixGenerator::generate_random_vector_gpu(size_t n_elements, int bit_width) {
    size_t total_bytes = n_elements * (bit_width / 8);
    // Align up to 8 bytes for kernel processing
    size_t n_words = (total_bytes + 7) / 8;
    size_t buffer_bytes = n_words * 8;

    LOG(LOG_DEBUG_1) << "[Generator-GPU] Generating random " << bit_width << "-bit vector (" << format_bytes(total_bytes) << ")..." << std::endl;

    uint64_t* d_V;
    CUDA_CHECK(cudaMalloc(&d_V, buffer_bytes));

    int block_size = 256;
    int grid_size = std::min(1024, (int)((n_words + block_size - 1) / block_size));

    // Use a different seed sequence than matrix generation
    k_fill_random_vector<<<grid_size, block_size>>>(seed_ + n_elements, d_V, n_words);
    CUDA_CHECK(cudaGetLastError()); 

    std::vector<uint8_t> h_V(total_bytes);
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, total_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_V));

    return h_V;
}

std::vector<uint8_t> MatrixGenerator::generate_random_vector_cpu(size_t n_elements, int bit_width) {
    size_t total_bytes = n_elements * (bit_width / 8);
    std::vector<uint8_t> h_V(total_bytes);
    LOG(LOG_DEBUG_1) << "[Generator-CPU] Generating random " << bit_width << "-bit vector (" << format_bytes(total_bytes) << ")..." << std::endl;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Use a different seed sequence than matrix generation
        std::mt19937_64 rng(seed_ + tid + n_elements); 
        uint64_t* data_ptr = (uint64_t*)h_V.data();
        size_t n_words = total_bytes / 8;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_words; ++i) {
            data_ptr[i] = rng();
        }

        // Handle remainder bytes if the total size is not a multiple of 8
        if (tid == 0 && (total_bytes % 8 != 0)) {
            uint64_t last_word = rng();
            std::memcpy(h_V.data() + n_words * 8, &last_word, total_bytes % 8);
        }
    }
    return h_V;
}

std::vector<uint8_t> MatrixGenerator::generate_random_vector(size_t n_elements, int bit_width, bool use_gpu) {
    if (bit_width % 8 != 0) {
        LOG(LOG_ERROR_CRITICAL) << "[Generator] Bit width must be a multiple of 8." << std::endl;
        throw std::runtime_error("[Generator] Bit width must be a multiple of 8.");
    }
    if (use_gpu) {
        return generate_random_vector_gpu(n_elements, bit_width);
    } else {
        return generate_random_vector_cpu(n_elements, bit_width);
    }
}
