// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/operations/bw_operator.h"
#include "hpc_logger.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "[BwOperator] CUDA error: " \
            << cudaGetErrorString(e__) \
            << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error( \
            std::string("CUDA error: ") + cudaGetErrorString(e__)); \
    } \
} while(0)
#endif

namespace lingen {

// -----------------------------------------------------------------------------
// Helper Kernels
// -----------------------------------------------------------------------------

/**
 * @brief Applies permutation: out[i] = in[perm[i]].
 */
static __global__ void k_gather_perm_u64(uint64_t* __restrict__ out, const uint64_t* __restrict__ in, const idx_t* __restrict__ perm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[(int)perm[i]];
}

// -----------------------------------------------------------------------------
// BwOperatorA Implementation
// -----------------------------------------------------------------------------

BwOperatorA::BwOperatorA(BlockWiedemannSpMM& spmm, int size) 
    : spmm_(spmm), size_(size) 
{}

void BwOperatorA::mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const {
    spmm_.execute_A((void*)dOut, (const void*)dIn, stream);
}

// -----------------------------------------------------------------------------
// BwOperatorAT Implementation
// -----------------------------------------------------------------------------

BwOperatorAT::BwOperatorAT(BlockWiedemannSpMM& spmm, int size) 
    : spmm_(spmm), size_(size) 
{}

void BwOperatorAT::mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const {
    spmm_.execute_AT((void*)dOut, (const void*)dIn, stream);
}

// -----------------------------------------------------------------------------
// BwOperatorColAtA Implementation
// -----------------------------------------------------------------------------

BwOperatorColAtA::BwOperatorColAtA(BlockWiedemannSpMM& spmm, int nrows, int ncols, Params p)
: spmm_(spmm), nrows_(nrows), ncols_(ncols) {
    // Allocate temps
    CUDA_CHECK(cudaMalloc(&dTmpCols0_, size_t(ncols_) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dTmpCols1_, size_t(ncols_) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dTmpRows_,  size_t(nrows_) * sizeof(uint64_t)));

    if (p.use_permutation) {
        std::vector<idx_t> h_perm(ncols_), h_invperm(ncols_);
        std::iota(h_perm.begin(), h_perm.end(), 0);
        std::mt19937_64 rng(p.seed);
        std::shuffle(h_perm.begin(), h_perm.end(), rng);
        
        // Compute Inverse Permutation P^T
        for (int i = 0; i < ncols_; ++i) h_invperm[h_perm[i]] = i;

        CUDA_CHECK(cudaMalloc(&d_perm_, size_t(ncols_) * sizeof(idx_t)));
        CUDA_CHECK(cudaMalloc(&d_invperm_, size_t(ncols_) * sizeof(idx_t)));
        CUDA_CHECK(cudaMemcpy(d_perm_, h_perm.data(), size_t(ncols_) * sizeof(idx_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invperm_, h_invperm.data(), size_t(ncols_) * sizeof(idx_t), cudaMemcpyHostToDevice));
    }
}

BwOperatorColAtA::~BwOperatorColAtA() {
    cudaFree(d_perm_); cudaFree(d_invperm_);
    cudaFree(dTmpCols0_); cudaFree(dTmpCols1_); cudaFree(dTmpRows_);
}

void BwOperatorColAtA::mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const {
    const int threads = 256;
    const int blocksC = (ncols_ + threads - 1) / threads;
    const uint64_t* in_use = dIn;

    // 1. Apply Permutation P (on caller's stream)
    if (d_perm_) {
        k_gather_perm_u64<<<blocksC, threads, 0, stream>>>(dTmpCols0_, dIn, d_invperm_, ncols_);
        in_use = dTmpCols0_;
    }

    // 2. Apply A (stream-aware, no sync)
    spmm_.execute_A((void*)dTmpRows_, (const void*)in_use, stream);

    // 3. Apply A^T (stream-aware, no sync)
    spmm_.execute_AT((void*)dTmpCols1_, (const void*)dTmpRows_, stream);

    // 4. Apply Inverse Permutation P^T (on caller's stream)
    if (d_perm_) {
        k_gather_perm_u64<<<blocksC, threads, 0, stream>>>(dOut, dTmpCols1_, d_perm_, ncols_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dOut, dTmpCols1_, size_t(ncols_) * sizeof(uint64_t),
                                    cudaMemcpyDeviceToDevice, stream));
    }
}

// -----------------------------------------------------------------------------
// BwOperatorRowAAt Implementation
// -----------------------------------------------------------------------------

BwOperatorRowAAt::BwOperatorRowAAt(BlockWiedemannSpMM& spmm, int nrows, int ncols)
: spmm_(spmm), nrows_(nrows), ncols_(ncols) {
    CUDA_CHECK(cudaMalloc(&dTmpCols_, size_t(ncols_) * sizeof(uint64_t)));
}

BwOperatorRowAAt::~BwOperatorRowAAt() {
    cudaFree(dTmpCols_);
}

void BwOperatorRowAAt::mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const {
    // A^T then A, both on caller's stream — in-order semantics guarantee ordering.
    spmm_.execute_AT((void*)dTmpCols_, (const void*)dIn, stream);
    spmm_.execute_A((void*)dOut, (const void*)dTmpCols_, stream);
}

} // namespace lingen
