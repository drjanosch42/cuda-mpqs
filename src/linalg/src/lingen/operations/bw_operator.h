// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "bw_spmm_interface.h" 

namespace lingen {

// Defined here for operator usage; equivalent to SpMM idx_t
using idx_t = uint32_t;

/**
 * @brief Abstract Linear Operator \f$ \mathcal{B}: V \to V \f$.
 * 
 * Defines the recurrence relation \f$ V_{t+1} = \mathcal{B}(V_t) \f$ used in the
 * Block Wiedemann algorithm. The vector space \f$ V \f$ is typically \f$ \mathbb{F}_2^N \f$.
 * 
 * Implementations wrap specific sparse matrix-vector multiplication (SpMM) routines.
 */
struct BwOperator {
    virtual ~BwOperator() = default;

    /**
     * @brief Returns the dimension \f$ N \f$ of the vector space (in bits).
     */
    virtual int dim_bits() const = 0; 

    /**
     * @brief Applies the operator: \f$ V_{out} \leftarrow \mathcal{B} \cdot V_{in} \f$.
     * 
     * @param dOut   Device pointer to output block \f$ V_{out} \f$ (size \f$ N \times n_{block} \f$).
     * @param dIn    Device pointer to input block \f$ V_{in} \f$ (size \f$ N \times n_{block} \f$).
     * @param stream CUDA stream for asynchronous execution.
     */
    virtual void mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const = 0;
};

/**
 * @brief Standard Operator \f$ \mathcal{B} = A \f$.
 * 
 * Implements \f$ v \mapsto A v \f$.
 * Used when finding the right kernel of a square matrix \f$ A \f$.
 */
class BwOperatorA final : public BwOperator {
public:
    BwOperatorA(BlockWiedemannSpMM& spmm, int size);
    ~BwOperatorA() = default;

    int dim_bits() const override { return size_; }
    void mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const override;

private:
    BlockWiedemannSpMM& spmm_;
    int size_;
};

/**
 * @brief Transposed Operator \f$ \mathcal{B} = A^T \f$.
 * 
 * Implements \f$ v \mapsto A^T v \f$.
 * Used when finding the left kernel \f$ x^T A = 0 \iff A^T x = 0 \f$.
 */
class BwOperatorAT final : public BwOperator {
public:
    BwOperatorAT(BlockWiedemannSpMM& spmm, int size);
    ~BwOperatorAT() = default;

    int dim_bits() const override { return size_; }
    void mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const override;

private:
    BlockWiedemannSpMM& spmm_;
    int size_;
};

/**
 * @brief Symmetrized Column-Space Operator \f$ \mathcal{B} = P^T A^T A P \f$.
 * 
 * Used for rectangular matrices where we iterate in the domain (columns).
 * Optionally applies a permutation \f$ P \f$ to precondition the vectors.
 */
class BwOperatorColAtA final : public BwOperator {
public:
    struct Params {
        uint64_t seed = 1;
        bool use_permutation = true;
    };
    
    /**
     * @param spmm   SpMM engine holding \f$ A \f$.
     * @param nrows  Row dimension \f$ M \f$ of \f$ A \f$.
     * @param ncols  Column dimension \f$ N \f$ of \f$ A \f$.
     * @param p      Parameters for permutation.
     */
    BwOperatorColAtA(BlockWiedemannSpMM& spmm, int nrows, int ncols, Params p);
    ~BwOperatorColAtA();

    int dim_bits() const override { return ncols_; }
    void mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const override;

private:
    BlockWiedemannSpMM& spmm_;
    int nrows_;
    int ncols_;
    
    idx_t* d_perm_ = nullptr;     // P
    idx_t* d_invperm_ = nullptr;  // P^T

    // Temporary buffers for the multi-step application A^T * (A * v)
    uint64_t* dTmpCols0_ = nullptr; 
    uint64_t* dTmpRows_  = nullptr; 
    uint64_t* dTmpCols1_ = nullptr; 
};

/**
 * @brief Symmetrized Row-Space Operator \f$ \mathcal{B} = A A^T \f$.
 * 
 * Used for rectangular matrices where we iterate in the codomain (rows).
 */
class BwOperatorRowAAt final : public BwOperator {
public:
    BwOperatorRowAAt(BlockWiedemannSpMM& spmm, int nrows, int ncols);
    ~BwOperatorRowAAt();

    int dim_bits() const override { return nrows_; }
    void mul(uint64_t* dOut, const uint64_t* dIn, cudaStream_t stream) const override;

private:
    BlockWiedemannSpMM& spmm_;
    int nrows_;
    int ncols_;

    uint64_t* dTmpCols_ = nullptr; 
};

} // namespace lingen
