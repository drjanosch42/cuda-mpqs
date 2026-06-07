// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <memory>

#include "mpqs_common.h"     // Defines HostMatrix
#include "mpqs_soa.h"        // Defines SOA Relation structs
#include "hpc_logger.h"      // Logging infrastructure

namespace mpqs {
namespace matrix {

/**
 * @brief Host-side Compressed Sparse Row (CSR) Matrix for GF(2).
 * Optimized for Block Wiedemann inputs.
 */
struct HostMatrixCSR {
    /// @brief Row pointers. Size = num_rows + 1.
    /// row_offsets[i] is the starting index in col_indices for row i.
    /// row_offsets[i+1] - row_offsets[i] = Hamming weight of row i.
    std::vector<uint32_t> row_offsets; 

    /// @brief Flattened column indices. Size = total_nnz.
    /// Contains the indices of the factor base primes present in the relations.
    /// Mapped as: 0 -> Sign, 1 -> Prime 2, k+2 -> FactorBase[k]
    std::vector<uint32_t> col_indices; 

    uint32_t n_rows = 0; 
    uint32_t n_cols = 0; 
};

// --- Validation & Conversion Helpers ---
bool ValidateHostMatrixCSR(const HostMatrixCSR& M);
HostMatrix ConvertFromCSR(const HostMatrixCSR& src);

// --- Diagnostic serializer (sqrt-failure investigation) ---------------------
//
// DumpHostMatrixCSR writes the GF(2) matrix `A` (the canonical BW input, stored
// as a vector<vector<col>> HostMatrix) to `path` in a tiny self-describing
// little-endian binary CSR format so the matrix can be parsed offline.
//
// On-disk layout (all integers little-endian):
//   char     magic[8]   = "MPQSMAT\0"
//   uint32_t version    = 1
//   uint32_t n_rows
//   uint32_t n_cols
//   uint64_t nnz                              (total stored entries)
//   uint64_t row_offsets[n_rows + 1]          (CSR row pointers, offsets[0]=0)
//   uint32_t col_indices[nnz]                 (per-row column indices, as stored)
//
// All matrix entries are 1 over GF(2); only column indices of set bits are
// stored. Column semantics are emitted separately by DumpMatrixColumnLegend.
// Returns true on success. This function performs NO computation on the matrix
// beyond reading it; it is only ever called behind the --dump_matrix flag.
bool DumpHostMatrixCSR(const std::string& path, const HostMatrix& A);

// DumpMatrixColumnLegend writes a human-readable text file mapping each matrix
// column index to its meaning, given the factor base used to build the matrix:
//   col 0            -> sign (parity of negative Q values)
//   col 1            -> prime 2 parity (val_2_exps)
//   col 2 + k        -> FactorBase[k]   (k = 0 .. fb_size-1)
//   cols after FB    -> "LP / character columns (expanded path)" — only present
//                       when n_cols > fb_size + 2 (preprocess matrix path).
// `n_cols` is the matrix column count (HostMatrix::n_cols). Returns true on success.
bool DumpMatrixColumnLegend(const std::string& path,
                            const std::vector<uint32_t>& factor_base,
                            uint32_t n_cols);

/**
 * @brief Handles the construction of the sparse matrix from GPU relation data.
 * Refactored to consume SoA structures directly.
 */
class MatrixConstructor {
public:
    MatrixConstructor();
    ~MatrixConstructor();

    /**
     * @brief Constructs a Host-side CSR Matrix directly from GPU SoA data.
     * Uses a chunked approach to minimize GPU memory usage during generation.
     *
     * Sets out_csr.n_rows and populates row_offsets/col_indices.
     * Caller must set out_csr.n_cols = fb_size + 2 (sign + parity columns)
     * before calling ValidateHostMatrixCSR on the result.
     *
     * @param batch_view View of the SoA data on GPU (large primes, factors, etc.).
     * @param num_relations Number of relations to process.
     * @param out_csr Reference to the output host matrix structure.
     */
    void constructFromSoA(
        mpqs::structures::RelationBatchView batch_view,
        size_t num_relations,
        HostMatrixCSR& out_csr
    );

private:
    cudaStream_t stream_compute_;

    // --- Temporary Device Buffers ---
    // These are resized dynamically but reused to avoid allocation overhead.
    
    // 1. Metadata (Size ~ Num_Relations)
    uint32_t* d_nnz_per_row = nullptr;   
    uint32_t* d_row_offsets = nullptr;   
    size_t   capacity_meta_ = 0;

    // 2. Chunk Data (Size ~ Chunk_Size * Avg_Weight)
    // Used to buffer a slice of the matrix column indices on GPU before transfer.
    uint32_t* d_col_chunk = nullptr;
    size_t   capacity_chunk_nnz_ = 0;
    
    // Unified-memory platform detection (Jetson Orin or similar)
    bool use_managed_ = false;
    bool use_mem_advise_ = false;  ///< true when concurrentManagedAccess supported (discrete GPU)
    int  device_id_ = 0;

    // Constants
    static const size_t MATRIX_GENERATION_CHUNK_SIZE = 4194304; // Process 4M rows at a time
};

} // namespace matrix
} // namespace mpqs
