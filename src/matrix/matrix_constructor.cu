// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
#include "matrix_constructor.h"
#include "matrix_utils.h"
#include "cuda_check.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace mpqs {
namespace matrix {

namespace kernels {

// -----------------------------------------------------------------------------
// Kernels (SoA)
// -----------------------------------------------------------------------------

/**
 * @brief Calculates the Hamming weight (NNZ) for each relation row.
 */
__global__ void calc_row_weights_soa_kernel(
    const uint64_t* __restrict__ offsets,
    const uint8_t* __restrict__ signs,
    const int32_t* __restrict__ val_2_exps,
    const uint8_t* __restrict__ factor_counts, 
    uint64_t num_relations,
    uint32_t* __restrict__ d_nnz_per_row
) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_relations) return;

    // SoA Factor Range
    uint64_t start = offsets[idx];
    uint64_t end   = offsets[idx+1];
    
    uint32_t weight = 0;
    
    // 1. Sign (Column 0)
    if (signs[idx] != 1) weight++;

    // 2. Factor 2 (Column 1)
    if (val_2_exps[idx] & 1) weight++;

    // 3. (Factors of A + Sieve Factors)
    for (uint64_t k = start; k < end; ++k) {
        if (factor_counts[k] & 1) {
            weight++;
        }
    }
    d_nnz_per_row[idx] = weight;
}

/**
 * @brief Fills a chunk of the sparse matrix column indices.
 * * @param chunk_start_row The global index of the first row in this chunk.
 * @param chunk_num_rows  Number of rows in this chunk.
 * @param d_row_offsets   Global offsets array (on device).
 * @param d_cols_out      Output buffer for this chunk (starts at index 0).
 */
__global__ void fill_matrix_soa_chunk_kernel(
    const uint64_t* __restrict__ factor_offsets,
    const uint32_t* __restrict__ factor_indices,
    const uint8_t* __restrict__ factor_counts,
    const uint8_t* __restrict__ signs,
    const int32_t* __restrict__ val_2_exps,
    uint64_t chunk_start_row,
    uint64_t chunk_num_rows,
    const uint32_t* __restrict__ d_row_offsets, 
    uint32_t* __restrict__ d_cols_out
) {
    uint64_t local_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (local_idx >= chunk_num_rows) return;

    uint64_t global_idx = chunk_start_row + local_idx;

    // Determine where to write in the localized chunk buffer
    // Global Pos = d_row_offsets[global_idx]
    // Chunk Start Offset = d_row_offsets[chunk_start_row]
    // Local Pos = Global Pos - Chunk Start Offset
    uint32_t write_pos = d_row_offsets[global_idx] - d_row_offsets[chunk_start_row];
    
    // 1. Sign (Column 0)
    if (signs[global_idx] != 1) {
        d_cols_out[write_pos++] = 0;
    }

    // 2. Factor 2 (Column 1)
    if (val_2_exps[global_idx] & 1) {
        d_cols_out[write_pos++] = 1;
    }

    // 3. Odd Factors (Columns 2 + Index)
    uint64_t start = factor_offsets[global_idx];
    uint64_t end   = factor_offsets[global_idx+1];

    for (uint64_t k = start; k < end; ++k) {
        if (factor_counts[k] & 1) {
            // Mapping: Factor Base Index i -> Matrix Column i + 2
            d_cols_out[write_pos++] = factor_indices[k] + 2;
        }
    }
}

} // namespace kernels

// -----------------------------------------------------------------------------
// MatrixConstructor Implementation
// -----------------------------------------------------------------------------

MatrixConstructor::MatrixConstructor() {
    CUDA_CHECK(cudaStreamCreate(&stream_compute_));

    // Detect unified-memory platform (Jetson Orin or similar).
    // prop.unifiedAddressing is true on ALL modern GPUs (unified address space),
    // so we use a compound check: SM 8.7 (Orin) or small VRAM (<12 GB).
    CUDA_CHECK(cudaGetDevice(&device_id_));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    use_managed_ = (prop.major == 8 && prop.minor == 7) ||
                   (prop.unifiedAddressing &&
                    prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);
    // cudaMemAdvise hints require concurrentManagedAccess (discrete GPUs only).
    // Integrated GPUs (Jetson) share physical memory — hints are unnecessary and unsupported.
    use_mem_advise_ = prop.concurrentManagedAccess != 0;
    if (use_managed_) {
        LOG(LOG_INFO) << "Unified-memory platform detected — using cudaMallocManaged for working buffers.";
    }
}

MatrixConstructor::~MatrixConstructor() {
    if (d_nnz_per_row) cudaFree(d_nnz_per_row);
    if (d_row_offsets) cudaFree(d_row_offsets);
    if (d_col_chunk)   cudaFree(d_col_chunk);
    cudaStreamDestroy(stream_compute_);
}

void MatrixConstructor::constructFromSoA(
    mpqs::structures::RelationBatchView batch_view,
    size_t num_relations,
    HostMatrixCSR& out_csr
) {
    LOG_SET_MODULE("Matrix");
    if (num_relations == 0) return;

    CUDA_CHECK(cudaGetLastError());

    LOG(LOG_DEBUG_1) << "Starting SoA Matrix Construction for " << num_relations << " relations.";

    // -------------------------------------------------------------------------
    // 1. Allocate Metadata Buffers
    // -------------------------------------------------------------------------
    // We need buffers for the weights and the offsets (inclusive scan result)
    // Size = num_relations + 1 (for offsets)
    if (capacity_meta_ < num_relations + 1) {
        LOG(LOG_DEBUG_2) << "capacity_meta < num_relations + 1";
        if (d_nnz_per_row) cudaFree(d_nnz_per_row);
        if (d_row_offsets) cudaFree(d_row_offsets);
        
        capacity_meta_ = std::max(static_cast<size_t>(num_relations * 1.2), num_relations + 1); // Growth factor, min num_relations+1
        size_t meta_bytes = capacity_meta_ * sizeof(uint32_t);
        LOG(LOG_DEBUG_2) << "Allocating metadata buffers of total size " << (2*meta_bytes)/(1024*1024) << " MB.";
        if (use_managed_) {
            // Managed allocation: host can read directly after kernel writes + sync.
            // SetPreferredLocation keeps data GPU-resident for kernel performance.
            CUDA_CHECK(cudaMallocManaged(&d_nnz_per_row, meta_bytes));
            CUDA_CHECK(cudaMallocManaged(&d_row_offsets, meta_bytes));
            if (use_mem_advise_) {
#if CUDART_VERSION >= 13000
                cudaMemLocation loc = {cudaMemLocationTypeDevice, device_id_};
                CUDA_CHECK(cudaMemAdvise(d_nnz_per_row, meta_bytes,
                                         cudaMemAdviseSetPreferredLocation, loc));
                CUDA_CHECK(cudaMemAdvise(d_row_offsets, meta_bytes,
                                         cudaMemAdviseSetPreferredLocation, loc));
#else
                CUDA_CHECK(cudaMemAdvise(d_nnz_per_row, meta_bytes,
                                         cudaMemAdviseSetPreferredLocation, device_id_));
                CUDA_CHECK(cudaMemAdvise(d_row_offsets, meta_bytes,
                                         cudaMemAdviseSetPreferredLocation, device_id_));
#endif
            }
        } else {
            CUDA_CHECK(cudaMalloc(&d_nnz_per_row, meta_bytes));
            CUDA_CHECK(cudaMalloc(&d_row_offsets, meta_bytes));
        }
    }

    LOG(LOG_DEBUG_2) << "Metadata buffers of total size " << (2*capacity_meta_*sizeof(uint32_t))/(1024*1024) << " MB allocated.";
    
    // -------------------------------------------------------------------------
    // 2. Calculate Weights & Scan (Offsets)
    // -------------------------------------------------------------------------
    int blockSize = 256;
    int numBlocks = (num_relations + blockSize - 1) / blockSize;

    // A. Compute Weights
    kernels::calc_row_weights_soa_kernel<<<numBlocks, blockSize, 0, stream_compute_>>>(
        batch_view.factor_offsets,
        batch_view.signs,
        batch_view.val_2_exps,
        batch_view.factor_counts,
        num_relations,
        d_nnz_per_row
    );
    CUDA_CHECK(cudaGetLastError());

    // B. Exclusive Scan to get Row Offsets
    // d_row_offsets[i] will contain sum(weights[0]...weights[i-1])
    // Zero-init sentinel element (kernel only writes [0..num_relations-1])
    CUDA_CHECK(cudaMemsetAsync(d_nnz_per_row + num_relations, 0, sizeof(uint32_t), stream_compute_));
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream_compute_),
        thrust::device_pointer_cast(d_nnz_per_row),
        thrust::device_pointer_cast(d_nnz_per_row + num_relations + 1), // +1 to get total NNZ at end
        thrust::device_pointer_cast(d_row_offsets)
    );

    // -------------------------------------------------------------------------
    // 3. Download Structure & Resize Host
    // -------------------------------------------------------------------------
    out_csr.n_rows = num_relations;
    out_csr.row_offsets.resize(num_relations + 1);

    // Copy offsets to host. On managed-memory platforms, read directly after sync.
    if (use_managed_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_compute_));
        std::memcpy(out_csr.row_offsets.data(), d_row_offsets,
                    (num_relations + 1) * sizeof(uint32_t));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            out_csr.row_offsets.data(),
            d_row_offsets,
            (num_relations + 1) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream_compute_
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute_));
    }

    uint64_t total_nnz = out_csr.row_offsets[num_relations];
    out_csr.col_indices.resize(total_nnz);
    
    // Compute matrix statistics
    uint32_t n_cols = out_csr.n_cols > 0 ? out_csr.n_cols : 1;
    double density = (double)total_nnz / ((double)num_relations * n_cols) * 100.0;
    double avg_per_row = (double)total_nnz / num_relations;
    LOG(LOG_INFO) << "Matrix Layout: " << fmtNum(num_relations) << " rows x " << fmtNum(n_cols)
                  << " cols, " << fmtNum(total_nnz) << " NNZ ("
                  << std::fixed << std::setprecision(3) << density << "% density, avg "
                  << std::setprecision(1) << avg_per_row << " per row)";

    // -------------------------------------------------------------------------
    // 4. Chunked Column Generation
    // -------------------------------------------------------------------------
    // We generate columns in chunks to keep GPU memory usage low.
    
    size_t chunk_size = MATRIX_GENERATION_CHUNK_SIZE;
    
    for (size_t offset = 0; offset < num_relations; offset += chunk_size) {
        size_t current_chunk_rows = std::min(chunk_size, num_relations - offset);
        
        // Host-side offsets for this chunk
        uint32_t start_nnz = out_csr.row_offsets[offset];
        uint32_t end_nnz   = out_csr.row_offsets[offset + current_chunk_rows];
        uint32_t chunk_nnz = end_nnz - start_nnz;

        if (chunk_nnz == 0) continue;

        // Reallocate chunk buffer if needed
        if (capacity_chunk_nnz_ < chunk_nnz) {
            if (d_col_chunk) cudaFree(d_col_chunk);
            capacity_chunk_nnz_ = chunk_nnz * 1.5; // Growth factor
            size_t chunk_bytes = capacity_chunk_nnz_ * sizeof(uint32_t);
            if (use_managed_) {
                CUDA_CHECK(cudaMallocManaged(&d_col_chunk, chunk_bytes));
                if (use_mem_advise_) {
#if CUDART_VERSION >= 13000
                    cudaMemLocation loc = {cudaMemLocationTypeDevice, device_id_};
                    CUDA_CHECK(cudaMemAdvise(d_col_chunk, chunk_bytes,
                                             cudaMemAdviseSetPreferredLocation, loc));
#else
                    CUDA_CHECK(cudaMemAdvise(d_col_chunk, chunk_bytes,
                                             cudaMemAdviseSetPreferredLocation, device_id_));
#endif
                }
            } else {
                CUDA_CHECK(cudaMalloc(&d_col_chunk, chunk_bytes));
            }
        }

        // Launch Fill Kernel for this chunk
        int chunk_blocks = (current_chunk_rows + blockSize - 1) / blockSize;
        
	kernels::fill_matrix_soa_chunk_kernel<<<chunk_blocks, blockSize, 0, stream_compute_>>>(
            batch_view.factor_offsets,
            batch_view.factor_indices,
            batch_view.factor_counts,
            batch_view.signs,
            batch_view.val_2_exps,
            offset,             // Start Row
            current_chunk_rows, // Num Rows
            d_row_offsets,      // Global Offsets (Device)
            d_col_chunk         // Local Output
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy chunk results to host vector
        if (use_managed_) {
            CUDA_CHECK(cudaStreamSynchronize(stream_compute_));
            std::memcpy(out_csr.col_indices.data() + start_nnz,
                        d_col_chunk, chunk_nnz * sizeof(uint32_t));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(
                out_csr.col_indices.data() + start_nnz,
                d_col_chunk,
                chunk_nnz * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream_compute_
            ));
        }
    }

    // Final Sync
    CUDA_CHECK(cudaStreamSynchronize(stream_compute_));
    LOG(LOG_DEBUG_1) << "SoA Construction Complete.";
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------


bool ValidateHostMatrixCSR(const HostMatrixCSR& M) {
    // 1. Structural Integrity Checks
    if (M.row_offsets.size() != M.n_rows + 1) {
        LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: row_offsets size mismatch.";
        return false;
    }
    if (M.row_offsets[0] != 0) {
        LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: row_offsets[0] must be 0.";
        return false;
    }
    if (M.row_offsets.back() != M.col_indices.size()) {
        LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: Last row offset does not match total NNZ.";
        return false;
    }

    // 2. Deep Data Integrity Checks
    for (uint32_t i = 0; i < M.n_rows; ++i) {
        uint32_t start = M.row_offsets[i];
        uint32_t end   = M.row_offsets[i+1];

        // Check A: Monotonicity (offsets must be non-decreasing)
        if (start > end) {
            LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: Negative row length at row " << i << ".";
            return false;
        }

        // Check B: Column Index Validity
        for (uint32_t k = start; k < end; ++k) {
            // Bound check
            if (M.col_indices[k] >= M.n_cols) {
                LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: Column index " << M.col_indices[k] 
                          << " out of bounds (n_cols=" << M.n_cols << ") at row " << i << ".";
                return false;
            }

            // Check C: Sorted and Unique (Strict CSR)
            // In MPQS, relations are sets of prime factors; duplicates imply cancellation.
            // Indices must be strictly increasing: col[k] > col[k-1]
            if (k > start) {
                if (M.col_indices[k] <= M.col_indices[k-1]) {
                    LOG(LOG_ERROR_CRITICAL) << "CSR Validation Failed: Unsorted or duplicate indices at row " 
                              << i << " (indices " << M.col_indices[k-1] << ", " 
                              << M.col_indices[k] << ").";
                    return false;
                }
            }
        }
    }

    return true;
}

HostMatrix ConvertFromCSR(const HostMatrixCSR& src) {
    HostMatrix out;
    out.n_rows = src.n_rows;
    out.n_cols = src.n_cols;

    // Resize output to match the number of rows in the CSR.
    out.rows.resize(src.n_rows);

    for (uint32_t i = 0; i < src.n_rows; ++i) {
        uint32_t start = src.row_offsets[i];
        uint32_t end   = src.row_offsets[i+1];
        uint32_t row_nnz = end - start;

        // Optimization: Reserve memory for the specific row density
        // Access via index; HostMatrix::rows is a std::vector<std::vector<uint32_t>>.
        out.rows[i].reserve(row_nnz);

        for (uint32_t k = start; k < end; ++k) {
            out.rows[i].push_back(src.col_indices[k]);
        }
    }
    return out;
}

} // namespace matrix
} // namespace mpqs
