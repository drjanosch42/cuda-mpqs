// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "device_csr.h"
#include "kernels.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

/// Gather row lengths in permuted order:
///   d_lengths[i] = d_row_ptr[d_perm[i]+1] - d_row_ptr[d_perm[i]]
__global__ void k_gather_row_lengths(
    const uint32_t* d_row_ptr, const uint32_t* d_perm,
    uint32_t* d_lengths, uint32_t n_rows)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        uint32_t old_row = d_perm[i];
        d_lengths[i] = d_row_ptr[old_row + 1] - d_row_ptr[old_row];
    }
}

/// Scatter-copy col_ind segments from old CSR to new CSR according to permutation.
/// Each thread handles one row: copies col_ind[old_start..old_end) to new position.
__global__ void k_scatter_csr_rows(
    const uint32_t* d_old_row_ptr, const uint32_t* d_old_col_ind,
    const uint32_t* d_perm,
    const uint32_t* d_new_row_ptr, uint32_t* d_new_col_ind,
    uint32_t n_rows)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        uint32_t old_row   = d_perm[i];
        uint32_t old_start = d_old_row_ptr[old_row];
        uint32_t old_end   = d_old_row_ptr[old_row + 1];
        uint32_t new_start = d_new_row_ptr[i];
        for (uint32_t j = 0; j < (old_end - old_start); ++j) {
            d_new_col_ind[new_start + j] = d_old_col_ind[old_start + j];
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceCSR member functions
// ---------------------------------------------------------------------------

void DeviceCSR::free() {
    if (d_row_ptr)      { cudaFree(d_row_ptr);      d_row_ptr      = nullptr; }
    if (d_col_ind)      { cudaFree(d_col_ind);      d_col_ind      = nullptr; }
    if (d_density_perm) { cudaFree(d_density_perm);  d_density_perm = nullptr; }
    n_rows = n_cols = n_dense_rows = 0;
    nnz = 0;
}

size_t DeviceCSR::device_bytes() const {
    size_t bytes = 0;
    if (d_row_ptr)      bytes += (n_rows + 1) * sizeof(uint32_t);
    if (d_col_ind)      bytes += nnz * sizeof(uint32_t);
    if (d_density_perm) bytes += n_rows * sizeof(uint32_t);
    return bytes;
}

DeviceCSRSlice DeviceCSR::slice(uint32_t start_row, uint32_t row_count) const {
    DeviceCSRSlice s;
    s.d_row_ptr = d_row_ptr + start_row;
    s.d_col_ind = d_col_ind;  // Shared — slice reads from row_ptr offsets
    s.start_row = start_row;
    s.n_rows    = row_count;
    s.n_cols    = n_cols;

    // Read row_ptr[start_row] and row_ptr[start_row + row_count] to compute NNZ
    uint32_t endpoints[2];
    CUDA_CHECK(cudaMemcpy(&endpoints[0], d_row_ptr + start_row,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&endpoints[1], d_row_ptr + start_row + row_count,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost));
    s.nnz = endpoints[1] - endpoints[0];
    return s;
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

DeviceCSR upload_host_matrix_to_device_csr(const HostMatrix& mat, cudaStream_t stream) {
    // 1. Build CSR on host (same logic as bw_spmm_interface.cpp:565-575)
    std::vector<uint32_t> row_ptr, col_ind;
    row_ptr.resize(mat.n_rows + 1);
    size_t total_nnz = 0;
    for (const auto& r : mat.rows) total_nnz += r.size();
    col_ind.reserve(total_nnz);
    row_ptr[0] = 0;
    for (size_t i = 0; i < mat.n_rows; ++i) {
        col_ind.insert(col_ind.end(), mat.rows[i].begin(), mat.rows[i].end());
        row_ptr[i + 1] = static_cast<uint32_t>(col_ind.size());
    }

    // 2. Upload: 2× cudaMalloc + 2× cudaMemcpyAsync
    DeviceCSR csr;
    csr.n_rows = mat.n_rows;
    csr.n_cols = mat.n_cols;
    csr.nnz    = total_nnz;

    CUDA_CHECK(cudaMalloc(&csr.d_row_ptr, (csr.n_rows + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&csr.d_col_ind, csr.nnz * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(csr.d_row_ptr, row_ptr.data(),
        (csr.n_rows + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(csr.d_col_ind, col_ind.data(),
        csr.nnz * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return csr;
}

void device_csr_permute_by_density(DeviceCSR& csr, uint32_t dense_threshold, cudaStream_t stream) {
    // 1. Compute density permutation (existing primitive: kernels.cu)
    csr.d_density_perm = gpu_compute_density_permutation(
        csr.d_row_ptr, csr.n_rows, /*ascending=*/false);

    // 2. Gather row lengths in permuted order
    uint32_t* d_new_lengths;
    CUDA_CHECK(cudaMalloc(&d_new_lengths, csr.n_rows * sizeof(uint32_t)));

    const uint32_t block_size = 256;
    const uint32_t grid_size = (csr.n_rows + block_size - 1) / block_size;

    k_gather_row_lengths<<<grid_size, block_size, 0, stream>>>(
        csr.d_row_ptr, csr.d_density_perm, d_new_lengths, csr.n_rows);

    // 3. Build new row_ptr via exclusive scan
    uint32_t* d_new_row_ptr;
    CUDA_CHECK(cudaMalloc(&d_new_row_ptr, (csr.n_rows + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_new_row_ptr, 0, sizeof(uint32_t), stream));

    thrust::device_ptr<uint32_t> p_len(d_new_lengths);
    thrust::device_ptr<uint32_t> p_ptr(d_new_row_ptr + 1);
    thrust::inclusive_scan(thrust::cuda::par.on(stream),
                           p_len, p_len + csr.n_rows, p_ptr);

    // 4. Scatter-copy col_ind segments
    uint32_t* d_new_col_ind;
    CUDA_CHECK(cudaMalloc(&d_new_col_ind, csr.nnz * sizeof(uint32_t)));

    k_scatter_csr_rows<<<grid_size, block_size, 0, stream>>>(
        csr.d_row_ptr, csr.d_col_ind, csr.d_density_perm,
        d_new_row_ptr, d_new_col_ind, csr.n_rows);

    // 5. Replace old arrays
    CUDA_CHECK(cudaFree(csr.d_row_ptr));
    CUDA_CHECK(cudaFree(csr.d_col_ind));
    csr.d_row_ptr = d_new_row_ptr;
    csr.d_col_ind = d_new_col_ind;
    CUDA_CHECK(cudaFree(d_new_lengths));

    // 6. Compute n_dense_rows
    uint32_t threshold = dense_threshold > 0 ? dense_threshold : csr.n_cols / 4;
    if (threshold > 0) {
        // Rows are sorted by descending NNZ. Find the first row below threshold.
        // Read permuted row lengths from new_row_ptr on host.
        // Binary search: we only need to find the boundary, so read row_ptr values
        // at candidate positions rather than the full array.
        // For simplicity and correctness, use a device-side approach:
        // row i has NNZ = row_ptr[i+1] - row_ptr[i]. We need the largest i
        // such that row_ptr[i+1] - row_ptr[i] >= threshold.
        // Since rows are sorted by descending NNZ, we can binary search on host
        // by reading O(log n) values from device.
        uint32_t lo = 0, hi = csr.n_rows;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            uint32_t endpoints[2];
            CUDA_CHECK(cudaMemcpy(&endpoints[0], csr.d_row_ptr + mid,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&endpoints[1], csr.d_row_ptr + mid + 1,
                                   sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t row_nnz = endpoints[1] - endpoints[0];
            if (row_nnz >= threshold) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        csr.n_dense_rows = lo;
    }
}

DeviceCSR device_csr_transpose(const DeviceCSR& csr, cudaStream_t stream) {
    DeviceCSR at;
    at.n_rows = csr.n_cols;
    at.n_cols = csr.n_rows;
    at.nnz    = csr.nnz;

    // Delegate to existing gpu_transpose_csr (kernels.cu)
    gpu_transpose_csr(csr.n_rows, csr.n_cols, csr.nnz,
                      csr.d_row_ptr, csr.d_col_ind,
                      &at.d_row_ptr, &at.d_col_ind);
    return at;
}
