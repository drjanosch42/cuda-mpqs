// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_packed_expanded.cu
//
// GPU packed expanded matrix construction: 4 kernels (E1–E4) + host driver.
// Builds the (F+2+L)-column packed CSR directly on device from the sieve
// stage's persistent RelationBatch (smooths) and LP witness batch (partials).
//
// Kernel E1: extract LP values (uint128 → uint64 cast)
// Kernel E2: count NNZ per row (ALL nonzero-exponent factors)
// Kernel E3: fill packed CSR entries with insertion sort on FB columns
// Kernel E4: copy per-row metadata (sqrt_Q, signs, val_2_exps)

#include "gpu_packed_expanded.cuh"
#include "matrix_utils.h"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <cstdint>
#include <iomanip>

namespace mpqs {
namespace matrix {

/// Predicate for thrust::copy_if: selects LP values > 1 (LP-combined smooths).
struct GreaterThanOne {
    __host__ __device__ bool operator()(uint64_t v) const { return v > 1; }
};

// ============================================================================
// Kernel E1: Extract LP values from unsigned __int128 to uint64_t
// ============================================================================

__global__ __launch_bounds__(256)
void extract_lp_values_kernel(
    const unsigned __int128* __restrict__ d_large_primes,
    uint64_t* __restrict__ d_lp_keys,
    uint32_t n_partial)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_partial) return;
    d_lp_keys[i] = static_cast<uint64_t>(d_large_primes[i]);
}

// ============================================================================
// Kernel E2: Count NNZ per row (all nonzero-exponent factors)
// ============================================================================

__global__ __launch_bounds__(256)
void count_row_nnz_packed_kernel(
    // Smooth batch fields
    const uint8_t*  __restrict__ smooth_signs,
    const int32_t*  __restrict__ smooth_val2,
    const uint64_t* __restrict__ smooth_foff,
    const uint8_t*  __restrict__ smooth_fcnt,
    const unsigned __int128* __restrict__ smooth_large_primes,   // for LP-combined detection
    // Partial batch fields
    const uint8_t*  __restrict__ partial_signs,
    const int32_t*  __restrict__ partial_val2,
    const uint64_t* __restrict__ partial_foff,
    const uint8_t*  __restrict__ partial_fcnt,
    // Dimensions
    uint32_t n_smooth,
    uint32_t total_rows,
    // Output
    uint32_t* __restrict__ d_row_nnz)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows) return;

    // Select batch and relation index
    const uint8_t*  signs;
    const int32_t*  val2;
    const uint64_t* foff;
    const uint8_t*  fcnt;
    uint32_t rel_idx;
    bool has_lp;

    if (r < n_smooth) {
        signs   = smooth_signs;
        val2    = smooth_val2;
        foff    = smooth_foff;
        fcnt    = smooth_fcnt;
        rel_idx = r;
        // LP-combined smooth: large_primes[r] > 1 means L² factor present
        has_lp  = (smooth_large_primes != nullptr &&
                   static_cast<uint64_t>(smooth_large_primes[r]) > 1);
    } else {
        signs   = partial_signs;
        val2    = partial_val2;
        foff    = partial_foff;
        fcnt    = partial_fcnt;
        rel_idx = r - n_smooth;
        has_lp  = true;
    }

    uint32_t count = 0;

    // Sign column (index 0, exponent 1): present if sign != 1
    if (signs[rel_idx] != 1) count++;

    // Prime-2 column (index 1, exponent = val_2_exp): present if val_2_exp != 0
    if (val2[rel_idx] != 0) count++;

    // FB columns: count ALL nonzero-exponent factors
    uint64_t off_start = foff[rel_idx];
    uint64_t off_end   = foff[rel_idx + 1];
    for (uint64_t j = off_start; j < off_end; ++j) {
        if (fcnt[j] != 0) count++;
    }

    // LP column: present for all partial rows AND LP-combined smooth rows
    if (has_lp) count++;

    d_row_nnz[r] = count;
}

// ============================================================================
// Kernel E3: Fill packed CSR entries with per-row insertion sort on FB columns
// ============================================================================

__global__ __launch_bounds__(256)
void fill_packed_csr_kernel(
    // Smooth batch fields
    const uint8_t*  __restrict__ smooth_signs,
    const int32_t*  __restrict__ smooth_val2,
    const uint64_t* __restrict__ smooth_foff,
    const uint32_t* __restrict__ smooth_fidx,
    const uint8_t*  __restrict__ smooth_fcnt,
    const unsigned __int128* __restrict__ smooth_large_primes,   // for LP-combined detection
    const uint32_t* __restrict__ d_smooth_lp_col_offset,         // LP col offsets for smooth rows
    // Partial batch fields
    const uint8_t*  __restrict__ partial_signs,
    const int32_t*  __restrict__ partial_val2,
    const uint64_t* __restrict__ partial_foff,
    const uint32_t* __restrict__ partial_fidx,
    const uint8_t*  __restrict__ partial_fcnt,
    // LP column offsets (per-partial)
    const uint32_t* __restrict__ d_lp_col_offset,
    // Dimensions
    uint32_t n_smooth,
    uint32_t total_rows,
    uint32_t fb_size,
    // CSR structure (output)
    const uint32_t* __restrict__ d_row_offsets,
    PackedEntry*    __restrict__ d_entries)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows) return;

    // Select batch and relation index
    const uint8_t*  signs;
    const int32_t*  val2;
    const uint64_t* foff;
    const uint32_t* fidx;
    const uint8_t*  fcnt;
    uint32_t rel_idx;
    bool has_lp;

    if (r < n_smooth) {
        signs   = smooth_signs;
        val2    = smooth_val2;
        foff    = smooth_foff;
        fidx    = smooth_fidx;
        fcnt    = smooth_fcnt;
        rel_idx = r;
        // LP-combined smooth: large_primes[r] > 1 means L² factor present
        has_lp  = (smooth_large_primes != nullptr &&
                   static_cast<uint64_t>(smooth_large_primes[r]) > 1);
    } else {
        signs   = partial_signs;
        val2    = partial_val2;
        foff    = partial_foff;
        fidx    = partial_fidx;
        fcnt    = partial_fcnt;
        rel_idx = r - n_smooth;
        has_lp  = true;
    }

    uint32_t write_pos = d_row_offsets[r];

    // Sign column (index 0, exponent 1)
    if (signs[rel_idx] != 1) {
        d_entries[write_pos++] = make_packed(0, 1);
    }

    // Prime-2 column (index 1, exponent = val_2_exp)
    if (val2[rel_idx] != 0) {
        d_entries[write_pos++] = make_packed(1, static_cast<uint8_t>(val2[rel_idx]));
    }

    // FB columns: all nonzero-exponent factors, then insertion sort by col_index
    uint32_t fb_start = write_pos;
    uint64_t off_start = foff[rel_idx];
    uint64_t off_end   = foff[rel_idx + 1];
    for (uint64_t j = off_start; j < off_end; ++j) {
        if (fcnt[j] != 0) {
            d_entries[write_pos++] = make_packed(fidx[j] + 2, fcnt[j]);
        }
    }

    // Insertion sort on col_index portion of FB entries
    uint32_t fb_count = write_pos - fb_start;
    for (uint32_t a = 1; a < fb_count; ++a) {
        PackedEntry key = d_entries[fb_start + a];
        uint32_t key_col = packed_col(key);
        int32_t b = static_cast<int32_t>(a) - 1;
        while (b >= 0 && packed_col(d_entries[fb_start + b]) > key_col) {
            d_entries[fb_start + b + 1] = d_entries[fb_start + b];
            b--;
        }
        d_entries[fb_start + b + 1] = key;
    }

    // LP column (always highest index, appending preserves sort order)
    if (has_lp) {
        if (r < n_smooth) {
            // LP-combined smooth: exponent 2 (L² factor from combining two partials)
            uint32_t lp_col = fb_size + 2 + d_smooth_lp_col_offset[rel_idx];
            d_entries[write_pos++] = make_packed(lp_col, 2);
        } else {
            // Raw partial: exponent 1
            uint32_t lp_col = fb_size + 2 + d_lp_col_offset[rel_idx];
            d_entries[write_pos++] = make_packed(lp_col, 1);
        }
    }
}

// ============================================================================
// Kernel E4: Copy per-row metadata from relation batches to DevicePackedCSR
// ============================================================================

__global__ __launch_bounds__(256)
void copy_row_metadata_kernel(
    // Smooth batch fields
    const mpqs::uint512* __restrict__ smooth_sqrt_Q,
    const uint8_t*       __restrict__ smooth_signs,
    const int32_t*       __restrict__ smooth_val2,
    // Partial batch fields
    const mpqs::uint512* __restrict__ partial_sqrt_Q,
    const uint8_t*       __restrict__ partial_signs,
    const int32_t*       __restrict__ partial_val2,
    // Dimensions
    uint32_t n_smooth,
    uint32_t total_rows,
    // Output
    mpqs::uint512* __restrict__ d_sqrt_Q,
    uint8_t*       __restrict__ d_signs,
    int32_t*       __restrict__ d_val_2_exps)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total_rows) return;

    if (r < n_smooth) {
        d_sqrt_Q[r]     = smooth_sqrt_Q[r];
        d_signs[r]      = smooth_signs[r];
        d_val_2_exps[r] = smooth_val2[r];
    } else {
        uint32_t idx = r - n_smooth;
        d_sqrt_Q[r]     = partial_sqrt_Q[idx];
        d_signs[r]      = partial_signs[idx];
        d_val_2_exps[r] = partial_val2[idx];
    }
}

// ============================================================================
// Host driver: gpuBuildPackedMatrix()
// ============================================================================

GpuPackedExpandedResult gpuBuildPackedMatrix(
    const structures::RelationBatchView& smooth_view,
    uint64_t n_smooth,
    const structures::RelationBatchView& partial_view,
    uint64_t n_partial,
    uint32_t fb_size)
{
    LOG_SET_MODULE("Matrix");

    const bool jetson = isJetsonDevice();
    if (jetson) {
        LOG(LOG_INFO) << "GPU packed matrix: using cudaMallocManaged (Jetson/integrated GPU).";
    }

    const uint32_t total_rows = static_cast<uint32_t>(n_smooth + n_partial);
    const uint32_t n_smooth32 = static_cast<uint32_t>(n_smooth);

    auto device_malloc = [&](void** ptr, size_t bytes) {
        if (jetson) {
            CUDA_CHECK(cudaMallocManaged(ptr, bytes));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, bytes));
        }
    };

    // ------------------------------------------------------------------
    // Phase 1: LP column assignment (Thrust sort + unique + lower_bound)
    // Includes LP values from BOTH raw partials AND LP-combined smooth rows.
    // LP-combined smooths have large_primes[i] > 1 (the real LP value from
    // global_combine_kernel) and represent L² factors in the factorization.
    // ------------------------------------------------------------------
    uint32_t n_unique_lp = 0;
    uint64_t* d_lp_sort_keys  = nullptr;   // combined sort buffer (partial + LP-combined smooth)
    uint64_t* d_lp_all_keys   = nullptr;   // unsorted partial keys for lower_bound
    uint32_t* d_lp_col_offset = nullptr;   // per-partial LP column offset
    uint64_t* d_smooth_lp_keys      = nullptr;  // smooth LP values (uint64 cast) for lower_bound
    uint32_t* d_smooth_lp_col_offset = nullptr; // per-smooth LP column offset

    if (n_partial > 0) {
        const uint32_t block = 256;

        // --- Extract partial LP values (unsorted copy for lower_bound) ---
        device_malloc(reinterpret_cast<void**>(&d_lp_all_keys),
                      n_partial * sizeof(uint64_t));
        device_malloc(reinterpret_cast<void**>(&d_lp_col_offset),
                      n_partial * sizeof(uint32_t));
        {
            const uint32_t grid = (static_cast<uint32_t>(n_partial) + block - 1) / block;
            extract_lp_values_kernel<<<grid, block>>>(
                partial_view.large_primes, d_lp_all_keys,
                static_cast<uint32_t>(n_partial));
            CUDA_CHECK(cudaGetLastError());
        }

        // --- Extract smooth LP values (for LP-combined detection + lower_bound) ---
        if (n_smooth > 0) {
            device_malloc(reinterpret_cast<void**>(&d_smooth_lp_keys),
                          n_smooth * sizeof(uint64_t));
            device_malloc(reinterpret_cast<void**>(&d_smooth_lp_col_offset),
                          n_smooth * sizeof(uint32_t));
            const uint32_t grid_s = (n_smooth32 + block - 1) / block;
            extract_lp_values_kernel<<<grid_s, block>>>(
                smooth_view.large_primes, d_smooth_lp_keys, n_smooth32);
            CUDA_CHECK(cudaGetLastError());
        }

        // --- Build combined sort buffer: all partial LPs + smooth LPs > 1 ---
        const uint64_t max_sort_keys = n_partial + n_smooth;
        device_malloc(reinterpret_cast<void**>(&d_lp_sort_keys),
                      max_sort_keys * sizeof(uint64_t));

        // Copy partial LPs into sort buffer
        CUDA_CHECK(cudaMemcpy(d_lp_sort_keys, d_lp_all_keys,
                               n_partial * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        uint64_t n_sort_keys = n_partial;

        // Append LP-combined smooth LP values (> 1) to sort buffer
        if (n_smooth > 0 && d_smooth_lp_keys != nullptr) {
            thrust::device_ptr<uint64_t> src_begin(d_smooth_lp_keys);
            thrust::device_ptr<uint64_t> dst(d_lp_sort_keys + n_partial);
            auto dst_end = thrust::copy_if(thrust::device,
                src_begin, src_begin + n_smooth, dst, GreaterThanOne());
            uint64_t n_smooth_lp = static_cast<uint64_t>(dst_end - dst);
            n_sort_keys += n_smooth_lp;
            if (n_smooth_lp > 0) {
                LOG(LOG_INFO) << "GPU packed matrix: " << n_smooth_lp
                              << " LP-combined smooth rows detected (L² entries).";
            }
        }

        // Sort + unique the combined LP keys
        thrust::device_ptr<uint64_t> sort_ptr(d_lp_sort_keys);
        thrust::sort(thrust::device, sort_ptr, sort_ptr + n_sort_keys);
        thrust::device_ptr<uint64_t> new_end =
            thrust::unique(thrust::device, sort_ptr, sort_ptr + n_sort_keys);
        n_unique_lp = static_cast<uint32_t>(new_end - sort_ptr);

        // lower_bound: map each partial's LP value to its column offset
        {
            thrust::device_ptr<uint64_t> all_ptr(d_lp_all_keys);
            thrust::device_ptr<uint32_t> offset_ptr(d_lp_col_offset);
            thrust::lower_bound(thrust::device,
                                sort_ptr, sort_ptr + n_unique_lp,
                                all_ptr, all_ptr + n_partial,
                                offset_ptr);
        }

        // lower_bound: map each smooth row's LP value to its column offset
        // (only meaningful for rows with large_primes > 1; others get unused offset)
        if (n_smooth > 0 && d_smooth_lp_keys != nullptr) {
            thrust::device_ptr<uint64_t> smooth_ptr(d_smooth_lp_keys);
            thrust::device_ptr<uint32_t> smooth_offset_ptr(d_smooth_lp_col_offset);
            thrust::lower_bound(thrust::device,
                                sort_ptr, sort_ptr + n_unique_lp,
                                smooth_ptr, smooth_ptr + n_smooth,
                                smooth_offset_ptr);
        }
    }

    const uint32_t total_cols = fb_size + 2 + n_unique_lp;

    LOG(LOG_INFO) << "GPU packed matrix: " << n_unique_lp << " distinct LP columns assigned";

    // ------------------------------------------------------------------
    // Phase 2: Count NNZ per row (E2) + prefix sum → row offsets
    // ------------------------------------------------------------------
    uint32_t* d_row_nnz = nullptr;
    device_malloc(reinterpret_cast<void**>(&d_row_nnz),
                  total_rows * sizeof(uint32_t));

    {
        const uint32_t block = 256;
        const uint32_t grid  = (total_rows + block - 1) / block;

        count_row_nnz_packed_kernel<<<grid, block>>>(
            smooth_view.signs, smooth_view.val_2_exps,
            smooth_view.factor_offsets, smooth_view.factor_counts,
            smooth_view.large_primes,  // LP-combined smooth detection
            partial_view.signs, partial_view.val_2_exps,
            partial_view.factor_offsets, partial_view.factor_counts,
            n_smooth32, total_rows,
            d_row_nnz);
        CUDA_CHECK(cudaGetLastError());
    }

    // Compute row offsets via inclusive scan: offsets[0] = 0, offsets[i+1] = sum(nnz[0..i])
    uint32_t* d_row_offsets_temp = nullptr;
    device_malloc(reinterpret_cast<void**>(&d_row_offsets_temp),
                  (total_rows + 1) * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_row_offsets_temp, 0, sizeof(uint32_t)));

    {
        thrust::device_ptr<uint32_t> nnz_ptr(d_row_nnz);
        thrust::device_ptr<uint32_t> offsets_ptr(d_row_offsets_temp + 1);
        thrust::inclusive_scan(thrust::device, nnz_ptr, nnz_ptr + total_rows, offsets_ptr);
    }

    // Read total NNZ from device
    uint32_t total_nnz = 0;
    CUDA_CHECK(cudaMemcpy(&total_nnz, d_row_offsets_temp + total_rows,
                           sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Phase 3: Allocate DevicePackedCSR + fill packed entries (E3) + metadata (E4)
    // ------------------------------------------------------------------
    GpuPackedExpandedResult result;
    result.csr.alloc(total_rows, total_cols, total_nnz, jetson);
    result.n_lp_cols = n_unique_lp;

    // Copy row offsets from temp buffer to DevicePackedCSR
    CUDA_CHECK(cudaMemcpy(result.csr.d_row_offsets, d_row_offsets_temp,
                           (total_rows + 1) * sizeof(uint32_t),
                           cudaMemcpyDeviceToDevice));

    // Launch E3: fill packed CSR entries
    if (total_nnz > 0) {
        const uint32_t block = 256;
        const uint32_t grid  = (total_rows + block - 1) / block;

        fill_packed_csr_kernel<<<grid, block>>>(
            smooth_view.signs, smooth_view.val_2_exps,
            smooth_view.factor_offsets, smooth_view.factor_indices,
            smooth_view.factor_counts,
            smooth_view.large_primes,   // LP-combined smooth detection
            d_smooth_lp_col_offset,     // LP col offsets for smooth rows
            partial_view.signs, partial_view.val_2_exps,
            partial_view.factor_offsets, partial_view.factor_indices,
            partial_view.factor_counts,
            d_lp_col_offset,
            n_smooth32, total_rows, fb_size,
            result.csr.d_row_offsets, result.csr.d_entries);
        CUDA_CHECK(cudaGetLastError());
    }

    // Launch E4: copy per-row metadata
    if (total_rows > 0) {
        const uint32_t block = 256;
        const uint32_t grid  = (total_rows + block - 1) / block;

        copy_row_metadata_kernel<<<grid, block>>>(
            smooth_view.sqrt_Q, smooth_view.signs, smooth_view.val_2_exps,
            partial_view.sqrt_Q, partial_view.signs, partial_view.val_2_exps,
            n_smooth32, total_rows,
            result.csr.d_sqrt_Q, result.csr.d_signs, result.csr.d_val_2_exps);
        CUDA_CHECK(cudaGetLastError());
    }

    // Synchronize before freeing temp buffers
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Cleanup: free all temporary device buffers
    // ------------------------------------------------------------------
    if (d_lp_sort_keys)         { CUDA_CHECK(cudaFree(d_lp_sort_keys)); }
    if (d_lp_all_keys)          { CUDA_CHECK(cudaFree(d_lp_all_keys)); }
    if (d_lp_col_offset)        { CUDA_CHECK(cudaFree(d_lp_col_offset)); }
    if (d_smooth_lp_keys)       { CUDA_CHECK(cudaFree(d_smooth_lp_keys)); }
    if (d_smooth_lp_col_offset) { CUDA_CHECK(cudaFree(d_smooth_lp_col_offset)); }
    if (d_row_nnz)              { CUDA_CHECK(cudaFree(d_row_nnz)); }
    if (d_row_offsets_temp)     { CUDA_CHECK(cudaFree(d_row_offsets_temp)); }

    // ------------------------------------------------------------------
    // Log matrix statistics
    // ------------------------------------------------------------------
    double density = (total_rows > 0 && total_cols > 0)
        ? static_cast<double>(total_nnz) / (static_cast<double>(total_rows) * total_cols) * 100.0
        : 0.0;
    double avg_per_row = (total_rows > 0)
        ? static_cast<double>(total_nnz) / total_rows
        : 0.0;
    LOG(LOG_INFO) << "GPU packed matrix: " << fmtNum(total_rows) << " rows x "
                  << fmtNum(total_cols) << " cols, " << fmtNum(total_nnz)
                  << " NNZ (" << std::fixed << std::setprecision(3) << density
                  << "% density, avg " << std::setprecision(1) << avg_per_row << " per row)";
    LOG(LOG_INFO) << "  Smooth rows: " << fmtNum(n_smooth)
                  << ", Partial rows: " << fmtNum(n_partial);

    return result;
}

} // namespace matrix
} // namespace mpqs
