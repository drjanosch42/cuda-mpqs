// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_compact_packed.cu
//
// GPU compaction of a post-merge scattered DevicePackedCSR (M10a).
// 4 CUDA kernels + host driver:
//   K1: mark_alive_kernel              — d_alive_mask from d_row_ptr
//   K2: compute_new_lengths_kernel     — resolve row lengths via indirection
//   K3: compact_entries_metadata_kernel — fused entry copy (with col remap) + metadata copy
//   K4: compute_col_weight_alive_kernel — accurate col weights from alive rows only
//
// Pipeline: K1 -> scan -> K2 -> scan -> K4 -> host col remap -> K3

#include "gpu_compact_packed.cuh"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace mpqs {
namespace matrix {

// ============================================================================
// K1: mark_alive_kernel
// ============================================================================

/// One thread per row. Writes 1 if the row is alive (not ROW_DEAD), 0 otherwise.
__global__ __launch_bounds__(256)
void mark_alive_kernel(
    const uint32_t* __restrict__ d_row_ptr,
    uint32_t n_rows,
    uint32_t* __restrict__ d_alive_mask)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    d_alive_mask[r] = (d_row_ptr[r] != ROW_DEAD) ? 1u : 0u;
}

// ============================================================================
// K2: compute_new_lengths_kernel
// ============================================================================

/// One thread per row. For each alive row, resolves its entry count via
/// d_row_ptr indirection (original CSR or workspace) and writes to
/// d_new_row_lengths at the compacted index from d_alive_scatter.
__global__ __launch_bounds__(256)
void compute_new_lengths_kernel(
    const uint32_t* __restrict__ d_row_ptr,
    const uint32_t* __restrict__ d_alive_mask,
    const uint32_t* __restrict__ d_alive_scatter,
    const uint32_t* __restrict__ orig_row_offsets,
    const uint32_t* __restrict__ ws_row_lengths,
    uint32_t n_rows,
    uint32_t* __restrict__ d_new_row_lengths)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || d_alive_mask[r] == 0) return;

    uint32_t new_r = d_alive_scatter[r];
    uint32_t ptr   = d_row_ptr[r];

    uint32_t len;
    if (ptr & ROW_WS_BIT) {
        uint32_t ws_idx = ptr & 0x7FFFFFFFu;
        len = ws_row_lengths[ws_idx];
    } else {
        len = orig_row_offsets[ptr + 1] - orig_row_offsets[ptr];
    }
    d_new_row_lengths[new_r] = len;
}

// ============================================================================
// K4: compute_col_weight_alive_kernel
// ============================================================================

/// One thread per row. For each alive row, resolves source entries via
/// d_row_ptr indirection and atomically increments column weights.
/// d_col_weight must be pre-zeroed before launch.
__global__ __launch_bounds__(256)
void compute_col_weight_alive_kernel(
    const uint32_t*    __restrict__ d_row_ptr,
    const uint32_t*    __restrict__ d_alive_mask,
    uint32_t n_rows,
    const uint32_t*    __restrict__ orig_row_offsets,
    const PackedEntry* __restrict__ orig_entries,
    const uint32_t*    __restrict__ ws_row_starts,
    const uint32_t*    __restrict__ ws_row_lengths,
    const PackedEntry* __restrict__ ws_entries,
    uint32_t*          d_col_weight)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || d_alive_mask[r] == 0) return;

    uint32_t ptr = d_row_ptr[r];
    const PackedEntry* row_data;
    uint32_t row_len;

    if (ptr & ROW_WS_BIT) {
        uint32_t ws_idx = ptr & 0x7FFFFFFFu;
        row_data = ws_entries + ws_row_starts[ws_idx];
        row_len  = ws_row_lengths[ws_idx];
    } else {
        row_data = orig_entries + orig_row_offsets[ptr];
        row_len  = orig_row_offsets[ptr + 1] - orig_row_offsets[ptr];
    }

    for (uint32_t i = 0; i < row_len; i++)
        atomicAdd(&d_col_weight[packed_col(row_data[i])], 1u);
}

// ============================================================================
// Plain-CSR GF(2) column weight kernel (for M12-S2 initial count)
// ============================================================================

/// One thread per row over a plain (no indirection) CSR. Atomically
/// increments GF(2) column weights for odd-exponent entries.
/// d_gf2_col_weight must be pre-zeroed before launch.
__global__ __launch_bounds__(256)
void compute_gf2_col_weights_plain_kernel(
    const uint32_t*    __restrict__ d_row_offsets,
    const PackedEntry* __restrict__ d_entries,
    uint32_t* __restrict__          d_gf2_col_weight,
    uint32_t n_rows)
{
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const uint32_t begin = d_row_offsets[r];
    const uint32_t end   = d_row_offsets[r + 1];
    for (uint32_t j = begin; j < end; ++j)
        if (packed_exp(d_entries[j]) & 1)
            atomicAdd(&d_gf2_col_weight[packed_col(d_entries[j])], 1u);
}

/// Helper: count GF(2)-alive columns (i.e. columns with at least one
/// odd-exponent entry) in a plain DevicePackedCSR. Used by M12-S2 to
/// snapshot `initial_gf2_cols` before any merge. Cost: one kernel launch
/// + one D→H of n_cols uint32_t.
static uint32_t countGf2AliveCols(const DevicePackedCSR& csr) {
    if (csr.n_rows == 0 || csr.n_cols == 0) return 0;
    bool jetson = isJetsonDevice();
    uint32_t* d_gf2 = nullptr;
    if (jetson) { CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&d_gf2),
                                                csr.n_cols * sizeof(uint32_t))); }
    else        { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gf2),
                                        csr.n_cols * sizeof(uint32_t))); }
    CUDA_CHECK(cudaMemset(d_gf2, 0, csr.n_cols * sizeof(uint32_t)));

    uint32_t blocks = (csr.n_rows + 255) / 256;
    compute_gf2_col_weights_plain_kernel<<<blocks, 256>>>(
        csr.d_row_offsets, csr.d_entries, d_gf2, csr.n_rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_gf2(csr.n_cols);
    CUDA_CHECK(cudaMemcpy(h_gf2.data(), d_gf2,
                          csr.n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gf2));

    uint32_t alive = 0;
    for (uint32_t c = 0; c < csr.n_cols; ++c)
        if (h_gf2[c] > 0) ++alive;
    return alive;
}

// ============================================================================
// K4-GF2: compute_gf2_col_weight_alive_kernel
// ============================================================================

/// GF(2) column weights from alive rows (odd-exponent entries only).
/// Mirrors compute_col_weight_alive_kernel with packed_exp parity guard.
/// d_gf2_col_weight must be pre-zeroed before launch.
__global__ __launch_bounds__(256)
void compute_gf2_col_weight_alive_kernel(
    const uint32_t*    __restrict__ d_row_ptr,
    const uint32_t*    __restrict__ d_alive_mask,
    uint32_t n_rows,
    const uint32_t*    __restrict__ orig_row_offsets,
    const PackedEntry* __restrict__ orig_entries,
    const uint32_t*    __restrict__ ws_row_starts,
    const uint32_t*    __restrict__ ws_row_lengths,
    const PackedEntry* __restrict__ ws_entries,
    uint32_t*          d_gf2_col_weight)
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || d_alive_mask[r] == 0) return;

    uint32_t ptr = d_row_ptr[r];
    const PackedEntry* row_data;
    uint32_t row_len;

    if (ptr & ROW_WS_BIT) {
        uint32_t ws_idx = ptr & 0x7FFFFFFFu;
        row_data = ws_entries + ws_row_starts[ws_idx];
        row_len  = ws_row_lengths[ws_idx];
    } else {
        row_data = orig_entries + orig_row_offsets[ptr];
        row_len  = orig_row_offsets[ptr + 1] - orig_row_offsets[ptr];
    }

    for (uint32_t i = 0; i < row_len; i++)
        if (packed_exp(row_data[i]) & 1)  // GF(2): odd exponents only
            atomicAdd(&d_gf2_col_weight[packed_col(row_data[i])], 1u);
}

// ============================================================================
// K3 (FUSED): compact_entries_metadata_kernel
// ============================================================================

/// One thread per row. For each alive row, copies entries (with column
/// remapping via d_col_remap) and per-row metadata (sqrt_Q, signs, val_2_exps)
/// from the source location (original CSR or workspace) into the fresh
/// contiguous output buffers.
__global__ __launch_bounds__(256)
void compact_entries_metadata_kernel(
    const uint32_t*    __restrict__ d_row_ptr,
    const uint32_t*    __restrict__ d_alive_mask,
    const uint32_t*    __restrict__ d_alive_scatter,
    uint32_t n_rows,
    // Original CSR
    const uint32_t*    __restrict__ orig_row_offsets,
    const PackedEntry* __restrict__ orig_entries,
    const uint512*     __restrict__ orig_sqrt_Q,
    const uint8_t*     __restrict__ orig_signs,
    const int32_t*     __restrict__ orig_val_2_exps,
    const uint32_t*    __restrict__ orig_char_bits,   // Stage 6: branch char vector
    // Workspace
    const uint32_t*    __restrict__ ws_row_starts,
    const uint32_t*    __restrict__ ws_row_lengths,
    const PackedEntry* __restrict__ ws_entries,
    const uint512*     __restrict__ ws_sqrt_Q,
    const uint8_t*     __restrict__ ws_signs,
    const int32_t*     __restrict__ ws_val_2_exps,
    const uint32_t*    __restrict__ ws_char_bits,     // Stage 6: branch char vector
    // New row offsets (output offsets for entries)
    const uint32_t*    __restrict__ d_new_row_offsets,
    // Column remap: old_col -> new_col (UINT32_MAX for dead columns)
    const uint32_t*    __restrict__ d_col_remap,
    // Outputs: new compacted CSR (entries + metadata)
    PackedEntry* __restrict__ d_new_entries,
    uint512*     __restrict__ d_new_sqrt_Q,
    uint8_t*     __restrict__ d_new_signs,
    int32_t*     __restrict__ d_new_val_2_exps,
    uint32_t*    __restrict__ d_new_char_bits)        // Stage 6: relocated char vector
{
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows || d_alive_mask[r] == 0) return;

    uint32_t new_r = d_alive_scatter[r];
    uint32_t ptr   = d_row_ptr[r];

    // Resolve source row
    const PackedEntry* src_entries;
    uint32_t           src_len;
    uint512            src_sqrt_Q;
    uint8_t            src_sign;
    int32_t            src_v2;
    uint32_t           src_char_bits;   // Stage 6: branch char vector

    if (ptr & ROW_WS_BIT) {
        uint32_t ws_idx = ptr & 0x7FFFFFFFu;
        src_entries     = ws_entries + ws_row_starts[ws_idx];
        src_len         = ws_row_lengths[ws_idx];
        src_sqrt_Q      = ws_sqrt_Q[ws_idx];
        src_sign        = ws_signs[ws_idx];
        src_v2          = ws_val_2_exps[ws_idx];
        src_char_bits   = ws_char_bits[ws_idx];
    } else {
        src_entries     = orig_entries + orig_row_offsets[ptr];
        src_len         = orig_row_offsets[ptr + 1] - orig_row_offsets[ptr];
        src_sqrt_Q      = orig_sqrt_Q[ptr];
        src_sign        = orig_signs[ptr];
        src_v2          = orig_val_2_exps[ptr];
        src_char_bits   = orig_char_bits[ptr];
    }

    // Copy entries with column remapping
    uint32_t dst = d_new_row_offsets[new_r];
    for (uint32_t i = 0; i < src_len; i++) {
        PackedEntry e   = src_entries[i];
        uint32_t old_c  = packed_col(e);
        uint32_t new_c  = d_col_remap[old_c];
        // new_c is always valid: K4 counted this column in an alive row,
        // so weight > 0 -> valid new index in col remap.
        d_new_entries[dst + i] = make_packed(new_c, packed_exp(e));
    }

    // Copy metadata
    d_new_sqrt_Q[new_r]     = src_sqrt_Q;
    d_new_signs[new_r]      = src_sign;
    d_new_val_2_exps[new_r] = src_v2;
    // Stage 6: compaction RELOCATES the row; the composed char vector carries
    // through unchanged (no XOR — that already happened in the merge kernel).
    d_new_char_bits[new_r]  = src_char_bits;
}

// ============================================================================
// Host driver
// ============================================================================

CompactResult gpuCompactPackedCSR(
    const DevicePackedCSR& csr,
    const BatchMergeResult& merge)
{
    LOG_SET_MODULE("Matrix");

    const uint32_t n_rows = csr.n_rows;
    const uint32_t n_cols = csr.n_cols;
    const DeviceMergeWorkspace& ws = merge.workspace;

    bool jetson = isJetsonDevice();

    // Early exit for degenerate inputs.
    if (n_rows == 0) {
        LOG(LOG_INFO) << "M10 compact: empty input (n_rows=0), returning empty result.";
        CompactResult empty;
        empty.csr.alloc(0, 0, 0, jetson);
        return empty;
    }

    // ---- Step 1: Allocate temporaries ----
    //
    // d_alive_mask[n_rows]          -- 1 for alive, 0 for dead rows
    // d_alive_scatter[n_rows]       -- exclusive scan of d_alive_mask -> new row indices
    // d_new_row_lengths[n_rows]     -- output lengths (only n_alive entries used)
    // d_new_row_offsets[n_rows + 1] -- exclusive scan of lengths + sentinel
    // d_accurate_col_weight[n_cols] -- col weights from alive rows only
    // d_col_remap[n_cols]           -- old_col -> new_col (UINT32_MAX for dead cols)

    uint32_t* d_alive_mask          = nullptr;
    uint32_t* d_alive_scatter       = nullptr;
    uint32_t* d_new_row_lengths     = nullptr;
    uint32_t* d_new_row_offsets     = nullptr;
    uint32_t* d_accurate_col_weight = nullptr;
    uint32_t* d_col_remap           = nullptr;
    uint32_t* d_gf2_col_weight      = nullptr;

    auto dev_alloc = [&](void** ptr, size_t sz) {
        if (jetson) { CUDA_CHECK(cudaMallocManaged(ptr, sz)); }
        else        { CUDA_CHECK(cudaMalloc(ptr, sz)); }
    };

    dev_alloc(reinterpret_cast<void**>(&d_alive_mask),          n_rows * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_alive_scatter),       n_rows * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_new_row_lengths),     n_rows * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_new_row_offsets),     (n_rows + 1) * sizeof(uint32_t));
    // Guard against n_cols == 0 for allocation (at least 1 element).
    const size_t safe_cols = (n_cols > 0) ? n_cols : 1u;
    dev_alloc(reinterpret_cast<void**>(&d_accurate_col_weight), safe_cols * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_col_remap),           safe_cols * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_gf2_col_weight),      safe_cols * sizeof(uint32_t));

    // ---- Step 2: K1 -- mark alive rows ----
    {
        uint32_t blocks = (n_rows + 255) / 256;
        mark_alive_kernel<<<blocks, 256>>>(ws.d_row_ptr, n_rows, d_alive_mask);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Step 3: Thrust exclusive_scan -> scatter map + n_alive ----
    {
        thrust::device_ptr<uint32_t> mask_ptr(d_alive_mask);
        thrust::device_ptr<uint32_t> scat_ptr(d_alive_scatter);
        thrust::exclusive_scan(thrust::device, mask_ptr, mask_ptr + n_rows, scat_ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // n_alive = d_alive_mask[n_rows-1] + d_alive_scatter[n_rows-1]
    uint32_t last_mask = 0, last_scat = 0;
    CUDA_CHECK(cudaMemcpy(&last_mask, d_alive_mask + n_rows - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_scat, d_alive_scatter + n_rows - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t n_alive = last_scat + last_mask;

    // Build h_row_map on host (compacted_row -> original_row)
    std::vector<uint32_t> h_alive_mask(n_rows), h_alive_scatter(n_rows);
    CUDA_CHECK(cudaMemcpy(h_alive_mask.data(), d_alive_mask,
                          n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_alive_scatter.data(), d_alive_scatter,
                          n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> h_row_map(n_alive);
    for (uint32_t r = 0; r < n_rows; r++) {
        if (h_alive_mask[r]) h_row_map[h_alive_scatter[r]] = r;
    }

    // ---- Step 4: K2 -- compute new row lengths ----
    {
        uint32_t blocks = (n_rows + 255) / 256;
        compute_new_lengths_kernel<<<blocks, 256>>>(
            ws.d_row_ptr, d_alive_mask, d_alive_scatter,
            csr.d_row_offsets, ws.d_ws_row_lengths,
            n_rows, d_new_row_lengths);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Step 5: Thrust exclusive_scan -> new row offsets + new_nnz ----
    uint32_t new_nnz = 0;
    if (n_alive > 0) {
        thrust::device_ptr<uint32_t> len_ptr(d_new_row_lengths);
        thrust::device_ptr<uint32_t> off_ptr(d_new_row_offsets);
        thrust::exclusive_scan(thrust::device, len_ptr, len_ptr + n_alive, off_ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t last_off = 0, last_len = 0;
        CUDA_CHECK(cudaMemcpy(&last_off, d_new_row_offsets + n_alive - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_len, d_new_row_lengths + n_alive - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        new_nnz = last_off + last_len;
        // Write sentinel at d_new_row_offsets[n_alive]
        CUDA_CHECK(cudaMemcpy(d_new_row_offsets + n_alive, &new_nnz,
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ---- Step 6: K4 -- compute accurate col weights from alive rows ----
    {
        CUDA_CHECK(cudaMemset(d_accurate_col_weight, 0, n_cols * sizeof(uint32_t)));
        uint32_t blocks = (n_rows + 255) / 256;
        compute_col_weight_alive_kernel<<<blocks, 256>>>(
            ws.d_row_ptr, d_alive_mask, n_rows,
            csr.d_row_offsets, csr.d_entries,
            ws.d_ws_row_starts, ws.d_ws_row_lengths, ws.d_ws_entries,
            d_accurate_col_weight);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Step 6b: K4-GF2 -- compute GF(2) col weights from alive rows ----
    {
        CUDA_CHECK(cudaMemset(d_gf2_col_weight, 0, n_cols * sizeof(uint32_t)));
        uint32_t blocks = (n_rows + 255) / 256;
        compute_gf2_col_weight_alive_kernel<<<blocks, 256>>>(
            ws.d_row_ptr, d_alive_mask, n_rows,
            csr.d_row_offsets, csr.d_entries,
            ws.d_ws_row_starts, ws.d_ws_row_lengths, ws.d_ws_entries,
            d_gf2_col_weight);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Step 7: Build col remap on host ----
    std::vector<uint32_t> h_accurate_col_weight(n_cols);
    if (n_cols > 0) {
        CUDA_CHECK(cudaMemcpy(h_accurate_col_weight.data(), d_accurate_col_weight,
                              n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    uint32_t new_n_cols = 0;
    std::vector<uint32_t> h_old_to_new(n_cols, UINT32_MAX);
    std::vector<uint32_t> h_col_map;
    h_col_map.reserve(n_cols);
    for (uint32_t c = 0; c < n_cols; c++) {
        if (h_accurate_col_weight[c] > 0) {
            h_old_to_new[c] = new_n_cols++;
            h_col_map.push_back(c);
        }
    }

    // Upload d_col_remap
    if (n_cols > 0) {
        CUDA_CHECK(cudaMemcpy(d_col_remap, h_old_to_new.data(),
                              n_cols * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ---- Step 8: Allocate fresh DevicePackedCSR ----
    DevicePackedCSR new_csr;
    new_csr.alloc(n_alive, new_n_cols, new_nnz, jetson);

    // Copy new row offsets into new_csr.d_row_offsets
    if (n_alive > 0) {
        CUDA_CHECK(cudaMemcpy(new_csr.d_row_offsets, d_new_row_offsets,
                              (n_alive + 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    }

    // ---- Step 9: K3 (FUSED) -- compact entries + metadata ----
    if (n_alive > 0) {
        uint32_t blocks = (n_rows + 255) / 256;
        compact_entries_metadata_kernel<<<blocks, 256>>>(
            ws.d_row_ptr, d_alive_mask, d_alive_scatter, n_rows,
            csr.d_row_offsets, csr.d_entries, csr.d_sqrt_Q, csr.d_signs, csr.d_val_2_exps,
            csr.d_char_bits,
            ws.d_ws_row_starts, ws.d_ws_row_lengths, ws.d_ws_entries,
            ws.d_ws_sqrt_Q, ws.d_ws_signs, ws.d_ws_val_2_exps, ws.d_ws_char_bits,
            d_new_row_offsets, d_col_remap,
            new_csr.d_entries, new_csr.d_sqrt_Q, new_csr.d_signs, new_csr.d_val_2_exps,
            new_csr.d_char_bits);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Step 10: Log compaction result (with GF(2) dimension) ----
    uint32_t gf2_new_n_cols = 0;
    {
        std::vector<uint32_t> h_gf2_col_weight(n_cols > 0 ? n_cols : 1u, 0u);
        if (n_cols > 0) {
            CUDA_CHECK(cudaMemcpy(h_gf2_col_weight.data(), d_gf2_col_weight,
                                  n_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        for (uint32_t c = 0; c < n_cols; ++c)
            if (h_gf2_col_weight[c] > 0) ++gf2_new_n_cols;
    }
    LOG(LOG_INFO) << "M10 compact: " << n_rows << " -> " << n_alive << " rows, "
                  << n_cols << " -> " << new_n_cols << " cols (GF2: "
                  << gf2_new_n_cols << "), " << new_nnz << " entries.";

    // ---- Step 11: Free temporaries ----
    CUDA_CHECK(cudaFree(d_alive_mask));
    CUDA_CHECK(cudaFree(d_alive_scatter));
    CUDA_CHECK(cudaFree(d_new_row_lengths));
    CUDA_CHECK(cudaFree(d_new_row_offsets));
    CUDA_CHECK(cudaFree(d_accurate_col_weight));
    CUDA_CHECK(cudaFree(d_col_remap));
    CUDA_CHECK(cudaFree(d_gf2_col_weight));

    return CompactResult{ std::move(new_csr), std::move(h_row_map), std::move(h_col_map),
                          gf2_new_n_cols };
}

// ============================================================================
// M10b: Multi-cycle compact-merge driver
// ============================================================================

/// Compute merge budget for one cycle.
/// budget = n_rows - ceil(truncation_factor × n_cols), or 0 if already at/below target.
static uint32_t computeCycleBudget(const DevicePackedCSR& csr, double truncation_factor) {
    if (truncation_factor <= 0.0) return 0;  // 0 = no cap = unlimited
    uint32_t target = static_cast<uint32_t>(std::ceil(truncation_factor * csr.n_cols));
    return (csr.n_rows > target) ? (csr.n_rows - target) : 0;
}

/// Count alive rows in h_row_ptr (entries != ROW_DEAD).
static uint32_t countAliveRows(const std::vector<uint32_t>& h_row_ptr) {
    uint32_t alive = 0;
    for (uint32_t v : h_row_ptr)
        if (v != ROW_DEAD) ++alive;
    return alive;
}

CompactMergeResult gpuCompactMergeCycles(
    DevicePackedCSR csr,
    const MontgomeryContext& mont,
    uint32_t k_max,
    uint32_t max_weight,
    double truncation_factor,
    uint32_t max_cycles,
    double gf2_floor_factor,
    uint32_t gf2_min_floor)
{
    LOG_SET_MODULE("Matrix");

    // Stop when fewer than this fraction of alive rows were merged in a cycle.
    constexpr double kConvergenceThreshold = 0.02;

    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();

    // Cumulative maps: initialized to identity (cycle 0 operates on post-singleton rows/cols).
    std::vector<uint32_t> cumulative_row_map(csr.n_rows);
    std::iota(cumulative_row_map.begin(), cumulative_row_map.end(), 0);

    std::vector<uint32_t> cumulative_col_map(csr.n_cols);
    std::iota(cumulative_col_map.begin(), cumulative_col_map.end(), 0);

    uint32_t total_merges = 0;
    uint32_t cycles_run   = 0;

    // [M12-S2] Snapshot initial GF(2) col count (post-singleton, pre-merge) and
    // derive the diversity floor. Cost: one extra kernel launch + small D→H copy.
    const uint32_t initial_gf2_cols = countGf2AliveCols(csr);
    const uint32_t computed_floor =
        static_cast<uint32_t>(gf2_floor_factor *
                              static_cast<double>(initial_gf2_cols));
    const uint32_t gf2_floor = std::max(gf2_min_floor, computed_floor);
    LOG(LOG_INFO) << "  [M12-S2] Initial GF(2) cols (post-singleton): "
                  << initial_gf2_cols
                  << ", floor=" << gf2_floor
                  << " (factor=" << std::fixed << std::setprecision(3)
                  << gf2_floor_factor
                  << ", min=" << gf2_min_floor << ").";

    for (uint32_t cycle = 0; cycle < max_cycles; ++cycle) {
        uint32_t rows_before = csr.n_rows;
        uint32_t cols_before = csr.n_cols;

        // 1. Per-cycle budget
        uint32_t budget = computeCycleBudget(csr, truncation_factor);

        // 2. GPU batch merges for this cycle
        auto t_merge_start = clock::now();
        auto merge = gpuBatchMerge(csr, mont, k_max, max_weight, budget);
        double merge_sec = std::chrono::duration<double>(clock::now() - t_merge_start).count();
        uint32_t cycle_merges = merge.w2_merges + merge.hw_merges;
        total_merges += cycle_merges;

        // 3. Count alive rows for convergence check
        uint32_t alive = countAliveRows(merge.h_row_ptr);

        // 4. Convergence check: < kConvergenceThreshold of alive rows merged this cycle
        if (cycle_merges < static_cast<uint32_t>(alive * kConvergenceThreshold)) {
            LOG(LOG_INFO) << "  M10b: Converged at cycle " << cycle
                          << " (" << cycle_merges << " merges < 2% of " << alive << ").";
            // The merge from this cycle IS the final merge — skip extra pass.
            ++cycles_run;
            double total_sec = std::chrono::duration<double>(clock::now() - t_total_start).count();
            LOG(LOG_INFO) << "  M10: " << cycles_run << " cycles, "
                          << (total_merges / 1000) << "K total merges → "
                          << csr.n_rows << " x " << csr.n_cols
                          << " [total " << std::fixed << std::setprecision(1) << total_sec << "s]";
            return {std::move(csr), std::move(merge),
                    std::move(cumulative_row_map), std::move(cumulative_col_map),
                    total_merges, cycles_run};
        }

        // [M12-S2] GF(2) column-diversity floor: count GF(2)-alive columns from
        // the merge's GF(2) col-weight vector (post-merge, pre-compaction).
        // If below the floor, exit BEFORE compacting and return this cycle's
        // merge directly — mirrors the convergence-return path so that
        // merge.h_row_ptr remains consistent with csr.
        uint32_t gf2_alive_pre_compact = 0;
        for (uint32_t c = 0; c < merge.h_gf2_col_weight.size(); ++c)
            if (merge.h_gf2_col_weight[c] > 0) ++gf2_alive_pre_compact;

        const double rows_merged_pct = (rows_before > 0)
            ? 100.0 * static_cast<double>(cycle_merges) / static_cast<double>(rows_before)
            : 0.0;
        const double gf2_pct = (initial_gf2_cols > 0)
            ? 100.0 * static_cast<double>(gf2_alive_pre_compact)
                    / static_cast<double>(initial_gf2_cols)
            : 0.0;
        LOG(LOG_INFO) << "  [M12-S2] cycle " << cycle
                      << ": rows_merged=" << cycle_merges
                      << " (" << std::fixed << std::setprecision(1) << rows_merged_pct << "%)"
                      << ", gf2_cols_alive=" << gf2_alive_pre_compact
                      << " (" << std::fixed << std::setprecision(1) << gf2_pct << "% of initial)"
                      << ", floor=" << gf2_floor;

        if (gf2_alive_pre_compact < gf2_floor) {
            LOG(LOG_WARNING) << "  [M12-S2] diversity floor reached, stopping compact-merge"
                             << " (gf2_cols_alive=" << gf2_alive_pre_compact
                             << " < floor=" << gf2_floor << ").";
            ++cycles_run;
            // Mirror the convergence path: return this cycle's merge directly,
            // without compacting. csr and merge.h_row_ptr remain consistent.
            double total_sec = std::chrono::duration<double>(clock::now() - t_total_start).count();
            LOG(LOG_INFO) << "  M10: " << cycles_run << " cycles, "
                          << (total_merges / 1000) << "K total merges → "
                          << csr.n_rows << " x " << csr.n_cols
                          << " [total " << std::fixed << std::setprecision(1) << total_sec
                          << "s, floor-stopped]";
            return {std::move(csr), std::move(merge),
                    std::move(cumulative_row_map), std::move(cumulative_col_map),
                    total_merges, cycles_run};
        }

        // 5. Compact: scatter alive rows into fresh contiguous DevicePackedCSR
        // Step 1: compact alive rows into fresh CSR (old CSR + workspace still live)
        auto t_compact_start = clock::now();
        auto compact = gpuCompactPackedCSR(csr, merge);
        double compact_sec = std::chrono::duration<double>(clock::now() - t_compact_start).count();

        // Per-cycle log: merges, row/col reduction, and per-phase timing
        LOG(LOG_INFO) << "  M10: Cycle " << cycle << ": "
                      << (cycle_merges / 1000) << "K merges ("
                      << (merge.w2_merges / 1000) << "K w2 + "
                      << (merge.hw_merges / 1000) << "K hw), "
                      << (rows_before / 1000) << "K → " << (compact.csr.n_rows / 1000) << "K rows, "
                      << (cols_before / 1000) << "K → " << (compact.csr.n_cols / 1000) << "K cols"
                      << " [" << std::fixed << std::setprecision(1) << merge_sec << "s merge + "
                      << compact_sec << "s compact]";

        // 6. Compose cumulative row map: new[i] = old[compact.row_map[i]]
        {
            std::vector<uint32_t> new_cumulative(compact.csr.n_rows);
            for (uint32_t i = 0; i < compact.csr.n_rows; ++i)
                new_cumulative[i] = cumulative_row_map[compact.row_map[i]];
            cumulative_row_map = std::move(new_cumulative);
        }

        // 7. Compose cumulative column map: new[c] = old[compact.col_map[c]]
        {
            std::vector<uint32_t> new_col_map(compact.csr.n_cols);
            for (uint32_t c = 0; c < compact.csr.n_cols; ++c)
                new_col_map[c] = cumulative_col_map[compact.col_map[c]];
            cumulative_col_map = std::move(new_col_map);
        }

        // Step 2: BatchMergeResult destructor frees workspace device memory
        // (handled by RAII — merge goes out of scope after this block)
        // Step 3: move-assign compacted CSR; old CSR freed via RAII
        csr = std::move(compact.csr);  // frees old CSR; compact.csr now invalid
        // Step 4: fresh workspace allocated at top of next cycle iteration

        ++cycles_run;
        // merge goes out of scope here; workspace freed.
    }

    // Final merge pass (no compaction — result feeds directly into GF2 extraction)
    uint32_t final_budget = computeCycleBudget(csr, truncation_factor);
    auto final_merge = gpuBatchMerge(csr, mont, k_max, max_weight, final_budget);
    total_merges += final_merge.w2_merges + final_merge.hw_merges;

    double total_sec = std::chrono::duration<double>(clock::now() - t_total_start).count();
    LOG(LOG_INFO) << "  M10: " << cycles_run << " cycles, "
                  << (total_merges / 1000) << "K total merges → "
                  << csr.n_rows << " x " << csr.n_cols
                  << " [total " << std::fixed << std::setprecision(1) << total_sec << "s]";

    return {std::move(csr), std::move(final_merge),
            std::move(cumulative_row_map), std::move(cumulative_col_map),
            total_merges, cycles_run};
}

} // namespace matrix
} // namespace mpqs
