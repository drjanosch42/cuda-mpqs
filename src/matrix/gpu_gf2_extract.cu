// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_gf2_extract.cu
//
// GF(2) extraction from packed merged CSR (M9f).
// Flatten alive rows into contiguous device buffer, then two-pass:
//   Pass 1: count odd-exponent entries per row
//   Prefix sum → row offsets
//   Pass 2: write column indices for odd-exponent entries
// Produces binary CSR identical to binary XOR merge output (audit v4 §10.5).

#include "gpu_gf2_extract.cuh"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <numeric>

namespace mpqs {
namespace matrix {

// ============================================================================
// Kernel 1: Flatten packed rows from original CSR or workspace into contiguous buffer
// ============================================================================

/// One thread per alive row. Copies entries from the source location
/// (original CSR or workspace) to a flat contiguous buffer.
__global__ __launch_bounds__(256) void flatten_packed_rows_kernel(
    // Original CSR
    const uint32_t*    __restrict__ orig_row_offsets,
    const PackedEntry* __restrict__ orig_entries,
    // Workspace
    const uint32_t*    __restrict__ ws_row_starts,
    const uint32_t*    __restrict__ ws_row_lengths,
    const PackedEntry* __restrict__ ws_entries,
    // Row indirection (alive rows only, compacted)
    const uint32_t*    __restrict__ row_ptr_alive,  // [n_alive] — row_ptr values
    // Output flat CSR
    const uint32_t*    __restrict__ flat_row_offsets,  // pre-computed prefix sum
    PackedEntry*       __restrict__ flat_entries,
    uint32_t n_alive)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_alive) return;

    uint32_t ptr_val = row_ptr_alive[idx];
    const PackedEntry* src;
    uint32_t len;

    if (ptr_val & ROW_WS_BIT) {
        uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
        src = ws_entries + ws_row_starts[ws_idx];
        len = ws_row_lengths[ws_idx];
    } else {
        src = orig_entries + orig_row_offsets[ptr_val];
        len = orig_row_offsets[ptr_val + 1] - orig_row_offsets[ptr_val];
    }

    uint32_t out_start = flat_row_offsets[idx];
    for (uint32_t i = 0; i < len; i++)
        flat_entries[out_start + i] = src[i];
}

// ============================================================================
// Kernel 2: Count odd-exponent entries per row
// ============================================================================

__global__ __launch_bounds__(256) void count_gf2_entries_kernel(
    const PackedEntry* __restrict__ entries,
    const uint32_t*    __restrict__ row_offsets,
    uint32_t n_rows,
    uint32_t* __restrict__ gf2_counts)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    uint32_t begin = row_offsets[row], end = row_offsets[row + 1];
    uint32_t count = 0;
    for (uint32_t i = begin; i < end; i++)
        if (entries[i] & 1)  // exponent is odd
            count++;
    gf2_counts[row] = count;
}

// ============================================================================
// Kernel 3: Write GF(2) column indices
// ============================================================================

__global__ __launch_bounds__(256) void extract_gf2_entries_kernel(
    const PackedEntry* __restrict__ entries,
    const uint32_t*    __restrict__ row_offsets,
    uint32_t n_rows,
    const uint32_t*    __restrict__ gf2_row_offsets,
    uint32_t* __restrict__ gf2_col_indices)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    uint32_t begin = row_offsets[row], end = row_offsets[row + 1];
    uint32_t out = gf2_row_offsets[row];
    for (uint32_t i = begin; i < end; i++)
        if (entries[i] & 1)
            gf2_col_indices[out++] = entries[i] >> 8;  // column index
}

// ============================================================================
// Host driver
// ============================================================================

GF2ExtractionResult gpuExtractGF2(
    const DevicePackedCSR& csr,
    const BatchMergeResult& merge_result)
{
    LOG_SET_MODULE("Matrix");
    LOG(LOG_INFO) << "M9f: GF(2) extraction from packed merged CSR...";

    const auto& h_row_ptr   = merge_result.h_row_ptr;
    const auto& h_col_weight = merge_result.h_col_weight;
    const auto& workspace   = merge_result.workspace;
    const uint32_t n_total  = static_cast<uint32_t>(h_row_ptr.size());

    // 1. Scan h_row_ptr → build row_map (alive rows) and row lengths
    std::vector<uint32_t> alive_ptr_vals;  // row_ptr values for alive rows
    std::vector<uint32_t> row_map;         // gf2_row → original logical row index
    std::vector<uint32_t> row_lengths;     // entry count per alive row

    alive_ptr_vals.reserve(n_total);
    row_map.reserve(n_total);
    row_lengths.reserve(n_total);

    // Download original row_offsets to host for length computation
    std::vector<uint32_t> h_orig_row_offsets(csr.n_rows + 1);
    CUDA_CHECK(cudaMemcpy(h_orig_row_offsets.data(), csr.d_row_offsets,
                          (csr.n_rows + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Download workspace row_starts and row_lengths to host
    uint64_t dual_counter_val = 0;
    CUDA_CHECK(cudaMemcpy(&dual_counter_val, workspace.d_dual_counter,
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t ws_row_count = static_cast<uint32_t>(dual_counter_val >> 32);

    std::vector<uint32_t> h_ws_row_starts(ws_row_count);
    std::vector<uint32_t> h_ws_row_lengths(ws_row_count);
    if (ws_row_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_ws_row_starts.data(), workspace.d_ws_row_starts,
                              ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ws_row_lengths.data(), workspace.d_ws_row_lengths,
                              ws_row_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    for (uint32_t r = 0; r < n_total; r++) {
        uint32_t ptr_val = h_row_ptr[r];
        if (ptr_val == ROW_DEAD) continue;

        alive_ptr_vals.push_back(ptr_val);
        row_map.push_back(r);

        if (ptr_val & ROW_WS_BIT) {
            uint32_t ws_idx = ptr_val & 0x7FFFFFFFu;
            row_lengths.push_back(h_ws_row_lengths[ws_idx]);
        } else {
            row_lengths.push_back(h_orig_row_offsets[ptr_val + 1] - h_orig_row_offsets[ptr_val]);
        }
    }

    const uint32_t n_alive = static_cast<uint32_t>(alive_ptr_vals.size());
    LOG(LOG_INFO) << "M9f: " << n_alive << " alive rows (from " << n_total << " total).";

    if (n_alive == 0) {
        GF2ExtractionResult empty;
        empty.gf2_csr.n_rows = 0;
        empty.gf2_csr.n_cols = 0;
        return empty;
    }

    // 2. Compute flat_row_offsets via host prefix sum
    std::vector<uint32_t> h_flat_row_offsets(n_alive + 1, 0);
    for (uint32_t i = 0; i < n_alive; i++)
        h_flat_row_offsets[i + 1] = h_flat_row_offsets[i] + row_lengths[i];
    uint32_t total_flat_nnz = h_flat_row_offsets[n_alive];

    LOG(LOG_DEBUG_1) << "M9f: Flattening " << total_flat_nnz << " packed entries.";

    // 3. Upload alive_ptr_vals and flat_row_offsets to device; allocate flat entries
    bool jetson = isJetsonDevice();
    auto dev_alloc = [&](void** ptr, size_t sz) {
        if (jetson) { CUDA_CHECK(cudaMallocManaged(ptr, sz)); }
        else        { CUDA_CHECK(cudaMalloc(ptr, sz)); }
    };

    uint32_t*    d_alive_ptr    = nullptr;
    uint32_t*    d_flat_offsets = nullptr;
    PackedEntry* d_flat_entries = nullptr;

    dev_alloc(reinterpret_cast<void**>(&d_alive_ptr),    n_alive * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_flat_offsets),  (n_alive + 1) * sizeof(uint32_t));
    if (total_flat_nnz > 0) {
        dev_alloc(reinterpret_cast<void**>(&d_flat_entries), total_flat_nnz * sizeof(PackedEntry));
    }

    CUDA_CHECK(cudaMemcpy(d_alive_ptr, alive_ptr_vals.data(),
                          n_alive * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_offsets, h_flat_row_offsets.data(),
                          (n_alive + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 4. Launch flatten kernel
    if (total_flat_nnz > 0) {
        constexpr uint32_t kBlock = 256;
        uint32_t grid = (n_alive + kBlock - 1) / kBlock;
        flatten_packed_rows_kernel<<<grid, kBlock>>>(
            csr.d_row_offsets, csr.d_entries,
            workspace.d_ws_row_starts, workspace.d_ws_row_lengths, workspace.d_ws_entries,
            d_alive_ptr,
            d_flat_offsets, d_flat_entries,
            n_alive);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Free alive_ptr — no longer needed
    CUDA_CHECK(cudaFree(d_alive_ptr));
    d_alive_ptr = nullptr;

    // 5. Pass 1: Count GF(2) entries per row
    uint32_t* d_gf2_counts = nullptr;
    dev_alloc(reinterpret_cast<void**>(&d_gf2_counts), n_alive * sizeof(uint32_t));
    CUDA_CHECK(cudaMemset(d_gf2_counts, 0, n_alive * sizeof(uint32_t)));

    if (total_flat_nnz > 0) {
        constexpr uint32_t kBlock = 256;
        uint32_t grid = (n_alive + kBlock - 1) / kBlock;
        count_gf2_entries_kernel<<<grid, kBlock>>>(
            d_flat_entries, d_flat_offsets, n_alive, d_gf2_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 6. Thrust exclusive_scan → GF(2) row offsets
    uint32_t* d_gf2_row_offsets = nullptr;
    dev_alloc(reinterpret_cast<void**>(&d_gf2_row_offsets), (n_alive + 1) * sizeof(uint32_t));

    {
        thrust::device_ptr<uint32_t> counts_ptr(d_gf2_counts);
        thrust::device_ptr<uint32_t> offsets_ptr(d_gf2_row_offsets);
        thrust::exclusive_scan(counts_ptr, counts_ptr + n_alive, offsets_ptr, 0u);
        // Write total NNZ at position n_alive
        // = offsets[n_alive-1] + counts[n_alive-1]
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read total GF(2) NNZ: offsets[n_alive-1] + counts[n_alive-1]
    uint32_t last_offset = 0, last_count = 0;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_gf2_row_offsets + (n_alive - 1),
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_count, d_gf2_counts + (n_alive - 1),
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t gf2_nnz = last_offset + last_count;

    // Write sentinel at position n_alive
    CUDA_CHECK(cudaMemcpy(d_gf2_row_offsets + n_alive, &gf2_nnz,
                          sizeof(uint32_t), cudaMemcpyHostToDevice));

    LOG(LOG_INFO) << "M9f: GF(2) matrix " << n_alive << " rows, "
                  << gf2_nnz << " NNZ.";

    // Free counts — no longer needed
    CUDA_CHECK(cudaFree(d_gf2_counts));
    d_gf2_counts = nullptr;

    // 7. Pass 2: Write GF(2) column indices
    uint32_t* d_gf2_col_indices = nullptr;
    if (gf2_nnz > 0) {
        dev_alloc(reinterpret_cast<void**>(&d_gf2_col_indices), gf2_nnz * sizeof(uint32_t));
    }

    if (gf2_nnz > 0 && total_flat_nnz > 0) {
        constexpr uint32_t kBlock = 256;
        uint32_t grid = (n_alive + kBlock - 1) / kBlock;
        extract_gf2_entries_kernel<<<grid, kBlock>>>(
            d_flat_entries, d_flat_offsets, n_alive,
            d_gf2_row_offsets, d_gf2_col_indices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Free flat entries — no longer needed
    if (d_flat_entries) {
        CUDA_CHECK(cudaFree(d_flat_entries));
        d_flat_entries = nullptr;
    }
    CUDA_CHECK(cudaFree(d_flat_offsets));
    d_flat_offsets = nullptr;

    // 8. Download GF(2) CSR to host
    GF2ExtractionResult result;
    result.row_map   = std::move(row_map);
    result.gf2_nnz   = gf2_nnz;

    result.gf2_csr.n_rows = n_alive;
    result.gf2_csr.row_offsets.resize(n_alive + 1);
    CUDA_CHECK(cudaMemcpy(result.gf2_csr.row_offsets.data(), d_gf2_row_offsets,
                          (n_alive + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result.gf2_csr.col_indices.resize(gf2_nnz);
    if (gf2_nnz > 0) {
        CUDA_CHECK(cudaMemcpy(result.gf2_csr.col_indices.data(), d_gf2_col_indices,
                              gf2_nnz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_gf2_row_offsets));
    if (d_gf2_col_indices) CUDA_CHECK(cudaFree(d_gf2_col_indices));

    // 9. Column compaction: remove zero-weight GF(2) columns.
    //    Build old_to_new_col[] from h_col_weight to determine which columns
    //    have nonzero GF(2) weight. We count actual GF(2) occurrences from the
    //    extracted CSR (h_col_weight from merge_result tracks packed weight, not GF(2)).
    {
        // Count GF(2) column weights from the extracted CSR
        uint32_t max_col = 0;
        for (uint32_t c : result.gf2_csr.col_indices) {
            if (c > max_col) max_col = c;
        }
        uint32_t n_cols_orig = max_col + 1;

        std::vector<uint32_t> gf2_col_weight(n_cols_orig, 0);
        for (uint32_t c : result.gf2_csr.col_indices)
            gf2_col_weight[c]++;

        // Build compaction map
        std::vector<uint32_t> old_to_new(n_cols_orig, UINT32_MAX);
        uint32_t new_col = 0;
        for (uint32_t c = 0; c < n_cols_orig; c++) {
            if (gf2_col_weight[c] > 0)
                old_to_new[c] = new_col++;
        }

        uint32_t cols_removed = n_cols_orig - new_col;
        if (cols_removed > 0) {
            // Remap column indices
            for (auto& c : result.gf2_csr.col_indices)
                c = old_to_new[c];
            LOG(LOG_INFO) << "M9f: Column compaction: " << n_cols_orig << " → "
                          << new_col << " (" << cols_removed << " empty columns removed).";
        }

        result.gf2_csr.n_cols = new_col;
    }

    LOG(LOG_INFO) << "M9f: GF(2) extraction complete: " << result.gf2_csr.n_rows
                  << " x " << result.gf2_csr.n_cols << ", NNZ=" << result.gf2_nnz << ".";

    return result;
}

} // namespace matrix
} // namespace mpqs
