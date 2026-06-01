// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/device_packed_csr.cu
//
// RAII implementation of DevicePackedCSR + isJetsonDevice() utility.

#include "device_packed_csr.cuh"
#include "cuda_check.h"

namespace mpqs {
namespace matrix {

// ============================================================================
// DevicePackedCSR RAII implementation
// ============================================================================

void DevicePackedCSR::alloc(uint32_t rows, uint32_t cols, uint32_t nnz_cap, bool jetson) {
    use_managed = jetson;
    n_rows = rows;
    n_cols = cols;
    nnz    = nnz_cap;

    auto device_malloc = [&](void** ptr, size_t bytes) {
        if (use_managed) {
            CUDA_CHECK(cudaMallocManaged(ptr, bytes));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, bytes));
        }
    };

    // Allocate at least 1 element to avoid zero-size alloc edge cases.
    const uint32_t safe_rows = (rows > 0) ? rows : 1u;
    const uint32_t safe_nnz  = (nnz_cap > 0) ? nnz_cap : 1u;

    device_malloc(reinterpret_cast<void**>(&d_row_offsets), (safe_rows + 1) * sizeof(uint32_t));
    device_malloc(reinterpret_cast<void**>(&d_entries),      safe_nnz * sizeof(PackedEntry));
    device_malloc(reinterpret_cast<void**>(&d_sqrt_Q),       safe_rows * sizeof(uint512));
    device_malloc(reinterpret_cast<void**>(&d_signs),        safe_rows * sizeof(uint8_t));
    device_malloc(reinterpret_cast<void**>(&d_val_2_exps),   safe_rows * sizeof(int32_t));
}

DevicePackedCSR::~DevicePackedCSR() {
    // cudaFree handles both cudaMalloc and cudaMallocManaged pointers.
    if (d_row_offsets) cudaFree(d_row_offsets);
    if (d_entries)     cudaFree(d_entries);
    if (d_sqrt_Q)      cudaFree(d_sqrt_Q);
    if (d_signs)       cudaFree(d_signs);
    if (d_val_2_exps)  cudaFree(d_val_2_exps);
}

DevicePackedCSR::DevicePackedCSR(DevicePackedCSR&& other) noexcept
    : d_row_offsets(other.d_row_offsets),
      d_entries    (other.d_entries),
      d_sqrt_Q     (other.d_sqrt_Q),
      d_signs      (other.d_signs),
      d_val_2_exps (other.d_val_2_exps),
      n_rows       (other.n_rows),
      n_cols       (other.n_cols),
      nnz          (other.nnz),
      use_managed  (other.use_managed)
{
    other.d_row_offsets = nullptr;
    other.d_entries     = nullptr;
    other.d_sqrt_Q      = nullptr;
    other.d_signs       = nullptr;
    other.d_val_2_exps  = nullptr;
    other.n_rows        = 0;
    other.n_cols        = 0;
    other.nnz           = 0;
}

DevicePackedCSR& DevicePackedCSR::operator=(DevicePackedCSR&& other) noexcept {
    if (this == &other) return *this;
    // Free own buffers first.
    if (d_row_offsets) cudaFree(d_row_offsets);
    if (d_entries)     cudaFree(d_entries);
    if (d_sqrt_Q)      cudaFree(d_sqrt_Q);
    if (d_signs)       cudaFree(d_signs);
    if (d_val_2_exps)  cudaFree(d_val_2_exps);
    // Transfer from other.
    d_row_offsets = other.d_row_offsets;
    d_entries     = other.d_entries;
    d_sqrt_Q      = other.d_sqrt_Q;
    d_signs       = other.d_signs;
    d_val_2_exps  = other.d_val_2_exps;
    n_rows        = other.n_rows;
    n_cols        = other.n_cols;
    nnz           = other.nnz;
    use_managed   = other.use_managed;
    // Null out other.
    other.d_row_offsets = nullptr;
    other.d_entries     = nullptr;
    other.d_sqrt_Q      = nullptr;
    other.d_signs       = nullptr;
    other.d_val_2_exps  = nullptr;
    other.n_rows        = 0;
    other.n_cols        = 0;
    other.nnz           = 0;
    return *this;
}

DevicePackedView DevicePackedCSR::view() const {
    return DevicePackedView{
        d_row_offsets, d_entries,
        d_sqrt_Q, d_signs, d_val_2_exps,
        n_rows, n_cols, nnz
    };
}

void DevicePackedCSR::downloadToHost(HostMatrixCSR& out, cudaStream_t stream) const {
    out.row_offsets.resize(n_rows + 1);
    out.col_indices.resize(nnz);

    CUDA_CHECK(cudaMemcpyAsync(out.row_offsets.data(), d_row_offsets,
                                (n_rows + 1) * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(out.col_indices.data(), d_entries,
                                nnz * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Extract column indices from packed entries: col = entry >> 8.
    for (uint32_t i = 0; i < nnz; ++i) {
        out.col_indices[i] >>= 8;
    }

    out.n_rows = n_rows;
    out.n_cols = n_cols;
}

// ============================================================================
// Jetson detection
// ============================================================================

bool isJetsonDevice() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        cached = 0;
        return false;
    }

    // SM 8.7 (Orin) or unified addressing with < 12 GB VRAM (integrated GPU).
    // NEVER uses concurrentManagedAccess — Blackwell SM 12.0 also reports 0.
    bool jetson = (prop.major == 8 && prop.minor == 7) ||
                  (prop.unifiedAddressing &&
                   prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);
    cached = jetson ? 1 : 0;
    return jetson;
}

} // namespace matrix
} // namespace mpqs
