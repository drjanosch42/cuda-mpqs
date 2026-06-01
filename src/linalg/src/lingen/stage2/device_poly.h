// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "hpc_logger.h"
#include "lingen/types.h"

namespace lingen {
namespace stage2 {

/**
 * @brief RAII Container for a device-resident polynomial matrix.
 * 
 * Stores P(x) as a contiguous sequence of coefficient matrices.
 * Uses a packed bit representation (PackedBitPolyView).
 * 
 * Optimized Storage:
 * Even if the host uses a larger stride (e.g., dim x dim), this container
 * allows storing compact matrices (e.g., dim x n) to save memory on the GPU.
 */
struct DevicePackedBitPoly {
    uint64_t* d_data;
    size_t length;         // Degree + 1
    int rows;
    int cols;
    size_t mat_stride_words; // Words per coefficient matrix
    size_t row_stride_words; // Words per row

    DevicePackedBitPoly(size_t len, int r, int c) 
        : d_data(nullptr), length(len), rows(r), cols(c) 
    {
        row_stride_words = (static_cast<size_t>(c) + 63) / 64;
        mat_stride_words = row_stride_words * r;
        size_t total_bytes = length * mat_stride_words * sizeof(uint64_t);

        if (total_bytes > 0) {
            cudaError_t err = cudaMalloc(&d_data, total_bytes);
            if (err != cudaSuccess) {
                LOG(LOG_ERROR_CRITICAL) << "[DevicePoly] Allocation failed: "
                                        << cudaGetErrorString(err)
                                        << " (" << total_bytes << " bytes)";
                throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
            }
            cudaMemset(d_data, 0, total_bytes);
        }
    }

    ~DevicePackedBitPoly() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }

    // Disable copy
    DevicePackedBitPoly(const DevicePackedBitPoly&) = delete;
    DevicePackedBitPoly& operator=(const DevicePackedBitPoly&) = delete;

    // View accessor
    PackedBitPolyView view() const {
        return PackedBitPolyView(d_data, length, rows, cols);
    }

    /**
     * @brief Uploads a flat host buffer with the SAME layout (rows x cols).
     */
    void upload_from_host(const std::vector<uint64_t>& h_data, cudaStream_t stream = 0) {
        size_t req_size = length * mat_stride_words;
        if (h_data.size() != req_size) {
            LOG(LOG_ERROR_CRITICAL) << "[DevicePoly] Upload size mismatch. Expected "
                                    << req_size << ", got " << h_data.size();
            throw std::runtime_error("DevicePoly upload size mismatch");
        }
        cudaMemcpyAsync(d_data, h_data.data(), req_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    /**
     * @brief Downloads to a flat host buffer with the SAME layout (rows x cols).
     */
    void download_to_host(std::vector<uint64_t>& h_data, cudaStream_t stream = 0) const {
        size_t req_size = length * mat_stride_words;
        h_data.resize(req_size);
        cudaMemcpyAsync(h_data.data(), d_data, req_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    }

    /**
     * @brief Uploads from a host pointer with a DIFFERENT row stride.
     * Useful for converting dim x dim (host) -> dim x n (device).
     * 
     * @param h_ptr Pointer to the start of the host data (flat).
     * @param h_len Number of coefficients to copy.
     * @param h_mat_stride_words Stride between matrices on host.
     * @param h_row_stride_words Stride between rows on host.
     */
    void upload_strided(const uint64_t* h_ptr, size_t h_len, size_t h_mat_stride_words, size_t h_row_stride_words, cudaStream_t stream = 0) {
        if (h_len != length) {
             LOG(LOG_ERROR_CRITICAL) << "[DevicePoly] Upload strided length mismatch.";
             throw std::runtime_error("DevicePoly upload strided length mismatch");
        }
        
        size_t copy_width_bytes = row_stride_words * sizeof(uint64_t); // Copy valid device width
        
        for(size_t k = 0; k < length; ++k) {
            const uint64_t* src_mat = h_ptr + k * h_mat_stride_words;
            uint64_t* dst_mat = d_data + k * mat_stride_words;

            cudaMemcpy2DAsync(
                dst_mat, 
                row_stride_words * sizeof(uint64_t), // dpitch
                src_mat, 
                h_row_stride_words * sizeof(uint64_t), // spitch
                copy_width_bytes, // width
                rows, // height
                cudaMemcpyHostToDevice, 
                stream
            );
        }
    }
};

} // namespace stage2
} // namespace lingen
