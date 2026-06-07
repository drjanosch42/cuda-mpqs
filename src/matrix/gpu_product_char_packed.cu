// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_product_char_packed.cu
//
// GPU product character columns from merged sqrt_Q values (M9f).
// Structurally identical to char_col_kernel — one thread per merged row,
// 32 Jacobi symbol evaluations per thread, packed uint32 bitmask output.
// Reuses jacobi_symbol_dev() from gpu_char_cols.cuh.

#include "gpu_product_char_packed.cuh"
#include "gpu_char_cols.cuh"       // jacobi_symbol_dev() — __host__ __device__
#include "device_packed_csr.cuh"   // isJetsonDevice()
#include "cuda_check.h"
#include "hpc_logger.h"

#include <vector>

namespace mpqs {
namespace matrix {

// ============================================================================
// Kernel: product character columns from merged sqrt_Q
// ============================================================================

/// One thread per merged row. Evaluates 32 Jacobi symbols on sqrt_Q^2 - N.
/// Structurally identical to char_col_kernel in gpu_char_cols.cu.
__global__ __launch_bounds__(256) void product_char_col_kernel(
    const uint512*  __restrict__ d_sqrt_Q,      // [n_rows] — merged sqrt_Q values
    const uint32_t* __restrict__ d_aux_primes,  // [k]
    const uint32_t* __restrict__ d_n_mod_q,     // [k]
    uint32_t*       __restrict__ d_packed_chars, // [n_rows]
    uint32_t n_rows,
    uint32_t k)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    const uint512 sq = d_sqrt_Q[i];
    uint32_t packed = 0u;

    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t q = d_aux_primes[j];

        // Reduce 512-bit sqrt_Q mod q
        const uint32_t sq_mod = sq % q;

        // Q_i mod q = (sq_mod^2 - N mod q) mod q
        // 64-bit multiply avoids overflow; add q before final mod for underflow safety
        const uint64_t sq2     = static_cast<uint64_t>(sq_mod) * static_cast<uint64_t>(sq_mod);
        const uint32_t Q_mod_q = static_cast<uint32_t>((sq2 % q + q - d_n_mod_q[j]) % q);

        const int ls = jacobi_symbol_dev(Q_mod_q, q);
        if (ls == -1) packed |= (1u << j);
    }

    d_packed_chars[i] = packed;
}

// ============================================================================
// Host driver
// ============================================================================

CharacterColumns gpuProductCharCols_packed(
    const uint512* d_merged_sqrt_Q,
    uint32_t n_rows,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q)
{
    // Guard: zero-row matrix has no product char cols to compute
    if (n_rows == 0) {
        CharacterColumns result;
        result.k = static_cast<uint32_t>(aux_primes.size());
        return result;
    }

    LOG_SET_MODULE("Matrix");
    LOG(LOG_INFO) << "M9f: Product char cols (Jacobi-only) for " << n_rows
                  << " merged rows, k=" << aux_primes.size() << ".";

    const uint32_t k = static_cast<uint32_t>(aux_primes.size());

    // Detect Jetson for managed memory allocation
    bool jetson = isJetsonDevice();

    auto dev_alloc = [&](void** ptr, size_t sz) {
        if (jetson) { CUDA_CHECK(cudaMallocManaged(ptr, sz)); }
        else        { CUDA_CHECK(cudaMalloc(ptr, sz)); }
    };

    // Allocate device buffers for aux_primes, n_mod_q, and packed output.
    // d_merged_sqrt_Q is already on device — passed directly, no upload needed.
    uint32_t* d_aux_primes_d = nullptr;
    uint32_t* d_n_mod_q_d    = nullptr;
    uint32_t* d_packed_chars = nullptr;

    dev_alloc(reinterpret_cast<void**>(&d_aux_primes_d), k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_n_mod_q_d),    k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_packed_chars), n_rows * sizeof(uint32_t));

    // Upload aux_primes and n_mod_q (tiny: 128 + 128 bytes for k=32)
    CUDA_CHECK(cudaMemcpy(d_aux_primes_d, aux_primes.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_mod_q_d, n_mod_q.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_packed_chars, 0, n_rows * sizeof(uint32_t)));

    // Launch kernel
    constexpr uint32_t kBlock = 256;
    const uint32_t grid = (n_rows + kBlock - 1) / kBlock;
    product_char_col_kernel<<<grid, kBlock>>>(
        d_merged_sqrt_Q, d_aux_primes_d, d_n_mod_q_d, d_packed_chars, n_rows, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download packed results
    std::vector<uint32_t> packed(n_rows);
    CUDA_CHECK(cudaMemcpy(packed.data(), d_packed_chars,
                          n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Free device buffers — NOT d_merged_sqrt_Q (owned by workspace)
    CUDA_CHECK(cudaFree(d_aux_primes_d));
    CUDA_CHECK(cudaFree(d_n_mod_q_d));
    CUDA_CHECK(cudaFree(d_packed_chars));

    // Unpack bitmasks into column-major CharacterColumns
    // Layout matches CharacterColumnComputer::compute() exactly.
    CharacterColumns result;
    result.k          = k;
    // CharacterColumns::aux_primes is 64-bit (Stage 2); widen the uint32 input.
    result.aux_primes.assign(aux_primes.begin(), aux_primes.end());
    result.columns.resize(k);
    for (uint32_t j = 0; j < k; ++j) result.columns[j].resize(n_rows, 0);

    for (uint32_t i = 0; i < n_rows; i++) {
        const uint32_t p = packed[i];
        for (uint32_t j = 0; j < k; ++j) {
            if ((p >> j) & 1u) result.columns[j][i] = 1;
        }
    }

    LOG(LOG_INFO) << "M9f: Product char cols complete.";
    return result;
}

} // namespace matrix
} // namespace mpqs
