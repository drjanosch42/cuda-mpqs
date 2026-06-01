// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2025-2026 Christoph Heinrichs and Fabian Januszewski
// src/matrix/gpu_char_cols.cu
// GPU-accelerated standard character column computation (M8c).
// One thread per relation; 32 Jacobi-symbol evaluations per thread.

#include "gpu_char_cols.cuh"
#include "cuda_check.h"
#include "hpc_logger.h"

#include <cstdint>
#include <vector>

namespace mpqs {
namespace matrix {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// One thread per relation.
///
/// For each of the k auxiliary primes q:
///   1. Compute sq_mod = sqrt_Q[i] % q  (uint512 → uint32_t)
///   2. Compute Q_mod_q = (sq_mod^2 - N_mod_q) mod q  (64-bit arithmetic)
///   3. Evaluate jacobi_symbol_dev(Q_mod_q, q)
///   4. Pack result as bit j of d_packed_chars[i]
///
/// The k=32 aux primes fit easily in L1/registers; q is broadcast across all threads.
__global__ __launch_bounds__(256) void char_col_kernel(
    const uint512* __restrict__ d_sqrt_Q,        // [n_rels]   — 64 bytes each
    const uint32_t* __restrict__ d_aux_primes,   // [k]
    const uint32_t* __restrict__ d_n_mod_q,      // [k]
    uint32_t* __restrict__ d_packed_chars,        // [n_rels]
    uint32_t n_rels,
    uint32_t k)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rels) return;

    const uint512 sq = d_sqrt_Q[i];
    uint32_t packed  = 0u;

    for (uint32_t j = 0; j < k; ++j) {
        const uint32_t q = d_aux_primes[j];

        // Reduce 512-bit sqrt_Q mod q — __host__ __device__ mod_uint32 path.
        const uint32_t sq_mod = sq % q;

        // Q_i mod q = (sq_mod^2 - N mod q) mod q.
        // 64-bit multiply avoids overflow; add q before final mod to prevent
        // negative underflow when sq_mod^2 % q < N mod q.
        const uint64_t sq2     = static_cast<uint64_t>(sq_mod) * static_cast<uint64_t>(sq_mod);
        const uint32_t Q_mod_q = static_cast<uint32_t>((sq2 % q + q - d_n_mod_q[j]) % q);

        const int ls = jacobi_symbol_dev(Q_mod_q, q);
        if (ls == -1) packed |= (1u << j);
    }

    d_packed_chars[i] = packed;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

CharacterColumns gpuComputeCharacterColumns(
    const structures::HostRelationBatch& batch,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q)
{
    LOG_SET_MODULE("Matrix");

    const uint32_t n_rels = static_cast<uint32_t>(batch.num_relations);
    const uint32_t k      = static_cast<uint32_t>(aux_primes.size());

    // Detect unified-memory platform (Jetson Orin or similar).
    // SM 8.7 = Orin; small VRAM heuristic covers other integrated GPUs.
    // cudaMallocManaged is used on these platforms; discrete GPUs use cudaMalloc.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const bool use_managed = (prop.major == 8 && prop.minor == 7) ||
                             (prop.unifiedAddressing &&
                              prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);

    // ---------- Device allocation ----------

    uint512*  d_sqrt_Q       = nullptr;
    uint32_t* d_aux_primes_d = nullptr;
    uint32_t* d_n_mod_q_d    = nullptr;
    uint32_t* d_packed_chars = nullptr;

    auto dev_alloc = [&](void** ptr, size_t sz) {
        if (use_managed) {
            CUDA_CHECK(cudaMallocManaged(ptr, sz));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, sz));
        }
    };

    dev_alloc(reinterpret_cast<void**>(&d_sqrt_Q),       n_rels * sizeof(uint512));
    dev_alloc(reinterpret_cast<void**>(&d_aux_primes_d), k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_n_mod_q_d),    k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_packed_chars), n_rels * sizeof(uint32_t));

    // ---------- Upload ----------

    CUDA_CHECK(cudaMemcpy(d_sqrt_Q, batch.sqrt_Q.data(),
                          n_rels * sizeof(uint512), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_aux_primes_d, aux_primes.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_mod_q_d, n_mod_q.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_packed_chars, 0, n_rels * sizeof(uint32_t)));

    // ---------- Kernel launch ----------

    constexpr uint32_t kBlock = 256;
    const uint32_t grid = (n_rels + kBlock - 1) / kBlock;
    char_col_kernel<<<grid, kBlock>>>(
        d_sqrt_Q, d_aux_primes_d, d_n_mod_q_d, d_packed_chars, n_rels, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Download ----------

    std::vector<uint32_t> packed(n_rels);
    CUDA_CHECK(cudaMemcpy(packed.data(), d_packed_chars,
                          n_rels * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // ---------- Free device buffers ----------

    CUDA_CHECK(cudaFree(d_sqrt_Q));
    CUDA_CHECK(cudaFree(d_aux_primes_d));
    CUDA_CHECK(cudaFree(d_n_mod_q_d));
    CUDA_CHECK(cudaFree(d_packed_chars));

    // ---------- Unpack bitmasks into column-major CharacterColumns ----------
    // columns[j][i] in {0,1}: bit j of packed[i] encodes char col j for relation i.
    // Layout matches CharacterColumnComputer::compute() exactly.

    CharacterColumns result;
    result.k          = k;
    result.aux_primes = aux_primes;
    result.columns.resize(k);
    for (uint32_t j = 0; j < k; ++j) result.columns[j].resize(n_rels, 0);

    for (uint32_t i = 0; i < n_rels; ++i) {
        const uint32_t p = packed[i];
        for (uint32_t j = 0; j < k; ++j) {
            if ((p >> j) & 1u) result.columns[j][i] = 1;
        }
    }

    LOG(LOG_INFO) << "GPU char cols: " << n_rels << " relations × " << k
                  << " aux primes ("
                  << (n_rels * sizeof(uint512) >> 20) << " MB sqrt_Q).";

    return result;
}

// ---------------------------------------------------------------------------
// Device-pointer overload (M9d)
// ---------------------------------------------------------------------------

/// Computes GPU character columns reading sqrt_Q directly from a device pointer.
/// Avoids the ~20 MB D→H→D sqrt_Q round-trip present in the HostRelationBatch overload.
/// The persistent RelationBatch remains unmodified; only the three auxiliary device
/// buffers allocated here are freed on return.
CharacterColumns gpuComputeCharacterColumns_device(
    const uint512* d_sqrt_Q,
    uint32_t n_rels,
    const std::vector<uint32_t>& aux_primes,
    const std::vector<uint32_t>& n_mod_q)
{
    LOG_SET_MODULE("Matrix");

    const uint32_t k = static_cast<uint32_t>(aux_primes.size());

    // Detect unified-memory platform (Jetson Orin or similar).
    // SM 8.7 = Orin; small VRAM heuristic covers other integrated GPUs.
    // Never use concurrentManagedAccess — Blackwell SM 12.0 also reports 0.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const bool use_managed = (prop.major == 8 && prop.minor == 7) ||
                             (prop.unifiedAddressing &&
                              prop.totalGlobalMem < 12ULL * 1024 * 1024 * 1024);

    // Allocate only aux_primes, n_mod_q, and packed output.
    // NO d_sqrt_Q allocation — caller provides device pointer directly.
    uint32_t* d_aux_primes_d = nullptr;
    uint32_t* d_n_mod_q_d    = nullptr;
    uint32_t* d_packed_chars = nullptr;

    auto dev_alloc = [&](void** ptr, size_t sz) {
        if (use_managed) {
            CUDA_CHECK(cudaMallocManaged(ptr, sz));
        } else {
            CUDA_CHECK(cudaMalloc(ptr, sz));
        }
    };

    dev_alloc(reinterpret_cast<void**>(&d_aux_primes_d), k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_n_mod_q_d),    k * sizeof(uint32_t));
    dev_alloc(reinterpret_cast<void**>(&d_packed_chars), n_rels * sizeof(uint32_t));

    // Upload aux_primes and n_mod_q (tiny: 128 + 128 bytes for k=32)
    CUDA_CHECK(cudaMemcpy(d_aux_primes_d, aux_primes.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_mod_q_d, n_mod_q.data(),
                          k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_packed_chars, 0, n_rels * sizeof(uint32_t)));

    // Launch the same char_col_kernel — d_sqrt_Q comes from the caller (device pointer)
    constexpr uint32_t kBlock = 256;
    const uint32_t grid = (n_rels + kBlock - 1) / kBlock;
    char_col_kernel<<<grid, kBlock>>>(
        d_sqrt_Q, d_aux_primes_d, d_n_mod_q_d, d_packed_chars, n_rels, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download packed results
    std::vector<uint32_t> packed(n_rels);
    CUDA_CHECK(cudaMemcpy(packed.data(), d_packed_chars,
                          n_rels * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Free only the buffers allocated here — NOT d_sqrt_Q (owned by persistent batch)
    CUDA_CHECK(cudaFree(d_aux_primes_d));
    CUDA_CHECK(cudaFree(d_n_mod_q_d));
    CUDA_CHECK(cudaFree(d_packed_chars));

    // Unpack bitmasks into column-major CharacterColumns (identical to existing overload)
    CharacterColumns result;
    result.k          = k;
    result.aux_primes = aux_primes;
    result.columns.resize(k);
    for (uint32_t j = 0; j < k; ++j) result.columns[j].resize(n_rels, 0);

    for (uint32_t i = 0; i < n_rels; ++i) {
        const uint32_t p = packed[i];
        for (uint32_t j = 0; j < k; ++j) {
            if ((p >> j) & 1u) result.columns[j][i] = 1;
        }
    }

    LOG(LOG_INFO) << "GPU char cols (device-resident sqrt_Q, 0 MB upload): "
                  << n_rels << " relations × " << k << " aux primes.";

    return result;
}

} // namespace matrix
} // namespace mpqs
