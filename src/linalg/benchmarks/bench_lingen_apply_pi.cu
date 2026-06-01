// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/*
 * benchmarks/bench_lingen_apply_pi.cu
 *
 * Validation and Benchmark for PolyMatVec operations.
 * Uses the new modular architecture.
 */

#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

#include "lingen/operations/poly_mat_vec.h"
#include "bw_version.h"
#include "hpc_logger.h"

using namespace lingen;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "CUDA Error: " << cudaGetErrorString(err) << " at " __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Random data helpers
static void fill_random(std::vector<uint64_t>& data) {
    std::mt19937_64 gen(1234);
    for (auto& x : data) x = gen();
}

template<int M, int G>
void run_test(int Pi_len, int L, int iters) {
    LOG(LOG_INFO) << "[Bench] Testing M=" << M << ", G=" << G << ", Pi_len=" << Pi_len << ", L=" << L;

    // 1. Setup Data
    // Pi: Pi_len * M * (M/64) words
    // V:  L * (M/64) words (padded to M for simplicity in storage)
    // W:  L * (M/64) words
    
    size_t words_per_mat = M * (M / 64);
    size_t words_per_vec = M / 64;
    
    std::vector<uint64_t> hPi(Pi_len * words_per_mat);
    std::vector<uint64_t> hV(L * words_per_vec);
    std::vector<uint64_t> hW_ref(L * words_per_vec);
    std::vector<uint64_t> hW_gpu(L * words_per_vec);

    fill_random(hPi);
    fill_random(hV);

    // 2. GPU Allocations
    uint64_t *dPi, *dV, *dW;
    CHECK_CUDA(cudaMalloc(&dPi, hPi.size() * 8));
    CHECK_CUDA(cudaMalloc(&dV, hV.size() * 8));
    CHECK_CUDA(cudaMalloc(&dW, hW_ref.size() * 8));

    CHECK_CUDA(cudaMemcpy(dPi, hPi.data(), hPi.size() * 8, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), hV.size() * 8, cudaMemcpyHostToDevice));

    PolyMatrixView<M> viewPi(dPi, Pi_len);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // -------------------------------------------------------------------------
    // VALIDATION: Compare CPU Backend vs GPU Backend
    // -------------------------------------------------------------------------
    
    // Run CPU Reference (Right)
    PolyMatVec::apply_right<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::CPU_Reference, stream);
    CHECK_CUDA(cudaMemcpy(hW_ref.data(), dW, hW_ref.size() * 8, cudaMemcpyDeviceToHost));

    // Run GPU Kernel (Right)
    CHECK_CUDA(cudaMemset(dW, 0, hW_ref.size() * 8));
    PolyMatVec::apply_right<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::GPU_Kernel, stream);
    CHECK_CUDA(cudaMemcpy(hW_gpu.data(), dW, hW_gpu.size() * 8, cudaMemcpyDeviceToHost));

    if (std::memcmp(hW_ref.data(), hW_gpu.data(), hW_ref.size() * 8) != 0) {
        LOG(LOG_ERROR_CRITICAL) << "Validation FAILED for Apply-Right M=" << M << " G=" << G;
        exit(1);
    } else {
        LOG(LOG_INFO) << "Validation PASSED (Right)";
    }

    // Run GPU Left (Check padding/transpose logic)
    // For random data, Left result != Right result, so we just check it runs without crashing
    // and compare against CPU Left.
    PolyMatVec::apply_left<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::CPU_Reference, stream);
    CHECK_CUDA(cudaMemcpy(hW_ref.data(), dW, hW_ref.size() * 8, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(dW, 0, hW_ref.size() * 8));
    PolyMatVec::apply_left<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::GPU_Kernel, stream);
    CHECK_CUDA(cudaMemcpy(hW_gpu.data(), dW, hW_gpu.size() * 8, cudaMemcpyDeviceToHost));

    if (std::memcmp(hW_ref.data(), hW_gpu.data(), hW_ref.size() * 8) != 0) {
        LOG(LOG_ERROR_CRITICAL) << "Validation FAILED for Apply-Left M=" << M << " G=" << G;
        exit(1);
    } else {
        LOG(LOG_INFO) << "Validation PASSED (Left)";
    }

    // -------------------------------------------------------------------------
    // BENCHMARK
    // -------------------------------------------------------------------------
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for(int i=0; i<3; ++i) 
        PolyMatVec::apply_right<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::GPU_Kernel, stream);

    cudaEventRecord(start, stream);
    for(int i=0; i<iters; ++i) {
        PolyMatVec::apply_right<M, G>(viewPi, dV, L, dW, L, PolyMatVecBackend::GPU_Kernel, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / iters;
    
    // Ops = L * Pi_len * G * G (bitwise ops approx)
    double ops = double(L) * Pi_len * G * G;
    double gops = (ops / 1e9) / (avg_ms / 1000.0);
    
    LOG(LOG_INFO) << "Benchmark Right: " << avg_ms << " ms | " << gops << " Gops";

    cudaFree(dPi); cudaFree(dV); cudaFree(dW);
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    LogConfig cfg;
    cfg.enable_cout = true;
    HPCLogger::Get().Init(cfg);

    LOG(LOG_INFO) << "=== bench_lingen_apply_pi " << lingen::version_string() << " ===";

    if(argc > 1) cudaSetDevice(atoi(argv[1]));

    // Standard cases
    run_test<64, 64>(32, 4096, 20);
    run_test<128, 128>(32, 4096, 20);
    
    // Padded case (e.g. N=64, M=128)
    run_test<128, 64>(32, 4096, 20);

    return 0;
}
