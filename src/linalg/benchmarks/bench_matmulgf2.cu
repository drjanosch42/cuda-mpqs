// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

/*
 * benchmarks/bench_matmulgf2.cu
 * Benchmark for dense binary matrix multiplication (MatMul) over GF(2).
 */

#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>

#include "lingen/operations/matmul_gf2.h"
#include "bw_version.h"
#include "hpc_logger.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "CUDA Error: " << cudaGetErrorString(err) << " at " __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void thrash_cache_kernel(int* garbage_ptr, size_t size_ints) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size_ints; i += stride) {
        garbage_ptr[i] = (int)i; 
    }
}

void run_cache_thrash(int* d_garbage, size_t size_bytes, cudaStream_t stream) {
    size_t size_ints = size_bytes / sizeof(int);
    int threads = 1024;
    int blocks = std::min((size_t)(size_ints + threads - 1) / threads, (size_t)8192);
    thrash_cache_kernel<<<blocks, threads, 0, stream>>>(d_garbage, size_ints);
}

template <int N>
void run_benchmark(size_t num_matrices, int iterations) {
    size_t row_bytes = N / 8;
    size_t mat_size_bytes = row_bytes * N;
    size_t total_data_size = num_matrices * mat_size_bytes;

    LOG(LOG_INFO) << "[MatMulGF2] ------------------------------------------------" << std::endl;
    LOG(LOG_INFO) << "[MatMulGF2] Benchmarking N = " << N << " (" << N << "x" << N << " bits)" << std::endl;
    LOG(LOG_INFO) << "[MatMulGF2] Batch Size: " << num_matrices << " matrices" << std::endl;
    LOG(LOG_INFO) << "[MatMulGF2] Data Volume: " << (total_data_size * 3.0 / 1024.0 / 1024.0) << " MB (A+B+C)" << std::endl;

    std::vector<uint8_t> h_A(total_data_size);
    std::vector<uint8_t> h_B(total_data_size);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (size_t i = 0; i < total_data_size; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    void *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, total_data_size));
    CHECK_CUDA(cudaMalloc(&d_B, total_data_size));
    CHECK_CUDA(cudaMalloc(&d_C, total_data_size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), total_data_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), total_data_size, cudaMemcpyHostToDevice));

    size_t thrash_size = 128 * 1024 * 1024; 
    int* d_garbage;
    CHECK_CUDA(cudaMalloc(&d_garbage, thrash_size));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    lingen::matmul_gf2_gpu(N, d_A, d_B, d_C, num_matrices, stream);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Hot Cache
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        lingen::matmul_gf2_gpu(N, d_A, d_B, d_C, num_matrices, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_hot = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_hot, start, stop));
    double avg_time_hot = (ms_hot / 1000.0) / iterations;

    // Cold Cache Overhead
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        run_cache_thrash(d_garbage, thrash_size, stream);
        cudaStreamSynchronize(stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_thrash_only = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_thrash_only, start, stop));

    // Cold Cache Combined
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; ++i) {
        run_cache_thrash(d_garbage, thrash_size, stream);
        cudaStreamSynchronize(stream); 
        lingen::matmul_gf2_gpu(N, d_A, d_B, d_C, num_matrices, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_cold_total = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cold_total, start, stop));
    
    float cold_ms_adjusted = std::max(0.0f, ms_cold_total - ms_thrash_only);
    double avg_time_cold = (cold_ms_adjusted / 1000.0) / iterations;
    if (avg_time_cold <= 0) avg_time_cold = avg_time_hot;

    // Metrics
    double ops_per_mat = (double)N * N * N; 
    double total_ops = ops_per_mat * num_matrices;

    double gops_hot = (total_ops / avg_time_hot) / 1e9;
    double gops_cold = (total_ops / avg_time_cold) / 1e9;
    
    double mat_sec_hot = num_matrices / avg_time_hot;
    double mat_sec_cold = num_matrices / avg_time_cold;

    LOG(LOG_INFO) << "[MatMulGF2] Hot Cache:  " << gops_hot << " Gops | " << mat_sec_hot << " Mats/sec | " << (avg_time_hot*1000) << " ms/batch" << std::endl;
    LOG(LOG_INFO) << "[MatMulGF2] Cold Cache: " << gops_cold << " Gops | " << mat_sec_cold << " Mats/sec | " << (avg_time_cold*1000) << " ms/batch" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_garbage);
    cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    int device = 0;
    if (argc > 1) device = atoi(argv[1]);

    LogConfig cfg;
    cfg.enable_cout = true;
    cfg.min_severity_cout = LOG_INFO;
    HPCLogger::Get().Init(cfg);

    LOG(LOG_INFO) << "=== bench_matmulgf2 " << lingen::version_string() << " ===";

    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    LOG(LOG_INFO) << "[MatMulGF2] Running on: " << prop.name << std::endl;
    LOG(LOG_INFO) << "[MatMulGF2] SM Count: " << prop.multiProcessorCount << std::endl;

    const int iterations = 50;

    run_benchmark<64>(400000, iterations);
    run_benchmark<128>(100000, iterations);
    run_benchmark<256>(25000, iterations);
    run_benchmark<512>(6000, iterations);

    return 0;
}
