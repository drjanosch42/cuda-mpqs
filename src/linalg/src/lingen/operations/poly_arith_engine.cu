// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "lingen/operations/poly_arith_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "hpc_logger.h"

namespace lingen {

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        LOG(LOG_ERROR_CRITICAL) << "CUDA Error: " << cudaGetErrorString(err) << " at " __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)
#endif

static PolyArithDeviceInfo get_device_info(int device_id) {
    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, device_id));
    PolyArithDeviceInfo di;
    di.name = p.name;
    di.cc_major = p.major;
    di.cc_minor = p.minor;
    di.sm_count = p.multiProcessorCount;
    di.l2_bytes = p.l2CacheSize;
    return di;
}

static void fill_random_bytes(std::vector<uint8_t>& v, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, 255);
    for (auto& b : v) b = (uint8_t)dist(gen);
}

// Cache-thrash kernel (same idea as the existing benches). [file:1][file:3]
__global__ void thrash_cache_kernel(int* garbage, size_t n_ints) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n_ints; i += stride) {
        garbage[i] ^= (int)i;
    }
}

static void run_cache_thrash(int* d_garbage, size_t thrash_bytes, cudaStream_t stream) {
    size_t n_ints = thrash_bytes / sizeof(int);
    int threads = 1024;
    int blocks = (int)std::min<size_t>((n_ints + threads - 1) / threads, (size_t)8192);
    thrash_cache_kernel<<<blocks, threads, 0, stream>>>(d_garbage, n_ints);
}

static inline uint32_t next_pow2_u32(uint32_t x) {
    if (x <= 1u) return 1u;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Your desired rule is on polynomial length L = degree+1. Pick L ~= pow2 and then subtract 1.
// Return representative degree (>=0).
static int pick_representative_degree(const PolyArithAutotuneHyperParams& hp) {
    const int deg_target = std::max(0, hp.bw_degree_target);
    const uint32_t L_target = (uint32_t)deg_target + 1u;

    uint32_t L_rep = next_pow2_u32(L_target);

    // Apply caps in "degree space" but while respecting the length logic.
    // Cap by converting deg caps -> length caps.
    const uint32_t L_min = (uint32_t)std::max(1, hp.deg_min_tune + 1);
    const uint32_t L_max = (uint32_t)std::max(1, hp.deg_max_tune + 1);

    if (L_rep < L_min) L_rep = next_pow2_u32(L_min);
    if (L_rep > L_max) L_rep = L_max;

    if (hp.poly_len_pow2_minus1) {
        // L_final = L_rep - 1 (must stay >= 1)
        if (L_rep > 1) L_rep -= 1;
    }

    int deg_rep = (int)L_rep - 1;
    if (deg_rep < 0) deg_rep = 0;
    return deg_rep;
}

// Compute the "ABC footprint" exactly like bench_karatsuba does: bytesA+bytesB+bytesC. [file:1]
template<int N>
static size_t footprint_ABC_bytes_for_degree(int degree) {
    const size_t lenA = (size_t)degree + 1;
    const size_t lenB = (size_t)degree + 1;
    const size_t lenC = lenA + lenB - 1;
    const size_t mat_bytes = PolyMatrixView<N>::MAT_SIZE_BYTES;
    return (lenA + lenB + lenC) * mat_bytes;
}

static double score_from_hot_cold(
    double hot_ms,
    double cold_ms,
    size_t footprint_bytes,
    int l2_bytes,
    const PolyArithAutotuneHyperParams& hp
) {
    // If cold wasn't measured, fall back to hot.
    if (cold_ms < 0.0 || l2_bytes <= 0) return hot_ms;

    const double f = (double)footprint_bytes;
    const double l2 = (double)l2_bytes;

    const double fit_thr = hp.l2_fit_factor * l2;
    const double dram_thr = hp.dram_factor * l2;

    if (f <= fit_thr) return hot_ms;
    if (f >= dram_thr) return cold_ms;

    // Linear blend between regimes.
    const double t = (f - fit_thr) / (dram_thr - fit_thr); // 0..1
    return (1.0 - t) * hot_ms + t * cold_ms;
}

template<int N>
static void bench_one_case_hot_cold_ms(
    int degree,
    const KaratsubaTuneConfig& kcfg,
    int warmup,
    int iterations,
    bool measure_cold,
    size_t thrash_bytes,
    double* out_hot_ms,
    double* out_cold_ms
) {
    const size_t lenA = (size_t)degree + 1;
    const size_t lenB = (size_t)degree + 1;
    const size_t lenC = lenA + lenB - 1;

    const size_t mat_bytes = PolyMatrixView<N>::MAT_SIZE_BYTES;
    const size_t bytesA = lenA * mat_bytes;
    const size_t bytesB = lenB * mat_bytes;
    const size_t bytesC = lenC * mat_bytes;

    std::vector<uint8_t> hA(bytesA), hB(bytesB);
    fill_random_bytes(hA, 123u + (uint32_t)degree);
    fill_random_bytes(hB, 999u + (uint32_t)degree);

    uint64_t* dA = nullptr;
    uint64_t* dB = nullptr;
    uint64_t* dC = nullptr;
    uint64_t* dWs = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    const size_t ws_bytes = karatsuba_workspace_size<N>(std::max(lenA, lenB));
    CHECK_CUDA(cudaMalloc(&dWs, ws_bytes));
    CHECK_CUDA(cudaMemset(dWs, 0, ws_bytes));

    int* dGarbage = nullptr;
    if (measure_cold) {
        CHECK_CUDA(cudaMalloc(&dGarbage, thrash_bytes));
        CHECK_CUDA(cudaMemset(dGarbage, 0, thrash_bytes));
    }

    PolyMatrixView<N> A(dA, lenA);
    PolyMatrixView<N> B(dB, lenB);
    PolyMatrixView<N> C(dC, lenC);

    cudaStream_t stream{};
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Warmup (hot)
    for (int i = 0; i < warmup; i++) {
        karatsuba_mul_tuned<N>(stream, C, A, B, dWs, kcfg);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start{}, stop{};
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Hot timing
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iterations; i++) {
        karatsuba_mul_tuned<N>(stream, C, A, B, dWs, kcfg);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_hot_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_hot_total, start, stop));
    const double hot_ms = (double)ms_hot_total / (double)iterations;

    double cold_ms = -1.0;

    if (measure_cold) {
        // Measure thrash-only overhead (same structure as existing benches). [file:1][file:3]
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; i++) {
            run_cache_thrash(dGarbage, thrash_bytes, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_thrash_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_thrash_total, start, stop));

        // Measure thrash + multiply
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; i++) {
            run_cache_thrash(dGarbage, thrash_bytes, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            karatsuba_mul_tuned<N>(stream, C, A, B, dWs, kcfg);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_cold_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_cold_total, start, stop));

        const double avg_thrash = (double)ms_thrash_total / (double)iterations;
        const double avg_cold_total = (double)ms_cold_total / (double)iterations;
        cold_ms = avg_cold_total - avg_thrash;
        if (cold_ms < 0.0) cold_ms = hot_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    if (dGarbage) CHECK_CUDA(cudaFree(dGarbage));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dWs));

    *out_hot_ms = hot_ms;
    *out_cold_ms = cold_ms;
}

static void append_csv(
    const char* path,
    const PolyArithDeviceInfo& di,
    const PolyArithBenchmarkPoint& pt
) {
    if (!path) return;
    std::FILE* f = std::fopen(path, "a");
    if (!f) return;

    // device,cc,sm,N,backend,leaf,threshold,degree,hot_ms,cold_ms,score_ms
    std::fprintf(
        f,
        "%s,%d.%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
        di.name.c_str(),
        di.cc_major, di.cc_minor,
        di.sm_count,
        pt.N,
        (int)pt.backend,
        (int)pt.leaf,
        pt.threshold,
        pt.degree,
        pt.hot_ms,
        pt.cold_ms,
        pt.score_ms
    );
    std::fclose(f);
}

PolyArithBestConfig PolyArithmeticEngine::initialize_and_autotune(
    int device_id,
    const PolyArithAutotuneHyperParams& hp,
    std::vector<PolyArithBenchmarkPoint>* out_points,
    PolyArithDeviceInfo* out_devinfo,
    const char* csv_path
) {
    CHECK_CUDA(cudaSetDevice(device_id));

    PolyArithDeviceInfo di = get_device_info(device_id);
    if (out_devinfo) *out_devinfo = di;

    PolyArithBestConfig best{};
    double best_score = 1e300;

    const int deg_rep = pick_representative_degree(hp);

    for (int N : hp.Ns) {
        LOG(LOG_DEBUG_1) << "[PolyArithEngine] [AutoTune] Evaluating N = " << N << std::endl;
        for (int thr = hp.threshold_min; thr <= hp.threshold_max; thr += hp.threshold_step) {
	    LOG(LOG_DEBUG_2) << "[PolyArithEngine] [AutoTune] Evaluating N = " << N << ", threshold = " << thr << std::endl;
            for (auto leaf : hp.leaf_kinds) {
                KaratsubaTuneConfig kcfg;
                kcfg.threshold = thr;
                kcfg.leaf_kind = leaf;

                double hot_ms = 0.0;
                double cold_ms = -1.0;
                size_t footprint = 0;

                switch (N) {
                    case 64:
                        bench_one_case_hot_cold_ms<64>(
                            deg_rep, kcfg, hp.warmup, hp.iterations,
                            hp.measure_cold_cache, hp.coldcache_thrash_bytes,
                            &hot_ms, &cold_ms
                        );
                        footprint = footprint_ABC_bytes_for_degree<64>(deg_rep);
                        break;
                    case 128:
                        bench_one_case_hot_cold_ms<128>(
                            deg_rep, kcfg, hp.warmup, hp.iterations,
                            hp.measure_cold_cache, hp.coldcache_thrash_bytes,
                            &hot_ms, &cold_ms
                        );
                        footprint = footprint_ABC_bytes_for_degree<128>(deg_rep);
                        break;
                    case 256:
                        bench_one_case_hot_cold_ms<256>(
                            deg_rep, kcfg, hp.warmup, hp.iterations,
                            hp.measure_cold_cache, hp.coldcache_thrash_bytes,
                            &hot_ms, &cold_ms
                        );
                        footprint = footprint_ABC_bytes_for_degree<256>(deg_rep);
                        break;
                    case 512:
                        bench_one_case_hot_cold_ms<512>(
                            deg_rep, kcfg, hp.warmup, hp.iterations,
                            hp.measure_cold_cache, hp.coldcache_thrash_bytes,
                            &hot_ms, &cold_ms
                        );
                        footprint = footprint_ABC_bytes_for_degree<512>(deg_rep);
                        break;
                    default:
                        continue;
                }

                const double score_ms = score_from_hot_cold(
                    hot_ms, cold_ms, footprint, di.l2_bytes, hp
                );

                PolyArithBenchmarkPoint pt;
                pt.N = N;
                pt.backend = PolyMulBackendKind::Karatsuba;
                pt.leaf = leaf;
                pt.threshold = thr;
                pt.degree = deg_rep;
                pt.hot_ms = hot_ms;
                pt.cold_ms = cold_ms;
                pt.score_ms = score_ms;

                if (out_points) out_points->push_back(pt);
                append_csv(csv_path, di, pt);

                if (score_ms < best_score) {
                    best_score = score_ms;
                    best.N = N;
                    best.backend = PolyMulBackendKind::Karatsuba;
                    best.karatsuba = kcfg;
                }
            }
        }
    }

    // Fallback
    if (best.N == 0) {
        best.N = 128;
        best.backend = PolyMulBackendKind::Karatsuba;
        best.karatsuba.threshold = 32;
        best.karatsuba.leaf_kind = PolyMulLeafKind::NaiveFused_SmemB;
    }

    return best;
}

size_t PolyArithmeticEngine::workspace_bytes(size_t max_len) const {
    switch (cfg_.N) {
        case 64:  return karatsuba_workspace_size<64>(max_len);
        case 128: return karatsuba_workspace_size<128>(max_len);
        case 256: return karatsuba_workspace_size<256>(max_len);
        case 512: return karatsuba_workspace_size<512>(max_len);
        default:  return 0;
    }
}

void PolyArithmeticEngine::poly_mul(
    cudaStream_t stream,
    void* C_,
    const void* A_,
    size_t lenA,
    const void* B_,
    size_t lenB,
    void* workspace_
) {
    if (cfg_.backend != PolyMulBackendKind::Karatsuba) {
        LOG(LOG_ERROR_CRITICAL) << "[PolyArithEngine] poly_mul: backend not implemented" << std::endl;
        std::abort();
    }

    const size_t lenC = lenA + lenB - 1;
    uint64_t* C = reinterpret_cast<uint64_t*>(C_);
    const uint64_t* A = reinterpret_cast<const uint64_t*>(A_);
    const uint64_t* B = reinterpret_cast<const uint64_t*>(B_);
    uint64_t* ws = reinterpret_cast<uint64_t*>(workspace_);

    switch (cfg_.N) {
        case 64: {
            PolyMatrixView<64> Av(const_cast<uint64_t*>(A), lenA);
            PolyMatrixView<64> Bv(const_cast<uint64_t*>(B), lenB);
            PolyMatrixView<64> Cv(C, lenC);
            karatsuba_mul_tuned<64>(stream, Cv, Av, Bv, ws, cfg_.karatsuba);
        } break;
        case 128: {
            PolyMatrixView<128> Av(const_cast<uint64_t*>(A), lenA);
            PolyMatrixView<128> Bv(const_cast<uint64_t*>(B), lenB);
            PolyMatrixView<128> Cv(C, lenC);
            karatsuba_mul_tuned<128>(stream, Cv, Av, Bv, ws, cfg_.karatsuba);
        } break;
        case 256: {
            PolyMatrixView<256> Av(const_cast<uint64_t*>(A), lenA);
            PolyMatrixView<256> Bv(const_cast<uint64_t*>(B), lenB);
            PolyMatrixView<256> Cv(C, lenC);
            karatsuba_mul_tuned<256>(stream, Cv, Av, Bv, ws, cfg_.karatsuba);
        } break;
        case 512: {
            PolyMatrixView<512> Av(const_cast<uint64_t*>(A), lenA);
            PolyMatrixView<512> Bv(const_cast<uint64_t*>(B), lenB);
            PolyMatrixView<512> Cv(C, lenC);
            karatsuba_mul_tuned<512>(stream, Cv, Av, Bv, ws, cfg_.karatsuba);
        } break;
        default:
	    LOG(LOG_ERROR_CRITICAL) << "[PolyArithEngine] poly_mul: unsupported N = " <<  cfg_.N << std::endl;
            std::abort();
    }
}

} // namespace lingen
