// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "lingen/types.h"
#include "lingen/operations/poly_arithmetic.h" // Fixed include
#include "lingen/operations/karatsuba.cuh"
#include "lingen/operations/matmul_gf2.h"      // Fixed include

namespace lingen {

// High-level backend (Karatsuba today, FFT later).
enum class PolyMulBackendKind : int {
    Karatsuba = 0,
    FFT_Future = 1,
};

// Autotune: representative degree policy + kernel sweep.
struct PolyArithAutotuneHyperParams {
    // Candidate N values to test
    std::vector<int> Ns = {64, 128, 256, 512};

    // Threshold sweep (inclusive)
    int threshold_min = 32;
    int threshold_max = 256;
    int threshold_step = 32;

    // Leaf variants to test (includes MPB leaf)
    std::vector<PolyMulLeafKind> leaf_kinds = {
        PolyMulLeafKind::NaiveFused_SmemB,
        PolyMulLeafKind::NaiveFused_Global,
        PolyMulLeafKind::NaiveFused_SmemB_MPB,
    };

    // Representative degree selection
    int bw_degree_target = 4096; // stage-2 polynomial degree from BW instance (adjustable)
    int deg_min_tune = 1024;
    int deg_max_tune = 32768;

    // If true: choose representative polynomial length L as next_pow2(L_target) - 1,
    // i.e. degree = (next_pow2(degree+1) - 1) - 1 = next_pow2(degree+1) - 2.
    // This implements your "polynomial length power-of-two minus 1" rule.
    bool poly_len_pow2_minus1 = true;

    // Cold-cache measurement (optional, but needed for cache-regime weighting).
    bool measure_cold_cache = true;
    size_t coldcache_thrash_bytes = 128ull * 1024ull * 1024ull; // matches existing benches [file:1][file:3]

    // Cache-regime scoring controls
    double l2_fit_factor = 0.80;  // treat <= 0.8*L2 as "L2 resident"
    double dram_factor   = 4.00;  // treat >= 4.0*L2 as "DRAM dominated"

    int warmup = 3;
    int iterations = 10;

    // Optional correctness checks during tuning (kept as hook; can be wired to GPU reference later)
    bool verify_small = false;
    int verify_degree = 32;
};

struct PolyArithBenchmarkPoint {
    int N = 0;
    PolyMulBackendKind backend = PolyMulBackendKind::Karatsuba;
    PolyMulLeafKind leaf = PolyMulLeafKind::NaiveFused_SmemB;
    int threshold = 0;

    int degree = 0;          // representative degree used
    double hot_ms = 0.0;     // avg ms per multiply (hot-cache)
    double cold_ms = -1.0;   // avg ms per multiply (cold-cache), if measured
    double score_ms = 0.0;   // scalar used for winner selection at this N
};

struct PolyArithDeviceInfo {
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    int l2_bytes = 0;
};

struct PolyArithBestConfig {
    int N = 0;
    PolyMulBackendKind backend = PolyMulBackendKind::Karatsuba;
    KaratsubaTuneConfig karatsuba;
};

class PolyArithmeticEngine {
public:
    PolyArithmeticEngine() = default;
    explicit PolyArithmeticEngine(const PolyArithBestConfig& cfg) : cfg_(cfg) {}

    const PolyArithBestConfig& config() const { return cfg_; }
    int N() const { return cfg_.N; }

    void poly_mul(cudaStream_t stream, void* C, const void* A, size_t lenA, const void* B, size_t lenB, void* workspace);
    size_t workspace_bytes(size_t max_len) const;

    // Autotune: returns best config across tested (N, threshold, leaf) and records points.
    static PolyArithBestConfig initialize_and_autotune(
        int device_id,
        const PolyArithAutotuneHyperParams& hp,
        std::vector<PolyArithBenchmarkPoint>* out_points = nullptr,
        PolyArithDeviceInfo* out_devinfo = nullptr,
        const char* csv_path = nullptr
    );

private:
    PolyArithBestConfig cfg_;
};

} // namespace lingen
