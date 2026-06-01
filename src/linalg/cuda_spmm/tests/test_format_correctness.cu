// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// test_format_correctness.cu — Validates GPU format converters produce bit-exact
// results vs CPU reference (MatrixPreprocessor) for all active formats.
// No test framework; standalone main() with assert() and PASS/FAIL logging.

#include "common.h"
#include "preprocessing.h"
#include "kernels.h"
#include "autotuner.h"
#include "gpu_autotuner.h"
#include "spmm_optimized.h"
#include "hpc_logger.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
template <typename T>
static std::vector<T> download(const T* d_ptr, size_t count) {
    std::vector<T> host(count);
    if (count > 0 && d_ptr)
        CUDA_CHECK(cudaMemcpy(host.data(), d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

/// Generate a HostMatrix with a specific density profile from a fixed seed.
enum DensityProfile { UNIFORM, POWER_LAW, BIMODAL };

static HostMatrix generate_test_matrix(row_idx_t rows, idx_t cols, size_t target_nnz,
                                       DensityProfile profile, uint64_t seed) {
    std::mt19937_64 rng(seed);
    HostMatrix mat;
    mat.n_rows = rows;
    mat.n_cols = cols;
    mat.rows.resize(rows);

    // Compute per-row NNZ weights
    std::vector<double> weights(rows, 1.0);
    if (profile == POWER_LAW) {
        // Top 10 rows get 50% of NNZ
        for (row_idx_t r = 0; r < std::min<row_idx_t>(10, rows); r++)
            weights[r] = (0.5 * target_nnz / 10.0);
        double remaining = 0.5 * target_nnz / std::max<row_idx_t>(1, rows - 10);
        for (row_idx_t r = 10; r < rows; r++)
            weights[r] = remaining;
    } else if (profile == BIMODAL) {
        // 100 dense rows + rest very sparse
        row_idx_t n_dense = std::min<row_idx_t>(100, rows);
        double dense_nnz = 0.8 * target_nnz / n_dense;
        double sparse_nnz = 0.2 * target_nnz / std::max<row_idx_t>(1, rows - n_dense);
        for (row_idx_t r = 0; r < n_dense; r++) weights[r] = dense_nnz;
        for (row_idx_t r = n_dense; r < rows; r++) weights[r] = sparse_nnz;
    } else {
        double avg = (double)target_nnz / rows;
        for (row_idx_t r = 0; r < rows; r++) weights[r] = avg;
    }

    // Generate column indices per row
    std::uniform_int_distribution<idx_t> col_dist(0, cols - 1);
    for (row_idx_t r = 0; r < rows; r++) {
        size_t nnz_r = std::max<size_t>(1, (size_t)(weights[r] + 0.5));
        nnz_r = std::min<size_t>(nnz_r, cols);
        std::vector<bool> used(cols, false);
        for (size_t k = 0; k < nnz_r; k++) {
            idx_t c;
            do { c = col_dist(rng); } while (used[c]);
            used[c] = true;
            mat.rows[r].push_back(c);
        }
        std::sort(mat.rows[r].begin(), mat.rows[r].end());
    }
    return mat;
}

/// Sort matrix rows by descending density (matches autotuner expectation).
static HostMatrix sort_by_density(const HostMatrix& A) {
    std::vector<row_idx_t> perm(A.n_rows);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](row_idx_t a, row_idx_t b) { return A.rows[a].size() > A.rows[b].size(); });
    HostMatrix out;
    out.n_rows = A.n_rows;
    out.n_cols = A.n_cols;
    out.rows.resize(A.n_rows);
    for (row_idx_t i = 0; i < A.n_rows; i++) out.rows[i] = A.rows[perm[i]];
    return out;
}

// ---------------------------------------------------------------------------
// SpMM comparison: run SpMM with both CPU-path and GPU-path DeviceMatrix and
// compare output bit-for-bit.
// ---------------------------------------------------------------------------
static bool compare_spmm_output(const HostMatrix& mat, const ExecutionPlan& plan_cpu,
                                const ExecutionPlan& plan_gpu, int vec_width,
                                std::ostream& log) {
    OptimizedSpMM engine_cpu, engine_gpu;
    engine_cpu.compile(mat, plan_cpu);
    engine_gpu.compile(mat, plan_gpu);

    size_t vec_bytes = (size_t)mat.n_cols * (vec_width / 8);
    size_t out_bytes = (size_t)mat.n_rows * (vec_width / 8);

    uint8_t *d_V = nullptr, *d_C_cpu = nullptr, *d_C_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_V, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_cpu, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_gpu, out_bytes));

    // Fill input with deterministic pattern
    std::vector<uint8_t> h_V(vec_bytes);
    std::mt19937_64 rng(0xDEADBEEF);
    for (size_t i = 0; i < vec_bytes; i++) h_V[i] = rng() & 0xFF;
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_cpu, 0, out_bytes));
    CUDA_CHECK(cudaMemset(d_C_gpu, 0, out_bytes));

    engine_cpu.execute(d_C_cpu, d_V);
    engine_gpu.execute(d_C_gpu, d_V);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_C_cpu = download(d_C_cpu, out_bytes);
    auto h_C_gpu = download(d_C_gpu, out_bytes);

    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_C_cpu));
    CUDA_CHECK(cudaFree(d_C_gpu));

    bool match = (memcmp(h_C_cpu.data(), h_C_gpu.data(), out_bytes) == 0);
    log << " spmm_match=" << (match ? "PASS" : "FAIL")
        << " out_bytes=" << out_bytes;
    return match;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    LogConfig lcfg;
    lcfg.enable_cout = true;
    lcfg.min_severity_cout = LOG_INFO;
    lcfg.enable_file = false;
    HPCLogger::Get().Init(lcfg);

    std::ofstream log("logs/format_correctness.log");
    log << "# format_correctness test log\n";

    int failures = 0;
    int total = 0;

    // Test matrix definitions
    struct TestDef {
        const char* name;
        row_idx_t rows;
        idx_t cols;
        size_t nnz;
        DensityProfile profile;
    };
    std::vector<TestDef> matrices = {
        {"tiny_uniform",    100,    100,    500,     UNIFORM},
        {"small_skewed",    1000,   1000,   10000,   POWER_LAW},
        {"medium_bimodal",  10000,  10000,  100000,  BIMODAL},
        {"wide_sparse",     1000,   100000, 5000,    UNIFORM},
        {"tall_dense",      100000, 1000,   500000,  UNIFORM},
    };

    constexpr int VEC_WIDTH = 128;

    for (auto& td : matrices) {
        fprintf(stderr, "[test_format_correctness] Testing matrix: %s (%u x %u, nnz~%zu)\n",
                td.name, td.rows, td.cols, td.nnz);

        HostMatrix raw = generate_test_matrix(td.rows, td.cols, td.nnz, td.profile, 42);
        HostMatrix mat = sort_by_density(raw);

        // --- CPU path: use legacy tune_global ---
        ExecutionPlan plan_cpu = SpMMAutoTuner::tune_global(mat, false, VEC_WIDTH, false);

        // --- GPU path: use GPUAutoTuner ---
        GPUAutoTuner::Config gpu_cfg;
        // Disable formats that may not have GPU converters yet
        // (they will use CPU fallback if allow_cpu_fallback is true)
        gpu_cfg.allow_cpu_fallback = true;
        ExecutionPlan plan_gpu = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);

        // Compare SpMM outputs
        total++;
        log << "matrix=" << td.name
            << " rows=" << td.rows << " cols=" << td.cols << " nnz=" << td.nnz;

        bool pass = compare_spmm_output(mat, plan_cpu, plan_gpu, VEC_WIDTH, log);
        log << " result=" << (pass ? "PASS" : "FAIL") << "\n";

        if (!pass) {
            fprintf(stderr, "  FAIL: SpMM output mismatch for %s\n", td.name);
            failures++;
        } else {
            fprintf(stderr, "  PASS: %s\n", td.name);
        }
    }

    log.close();

    fprintf(stderr, "\n========================================\n");
    if (failures == 0) {
        fprintf(stderr, "ALL TESTS PASSED (%d tests)\n", total);
    } else {
        fprintf(stderr, "FAILED: %d tests (out of %d)\n", failures, total);
    }
    return failures > 0 ? 1 : 0;
}
