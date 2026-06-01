// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// test_memory_leak.cu — Verifies that repeated tune + compile + execute + destroy
// cycles do not leak GPU memory beyond CUDA allocator fragmentation tolerance.
// No test framework; standalone main() with assert() and PASS/FAIL logging.

#include "common.h"
#include "gpu_autotuner.h"
#include "spmm_optimized.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Matrix generation
// ---------------------------------------------------------------------------
static HostMatrix generate_test_matrix(row_idx_t rows, idx_t cols, size_t target_nnz,
                                       uint64_t seed) {
    std::mt19937_64 rng(seed);
    HostMatrix mat;
    mat.n_rows = rows;
    mat.n_cols = cols;
    mat.rows.resize(rows);

    double avg = (double)target_nnz / rows;
    std::uniform_int_distribution<idx_t> col_dist(0, cols - 1);

    for (row_idx_t r = 0; r < rows; r++) {
        size_t nnz_r = std::max<size_t>(1, (size_t)(avg + 0.5));
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
// Main
// ---------------------------------------------------------------------------
int main() {
    LogConfig lcfg;
    lcfg.enable_cout = true;
    lcfg.min_severity_cout = LOG_INFO;
    lcfg.enable_file = false;
    HPCLogger::Get().Init(lcfg);

    constexpr int VEC_WIDTH = 128;
    constexpr int ITERATIONS = 5;
    constexpr size_t TOLERANCE_BYTES = 1 * 1024 * 1024; // 1 MB

    HostMatrix raw = generate_test_matrix(10000, 10000, 50000, 42);
    HostMatrix mat = sort_by_density(raw);

    size_t vec_bytes = (size_t)mat.n_cols * (VEC_WIDTH / 8);
    size_t out_bytes = (size_t)mat.n_rows * (VEC_WIDTH / 8);

    // Allocate persistent input/output buffers
    uint8_t *d_V = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_V, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, out_bytes));
    CUDA_CHECK(cudaMemset(d_V, 0xFF, vec_bytes));

    GPUAutoTuner::Config gpu_cfg;
    gpu_cfg.allow_cpu_fallback = true;

    // Warm up: run one full cycle to prime the CUDA allocator
    {
        ExecutionPlan plan = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);
        OptimizedSpMM engine;
        engine.compile(mat, plan);
        engine.execute(d_C, d_V);
        CUDA_CHECK(cudaDeviceSynchronize());
        // engine destructor frees resources
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record baseline memory
    size_t free_before = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));
    fprintf(stderr, "[test_memory_leak] Baseline: free=%zu MB total=%zu MB\n",
            free_before / (1024 * 1024), total_mem / (1024 * 1024));

    // Run N iterations
    for (int i = 0; i < ITERATIONS; i++) {
        ExecutionPlan plan = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);
        OptimizedSpMM engine;
        engine.compile(mat, plan);
        engine.execute(d_C, d_V);
        CUDA_CHECK(cudaDeviceSynchronize());
        // engine destructor runs here, calling free_resources()
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Check final memory
    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));
    fprintf(stderr, "[test_memory_leak] After %d iterations: free=%zu MB\n",
            ITERATIONS, free_after / (1024 * 1024));

    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_C));

    // Allow for CUDA allocator fragmentation
    bool pass = true;
    if (free_before > free_after) {
        size_t drift = free_before - free_after;
        fprintf(stderr, "[test_memory_leak] Memory drift: %zu bytes (tolerance: %zu)\n",
                drift, TOLERANCE_BYTES);
        if (drift > TOLERANCE_BYTES) {
            fprintf(stderr, "  FAIL: Memory leak detected (%zu bytes > %zu tolerance)\n",
                    drift, TOLERANCE_BYTES);
            pass = false;
        }
    }

    fprintf(stderr, "\n========================================\n");
    if (pass) {
        fprintf(stderr, "ALL TESTS PASSED\n");
    } else {
        fprintf(stderr, "FAILED: 1 tests\n");
    }
    return pass ? 0 : 1;
}
