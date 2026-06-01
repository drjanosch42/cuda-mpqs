// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// test_spmm_endtoend.cu — Full pipeline test: tune + compile + execute with
// both old (CPU) and new (GPU) paths, compare SpMM output bit-for-bit.
// No test framework; standalone main() with assert() and PASS/FAIL logging.

#include "common.h"
#include "autotuner.h"
#include "gpu_autotuner.h"
#include "spmm_optimized.h"
#include "generator.h"
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
enum DensityProfile { UNIFORM, POWER_LAW, BIMODAL };

static HostMatrix generate_test_matrix(row_idx_t rows, idx_t cols, size_t target_nnz,
                                       DensityProfile profile, uint64_t seed) {
    std::mt19937_64 rng(seed);
    HostMatrix mat;
    mat.n_rows = rows;
    mat.n_cols = cols;
    mat.rows.resize(rows);

    std::vector<double> weights(rows, 1.0);
    if (profile == POWER_LAW) {
        for (row_idx_t r = 0; r < std::min<row_idx_t>(10, rows); r++)
            weights[r] = (0.5 * target_nnz / 10.0);
        double remaining = 0.5 * target_nnz / std::max<row_idx_t>(1, rows - 10);
        for (row_idx_t r = 10; r < rows; r++) weights[r] = remaining;
    } else if (profile == BIMODAL) {
        row_idx_t n_dense = std::min<row_idx_t>(100, rows);
        double dense_nnz = 0.8 * target_nnz / n_dense;
        double sparse_nnz = 0.2 * target_nnz / std::max<row_idx_t>(1, rows - n_dense);
        for (row_idx_t r = 0; r < n_dense; r++) weights[r] = dense_nnz;
        for (row_idx_t r = n_dense; r < rows; r++) weights[r] = sparse_nnz;
    } else {
        double avg = (double)target_nnz / rows;
        for (row_idx_t r = 0; r < rows; r++) weights[r] = avg;
    }

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

/// Run end-to-end comparison for a given matrix and transposition flag.
static bool run_e2e_test(const HostMatrix& mat, bool is_transposed, int vec_width) {
    const char* label = is_transposed ? "AT" : "A";

    // 1. Old path: CPU tune + compile + execute
    ExecutionPlan plan_old = SpMMAutoTuner::tune_global(mat, is_transposed, vec_width, false);
    OptimizedSpMM engine_old;
    engine_old.compile(mat, plan_old);

    // 2. New path: GPU tune + compile + execute
    GPUAutoTuner::Config gpu_cfg;
    gpu_cfg.allow_cpu_fallback = true;
    ExecutionPlan plan_new = GPUAutoTuner::tune(mat, is_transposed, vec_width, gpu_cfg, false);
    OptimizedSpMM engine_new;
    engine_new.compile(mat, plan_new);

    // 3. Allocate input/output buffers
    size_t vec_bytes = (size_t)mat.n_cols * (vec_width / 8);
    size_t out_bytes = (size_t)mat.n_rows * (vec_width / 8);

    uint8_t *d_V = nullptr, *d_C_old = nullptr, *d_C_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_V, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_old, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_new, out_bytes));

    // Fixed-seed random input vector (same for both paths)
    std::vector<uint8_t> h_V(vec_bytes);
    std::mt19937_64 rng(0xCAFEBABE);
    for (size_t i = 0; i < vec_bytes; i++) h_V[i] = rng() & 0xFF;
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_old, 0, out_bytes));
    CUDA_CHECK(cudaMemset(d_C_new, 0, out_bytes));

    // 4. Execute both paths
    engine_old.execute(d_C_old, d_V);
    engine_new.execute(d_C_new, d_V);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Download and compare
    std::vector<uint8_t> h_C_old(out_bytes), h_C_new(out_bytes);
    CUDA_CHECK(cudaMemcpy(h_C_old.data(), d_C_old, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_new.data(), d_C_new, out_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_C_old));
    CUDA_CHECK(cudaFree(d_C_new));

    bool match = (memcmp(h_C_old.data(), h_C_new.data(), out_bytes) == 0);
    if (!match) {
        // Find first mismatch for diagnostics
        for (size_t i = 0; i < out_bytes; i++) {
            if (h_C_old[i] != h_C_new[i]) {
                fprintf(stderr, "    First mismatch at byte %zu: old=0x%02x new=0x%02x (%s)\n",
                        i, h_C_old[i], h_C_new[i], label);
                break;
            }
        }
    }
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

    int failures = 0;
    int total = 0;
    constexpr int VEC_WIDTH = 128;

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

    for (auto& td : matrices) {
        fprintf(stderr, "[test_spmm_endtoend] Matrix: %s (%u x %u, nnz~%zu)\n",
                td.name, td.rows, td.cols, td.nnz);

        HostMatrix raw = generate_test_matrix(td.rows, td.cols, td.nnz, td.profile, 42);
        HostMatrix mat = sort_by_density(raw);

        // Test non-transposed
        total++;
        bool pass_A = run_e2e_test(mat, false, VEC_WIDTH);
        fprintf(stderr, "  A:  %s\n", pass_A ? "PASS" : "FAIL");
        if (!pass_A) failures++;

        // Test transposed — need physical transpose
        HostMatrix matT = MatrixGenerator::transpose(raw);
        HostMatrix matT_sorted = sort_by_density(matT);
        total++;
        bool pass_AT = run_e2e_test(matT_sorted, true, VEC_WIDTH);
        fprintf(stderr, "  AT: %s\n", pass_AT ? "PASS" : "FAIL");
        if (!pass_AT) failures++;
    }

    fprintf(stderr, "\n========================================\n");
    if (failures == 0) {
        fprintf(stderr, "ALL TESTS PASSED (%d tests)\n", total);
    } else {
        fprintf(stderr, "FAILED: %d tests (out of %d)\n", failures, total);
    }
    return failures > 0 ? 1 : 0;
}
