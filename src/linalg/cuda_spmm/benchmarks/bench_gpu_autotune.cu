// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// bench_gpu_autotune.cu — Standalone benchmark comparing legacy tune_global()
// vs GPU-only GPUAutoTuner::tune() across multiple matrix sizes.
// Produces structured log output to logs/bench_gpu_autotune.log.

#include "common.h"
#include "autotuner.h"
#include "gpu_autotuner.h"
#include "spmm_optimized.h"
#include "generator.h"
#include "hpc_logger.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
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

/// Take median of N runs for stable timing.
static double median_time(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
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

    std::ofstream log("logs/bench_gpu_autotune.log");
    log << "# bench_gpu_autotune log\n";
    log << "# matrix rows cols nnz old_ms new_ms speedup old_segments new_segments old_est_ms new_est_ms\n";

    constexpr int VEC_WIDTH = 128;
    constexpr int RUNS = 3;

    struct TestCase {
        const char* name;
        row_idx_t rows;
        idx_t cols;
        size_t nnz;
        DensityProfile profile;
    };

    std::vector<TestCase> cases = {
        {"1k_uniform",    1000,   1000,   5000,      UNIFORM},
        {"10k_skewed",    10000,  10000,  50000,     POWER_LAW},
        {"100k_bimodal",  100000, 100000, 500000,    BIMODAL},
        {"500k_sparse",   500000, 500000, 2500000,   UNIFORM},
    };

    GPUAutoTuner::Config gpu_cfg;
    gpu_cfg.allow_cpu_fallback = true;

    for (auto& tc : cases) {
        fprintf(stderr, "[bench_gpu_autotune] %s: %u x %u, nnz=%zu ...\n",
                tc.name, tc.rows, tc.cols, tc.nnz);

        HostMatrix raw = generate_test_matrix(tc.rows, tc.cols, tc.nnz, tc.profile, 42);
        HostMatrix mat = sort_by_density(raw);

        // Benchmark old path (median of RUNS)
        std::vector<double> old_times;
        ExecutionPlan plan_old;
        for (int i = 0; i < RUNS; i++) {
            auto t0 = std::chrono::steady_clock::now();
            plan_old = SpMMAutoTuner::tune_global(mat, false, VEC_WIDTH, false);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            old_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double ms_old = median_time(old_times);

        // Benchmark new path (median of RUNS)
        std::vector<double> new_times;
        ExecutionPlan plan_new;
        for (int i = 0; i < RUNS; i++) {
            auto t0 = std::chrono::steady_clock::now();
            plan_new = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);
            CUDA_CHECK(cudaDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            new_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double ms_new = median_time(new_times);

        double speedup = ms_old / ms_new;

        log << "matrix=" << tc.name
            << " rows=" << tc.rows
            << " cols=" << tc.cols
            << " nnz=" << tc.nnz
            << " old_ms=" << ms_old
            << " new_ms=" << ms_new
            << " speedup=" << speedup
            << " old_segments=" << plan_old.segments.size()
            << " new_segments=" << plan_new.segments.size()
            << " old_est_ms=" << plan_old.estimated_total_time_ms
            << " new_est_ms=" << plan_new.estimated_total_time_ms
            << "\n";

        fprintf(stderr, "  old=%.1f ms  new=%.1f ms  speedup=%.2fx  segs=%zu/%zu\n",
                ms_old, ms_new, speedup,
                plan_old.segments.size(), plan_new.segments.size());
    }

    log.close();
    fprintf(stderr, "\n[bench_gpu_autotune] Results written to logs/bench_gpu_autotune.log\n");
    return 0;
}
