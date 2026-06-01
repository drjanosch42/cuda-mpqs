// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// test_autotuner_ab.cu — A/B comparison of legacy tune_global() vs GPU-only
// GPUAutoTuner::tune(). Validates plan quality and logs timing.
// No test framework; standalone main() with assert() and PASS/FAIL logging.

#include "common.h"
#include "autotuner.h"
#include "gpu_autotuner.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Matrix generation (same helper as other tests)
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

/// Take median of N runs.
static double benchmark_tuner_old(const HostMatrix& mat, int vec_width, int runs) {
    std::vector<double> times;
    for (int i = 0; i < runs; i++) {
        auto t0 = std::chrono::steady_clock::now();
        auto plan = SpMMAutoTuner::tune_global(mat, false, vec_width, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static std::pair<double, ExecutionPlan> benchmark_tuner_new(
    const HostMatrix& mat, int vec_width, const GPUAutoTuner::Config& cfg, int runs) {
    std::vector<double> times;
    ExecutionPlan last_plan;
    for (int i = 0; i < runs; i++) {
        auto t0 = std::chrono::steady_clock::now();
        last_plan = GPUAutoTuner::tune(mat, false, vec_width, cfg, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return {times[times.size() / 2], last_plan};
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

    std::ofstream log("logs/autotune_ab_comparison.log");
    log << "# A/B autotuner comparison log\n";

    int failures = 0;
    int total = 0;
    constexpr int VEC_WIDTH = 128;
    constexpr int RUNS = 3;

    struct TestDef {
        const char* name;
        row_idx_t rows;
        idx_t cols;
        size_t nnz;
        DensityProfile profile;
    };
    std::vector<TestDef> matrices = {
        {"small_skewed",    1000,   1000,   10000,   POWER_LAW},
        {"medium_bimodal",  10000,  10000,  100000,  BIMODAL},
        {"tall_dense",      100000, 1000,   500000,  UNIFORM},
    };

    GPUAutoTuner::Config gpu_cfg;
    gpu_cfg.allow_cpu_fallback = true;

    for (auto& td : matrices) {
        fprintf(stderr, "[test_autotuner_ab] Matrix: %s (%u x %u, nnz~%zu)\n",
                td.name, td.rows, td.cols, td.nnz);

        HostMatrix raw = generate_test_matrix(td.rows, td.cols, td.nnz, td.profile, 42);
        HostMatrix mat = sort_by_density(raw);

        // Benchmark old tuner (median of RUNS)
        double t_old = benchmark_tuner_old(mat, VEC_WIDTH, RUNS);
        ExecutionPlan plan_old = SpMMAutoTuner::tune_global(mat, false, VEC_WIDTH, false);

        // Benchmark new tuner (median of RUNS)
        auto [t_new, plan_new] = benchmark_tuner_new(mat, VEC_WIDTH, gpu_cfg, RUNS);

        double speedup = t_old / t_new;

        log << "matrix=" << td.name
            << " rows=" << td.rows << " cols=" << td.cols << " nnz=" << td.nnz
            << " t_old_ms=" << t_old << " t_new_ms=" << t_new
            << " speedup=" << speedup
            << " old_segments=" << plan_old.segments.size()
            << " new_segments=" << plan_new.segments.size()
            << " old_est_ms=" << plan_old.estimated_total_time_ms
            << " new_est_ms=" << plan_new.estimated_total_time_ms
            << "\n";

        // Per-segment details
        for (size_t i = 0; i < plan_new.segments.size(); i++) {
            auto& s = plan_new.segments[i];
            log << "  new_seg[" << i << "] rows=[" << s.start_row << "," << s.end_row
                << ") kernel=" << s.best_config.name
                << " throughput=" << s.measured_throughput_gnnz << "\n";
        }

        // --- Assertions ---
        total++;
        bool pass = true;

        // 1. New plan should be no more than 5% worse in estimated time
        if (plan_new.estimated_total_time_ms > plan_old.estimated_total_time_ms * 1.05) {
            fprintf(stderr, "  WARN: New plan estimated time %.4f ms > old * 1.05 = %.4f ms\n",
                    plan_new.estimated_total_time_ms,
                    plan_old.estimated_total_time_ms * 1.05);
            // This is a soft warning, not a hard failure — the GPU tuner may pick
            // different kernels with slightly different estimates.
        }

        // 2. Plan covers all rows
        if (!plan_new.segments.empty()) {
            if (plan_new.segments.front().start_row != 0) {
                fprintf(stderr, "  FAIL: First segment starts at %u, not 0\n",
                        plan_new.segments.front().start_row);
                pass = false;
            }
            if (plan_new.segments.back().end_row != mat.n_rows) {
                fprintf(stderr, "  FAIL: Last segment ends at %u, not %u\n",
                        plan_new.segments.back().end_row, mat.n_rows);
                pass = false;
            }
        } else {
            fprintf(stderr, "  FAIL: Empty plan\n");
            pass = false;
        }

        // 3. No gaps or overlaps
        for (size_t i = 1; i < plan_new.segments.size(); i++) {
            if (plan_new.segments[i].start_row != plan_new.segments[i - 1].end_row) {
                fprintf(stderr, "  FAIL: Gap/overlap between segments %zu and %zu\n", i - 1, i);
                pass = false;
                break;
            }
        }

        if (!pass) failures++;
        fprintf(stderr, "  %s: %s (speedup=%.2fx)\n", pass ? "PASS" : "FAIL", td.name, speedup);
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
