// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

// test_golden_regression.cu — Structural regression tests on GPU autotuner plans.
// No binary golden files — generates matrices from fixed seeds and asserts
// structural properties of the resulting execution plans.
// No test framework; standalone main() with assert() and PASS/FAIL logging.

#include "common.h"
#include "gpu_autotuner.h"
#include "generator.h"
#include "hpc_logger.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Matrix generation with fixed seed
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

/// Validate structural properties of an execution plan.
static bool validate_plan(const ExecutionPlan& plan, row_idx_t expected_rows,
                          const char* label) {
    bool pass = true;

    // Non-empty plan
    if (plan.segments.empty()) {
        fprintf(stderr, "  FAIL [%s]: Plan has no segments\n", label);
        return false;
    }

    // Plan covers all rows [0, expected_rows) with no gaps
    if (plan.segments.front().start_row != 0) {
        fprintf(stderr, "  FAIL [%s]: First segment starts at %u, expected 0\n",
                label, plan.segments.front().start_row);
        pass = false;
    }
    if (plan.segments.back().end_row != expected_rows) {
        fprintf(stderr, "  FAIL [%s]: Last segment ends at %u, expected %u\n",
                label, plan.segments.back().end_row, expected_rows);
        pass = false;
    }

    // No gaps or overlaps
    for (size_t i = 1; i < plan.segments.size(); i++) {
        if (plan.segments[i].start_row != plan.segments[i - 1].end_row) {
            fprintf(stderr, "  FAIL [%s]: Gap/overlap at segment %zu: prev_end=%u next_start=%u\n",
                    label, i, plan.segments[i - 1].end_row, plan.segments[i].start_row);
            pass = false;
            break;
        }
    }

    // Segment count in reasonable range (exponential blocks → ~19 max for 10k rows)
    size_t n_seg = plan.segments.size();
    if (n_seg < 1 || n_seg > 50) {
        fprintf(stderr, "  FAIL [%s]: Segment count %zu out of expected range [1, 50]\n",
                label, n_seg);
        pass = false;
    }

    // All segment throughputs > 0
    for (size_t i = 0; i < plan.segments.size(); i++) {
        if (plan.segments[i].measured_throughput_gnnz <= 0) {
            fprintf(stderr, "  FAIL [%s]: Segment %zu has non-positive throughput %.4f\n",
                    label, i, plan.segments[i].measured_throughput_gnnz);
            pass = false;
        }
    }

    // Estimated total time sanity
    if (plan.estimated_total_time_ms <= 0 || plan.estimated_total_time_ms > 1000) {
        fprintf(stderr, "  FAIL [%s]: Estimated time %.4f ms out of range (0, 1000]\n",
                label, plan.estimated_total_time_ms);
        pass = false;
    }

    // First segment should be M4RM for density-sorted matrices (dense head)
    if (plan.segments.front().best_config.id == KernelID::M4RM) {
        fprintf(stderr, "  INFO [%s]: First segment is M4RM (expected for dense head)\n", label);
    }

    return pass;
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
    constexpr uint64_t GOLDEN_SEED = 12345;

    GPUAutoTuner::Config gpu_cfg;
    gpu_cfg.allow_cpu_fallback = true;

    // --- Test 1: Non-transposed plan ---
    {
        fprintf(stderr, "[test_golden_regression] Test 1: 10k x 10k, seed=%lu, A\n",
                (unsigned long)GOLDEN_SEED);
        HostMatrix raw = generate_test_matrix(10000, 10000, 50000, GOLDEN_SEED);
        HostMatrix mat = sort_by_density(raw);

        ExecutionPlan plan = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);

        total++;
        bool pass = validate_plan(plan, mat.n_rows, "A");
        if (!pass) failures++;
        fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");

        // Log plan details
        fprintf(stderr, "  Plan: %zu segments, est_time=%.4f ms\n",
                plan.segments.size(), plan.estimated_total_time_ms);
        for (size_t i = 0; i < plan.segments.size(); i++) {
            auto& s = plan.segments[i];
            fprintf(stderr, "    [%u, %u) %s throughput=%.4f GNNz/s\n",
                    s.start_row, s.end_row, s.best_config.name.c_str(),
                    s.measured_throughput_gnnz);
        }
    }

    // --- Test 2: Transposed plan ---
    {
        fprintf(stderr, "[test_golden_regression] Test 2: 10k x 10k, seed=%lu, AT\n",
                (unsigned long)GOLDEN_SEED);
        HostMatrix raw = generate_test_matrix(10000, 10000, 50000, GOLDEN_SEED);
        HostMatrix matT = MatrixGenerator::transpose(raw);
        HostMatrix matT_sorted = sort_by_density(matT);

        ExecutionPlan plan = GPUAutoTuner::tune(matT_sorted, true, VEC_WIDTH, gpu_cfg, false);

        total++;
        bool pass = validate_plan(plan, matT_sorted.n_rows, "AT");
        if (!pass) failures++;
        fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");
    }

    // --- Test 3: Determinism — same seed, same plan structure ---
    {
        fprintf(stderr, "[test_golden_regression] Test 3: Determinism check\n");
        HostMatrix raw = generate_test_matrix(10000, 10000, 50000, GOLDEN_SEED);
        HostMatrix mat = sort_by_density(raw);

        ExecutionPlan plan1 = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);
        ExecutionPlan plan2 = GPUAutoTuner::tune(mat, false, VEC_WIDTH, gpu_cfg, false);

        total++;
        bool pass = true;

        if (plan1.segments.size() != plan2.segments.size()) {
            fprintf(stderr, "  FAIL: Segment count differs: %zu vs %zu\n",
                    plan1.segments.size(), plan2.segments.size());
            pass = false;
        } else {
            for (size_t i = 0; i < plan1.segments.size(); i++) {
                if (plan1.segments[i].start_row != plan2.segments[i].start_row ||
                    plan1.segments[i].end_row != plan2.segments[i].end_row ||
                    plan1.segments[i].best_config.id != plan2.segments[i].best_config.id) {
                    fprintf(stderr, "  FAIL: Segment %zu differs between runs\n", i);
                    pass = false;
                    break;
                }
            }
        }
        if (!pass) failures++;
        fprintf(stderr, "  %s\n", pass ? "PASS" : "FAIL");
    }

    fprintf(stderr, "\n========================================\n");
    if (failures == 0) {
        fprintf(stderr, "ALL TESTS PASSED (%d tests)\n", total);
    } else {
        fprintf(stderr, "FAILED: %d tests (out of %d)\n", failures, total);
    }
    return failures > 0 ? 1 : 0;
}
