// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "bw_spmm_interface.h"
#include "autotuner.h"
#include "gpu_autotuner.h"
#include "generator.h"
#include "preprocessing.h"
#include "verification.h"
#include "hpc_logger.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cstring>
#include <cassert>
#include <map>
#include <sstream>

// Helper to efficiently slice a host matrix (Deep Copy)
static HostMatrix slice_matrix_rows(const HostMatrix& src, row_idx_t start, row_idx_t count) {
    HostMatrix mat;
    mat.n_rows = count;
    mat.n_cols = src.n_cols;
    mat.rows.resize(count);
    
    #pragma omp parallel for schedule(static)
    for (row_idx_t i = 0; i < count; ++i) {
        if (start + i < src.rows.size()) {
            mat.rows[i] = src.rows[start + i];
        }
    }
    return mat;
}

static size_t estimate_host_matrix_size(const HostMatrix& m) {
    size_t overhead = m.rows.size() * sizeof(std::vector<idx_t>);
    size_t data = 0;
    for(const auto& r : m.rows) data += r.size() * sizeof(idx_t);
    return overhead + data;
}

// =================================================================================
// Constructor & Initialization
// =================================================================================

BlockWiedemannSpMM::BlockWiedemannSpMM(const HostMatrix& raw_A, idx_t padded_n_cols, bool sort_rows, bool sort_cols) {
    logical_n_cols_ = (padded_n_cols == 0) ? raw_A.n_cols : std::max(raw_A.n_cols, padded_n_cols);
    
    total_nnz_ = 0;
    for(const auto& r : raw_A.rows) total_nnz_ += r.size();

    // Compute Permutations
    if (sort_rows) {
        P_row_ = compute_density_permutation(raw_A, logical_n_cols_, false);
    } else {
        P_row_.map.resize(raw_A.n_rows); P_row_.inv.resize(raw_A.n_rows);
        std::iota(P_row_.map.begin(), P_row_.map.end(), 0);
        std::iota(P_row_.inv.begin(), P_row_.inv.end(), 0);
    }

    if (sort_cols) {
        P_col_ = compute_density_permutation(raw_A, logical_n_cols_, true);
    } else {
        P_col_.map.resize(logical_n_cols_); P_col_.inv.resize(logical_n_cols_);
        std::iota(P_col_.map.begin(), P_col_.map.end(), 0);
        std::iota(P_col_.inv.begin(), P_col_.inv.end(), 0);
    }

    // Construct Matrices
    mat_A_.host_data = apply_permutation_explicit(raw_A, logical_n_cols_, P_row_, P_col_);
    mat_A_.host_mem_usage = estimate_host_matrix_size(mat_A_.host_data);
    mat_A_.device_engine = std::make_unique<OptimizedSpMM>();

    mat_AT_.host_data = MatrixGenerator::transpose(mat_A_.host_data);
    mat_AT_.host_mem_usage = estimate_host_matrix_size(mat_AT_.host_data);
    mat_AT_.device_engine = std::make_unique<OptimizedSpMM>();

    print_memory_statistics();
}

BlockWiedemannSpMM::~BlockWiedemannSpMM() {
    unload_A_from_device();
    unload_AT_from_device();
    if (P_row_.d_map) cudaFree(P_row_.d_map);
    if (P_row_.d_inv) cudaFree(P_row_.d_inv);
    if (P_col_.d_map) cudaFree(P_col_.d_map);
    if (P_col_.d_inv) cudaFree(P_col_.d_inv);
}

BlockWiedemannSpMM::PermutationMap BlockWiedemannSpMM::compute_density_permutation(const HostMatrix& m, idx_t n_cols, bool transpose_view) {
    PermutationMap pm;
    size_t size = transpose_view ? n_cols : m.n_rows;
    pm.map.resize(size);
    std::iota(pm.map.begin(), pm.map.end(), 0);

    if (!transpose_view) {
        std::sort(pm.map.begin(), pm.map.end(), [&](row_idx_t a, row_idx_t b) { return m.rows[a].size() > m.rows[b].size(); });
    } else {
        std::vector<size_t> col_counts(n_cols, 0);
        #pragma omp parallel
        {
            std::vector<size_t> local_counts(n_cols, 0);
            #pragma omp for
            for(size_t r=0; r<m.n_rows; ++r) for(idx_t c : m.rows[r]) if(c < n_cols) local_counts[c]++;
            #pragma omp critical
            { for(size_t i=0; i<n_cols; ++i) col_counts[i] += local_counts[i]; }
        }
        std::sort(pm.map.begin(), pm.map.end(), [&](row_idx_t a, row_idx_t b) { return col_counts[a] > col_counts[b]; });
    }
    pm.inv.resize(size);
    #pragma omp parallel for
    for(size_t i=0; i<size; ++i) pm.inv[pm.map[i]] = i;
    return pm;
}

HostMatrix BlockWiedemannSpMM::apply_permutation_explicit(const HostMatrix& src, idx_t padded_cols, const PermutationMap& p_row, const PermutationMap& p_col) {
    HostMatrix out;
    out.n_rows = src.n_rows;
    out.n_cols = padded_cols;
    out.rows.resize(src.n_rows);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (row_idx_t i = 0; i < src.n_rows; ++i) {
        row_idx_t old_row_idx = p_row.map[i];
        if (old_row_idx >= src.rows.size()) continue;
        const auto& old_row = src.rows[old_row_idx];
        out.rows[i].reserve(old_row.size());
        for (idx_t old_c : old_row) if (old_c < p_col.inv.size()) out.rows[i].push_back(p_col.inv[old_c]);
        std::sort(out.rows[i].begin(), out.rows[i].end());
    }
    return out;
}

// =================================================================================
// Setup & Hybrid Construction
// =================================================================================

void BlockWiedemannSpMM::setup_hybrid_path(ManagedMatrix& mat, int bit_width) {
    int m4rm_rows = 8;
    HostMatrix h_m4rm = slice_matrix_rows(mat.host_data, 0, m4rm_rows);
    mat.hybrid_data.m4rm_ctx = MatrixPreprocessor::preprocess_m4rm(h_m4rm, m4rm_rows);

    row_idx_t sparse_start = m4rm_rows;
    row_idx_t sparse_count = mat.host_data.n_rows - m4rm_rows;
    HostMatrix h_sparse = slice_matrix_rows(mat.host_data, sparse_start, sparse_count);

    idx_t dense_lim = 0; 
    row_idx_t heavy_lim = 1024;
    if (heavy_lim > h_sparse.n_rows) heavy_lim = h_sparse.n_rows;

    CompressedMatrix comp_sparse = MatrixPreprocessor::preprocess(
        h_sparse, dense_lim, heavy_lim, false, 
        0.90f, 256
    );
    
    mat.hybrid_data.sparse_mat = SpMMKernels::allocate_and_copy(comp_sparse);
    
    mat.hybrid_config = SpMMConfig();
    mat.hybrid_config.vector_width_bits = bit_width;
    mat.hybrid_config.enable_sparse = true;
    mat.hybrid_config.enable_sparse_tiled_coo = true;
    mat.hybrid_config.enable_sparse_pfor = true; 
    mat.hybrid_config.pfor_exception_threshold = 0.90f;
    mat.hybrid_config.tiled_row_block_size = 256;
    mat.hybrid_config.enable_m4rm = true;
    mat.hybrid_config.enable_heavy_rows = true;
    mat.hybrid_config.enable_heavy_tiled_coo = true; 

    mat.use_hybrid = true;
}

void BlockWiedemannSpMM::free_hybrid_path(ManagedMatrix& mat) {
    if (mat.use_hybrid) {
        free_m4rm_context(mat.hybrid_data.m4rm_ctx);
        SpMMKernels::free_matrix(mat.hybrid_data.sparse_mat);
        mat.use_hybrid = false;
    }
}

void BlockWiedemannSpMM::load_A_to_device(int bit_width) {
    if (mat_A_.is_on_device) return;
    
    size_t f, t; cudaMemGetInfo(&f, &t); size_t start_mem = f;
    
    mat_A_.device_engine->compile(mat_A_.host_data, mat_A_.plan);
    mat_A_.is_on_device = true;

    cudaMemGetInfo(&f, &t);
    mat_A_.device_mem_usage = start_mem - f;
}

void BlockWiedemannSpMM::unload_A_from_device() {
    if (!mat_A_.is_on_device) return;
    mat_A_.device_engine->free_resources();
    mat_A_.is_on_device = false;
    mat_A_.device_mem_usage = 0;
}

void BlockWiedemannSpMM::load_AT_to_device(int bit_width) {
    if (mat_AT_.is_on_device) return;
    
    size_t f, t; cudaMemGetInfo(&f, &t); size_t start_mem = f;

    // FIX: Use the tuned engine (like A) instead of hardcoded hybrid path
    mat_AT_.device_engine->compile(mat_AT_.host_data, mat_AT_.plan);
    mat_AT_.is_on_device = true;

    cudaMemGetInfo(&f, &t);
    mat_AT_.device_mem_usage = start_mem - f;
}

void BlockWiedemannSpMM::unload_AT_from_device() {
    if (!mat_AT_.is_on_device) return;

    // FIX: Cleanup based on how it was loaded
    if (mat_AT_.use_hybrid) {
        free_hybrid_path(mat_AT_);
    } else {
        mat_AT_.device_engine->free_resources();
    }
    
    mat_AT_.is_on_device = false;
    mat_AT_.device_mem_usage = 0;
}

// =================================================================================
// Introspection & Logging
// =================================================================================

std::string BlockWiedemannSpMM::get_kernel_summary(bool is_AT) const {
    const ManagedMatrix& mat = is_AT ? mat_AT_ : mat_A_;
    std::stringstream ss;
    
    if (is_AT && mat.use_hybrid) {
        ss << "M4RM (8 Rows), Hybrid Sparse PFor/Tiled (" << (mat.host_data.n_rows - 8) << " Rows)";
        return ss.str();
    }
    
    if (!mat.is_on_device) return "Not Loaded";

    std::map<std::string, row_idx_t> counts;
    for (const auto& seg : mat.plan.segments) {
        counts[seg.best_config.name] += (seg.end_row - seg.start_row);
    }
    
    bool first = true;
    for (const auto& pair : counts) {
        if (!first) ss << ", ";
        ss << pair.first << " (" << pair.second << " Rows)";
        first = false;
    }
    return ss.str();
}

size_t BlockWiedemannSpMM::get_device_memory_bytes(bool is_AT) const {
    return is_AT ? mat_AT_.device_mem_usage : mat_A_.device_mem_usage;
}

size_t BlockWiedemannSpMM::get_host_memory_bytes(bool is_AT) const {
    return is_AT ? mat_AT_.host_mem_usage : mat_A_.host_mem_usage;
}

// =================================================================================
// Verification
// =================================================================================

// --- Verify A ---
bool BlockWiedemannSpMM::verify_A(int bit_width) {
    LOG(LOG_STATS) << "[BwSpmmInterface] Verifying x -> Ax (BitWidth=" << bit_width << ")...";
    bool passed = true;
    
    {
        size_t byte_width = bit_width / 8;
        size_t n_in = logical_n_cols_;
        size_t n_out = mat_A_.host_data.n_rows;
        
        std::vector<uint8_t> h_in(n_in * byte_width);
        std::vector<uint8_t> h_out_gpu(n_out * byte_width);
        std::vector<uint8_t> h_out_ref(n_out * byte_width);
        
        // Randomize input
        for(auto& b : h_in) b = rand() % 256;

        void *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, h_in.size()));
        CUDA_CHECK(cudaMalloc(&d_out, h_out_gpu.size()));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, h_out_gpu.size()));

        // Run GPU
	LOG(LOG_DEBUG_2) << "[BwSpmmInterface] Executing x -> Ax on GPU...";
        load_A_to_device(bit_width);
        execute_A(d_out, d_in);
        CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, h_out_gpu.size(), cudaMemcpyDeviceToHost));
        
        // Run CPU Ref (Identity permutation because mat_A_.host_data is already the permuted matrix used on GPU)
	LOG(LOG_DEBUG_2) << "[BwSpmmInterface] Executing x -> Ax on CPU...";
        std::vector<row_idx_t> identity(n_out);
        std::iota(identity.begin(), identity.end(), 0);
        Verification::cpu_spmm_gf2(mat_A_.host_data, h_in.data(), h_out_ref.data(), identity, bit_width);

        // Compare
        if (!Verification::check_results(h_out_gpu.data(), h_out_ref.data(), n_out, bit_width)) {
            LOG(LOG_ERROR_CRITICAL) << "[BwSpmmInterface] Matrix A Verification FAILED!";
            passed = false;
        } else {
            LOG(LOG_INFO) << "[BwSpmmInterface] Matrix A Verification PASSED.";
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        unload_A_from_device();
    }

    return passed;
}

// --- Verify AT ---
bool BlockWiedemannSpMM::verify_AT(int bit_width) {
    LOG(LOG_STATS) << "[BwSpmmInterface] Verifying x -> ATx (BitWidth=" << bit_width << ")...";
    bool passed = true;
    {
        size_t byte_width = bit_width / 8;
        size_t n_in = mat_A_.host_data.n_rows; // AT inputs  = A rows
        size_t n_out = logical_n_cols_;        // AT outputs = A cols
        
        std::vector<uint8_t> h_in(n_in * byte_width);
        std::vector<uint8_t> h_out_gpu(n_out * byte_width);
        std::vector<uint8_t> h_out_ref(n_out * byte_width);
        
        for(auto& b : h_in) b = rand() % 256;

        void *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, h_in.size()));
        CUDA_CHECK(cudaMalloc(&d_out, h_out_gpu.size()));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_out, 0, h_out_gpu.size()));

        // Run GPU
	LOG(LOG_DEBUG_2) << "[BwSpmmInterface] Executing x -> ATx on GPU...";
        load_AT_to_device(bit_width);
        execute_AT(d_out, d_in);
        CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, h_out_gpu.size(), cudaMemcpyDeviceToHost));
        
        // Run CPU Ref
	LOG(LOG_DEBUG_2) << "[BwSpmmInterface] Executing x -> ATx on CPU...";
        std::vector<row_idx_t> identity(n_out);
        std::iota(identity.begin(), identity.end(), 0);
        Verification::cpu_spmm_gf2(mat_AT_.host_data, h_in.data(), h_out_ref.data(), identity, bit_width);

        // Compare
        if (!Verification::check_results(h_out_gpu.data(), h_out_ref.data(), n_out, bit_width)) {
            LOG(LOG_ERROR_CRITICAL) << "[BwSpmmInterface] Matrix AT Verification FAILED!";
            passed = false;
        } else {
             LOG(LOG_INFO) << "[BwSpmmInterface] Matrix AT Verification PASSED.";
        }

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        unload_AT_from_device();
    }
    
    return passed;
}

// =================================================================================
// Execution & Benchmarking
// =================================================================================

SpMMPerformanceReport BlockWiedemannSpMM::setup_and_benchmark(
    int bit_width_m, int bit_width_n,
    const SpMMAutoTuneConfig& autotune_cfg) {

    SpMMPerformanceReport report = {};
    report.bit_width_m = bit_width_m;
    report.bit_width_n = bit_width_n;
    int bit_width = bit_width_n;
    size_t byte_width = bit_width / 8;

    const bool gpu_only_enabled = autotune_cfg.gpu_only;
    const auto& gpu_cfg = autotune_cfg.gpu_config;

    // --- Tune A ---
    LOG_INCREMENT_STAGE(10);
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] [BW-Bench] Tuning Architecture for A..." << std::endl;

    if (gpu_only_enabled) {
        try {
            mat_A_.plan = GPUAutoTuner::tune(
                mat_A_.host_data, false, bit_width, gpu_cfg, true);
            LOG(LOG_DEBUG_1) << "[BwSpmmInterface] GPU-only tuning for A succeeded ("
                             << mat_A_.plan.segments.size() << " segments)" << std::endl;
        } catch (const std::exception& e) {
            LOG(LOG_WARNING) << "[BwSpmmInterface] GPU-only tuning for A failed: " << e.what()
                            << ". Falling back to legacy tune_global()." << std::endl;
            mat_A_.plan = SpMMAutoTuner::tune_global(mat_A_.host_data, false, bit_width, true);
        }
    } else {
        mat_A_.plan = SpMMAutoTuner::tune_global(mat_A_.host_data, false, bit_width, true);
    }

    LOG_INCREMENT_STAGE(10);
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] [BW-Bench] Processing A (Standard)..." << std::endl;
    load_A_to_device(bit_width);
    size_t v_bytes_A = mat_A_.host_data.n_cols * byte_width;
    size_t c_bytes_A = mat_A_.host_data.n_rows * byte_width;
    uint64_t *dV_A, *dC_A;
    CUDA_CHECK(cudaMalloc(&dV_A, v_bytes_A));
    CUDA_CHECK(cudaMalloc(&dC_A, c_bytes_A));
    mat_A_.device_engine->tune_execution_strategy(dC_A, dV_A);
    cudaFree(dV_A); cudaFree(dC_A);
    report.time_A_ms = measure_hot_execution(64, v_bytes_A, c_bytes_A, false);
    report.peak_gpu_mem_bytes = std::max(report.peak_gpu_mem_bytes, get_device_memory_bytes(false));

    // --- Tune AT ---
    bit_width = bit_width_m;
    LOG_INCREMENT_STAGE(10);
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] [BW-Bench] Tuning Architecture for AT..." << std::endl;

    if (gpu_only_enabled) {
        try {
            mat_AT_.plan = GPUAutoTuner::tune(
                mat_AT_.host_data, true, bit_width, gpu_cfg, true);
            LOG(LOG_DEBUG_1) << "[BwSpmmInterface] GPU-only tuning for AT succeeded ("
                             << mat_AT_.plan.segments.size() << " segments)" << std::endl;
        } catch (const std::exception& e) {
            LOG(LOG_WARNING) << "[BwSpmmInterface] GPU-only tuning for AT failed: " << e.what()
                            << ". Falling back to legacy tune_global()." << std::endl;
            mat_AT_.plan = SpMMAutoTuner::tune_global(mat_AT_.host_data, true, bit_width, true);
        }
    } else {
        mat_AT_.plan = SpMMAutoTuner::tune_global(mat_AT_.host_data, true, bit_width, true);
    }
    
    LOG_INCREMENT_STAGE(10);
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] [BW-Bench] Processing AT..." << std::endl;
    load_AT_to_device(bit_width);
    size_t v_bytes_AT = mat_AT_.host_data.n_cols * byte_width;
    size_t c_bytes_AT = mat_AT_.host_data.n_rows * byte_width;
    uint64_t *dV_AT, *dC_AT;
    CUDA_CHECK(cudaMalloc(&dV_AT, v_bytes_AT));
    CUDA_CHECK(cudaMalloc(&dC_AT, c_bytes_AT));
    mat_AT_.device_engine->tune_execution_strategy(dC_AT, dV_AT);
    cudaFree(dV_AT); cudaFree(dC_AT);
    report.time_AT_ms = measure_hot_execution(64, v_bytes_AT, c_bytes_AT, true);
    report.peak_gpu_mem_bytes = get_device_memory_bytes(true); 
    unload_AT_from_device();

    LOG_INCREMENT_STAGE(10);
    report.throughput_A_gnnz = (double)total_nnz_ / 1e9 / (report.time_A_ms / 1000.0);
    report.throughput_AT_gnnz = (double)total_nnz_ / 1e9 / (report.time_AT_ms / 1000.0);
    report.ratio_A_over_AT = (report.throughput_AT_gnnz > 0) ? (report.throughput_A_gnnz / report.throughput_AT_gnnz) : 0.0;
    report.host_mem_bytes = mat_A_.host_mem_usage + mat_AT_.host_mem_usage;

    {
        LOG(LOG_STATS) << "[BwSpmmInterface] --------------------------------------------------------";
        LOG(LOG_STATS) << "[BwSpmmInterface]  [BW-Result] A  Time: " << std::fixed << std::setprecision(3)
			 << report.time_A_ms << " ms | "
			 << report.throughput_A_gnnz << " GNNz/s with " << bit_width_n << " bits";
        LOG(LOG_STATS) << "[BwSpmmInterface]  [BW-Result] AT Time: " << report.time_AT_ms << " ms | "
			 << report.throughput_AT_gnnz << " GNNz/s with " << bit_width_m << " bits";
        LOG(LOG_STATS) << "[BwSpmmInterface]  [BW-Result] Ratio (A/AT): " << std::setprecision(4)
			 << report.ratio_A_over_AT;
        LOG(LOG_STATS) << "[BwSpmmInterface] --------------------------------------------------------";
        print_memory_statistics();
    }
    return report;
}

double BlockWiedemannSpMM::measure_hot_execution(int iterations, size_t v_bytes, size_t c_bytes, bool is_AT) {
    uint64_t *d_V, *d_C;
    CUDA_CHECK(cudaMalloc(&d_V, v_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemset(d_V, 0xAA, v_bytes));
    CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
    
    if (is_AT) execute_AT(d_C, d_V);
    else execute_A(d_C, d_V);

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        if (is_AT) execute_AT(d_C, d_V);
        else execute_A(d_C, d_V);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaFree(d_V); cudaFree(d_C);
    return std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iterations;
}

void BlockWiedemannSpMM::execute_A(void* d_C, const void* d_V) {
    if (!mat_A_.is_on_device) {
        if (current_bit_width_ == 0) throw std::runtime_error("[BW-Error] Cannot auto-load A.");
        load_A_to_device(current_bit_width_);
    }
    mat_A_.device_engine->execute(d_C, d_V);
    cudaDeviceSynchronize();
}

void BlockWiedemannSpMM::execute_AT(void* d_C, const void* d_V) {
    if (!mat_AT_.is_on_device) {
        if (current_bit_width_ == 0) throw std::runtime_error("[BW-Error] Cannot auto-load AT.");
        load_AT_to_device(current_bit_width_);
    }

    if (mat_AT_.use_hybrid) {
        SpMMKernels::run_hybrid(mat_AT_.hybrid_data, d_C, d_V, mat_AT_.hybrid_config);
        cudaDeviceSynchronize();
    } else {
        mat_AT_.device_engine->execute(d_C, d_V);
        cudaDeviceSynchronize();
    }
}

void BlockWiedemannSpMM::execute_A(void* d_C, const void* d_V, cudaStream_t stream) {
    assert(mat_A_.is_on_device && "execute_A(stream): matrix must be pre-loaded for stream-aware path");
    if (!mat_A_.is_on_device) {
        if (current_bit_width_ == 0) throw std::runtime_error("[BW-Error] Cannot auto-load A.");
        load_A_to_device(current_bit_width_);
    }
    mat_A_.device_engine->execute(d_C, d_V, stream);
}

void BlockWiedemannSpMM::execute_AT(void* d_C, const void* d_V, cudaStream_t stream) {
    assert(mat_AT_.is_on_device && "execute_AT(stream): matrix must be pre-loaded for stream-aware path");
    if (!mat_AT_.is_on_device) {
        if (current_bit_width_ == 0) throw std::runtime_error("[BW-Error] Cannot auto-load AT.");
        load_AT_to_device(current_bit_width_);
    }

    if (mat_AT_.use_hybrid) {
        SpMMKernels::run_hybrid(mat_AT_.hybrid_data, d_C, d_V, mat_AT_.hybrid_config, stream);
    } else {
        mat_AT_.device_engine->execute(d_C, d_V, stream);
    }
}

// =================================================================================
// Vector Transformations (CPU)
// =================================================================================

void BlockWiedemannSpMM::apply_permutation_cpu(const void* src, void* dst, size_t n, size_t elem_size, const std::vector<row_idx_t>& mapping) const {
    const char* s = (const char*)src;
    char* d = (char*)dst;
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        if (i < mapping.size()) {
            std::memcpy(d + i * elem_size, s + mapping[i] * elem_size, elem_size);
        }
    }
}

void BlockWiedemannSpMM::preprocess_domain_A(const void* h_in, void* h_out, size_t num_elements, int bit_width) const {
    apply_permutation_cpu(h_in, h_out, num_elements, bit_width/8, P_col_.map);
}
void BlockWiedemannSpMM::postprocess_domain_AT(const void* h_in, void* h_out, size_t num_elements, int bit_width) const {
    apply_permutation_cpu(h_in, h_out, num_elements, bit_width/8, P_col_.inv);
}
void BlockWiedemannSpMM::preprocess_domain_AT(const void* h_in, void* h_out, size_t num_elements, int bit_width) const {
    apply_permutation_cpu(h_in, h_out, num_elements, bit_width/8, P_row_.map);
}
void BlockWiedemannSpMM::postprocess_domain_A(const void* h_in, void* h_out, size_t num_elements, int bit_width) const {
    apply_permutation_cpu(h_in, h_out, num_elements, bit_width/8, P_row_.inv);
}

// =================================================================================
// Vector Transformations (GPU)
// =================================================================================
void BlockWiedemannSpMM::permute_vec_rows_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width) {
    if (!P_row_.d_map) {
        CUDA_CHECK(cudaMalloc(&P_row_.d_map, P_row_.map.size() * 4));
        CUDA_CHECK(cudaMemcpy(P_row_.d_map, P_row_.map.data(), P_row_.map.size() * 4, cudaMemcpyHostToDevice));
    }
    gpu_permute_vector(d_in, d_out, P_row_.d_map, num_elements, bit_width);
}

void BlockWiedemannSpMM::unpermute_vec_rows_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width) {
    if (!P_row_.d_inv) {
        CUDA_CHECK(cudaMalloc(&P_row_.d_inv, P_row_.inv.size() * 4));
        CUDA_CHECK(cudaMemcpy(P_row_.d_inv, P_row_.inv.data(), P_row_.inv.size() * 4, cudaMemcpyHostToDevice));
    }
    gpu_permute_vector(d_in, d_out, P_row_.d_inv, num_elements, bit_width);
}

void BlockWiedemannSpMM::permute_vec_cols_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width) {
    if (!P_col_.d_map) {
        CUDA_CHECK(cudaMalloc(&P_col_.d_map, P_col_.map.size() * 4));
        CUDA_CHECK(cudaMemcpy(P_col_.d_map, P_col_.map.data(), P_col_.map.size() * 4, cudaMemcpyHostToDevice));
    }
    gpu_permute_vector(d_in, d_out, P_col_.d_map, num_elements, bit_width);
}

void BlockWiedemannSpMM::unpermute_vec_cols_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width) {
    if (!P_col_.d_inv) {
        CUDA_CHECK(cudaMalloc(&P_col_.d_inv, P_col_.inv.size() * 4));
        CUDA_CHECK(cudaMemcpy(P_col_.d_inv, P_col_.inv.data(), P_col_.inv.size() * 4, cudaMemcpyHostToDevice));
    }
    gpu_permute_vector(d_in, d_out, P_col_.d_inv, num_elements, bit_width);
}

void BlockWiedemannSpMM::host_matrix_to_csr_arrays(const HostMatrix& m, std::vector<uint32_t>& row_ptr, std::vector<uint32_t>& col_ind) {
    row_ptr.resize(m.n_rows + 1);
    size_t nnz = 0;
    for(const auto& r : m.rows) nnz += r.size();
    col_ind.reserve(nnz);
    row_ptr[0] = 0;
    for(size_t i=0; i<m.n_rows; ++i) {
        col_ind.insert(col_ind.end(), m.rows[i].begin(), m.rows[i].end());
        row_ptr[i+1] = (uint32_t)col_ind.size();
    }
}

void BlockWiedemannSpMM::print_memory_statistics() const {
    // ... implementation preserved ...
    size_t perm_size = (P_row_.map.size() + P_row_.inv.size() + P_col_.map.size() + P_col_.inv.size()) * sizeof(row_idx_t);
    size_t a_host = mat_A_.host_mem_usage;
    size_t at_host = mat_AT_.host_mem_usage;
    size_t orig_a_est = (P_row_.map.size() + 1) * sizeof(uint32_t) + total_nnz_ * sizeof(uint32_t);

    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] === [BW Memory Statistics] ===" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  [Original] Matrix A (Ref):   " << format_bytes(orig_a_est) << " (Estimate)" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  [Internal] Tilde A (Host):   " << format_bytes(a_host) << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  [Internal] Tilde AT (Host):  " << format_bytes(at_host) << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  [Internal] Permutations:     " << format_bytes(perm_size) << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  ----------------------------" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  Total Internal Host Alloc:   " << format_bytes(a_host + at_host + perm_size) << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  [Device Status]" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  Matrix A:  " << (mat_A_.is_on_device ? "GPU Resident" : "Unloaded") << " (" << format_bytes(mat_A_.device_mem_usage) << ")" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface]  Matrix AT: " << (mat_AT_.is_on_device ? "GPU Resident" : "Unloaded") << " (" << format_bytes(mat_AT_.device_mem_usage) << ")" << std::endl;
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] ==============================" << std::endl;
}

extern uint32_t* gpu_compute_density_permutation(const uint32_t* d_row_offsets, size_t n_rows, bool ascending);
extern void gpu_transpose_csr(size_t n_rows, size_t n_cols, size_t nnz, const uint32_t* d_csr_row_ptr, const uint32_t* d_csr_col_ind, uint32_t** d_out_row_ptr, uint32_t** d_out_col_ind);

bool BlockWiedemannSpMM::benchmark_preprocessing() {
    LOG(LOG_DEBUG_1) << "[BwSpmmInterface] [BW-Bench] Benchmarking Preprocessing Primitives..." << std::endl;
    
    std::vector<uint32_t> row_ptr, col_ind;
    host_matrix_to_csr_arrays(mat_A_.host_data, row_ptr, col_ind);
    size_t n_rows = mat_A_.host_data.n_rows;
    
    LOG_INCREMENT_STAGE(10);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_cpu = compute_density_permutation(mat_A_.host_data, logical_n_cols_, false);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu_sort = std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    
    LOG_INCREMENT_STAGE(10);
    uint32_t *d_ptr;
    cudaMalloc(&d_ptr, row_ptr.size() * sizeof(uint32_t));
    cudaMemcpy(d_ptr, row_ptr.data(), row_ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    uint32_t* d_perm = gpu_compute_density_permutation(d_ptr, n_rows, false);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_gpu_sort = std::chrono::duration<double>(t3 - t2).count() * 1000.0;
    cudaFree(d_ptr); cudaFree(d_perm);

    LOG_INCREMENT_STAGE(10);
    auto t4 = std::chrono::high_resolution_clock::now();
    HostMatrix ht = MatrixGenerator::transpose(mat_A_.host_data);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_cpu_trans = std::chrono::duration<double>(t5 - t4).count() * 1000.0;

    LOG_INCREMENT_STAGE(10);
    uint32_t *d_c_ind, *d_out_r, *d_out_c;
    cudaMalloc(&d_ptr, row_ptr.size() * sizeof(uint32_t));
    cudaMalloc(&d_c_ind, col_ind.size() * sizeof(uint32_t));
    cudaMemcpy(d_ptr, row_ptr.data(), row_ptr.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_ind, col_ind.data(), col_ind.size() * 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();
    gpu_transpose_csr(n_rows, logical_n_cols_, col_ind.size(), d_ptr, d_c_ind, &d_out_r, &d_out_c);
    cudaDeviceSynchronize();
    auto t7 = std::chrono::high_resolution_clock::now();
    double ms_gpu_trans = std::chrono::duration<double>(t7 - t6).count() * 1000.0;
    cudaFree(d_ptr); cudaFree(d_c_ind); cudaFree(d_out_r); cudaFree(d_out_c);

    LOG_INCREMENT_STAGE(10);
    {
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   -------------------------------------------------" << std::endl;
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   Operation      | CPU (ms)   | GPU (ms)   | Speedup" << std::endl;
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   -------------------------------------------------" << std::endl;
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   Density Sort   | " << std::setw(10) << ms_cpu_sort << " | " << std::setw(10) << ms_gpu_sort << " | " << ms_cpu_sort/ms_gpu_sort << "x" << std::endl;
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   Transpose      | " << std::setw(10) << ms_cpu_trans << " | " << std::setw(10) << ms_gpu_trans << " | " << ms_cpu_trans/ms_gpu_trans << "x" << std::endl;
        LOG(LOG_DEBUG_1) << "[BwSpmmInterface]   -------------------------------------------------" << std::endl;
    }
    return (ms_gpu_trans < ms_cpu_trans);
}
