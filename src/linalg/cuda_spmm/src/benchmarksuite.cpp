// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "benchmarksuite.h"
#include "generator.h"
#include "preprocessing.h"
#include "reordering.h"
#include "kernels.h"
#include "m4rm_data.h"
#include "verification.h"
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring> // For memset

#include "hpc_logger.h"

// =================================================================================
// Performance Helpers
// =================================================================================

struct PerfResult {
    double time_ms;
    double gnnz_s; // Giga NNZ / s
};

// Helper to efficiently slice a host matrix (Deep Copy)
static HostMatrix slice_matrix_simple(const HostMatrix& src, row_idx_t start, row_idx_t count) {
    HostMatrix mat;
    mat.n_rows = count;
    mat.n_cols = src.n_cols;
    mat.rows.resize(count);
    
    // Parallel copy for speed on host
    #pragma omp parallel for schedule(static)
    for (row_idx_t i = 0; i < count; ++i) {
        if (start + i < src.rows.size()) {
            mat.rows[i] = src.rows[start + i];
        }
    }
    return mat;
}

// Unified Kernel Measurement Helper (Handles alloc/run/measure/free to save VRAM)
static PerfResult measure_kernel_perf(
    const DeviceMatrix& d_mat,
    size_t n_rows, // Output rows
    size_t n_cols, // Input cols
    SpMMConfig config
) {
    // Dynamic size calculation based on bit width
    size_t width_bytes = config.vector_width_bits / 8;
    size_t v_size_bytes = n_cols * width_bytes;
    size_t c_size_bytes = n_rows * width_bytes;

    // Use byte pointers for allocation to ensure correct sizing
    uint8_t *d_V_bytes, *d_C_bytes;
    CUDA_CHECK(cudaMalloc(&d_V_bytes, v_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_bytes, c_size_bytes));
    
    // Generate a random vector for benchmarking instead of a fixed pattern
    MatrixGenerator local_gen(999); // Fixed seed for reproducible benchmarks
    std::vector<uint8_t> h_V = local_gen.generate_random_vector(n_cols, config.vector_width_bits, true);
    CUDA_CHECK(cudaMemcpy(d_V_bytes, h_V.data(), v_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_bytes, 0, c_size_bytes));

    // Cast to uint64_t* for kernel interface compatibility
    uint64_t* d_V = (uint64_t*)d_V_bytes;
    uint64_t* d_C = (uint64_t*)d_C_bytes;

    // Warmup
    SpMMKernels::run_spmm(d_mat, d_C, d_V, config);
    cudaDeviceSynchronize();

    int iterations = 20;
    if (n_rows > 500000) iterations = 10;
    
    double total_ms = 0;

    for(int i=0; i<iterations; ++i) {
        // FLUSH L2 to ensure Matrix A is fetched from DRAM
        SpMMKernels::flush_l2_cache();
        
        auto t0 = std::chrono::high_resolution_clock::now();
        SpMMKernels::run_spmm(d_mat, d_C, d_V, config);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    }

    double avg_ms = total_ms / iterations;
    
    cudaFree(d_V_bytes);
    cudaFree(d_C_bytes);

    return {avg_ms, 0.0};
}

// =================================================================================
// BenchmarkSuite Implementation
// =================================================================================

void BenchmarkSuite::run_suite(
    const std::vector<MatrixParameter>& params,
    const std::vector<BenchmarkScenario>& scenarios,
    bool enable_layer_analysis
) {
    for (const auto& param : params) {
        LOG(LOG_INFO) << "##########################################################";
        LOG(LOG_INFO) << " Processing Matrix Config: "
                      << param.n_rows << "x" << param.n_cols
                      << " (alpha=" << param.alpha << ")";
        LOG(LOG_INFO) << "##########################################################";
        
        process_single_matrix(param, scenarios, enable_layer_analysis);
    }
    
    // Cleanup Global Buffer
    SpMMKernels::cleanup_l2_cache();
}

void BenchmarkSuite::process_single_matrix(
    const MatrixParameter& param,
    const std::vector<BenchmarkScenario>& scenarios,
    bool enable_layer_analysis
) {
    // 1. Matrix Generation
    MatrixGenerator gen(param.seed);
    gen.generate_factor_base(param.n_cols);
    
    LOG(LOG_INFO) << "[Generator] Creating Matrix A...";
    HostMatrix A = gen.generate_matrix(param.n_rows, param.alpha);

    LOG(LOG_INFO) << "[Generator] Creating Matrix AT (Transpose)...";
    HostMatrix AT = MatrixGenerator::transpose(A);

    // Setup Limits
    row_idx_t heavy_limit = 1024; 
    if (heavy_limit > AT.n_rows) heavy_limit = AT.n_rows;
    idx_t dense_limit_A = std::min((idx_t)64, param.n_cols);

    // 2. Run Scenarios
    for (const auto& scen : scenarios) {
        const HostMatrix* source_mat_ptr = nullptr;
        
        idx_t dense_lim = 0;
        row_idx_t heavy_lim = 0;

        // Determine Source (A vs AT)
        if (!scen.use_transpose) {
            source_mat_ptr = &A;
            dense_lim = dense_limit_A;
            heavy_lim = 0;
        } else {
            source_mat_ptr = &AT;
            dense_lim = 0;
            heavy_lim = heavy_limit;
        }

        // --- A. Hybrid M4RM Execution ---
        if (scen.config.enable_m4rm) {
             LOG(LOG_INFO) << "----------------------------------------------------------";
             LOG(LOG_INFO) << "   Benchmark Phase: " << scen.name << " (Hybrid M4RM)";
             LOG(LOG_INFO) << "----------------------------------------------------------";
             
             int m4rm_rows = scen.config.m4rm_rows;
             
             // 1. Prepare M4RM Context (First 8 rows)
             M4RMContext ctx = MatrixPreprocessor::preprocess_m4rm(*source_mat_ptr, m4rm_rows);
             
             // 2. Prepare Sparse Context (Rows 8 to End)
             row_idx_t sparse_start = m4rm_rows;
             row_idx_t sparse_count = source_mat_ptr->n_rows - m4rm_rows;
             HostMatrix h_sparse = slice_matrix_simple(*source_mat_ptr, sparse_start, sparse_count);
             
             CompressedMatrix comp_sparse = MatrixPreprocessor::preprocess(
                 h_sparse, dense_lim, heavy_lim, false, 
                 scen.config.pfor_exception_threshold, scen.config.tiled_row_block_size
             );
             
             DeviceMatrix d_sparse = SpMMKernels::allocate_and_copy(comp_sparse);
             HybridDeviceMatrix d_hybrid = { ctx, d_sparse };
             
             // 3. Setup Benchmark Resources
             size_t width_bytes = scen.config.vector_width_bits / 8;
             size_t v_size_bytes = source_mat_ptr->n_cols * width_bytes;
             size_t c_size_bytes = source_mat_ptr->n_rows * width_bytes;
             
             uint8_t *d_V_bytes, *d_C_bytes;
             CUDA_CHECK(cudaMalloc(&d_V_bytes, v_size_bytes));
             CUDA_CHECK(cudaMalloc(&d_C_bytes, c_size_bytes));

             // Generate random vector and copy to device
             std::vector<uint8_t> h_V_rand = gen.generate_random_vector(source_mat_ptr->n_cols, scen.config.vector_width_bits);
             CUDA_CHECK(cudaMemcpy(d_V_bytes, h_V_rand.data(), v_size_bytes, cudaMemcpyHostToDevice));
             CUDA_CHECK(cudaMemset(d_C_bytes, 0, c_size_bytes));

             uint64_t* d_V = (uint64_t*)d_V_bytes;
             uint64_t* d_C = (uint64_t*)d_C_bytes;

             // 4. Verification
             SpMMKernels::run_hybrid(d_hybrid, d_C, d_V, scen.config);
             cudaDeviceSynchronize();
             
             if (scen.verify) {
                 LOG(LOG_INFO) << "[Verification] Computing CPU Reference...";
                 // Host buffers as bytes (uint8_t) to handle any bit width genericly
                 std::vector<uint8_t> h_C(c_size_bytes);
                 std::vector<uint8_t> h_C_ref(c_size_bytes);
                 
                 CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_size_bytes, cudaMemcpyDeviceToHost));
                 
                 // Construct Combined Permutation
                 std::vector<row_idx_t> combined_perm(source_mat_ptr->n_rows);
                 for(int i=0; i<m4rm_rows; ++i) combined_perm[i] = i;
                 for(size_t i=0; i<comp_sparse.permutation.size(); ++i) {
                     combined_perm[m4rm_rows + i] = m4rm_rows + comp_sparse.permutation[i];
                 }
                 
                 Verification::cpu_spmm_gf2(*source_mat_ptr, h_V_rand.data(), h_C_ref.data(), combined_perm, scen.config.vector_width_bits);
                 Verification::check_results(h_C.data(), h_C_ref.data(), source_mat_ptr->n_rows, scen.config.vector_width_bits);
             }
             
             // 5. Timing
             SpMMKernels::flush_l2_cache();
             auto t0 = std::chrono::high_resolution_clock::now();
             int iterations = 20; 
             if (source_mat_ptr->n_rows > 500000) iterations = 10;
             
             for(int i=0; i<iterations; ++i) {
                 SpMMKernels::run_hybrid(d_hybrid, d_C, d_V, scen.config);
             }
             cudaDeviceSynchronize();
             auto t1 = std::chrono::high_resolution_clock::now();
             double time_avg_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iterations;
             
             // Metrics
             size_t total_nnz = 0;
             for(const auto& r : source_mat_ptr->rows) total_nnz += r.size();
             double throughput = (double)total_nnz / 1e9 / (time_avg_ms / 1000.0);
             double eff_bw = (double)(total_nnz * 4 + source_mat_ptr->n_rows * 4) / 1e9 / (time_avg_ms / 1000.0);
             
             LOG(LOG_INFO) << "Results for " << scen.name << ":";
             LOG(LOG_INFO) << "  Runtime:       " << std::fixed << std::setprecision(4) << time_avg_ms << " ms";
             LOG(LOG_INFO) << "  Throughput:    " << throughput << " GNNz/s";
             LOG(LOG_INFO) << "  CSR Eff BW:    " << eff_bw << " GB/s";
             
             cudaFree(d_V_bytes);
             cudaFree(d_C_bytes);
             free_m4rm_context(ctx);
             SpMMKernels::free_matrix(d_sparse);

        } 
        // --- B. Sliced Execution (Legacy) ---
        else if (scen.config.enable_vertical_slicing) {
            LOG(LOG_INFO) << "[Setup] Slicing Matrix (Legacy) for " << scen.name << "...";
            // Generate Slices
            std::vector<HostMatrix> h_slices = MatrixPreprocessor::slice_matrix_vertical(
                *source_mat_ptr, dense_lim, scen.config.tiled_row_block_size, scen.config.max_nnz_per_slice_block
            );
            
            std::vector<CompressedMatrix> comp_slices;
            size_t total_nnz_all_slices = 0;
            for(const auto& h_s : h_slices) {
                CompressedMatrix cm = MatrixPreprocessor::preprocess(
                    h_s, (h_s.n_cols == h_slices[0].n_cols ? dense_lim : 0), 0, false, 
                    scen.config.pfor_exception_threshold, scen.config.tiled_row_block_size
                );
                comp_slices.push_back(cm);
                total_nnz_all_slices += cm.total_nnz;
            }
            SlicedDeviceMatrix dSlices = SpMMKernels::allocate_sliced(comp_slices);
            
            // Measure
            size_t width_bytes = scen.config.vector_width_bits / 8;
            size_t v_size_bytes = source_mat_ptr->n_cols * width_bytes;
            size_t c_size_bytes = source_mat_ptr->n_rows * width_bytes;
            
            uint8_t *d_V_bytes, *d_C_bytes;
            CUDA_CHECK(cudaMalloc(&d_V_bytes, v_size_bytes));
            CUDA_CHECK(cudaMalloc(&d_C_bytes, c_size_bytes));

            // Generate random vector and copy to device
            std::vector<uint8_t> h_V_rand = gen.generate_random_vector(source_mat_ptr->n_cols, scen.config.vector_width_bits);
            CUDA_CHECK(cudaMemcpy(d_V_bytes, h_V_rand.data(), v_size_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_C_bytes, 0, c_size_bytes));
            
            uint64_t *d_V = (uint64_t*)d_V_bytes;
            uint64_t *d_C = (uint64_t*)d_C_bytes;

            // Verify
            SpMMKernels::run_spmm_sliced(dSlices, d_C, d_V, scen.config);
            cudaDeviceSynchronize();
            if (scen.verify) {
                 LOG(LOG_INFO) << "[Verification] Skipped for Sliced scenario in this view.";
            }

            // Time
            SpMMKernels::flush_l2_cache();
            auto t0 = std::chrono::high_resolution_clock::now();
            int iterations = 10;
            for(int i=0; i<iterations; ++i) SpMMKernels::run_spmm_sliced(dSlices, d_C, d_V, scen.config);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            double time_avg_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / iterations;

            LOG(LOG_INFO) << "Results for " << scen.name << ":";
            LOG(LOG_INFO) << "  Runtime:       " << time_avg_ms << " ms";
            LOG(LOG_INFO) << "  Throughput:    " << ((double)total_nnz_all_slices/1e9)/(time_avg_ms/1000.0) << " GNNz/s";

            cudaFree(d_V_bytes);
            cudaFree(d_C_bytes);
            SpMMKernels::free_sliced(dSlices);
        }
        // --- C. Standard Sparse/RCM Execution ---
        else {
             HostMatrix rcm_temp_mat;
             const HostMatrix* current_src = source_mat_ptr;

             if (scen.config.enable_rcm) {
                std::vector<idx_t> p = MatrixReordering::compute_rcm_permutation(*source_mat_ptr, dense_lim);
                rcm_temp_mat = MatrixReordering::apply_permutation(*source_mat_ptr, p, dense_lim);
                current_src = &rcm_temp_mat; 
             }
             
             CompressedMatrix comp = MatrixPreprocessor::preprocess(
                  *current_src, dense_lim, heavy_lim, false,
                  scen.config.pfor_exception_threshold, scen.config.tiled_row_block_size
             );
            
             DeviceMatrix dMat = SpMMKernels::allocate_and_copy(comp);
             
             LOG(LOG_INFO) << "----------------------------------------------------------";
             LOG(LOG_INFO) << "   Benchmark Phase: " << scen.name;
             LOG(LOG_INFO) << "----------------------------------------------------------";

             // Verification Flow
             if (scen.verify) {
                 size_t width_bytes = scen.config.vector_width_bits / 8;
                 size_t v_size_bytes = current_src->n_cols * width_bytes;
                 size_t c_size_bytes = current_src->n_rows * width_bytes;
                 
                 uint8_t *d_V_bytes, *d_C_bytes;
                 CUDA_CHECK(cudaMalloc(&d_V_bytes, v_size_bytes));
                 CUDA_CHECK(cudaMalloc(&d_C_bytes, c_size_bytes));
                 
                 // Generate random vector for verification
                 std::vector<uint8_t> h_V_rand = gen.generate_random_vector(current_src->n_cols, scen.config.vector_width_bits);
                 CUDA_CHECK(cudaMemcpy(d_V_bytes, h_V_rand.data(), v_size_bytes, cudaMemcpyHostToDevice));
                 CUDA_CHECK(cudaMemset(d_C_bytes, 0, c_size_bytes));
                 
                 uint64_t *d_V = (uint64_t*)d_V_bytes;
                 uint64_t *d_C = (uint64_t*)d_C_bytes;

                 SpMMKernels::run_spmm(dMat, d_C, d_V, scen.config);
                 cudaDeviceSynchronize();

                 LOG(LOG_INFO) << "[Verification] Computing CPU Reference...";
                 std::vector<uint8_t> h_C(c_size_bytes);
                 std::vector<uint8_t> h_C_ref(c_size_bytes);
                 
                 CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_size_bytes, cudaMemcpyDeviceToHost));
                 
                 Verification::cpu_spmm_gf2(*current_src, h_V_rand.data(), h_C_ref.data(), comp.permutation, scen.config.vector_width_bits);
                 Verification::check_results(h_C.data(), h_C_ref.data(), current_src->n_rows, scen.config.vector_width_bits);

                 cudaFree(d_V_bytes);
                 cudaFree(d_C_bytes);
             }

             // Timing using helper
             PerfResult res = measure_kernel_perf(dMat, current_src->n_rows, current_src->n_cols, scen.config);
             
             size_t total_nnz = 0;
             for(const auto& r : current_src->rows) total_nnz += r.size();
             double throughput = (double)total_nnz / 1e9 / (res.time_ms / 1000.0);
             
             LOG(LOG_INFO) << "Results for " << scen.name << ":";
             LOG(LOG_INFO) << "  Runtime:       " << res.time_ms << " ms";
             LOG(LOG_INFO) << "  Throughput:    " << throughput << " GNNz/s";

             SpMMKernels::free_matrix(dMat);
        }
    }

    // 3. Post-Benchmark Analysis
    if (enable_layer_analysis) {
        execute_dense_scaling(AT);
        execute_density_profile(AT, 0); 
    }
}

// =================================================================================
// DENSITY PROFILE ANALYSIS (Dynamic Slicing)
// =================================================================================

void BenchmarkSuite::execute_density_profile(const HostMatrix& AT, idx_t dense_limit_cols) {
    LOG(LOG_INFO) << "========================================================================================================================";
    LOG(LOG_INFO) << "   DENSITY PROFILE ANALYSIS (Fine-grained partitioning for Kernel Selection)";
    LOG(LOG_INFO) << "========================================================================================================================";

    struct BlockDef {
        row_idx_t start;
        row_idx_t count;
        double avg_nnz;
        size_t total_nnz;
        std::string label;
    };
    std::vector<BlockDef> blocks;
    
    // 1. Fixed Transition Ranges
    std::vector<std::pair<row_idx_t, row_idx_t>> fixed_ranges = {
      {32, 64}, {64, 128}, {128, 256}, {256, 512}, {512, 1024}, {1024, 2048}, {2048, 4096}, {4096, 8192}, {8192, 16384}, {16384, 32768}, {32768, 65536}, {65536, 131072}, {131072, 262144}, {262144, 524288}, {524288, 1048576}
    };

    for(auto& range : fixed_ranges) {
        if (range.first >= AT.n_rows) break;
        row_idx_t end = std::min(range.second, AT.n_rows);
        row_idx_t count = end - range.first;
        
        HostMatrix h_slice = slice_matrix_simple(AT, range.first, count);
        size_t total_nnz = 0;
        for(const auto& r : h_slice.rows) total_nnz += r.size();
        
        blocks.push_back({range.first, count, (double)total_nnz/count, total_nnz, "Fixed"});
    }

    // 2. Dynamic Ranges (65536+)
    size_t target_bytes = 64 * 1024 * 1024; // 64 MB
    row_idx_t current_start = 1048576;
    
    if (current_start < AT.n_rows) {
        size_t current_bytes = 0;
        size_t current_nnz = 0;

        for (row_idx_t r = current_start; r < AT.n_rows; ++r) {
            size_t row_nnz = AT.rows[r].size();
            current_bytes += (row_nnz * 4 + 4); 
            current_nnz += row_nnz;

            if (current_bytes >= target_bytes || r == AT.n_rows - 1) {
                blocks.push_back({current_start, (r - current_start) + 1, (double)current_nnz / ((r - current_start) + 1), current_nnz, "Dynamic"});
                current_start = r + 1;
                current_bytes = 0;
                current_nnz = 0;
            }
        }
    }

    LOG(LOG_INFO) << "Generated " << blocks.size() << " sparse blocks.";
    LOG(LOG_INFO) << "Row Range       | Density | Golomb | WarpCSR | TiledCOO | Delta16 | PFor(0.8) | PFor(0.9) | PFor(0.98)";
    LOG(LOG_INFO) << "----------------|---------|--------|---------|----------|---------|-----------|-----------|-----------";

    for (size_t b_idx = 0; b_idx < blocks.size(); ++b_idx) {
        const auto& block = blocks[b_idx];
        if (block.count == 0) continue;

        HostMatrix h_slice = slice_matrix_simple(AT, block.start, block.count);
        
        std::vector<std::pair<std::string, SpMMConfig>> configs;
        
        // DEFAULT TO 128-BIT for Analysis
        int default_bits = 128;

        // 1. Golomb (Standard)
        SpMMConfig cfg_g; cfg_g.enable_sparse = true; cfg_g.vector_width_bits = default_bits;
        configs.push_back({"Golomb", cfg_g});

        // 2. WarpCSR
        SpMMConfig cfg_w; cfg_w.enable_sparse = true; cfg_w.enable_heavy_warp_csr = true; cfg_w.vector_width_bits = default_bits;
        configs.push_back({"WarpCSR", cfg_w});

        // 3. TiledCOO
        SpMMConfig cfg_t; cfg_t.enable_sparse = true; cfg_t.enable_sparse_tiled_coo = true; cfg_t.vector_width_bits = default_bits;
        configs.push_back({"TiledCOO", cfg_t});

        // 4. Delta16
        SpMMConfig cfg_d; cfg_d.enable_sparse = true; cfg_d.enable_sparse_delta_16 = true; cfg_d.vector_width_bits = default_bits;
        configs.push_back({"Delta16", cfg_d});

        // 5. PFor Variants
        float p_thresh[] = {0.80f, 0.90f, 0.98f};
        for (float t : p_thresh) {
            SpMMConfig cfg_p; cfg_p.enable_sparse = true; cfg_p.enable_sparse_pfor_bit_exact = true; cfg_p.pfor_exception_threshold = t; cfg_p.vector_width_bits = default_bits;
            configs.push_back({"PFor", cfg_p});
        }

        std::ostringstream row_ss;
        row_ss << "[" << std::setw(6) << block.start << "-" << std::setw(6) << block.start + block.count << ") | "
               << std::setw(7) << std::fixed << std::setprecision(1) << block.avg_nnz << " | ";

        for (size_t i = 0; i < configs.size(); ++i) {
            auto& pair = configs[i];
            CompressedMatrix comp = MatrixPreprocessor::preprocess(
                h_slice, dense_limit_cols, 0 /* heavy lim */, false,
                pair.second.pfor_exception_threshold
            );

            DeviceMatrix d_mat = SpMMKernels::allocate_and_copy(comp);

            PerfResult res = measure_kernel_perf(d_mat, h_slice.n_rows, h_slice.n_cols, pair.second);
            double gnnz = (double)block.total_nnz / 1e9 / (res.time_ms / 1000.0);

            int width = 6;
            if (i == 1) width = 7;
            else if (i == 2) width = 8;
            else if (i == 3) width = 7;
            else if (i >= 4) width = 9;

            row_ss << std::setw(width) << std::setprecision(2) << gnnz << " | ";

            SpMMKernels::free_matrix(d_mat);
        }
        LOG(LOG_INFO) << row_ss.str();
    }
}

// =================================================================================
// DENSE SCALING ANALYSIS (Isolated Intervals)
// =================================================================================

void BenchmarkSuite::execute_dense_scaling(const HostMatrix& AT) {
    LOG(LOG_INFO) << "============================================================================================================";
    LOG(LOG_INFO) << "   DENSE KERNEL ANALYSIS (Isolated Row Ranges of AT)";
    LOG(LOG_INFO) << "============================================================================================================";

    std::vector<row_idx_t> cuts = {0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int analysis_width = 128; // Default to 128-bit for this specific analysis view
    
    LOG(LOG_INFO) << "Row Range       | M4RM (G/s)  | Bitslice (G/s) | WarpCSR (G/s) | Bitmap (G/s)";
    LOG(LOG_INFO) << "----------------|-------------|----------------|---------------|-------------";

    for (size_t i = 0; i < cuts.size() - 1; ++i) {
        row_idx_t start = cuts[i];
        row_idx_t end = cuts[i+1];
        if (start >= AT.n_rows) break;
        if (end > AT.n_rows) end = AT.n_rows;
        
        row_idx_t count = end - start;
        if (count == 0) break;

        HostMatrix h_slice = slice_matrix_simple(AT, start, count);
        size_t total_nnz = 0;
        for(const auto& r : h_slice.rows) total_nnz += r.size();

        // 0. M4RM (Batched Processing)
        double m4rm_throughput = 0.0;
        if (count >= 8) {
            int num_chunks = count / 8;
            if (num_chunks > 0) {
                std::vector<M4RMContext> ctxs;
                std::vector<int> offsets;

                for(int c=0; c<num_chunks; ++c) {
                    int chunk_start = c * 8;
                    HostMatrix chunk = slice_matrix_simple(h_slice, chunk_start, 8);
                    ctxs.push_back(MatrixPreprocessor::preprocess_m4rm(chunk, 8)); 
                    offsets.push_back(chunk_start);
                }

                size_t width_bytes = analysis_width / 8;
                size_t v_size = AT.n_cols * width_bytes;
                size_t c_size = count * width_bytes;
                uint8_t *dV_bytes, *dC_bytes;
                CUDA_CHECK(cudaMalloc(&dV_bytes, v_size)); 
                CUDA_CHECK(cudaMalloc(&dC_bytes, c_size));

                // Use random vector for M4RM benchmark
                MatrixGenerator local_gen(999);
                std::vector<uint8_t> h_V_rand = local_gen.generate_random_vector(AT.n_cols, analysis_width, true);
                CUDA_CHECK(cudaMemcpy(dV_bytes, h_V_rand.data(), v_size, cudaMemcpyHostToDevice));
                
                uint64_t* dV = (uint64_t*)dV_bytes;

                int m4rm_iters = 20;
                double total_m4rm_ms = 0;
                
                for(int iter=0; iter<m4rm_iters; ++iter) {
                    SpMMKernels::flush_l2_cache();
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for(int c=0; c<num_chunks; ++c) {
                        // Offset C pointer by rows
                        // C is count * width_bytes.
                        // dC_bytes + chunk_start * width_bytes
                        uint8_t* ptr = dC_bytes + offsets[c] * width_bytes;
			uint64_t* dC = (uint64_t*)ptr;
                        launch_m4rm_full(ctxs[c], dV, dC, analysis_width, 0);
                    }
                    cudaDeviceSynchronize();
                    auto t1 = std::chrono::high_resolution_clock::now();
                    total_m4rm_ms += std::chrono::duration<double>(t1 - t0).count() * 1000.0;
                }
                
                double avg_ms = total_m4rm_ms / m4rm_iters;
                m4rm_throughput = (double)total_nnz / 1e9 / (avg_ms / 1000.0);

                cudaFree(dV_bytes); cudaFree(dC_bytes);
                for(auto& ctx : ctxs) free_m4rm_context(ctx);
            }
        }

        // 1. Bitslice
        CompressedMatrix cm_b = MatrixPreprocessor::preprocess(h_slice, AT.n_cols, 0, false);
        DeviceMatrix d_b = SpMMKernels::allocate_and_copy(cm_b);
        SpMMConfig cfg_b; cfg_b.enable_dense_bitslice = true; cfg_b.enable_sparse = false; cfg_b.vector_width_bits = analysis_width;
        PerfResult r_b = measure_kernel_perf(d_b, count, AT.n_cols, cfg_b);
        SpMMKernels::free_matrix(d_b);

        // 2. Warp CSR (Heavy)
        CompressedMatrix cm_w = MatrixPreprocessor::preprocess(h_slice, 0, count, false);
        DeviceMatrix d_w = SpMMKernels::allocate_and_copy(cm_w);
        SpMMConfig cfg_w; cfg_w.enable_heavy_warp_csr = true; cfg_w.enable_sparse = false; cfg_w.vector_width_bits = analysis_width;
        PerfResult r_w = measure_kernel_perf(d_w, count, AT.n_cols, cfg_w);
        SpMMKernels::free_matrix(d_w);

        // 3. Bitmap (Heavy)
        CompressedMatrix cm_m = MatrixPreprocessor::preprocess(h_slice, 0, count, false);
        DeviceMatrix d_m = SpMMKernels::allocate_and_copy(cm_m);
        SpMMConfig cfg_m; cfg_m.enable_heavy_bitmap = true; cfg_m.enable_sparse = false; cfg_m.vector_width_bits = analysis_width;
        PerfResult r_m = measure_kernel_perf(d_m, count, AT.n_cols, cfg_m);
        SpMMKernels::free_matrix(d_m);
        
        LOG(LOG_INFO) << "[" << std::setw(6) << start << "-" << std::setw(6) << end << ") | "
                      << std::setw(11) << std::fixed << std::setprecision(2) << m4rm_throughput << " | "
                      << std::setw(14) << (total_nnz/1e9)/(r_b.time_ms/1000.0) << " | "
                      << std::setw(13) << (total_nnz/1e9)/(r_w.time_ms/1000.0) << " | "
                      << std::setw(11) << (total_nnz/1e9)/(r_m.time_ms/1000.0);
    }
}
