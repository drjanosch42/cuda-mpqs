// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#pragma once
#include "common.h"
#include "spmm_optimized.h"
#include "kernels.h"
#include "m4rm_data.h"
#include "gpu_autotuner.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

/// @brief Configuration passed to setup_and_benchmark() to control the
/// GPU-only autotuning pipeline.  Lives here (not in bw_solver.h) to
/// avoid a circular include between spmm_core and the parent project.
/// The solver translates BWAutoTuneConfig → SpMMAutoTuneConfig at the call site.
struct SpMMAutoTuneConfig {
    bool gpu_only = true;
    GPUAutoTuner::Config gpu_config = {};
};

struct SpMMPerformanceReport {
    int bit_width_m;
    int bit_width_n;
    double time_A_ms;
    double time_AT_ms;
    double throughput_A_gnnz;
    double throughput_AT_gnnz;
    double ratio_A_over_AT; 
    size_t peak_gpu_mem_bytes;
    size_t host_mem_bytes;
};

class BlockWiedemannSpMM {
public:
    BlockWiedemannSpMM(const HostMatrix& raw_A, idx_t padded_n_cols = 0, bool sort_rows = true, bool sort_cols = true);
    ~BlockWiedemannSpMM();

    SpMMPerformanceReport setup_and_benchmark(
        int bit_width_m, int bit_width_n,
        const SpMMAutoTuneConfig& autotune_cfg = SpMMAutoTuneConfig{});

    /**
     * @brief Verifies the currently optimized/loaded kernels against the CPU reference.
     * Generates random vectors, runs GPU SpMM (A and AT), runs CPU reference, and compares.
     * Must be called AFTER setup_and_benchmark.
     */
    bool verify_A(int bit_width);
    bool verify_AT(int bit_width);
    
    // Execution (legacy, synchronous)
    void execute_A(void* d_C, const void* d_V);
    void execute_AT(void* d_C, const void* d_V);

    // Execution (stream-aware, no internal sync — caller manages ordering)
    void execute_A(void* d_C, const void* d_V, cudaStream_t stream);
    void execute_AT(void* d_C, const void* d_V, cudaStream_t stream);

    void load_A_to_device(int bit_width);
    void unload_A_from_device();
    void load_AT_to_device(int bit_width);
    void unload_AT_from_device();

    void print_memory_statistics() const;
    bool benchmark_preprocessing();

    // Introspection / Logging
    std::string get_kernel_summary(bool is_AT) const;
    size_t get_device_memory_bytes(bool is_AT) const;
    size_t get_host_memory_bytes(bool is_AT) const;

    // Generic Vector Transformations (Host)
    void preprocess_domain_A(const void* h_in, void* h_out, size_t num_elements, int bit_width) const;
    void postprocess_domain_AT(const void* h_in, void* h_out, size_t num_elements, int bit_width) const;
    void preprocess_domain_AT(const void* h_in, void* h_out, size_t num_elements, int bit_width) const;
    void postprocess_domain_A(const void* h_in, void* h_out, size_t num_elements, int bit_width) const;

    // Generic Vector Transformations (Device)
    void permute_vec_rows_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width);
    void unpermute_vec_rows_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width);
    void permute_vec_cols_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width);
    void unpermute_vec_cols_gpu(const void* d_in, void* d_out, size_t num_elements, int bit_width);

private:
    struct ManagedMatrix {
        HostMatrix host_data;
        std::unique_ptr<OptimizedSpMM> device_engine;
        ExecutionPlan plan;
        bool is_on_device = false;
        size_t host_mem_usage = 0;
        size_t device_mem_usage = 0; // Tracked via cudaMemGetInfo

        // Hybrid Execution Support (Golden Path)
        bool use_hybrid = false;
        HybridDeviceMatrix hybrid_data = {};
        SpMMConfig hybrid_config = {};
    };

    struct PermutationMap {
        std::vector<row_idx_t> map;
        std::vector<row_idx_t> inv;
        row_idx_t* d_map = nullptr;
        row_idx_t* d_inv = nullptr;
    };

    idx_t logical_n_cols_;
    size_t total_nnz_;
    int current_bit_width_ = 0;

    ManagedMatrix mat_A_;
    ManagedMatrix mat_AT_;

    PermutationMap P_row_;
    PermutationMap P_col_;

    void host_matrix_to_csr_arrays(const HostMatrix& m, std::vector<uint32_t>& row_ptr, std::vector<uint32_t>& col_ind);
    static PermutationMap compute_density_permutation(const HostMatrix& m, idx_t n_cols, bool transpose_view);
    static HostMatrix apply_permutation_explicit(const HostMatrix& src, idx_t padded_cols, const PermutationMap& p_row, const PermutationMap& p_col);

    double measure_hot_execution(int iterations, size_t v_bytes, size_t c_bytes, bool is_AT);
    void apply_permutation_cpu(const void* src, void* dst, size_t n, size_t elem_size, const std::vector<row_idx_t>& mapping) const;
    
    void setup_hybrid_path(ManagedMatrix& mat, int bit_width);
    void free_hybrid_path(ManagedMatrix& mat);
};
