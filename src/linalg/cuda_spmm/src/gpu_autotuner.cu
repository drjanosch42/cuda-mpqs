// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 2024-2026 Fabian Januszewski
// This file is part of the Block Wiedemann implementation.
// See LICENSE for licensing terms, including the NVIDIA CUDA Toolkit exception.

#include "gpu_autotuner.h"
#include "kernels.h"
#include "m4rm_data.h"
#include "device_csr.h"
#include "device_format_convert.h"
#include "format_arena.h"
#include "hpc_logger.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Block-size threshold (in rows) above which CPU-fallback formats (PFor_BE,
// Golomb) are excluded from benchmarking.  These formats use full CPU
// preprocessing whose cost scales linearly with rows×weight, making them
// non-competitive for large blocks while GPU-native formats (TiledCOO,
// Delta16) handle the same range efficiently on-device.
//
// Raised from 2048 to 16384 to allow PForDelta on AT's moderately dense
// sparse-tail blocks (e.g., 6247-8746 rows for 50k/70k matrices).
// CPU preprocessing cost is acceptable for one-time autotuning.
static constexpr uint32_t CPU_FALLBACK_BLOCK_THRESHOLD = 16384;

// ---------------------------------------------------------------------------
// config_to_spmm_config — copied from autotuner.cpp:76-100 (static helper)
// ---------------------------------------------------------------------------
static SpMMConfig config_to_spmm_config(const KernelConfig& kc) {
    SpMMConfig cfg;
    cfg.vector_width_bits = kc.vector_width_bits;
    cfg.enable_dense_bitslice = false;
    cfg.enable_heavy_rows = false;
    cfg.enable_sparse = false;

    switch (kc.id) {
        case KernelID::M4RM: cfg.enable_m4rm = true; cfg.m4rm_rows = kc.params.m4rm_rows; break;
        case KernelID::Dense_Bitslice:
            cfg.enable_dense_bitslice = true;
            cfg.enable_sparse = true;
            cfg.enable_heavy_warp_csr = true;
            break;
        case KernelID::Sparse_WarpCSR: cfg.enable_sparse = true; cfg.enable_sparse_warp_csr = true; break;
        case KernelID::Sparse_TiledCOO: cfg.enable_sparse = true; cfg.enable_sparse_tiled_coo = true; cfg.tiled_row_block_size = kc.params.tiled_block_size; break;
        case KernelID::Sparse_TiledCOO_Unrolled: cfg.enable_sparse = true; cfg.enable_sparse_tiled_coo_unrolled = true; cfg.tiled_row_block_size = kc.params.tiled_block_size; break;
        case KernelID::Sparse_Delta16: cfg.enable_sparse = true; cfg.enable_sparse_delta_16 = true; break;
        case KernelID::Sparse_PForDelta: cfg.enable_sparse = true; cfg.enable_sparse_pfor = true; cfg.pfor_exception_threshold = kc.params.pfor_threshold; break;
        case KernelID::Sparse_PForDelta_BitExact: cfg.enable_sparse = true; cfg.enable_sparse_pfor_bit_exact = true; cfg.pfor_exception_threshold = kc.params.pfor_threshold; break;
        case KernelID::Sparse_Golomb: cfg.enable_sparse = true; break;
        default: break;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// select_tile_size — density-based tile size heuristic for TiledCOO
//
// Larger tiles pack more NNZ per block, improving GPU utilization on
// ultra-sparse rows.  Shared memory limits 2048-row tiles to vec_width < 256.
// ---------------------------------------------------------------------------
static int select_tile_size(double avg_weight, int vec_width) {
    int max_tile = (vec_width >= 256) ? 1024 : 2048;
    int tile;
    if (avg_weight >= 32.0)      tile = 256;
    else if (avg_weight >= 16.0) tile = 512;
    else if (avg_weight >= 8.0)  tile = 1024;
    else                         tile = 2048;
    return std::min(tile, max_tile);
}

// ---------------------------------------------------------------------------
// generate_candidate_configs — adapted from autotuner.cpp:14-74
//
// TiledCOO now emits one candidate per tile size (256/512/1024/2048).
// The benchmark loop prunes non-optimal tile sizes per-block using the
// select_tile_size() heuristic — zero extra benchmark overhead.
// ---------------------------------------------------------------------------
std::vector<KernelConfig> GPUAutoTuner::generate_candidate_configs(
    int global_vec_width, const Config& config
) {
    std::vector<KernelConfig> configs;

    // 1. M4RM
    if (config.enable_m4rm) {
        KernelConfig kc; kc.id = KernelID::M4RM; kc.vector_width_bits = global_vec_width;
        kc.params.m4rm_rows = 8; kc.name = "M4RM (8 Rows)";
        configs.push_back(kc);
    }

    // 2. Tiled COO — one candidate per tile size
    if (config.enable_tiledcoo) {
        int max_tile = (global_vec_width >= 256) ? 1024 : 2048;
        for (int tile_size : {256, 512, 1024, 2048}) {
            if (tile_size > max_tile) continue;  // shared memory guard
            KernelConfig kc; kc.id = KernelID::Sparse_TiledCOO;
            kc.vector_width_bits = global_vec_width;
            kc.params.tiled_block_size = tile_size;
            kc.name = "Tiled COO (" + std::to_string(tile_size) + ")";
            configs.push_back(kc);
        }
    }

    // 2a. Tiled COO Unrolled — same tile sizes, unrolled kernel variant
    if (config.enable_tiledcoo_unrolled) {
        int max_tile = (global_vec_width >= 256) ? 1024 : 2048;
        for (int tile_size : {256, 512, 1024, 2048}) {
            if (tile_size > max_tile) continue;
            KernelConfig kc; kc.id = KernelID::Sparse_TiledCOO_Unrolled;
            kc.vector_width_bits = global_vec_width;
            kc.params.tiled_block_size = tile_size;
            kc.name = "Tiled COO Unrolled (" + std::to_string(tile_size) + ")";
            configs.push_back(kc);
        }
    }

    // 2b. Warp-CSR (for ultra-sparse rows)
    if (config.enable_warp_csr) {
        KernelConfig kc; kc.id = KernelID::Sparse_WarpCSR; kc.vector_width_bits = global_vec_width;
        kc.name = "Warp-CSR";
        configs.push_back(kc);
    }

    // 3. Delta-16
    if (config.enable_delta16) {
        KernelConfig kc; kc.id = KernelID::Sparse_Delta16; kc.vector_width_bits = global_vec_width;
        kc.name = "Delta-16";
        configs.push_back(kc);
    }

    // 4. PForDelta BitExact
    if (config.enable_pfor_be) {
        KernelConfig kc; kc.id = KernelID::Sparse_PForDelta_BitExact; kc.vector_width_bits = global_vec_width;
        kc.params.pfor_threshold = 0.90f; kc.name = "PForDelta (BE, 0.90)";
        configs.push_back(kc);
    }

    // 5. Golomb-Rice
    if (config.enable_golomb) {
        KernelConfig kc; kc.id = KernelID::Sparse_Golomb; kc.vector_width_bits = global_vec_width;
        kc.name = "Golomb-Rice";
        configs.push_back(kc);
    }

    return configs;
}

// ---------------------------------------------------------------------------
// create_exponential_blocks — Two-phase adaptive partitioning
//
// Phase 1 (dense head): 8-row blocks for M4RM candidates, covering only the
//   first n_dense_rows (rounded up to a multiple of 8).
// Phase 2 (sparse tail): Blocks sized adaptively based on remaining rows,
//   targeting ~4-8 blocks with exponential growth.
//
// Uses a single bulk D→H copy of d_row_ptr instead of per-block slice()
// calls, eliminating 2×cudaMemcpy per block.
// ---------------------------------------------------------------------------
std::vector<GPUAutoTuner::AtomicBlock> GPUAutoTuner::create_exponential_blocks(
    const DeviceCSR& csr, int initial_size, int max_size, bool is_transposed
) {
    // Bulk-download row_ptr — one D→H copy instead of 2 per block
    std::vector<uint32_t> h_row_ptr(csr.n_rows + 1);
    CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), csr.d_row_ptr,
        (csr.n_rows + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<AtomicBlock> blocks;
    uint32_t current = 0;

    // Phase 1: Dense head — 8-row blocks for M4RM candidates
    // AT matrices with sparse dense heads: cap to 2 blocks (16 rows) to prevent
    // throughput collapse. M4RM degrades 87% (14→1.8 GNNz/s) on ultra-sparse AT
    // rows. But for dense matrices (avg weight >= 16), M4RM is the best kernel
    // and capping it pushes rows to slower Warp-CSR.
    uint32_t dense_end = std::min((csr.n_dense_rows + 7u) & ~7u, csr.n_rows);
    if (is_transposed && dense_end > 16u) {
        double avg_dense_weight = (double)h_row_ptr[dense_end] / dense_end;
        if (avg_dense_weight < 16.0) {
            dense_end = 16u;  // cap M4RM blocks only for sparse AT heads
        }
    }
    while (current < dense_end) {
        uint32_t next = std::min(current + 8u, csr.n_rows);
        uint32_t n = next - current;
        AtomicBlock b;
        b.start_row  = current;
        b.end_row    = next;
        b.nnz        = h_row_ptr[next] - h_row_ptr[current];
        b.avg_weight = (n > 0) ? (double)b.nnz / n : 0;
        b.dense_head = true;
        blocks.push_back(b);
        current = next;
    }

    // Phase 2: Sparse tail
    // For AT (is_transposed): fixed 1024-row blocks give the DP solver
    // finer density resolution that tracks AT's clustered profile.
    // For A: exponential growth targets ~4-8 blocks (original behavior).
    uint32_t remaining = csr.n_rows - current;

    if (is_transposed) {
        // AT: fixed small blocks for finer density resolution
        constexpr int AT_SPARSE_BLOCK_SIZE = 1024;
        while (current < csr.n_rows) {
            uint32_t next = std::min(current + (uint32_t)AT_SPARSE_BLOCK_SIZE, csr.n_rows);
            uint32_t n = next - current;
            AtomicBlock b;
            b.start_row  = current;
            b.end_row    = next;
            b.nnz        = h_row_ptr[next] - h_row_ptr[current];
            b.avg_weight = (n > 0) ? (double)b.nnz / n : 0;
            b.dense_head = false;
            blocks.push_back(b);
            current = next;
        }
    } else {
        // A: exponential growth — initial step scaled to target ~4-8 blocks
        int step = (remaining > 0)
            ? std::min(std::max(initial_size, (int)(remaining / 8)), max_size)
            : initial_size;

        while (current < csr.n_rows) {
            uint32_t next = std::min(current + (uint32_t)step, csr.n_rows);
            uint32_t n = next - current;
            AtomicBlock b;
            b.start_row  = current;
            b.end_row    = next;
            b.nnz        = h_row_ptr[next] - h_row_ptr[current];
            b.avg_weight = (n > 0) ? (double)b.nnz / n : 0;
            b.dense_head = false;
            blocks.push_back(b);
            current = next;
            step = std::min(step * 2, max_size);
        }
    }

    return blocks;
}

// ---------------------------------------------------------------------------
// tune() — Main Orchestration
// ---------------------------------------------------------------------------
ExecutionPlan GPUAutoTuner::tune(
    const HostMatrix& mat, bool is_transposed,
    int global_vec_width, const Config& config, bool verbose
) {
    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] Starting GPU-only tuning for "
                     << (is_transposed ? "AT" : "A")
                     << " (" << mat.n_rows << "x" << mat.n_cols
                     << ", vec_width=" << global_vec_width << ")" << std::endl;

    // ─── Phase 0: Setup ───
    DeviceCSR csr = upload_host_matrix_to_device_csr(mat);
    device_csr_permute_by_density(csr);
    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] DeviceCSR: " << csr.device_bytes() / (1024*1024)
                     << " MB, n_dense_rows=" << csr.n_dense_rows << std::endl;

    // ─── Device Info ───
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int sm_count = (config.sm_count > 0) ? config.sm_count : prop.multiProcessorCount;
    int l2_size_bytes;
    cudaDeviceGetAttribute(&l2_size_bytes, cudaDevAttrL2CacheSize, device_id);
    LOG(LOG_STATS) << "[GPUAutoTuner] Device: SM " << prop.major << "." << prop.minor
                  << ", " << sm_count << " SMs, "
                  << (prop.totalGlobalMem >> 20) << " MB, "
                  << "L2=" << (l2_size_bytes >> 10) << " KB";

    auto configs = generate_candidate_configs(global_vec_width, config);
    auto blocks  = create_exponential_blocks(csr, config.initial_block_size, config.max_block_size, is_transposed);

    // Pre-compute DeviceCSRSlice per block — avoids 2× D→H memcpy per
    // (block, format) pair in the benchmark loop.
    std::vector<DeviceCSRSlice> block_slices(blocks.size());
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        block_slices[bi].d_row_ptr = csr.d_row_ptr + blocks[bi].start_row;
        block_slices[bi].d_col_ind = csr.d_col_ind;
        block_slices[bi].start_row = blocks[bi].start_row;
        block_slices[bi].n_rows    = blocks[bi].end_row - blocks[bi].start_row;
        block_slices[bi].n_cols    = csr.n_cols;
        block_slices[bi].nnz       = blocks[bi].nnz;
    }

    // Compute effective n_spmm_calls for conversion cost amortization
    uint32_t effective_spmm_calls = config.n_spmm_calls;
    if (effective_spmm_calls == 0) {
        uint32_t N = std::max(mat.n_rows, mat.n_cols);
        effective_spmm_calls = std::max(100u, 2 * N / 64 + 100);
    }
    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] Amortizing conversion cost over "
                     << effective_spmm_calls << " SpMM calls" << std::endl;

    // Build a permuted host matrix only when CPU-fallback formats (PFor_BE,
    // Golomb) are enabled AND at least one non-pruned block needs them.
    // For AT, the density pruning threshold is lowered to 8.0 (see
    // density-based pruning below), so blocks with avg_weight 8-16 may
    // need CPU fallback formats.
    double cpu_fb_density_threshold = is_transposed ? 8.0 : 16.0;
    bool need_cpu_fallback = false;
    if (config.enable_pfor_be || config.enable_golomb) {
        for (const auto& b : blocks) {
            if (b.dense_head) continue;
            if (b.avg_weight < cpu_fb_density_threshold) continue;
            if ((b.end_row - b.start_row) <= CPU_FALLBACK_BLOCK_THRESHOLD) {
                need_cpu_fallback = true;
                break;
            }
        }
    }

    HostMatrix permuted_mat;
    if (need_cpu_fallback) {
        std::vector<uint32_t> h_perm(csr.n_rows);
        cudaMemcpy(h_perm.data(), csr.d_density_perm, csr.n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        permuted_mat.n_rows = mat.n_rows;
        permuted_mat.n_cols = mat.n_cols;
        permuted_mat.rows.resize(mat.n_rows);
        for (uint32_t i = 0; i < mat.n_rows; i++)
            permuted_mat.rows[i] = mat.rows[h_perm[i]];
    }
    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] " << blocks.size() << " blocks, "
                     << configs.size() << " candidates" << std::endl;

    // Initialize benchmark_times
    for (auto& b : blocks)
        b.benchmark_times.assign(configs.size(), std::numeric_limits<double>::infinity());

    // Allocate shared benchmark buffers
    size_t vec_bytes = (size_t)mat.n_cols * (global_vec_width / 8);
    row_idx_t max_rows = 0;
    for (const auto& b : blocks) max_rows = std::max(max_rows, b.end_row - b.start_row);
    size_t max_out = (size_t)max_rows * (global_vec_width / 8);

    uint64_t *d_V, *d_C;
    CUDA_CHECK(cudaMalloc(&d_V, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, max_out));
    CUDA_CHECK(cudaMemset(d_V, 0xFF, vec_bytes));

    // L2 flush buffer — device_id and l2_size_bytes already queried above
    size_t flush_bytes = (l2_size_bytes > 0) ? (size_t)l2_size_bytes : 2 * 1024 * 1024;
    uint8_t* d_flush;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    FormatArena arena;

    // ─── Format Cache ───
    // Cache converted format data per (block, format) to avoid re-conversion
    // in Phase 7.  Each entry stores either a DeviceMatrix or M4RMContext.
    struct CachedFormat {
        DeviceMatrix dm = {};
        M4RMContext m4rm = {};
        bool valid = false;
        bool is_m4rm = false;
        bool is_arena_managed = false;  // TiledCOO, Delta16, M4RM use arena
    };
    std::vector<std::vector<CachedFormat>> format_cache(
        blocks.size(), std::vector<CachedFormat>(configs.size()));

    // ─── Phases 1-5: Format-Grouped Benchmarking ───
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        const auto& conf = configs[ci];
        LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Benchmarking format: " << conf.name << std::endl;

        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            auto& block = blocks[bi];
            uint32_t n_rows_block = block.end_row - block.start_row;

            // Dense head blocks: only benchmark M4RM
            if (block.dense_head && conf.id != KernelID::M4RM) continue;

            // M4RM: only on dense-head 8-row blocks
            if (conf.id == KernelID::M4RM && (!block.dense_head || n_rows_block != 8)) continue;

            // Skip CPU-fallback formats (PFor_BE, Golomb) for large blocks
            // where CPU preprocessing cost makes them non-competitive.
            if (n_rows_block > CPU_FALLBACK_BLOCK_THRESHOLD &&
                (conf.id == KernelID::Sparse_PForDelta_BitExact ||
                 conf.id == KernelID::Sparse_Golomb)) {
                block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                continue;
            }

            // WarpCSR: skip for dense blocks where TiledCOO/Delta16 are better
            if (conf.id == KernelID::Sparse_WarpCSR && block.avg_weight > 64.0) {
                block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                continue;
            }

            // TiledCOO: prune non-optimal tile sizes per-block (density heuristic)
            // Each block benchmarks exactly ONE tile size — no extra overhead.
            if (conf.id == KernelID::Sparse_TiledCOO && !block.dense_head) {
                int optimal_tile = select_tile_size(block.avg_weight, global_vec_width);
                if (conf.params.tiled_block_size != optimal_tile) {
                    block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                    continue;
                }
                // Column index overflow guard: verify tile size is safe for this matrix
                int row_bits = 0;
                { uint32_t tmp = conf.params.tiled_block_size - 1; while (tmp > 0) { row_bits++; tmp >>= 1; } }
                int col_bits = 32 - row_bits;
                if (csr.n_cols > ((1u << col_bits) - 1)) {
                    block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                    continue;
                }
            }

            // TiledCOO_Unrolled: same tile-size pruning as TiledCOO
            if (conf.id == KernelID::Sparse_TiledCOO_Unrolled && !block.dense_head) {
                int optimal_tile = select_tile_size(block.avg_weight, global_vec_width);
                if (conf.params.tiled_block_size != optimal_tile) {
                    block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                    continue;
                }
                int row_bits = 0;
                { uint32_t tmp = conf.params.tiled_block_size - 1; while (tmp > 0) { row_bits++; tmp >>= 1; } }
                int col_bits = 32 - row_bits;
                if (csr.n_cols > ((1u << col_bits) - 1)) {
                    block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                    continue;
                }
            }

            // Density-based format pruning: very sparse blocks only benchmark
            // formats suited to low-density rows. For AT (is_transposed), use
            // a lower threshold (8.0) — finer blocks have tighter density
            // ranges, so blocks with avg_weight 8-16 may benefit from PFor_BE.
            {
                double density_prune_threshold = is_transposed ? 8.0 : 16.0;
                if (!block.dense_head && block.avg_weight < density_prune_threshold &&
                    conf.id != KernelID::Sparse_Delta16 &&
                    conf.id != KernelID::Sparse_TiledCOO &&
                    conf.id != KernelID::Sparse_TiledCOO_Unrolled &&
                    conf.id != KernelID::Sparse_WarpCSR) {
                    continue;
                }
            }

            if (block.nnz == 0) {
                block.benchmark_times[ci] = 0.0001;
                continue;
            }

            size_t arena_mark = arena.get_watermark();  // P2-B: scoped cleanup
            try {
                DeviceCSRSlice slice = block_slices[bi];
                DeviceMatrix dm = {};
                M4RMContext m4rm = {};
                bool used_m4rm = false;
                bool arena_managed = false;

                // Time format conversion (one-time cost, amortized over L calls)
                CUDA_CHECK(cudaDeviceSynchronize());  // drain pending work
                auto t_conv_start = std::chrono::high_resolution_clock::now();

                // Convert CSR → format
                switch (conf.id) {
                    case KernelID::M4RM:
                        gpu_convert_csr_to_m4rm(slice, conf.params.m4rm_rows, arena, m4rm);
                        used_m4rm = true;
                        arena_managed = true;
                        break;
                    case KernelID::Sparse_TiledCOO:
                        gpu_convert_csr_to_tiledcoo(slice, conf.params.tiled_block_size, arena, dm);
                        arena_managed = true;
                        break;
                    case KernelID::Sparse_TiledCOO_Unrolled:
                        gpu_convert_csr_to_tiledcoo(slice, conf.params.tiled_block_size, arena, dm);
                        arena_managed = true;
                        break;
                    case KernelID::Sparse_Delta16:
                        gpu_convert_csr_to_delta16(slice, arena, dm);
                        arena_managed = true;
                        break;
                    case KernelID::Sparse_PForDelta_BitExact:
                        gpu_convert_csr_to_pfor_be(slice, conf.params.pfor_threshold,
                            arena, dm, config.allow_cpu_fallback, &permuted_mat);
                        break;
                    case KernelID::Sparse_WarpCSR:
                        // No format conversion — reference the DeviceCSR slice directly.
                        dm.warp_csr_row_ptr  = slice.d_row_ptr;
                        dm.warp_csr_col_ind  = slice.d_col_ind;
                        dm.warp_csr_n_rows   = slice.n_rows;
                        dm.warp_csr_start_row = 0;  // relative to slice for benchmarking
                        dm.n_rows            = slice.n_rows;
                        arena_managed = true;  // non-owning pointers — skip free_matrix in cleanup
                        break;
                    case KernelID::Sparse_Golomb:
                        gpu_convert_csr_to_golomb(slice, arena, dm,
                            config.allow_cpu_fallback, &permuted_mat);
                        break;
                    default: continue;
                }

                CUDA_CHECK(cudaDeviceSynchronize());  // ensure conversion is complete
                auto t_conv_end = std::chrono::high_resolution_clock::now();
                double conversion_ms = std::chrono::duration<double>(t_conv_end - t_conv_start).count() * 1000.0;

                // Benchmark
                size_t out_bytes = (size_t)n_rows_block * (global_vec_width / 8);
                CUDA_CHECK(cudaMemset(d_C, 0, out_bytes));

                SpMMConfig cfg = config_to_spmm_config(conf);

                // P2-C: Warmup (untimed) — warm instruction cache
                cudaGetLastError();  // clear any residual error state
                if (used_m4rm)
                    launch_m4rm_full(m4rm, d_V, d_C, global_vec_width);
                else
                    SpMMKernels::run_spmm(dm, d_C, d_V, cfg);
                cudaError_t warmup_err = cudaDeviceSynchronize();
                if (warmup_err != cudaSuccess) {
                    LOG(LOG_WARNING) << "[GPUAutoTuner] Warmup launch failed for "
                                     << conf.name << " on block [" << block.start_row
                                     << "," << block.end_row << "): "
                                     << cudaGetErrorString(warmup_err);
                    cudaGetLastError();  // clear sticky error
                    block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                    if (!arena_managed) {
                        SpMMKernels::free_matrix(dm);
                    }
                    arena.free_since(arena_mark);
                    continue;
                }

                // L2 cache flush — cold data cache for timed iterations
                CUDA_CHECK(cudaMemset(d_flush, 0xAA, flush_bytes));

                // Timed iterations (M1.2: 3 for M4RM, 5 for GPU-native sparse)
                int n_timed = (conf.id == KernelID::M4RM) ? 3 : 5;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int k = 0; k < n_timed; ++k) {
                    if (used_m4rm)
                        launch_m4rm_full(m4rm, d_V, d_C, global_vec_width);
                    else
                        SpMMKernels::run_spmm(dm, d_C, d_V, cfg);
                }
                {
                    cudaError_t sync_err = cudaDeviceSynchronize();
                    if (sync_err != cudaSuccess) {
                        // Kernel error during benchmark — mark as unusable and
                        // reset the device error state so subsequent CUDA calls
                        // are not poisoned.
                        LOG(LOG_WARNING) << "[GPUAutoTuner] Benchmark kernel error for "
                                         << conf.name << " on block [" << block.start_row
                                         << "," << block.end_row << "): "
                                         << cudaGetErrorString(sync_err) << std::endl;
                        cudaGetLastError();  // clear sticky error
                        block.benchmark_times[ci] = std::numeric_limits<double>::infinity();

                        // Free the failed conversion immediately
                        if (!arena_managed) {
                            SpMMKernels::free_matrix(dm);
                        }
                        arena.free_since(arena_mark);  // P2-B: scoped cleanup
                        continue;
                    }
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / n_timed;

                // Amortize one-time conversion cost over expected SpMM invocations
                double amortized_conversion_ms = conversion_ms / effective_spmm_calls;
                double total_ms = ms + amortized_conversion_ms;

                block.benchmark_times[ci] = total_ms;

                // Cache the converted format data instead of freeing it.
                // Phase 7 will retrieve winners from the cache, avoiding
                // redundant re-conversion.
                auto& cached = format_cache[bi][ci];
                cached.dm = dm;
                cached.m4rm = m4rm;
                cached.valid = true;
                cached.is_m4rm = used_m4rm;
                cached.is_arena_managed = arena_managed;

            } catch (...) {
                block.benchmark_times[ci] = std::numeric_limits<double>::infinity();
                arena.free_since(arena_mark);  // P2-B: scoped cleanup
            }
        }
    }

    // ─── Benchmark diagnostic ───
    if (verbose) {
        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            auto& block = blocks[bi];
            if (block.dense_head) continue;
            LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Block [" << block.start_row << "," << block.end_row
                             << ") rows=" << (block.end_row - block.start_row)
                             << " nnz=" << block.nnz << " avg_w=" << std::fixed << std::setprecision(1) << block.avg_weight << ":";
            for (size_t ci = 0; ci < configs.size(); ++ci) {
                double t = block.benchmark_times[ci];
                if (t < std::numeric_limits<double>::infinity())
                    LOG(LOG_DEBUG_2) << "  " << configs[ci].name << ": " << std::setprecision(4) << t << " ms";
            }
        }
        LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Conversion cost amortization: "
                         << effective_spmm_calls << " calls" << std::endl;
    }

    // ─── Phase 5b: Whole-Range Delta-16 Benchmark ───
    // Per-block benchmarks underestimate Delta-16 for small blocks because
    // Delta-16's serial dependency chain requires high GPU occupancy (many CUDA
    // blocks) to hide memory latency.  A single-segment Delta-16 covering the
    // entire sparse tail achieves much higher occupancy than individual blocks.
    // Benchmark Delta-16 on the full sparse range and inject this time into the
    // DP solver so it can compete fairly against multi-segment alternatives.
    //
    // Use a separate arena so cleanup doesn't free cached per-block format data.
    int delta16_ci = -1;
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        if (configs[ci].id == KernelID::Sparse_Delta16) { delta16_ci = (int)ci; break; }
    }

    // Identify first and last non-dense blocks
    int first_sparse_bi = -1, last_sparse_bi = -1;
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        if (blocks[bi].dense_head) continue;
        if (first_sparse_bi < 0) first_sparse_bi = (int)bi;
        last_sparse_bi = (int)bi;
    }

    double delta16_whole_range_ms = std::numeric_limits<double>::infinity();
    if (delta16_ci >= 0 && first_sparse_bi >= 0 && first_sparse_bi < last_sparse_bi) {
        // Build a whole-range CSR slice covering all sparse blocks
        uint32_t wr_start = blocks[first_sparse_bi].start_row;
        uint32_t wr_end   = blocks[last_sparse_bi].end_row;
        uint32_t wr_nrows = wr_end - wr_start;
        uint64_t wr_nnz = 0;
        for (int bi = first_sparse_bi; bi <= last_sparse_bi; ++bi) wr_nnz += blocks[bi].nnz;

        if (wr_nrows > 0 && wr_nnz > 0) {
            DeviceCSRSlice wr_slice;
            wr_slice.d_row_ptr = csr.d_row_ptr + wr_start;
            wr_slice.d_col_ind = csr.d_col_ind;
            wr_slice.start_row = wr_start;
            wr_slice.n_rows    = wr_nrows;
            wr_slice.n_cols    = csr.n_cols;
            wr_slice.nnz       = wr_nnz;

            FormatArena wr_arena;  // isolated arena for whole-range benchmark
            try {
                DeviceMatrix dm = {};
                gpu_convert_csr_to_delta16(wr_slice, wr_arena, dm);
                CUDA_CHECK(cudaMemset(d_flush, 0xAA, flush_bytes));
                size_t out_bytes = (size_t)wr_nrows * (global_vec_width / 8);
                // d_C may be too small for whole-range — allocate temporary if needed
                void* d_C_wr = d_C;
                bool allocated_wr = false;
                if (out_bytes > max_out) {
                    CUDA_CHECK(cudaMalloc(&d_C_wr, out_bytes));
                    allocated_wr = true;
                }
                CUDA_CHECK(cudaMemset(d_C_wr, 0, out_bytes));
                SpMMConfig cfg = config_to_spmm_config(configs[delta16_ci]);
                // Warmup + sync to get accurate timing
                SpMMKernels::run_spmm(dm, d_C_wr, d_V, cfg);
                CUDA_CHECK(cudaDeviceSynchronize());
                int n_iter = 5;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int k = 0; k < n_iter; ++k)
                    SpMMKernels::run_spmm(dm, d_C_wr, d_V, cfg);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                delta16_whole_range_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / n_iter;
                if (verbose) {
                    LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Whole-range Delta-16 ["
                                     << wr_start << "," << wr_end << "): "
                                     << std::fixed << std::setprecision(4) << delta16_whole_range_ms
                                     << " ms, " << std::setprecision(2)
                                     << (wr_nnz / 1e9) / (delta16_whole_range_ms / 1000.0)
                                     << " GNNz/s" << std::endl;
                }
                if (allocated_wr) CUDA_CHECK(cudaFree(d_C_wr));
                wr_arena.free_all();
            } catch (...) {
                wr_arena.free_all();
            }
        }
    }

    // ─── Phase 5c: Whole-Range Warp-CSR Benchmark ───
    // Mirror Phase 5b for Warp-CSR.  Warp-CSR uses CSR directly (no format
    // conversion), so this is simpler.  Gives Phase 6b a fair like-for-like
    // comparison between single-segment alternatives.
    int warp_csr_ci = -1;
    for (size_t ci = 0; ci < configs.size(); ++ci) {
        if (configs[ci].id == KernelID::Sparse_WarpCSR) { warp_csr_ci = (int)ci; break; }
    }

    double warp_csr_whole_range_ms = std::numeric_limits<double>::infinity();
    if (warp_csr_ci >= 0 && first_sparse_bi >= 0 && first_sparse_bi < last_sparse_bi) {
        uint32_t wr_start = blocks[first_sparse_bi].start_row;
        uint32_t wr_end   = blocks[last_sparse_bi].end_row;
        uint32_t wr_nrows = wr_end - wr_start;
        uint64_t wr_nnz = 0;
        for (int bi = first_sparse_bi; bi <= last_sparse_bi; ++bi) wr_nnz += blocks[bi].nnz;

        if (wr_nrows > 0 && wr_nnz > 0) {
            try {
                DeviceMatrix dm = {};
                DeviceCSRSlice wr_slice;
                wr_slice.d_row_ptr  = csr.d_row_ptr + wr_start;
                wr_slice.d_col_ind  = csr.d_col_ind;
                wr_slice.start_row  = wr_start;
                wr_slice.n_rows     = wr_nrows;
                wr_slice.n_cols     = csr.n_cols;
                wr_slice.nnz        = wr_nnz;

                dm.warp_csr_row_ptr   = wr_slice.d_row_ptr;
                dm.warp_csr_col_ind   = wr_slice.d_col_ind;
                dm.warp_csr_n_rows    = wr_nrows;
                dm.warp_csr_start_row = 0;
                dm.n_rows             = wr_nrows;

                CUDA_CHECK(cudaMemset(d_flush, 0xAA, flush_bytes));
                size_t out_bytes = (size_t)wr_nrows * (global_vec_width / 8);
                void* d_C_wr = d_C;
                bool allocated_wr = false;
                if (out_bytes > max_out) {
                    CUDA_CHECK(cudaMalloc(&d_C_wr, out_bytes));
                    allocated_wr = true;
                }
                CUDA_CHECK(cudaMemset(d_C_wr, 0, out_bytes));
                SpMMConfig cfg = config_to_spmm_config(configs[warp_csr_ci]);

                SpMMKernels::run_spmm(dm, d_C_wr, d_V, cfg);
                CUDA_CHECK(cudaDeviceSynchronize());
                int n_iter = 5;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int k = 0; k < n_iter; ++k)
                    SpMMKernels::run_spmm(dm, d_C_wr, d_V, cfg);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                warp_csr_whole_range_ms = std::chrono::duration<double>(t1 - t0).count() * 1000.0 / n_iter;

                if (verbose) {
                    LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Whole-range Warp-CSR ["
                                     << wr_start << "," << wr_end << "): "
                                     << std::fixed << std::setprecision(4) << warp_csr_whole_range_ms
                                     << " ms, " << std::setprecision(2)
                                     << (wr_nnz / 1e9) / (warp_csr_whole_range_ms / 1000.0)
                                     << " GNNz/s" << std::endl;
                }
                if (allocated_wr) CUDA_CHECK(cudaFree(d_C_wr));
            } catch (...) {}
        }
    }

    // ─── Phase 6: DP Solver ───
    const size_t B = blocks.size();
    const size_t K = configs.size();
    std::vector<double> dp(B + 1, std::numeric_limits<double>::infinity());
    std::vector<std::pair<int,int>> path(B + 1, {-1, -1});
    dp[0] = 0.0;
    const double KERNEL_OVERHEAD_MS = 0.050;

    for (size_t i = 1; i <= B; ++i) {
        for (size_t j = 0; j < i; ++j) {
            for (size_t k = 0; k < K; ++k) {
                if (configs[k].id == KernelID::M4RM) {
                    row_idx_t seg_rows = blocks[i-1].end_row - blocks[j].start_row;
                    if (seg_rows != 8) continue;
                }
                double exec_time = 0.0;
                bool possible = true;
                for (size_t b = j; b < i; ++b) {
                    double t = blocks[b].benchmark_times[k];
                    if (t == std::numeric_limits<double>::infinity()) {
                        possible = false; break;
                    }
                    exec_time += t;
                }
                if (!possible) continue;
                double cost = dp[j] + exec_time + KERNEL_OVERHEAD_MS;
                if (cost < dp[i]) {
                    dp[i] = cost;
                    path[i] = {(int)j, (int)k};
                }
            }
        }
    }

    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] DP solver: " << B << " blocks x "
                     << K << " candidates = " << (B * B * K) << " iterations" << std::endl;

    // ─── Phase 6b: Plan Consolidation (A only) ───
    // Compare DP solution's sparse-tail cost against the best whole-range
    // single-segment kernel (Delta-16 or Warp-CSR).  A single segment covering
    // all sparse blocks avoids kernel launch overhead and achieves higher GPU
    // occupancy than the per-block benchmarks suggest.
    //
    // Disabled for AT (threshold=0): AT's column-frequency-sorted rows create
    // a heterogeneous sparse tail where the DP plan selects specialized formats
    // (PForDelta for dense blocks, Delta-16/Warp-CSR for sparse).  Consolidation
    // destroys this diversity and regresses AT throughput 2-5x.
    //
    // Hoisted to tune() scope so Phase 7 can use these for throughput reporting.
    double best_whole_range_ms = std::numeric_limits<double>::infinity();
    int best_whole_range_ci = -1;

    if (first_sparse_bi >= 0) {

        if (delta16_ci >= 0 && delta16_whole_range_ms < best_whole_range_ms) {
            best_whole_range_ms = delta16_whole_range_ms;
            best_whole_range_ci = delta16_ci;
        }
        if (warp_csr_ci >= 0 && warp_csr_whole_range_ms < best_whole_range_ms) {
            best_whole_range_ms = warp_csr_whole_range_ms;
            best_whole_range_ci = warp_csr_ci;
        }

        if (best_whole_range_ci >= 0) {
            double dp_sparse_cost = dp[B] - dp[first_sparse_bi];
            double whole_range_cost = best_whole_range_ms + KERNEL_OVERHEAD_MS;

            if (verbose) {
                LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Plan consolidation: DP sparse="
                                 << std::fixed << std::setprecision(4) << dp_sparse_cost
                                 << " ms, best whole-range ("
                                 << configs[best_whole_range_ci].name << ")="
                                 << whole_range_cost << " ms"
                                 << (is_transposed ? " [AT: consolidation disabled]" : "")
                                 << std::endl;
            }

            double consolidation_threshold = is_transposed ? 0.0 : 1.0;
            if (whole_range_cost < dp_sparse_cost * consolidation_threshold) {
                LOG(LOG_DEBUG_2) << "[GPUAutoTuner] Overriding DP: whole-range "
                                 << configs[best_whole_range_ci].name
                                 << " saves " << std::fixed << std::setprecision(4)
                                 << (dp_sparse_cost - whole_range_cost) << " ms" << std::endl;
                dp[B] = dp[first_sparse_bi] + whole_range_cost;
                path[B] = {first_sparse_bi, best_whole_range_ci};
                for (int bi = first_sparse_bi + 1; bi < (int)B; ++bi)
                    path[bi] = {-1, -1};
            }
        }
    }

    // ─── Phase 7: Build Plan from Cached Format Data ───
    // Instead of re-converting winners, retrieve them from the format cache
    // built during benchmarking.
    ExecutionPlan plan;
    plan.is_transposed = is_transposed;
    plan.estimated_total_time_ms = dp[B];

    // Track which cache entries are winners (for cleanup)
    std::vector<std::vector<bool>> is_winner(blocks.size(),
        std::vector<bool>(configs.size(), false));

    // Backtrack DP path
    std::vector<SegmentRecipe> segs;
    int curr = (int)B;
    while (curr > 0) {
        int prev = path[curr].first;
        int ki   = path[curr].second;

        SegmentRecipe seg;
        seg.start_row   = blocks[prev].start_row;
        seg.end_row     = blocks[curr - 1].end_row;
        seg.best_config = configs[ki];

        // Compute throughput — handle Infinity from Phase 6b overrides
        uint64_t seg_nnz = 0;
        for (int b = prev; b < curr; ++b) seg_nnz += blocks[b].nnz;
        double seg_time = 0;
        bool seg_time_valid = true;
        for (int b = prev; b < curr; ++b) {
            double t = blocks[b].benchmark_times[ki];
            if (!std::isfinite(t)) { seg_time_valid = false; break; }
            seg_time += t;
        }

        if (seg_time_valid && seg_time > 0) {
            seg.measured_throughput_gnnz = (seg_nnz / 1e9) / (seg_time / 1000.0);
        } else if (ki == best_whole_range_ci &&
                   std::isfinite(best_whole_range_ms) && best_whole_range_ms > 0) {
            // Phase 6b override: use the whole-range benchmark time
            seg.measured_throughput_gnnz = (seg_nnz / 1e9) / (best_whole_range_ms / 1000.0);
        } else {
            seg.measured_throughput_gnnz = 0.0;  // no valid measurement available
        }

        // Retrieve winner format data from cache.
        // A DP segment may span multiple atomic blocks — if so, we need to
        // re-convert for the merged range (cache only has per-block data).
        int n_seg_blocks = curr - prev;
        if (n_seg_blocks == 1 && format_cache[prev][ki].valid) {
            // Single-block segment: use cached data directly
            auto& cached = format_cache[prev][ki];
            if (cached.is_m4rm) {
                seg.m4rm_data = cached.m4rm;
            } else {
                seg.device_data = cached.dm;
            }
            is_winner[prev][ki] = true;
        } else {
            // Multi-block segment or missing cache: must convert the merged range
            uint32_t n = seg.end_row - seg.start_row;
            DeviceCSRSlice slice = csr.slice(seg.start_row, n);
            switch (seg.best_config.id) {
                case KernelID::M4RM:
                    gpu_convert_csr_to_m4rm(slice, seg.best_config.params.m4rm_rows,
                        arena, seg.m4rm_data);
                    break;
                case KernelID::Sparse_TiledCOO:
                    gpu_convert_csr_to_tiledcoo(slice, seg.best_config.params.tiled_block_size,
                        arena, seg.device_data);
                    break;
                case KernelID::Sparse_TiledCOO_Unrolled:
                    gpu_convert_csr_to_tiledcoo(slice, seg.best_config.params.tiled_block_size,
                        arena, seg.device_data);
                    break;
                case KernelID::Sparse_Delta16:
                    gpu_convert_csr_to_delta16(slice, arena, seg.device_data);
                    break;
                case KernelID::Sparse_PForDelta_BitExact:
                    gpu_convert_csr_to_pfor_be(slice, seg.best_config.params.pfor_threshold,
                        arena, seg.device_data, config.allow_cpu_fallback, &permuted_mat);
                    break;
                case KernelID::Sparse_WarpCSR: {
                    // No format conversion — set non-owning pointers into CSR
                    seg.device_data.warp_csr_row_ptr  = slice.d_row_ptr;
                    seg.device_data.warp_csr_col_ind  = slice.d_col_ind;
                    seg.device_data.warp_csr_n_rows   = slice.n_rows;
                    seg.device_data.warp_csr_start_row = 0;
                    seg.device_data.n_rows            = slice.n_rows;
                    break;
                }
                case KernelID::Sparse_Golomb:
                    gpu_convert_csr_to_golomb(slice, arena, seg.device_data,
                        config.allow_cpu_fallback, &permuted_mat);
                    break;
                default: break;
            }
        }

        // Promote surviving device pointers to persistent so that
        // arena.free_temporaries() does not free the plan's live data.
        switch (seg.best_config.id) {
            case KernelID::Sparse_TiledCOO:
                arena.promote_to_persistent(seg.device_data.d_sparse_tiled_coords);
                arena.promote_to_persistent(seg.device_data.d_sparse_tiled_ptr);
                break;
            case KernelID::Sparse_TiledCOO_Unrolled:
                arena.promote_to_persistent(seg.device_data.d_sparse_tiled_coords);
                arena.promote_to_persistent(seg.device_data.d_sparse_tiled_ptr);
                break;
            case KernelID::Sparse_Delta16:
                arena.promote_to_persistent(seg.device_data.d_delta_16_stream);
                arena.promote_to_persistent(seg.device_data.d_delta_16_offsets);
                break;
            case KernelID::M4RM:
                arena.promote_to_persistent(seg.m4rm_data.d_pattern_stream);
                break;
            case KernelID::Sparse_WarpCSR:
                // No arena allocation — CSR data will become owned copies below
                break;
            default: break;  // PFor_BE/Golomb: allocated outside arena
        }

        segs.push_back(seg);
        curr = prev;
    }
    std::reverse(segs.begin(), segs.end());
    plan.segments = segs;

    // Download the density permutation so the CPU fallback in compile()
    // can map double-permuted segment indices back to host_mat row order.
    plan.density_perm.resize(csr.n_rows);
    CUDA_CHECK(cudaMemcpy(plan.density_perm.data(), csr.d_density_perm,
               csr.n_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // WarpCSR segments hold non-owning pointers into the global CSR.
    // Before csr.free(), make owned copies so the plan survives.
    for (auto& seg : plan.segments) {
        if (seg.best_config.id != KernelID::Sparse_WarpCSR) continue;
        if (!seg.device_data.warp_csr_row_ptr) continue;

        uint32_t n = seg.end_row - seg.start_row;

        // Read row_ptr slice to host (n+1 elements) — compute NNZ range
        std::vector<uint32_t> h_rp(n + 1);
        CUDA_CHECK(cudaMemcpy(h_rp.data(), csr.d_row_ptr + seg.start_row,
            (n + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t nnz_start = h_rp[0];
        uint32_t nnz_end   = h_rp[n];
        uint32_t seg_nnz   = nnz_end - nnz_start;

        // Adjust row_ptr to be zero-based for the owned col_ind copy
        for (uint32_t i = 0; i <= n; i++) h_rp[i] -= nnz_start;

        // Upload adjusted row_ptr
        uint32_t* owned_row_ptr;
        CUDA_CHECK(cudaMalloc(&owned_row_ptr, (n + 1) * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(owned_row_ptr, h_rp.data(),
            (n + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Copy col_ind range (seg_nnz elements)
        uint32_t* owned_col_ind = nullptr;
        if (seg_nnz > 0) {
            CUDA_CHECK(cudaMalloc(&owned_col_ind, seg_nnz * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(owned_col_ind, csr.d_col_ind + nnz_start,
                seg_nnz * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }

        seg.device_data.warp_csr_row_ptr = owned_row_ptr;
        seg.device_data.warp_csr_col_ind = owned_col_ind;
    }

    // Compute overall throughput
    uint64_t total_nnz = 0;
    for (const auto& b : blocks) total_nnz += b.nnz;
    plan.estimated_throughput_gnnz = (total_nnz / 1e9) / (plan.estimated_total_time_ms / 1000.0);

    // Verbose logging
    if (verbose) {
        LOG(LOG_DEBUG_1) << "[GPUAutoTuner] ─── Execution Plan ───" << std::endl;
        for (const auto& s : plan.segments) {
            LOG(LOG_DEBUG_1) << "[GPUAutoTuner]  [" << std::setw(7) << s.start_row
                             << " - " << std::setw(7) << s.end_row << ")  "
                             << std::setw(28) << s.best_config.name
                             << "  " << std::fixed << std::setprecision(2)
                             << s.measured_throughput_gnnz << " GNNZ/s"
                             << (s.has_device_data() ? " [GPU]" : " [CPU]") << std::endl;
        }
        LOG(LOG_DEBUG_1) << "[GPUAutoTuner] Est. total: " << std::setprecision(3)
                         << plan.estimated_total_time_ms << " ms, "
                         << plan.estimated_throughput_gnnz << " GNNZ/s" << std::endl;
    }

    // Free non-winning cached formats.
    // CPU-fallback formats (PFor_BE, Golomb) are NOT tracked by the arena
    // and must be freed individually.  Arena-managed formats (TiledCOO,
    // Delta16, M4RM) will be freed by arena.free_temporaries() below.
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        for (size_t ci = 0; ci < configs.size(); ++ci) {
            auto& cached = format_cache[bi][ci];
            if (!cached.valid || is_winner[bi][ci]) continue;
            if (!cached.is_arena_managed) {
                // CPU-fallback format: free the DeviceMatrix directly
                SpMMKernels::free_matrix(cached.dm);
            }
            cached.valid = false;
        }
    }

    // Cleanup: free DeviceCSR (format data in SegmentRecipes survives)
    csr.free();
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_flush));
    arena.free_temporaries();  // Frees non-persistent arena data (losing formats)

    LOG(LOG_DEBUG_1) << "[GPUAutoTuner] GPU-only tuning complete" << std::endl;

    return plan;
}
